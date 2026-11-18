"""Hash token prefixes, apply cache lifecycle events, and compare routing costs."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import struct
from typing import Iterable


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} is too large") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _name(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _tokens(values: Iterable[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, dict, set, frozenset)):
        raise ValueError("tokens must be an iterable of integer token IDs")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError("tokens must be an iterable of integer token IDs") from exc
    for token in result:
        _integer(token, "token ID")
        if token > 0xFFFFFFFF:
            raise ValueError("token IDs must fit in an unsigned 32-bit integer")
    return result


@dataclass(frozen=True)
class Scope:
    """Cache identity; all values must describe the actual inference context."""

    model_revision: str = "example-model-v1"
    tokenizer_revision: str = "example-tokenizer-v1"
    tenant: str = "demo-tenant"
    adapter: str = "base"

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            _name(value, key)


def hash_blocks(tokens: Iterable[int], block_size: int, scope: Scope) -> tuple[str, ...]:
    """Return chained hashes for full blocks; incomplete final blocks are omitted."""
    _integer(block_size, "block_size", 1)
    if not isinstance(scope, Scope):
        raise ValueError("scope must be a Scope")
    token_ids = _tokens(tokens)
    identity = {"format": "prefix-router-v1", "block_size": block_size, **asdict(scope)}
    parent = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=True,
                                       separators=(",", ":")).encode("ascii")).digest()
    hashes = []
    for offset in range(0, len(token_ids) - block_size + 1, block_size):
        digest = hashlib.sha256(parent)
        for token in token_ids[offset:offset + block_size]:
            digest.update(struct.pack(">I", token))
        parent = digest.digest()
        hashes.append(parent.hex())
    return tuple(hashes)


@dataclass(frozen=True)
class CacheEvent:
    worker: str
    sequence: int
    kind: str
    blocks: tuple[str, ...] = ()


@dataclass(frozen=True)
class EventResult:
    applied: bool
    gap_detected: bool


class CacheIndex:
    """Block-to-worker ownership with ordered per-worker lifecycle events."""

    def __init__(self, workers: Iterable[str]):
        if isinstance(workers, (str, bytes, dict)):
            raise ValueError("workers must be an iterable of unique worker names")
        try:
            worker_ids = tuple(workers)
        except TypeError as exc:
            raise ValueError("workers must be an iterable of unique worker names") from exc
        for worker in worker_ids:
            _name(worker, "worker")
        if not worker_ids or len(set(worker_ids)) != len(worker_ids):
            raise ValueError("provide at least one worker with unique names")
        self.workers = tuple(sorted(worker_ids))
        self._blocks = {worker: set() for worker in self.workers}
        self._owners: dict[str, set[str]] = {}
        self._sequence = {worker: 0 for worker in self.workers}

    def _worker(self, worker: str) -> None:
        if not isinstance(worker, str) or worker not in self._blocks:
            raise ValueError(f"unknown worker: {worker!r}")

    def _remove(self, worker: str, blocks: Iterable[str]) -> None:
        for block in tuple(blocks):
            self._blocks[worker].discard(block)
            owners = self._owners.get(block)
            if owners is not None:
                owners.discard(worker)
                if not owners:
                    del self._owners[block]

    def apply(self, event: CacheEvent) -> EventResult:
        """Ignore stale events; clear a worker's old index if a sequence is missing."""
        if not isinstance(event, CacheEvent):
            raise ValueError("event must be a CacheEvent")
        self._worker(event.worker)
        _integer(event.sequence, "event sequence", 1)
        if not isinstance(event.kind, str) or event.kind not in {"created", "evicted", "cleared"}:
            raise ValueError("event kind must be created, evicted, or cleared")
        if not isinstance(event.blocks, (tuple, list)):
            raise ValueError("event blocks must be a list or tuple of SHA256 hex hashes")
        for block in event.blocks:
            if not isinstance(block, str) or re.fullmatch(r"[0-9a-f]{64}", block) is None:
                raise ValueError("event blocks must contain lowercase SHA256 hex hashes")
        if event.kind == "cleared" and event.blocks:
            raise ValueError("cleared events cannot include block hashes")
        previous = self._sequence[event.worker]
        if event.sequence <= previous:
            return EventResult(applied=False, gap_detected=False)
        gap = event.sequence != previous + 1
        if gap or event.kind == "cleared":
            self._remove(event.worker, self._blocks[event.worker])
        if event.kind == "created":
            for block in event.blocks:
                self._blocks[event.worker].add(block)
                self._owners.setdefault(block, set()).add(event.worker)
        elif event.kind == "evicted":
            self._remove(event.worker, event.blocks)
        self._sequence[event.worker] = event.sequence
        return EventResult(applied=True, gap_detected=gap)

    def owners(self, block: str) -> tuple[str, ...]:
        return tuple(sorted(self._owners.get(block, ())))

    def prefix_hits(self, worker: str, request_blocks: Iterable[str]) -> int:
        """Only resident blocks before the first miss contribute cache credit."""
        self._worker(worker)
        hits = 0
        for block in request_blocks:
            if worker not in self._owners.get(block, ()):
                break
            hits += 1
        return hits


@dataclass(frozen=True)
class WorkerScore:
    worker: str
    cached_blocks: int
    reused_tokens: int
    uncached_tokens: int
    uncached_prefill_blocks: float
    decode_load: float
    score: float


@dataclass(frozen=True)
class Decision:
    worker: str
    prompt_tokens: int
    complete_blocks: int
    tail_tokens: int
    prefill_weight: float
    candidates: tuple[WorkerScore, ...]

    def as_dict(self) -> dict:
        return asdict(self)


class Router:
    def __init__(self, index: CacheIndex, block_size: int = 16, prefill_weight: float = 1.0):
        if not isinstance(index, CacheIndex):
            raise ValueError("index must be a CacheIndex")
        _integer(block_size, "block_size", 1)
        self.index = index
        self.block_size = block_size
        self.prefill_weight = _number(prefill_weight, "prefill_weight")

    def route(self, tokens: Iterable[int], scope: Scope, decode_load: dict[str, float]) -> Decision:
        """Select minimum cost; routing alone never declares blocks to be cached."""
        token_ids = _tokens(tokens)
        if not token_ids:
            raise ValueError("a routing request must have at least one token")
        blocks = hash_blocks(token_ids, self.block_size, scope)
        if not isinstance(decode_load, dict) or set(decode_load) != set(self.index.workers):
            raise ValueError("decode_load must contain exactly one entry per worker")
        candidates = []
        for worker in self.index.workers:
            load = _number(decode_load[worker], f"decode_load[{worker}]")
            hits = self.index.prefix_hits(worker, blocks)
            reused = hits * self.block_size
            uncached = len(token_ids) - reused
            prefill = uncached / self.block_size
            score = load + self.prefill_weight * prefill
            if not math.isfinite(score):
                raise ValueError("routing score overflow; reduce loads or prefill_weight")
            candidates.append(WorkerScore(worker, hits, reused, uncached, prefill, load, score))
        chosen = min(candidates, key=lambda candidate: (candidate.score, candidate.worker))
        return Decision(chosen.worker, len(token_ids), len(blocks),
                        len(token_ids) % self.block_size, self.prefill_weight, tuple(candidates))
