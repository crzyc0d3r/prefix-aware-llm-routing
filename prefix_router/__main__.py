"""Replay a JSON scenario and print each routing decision as JSON."""

import argparse
import json
from pathlib import Path
import sys

from .core import CacheEvent, CacheIndex, Router, Scope, hash_blocks, _integer, _number


def _object(value, required, optional=()):
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    if not set(required) <= set(value) or set(value) - set(required) - set(optional):
        raise ValueError(f"expected required fields {sorted(required)}; optional {sorted(optional)}")
    return value


def _scope(value):
    _object(value, (), ("model_revision", "tokenizer_revision", "tenant", "adapter"))
    return Scope(**value)


def replay(scenario):
    """Replay explicit cache events and load observations, without simulated inference."""
    _object(scenario, ("workers", "operations"), ("block_size", "prefill_weight", "scope"))
    workers = scenario["workers"]
    if not isinstance(workers, dict):
        raise ValueError("workers must map worker names to decode loads")
    index = CacheIndex(workers.keys())
    loads = {worker: _number(load, "decode_load") for worker, load in workers.items()}
    router = Router(index, scenario.get("block_size", 16), scenario.get("prefill_weight", 1.0))
    base_scope = _scope(scenario.get("scope", {}))
    operations = scenario["operations"]
    if not isinstance(operations, list):
        raise ValueError("operations must be a list")
    sequences = {worker: 0 for worker in index.workers}
    decisions = []
    events = []
    for position, operation in enumerate(operations):
        try:
            if not isinstance(operation, dict) or "type" not in operation:
                raise ValueError("each operation needs a type")
            kind = operation["type"]
            if kind == "route":
                _object(operation, ("type", "tokens"), ("label", "scope"))
                scope = _scope(operation["scope"]) if "scope" in operation else base_scope
                decision = router.route(operation["tokens"], scope, loads).as_dict()
                label = operation.get("label", f"request-{len(decisions) + 1}")
                if not isinstance(label, str):
                    raise ValueError("label must be a string")
                decisions.append({"label": label, **decision})
            elif kind == "load":
                _object(operation, ("type", "worker", "decode_load"))
                index._worker(operation["worker"])
                loads[operation["worker"]] = _number(operation["decode_load"], "decode_load")
            elif kind in {"cache", "evict", "clear"}:
                required = ("type", "worker") if kind == "clear" else ("type", "worker", "tokens")
                optional = () if kind == "clear" else (("scope", "block_indexes") if kind == "evict" else ("scope",))
                _object(operation, required, optional)
                worker = operation["worker"]
                index._worker(worker)
                blocks = ()
                if kind != "clear":
                    scope = _scope(operation["scope"]) if "scope" in operation else base_scope
                    blocks = hash_blocks(operation["tokens"], router.block_size, scope)
                    if "block_indexes" in operation:
                        indexes = operation["block_indexes"]
                        if not isinstance(indexes, list):
                            raise ValueError("block_indexes must be a list")
                        for item in indexes:
                            _integer(item, "block index")
                            if item >= len(blocks):
                                raise ValueError("block index exceeds the number of complete blocks")
                        blocks = tuple(blocks[item] for item in indexes)
                sequences[worker] += 1
                event_kind = {"cache": "created", "evict": "evicted", "clear": "cleared"}[kind]
                index.apply(CacheEvent(worker, sequences[worker], event_kind, blocks))
                events.append({"worker": worker, "sequence": sequences[worker],
                               "kind": event_kind, "block_count": len(blocks)})
            else:
                raise ValueError(f"unknown operation type: {kind!r}")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"operation {position}: {exc}") from exc
    return {"simulation_only": True, "score_units": "relative block-equivalent cost",
            "events": events, "decisions": decisions}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path, help="JSON scenario to replay")
    args = parser.parse_args(argv)
    try:
        def reject_constant(value):
            raise ValueError(f"non-finite JSON value: {value}")
        scenario = json.loads(args.scenario.read_text(encoding="utf-8"), parse_constant=reject_constant)
        result = replay(scenario)
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
