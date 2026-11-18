"""Behavioral tests for prefix identity, lifecycle events, and load-aware choices."""

from dataclasses import replace
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from prefix_router import CacheEvent, CacheIndex, Router, Scope, hash_blocks
from prefix_router.__main__ import main, replay


ROOT = Path(__file__).resolve().parents[1]


class HashTests(unittest.TestCase):
    def test_chain_changes_only_from_changed_block_onward(self):
        original = hash_blocks([1, 2, 3, 4, 5, 6], 2, Scope())
        changed = hash_blocks([1, 2, 3, 99, 5, 6], 2, Scope())
        self.assertEqual(original[0], changed[0])
        self.assertNotEqual(original[1], changed[1])
        self.assertNotEqual(original[2], changed[2])

    def test_incomplete_block_never_becomes_cached(self):
        self.assertEqual(hash_blocks([1, 2, 3], 2, Scope()), hash_blocks([1, 2], 2, Scope()))
        self.assertEqual(hash_blocks([1], 2, Scope()), ())
        self.assertEqual(hash_blocks([], 2, Scope()), ())

    def test_identical_suffix_with_different_parent_is_not_a_hit(self):
        first = hash_blocks([1, 2, 7, 8], 2, Scope())
        other = hash_blocks([3, 4, 7, 8], 2, Scope())
        self.assertNotEqual(first[1], other[1])

    def test_every_scope_field_isolates_cache_identity(self):
        scope = Scope()
        original = hash_blocks([1, 2], 2, scope)
        for name in ("model_revision", "tokenizer_revision", "tenant", "adapter"):
            with self.subTest(field=name):
                self.assertNotEqual(original, hash_blocks([1, 2], 2, replace(scope, **{name: "other"})))

    def test_block_size_is_part_of_identity(self):
        self.assertNotEqual(hash_blocks([1, 2], 2, Scope())[0],
                            hash_blocks([1, 2], 1, Scope())[1])

    def test_hashes_are_stable_across_processes(self):
        code = "from prefix_router import *; print(hash_blocks([1,2,3,4],2,Scope())[1])"
        first = subprocess.check_output([sys.executable, "-B", "-c", code], cwd=ROOT, text=True).strip()
        self.assertEqual(first, hash_blocks([1, 2, 3, 4], 2, Scope())[1])

    def test_bad_token_ids_and_unordered_inputs_are_rejected(self):
        for tokens in ([True], [-1], [1.5], [2**32], "12", None, {1, 2}):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                hash_blocks(tokens, 2, Scope())
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                hash_blocks([1, 2], size, Scope())
        with self.assertRaises(ValueError):
            Scope(tenant=" ")


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.index = CacheIndex(["a", "b"])
        self.blocks = hash_blocks(range(8), 2, Scope())

    def test_created_event_updates_owners_for_multiple_workers(self):
        self.index.apply(CacheEvent("a", 1, "created", self.blocks))
        self.index.apply(CacheEvent("b", 1, "created", self.blocks[:2]))
        self.assertEqual(self.index.owners(self.blocks[0]), ("a", "b"))
        self.assertEqual(self.index.prefix_hits("a", self.blocks), 4)
        self.assertEqual(self.index.prefix_hits("b", self.blocks), 2)

    def test_hits_stop_at_first_missing_block(self):
        self.index.apply(CacheEvent("a", 1, "created", (self.blocks[0], self.blocks[2], self.blocks[3])))
        self.assertEqual(self.index.prefix_hits("a", self.blocks), 1)
        self.index.apply(CacheEvent("a", 2, "created", self.blocks[1:2]))
        self.assertEqual(self.index.prefix_hits("a", self.blocks), 4)

    def test_eviction_breaks_prefix_and_keeps_other_workers(self):
        for worker in ("a", "b"):
            self.index.apply(CacheEvent(worker, 1, "created", self.blocks))
        self.index.apply(CacheEvent("a", 2, "evicted", self.blocks[1:2]))
        self.assertEqual(self.index.prefix_hits("a", self.blocks), 1)
        self.assertEqual(self.index.prefix_hits("b", self.blocks), 4)
        self.assertEqual(self.index.owners(self.blocks[1]), ("b",))

    def test_clearing_removes_all_worker_ownership(self):
        self.index.apply(CacheEvent("a", 1, "created", self.blocks))
        self.index.apply(CacheEvent("a", 2, "cleared"))
        self.assertEqual(self.index.prefix_hits("a", self.blocks), 0)
        self.assertEqual(self.index.owners(self.blocks[0]), ())

    def test_stale_or_repeated_event_cannot_resurrect_evicted_blocks(self):
        self.index.apply(CacheEvent("a", 1, "created", self.blocks))
        self.index.apply(CacheEvent("a", 2, "evicted", self.blocks))
        result = self.index.apply(CacheEvent("a", 1, "created", self.blocks))
        self.assertFalse(result.applied)
        self.assertEqual(self.index.prefix_hits("a", self.blocks), 0)

    def test_gap_discards_potentially_stale_ownership(self):
        self.index.apply(CacheEvent("a", 1, "created", self.blocks))
        result = self.index.apply(CacheEvent("a", 3, "created", self.blocks[2:]))
        self.assertTrue(result.gap_detected)
        self.assertEqual(self.index.owners(self.blocks[0]), ())
        self.assertEqual(self.index.prefix_hits("a", self.blocks), 0)

    def test_validation_happens_before_mutation(self):
        self.index.apply(CacheEvent("a", 1, "created", self.blocks))
        for event in (CacheEvent("unknown", 1, "created"), CacheEvent("a", 0, "created"),
                      CacheEvent("a", 2, "removed"), CacheEvent("a", 2, "created", ("invalid",)),
                      CacheEvent("a", 2, "cleared", self.blocks)):
            with self.subTest(event=event), self.assertRaises(ValueError):
                self.index.apply(event)
        self.assertEqual(self.index.prefix_hits("a", self.blocks), 4)

    def test_duplicate_hashes_and_absent_evictions_are_idempotent(self):
        self.index.apply(CacheEvent("a", 1, "evicted", self.blocks))
        self.index.apply(CacheEvent("a", 2, "created", self.blocks + self.blocks))
        self.assertEqual(self.index.prefix_hits("a", self.blocks), 4)


class RouterTests(unittest.TestCase):
    def test_lower_load_beats_best_cache_match(self):
        index = CacheIndex(["worker-1", "worker-2", "worker-3"])
        blocks = hash_blocks(range(20), 2, Scope())
        for worker, count in (("worker-1", 2), ("worker-2", 5), ("worker-3", 8)):
            index.apply(CacheEvent(worker, 1, "created", blocks[:count]))
        result = Router(index, 2).route(range(20), Scope(), {"worker-1": 10, "worker-2": 5, "worker-3": 9})
        self.assertEqual(result.worker, "worker-2")
        self.assertEqual([row.score for row in result.candidates], [18.0, 10.0, 11.0])

    def test_partial_tail_is_always_prefilled_and_scored_fractionally(self):
        index = CacheIndex(["a"])
        index.apply(CacheEvent("a", 1, "created", hash_blocks([1, 2, 3], 2, Scope())))
        result = Router(index, 2).route([1, 2, 3], Scope(), {"a": 0})
        self.assertEqual(result.tail_tokens, 1)
        self.assertEqual(result.candidates[0].uncached_tokens, 1)
        self.assertEqual(result.candidates[0].score, 0.5)

    def test_higher_weight_can_choose_busier_warmer_worker(self):
        index = CacheIndex(["cold", "warm"])
        index.apply(CacheEvent("warm", 1, "created", hash_blocks(range(20), 2, Scope())))
        loads = {"cold": 0, "warm": 11}
        self.assertEqual(Router(index, 2, 1).route(range(20), Scope(), loads).worker, "cold")
        self.assertEqual(Router(index, 2, 2).route(range(20), Scope(), loads).worker, "warm")

    def test_zero_weight_uses_load_only_and_ties_sort_by_worker(self):
        index = CacheIndex(["z", "a"])
        index.apply(CacheEvent("z", 1, "created", hash_blocks(range(4), 2, Scope())))
        result = Router(index, 2, 0).route(range(4), Scope(), {"a": 3, "z": 3})
        self.assertEqual(result.worker, "a")

    def test_routing_does_not_predict_cache_events(self):
        index = CacheIndex(["a"])
        router = Router(index, 2)
        router.route(range(4), Scope(), {"a": 0})
        second = router.route(range(4), Scope(), {"a": 0})
        self.assertEqual(second.candidates[0].cached_blocks, 0)

    def test_invalid_inputs_and_score_overflow_fail_cleanly(self):
        index = CacheIndex(["a"])
        for weight in (-1, True, float("nan"), float("inf"), 10**1000):
            with self.subTest(weight=str(weight)), self.assertRaises(ValueError):
                Router(index, 2, weight)
        router = Router(index, 2)
        for loads in ({}, {"a": 0, "b": 0}, {"a": -1}, {"a": True}, {"a": float("nan")}):
            with self.subTest(loads=loads), self.assertRaises(ValueError):
                router.route([1], Scope(), loads)
        with self.assertRaises(ValueError):
            router.route([], Scope(), {"a": 0})
        with self.assertRaises(ValueError):
            Router(index, 1, 1e308).route([1, 2, 3], Scope(), {"a": 1e308})


class ScenarioTests(unittest.TestCase):
    def test_demo_replays_expected_decisions(self):
        result = replay(json.loads((ROOT / "examples" / "demo.json").read_text(encoding="utf-8")))
        self.assertTrue(result["simulation_only"])
        self.assertEqual([row["worker"] for row in result["decisions"]],
                         ["worker-2", "worker-3", "worker-2", "worker-2", "worker-2"])
        self.assertEqual(result["decisions"][2]["candidates"][1]["uncached_tokens"], 1)
        self.assertEqual([row["cached_blocks"] for row in result["decisions"][3]["candidates"]], [0, 0, 0])

    def test_scenario_validation_catches_typos_and_invalid_eviction_indexes(self):
        for operation in ({"type": "unknown"}, {"type": "clear", "worker": "missing"},
                          {"type": "cache", "worker": "a", "tokens": [1, 2], "typo": True},
                          {"type": "evict", "worker": "a", "tokens": [1, 2], "block_indexes": [1]}):
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, "operation 0"):
                replay({"block_size": 2, "workers": {"a": 0}, "operations": [operation]})

    def test_cli_runs_demo_and_rejects_nonfinite_json_without_traceback(self):
        completed = subprocess.run([sys.executable, "-B", "-m", "prefix_router", "examples/demo.json"],
                                   cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(json.loads(completed.stdout)["decisions"]), 5)
        errors = io.StringIO()
        with patch.object(Path, "read_text", return_value='{"workers":{"a":1e999},"operations":[]}'):
            with redirect_stderr(errors):
                status = main(["invalid.json"])
        self.assertEqual(status, 2)
        self.assertIn("finite", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
