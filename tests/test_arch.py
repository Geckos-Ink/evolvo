"""Tests of the GFSL-NN architecture language, learned statistics and search.

The language/statistics/search checks are pure Python; the build/train checks skip without
PyTorch. Run: ``PYTHONPATH=src python -m unittest discover -s tests``.
"""

from __future__ import annotations

import json
import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evolvo.arch import (  # noqa: E402
    AdaptivePolicy, ArchGenome, ArchSpace, ArchitectureSearch, ArgSpec, Chooser, ContextualStats,
    EfficiencyModel, Knowledge, PlateauStopper, Row, default_registry, fit_power_law, log_bucket,
    register_macro,
)

try:  # noqa: E402
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None


def fake_evaluate(space, genome, info):
    """Loss falls with size like a power law, plus a little per primitive (no torch)."""
    graph = space.lower(genome)
    return {"loss": 0.1 + 20.0 / graph.params ** 0.5 + 0.002 * len(graph.nodes),
            "params": graph.params, "stop_reason": "plateau", "epochs": 1}


class StatsTests(unittest.TestCase):
    def test_backoff_and_discovered_order(self):
        stats = ContextualStats(pinned=("choice",), reorder_every=10)
        rng = random.Random(0)
        for _ in range(400):
            choice = rng.choice("ab")
            size = rng.choice([1, 2, 3])
            noise = rng.choice(["x", "y", "z", "w"])
            # Reward depends on choice x size; "noise" is irrelevant.
            value = (1.0 if choice == "a" else 0.0) + (0.5 * size if choice == "b" else 0.0)
            stats.observe({"choice": choice, "size": size, "noise": noise}, value + rng.gauss(0, 0.05))
        self.assertEqual(stats.order()[:2], ["choice", "size"])
        est = stats.estimate({"choice": "b", "size": 3, "noise": "x"})
        self.assertAlmostEqual(est.mean, 1.5, delta=0.1)
        # An unseen context value backs off to its parent group with zero own count.
        unseen = stats.estimate({"choice": "b", "size": 99})
        self.assertEqual(unseen.count, 0.0)
        self.assertGreaterEqual(unseen.matched, 1)
        self.assertAlmostEqual(unseen.mean, stats.estimate({"choice": "b"}).mean, delta=1e-9)
        # Shrinkage: a 1-observation group sits between its own value and the parent mean.
        stats.observe({"choice": "a", "size": 7}, 10.0)
        lone = stats.estimate({"choice": "a", "size": 7})
        self.assertTrue(1.0 < lone.mean < 10.0)
        clone = ContextualStats.from_dict(json.loads(json.dumps(stats.to_dict())))
        self.assertEqual(clone.order(), stats.order())
        self.assertAlmostEqual(clone.estimate({"choice": "b", "size": 3}).mean,
                               stats.estimate({"choice": "b", "size": 3}).mean, delta=1e-12)
        with self.assertRaises(ValueError):
            stats.observe({"choice": "a"}, math.nan)

    def test_log_bucket(self):
        self.assertEqual(log_bucket(1), 0)
        self.assertEqual(log_bucket(4, 2), 4)
        self.assertEqual(log_bucket(0), log_bucket(-3))


class LanguageTests(unittest.TestCase):
    def test_random_genomes_are_valid_and_budgeted(self):
        for inputs in ([("vec", 6)], [("set", 5), ("vec", 2)]):
            space = ArchSpace(inputs, 2, max_rows=7, max_params=40_000)
            chooser = Chooser(random.Random(1))
            for _ in range(300):
                genome = space.random_genome(chooser)
                self.assertEqual(space.validate(genome), [])
                self.assertLessEqual(space.lower(genome).params, space.max_params)
                for slots in space.to_slots(genome):
                    self.assertEqual(len(slots), 7)

    def test_cascading_options(self):
        space = ArchSpace([("vec", 6)], 1)
        ops = space.op_options(space.inputs, 0)
        self.assertNotIn("attend", ops)          # needs a set
        self.assertNotIn("transformer", ops)
        self.assertNotIn("add", ops)             # needs a second, matching value
        self.assertIn("tokenize", ops)
        heads = space.registry.get("attend").args[0]
        from evolvo.arch import Shape
        self.assertEqual(space.arg_options(heads, Shape("set", 12, "n6")), [1, 2, 4])
        tight = ArchSpace([("vec", 6)], 1, max_params=200)
        units = tight.feasible_args(tight.registry.get("linear"), (tight.inputs[0],), 0, {},
                                    tight.registry.get("linear").args[0])
        self.assertTrue(units and all(6 * u + u + u + 1 <= 200 for u in units))

    def test_macro_expansion_and_introns(self):
        space = ArchSpace([("vec", 4)], 1, max_rows=5, max_params=50_000)
        genome = ArchGenome([
            Row("tokenize", 0, -1, (16,)),
            Row("norm", 0),                            # intron: never reaches the output
            Row("transformer", 1, -1, (2, 4, 2)),
            Row("pool", 3, -1, ("mean",)),
        ])
        self.assertEqual(space.validate(genome), [])
        self.assertEqual(space.effective_rows(genome), [0, 2, 3])
        graph = space.lower(genome)
        self.assertEqual(sum(1 for n in graph.nodes if n.op == "attend"), 2)
        # tokenize 2*4*16 + 2 x (2 norms + attention + 2 linears) + head
        block = 2 * 2 * 16 + (4 * 16 * 16 + 4 * 16) + (16 * 32 + 32) + (32 * 16 + 16)
        self.assertEqual(graph.params, 2 * 4 * 16 + 2 * block + 17)
        without = ArchGenome([genome.rows[0], Row("transformer", 1, -1, (2, 4, 2)), Row("pool", 2, -1, ("mean",))])
        self.assertEqual(space.signature(genome), space.signature(without))

    def test_user_macro_and_validation(self):
        reg = default_registry()

        def gated(em, src, a):
            x = src[0]
            h = em.emit("linear", x, units=a["units"])
            g = em.emit("act", em.emit("linear", x, units=a["units"]), fn="sigmoid")
            return em.emit("mul", h, g)

        register_macro(reg, "glu", [ArgSpec("units", "width")], gated)
        space = ArchSpace([("vec", 3)], 1, registry=reg, activations=("relu", "sigmoid"))
        genome = ArchGenome([Row("glu", 0, -1, (16,))])
        self.assertEqual(space.validate(genome), [])
        self.assertEqual(space.lower(genome).params, 2 * (3 * 16 + 16) + 17)
        bad = ArchGenome([Row("glu", 0, -1, (17,))])
        self.assertTrue(space.validate(bad))
        with self.assertRaises(ValueError):
            register_macro(reg, "glu", [], gated)

    def test_repair_preserves_intent(self):
        space = ArchSpace([("vec", 4)], 1, max_rows=4, max_params=50_000)
        chooser = Chooser(random.Random(2))
        broken = ArchGenome([
            Row("tokenize", 0, -1, (6,)),
            Row("attend", 1, -1, (8,)),        # 8 heads do not divide 6
            Row("pool", 9, -1, ("mean",)),     # dangling source
        ])
        fixed, changed = space.repair(broken, chooser)
        self.assertEqual(space.validate(fixed), [])
        self.assertEqual(fixed.rows[0].args, (6,))
        self.assertEqual(fixed.rows[1].args, (2,))    # nearest valid divisor, not a re-roll
        self.assertGreaterEqual(changed, 2)


class EfficiencyAndStoppingTests(unittest.TestCase):
    def test_power_law_and_pressure(self):
        pts = [(p, 0.05 + 3.0 * p ** -0.5) for p in (100, 300, 1000, 3000, 10_000, 30_000)]
        fit = fit_power_law(pts)
        self.assertAlmostEqual(fit.alpha, 0.5, delta=0.05)
        self.assertAlmostEqual(fit.l_inf, 0.05, delta=0.01)
        model = EfficiencyModel(target_gain=0.02)
        for p, l in pts:
            model.observe(p, l)
        star = model.efficient_size()
        self.assertTrue(1000 < star < 1e6)
        self.assertGreater(model.pressure(star / 8), 0.0)
        self.assertLess(model.pressure(star * 8), 0.0)

    def test_plateau_and_learned_patience(self):
        stats = ContextualStats(pinned=("metric",))
        stopper = PlateauStopper(stats=stats, context={"size": 3}, max_epochs=500)
        epoch = 0
        while not stopper.update(epoch, 1.0 / (1 + epoch) if epoch < 40 else 1.0 / 41):
            epoch += 1
        self.assertEqual(stopper.stop_reason, "plateau")
        self.assertEqual(stopper.best_epoch, 40)
        self.assertEqual(epoch, 40 + stopper.patience)
        stopper.record()
        learned = PlateauStopper(stats=stats, context={"size": 3})
        self.assertIn("gap_mean", learned.learned)
        self.assertLess(learned.patience, 25)
        self.assertEqual(PlateauStopper(max_epochs=3).max_epochs, 5)    # min_epochs floor

    def test_hopeless_needs_equal_effort_deficit(self):
        reference = [1.0 / (1 + t) for t in range(200)]
        behind = PlateauStopper(incumbent=0.005, reference_curve=reference, max_epochs=200)
        epoch = 0
        while not behind.update(epoch, 0.5 + 0.1 / (1 + epoch)):
            epoch += 1
        self.assertEqual(behind.stop_reason, "hopeless")
        self.assertEqual(epoch + 1, behind.hopeless_min_epochs)
        pacing = PlateauStopper(incumbent=0.005, reference_curve=reference, max_epochs=60)
        epoch = 0
        while not pacing.update(epoch, 1.1 / (1 + epoch)):
            epoch += 1
        self.assertNotEqual(pacing.stop_reason, "hopeless")
        self.assertTrue(PlateauStopper().update(0, math.inf))


class SearchTests(unittest.TestCase):
    def test_deterministic_valid_and_learning(self):
        def run(seed):
            space = ArchSpace([("vec", 6)], 2, max_rows=6, max_params=20_000)
            search = ArchitectureSearch(space, fake_evaluate, seed=seed, population=8)
            return search, search.run(generations=6, patience=10)

        a, ra = run(4)
        b, rb = run(4)
        self.assertEqual(ra["best"]["signature"], rb["best"]["signature"])
        self.assertEqual([h["signature"] for h in ra["history"]], [h["signature"] for h in rb["history"]])
        for cand in a.history:
            self.assertEqual(a.space.validate(cand.genome), [])
        self.assertGreater(ra["policy"]["stats"]["records"], 0)
        self.assertEqual(sorted(ra["policy"]["mutation_preference"]), sorted(
            ["arg", "replace", "insert", "delete", "rewire", "crossover"]))
        # Selection prefers the smaller of two equal losses.
        self.assertLess(a.score(0.1, 100), a.score(0.1, 1000))
        # The search moves the frontier: the best is no worse than the first generation's.
        first = min(h["score"] for h in ra["history"][:8] if h["score"] is not None)
        self.assertLessEqual(ra["best"]["score"], first)

    def test_budgets_and_knowledge_roundtrip(self):
        space = ArchSpace([("vec", 3)], 1, max_rows=4, max_params=5000)
        search = ArchitectureSearch(space, fake_evaluate, seed=0, population=4)
        report = search.run(generations=50, max_evaluations=6, patience=50)
        self.assertEqual(report["stop_reason"], "evaluation_budget")
        self.assertEqual(report["evaluations"], 6)
        data = json.loads(json.dumps(search.knowledge.to_dict()))
        again = Knowledge.from_dict(data)
        self.assertEqual(again.policy.count, search.knowledge.policy.count)
        failing = ArchitectureSearch(space, lambda *a: (_ for _ in ()).throw(RuntimeError("boom")),
                                     seed=0, population=3)
        self.assertIsNone(failing.run(generations=1)["best"])


@unittest.skipIf(torch is None, "PyTorch not installed")
class TorchTests(unittest.TestCase):
    def test_build_parity_trace_and_training(self):
        from evolvo.arch.build import build_module, parameter_count, trace_module
        from evolvo.arch.train import TorchTask

        chooser = Chooser(random.Random(5))
        for inputs in ([("vec", 5)], [("set", 4), ("vec", 2)]):
            space = ArchSpace(inputs, 2, max_rows=6, max_params=40_000)
            for i in range(25):
                genome = space.random_genome(chooser)
                module = build_module(space, genome, seed=i)
                self.assertEqual(parameter_count(module), space.lower(genome).params)
                xs = [torch.randn(3, w) if k == "vec" else torch.randn(3, 4, w) for k, w in inputs]
                traced = trace_module(module, xs)
                xs = [torch.randn(6, w) if k == "vec" else torch.randn(6, 4, w) for k, w in inputs]
                self.assertTrue(torch.allclose(traced(*xs), module(*xs), atol=1e-5))

        torch.manual_seed(0)
        x = torch.rand(300, 3) * 2 - 1
        y = (x[:, :1] * x[:, 1:2] + torch.sin(2 * x[:, 2:3])).float()
        task = TorchTask((x[:200], y[:200]), (x[200:], y[200:]), lr=1e-2, hard_max_epochs=150)
        space = ArchSpace([("vec", 3)], 1, max_rows=3, max_params=5000)
        search = ArchitectureSearch(space, task, seed=0, population=4)
        report = search.run(generations=2, patience=5)
        self.assertLess(report["best"]["loss"], float(y[200:].var()))
        self.assertIsNotNone(task.best_module)
        self.assertGreater(search.knowledge.epochs.count, 0)
        self.assertIn(report["best"]["training"]["stop_reason"], ("plateau", "max_epochs"))


if __name__ == "__main__":
    unittest.main()
