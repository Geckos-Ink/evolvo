"""Tests for GFSLPolicyGuide (torch reinforcement-learning mutation guide). Skipped without torch."""

from __future__ import annotations

import os
import random
import tempfile
import unittest
import warnings

import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - the GFSL runtime needs NumPy
    raise unittest.SkipTest("NumPy not installed")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from evolvo import DataType, GFSLEvolver, GFSLExecutor, InstructionProfile


@unittest.skipIf(torch is None, "torch not installed")
class PolicyGuideTests(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", RuntimeWarning)
        from evolvo import GFSLPolicyGuide

        random.seed(0)
        self.Guide = GFSLPolicyGuide
        self.ev = GFSLEvolver(population_size=30)
        self.ev.initialize_population("algorithm", 6, inputs=2, outputs=[(DataType.DECIMAL, 2)], profile=InstructionProfile.formula(allow_custom=False))

    def test_reward_is_relative_and_clipped(self):
        g = self.Guide(device="cpu", seed=1)
        self.assertAlmostEqual(g.reward(1.5, 1.0), 0.5)
        self.assertAlmostEqual(g.reward(-3.0, -2.0), -0.5)
        self.assertEqual(g.reward(1e9, 1.0), 5.0)
        self.assertEqual(g.reward(float("-inf"), 1.0), -5.0)
        self.assertEqual(g.reward(None, 1.0), -5.0)

    def test_children_respect_language_and_run(self):
        g = self.Guide(device="cpu", seed=2)
        ex = GFSLExecutor(compute_backend="cpu")
        allowed = {int(op) for op in InstructionProfile.formula(allow_custom=False).operations}
        for parent in self.ev.population:
            child = g.propose_mutation(self.ev, parent)
            self.assertEqual(child.input_counts, parent.input_counts)
            for ins in child.instructions:
                self.assertIn(ins.operation, allowed)
            out = ex.execute_batch(child, {"d$0": np.linspace(-1, 1, 7), "d$1": 0.5}, track_activity=False)
            self.assertEqual(out["d$2"].shape, (7,))
        self.assertGreater(len(g._pending), 0)

    def test_schedule_lowers_temperature_and_entropy(self):
        g = self.Guide(device="cpu", seed=3, q_target=1.0, temperature=(2.0, 0.5))
        pop = self.ev.population
        for genome, f in zip(pop, np.linspace(0, 0.2, len(pop))):
            genome.fitness = float(f)
        g.observe_population(pop)
        t0, e0 = g.temperature, g.entropy_coef
        for genome, f in zip(pop, np.linspace(0, 0.9, len(pop))):
            genome.fitness = float(f)
        g.observe_population(pop)
        self.assertLess(g.temperature, t0)
        self.assertLess(g.entropy_coef, e0)
        self.assertGreater(g.temperature, 0.5 - 1e-9)

    def test_learns_to_prefer_rewarded_operator(self):
        """Bandit check: children made with 'add' earn reward, others do not -> P(add) rises."""
        from evolvo.policy_guide import OPERATORS

        g = self.Guide(device="cpu", seed=4, batch_min=8, lr=2e-2, uniform_floor=0.02)
        parent = max(self.ev.population, key=lambda x: len(x.instructions))
        parent = self.ev.mutate(parent)
        add_idx = OPERATORS.index("add")

        def p_add():
            feats = torch.as_tensor(g._features(parent))
            with torch.no_grad():
                ctx = g.net.context(feats)
                mask = [True] * len(OPERATORS)
                return float(g._mix_masked(g.net.operator_logits(ctx), mask)[add_idx])

        before = p_add()
        for _ in range(25):
            children = [g.propose_mutation(self.ev, parent) for _ in range(16)]
            for c in children:
                trace = g._pending.get(getattr(c, "_policy_token", None))
                c.fitness = 2.0 if trace is not None and trace["op"] == add_idx else 0.0
            elite = [self.ev.population[0]]
            elite[0].fitness = 1.0
            g.observe_population(children + elite)
        self.assertGreater(g.updates, 5)
        self.assertGreater(p_add(), before + 0.15)

    def test_save_load_roundtrip(self):
        g = self.Guide(device="cpu", seed=5, batch_min=4)
        for genome, f in zip(self.ev.population, np.linspace(0, 1, len(self.ev.population))):
            genome.fitness = float(f)
        kids = [g.propose_mutation(self.ev, p) for p in self.ev.population[:8]]
        for k, f in zip(kids, np.linspace(-1, 2, len(kids))):
            k.fitness = float(f)
        g.observe_population(self.ev.population + kids)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "guide.pt")
            g.save(path)
            h = self.Guide.load(path, device="cpu", q_target=3.0)
            self.assertEqual(h.loaded_from, path)
            self.assertEqual(h.updates, g.updates)
            self.assertEqual(h.q_target, 3.0)
            for (k1, v1), (k2, v2) in zip(g.net.state_dict().items(), h.net.state_dict().items()):
                self.assertEqual(k1, k2)
                self.assertTrue(torch.equal(v1.cpu(), v2.cpu()))
            with self.assertRaises(ValueError):
                self.Guide.load(path, hidden=32)

    def test_plugs_into_evolver(self):
        rng = np.random.default_rng(0)
        x0, x1 = rng.uniform(-2, 2, 200), rng.uniform(-2, 2, 200)
        y = x0 * x1
        ex = GFSLExecutor(compute_backend="cpu")
        guide = self.Guide(device="cpu", seed=6, batch_min=8)
        ev = GFSLEvolver(population_size=40, supervised_guide=guide)
        ev.initialize_population("algorithm", 6, inputs=2, outputs=[(DataType.DECIMAL, 2)], profile=InstructionProfile.formula(allow_custom=False))

        def fitness(genome):
            pred = ex.execute_batch(genome, {"d$0": x0, "d$1": x1}, track_activity=False)["d$2"]
            mse = float(np.mean((pred - y) ** 2))
            return -mse if np.isfinite(mse) else -1e9

        ev.evolve(8, fitness)
        self.assertGreater(guide.observations, 0)
        self.assertGreater(guide.updates, 0)
        self.assertEqual(len(guide.history), guide.observations)


if __name__ == "__main__":
    unittest.main()
