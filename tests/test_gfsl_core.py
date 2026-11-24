"""GFSL core tests: declared inputs, instruction profiles, batched execution (scalar parity,
protected numerics, custom ops), the fitness-epoch hook, and an evolver smoke run.

Needs NumPy; run with ``PYTHONPATH=src python -m unittest discover -s tests``.
"""

from __future__ import annotations

import random
import unittest
import uuid
import warnings

import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - the GFSL runtime needs NumPy
    raise unittest.SkipTest("NumPy not installed")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evolvo import (
    Category,
    DataType,
    GFSLEvolver,
    GFSLExecutor,
    GFSLGenome,
    GFSLInstruction,
    InstructionProfile,
    Operation,
    pack_type_index,
    register_custom_operation,
)

SPECIAL = np.array([0.0, -0.0, 1.0, -1.0, 2.0, -2.5, 0.5, 1e-12, -1e-300, 1e308, -1e308, np.inf, -np.inf, np.nan, 3.0, 1e-9])


def _inputs(rng: np.random.Generator, k: int, n: int = 48) -> dict:
    out = {}
    for i in range(k):
        noise = rng.normal(0, 3, size=n)
        out[f"d${i}"] = np.concatenate([noise, rng.permutation(SPECIAL)])
    return out


def _assert_same(test: unittest.TestCase, a: dict, b: dict, genome: GFSLGenome) -> None:
    test.assertEqual(set(a), set(b))
    for key in a:
        ok = np.isclose(a[key], b[key], rtol=1e-12, atol=0.0, equal_nan=True) | (a[key] == b[key])
        if not ok.all():
            bad = int(np.nonzero(~ok)[0][0])
            test.fail(f"{key} row {bad}: vector {a[key][bad]!r} != scalar {b[key][bad]!r}\n" + "\n".join(genome.to_human_readable()))


def _instr(target_idx, op, s1, s2=None):
    """d$target = op(s1, s2) where sources are ('v', idx) variables or ('x', value_index) values."""
    def src(s):
        if s is None:
            return [Category.NONE, 0]
        kind, idx = s
        if kind == "v":
            return [Category.VARIABLE, pack_type_index(DataType.DECIMAL, idx)]
        return [Category.VALUE, idx]

    return GFSLInstruction([Category.VARIABLE, pack_type_index(DataType.DECIMAL, target_idx), op, *src(s1), *src(s2)])


class InputDeclarationTests(unittest.TestCase):
    def test_declared_inputs_are_readable_and_persist(self):
        g = GFSLGenome("algorithm", inputs=2)
        self.assertEqual(g.input_keys(), ["d$0", "d$1"])
        self.assertTrue(g.add_instruction(_instr(2, Operation.ADD, ("v", 0), ("v", 1))))
        g.mark_output(DataType.DECIMAL, 2)
        g.rebuild_validator_state()
        self.assertEqual(g.validator.variable_counts[int(DataType.DECIMAL)], 3)
        out = GFSLExecutor(compute_backend="cpu").execute(g, {"d$0": 1.5, "d$1": 2.0})
        self.assertEqual(out["d$2"], 3.5)

    def test_in_place_update_keeps_its_history(self):
        g = GFSLGenome("algorithm", inputs=2)
        self.assertTrue(g.add_instruction(_instr(2, Operation.MUL, ("v", 0), ("v", 1))))
        self.assertTrue(g.add_instruction(_instr(2, Operation.ADD, ("v", 2), ("v", 0))))
        g.mark_output(DataType.DECIMAL, 2)
        self.assertEqual(g.extract_effective_algorithm(), [0, 1])
        ex = GFSLExecutor(compute_backend="cpu")
        self.assertEqual(ex.execute(g, {"d$0": 3.0, "d$1": 4.0})["d$2"], 15.0)
        out = ex.execute_batch(g, {"d$0": np.array([3.0, 1.0]), "d$1": 4.0}, mode="vector")
        np.testing.assert_array_equal(out["d$2"], [15.0, 5.0])

    def test_without_declaration_inputs_are_not_offered(self):
        g = GFSLGenome("algorithm")
        self.assertFalse(g.add_instruction(_instr(0, Operation.ADD, ("v", 0), ("v", 1))))

    def test_evolver_population_reads_inputs_and_keeps_declarations(self):
        random.seed(3)
        ev = GFSLEvolver(population_size=60)
        ev.initialize_population("algorithm", 10, inputs={DataType.DECIMAL: 3}, outputs=[(DataType.DECIMAL, 3)], profile=InstructionProfile.formula())
        reads = 0
        for g in ev.population:
            self.assertEqual(g.input_counts, {int(DataType.DECIMAL): 3})
            self.assertIn((1, int(DataType.DECIMAL), 3), g.outputs)
            eff = g.extract_effective_algorithm()
            reads += any(
                g.instructions[i].source1_cat == Category.VARIABLE and g.instructions[i].source1_value < 3 for i in eff
            )
        self.assertGreater(reads, 10)
        kids = [ev.mutate(g) for g in ev.population[:20]] + [ev.crossover(ev.population[i], ev.population[i + 1]) for i in range(20)]
        for k in kids:
            self.assertEqual(k.input_counts, {int(DataType.DECIMAL): 3})
            self.assertIsNotNone(k.profile)
        immigrant = ev._random_genome(8, "algorithm")
        self.assertEqual(immigrant.outputs, ev.genome_outputs)
        self.assertEqual(immigrant.input_counts, {int(DataType.DECIMAL): 3})


class ValidatorRegressionTests(unittest.TestCase):
    def test_global_operation_groups_are_not_mutated(self):
        from evolvo.enums import DECIMAL_OPS, TENSOR_OPS

        before = (list(DECIMAL_OPS), list(TENSOR_OPS))
        random.seed(9)
        ev = GFSLEvolver(population_size=40)
        ev.initialize_population("algorithm", 16)  # unrestricted language builds lists and functions
        for g in ev.population[:20]:
            ev.mutate(g)
        self.assertEqual((list(DECIMAL_OPS), list(TENSOR_OPS)), before)


class ProfileTests(unittest.TestCase):
    def test_formula_profile_restricts_language(self):
        random.seed(0)
        profile = InstructionProfile.formula()
        ev = GFSLEvolver(population_size=40)
        ev.initialize_population("algorithm", 12, inputs=2, outputs=[(DataType.DECIMAL, 2)], profile=profile)
        from evolvo.custom_ops import custom_operations
        from evolvo.enums import DECIMAL_OPS

        allowed = {int(op) for op in profile.operations}
        self.assertEqual(allowed, {int(op) for op in DECIMAL_OPS})
        for g in ev.population:
            for ins in g.instructions:
                self.assertEqual(ins.target_cat, Category.VARIABLE)
                self.assertTrue(ins.operation in allowed or custom_operations.get(ins.operation) is not None)
        self.assertEqual(InstructionProfile.from_dict(profile.to_dict()), profile)


class BatchExecutionTests(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", RuntimeWarning)
        self.ex = GFSLExecutor(compute_backend="cpu")

    def test_vector_matches_scalar_on_random_genomes(self):
        random.seed(11)
        rng = np.random.default_rng(11)
        checked = {"vector": 0, "scalar": 0}
        for profile in (InstructionProfile.formula(boolean=True, constants=True), None):
            ev = GFSLEvolver(population_size=150)
            ev.initialize_population("algorithm", 14, inputs=3, outputs=[(DataType.DECIMAL, 3)], profile=profile)
            for g in ev.population + [ev.mutate(g) for g in ev.population[:50]]:
                x = _inputs(rng, 3)
                vec = self.ex.execute_batch(g, x, track_activity=False)
                checked[self.ex.last_batch_mode] += 1
                ref = self.ex.execute_batch(g, x, mode="scalar", track_activity=False)
                _assert_same(self, vec, ref, g)
        self.assertGreater(checked["vector"], 150)
        self.assertGreater(checked["scalar"], 10)  # list/CALL genomes take the scalar path in auto mode

    def test_protected_numerics_every_operation(self):
        ops = [Operation.ADD, Operation.SUB, Operation.MUL, Operation.DIV, Operation.POW, Operation.SQRT, Operation.ABS,
               Operation.SIN, Operation.COS, Operation.EXP, Operation.LOG, Operation.MOD]
        a, b = np.meshgrid(SPECIAL, SPECIAL)
        x = {"d$0": a.ravel(), "d$1": b.ravel()}
        unary = {Operation.SQRT, Operation.ABS, Operation.SIN, Operation.COS, Operation.EXP, Operation.LOG}
        for op in ops:
            g = GFSLGenome("algorithm", inputs=2)
            second = None if op in unary else ("v", 1)
            self.assertTrue(g.add_instruction(_instr(2, op, ("v", 0), second)), op)
            g.mark_output(DataType.DECIMAL, 2)
            vec = self.ex.execute_batch(g, x, mode="vector", track_activity=False)
            ref = self.ex.execute_batch(g, x, mode="scalar", track_activity=False)
            _assert_same(self, vec, ref, g)

    def test_boolean_ops_and_outputs(self):
        for op in (Operation.GT, Operation.LT, Operation.EQ, Operation.GTE, Operation.NEQ):
            g = GFSLGenome("algorithm", inputs=2)
            g.validator.activate_type(DataType.BOOLEAN)
            ins = GFSLInstruction([Category.VARIABLE, pack_type_index(DataType.BOOLEAN, 0), op,
                                   Category.VARIABLE, pack_type_index(DataType.DECIMAL, 0),
                                   Category.VARIABLE, pack_type_index(DataType.DECIMAL, 1)])
            self.assertTrue(g.add_instruction(ins), op)
            g.mark_output(DataType.BOOLEAN, 0)
            a, b = np.meshgrid(SPECIAL, SPECIAL)
            x = {"d$0": a.ravel(), "d$1": b.ravel()}
            vec = self.ex.execute_batch(g, x, mode="vector", track_activity=False)
            ref = self.ex.execute_batch(g, x, mode="scalar", track_activity=False)
            np.testing.assert_array_equal(vec["b$0"], ref["b$0"])
            self.assertEqual(vec["b$0"].dtype, bool)

    def test_shapes_scalars_and_activity(self):
        g = GFSLGenome("algorithm", inputs=2)
        g.add_instruction(_instr(2, Operation.MUL, ("v", 0), ("v", 1)))
        g.mark_output(DataType.DECIMAL, 2)
        out = self.ex.execute_batch(g, {"d$0": np.arange(5.0), "d$1": 2.0})
        np.testing.assert_array_equal(out["d$2"], np.arange(5.0) * 2)
        self.assertEqual(g.instruction_activity[0].hits, 1)  # one batch = one activity record
        with self.assertRaises(ValueError):
            self.ex.execute_batch(g, {"d$0": np.zeros(3), "d$1": np.zeros(4)})
        with self.assertRaises(ValueError):
            self.ex.execute_batch(g, {"d$0": np.zeros((2, 2))})

    def test_custom_operation_batch_function_and_fallback(self):
        tag = uuid.uuid4().hex[:8]

        def scalar_fn(a, b):
            return a * a - b

        vec_code = register_custom_operation(f"sqdiff_{tag}", DataType.DECIMAL, scalar_fn, arity=2,
                                             batch_function=lambda a, b: a * a - b)
        loop_code = register_custom_operation(f"sqdiff_loop_{tag}", DataType.DECIMAL, scalar_fn, arity=2)
        broken_code = register_custom_operation(f"sqdiff_broken_{tag}", DataType.DECIMAL, scalar_fn, arity=2,
                                                batch_function=lambda a, b: 1 / 0)
        x = {"d$0": np.linspace(-3, 3, 31), "d$1": np.linspace(1, 2, 31)}
        expected = x["d$0"] ** 2 - x["d$1"]
        for code in (vec_code, loop_code, broken_code):
            g = GFSLGenome("algorithm", inputs=2)
            self.assertTrue(g.add_instruction(_instr(2, code, ("v", 0), ("v", 1))))
            g.mark_output(DataType.DECIMAL, 2)
            out = self.ex.execute_batch(g, x, track_activity=False)
            self.assertEqual(self.ex.last_batch_mode, "vector")
            np.testing.assert_allclose(out["d$2"], expected, rtol=1e-14)
            _assert_same(self, out, self.ex.execute_batch(g, x, mode="scalar", track_activity=False), g)


class QLearningGuideTests(unittest.TestCase):
    def test_plugs_into_evolver_and_learns(self):
        from evolvo import GFSLQLearningGuide

        warnings.simplefilter("ignore", RuntimeWarning)
        random.seed(4)
        rng = np.random.default_rng(4)
        x0 = rng.uniform(-2, 2, 200)
        y = 3 * x0
        ex = GFSLExecutor(compute_backend="cpu")
        guide = GFSLQLearningGuide()
        ev = GFSLEvolver(population_size=30, supervised_guide=guide)
        ev.initialize_population("algorithm", 5, inputs=1, outputs=[(DataType.DECIMAL, 1)], profile=InstructionProfile.formula(allow_custom=False))

        def fitness(g):
            pred = ex.execute_batch(g, {"d$0": x0}, track_activity=False)["d$1"]
            mse = float(np.mean((pred - y) ** 2))
            return 1.0 / (1.0 + mse) if np.isfinite(mse) else 0.0

        ev.evolve(6, fitness)
        self.assertGreater(guide.observations, 0)
        self.assertGreater(guide.updates, 0)
        self.assertGreater(len(guide.q_table), 0)
        for g in ev.population:
            self.assertEqual(g.input_counts, {int(DataType.DECIMAL): 1})


class FitnessEpochTests(unittest.TestCase):
    def test_epoch_change_rescores_elite_and_resets_tracker(self):
        random.seed(5)
        ev = GFSLEvolver(population_size=12)
        ev.initialize_population("algorithm", 6, inputs=1, outputs=[(DataType.DECIMAL, 1)], profile=InstructionProfile.formula())
        epoch = {"size": 100}
        seen = []

        def evaluator(genome):
            seen.append((ev.generation, id(genome)))
            return random.random() * epoch["size"]

        def epoch_fn():
            if ev.generation == 3:
                epoch["size"] = 1000
            return epoch["size"]

        ev.evolve(6, evaluator, fitness_epoch=epoch_fn)
        self.assertEqual([e for _, e in ev.fitness_epoch_changes], [100, 1000])
        evaluated_gen3 = [gid for gen, gid in seen if gen == 3]
        self.assertEqual(len(evaluated_gen3), ev.population_size)  # whole population, elite included
        for g in ev.population:
            if g.fitness is not None:
                self.assertEqual(g.fitness_epoch, 1000)
        self.assertEqual(ev.evaluations, len(seen))


class EvolverSmokeTests(unittest.TestCase):
    def test_batched_formula_search_improves(self):
        warnings.simplefilter("ignore", RuntimeWarning)
        random.seed(2)
        rng = np.random.default_rng(2)
        x0, x1 = rng.uniform(-2, 2, 400), rng.uniform(-2, 2, 400)
        y = x0 * x1 + x0
        ex = GFSLExecutor(compute_backend="cpu")
        ev = GFSLEvolver(population_size=80)
        ev.initialize_population("algorithm", 6, inputs=2, outputs=[(DataType.DECIMAL, 2)], profile=InstructionProfile.formula())

        def fitness(g):
            pred = ex.execute_batch(g, {"d$0": x0, "d$1": x1}, track_activity=False)["d$2"]
            mse = float(np.mean((pred - y) ** 2))
            return -mse if np.isfinite(mse) else -1e9

        history = []
        ev.evolve(25, fitness, progress_callback=lambda gen, best, f: history.append(f))
        self.assertGreater(history[-1], history[0])
        self.assertGreater(history[-1], -float(np.var(y)))  # beats predicting a constant


if __name__ == "__main__":
    unittest.main()
