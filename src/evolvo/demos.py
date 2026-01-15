"""Legacy example workflows preserved for quick experimentation."""

from __future__ import annotations

from typing import Optional

import numpy as np

_TORCH_IMPORT_ERROR: Optional[ImportError] = None
try:
    import torch
except ImportError as exc:
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def _require_torch(feature: str) -> None:
    if torch is None:
        raise ModuleNotFoundError(
            f"`torch` is required for {feature}. Install it via `pip install torch`."
        ) from _TORCH_IMPORT_ERROR

from .enums import Category, ConfigProperty, DataType, Operation
from .executor import GFSLExecutor
from .genome import GFSLGenome
from .instruction import GFSLInstruction
from .evaluator import RealTimeEvaluator
from .evolver import GFSLEvolver
from .model import RecursiveModelBuilder
from .slots import pack_type_index


def example_formula_discovery():
    """Example: Discover mathematical formulas."""
    print("=== GFSL Formula Discovery ===\n")

    test_cases = []
    expected_outputs = []

    for x in range(-5, 6):
        test_cases.append({"d$0": float(x)})
        expected_outputs.append({"d$1": float(x ** 2 + 2 * x + 1)})

    evaluator_obj = RealTimeEvaluator(test_cases, expected_outputs)

    def fitness_func(genome):
        return evaluator_obj.evaluate(genome)

    evolver = GFSLEvolver(population_size=30)
    evolver.initialize_population("algorithm", initial_instructions=15)

    for genome in evolver.population:
        genome.mark_output(DataType.DECIMAL, 1)

    def progress_callback(gen, best_genome, best_fitness):
        if gen % 20 == 0:
            print(f"\nGeneration {gen}:")
            print(f"  Best fitness: {best_fitness:.6f}")
            print("  Effective algorithm:")
            for line in best_genome.to_human_readable():
                if line.startswith("✓"):
                    print(f"    {line}")

    evolver.evolve(100, fitness_func, progress_callback)

    best = evolver.population[0]
    print("\n=== Best Formula Found ===")
    print(f"Fitness: {best.fitness:.6f}")
    print(f"Signature: {best.get_signature()[:16]}...")
    print("\nFull Algorithm:")
    for line in best.to_human_readable():
        print(f"  {line}")

    print("\n=== Testing Formula ===")
    executor = GFSLExecutor()
    for x in [-2, 0, 1, 3]:
        result = executor.execute(best, {"d$0": float(x)})
        expected = x ** 2 + 2 * x + 1
        print(
            f"  x={x}: Result={result.get('d$1', 0):.2f}, Expected={expected:.2f}"
        )


def example_neural_architecture_search():
    """Example: Evolve neural network architecture."""
    _require_torch("example_neural_architecture_search")
    print("\n=== GFSL Neural Architecture Search ===\n")

    builder = RecursiveModelBuilder()

    genome = GFSLGenome("neural")

    genome.validator.activate_type(DataType.TENSOR)

    set_channels = GFSLInstruction(
        [
            Category.NONE,
            0,
            Operation.SET,
            Category.CONFIG,
            ConfigProperty.CHANNELS,
            Category.VALUE,
            5,
        ]
    )
    genome.add_instruction(set_channels)

    set_kernel = GFSLInstruction(
        [
            Category.NONE,
            0,
            Operation.SET,
            Category.CONFIG,
            ConfigProperty.KERNEL,
            Category.VALUE,
            1,
        ]
    )
    genome.add_instruction(set_kernel)

    conv = GFSLInstruction(
        [
            Category.VARIABLE,
            pack_type_index(DataType.TENSOR, 0),
            Operation.CONV,
            Category.VARIABLE,
            pack_type_index(DataType.TENSOR, 0),
            Category.NONE,
            0,
        ]
    )
    genome.add_instruction(conv)

    relu = GFSLInstruction(
        [
            Category.VARIABLE,
            pack_type_index(DataType.TENSOR, 0),
            Operation.RELU,
            Category.VARIABLE,
            pack_type_index(DataType.TENSOR, 0),
            Category.NONE,
            0,
        ]
    )
    genome.add_instruction(relu)

    model = builder.build_from_genome(genome, (3, 32, 32))

    print("Generated Model Architecture:")
    print(model)

    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    print("\nGFSL Instructions:")
    for line in genome.to_human_readable():
        print(f"  {line}")


__all__ = ["example_formula_discovery", "example_neural_architecture_search"]
