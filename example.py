"""
Example workflow demonstrating GFSL evolution with the supervised guide.

Usage:
  python example.py
  python example.py personalization
  python example.py both
  python example.py expression
  python example.py neural
  python example.py extraction
"""

import random
import runpy
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_TORCH_IMPORT_ERROR = None
try:
    import torch
except ImportError as exc:
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc


def _require_torch(feature: str) -> None:
    if torch is None:
        raise ModuleNotFoundError(
            f"`torch` is required for {feature}. Install it via `pip install torch`."
        ) from _TORCH_IMPORT_ERROR

SRC_PATH = Path(__file__).resolve().parent / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from evolvo import (
    Category,
    DataType,
    GFSLExecutor,
    GFSLGenome,
    GFSLInstruction,
    GFSLEvolver,
    GFSLSupervisedGuide,
    RealTimeEvaluator,
    pack_type_index,
    register_custom_operation,
)


def build_formula_dataset() -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """Generate training data for a quadratic formula."""
    test_cases: List[Dict[str, float]] = []
    expected: List[Dict[str, float]] = []
    for x in np.linspace(-3.0, 3.0, num=13):
        test_cases.append({"d$0": float(x)})
        # Target function y = 0.5*x^2 - x + 2
        expected.append({"d$1": float(0.5 * x * x - x + 2.0)})
    return test_cases, expected


def personalization_demo() -> None:
    """Show how to register an ad-hoc decimal operation and execute it."""

    def affine_bias(value: float, bias: float, context=None) -> float:
        """Simple affine transform used by the custom opcode."""
        return float(value) * 1.5 + float(bias if bias is not None else 0.0)

    opcode = register_custom_operation(
        "affine_bias",
        target_type=DataType.DECIMAL,
        function=affine_bias,
        arity=2,
        value_enumeration=[-1.0, 0.0, 0.5, 2.0],
        doc="Scales the first source and adds a bias selected from the enumeration.",
    )

    genome = GFSLGenome("algorithm")
    # Make d$0 available as an input slot without assigning it first.
    genome.validator.variable_counts[int(DataType.DECIMAL)] = 1
    instruction = GFSLInstruction(
        [
            Category.VARIABLE,
            pack_type_index(DataType.DECIMAL, 1),
            opcode,
            Category.VARIABLE,
            pack_type_index(DataType.DECIMAL, 0),
            Category.VALUE,
            3,  # Uses the inline value 2.0 as the bias term
        ]
    )
    if not genome.add_instruction(instruction):
        raise RuntimeError("Failed to insert custom instruction into the genome.")
    genome.mark_output(DataType.DECIMAL, 1)

    executor = GFSLExecutor()
    print("\nCustom operation demonstration:")
    for x in [0.0, 1.0, -2.0]:
        outputs = executor.execute(genome, {"d$0": x})
        print(f"  x={x:+.1f} -> y={outputs.get('d$1', float('nan')):+.3f}")

    print("Human-readable trace:")
    for line in genome.to_human_readable():
        print(" ", line)


def print_usage() -> None:
    print(
        "Usage: python example.py [mode]\n"
        "  evolution (default)  Run the supervised evolution demo\n"
        "  personalization      Run the custom-operation demo\n"
        "  both                 Run personalization then evolution\n"
        "  expression           Run examples/expression_flow.py\n"
        "  neural               Run examples/neural_flow.py\n"
        "  extraction           Run examples/operation_extraction.py\n"
        "\n"
        "You can also run the scripts directly:\n"
        "  python examples/expression_flow.py\n"
        "  python examples/neural_flow.py\n"
        "  python examples/operation_extraction.py\n"
    )


def run_example_script(filename: str) -> None:
    script_path = Path(__file__).resolve().parent / "examples" / filename
    if not script_path.exists():
        raise FileNotFoundError(f"Example script not found: {script_path}")
    runpy.run_path(str(script_path), run_name="__main__")


def main():
    _require_torch("example.py supervised evolution demo")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    test_cases, expected_outputs = build_formula_dataset()
    evaluator = RealTimeEvaluator(test_cases, expected_outputs)

    guide = GFSLSupervisedGuide(
        hidden_layers=[256, 128],
        buffer_size=512,
        min_buffer=64,
        candidate_pool=4,
    )

    evolver = GFSLEvolver(population_size=32, supervised_guide=guide)
    evolver.initialize_population("algorithm", initial_instructions=12)

    for genome in evolver.population:
        genome.mark_output(DataType.DECIMAL, 1)

    def fitness(genome):
        return evaluator.evaluate(genome)

    def progress_callback(gen, best, best_fitness):
        if gen % 10 == 0:
            guide_estimate = float("nan")
            if guide.trained:
                guide_estimate = float(guide.predict([best])[0])
            print(
                f"Gen {gen:03d} | fitness={best_fitness:.6f} | guide≈{guide_estimate:.6f}"
            )

    evolver.evolve(60, fitness, progress_callback)

    best = evolver.population[0]
    print("\nBest genome fitness:", best.fitness)
    print("Signature:", best.get_signature()[:16] + "...")

    print("\nEffective instruction trace:")
    for line in best.to_human_readable():
        if line.startswith("✓"):
            print("  " + line)

    executor = GFSLExecutor()
    print("\nSpot-check predictions:")
    for x in [-2.0, -0.5, 1.5, 2.5]:
        result = executor.execute(best, {"d$0": float(x)})
        predicted = result.get("d$1", 0.0)
        expected = 0.5 * x * x - x + 2.0
        print(f"  x={x:+.2f} | predicted={predicted:.4f} | expected={expected:.4f}")

    if guide.loss_history:
        print(
            f"\nGuide trained for {len(guide.loss_history)} epochs, "
            f"last loss={guide.loss_history[-1]:.5f}"
        )


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "evolution"
    if mode in {"-h", "--help", "help", "examples"}:
        print_usage()
        raise SystemExit(0)
    if mode == "personalization":
        personalization_demo()
    elif mode == "both":
        personalization_demo()
        main()
    elif mode in {"expression", "expression_flow"}:
        run_example_script("expression_flow.py")
    elif mode in {"neural", "neural_flow"}:
        run_example_script("neural_flow.py")
    elif mode in {"extraction", "operation", "operation_extraction"}:
        run_example_script("operation_extraction.py")
    elif mode == "evolution":
        main()
    else:
        print_usage()
        raise SystemExit(1)
