"""Example workflow demonstrating GFSL evolution with the supervised guide."""

import random
from typing import Dict, List, Tuple

import numpy as np
import torch

from evolvo import (
    DataType,
    GFSLExecutor,
    GFSLEvolver,
    GFSLSupervisedGuide,
    RealTimeEvaluator,
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


def main():
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
    main()
