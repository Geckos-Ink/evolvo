"""Real-time evaluation utilities."""

from typing import Callable, Dict, List, Optional

from .executor import GFSLExecutor
from .genome import GFSLGenome


class RealTimeEvaluator:
    """
    Evaluates genomes multiple times with different inputs,
    supporting real-time scoring and aggregation.
    """

    def __init__(
        self,
        test_cases: List[Dict[str, float]],
        expected_outputs: Optional[List[Dict[str, float]]] = None,
        score_aggregator: Optional[Callable] = None,
    ):
        """
        Args:
            test_cases: List of input dictionaries for each test
            expected_outputs: Optional expected outputs for each test
            score_aggregator: Custom function to aggregate scores across tests
        """
        self.test_cases = test_cases
        self.expected_outputs = expected_outputs or [{}] * len(test_cases)
        self.score_aggregator = score_aggregator or self._default_aggregator
        self.executor = GFSLExecutor()

    def _default_aggregator(self, scores: List[float]) -> float:
        """Default score aggregation: mean with penalty for failures."""
        if not scores:
            return -float("inf")

        valid_scores = [s for s in scores if s != -float("inf")]
        if not valid_scores:
            return -float("inf")

        success_rate = len(valid_scores) / len(scores)
        mean_score = sum(valid_scores) / len(valid_scores)

        return mean_score * success_rate

    def evaluate(
        self,
        genome: GFSLGenome,
        callback: Optional[Callable[[int, Dict, float], None]] = None,
    ) -> float:
        """
        Evaluate genome on all test cases.

        Args:
            genome: The genome to evaluate
            callback: Optional callback(test_idx, output, score) for real-time feedback

        Returns:
            Aggregated fitness score
        """
        scores = []

        for idx, (inputs, expected) in enumerate(
            zip(self.test_cases, self.expected_outputs)
        ):
            try:
                outputs = self.executor.execute(genome, inputs)
                score = self._calculate_score(outputs, expected)
                scores.append(score)

                if callback:
                    callback(idx, outputs, score)

            except Exception:
                scores.append(-float("inf"))
                if callback:
                    callback(idx, {}, -float("inf"))

        return self.score_aggregator(scores)

    def _calculate_score(self, outputs: Dict[str, float], expected: Dict[str, float]) -> float:
        """Calculate score for a single test case."""
        if not expected:
            return 1.0 if outputs else 0.0

        total_error = 0.0
        for key, expected_val in expected.items():
            if key in outputs:
                error = abs(outputs[key] - expected_val)
                total_error += error
            else:
                total_error += abs(expected_val)

        return 1.0 / (1.0 + total_error)


__all__ = ["RealTimeEvaluator"]
