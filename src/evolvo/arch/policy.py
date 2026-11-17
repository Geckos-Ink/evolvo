"""Learned slot-choice policy: dynamic option weights instead of fixed settings.

Every slot choice (op, source distance, width, depth, heads, activation, genome length, even
which mutation to apply) is drawn from the options the language allows, weighted by what the
search has learned so far. The learned value of an option is a :class:`ContextualStats`
estimate of the *reward* that genomes making that choice received (their fitness quantile,
1 = best), with ``choice`` pinned and every other context feature (slot, op, source kind and
width, position, remaining budget, task signature) left to the store to group or ignore.

Sampling is a softmax over ``(estimate + optimism bonus) / temperature`` mixed with a uniform
floor, so rarely tried options keep a chance and no option can be driven to zero. Callers can
add *prior logits* per option, which is how the efficiency model pushes size arguments up or
down.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence

from .language import Chooser, Decision
from .stats import ContextualStats


def _choice_key(value: Any) -> Hashable:
    if isinstance(value, float) and value.is_integer():
        return value
    return value


class AdaptivePolicy(Chooser):
    """A :class:`Chooser` whose option weights are learned from rewards."""

    def __init__(
        self,
        stats: Optional[ContextualStats] = None,
        *,
        rng: Optional[random.Random] = None,
        temperature: float = 0.12,
        explore: float = 0.15,
        optimism: float = 0.5,
    ) -> None:
        super().__init__(rng)
        if temperature <= 0 or not 0 <= explore <= 1 or optimism < 0:
            raise ValueError("temperature > 0, 0 <= explore <= 1 and optimism >= 0 are required")
        self.stats = stats or ContextualStats(pinned=("slot", "choice"), default_mean=0.5,
                                              default_var=0.08)
        self.temperature = float(temperature)
        self.explore = float(explore)
        self.optimism = float(optimism)
        self.priors: Dict[str, Any] = {}

    def weights(self, slot: str, context: Mapping[str, Hashable], options: Sequence[Any],
                prior: Optional[Sequence[float]] = None) -> List[float]:
        logits = []
        for i, option in enumerate(options):
            est = self.stats.estimate({**context, "slot": slot, "choice": _choice_key(option)})
            bonus = self.optimism * math.sqrt(max(est.var, 1e-6)) / math.sqrt(1.0 + est.count)
            logit = (est.mean + bonus) / self.temperature
            if prior is not None:
                logit += prior[i]
            logits.append(logit)
        top = max(logits)
        exp = [math.exp(v - top) for v in logits]
        total = sum(exp)
        n = len(options)
        return [(1.0 - self.explore) * e / total + self.explore / n for e in exp]

    def __call__(self, slot: str, context: Mapping[str, Hashable], options: Sequence[Any]) -> Any:
        if not options:
            raise ValueError(f"no options for {slot}")
        if len(options) == 1:
            return options[0]
        hook = self.priors.get(slot)
        prior = hook(context, options) if callable(hook) else None
        weights = self.weights(slot, context, options, prior)
        r = self.rng.random()
        acc = 0.0
        for option, w in zip(options, weights):
            acc += w
            if r <= acc:
                return option
        return options[-1]

    def update(self, decisions: Iterable[Decision], reward: float, weight: float = 1.0) -> None:
        """Credit every decision of one evaluated genome with its reward (0 worst, 1 best)."""
        for d in decisions:
            self.stats.observe({**d.context, "choice": _choice_key(d.choice)}, reward, weight)

    def preference(self, slot: str, context: Mapping[str, Hashable], options: Sequence[Any]) -> Dict[Any, float]:
        """Current sampling probabilities (for reports)."""
        return dict(zip(options, self.weights(slot, context, options)))


def size_prior(pressure: float, strength: float = 2.0):
    """Prior-logit hook that tilts ascending numeric options by the efficiency pressure.

    ``pressure`` > 0 favours larger options (growing still pays), < 0 smaller ones.
    """

    def hook(context: Mapping[str, Hashable], options: Sequence[Any]) -> Optional[List[float]]:
        if not options or not all(isinstance(o, (int, float)) for o in options):
            return None
        n = len(options)
        if n == 1:
            return [0.0]
        return [strength * pressure * (2.0 * i / (n - 1) - 1.0) for i in range(n)]

    return hook


__all__ = ["AdaptivePolicy", "size_prior"]
