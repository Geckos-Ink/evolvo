"""Automatic epoch stopping that learns how long training tends to keep paying.

:class:`PlateauStopper` watches a validation loss once per epoch and stops for one of:

* ``plateau``: no relative improvement larger than ``rel_tol`` for ``patience`` epochs. Progress
  is measured from the best loss at the last counted improvement, so slow steady gains add up:
  a run whose loss falls by less than ``rel_tol`` per epoch but by more over the patience
  window keeps training;
* ``hopeless``: the run is behind the incumbent (the best run so far) *at equal effort*, i.e.
  its best loss after ``t`` epochs is worse than the incumbent's best after ``t`` epochs by more
  than ``hopeless_margin``, it is also worse than the incumbent's final loss by that margin,
  and a fitted learning curve ``L(t) = a + b t^-alpha`` projects an asymptote that still
  misses it. Then the remaining epochs cannot change the selection. Never before
  ``hopeless_min_epochs``;
* ``diverged``: a non-finite loss;
* ``max_epochs``: the hard cap (a budget guard, never the normal way a run ends).

``patience`` does not have to be set by hand. With a
:class:`~evolvo.arch.stats.ContextualStats` store and a context (task signature, model size
bucket, ...), the stopper reads the learned tendency of *epochs between improvements* for that
kind of run and derives the patience from it; the learned *epoch of the best loss* delays the
hopeless check for slow learners. It then records its own outcome so the next run in a similar
context starts better calibrated. Patience never falls below ``relative_patience`` times the
epochs run so far, because improvements thin out late in training. A run that is still
improving is never stopped by a learned value, only by ``max_epochs``.
"""

from __future__ import annotations

import math
from typing import Dict, Hashable, List, Mapping, Optional, Tuple

from .stats import ContextualStats


def fit_learning_curve(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
    """Least-squares ``loss = a + b * t^-alpha`` over an alpha grid; returns (a, b, alpha)."""
    pts = [(float(t), float(y)) for t, y in points if t > 0 and math.isfinite(y)]
    if len(pts) < 4:
        return None
    best, best_sse = None, math.inf
    for k in range(1, 41):
        alpha = 0.05 * k
        xs = [t ** (-alpha) for t, _ in pts]
        ys = [y for _, y in pts]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx <= 0:
            continue
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        if b < 0:  # a curve rising with t: no decreasing fit
            continue
        a = my - b * mx
        sse = sum((a + b * x - y) ** 2 for x, y in zip(xs, ys))
        if sse < best_sse:
            best, best_sse = (a, b, alpha), sse
    return best


class PlateauStopper:
    """Per-run epoch controller; see the module docstring."""

    def __init__(
        self,
        *,
        max_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        min_epochs: int = 5,
        rel_tol: float = 1e-3,
        incumbent: Optional[float] = None,
        reference_curve: Optional[List[float]] = None,
        hopeless_margin: float = 0.5,
        hopeless_min_epochs: int = 20,
        hopeless_check_every: int = 5,
        stats: Optional[ContextualStats] = None,
        context: Optional[Mapping[str, Hashable]] = None,
        hard_max_epochs: int = 2000,
        min_patience: int = 8,
        max_patience: int = 200,
        default_patience: int = 25,
        relative_patience: float = 0.1,
    ) -> None:
        if min_epochs < 1 or hard_max_epochs < 1 or rel_tol < 0:
            raise ValueError("min_epochs and hard_max_epochs must be positive, rel_tol >= 0")
        self.stats = stats
        self.context = dict(context or {})
        self.rel_tol = float(rel_tol)
        self.min_epochs = int(min_epochs)
        self.incumbent = incumbent if incumbent is not None and math.isfinite(incumbent) else None
        self.hopeless_margin = float(hopeless_margin)
        self.reference = [float(v) for v in reference_curve] if reference_curve else None
        self.hopeless_min_epochs = max(int(hopeless_min_epochs), self.min_epochs)
        self.hopeless_check_every = max(1, int(hopeless_check_every))
        self.learned: Dict[str, float] = {}
        self.relative_patience = max(0.0, float(relative_patience))
        if patience is None:
            patience = default_patience
            if stats is not None:
                est = stats.estimate({**self.context, "metric": "gap"})
                if est.matched >= 1:
                    # Gaps between improvements are heavy-tailed: wait twice a high one.
                    spread = math.sqrt(max(est.var, 0.0))
                    patience = int(math.ceil(2.0 * (est.mean + 2.0 * spread) + 3))
                    self.learned["gap_mean"] = est.mean
            patience = max(min_patience, min(max_patience, patience))
        if stats is not None:
            est = stats.estimate({**self.context, "metric": "best_epoch"})
            if est.matched >= 1:
                self.learned["best_epoch_mean"] = est.mean
        if max_epochs is None:
            max_epochs = hard_max_epochs
        max_epochs = max(self.min_epochs, min(hard_max_epochs, int(max_epochs)))
        self.patience = int(patience)
        self.max_epochs = int(max_epochs)
        self.best = math.inf
        self.best_epoch = -1
        self.last_improvement = -1
        # Best loss when the last improvement was counted: the anchor progress is measured from.
        self.anchor = math.inf
        self.gaps: List[int] = []
        self.curve: List[float] = []
        self.stop_reason: Optional[str] = None
        self.projection: Optional[float] = None

    def update(self, epoch: int, loss: float) -> bool:
        """Record epoch ``epoch``'s validation loss; True when training should stop."""
        if self.stop_reason is not None:
            return True
        loss = float(loss)
        self.curve.append(loss)
        if not math.isfinite(loss):
            self.stop_reason = "diverged"
            return True
        threshold = self.anchor - self.rel_tol * abs(self.anchor) if math.isfinite(self.anchor) else math.inf
        if loss < threshold:
            if self.last_improvement >= 0:
                self.gaps.append(epoch - self.last_improvement)
            self.last_improvement = epoch
            self.anchor = loss
        if loss < self.best:
            self.best = loss
            self.best_epoch = epoch
        if epoch + 1 >= self.max_epochs:
            self.stop_reason = "max_epochs"
        elif epoch + 1 >= self.min_epochs and epoch - self.last_improvement >= max(
                self.patience, int(math.ceil(self.relative_patience * (epoch + 1)))):
            self.stop_reason = "plateau"
        elif self._hopeless(epoch):
            self.stop_reason = "hopeless"
        return self.stop_reason is not None

    def _hopeless(self, epoch: int) -> bool:
        if self.incumbent is None or epoch + 1 < self.hopeless_min_epochs:
            return False
        if (epoch + 1) % self.hopeless_check_every:
            return False
        if "best_epoch_mean" in self.learned and epoch + 1 < 0.25 * self.learned["best_epoch_mean"]:
            return False

        def raised(value: float) -> float:
            return value * (1.0 + self.hopeless_margin) if value >= 0 else value * (1.0 - self.hopeless_margin)

        bar = raised(self.incumbent)
        if self.best <= bar:
            return False
        if self.reference:
            # Equal effort: the incumbent's best loss after the same number of epochs.
            at_same = min(self.reference[: epoch + 1])
            if self.best <= raised(at_same):
                return False
        # Fit the lower envelope (running minimum) of the curve.
        running, pts = math.inf, []
        for t, y in enumerate(self.curve):
            running = min(running, y)
            pts.append((t + 1.0, running))
        fit = fit_learning_curve(pts)
        if fit is None:
            return False
        self.projection = fit[0]
        return min(fit[0], self.best) > bar

    def record(self) -> None:
        """Teach the stats store how this run behaved (call once training ends)."""
        if self.stats is None or self.best_epoch < 0:
            return
        self.stats.observe({**self.context, "metric": "best_epoch"}, float(self.best_epoch))
        for gap in self.gaps:
            self.stats.observe({**self.context, "metric": "gap"}, float(gap))
        if not self.gaps:
            self.stats.observe({**self.context, "metric": "gap"}, float(max(1, self.best_epoch)))

    def summary(self) -> dict:
        return {"stop_reason": self.stop_reason, "epochs": len(self.curve), "best_epoch": self.best_epoch,
                "best_loss": self.best if math.isfinite(self.best) else None, "patience": self.patience,
                "max_epochs": self.max_epochs, "learned": dict(self.learned),
                "projected_asymptote": self.projection}


__all__ = ["PlateauStopper", "fit_learning_curve"]
