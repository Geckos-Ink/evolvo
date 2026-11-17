"""Learned size efficiency: how much loss another doubling of the model still buys.

Every evaluated architecture contributes ``(parameters, loss)`` under a context (the task
signature, optionally more). For a context the model keeps the *achievable frontier*, the best
loss seen in each half-octave of size, and fits a saturating power law to it:

    L(P) = L_inf + B * P^(-alpha),   L_inf >= 0, B >= 0

(``alpha`` on a grid, ``L_inf``/``B`` by constrained least squares weighted by ``1/L^2``, i.e.
on relative error, so the small losses of the large models, which decide where growth stops
paying, are fitted as carefully as the large losses of tiny models). From the fit it reports:

* ``gain_per_doubling(P)``: the relative loss reduction ``1 - L(2P)/L(P)``;
* ``efficient_size()``: the size P* where that gain falls to ``target_gain``, i.e. where
  growing stops paying for its cost;
* ``pressure(P)`` in [-1, 1]: positive when a candidate of size P is still below P* (grow),
  negative above it (shrink).

The search ties ``target_gain`` to its size cost (``score = ln(loss) + c * ln(params)``: a
doubling pays when it cuts the loss by more than ``1 - 2^-c``), so the pressure and the
selection objective agree. Contexts with too few frontier points back off to all points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Hashable, List, Mapping, Optional, Tuple

from .stats import log_bucket


@dataclass(frozen=True)
class PowerLawFit:
    l_inf: float
    b: float
    alpha: float
    points: int
    rmse: float  # relative (weighted) residual

    def loss(self, params: float) -> float:
        return self.l_inf + self.b * max(params, 1.0) ** (-self.alpha)

    def gain_per_doubling(self, params: float) -> float:
        now = self.loss(params)
        return 0.0 if now <= 0 else 1.0 - self.loss(2.0 * params) / now


def fit_power_law(points: List[Tuple[float, float]]) -> Optional[PowerLawFit]:
    """Fit ``loss = l_inf + b * P^-alpha`` to ``(P, loss)`` points (needs three or more)."""
    pts = [(float(p), float(l)) for p, l in points if p > 0 and math.isfinite(l) and l >= 0]
    if len(pts) < 3 or len({p for p, _ in pts}) < 2:
        return None
    best: Optional[PowerLawFit] = None
    best_sse = math.inf
    ys = [l for _, l in pts]
    floor = max(1e-12, 1e-6 * max(ys))
    ws = [1.0 / max(y, floor) ** 2 for y in ys]
    sw = sum(ws)
    n = len(pts)
    for k in range(1, 61):
        alpha = 0.025 * k
        xs = [p ** (-alpha) for p, _ in pts]
        mx = sum(w * x for w, x in zip(ws, xs)) / sw
        my = sum(w * y for w, y in zip(ws, ys)) / sw
        sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
        if sxx <= 0:
            continue
        b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx
        l_inf = my - b * mx
        if b < 0:
            b, l_inf = 0.0, my
        if l_inf < 0:
            l_inf = 0.0
            b = max(0.0, sum(w * x * y for w, x, y in zip(ws, xs, ys)) /
                    max(sum(w * x * x for w, x in zip(ws, xs)), 1e-300))
        sse = sum(w * (l_inf + b * x - y) ** 2 for w, x, y in zip(ws, xs, ys))
        if sse < best_sse - 1e-15:
            best_sse = sse
            best = PowerLawFit(l_inf, b, alpha, n, math.sqrt(sse / n))
    return best


class EfficiencyModel:
    """Per-context frontier of loss versus size, with a power-law fit and a size pressure."""

    def __init__(self, *, target_gain: float = 0.02, min_points: int = 4, per_octave: int = 2) -> None:
        if not 0 < target_gain < 1:
            raise ValueError("target_gain must be in (0, 1)")
        self.target_gain = float(target_gain)
        self.min_points = int(min_points)
        self.per_octave = int(per_octave)
        self._obs: List[Tuple[Tuple[Tuple[str, Hashable], ...], float, float]] = []

    @staticmethod
    def _key(context: Optional[Mapping[str, Hashable]]) -> Tuple[Tuple[str, Hashable], ...]:
        return tuple(sorted((context or {}).items()))

    def observe(self, params: float, loss: float, context: Optional[Mapping[str, Hashable]] = None) -> None:
        if params > 0 and math.isfinite(loss) and loss >= 0:
            self._obs.append((self._key(context), float(params), float(loss)))

    def frontier(self, context: Optional[Mapping[str, Hashable]] = None) -> List[Tuple[float, float]]:
        key = self._key(context)
        own = [(p, l) for k, p, l in self._obs if k == key]
        pts = own if len({log_bucket(p, self.per_octave) for p, _ in own}) >= self.min_points else \
            [(p, l) for _, p, l in self._obs]
        best: Dict[int, Tuple[float, float]] = {}
        for p, l in pts:
            b = log_bucket(p, self.per_octave)
            if b not in best or l < best[b][1]:
                best[b] = (p, l)
        return sorted(best.values())

    def fit(self, context: Optional[Mapping[str, Hashable]] = None) -> Optional[PowerLawFit]:
        front = self.frontier(context)
        if len(front) < self.min_points:
            return None
        return fit_power_law(front)

    def efficient_size(self, context: Optional[Mapping[str, Hashable]] = None) -> Optional[float]:
        fit = self.fit(context)
        if fit is None or fit.b <= 0:
            return None
        lo, hi = 0.0, 30.0  # log2 of the parameter count
        if fit.gain_per_doubling(2 ** lo) < self.target_gain:
            return 1.0
        if fit.gain_per_doubling(2 ** hi) > self.target_gain:
            return 2 ** hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if fit.gain_per_doubling(2 ** mid) > self.target_gain:
                lo = mid
            else:
                hi = mid
        return 2 ** (0.5 * (lo + hi))

    def pressure(self, params: float, context: Optional[Mapping[str, Hashable]] = None) -> float:
        """+1 grow … 0 neutral/unknown … −1 shrink, for a model of ``params`` parameters."""
        fit = self.fit(context)
        if fit is None:
            return 0.0
        gain = max(fit.gain_per_doubling(params), 1e-9)
        return math.tanh(math.log(gain / self.target_gain))

    def report(self, context: Optional[Mapping[str, Hashable]] = None) -> dict:
        fit = self.fit(context)
        return {
            "target_gain_per_doubling": self.target_gain,
            "observations": len(self._obs),
            "frontier": [[p, l] for p, l in self.frontier(context)],
            "fit": None if fit is None else {"l_inf": fit.l_inf, "b": fit.b, "alpha": fit.alpha,
                                            "points": fit.points, "rmse": fit.rmse},
            "efficient_params": self.efficient_size(context),
        }

    def to_dict(self) -> dict:
        return {"schema": "evolvo_efficiency_v1", "target_gain": self.target_gain,
                "min_points": self.min_points, "per_octave": self.per_octave,
                "observations": [[[[k, v] for k, v in key], p, l] for key, p, l in self._obs]}

    @classmethod
    def from_dict(cls, data: Mapping, *, target_gain: Optional[float] = None) -> "EfficiencyModel":
        if data.get("schema") != "evolvo_efficiency_v1":
            raise ValueError("not an evolvo_efficiency_v1 block")
        model = cls(target_gain=target_gain or data["target_gain"], min_points=data["min_points"],
                    per_octave=data["per_octave"])
        for key, p, l in data["observations"]:
            model._obs.append((tuple((k, v) for k, v in key), float(p), float(l)))
        return model


__all__ = ["EfficiencyModel", "PowerLawFit", "fit_power_law"]
