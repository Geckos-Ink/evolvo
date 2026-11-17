"""Contextual statistics whose group types are discovered, not enumerated.

``ContextualStats`` learns the tendency of a value (a reward, an epoch count, a loss) as a
function of a *context*: a mapping of feature name to a hashable value. Nothing declares which
combinations of features form a "group". Instead the store:

1. keeps the raw observations (a bounded sliding window, so tendencies can drift);
2. orders the context features greedily by how much *extra* variance each explains once the
   features before it are known, after paying for the groups it adds (an adjusted
   between-group sum of squares, the same quantity an F-test uses), and re-orders every
   ``reorder_every`` observations;
3. aggregates the observations at every prefix of that order, so the groups are the cells of a
   hierarchy that the data chose: ``()`` → ``(f1,)`` → ``(f1, f2)`` → …;
4. estimates a context by walking down that hierarchy and shrinking each level towards the one
   above (empirical-Bayes backoff with ``prior_strength`` pseudo-observations). A sparse,
   specific group borrows from its parents; a well-observed group speaks for itself.

``pinned`` features always lead the order. A policy pins ``choice`` so every estimate is "the
value of this option" refined by whichever context matters most.

Pure Python (no NumPy), deterministic, JSON round-trippable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

ContextKey = Tuple[Tuple[str, Hashable], ...]

_NEG_BUCKET = -(10 ** 6)


def log_bucket(value: float, per_octave: int = 2) -> int:
    """Discretize a positive magnitude on a log2 ladder (``per_octave`` buckets per doubling).

    Zero, negative and non-finite values share one sentinel bucket so they never collide with a
    real magnitude.
    """
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        return _NEG_BUCKET
    return int(math.floor(math.log2(value) * per_octave + 1e-9))


class _Acc:
    """Weighted Welford accumulator."""

    __slots__ = ("n", "mean", "m2")

    def __init__(self) -> None:
        self.n = 0.0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, x: float, w: float = 1.0) -> None:
        self.n += w
        d = x - self.mean
        self.mean += w * d / self.n
        self.m2 += w * d * (x - self.mean)

    @property
    def var(self) -> float:
        return self.m2 / self.n if self.n > 0 else 0.0


@dataclass(frozen=True)
class Estimate:
    """A shrunk estimate: ``mean``/``var`` blended down the hierarchy.

    ``count`` is the weight of the most specific group of the context (0 when that group has
    never been observed, so a policy can treat it as untried), ``matched`` how many ordered
    features the deepest *observed* group fixes, ``evidence`` that group's weight, and ``path``
    the feature names walked.
    """

    mean: float
    var: float
    count: float
    depth: int
    path: Tuple[str, ...]
    matched: int = 0
    evidence: float = 0.0


def _key(context: Mapping[str, Hashable]) -> ContextKey:
    items = []
    for name, value in context.items():
        if not isinstance(name, str) or not name:
            raise ValueError("context feature names must be non-empty strings")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"context feature {name!r} is not finite")
        hash(value)
        items.append((name, value))
    return tuple(sorted(items, key=lambda kv: kv[0]))


class ContextualStats:
    """Hierarchical statistics over dynamically discovered context groups."""

    def __init__(
        self,
        *,
        pinned: Sequence[str] = (),
        prior_strength: float = 3.0,
        max_records: int = 20000,
        reorder_every: int = 50,
        default_mean: float = 0.0,
        default_var: float = 1.0,
    ) -> None:
        if prior_strength <= 0 or not math.isfinite(prior_strength):
            raise ValueError("prior_strength must be positive")
        if max_records < 1 or reorder_every < 1:
            raise ValueError("max_records and reorder_every must be positive")
        self.pinned: Tuple[str, ...] = tuple(pinned)
        self.prior_strength = float(prior_strength)
        self.max_records = int(max_records)
        self.reorder_every = int(reorder_every)
        self.default_mean = float(default_mean)
        self.default_var = float(default_var)
        self._records: List[Tuple[ContextKey, float, float]] = []
        self._features: List[str] = []
        self._order: List[str] = []
        self._gains: Dict[str, float] = {}
        self._global = _Acc()
        self._levels: List[Dict[tuple, _Acc]] = []
        self._dirty = False
        self._since_reorder = 0

    # ------------------------------------------------------------------ observation
    def observe(self, context: Mapping[str, Hashable], value: float, weight: float = 1.0) -> None:
        value = float(value)
        weight = float(weight)
        if not math.isfinite(value) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("observations need a finite value and a positive finite weight")
        key = _key(context)
        self._records.append((key, value, weight))
        for name, _ in key:
            if name not in self._features:
                # A new feature joins the hierarchy at once (last); the next reorder places it.
                self._features.append(name)
                if name not in self.pinned:
                    self._order.append(name)
                self._dirty = True
        if len(self._records) > self.max_records:
            del self._records[: len(self._records) - self.max_records]
            self._dirty = True
        self._since_reorder += 1
        # Re-learn the grouping after a fixed number of observations or a quarter of the
        # window, whichever is larger, so the cost of discovery stays logarithmic in history.
        if self._since_reorder >= max(self.reorder_every, len(self._records) // 4):
            self.reorder()
        elif self._dirty:
            self._rebuild()
        else:
            self._add(key, value, weight)

    def _add(self, key: ContextKey, value: float, weight: float) -> None:
        self._global.add(value, weight)
        values = dict(key)
        order = self.order()
        prefix: List[Hashable] = []
        for depth, name in enumerate(order):
            if name not in values:
                break
            prefix.append(values[name])
            self._levels[depth].setdefault(tuple(prefix), _Acc()).add(value, weight)

    def _rebuild(self) -> None:
        self._global = _Acc()
        self._levels = [dict() for _ in self.order()]
        for key, value, weight in self._records:
            self._add(key, value, weight)
        self._dirty = False

    # ------------------------------------------------------------------ group discovery
    def order(self) -> List[str]:
        """Feature order of the group hierarchy: pinned features, then the learned order."""
        pinned = [name for name in self.pinned if name in self._features]
        return pinned + [name for name in self._order if name not in pinned]

    @staticmethod
    def _ss_between(rows, prefix: Sequence[str]) -> Tuple[float, int, float]:
        """Between-group sum of squares, group count and within SS for groups over ``prefix``.

        Records missing any prefix feature are grouped under a "missing" marker so every
        candidate is compared on the same observations.
        """
        groups: Dict[tuple, _Acc] = {}
        total = _Acc()
        for values, value, weight in rows:
            gkey = tuple(values.get(name, ("<missing>",)) for name in prefix)
            groups.setdefault(gkey, _Acc()).add(value, weight)
            total.add(value, weight)
        between = sum(acc.n * (acc.mean - total.mean) ** 2 for acc in groups.values())
        within = sum(acc.m2 for acc in groups.values())
        return between, len(groups), within

    def reorder(self) -> None:
        """Re-learn which context features define groups, and in which order."""
        self._since_reorder = 0
        pinned = [name for name in self.pinned if name in self._features]
        remaining = sorted(name for name in self._features if name not in pinned)
        order: List[str] = []
        gains: Dict[str, float] = {}
        rows = [(dict(key), value, weight) for key, value, weight in self._records]
        n = sum(w for _, _, w in rows)
        base_between, base_groups, _ = self._ss_between(rows, pinned)
        while remaining:
            best_name, best_gain, best_state = None, -math.inf, None
            for name in remaining:
                between, groups, within = self._ss_between(rows, pinned + order + [name])
                dof = max(1.0, n - groups)
                msw = within / dof
                # Variance explained beyond the current prefix, minus what that many extra
                # groups would explain by chance: only real structure creates groups.
                gain = (between - base_between) - (groups - base_groups) * msw
                if gain > best_gain or (gain == best_gain and best_name is not None and name < best_name):
                    best_name, best_gain, best_state = name, gain, (between, groups)
            order.append(best_name)
            gains[best_name] = best_gain
            remaining.remove(best_name)
            base_between, base_groups = best_state
        self._order = order
        self._gains = gains
        self._rebuild()

    # ------------------------------------------------------------------ estimation
    def estimate(self, context: Mapping[str, Hashable]) -> Estimate:
        if self._dirty:
            self._rebuild()
        values = dict(_key(context))
        if self._global.n <= 0:
            return Estimate(self.default_mean, self.default_var, 0.0, 0, ())
        kappa = self.prior_strength
        mean = self._global.mean
        var = self._global.var if self._global.n > 1 else self.default_var
        count = self._global.n
        path: List[str] = []
        prefix: List[Hashable] = []
        matched, evidence = 0, self._global.n
        for depth, name in enumerate(self.order()):
            if name not in values:
                break
            prefix.append(values[name])
            acc = self._levels[depth].get(tuple(prefix))
            if acc is None:
                # An unseen specific group: inherit the parent's tendency with no own evidence.
                count = 0.0
                path.append(name)
                break
            group_var = acc.var if acc.n > 1 else var
            mean = (acc.n * acc.mean + kappa * mean) / (acc.n + kappa)
            var = (acc.n * group_var + kappa * var) / (acc.n + kappa)
            count = acc.n
            matched, evidence = depth + 1, acc.n
            path.append(name)
        return Estimate(mean, var, count, len(path), tuple(path), matched, evidence)

    # ------------------------------------------------------------------ reporting
    @property
    def count(self) -> int:
        return len(self._records)

    def groups(self, min_count: float = 3.0, limit: int = 50) -> List[dict]:
        """The most specific discovered groups with at least ``min_count`` weight."""
        if self._dirty:
            self._rebuild()
        order = self.order()
        out = []
        for depth in range(len(self._levels) - 1, -1, -1):
            for key, acc in self._levels[depth].items():
                if acc.n >= min_count:
                    out.append({"features": dict(zip(order[: depth + 1], key)),
                                "count": acc.n, "mean": acc.mean, "std": math.sqrt(acc.var)})
        out.sort(key=lambda g: (-len(g["features"]), -g["count"]))
        return out[:limit]

    def summary(self) -> dict:
        if self._dirty:
            self._rebuild()
        return {
            "records": len(self._records),
            "order": self.order(),
            "gains": {name: self._gains.get(name) for name in self.order() if name in self._gains},
            "groups_per_level": [len(level) for level in self._levels],
            "mean": self._global.mean if self._global.n else None,
        }

    def to_dict(self) -> dict:
        return {
            "schema": "evolvo_contextual_stats_v1",
            "pinned": list(self.pinned),
            "prior_strength": self.prior_strength,
            "max_records": self.max_records,
            "reorder_every": self.reorder_every,
            "default_mean": self.default_mean,
            "default_var": self.default_var,
            "records": [[[[k, v] for k, v in key], value, weight] for key, value, weight in self._records],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "ContextualStats":
        if data.get("schema") != "evolvo_contextual_stats_v1":
            raise ValueError("not an evolvo_contextual_stats_v1 block")
        stats = cls(pinned=data.get("pinned", ()), prior_strength=data.get("prior_strength", 3.0),
                    max_records=data.get("max_records", 20000),
                    reorder_every=data.get("reorder_every", 50),
                    default_mean=data.get("default_mean", 0.0),
                    default_var=data.get("default_var", 1.0))
        for key, value, weight in data.get("records", []):
            context = {k: (tuple(v) if isinstance(v, list) else v) for k, v in key}
            k = _key(context)
            stats._records.append((k, float(value), float(weight)))
            for name, _ in k:
                if name not in stats._features:
                    stats._features.append(name)
        stats.reorder()
        return stats

    def merge(self, records: Iterable[Tuple[Mapping[str, Hashable], float]]) -> None:
        for context, value in records:
            self.observe(context, value)


__all__ = ["ContextualStats", "Estimate", "log_bucket"]
