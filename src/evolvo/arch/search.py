"""Evolutionary architecture search over the GFSL-NN language.

:class:`ArchitectureSearch` evolves :class:`~evolvo.arch.language.ArchGenome` populations.
Each part learns while it runs:

* **Slot choices** come from an :class:`~evolvo.arch.policy.AdaptivePolicy`. After every
  evaluation, each effective decision of the genome (op, source distance, width, depth, heads,
  activation, length) is credited with the genome's fitness quantile among all evaluations.
* **Mutation operators** (argument step, row replacement, insertion, deletion, rewiring,
  crossover) are chosen by the same policy (slot ``mutation``), credited with the child's
  quantile, so the search learns which edits pay on this problem.
* **Size** is steered by an :class:`~evolvo.arch.efficiency.EfficiencyModel`: its pressure
  tilts size arguments and argument steps up or down, and the selection score
  ``ln(loss) + size_cost * ln(params)`` prefers the smaller of two equally good models.
* **The language** widens its width window towards sizes the elite uses at an edge.
* **Training** stops per candidate when learning stops paying (``TorchTask`` +
  ``PlateauStopper``); the search itself stops when the best score has not improved by
  ``min_improvement`` for ``patience`` generations (``converged``), or on a budget.

Everything the search learned (policy statistics, epoch statistics, efficiency observations)
is a :class:`Knowledge` that can be saved and loaded, so the next search on a similar task
starts from those tendencies rather than from uniform choices.

Determinism: one ``random.Random(seed)`` drives every choice; candidate ``i`` gets the torch
seed ``seed * 1_000_003 + i``.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .efficiency import EfficiencyModel
from .language import NONE, ArchGenome, ArchSpace, Row
from .policy import AdaptivePolicy, size_prior
from .stats import ContextualStats, log_bucket

MUTATIONS = ("arg", "replace", "insert", "delete", "rewire", "crossover")
SIZE_SLOTS = ("arg:units", "arg:dim", "arg:depth", "arg:ratio", "arg:ff")


@dataclass
class Knowledge:
    """Learned tendencies that outlive one search."""

    policy: ContextualStats = field(default_factory=lambda: ContextualStats(
        pinned=("slot", "choice"), default_mean=0.5, default_var=0.08))
    epochs: ContextualStats = field(default_factory=lambda: ContextualStats(pinned=("metric",)))
    efficiency: Optional[EfficiencyModel] = None

    def to_dict(self) -> dict:
        return {"schema": "evolvo_arch_knowledge_v1", "policy": self.policy.to_dict(),
                "epochs": self.epochs.to_dict(),
                "efficiency": None if self.efficiency is None else self.efficiency.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping) -> "Knowledge":
        if data.get("schema") != "evolvo_arch_knowledge_v1":
            raise ValueError("not an evolvo_arch_knowledge_v1 file")
        eff = data.get("efficiency")
        return cls(ContextualStats.from_dict(data["policy"]), ContextualStats.from_dict(data["epochs"]),
                   None if eff is None else EfficiencyModel.from_dict(eff))

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Knowledge":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class Candidate:
    genome: ArchGenome
    signature: str
    origin: str
    generation: int
    index: int
    params: int = 0
    loss: float = math.inf
    score: float = math.inf
    result: Dict[str, Any] = field(default_factory=dict)
    cached: bool = False
    parent_score: Optional[float] = None

    def brief(self) -> dict:
        return {"index": self.index, "generation": self.generation, "origin": self.origin,
                "signature": self.signature, "params": self.params,
                "loss": self.loss if math.isfinite(self.loss) else None,
                "score": self.score if math.isfinite(self.score) else None,
                "epochs": self.result.get("epochs"), "stop_reason": self.result.get("stop_reason"),
                "cached": self.cached}


Evaluate = Callable[[ArchSpace, ArchGenome, Mapping[str, Any]], Mapping[str, Any]]


class ArchitectureSearch:
    """See the module docstring. ``evaluate(space, genome, info)`` returns at least ``loss``
    and ``params``; ``info`` carries ``seed``, ``incumbent`` (best loss so far) and
    ``signature``. :class:`~evolvo.arch.train.TorchTask` is the PyTorch evaluator."""

    def __init__(
        self,
        space: ArchSpace,
        evaluate: Evaluate,
        *,
        seed: int = 0,
        population: int = 12,
        elite: Optional[int] = None,
        tournament: int = 3,
        size_cost: float = 0.03,
        knowledge: Optional[Knowledge] = None,
        temperature: float = 0.12,
        explore: float = 0.15,
    ) -> None:
        if elite is None:
            elite = max(1, population // 4)
        if population < 2 or not 1 <= elite < population or tournament < 1:
            raise ValueError("population >= 2, 1 <= elite < population and tournament >= 1")
        if size_cost < 0 or not math.isfinite(size_cost):
            raise ValueError("size_cost must be finite and non-negative")
        self.space = space
        self.evaluate = evaluate
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.population_size = int(population)
        self.elite = int(elite)
        self.tournament = int(tournament)
        self.size_cost = float(size_cost)
        self.knowledge = knowledge or Knowledge()
        target_gain = 1.0 - 2.0 ** (-max(size_cost, 1e-4))
        if self.knowledge.efficiency is None:
            self.knowledge.efficiency = EfficiencyModel(target_gain=target_gain)
        else:
            self.knowledge.efficiency.target_gain = target_gain
        if hasattr(evaluate, "epoch_stats"):
            # One store of epoch tendencies: the evaluator's stopper reads and teaches it, the
            # knowledge saves it.
            if knowledge is None:
                self.knowledge.epochs = evaluate.epoch_stats
            else:
                evaluate.epoch_stats = self.knowledge.epochs
        self.policy = AdaptivePolicy(self.knowledge.policy, rng=self.rng,
                                     temperature=temperature, explore=explore)
        self.cache: Dict[str, Candidate] = {}
        self.history: List[Candidate] = []
        self.scores: List[float] = []
        self._pressure = 0.0

    # ------------------------------------------------------------------ scoring
    def score(self, loss: float, params: int) -> float:
        if not math.isfinite(loss):
            return math.inf
        return math.log(max(loss, 1e-12)) + self.size_cost * math.log(max(params, 1))

    def quantile(self, score: float) -> float:
        """1 for the best score seen so far, 0 for the worst (or a failure)."""
        finite = [s for s in self.scores if math.isfinite(s)]
        if not math.isfinite(score) or not finite:
            return 0.0
        if len(finite) == 1:
            return 0.5
        worse = sum(1 for s in finite if s > score)
        ties = sum(1 for s in finite if s == score) - 1
        return (worse + 0.5 * ties) / (len(finite) - 1)

    def _set_pressure(self, params: Optional[float]) -> None:
        eff = self.knowledge.efficiency
        self._pressure = 0.0 if params is None else eff.pressure(params, self.space.task)
        for slot in SIZE_SLOTS:
            self.policy.priors[slot] = size_prior(self._pressure)

    # ------------------------------------------------------------------ evaluation
    def _evaluate(self, genome: ArchGenome, origin: str, generation: int,
                  parent_score: Optional[float]) -> Candidate:
        signature = self.space.signature(genome)
        cand = Candidate(genome, signature, origin, generation, len(self.history), parent_score=parent_score)
        hit = self.cache.get(signature)
        if hit is not None:
            cand.params, cand.loss, cand.score, cand.result, cand.cached = hit.params, hit.loss, hit.score, hit.result, True
        else:
            finite = [c.loss for c in self.cache.values() if math.isfinite(c.loss)]
            info = {"seed": self.seed * 1_000_003 + cand.index, "signature": signature,
                    "incumbent": min(finite) if finite else None, "generation": generation}
            try:
                result = dict(self.evaluate(self.space, genome, info))
                cand.loss = float(result.get("loss", math.inf))
                cand.params = int(result.get("params", self.space.lower(genome).params))
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                result = {"error": str(exc), "stop_reason": "error"}
                cand.loss = math.inf
                cand.params = self.space.lower(genome).params
            cand.result = result
            # A run stopped as hopeless is a censored loss: it ranks, but it would bias the
            # size frontier, which needs converged losses.
            cand.score = self.score(cand.loss, cand.params)
            self.cache[signature] = cand
            if math.isfinite(cand.loss) and result.get("stop_reason") != "hopeless":
                self.knowledge.efficiency.observe(cand.params, cand.loss, self.space.task)
        self.history.append(cand)
        self.scores.append(cand.score)
        reward = self.quantile(cand.score)
        if not cand.cached:
            self.policy.update(self.space.decisions(cand.genome), reward)
        if origin.startswith("mutate:") and parent_score is not None:
            kind = origin.split(":", 1)[1]
            improved = 1.0 if cand.score < parent_score else (0.5 if cand.score == parent_score else 0.0)
            self.policy.update([_mutation_decision(self.space, kind, self._pressure)], 0.5 * (reward + improved))
        return cand

    # ------------------------------------------------------------------ variation
    def _tournament(self, pool: Sequence[Candidate]) -> Candidate:
        picks = [pool[self.rng.randrange(len(pool))] for _ in range(min(self.tournament, len(pool)))]
        return min(picks, key=lambda c: (c.score, c.params, c.index))

    def _vary(self, parent: Candidate, pool: Sequence[Candidate]) -> (ArchGenome, str):
        self._set_pressure(parent.params)
        ctx = _mutation_context(self.space, self._pressure)
        kinds = list(MUTATIONS) if len(pool) > 1 else [k for k in MUTATIONS if k != "crossover"]
        kind = self.policy("mutation", ctx, kinds)
        rows = list(parent.genome.rows)
        n_in = len(self.space.inputs)
        eff = self.space.effective_rows(parent.genome) or list(range(len(rows)))
        if kind == "arg" and rows:
            with_args = [i for i in eff if self.space.registry.get(rows[i].op).args]
            if with_args:
                i = with_args[self.rng.randrange(len(with_args))]
                rows[i] = self._step_arg(rows, i)
        elif kind == "replace" and rows:
            i = eff[self.rng.randrange(len(eff))]
            rows[i] = Row("<resample>", rows[i].a)
        elif kind == "insert" and len(rows) < self.space.max_rows:
            p = self.rng.randrange(len(rows) + 1)
            shifted = [_shift(r, n_in + p, +1) for r in rows[p:]]
            new = Row("<resample>", n_in + p - 1 if p > 0 else 0)
            rows = rows[:p] + [new] + shifted
            if p + 1 < len(rows) and self.rng.random() < 0.7:
                nxt = rows[p + 1]
                rows[p + 1] = Row(nxt.op, n_in + p, nxt.b, nxt.args)
        elif kind == "delete" and len(rows) > 1:
            p = eff[self.rng.randrange(len(eff))] if eff else len(rows) - 1
            bypass = rows[p].a
            rest = []
            for r in rows[p + 1:]:
                a = bypass if r.a == n_in + p else r.a
                b = bypass if r.b == n_in + p else r.b
                rest.append(_shift(Row(r.op, a, b, r.args), n_in + p + 1, -1))
            rows = rows[:p] + rest
        elif kind == "rewire" and rows:
            i = eff[self.rng.randrange(len(eff))]
            r = rows[i]
            target = self.rng.randrange(n_in + i)
            rows[i] = Row(r.op, target, r.b, r.args) if self.rng.random() < 0.6 or r.b == NONE \
                else Row(r.op, r.a, target, r.args)
        elif kind == "crossover":
            other = self._tournament(pool).genome.rows
            ca = self.rng.randrange(len(rows) + 1)
            cb = self.rng.randrange(len(other) + 1)
            tail = []
            for r in other[cb:]:
                tail.append(Row(r.op, _remap(r.a, n_in, cb, ca), _remap(r.b, n_in, cb, ca), r.args))
            rows = rows[:ca] + tail
            if not rows:
                rows = list(parent.genome.rows)
        genome, _ = self.space.repair(ArchGenome(rows), self.policy)
        if not genome.rows and parent.genome.rows:
            genome = parent.genome.copy()
        return genome, kind

    def _step_arg(self, rows: List[Row], i: int) -> Row:
        row = rows[i]
        spec = self.space.registry.get(row.op)
        j = self.rng.randrange(len(spec.args))
        arg = spec.args[j]
        shapes = self.space.shapes(ArchGenome(rows[:i]))
        options = self.space.arg_options(arg, shapes[row.a])
        args = list(row.args)
        if arg.numeric and row.args[j] in options and len(options) > 1:
            k = options.index(row.args[j])
            up = self.rng.random() < 0.5 + 0.4 * self._pressure if arg.name in _SIZE_ARGS else self.rng.random() < 0.5
            step = 1 if self.rng.random() < 0.75 else 2
            k = min(len(options) - 1, k + step) if up else max(0, k - step)
            args[j] = options[k]
        else:
            ctx = self.space.slot_context("arg:" + arg.name, i, shapes, 0, row.op, shapes[row.a])
            args[j] = self.policy("arg:" + arg.name, ctx, options) if options else row.args[j]
        return Row(row.op, row.a, row.b, tuple(args))

    # ------------------------------------------------------------------ main loop
    def run(
        self,
        *,
        generations: int = 10,
        max_evaluations: Optional[int] = None,
        seconds: float = 0.0,
        patience: int = 3,
        min_improvement: float = 1e-3,
        seeds: Sequence[ArchGenome] = (),
        progress: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        if generations < 1 or patience < 1:
            raise ValueError("generations and patience must be positive")
        started = time.monotonic()
        budget_hit = lambda: (seconds > 0 and time.monotonic() - started >= seconds) or \
            (max_evaluations is not None and len(self.history) >= max_evaluations)  # noqa: E731
        population: List[Candidate] = []
        stop_reason = "generation_limit"
        best_score, stale = math.inf, 0
        per_generation = []
        for gen in range(generations):
            new: List[Candidate] = []
            if gen == 0:
                for genome in seeds:
                    if budget_hit():
                        break
                    fixed, _ = self.space.repair(genome, self.policy)
                    new.append(self._evaluate(fixed, "seed", gen, None))
                while len(new) < self.population_size and not budget_hit():
                    new.append(self._evaluate(self.space.random_genome(self.policy), "random", gen, None))
            else:
                elite = sorted(population, key=lambda c: (c.score, c.params, c.index))[: self.elite]
                while len(new) < self.population_size - len(elite) and not budget_hit():
                    parent = self._tournament(population)
                    genome, kind = self._vary(parent, population)
                    new.append(self._evaluate(genome, "mutate:" + kind, gen, parent.score))
                new = elite + new
            population = new or population
            ranked = sorted(population, key=lambda c: (c.score, c.params, c.index))
            if ranked:
                widths = [w for c in ranked[: self.elite] for w in self.space.width_values(c.genome)]
                self.space.adapt_window(widths)
                self._set_pressure(ranked[0].params)
            gen_best = ranked[0].score if ranked else math.inf
            per_generation.append({"generation": gen, "best_score": gen_best if math.isfinite(gen_best) else None,
                                   "best_loss": ranked[0].loss if ranked and math.isfinite(ranked[0].loss) else None,
                                   "best_params": ranked[0].params if ranked else None,
                                   "evaluations": len(self.history), "pressure": self._pressure,
                                   "window": [self.space.widths()[0], self.space.widths()[-1]]})
            if progress:
                progress(per_generation[-1])
            if gen_best < best_score - min_improvement * max(1.0, abs(best_score) if math.isfinite(best_score) else 1.0):
                best_score, stale = gen_best, 0
            else:
                stale += 1
            if budget_hit():
                stop_reason = "time_budget" if seconds > 0 and time.monotonic() - started >= seconds else "evaluation_budget"
                break
            if stale >= patience:
                stop_reason = "converged"
                break
        return self.report(stop_reason, per_generation, time.monotonic() - started)

    def best(self) -> Optional[Candidate]:
        finite = [c for c in self.cache.values() if math.isfinite(c.score)]
        return min(finite, key=lambda c: (c.score, c.params, c.index)) if finite else None

    def pareto(self) -> List[Candidate]:
        front, best_loss = [], math.inf
        for c in sorted((c for c in self.cache.values() if math.isfinite(c.loss)), key=lambda c: (c.params, c.loss)):
            if c.loss < best_loss:
                front.append(c)
                best_loss = c.loss
        return front

    def report(self, stop_reason: str, per_generation: List[dict], seconds: float) -> dict:
        best = self.best()
        eff = self.knowledge.efficiency
        return {
            "schema": "evolvo_arch_search_v1",
            "seed": self.seed,
            "stop_reason": stop_reason,
            "seconds": seconds,
            "evaluations": len(self.history),
            "unique_architectures": len(self.cache),
            "cache_hits": sum(1 for c in self.history if c.cached),
            "size_cost": self.size_cost,
            "best": None if best is None else {
                **best.brief(), "genome": best.genome.to_dict(), "describe": self.space.describe(best.genome),
                "slots": [list(s) for s in self.space.to_slots(best.genome)],
                "lowered": self.space.lower(best.genome).describe(), "training": _training_brief(best.result)},
            "pareto": [c.brief() for c in self.pareto()],
            "generations": per_generation,
            "efficiency": {**eff.report(self.space.task),
                           "budget_params": self.space.max_params},
            "width_window": [self.space.widths()[0], self.space.widths()[-1]],
            "window_events": list(self.space.window_events),
            "policy": {"stats": self.knowledge.policy.summary(),
                       "groups": self.knowledge.policy.groups(min_count=5, limit=12),
                       "mutation_preference": self.policy.preference(
                           "mutation", _mutation_context(self.space, self._pressure), list(MUTATIONS))},
            "epochs": {"stats": self.knowledge.epochs.summary(),
                       "groups": self.knowledge.epochs.groups(min_count=2, limit=12)},
            "history": [c.brief() for c in self.history],
        }


_SIZE_ARGS = {"units", "dim", "depth", "ratio", "ff"}


def _training_brief(result: Mapping[str, Any]) -> dict:
    return {k: result.get(k) for k in ("epochs", "best_epoch", "stop_reason", "patience", "max_epochs",
                                       "learned", "projected_asymptote", "seconds")}


def _mutation_context(space: ArchSpace, pressure: float) -> Dict[str, Any]:
    return {**space.task, "slot": "mutation", "pressure": int(round(2 * pressure))}


def _mutation_decision(space: ArchSpace, kind: str, pressure: float):
    from .language import Decision

    return Decision(_mutation_context(space, pressure), kind)


def _shift(row: Row, threshold: int, delta: int) -> Row:
    a = row.a + delta if row.a >= threshold else row.a
    b = row.b + delta if row.b != NONE and row.b >= threshold else row.b
    return Row(row.op, a, b, row.args)


def _remap(ref: int, n_in: int, cb: int, ca: int) -> int:
    """Map a reference of the donor's suffix (starting at row ``cb``) into the child (row ``ca``)."""
    if ref == NONE or ref < n_in:
        return ref
    return max(0, ref + (ca - cb)) if ref + (ca - cb) >= n_in else n_in - 1


__all__ = ["ArchitectureSearch", "Candidate", "Knowledge", "MUTATIONS"]
