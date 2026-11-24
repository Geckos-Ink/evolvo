"""Reinforcement-learning mutation guide for GFSL evolution (PyTorch, policy gradient).

:class:`GFSLPolicyGuide` plugs into :class:`~evolvo.evolver.GFSLEvolver` exactly like
:class:`~evolvo.supervised.GFSLSupervisedGuide` (``supervised_guide=...``): the evolver calls
``propose_mutation(evolver, genome)`` whenever it mutates a child and ``observe_population(pop)``
once per generation after evaluation.

Policy
    * Operator head: which edit to apply (``slot``, ``add``, ``swap``, ``remove``), conditioned
      on the parent's :class:`~evolvo.supervised.GFSLFeatureExtractor` features and the
      current progress.
    * Slot head: whenever the edit has to choose a slot value (new value for a mutated slot,
      repairs of the following slots, every slot of an added instruction), it scores each
      *valid* option. Options are hashed with the slot index and the previous slot's value into
      ``buckets`` embeddings; score = bias + <embedding, context>. Only options the cascading
      validator offers are ever sampled, so every child stays valid.
    * Sampling: softmax at temperature ``T`` mixed with a uniform floor ``uniform_floor`` so no
      valid option ever gets probability 0 (the search cannot freeze).

Reward (relative to the average best scores)
    ``r = (f_child - q) / (|q| + eps)``, clipped to ``[-reward_clip, reward_clip]``, where ``q``
    is the mean fitness of the top ``top_fraction`` of the population observed in the same
    generation as the child's evaluation. Children and ``q`` therefore always come from the same
    evaluation epoch (see ``GFSLEvolver.evolve(fitness_epoch=...)``). Non-finite child fitness
    gets ``-reward_clip``.

Learning
    REINFORCE with a running-mean baseline, an entropy bonus, AdamW; one update per generation
    once ``batch_min`` rewarded children are buffered.

Less dispersion over time
    ``progress`` in [0, 1] lowers the temperature from ``temperature[0]`` to ``temperature[1]``
    and the entropy bonus from ``entropy_coef`` to 0. Progress is ``(q - q0) / (q_target - q0)``
    when ``q_target`` is given (``q0`` = first observed ``q``), otherwise whatever the caller
    sets with :meth:`set_progress`.

Persistence
    :meth:`save` / :meth:`load` store the weights, optimizer-free config and baseline so later
    runs start from what earlier runs learned.
"""

from __future__ import annotations

import copy
import itertools
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .genome import GFSLGenome
from .instruction import GFSLInstruction
from .supervised import GFSLFeatureExtractor, _require_torch, resolve_torch_accelerator

try:  # pragma: no cover - exercised only without torch
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

OPERATORS: Tuple[str, ...] = ("slot", "add", "swap", "remove")
_TOKEN = itertools.count(1)
POLICY_GUIDE_FORMAT = "gfsl-policy-guide-1"


def _option_bucket(slot_idx: int, prev_value: int, option: int, buckets: int) -> int:
    key = ((int(slot_idx) * 7919 + int(prev_value) + 1) * 1_000_003 + int(option)) & 0x7FFFFFFF
    return key % buckets


if torch is not None:

    class _PolicyNet(nn.Module):
        def __init__(self, feature_dim: int, hidden: int, buckets: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(feature_dim + 1)
            self.encoder = nn.Sequential(nn.Linear(feature_dim + 1, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
            self.operator_head = nn.Linear(hidden, len(OPERATORS))
            self.option_embedding = nn.Embedding(buckets, hidden)
            self.option_bias = nn.Embedding(buckets, 1)
            nn.init.normal_(self.option_embedding.weight, std=0.02)
            nn.init.zeros_(self.option_bias.weight)
            self.scale = 1.0 / math.sqrt(hidden)

        def context(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.encoder(self.norm(x))

        def operator_logits(self, ctx: "torch.Tensor") -> "torch.Tensor":
            return self.operator_head(ctx)

        def option_logits(self, ctx: "torch.Tensor", buckets: "torch.Tensor") -> "torch.Tensor":
            emb = self.option_embedding(buckets)
            return self.option_bias(buckets).squeeze(-1) + (emb @ ctx) * self.scale


class GFSLPolicyGuide:
    """Policy-gradient guide that learns which genome edits pay off (see module docstring)."""

    def __init__(
        self,
        *,
        hidden: int = 64,
        buckets: int = 4096,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        entropy_coef: float = 0.01,
        temperature: Tuple[float, float] = (1.5, 0.5),
        uniform_floor: float = 0.05,
        top_fraction: float = 0.10,
        reward_clip: float = 5.0,
        batch_min: int = 16,
        baseline_momentum: float = 0.9,
        q_target: Optional[float] = None,
        max_slot_trials: int = 6,
        device: Optional[str] = "auto",
        seed: Optional[int] = None,
    ) -> None:
        _require_torch("GFSLPolicyGuide")
        self.config: Dict[str, Any] = {
            "hidden": int(hidden),
            "buckets": int(buckets),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "entropy_coef": float(entropy_coef),
            "temperature": [float(temperature[0]), float(temperature[1])],
            "uniform_floor": float(uniform_floor),
            "top_fraction": float(top_fraction),
            "reward_clip": float(reward_clip),
            "batch_min": int(batch_min),
            "baseline_momentum": float(baseline_momentum),
            "max_slot_trials": int(max_slot_trials),
        }
        if not 0.0 <= uniform_floor < 1.0:
            raise ValueError("uniform_floor must be in [0, 1)")
        self.q_target = q_target
        self.rng = random.Random(seed)
        if seed is not None:
            torch.manual_seed(int(seed))
        self.device_backend, device_name, _, _ = resolve_torch_accelerator(device or "auto")
        self.device = torch.device(device_name)
        self.feature_extractor = GFSLFeatureExtractor()
        self.net = _PolicyNet(self.feature_extractor.feature_dim, self.config["hidden"], self.config["buckets"]).to(self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])

        self.progress = 0.0
        self.q0: Optional[float] = None
        self.q: Optional[float] = None
        self.baseline: Optional[float] = None
        self._pending: Dict[int, Dict[str, Any]] = {}
        self._buffer: List[Tuple[Dict[str, Any], float]] = []
        self.updates = 0
        self.observations = 0
        self.history: List[Dict[str, float]] = []
        self.loaded_from: Optional[str] = None

    # ------------------------------------------------------------------ schedule

    @property
    def temperature(self) -> float:
        hi, lo = self.config["temperature"]
        return hi + (lo - hi) * self.progress

    @property
    def entropy_coef(self) -> float:
        return self.config["entropy_coef"] * (1.0 - self.progress)

    def set_progress(self, progress: float) -> None:
        """Set progress in [0, 1] directly (used when no ``q_target`` is configured)."""
        self.progress = float(min(1.0, max(0.0, progress)))

    # ------------------------------------------------------------------ sampling

    def _features(self, genome: GFSLGenome) -> np.ndarray:
        f = self.feature_extractor.encode(genome).numpy()
        return np.append(f, np.float32(self.progress)).astype(np.float32)

    def _mix(self, logits: "torch.Tensor") -> "torch.Tensor":
        probs = torch.softmax(logits / max(self.temperature, 1e-3), dim=-1)
        floor = self.config["uniform_floor"]
        return (1.0 - floor) * probs + floor / probs.shape[-1]

    def _sample(self, probs: "torch.Tensor") -> int:
        p = probs.detach().cpu().numpy().astype(np.float64)
        p = p / p.sum()
        return int(self.rng.choices(range(len(p)), weights=p, k=1)[0])

    def _choose_option(self, ctx: "torch.Tensor", trace: Dict[str, Any], slot_idx: int, prev: int, options: Sequence[int]) -> int:
        options = [int(o) for o in options]
        if len(options) == 1:
            return options[0]
        buckets = [_option_bucket(slot_idx, prev, o, self.config["buckets"]) for o in options]
        with torch.no_grad():
            logits = self.net.option_logits(ctx, torch.tensor(buckets, device=self.device))
            choice = self._sample(self._mix(logits))
        trace["slots"].append((buckets, choice))
        return options[choice]

    @staticmethod
    def _prev(instruction: GFSLInstruction, slot_idx: int) -> int:
        return int(instruction.slots[slot_idx - 1]) if slot_idx > 0 else -1

    def _fill_instruction(self, genome: GFSLGenome, ctx, trace, instruction: GFSLInstruction, start: int, *, keep_valid: bool) -> bool:
        """Choose slots ``start..`` with the policy; with ``keep_valid`` keep still-valid values."""
        for slot_idx in range(start, len(instruction.slots)):
            options = genome.validator.get_valid_options(instruction, slot_idx)
            if not options:
                return False
            if keep_valid and instruction.slots[slot_idx] in options:
                continue
            instruction.slots[slot_idx] = self._choose_option(ctx, trace, slot_idx, self._prev(instruction, slot_idx), options)
        return True

    def propose_mutation(self, evolver, genome: GFSLGenome) -> GFSLGenome:
        """Return a mutated copy of ``genome`` built with the learned policy."""
        from .evolver import _invalidate_genome_caches

        child = copy.deepcopy(genome)
        child.rebuild_validator_state()
        features = self._features(genome)
        trace: Dict[str, Any] = {"features": features, "slots": []}
        with torch.no_grad():
            ctx = self.net.context(torch.as_tensor(features, device=self.device))
            op_logits = self.net.operator_logits(ctx)
        n_instr = len(child.instructions)
        mask = [
            n_instr > 0,
            n_instr < int(getattr(evolver, "max_instruction_count", 128)),
            n_instr > 1,
            n_instr > 1,
        ]
        if not any(mask):
            mask[1] = True
        masked = op_logits.masked_fill(~torch.tensor(mask, device=self.device), float("-inf"))
        probs = self._mix_masked(masked, mask)
        op_idx = self._sample(probs)
        trace["op"] = op_idx
        trace["op_mask"] = mask
        changed = self._apply(evolver, child, OPERATORS[op_idx], ctx, trace)
        if not changed:
            child = evolver.mutate(genome)  # nothing applicable: plain mutation, no trace
            return child
        child.rebuild_validator_state()
        evolver._enforce_complexity_budget(child)
        _invalidate_genome_caches(child)
        child.fitness = None
        child.generation = genome.generation
        token = next(_TOKEN)
        child._policy_token = token  # type: ignore[attr-defined]
        self._pending[token] = trace
        return child

    def _mix_masked(self, logits: "torch.Tensor", mask: Sequence[bool]) -> "torch.Tensor":
        probs = torch.softmax(logits / max(self.temperature, 1e-3), dim=-1)
        m = torch.tensor(mask, dtype=probs.dtype, device=probs.device)
        floor = self.config["uniform_floor"]
        return (1.0 - floor) * probs + floor * m / m.sum()

    def _apply(self, evolver, child: GFSLGenome, op: str, ctx, trace) -> bool:
        if op == "slot":
            for _ in range(self.config["max_slot_trials"]):
                i = self.rng.randrange(len(child.instructions))
                candidate = child.instructions[i].copy()
                j = self.rng.randrange(len(candidate.slots))
                options = [o for o in child.validator.get_valid_options(candidate, j) if o != candidate.slots[j]]
                if not options:
                    continue
                mark = len(trace["slots"])
                candidate.slots[j] = self._choose_option(ctx, trace, j, self._prev(candidate, j), options)
                if self._fill_instruction(child, ctx, trace, candidate, j + 1, keep_valid=True):
                    child.instructions[i] = candidate
                    return True
                del trace["slots"][mark:]
            return False
        if op == "add":
            instruction = GFSLInstruction(slot_count=max(1, int(child.validator.slot_count)))
            mark = len(trace["slots"])
            if not self._fill_instruction(child, ctx, trace, instruction, 0, keep_valid=False):
                del trace["slots"][mark:]
                return False
            child.instructions.append(instruction)
            child.validator.update_state(instruction)
            child._ensure_activity_size()
            return True
        if op == "swap" and len(child.instructions) > 1:
            a, b = self.rng.sample(range(len(child.instructions)), 2)
            child.instructions[a], child.instructions[b] = child.instructions[b], child.instructions[a]
            child._ensure_activity_size()
            acts = child.instruction_activity
            acts[a], acts[b] = acts[b], acts[a]
            return True
        if op == "remove" and len(child.instructions) > 1:
            idx = self.rng.randrange(len(child.instructions))
            child.instructions.pop(idx)
            if idx < len(child.instruction_activity):
                child.instruction_activity.pop(idx)
            return True
        return False

    # ------------------------------------------------------------------ learning

    def reward(self, fitness: Optional[float], q: float) -> float:
        clip = self.config["reward_clip"]
        if fitness is None or not math.isfinite(float(fitness)):
            return -clip
        r = (float(fitness) - q) / (abs(q) + 1e-8)
        return float(min(clip, max(-clip, r)))

    def observe_population(self, population: Sequence[GFSLGenome]) -> None:
        """Reward the children evaluated this generation, update the schedule, maybe learn."""
        finite = sorted((float(g.fitness) for g in population if g.fitness is not None and math.isfinite(float(g.fitness))), reverse=True)
        if not finite:
            self._pending.clear()
            return
        k = max(1, int(math.ceil(self.config["top_fraction"] * len(finite))))
        q = float(np.mean(finite[:k]))
        self.q = q
        if self.q0 is None:
            self.q0 = q
        if self.q_target is not None and self.q_target != self.q0:
            self.set_progress((q - self.q0) / (self.q_target - self.q0))
        rewards = []
        for genome in population:
            token = getattr(genome, "_policy_token", None)
            trace = self._pending.pop(token, None) if token is not None else None
            if trace is None or genome.fitness is None:
                continue
            r = self.reward(genome.fitness, q)
            self._buffer.append((trace, r))
            rewards.append(r)
        self._pending.clear()  # children not evaluated in this epoch are not rewarded later
        self.observations += 1
        loss = None
        if len(self._buffer) >= self.config["batch_min"]:
            loss = self._update()
        self.history.append({
            "q": q,
            "progress": self.progress,
            "temperature": self.temperature,
            "rewarded": float(len(rewards)),
            "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
            "loss": float("nan") if loss is None else float(loss),
        })

    def _log_prob_and_entropy(self, trace: Dict[str, Any]) -> Tuple["torch.Tensor", "torch.Tensor"]:
        ctx = self.net.context(torch.as_tensor(trace["features"], device=self.device))
        mask = trace["op_mask"]
        logits = self.net.operator_logits(ctx).masked_fill(~torch.tensor(mask, device=self.device), float("-inf"))
        probs = self._mix_masked(logits, mask)
        logp = torch.log(probs[trace["op"]])
        ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
        for buckets, choice in trace["slots"]:
            p = self._mix(self.net.option_logits(ctx, torch.tensor(buckets, device=self.device)))
            logp = logp + torch.log(p[choice])
            ent = ent - (p * torch.log(p.clamp_min(1e-12))).sum()
        return logp, ent

    def _update(self) -> float:
        rewards = np.array([r for _, r in self._buffer], dtype=np.float64)
        mean_r = float(rewards.mean())
        if self.baseline is None:
            self.baseline = mean_r
        advantages = rewards - self.baseline
        m = self.config["baseline_momentum"]
        self.baseline = m * self.baseline + (1.0 - m) * mean_r
        scale = float(advantages.std()) or 1.0
        self.net.train()
        self.optimizer.zero_grad()
        total = torch.zeros((), device=self.device)
        for (trace, _), adv in zip(self._buffer, advantages / scale):
            logp, ent = self._log_prob_and_entropy(trace)
            total = total - float(adv) * logp - self.entropy_coef * ent
        loss = total / len(self._buffer)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        self.net.eval()
        self._buffer.clear()
        self.updates += 1
        return float(loss.detach())

    # ------------------------------------------------------------------ persistence

    def save(self, path: str) -> None:
        torch.save(
            {
                "format": POLICY_GUIDE_FORMAT,
                "config": self.config,
                "feature_dim": self.feature_extractor.feature_dim,
                "state_dict": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
                "baseline": self.baseline,
                "updates": self.updates,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, **overrides: Any) -> "GFSLPolicyGuide":
        """Build a guide from a saved file; ``overrides`` change schedule settings
        (``q_target``, ``temperature``, ``device``, ``seed``, ...) but not the network shape."""
        _require_torch("GFSLPolicyGuide")
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob.get("format") != POLICY_GUIDE_FORMAT:
            raise ValueError(f"{path} is not a {POLICY_GUIDE_FORMAT} file")
        cfg = dict(blob["config"])
        for key in ("hidden", "buckets"):
            if key in overrides and overrides[key] != cfg[key]:
                raise ValueError(f"cannot override network shape '{key}' when loading")
        kwargs = {k: v for k, v in cfg.items()}
        kwargs["temperature"] = tuple(kwargs["temperature"])
        kwargs.update(overrides)
        guide = cls(**kwargs)
        if blob["feature_dim"] != guide.feature_extractor.feature_dim:
            raise ValueError("saved guide was trained with a different feature extractor")
        guide.net.load_state_dict(blob["state_dict"])
        guide.baseline = blob.get("baseline")
        guide.updates = int(blob.get("updates", 0))
        guide.loaded_from = str(path)
        return guide

    def runtime_summary(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "backend": self.device_backend,
            "updates": self.updates,
            "observations": self.observations,
            "progress": self.progress,
            "temperature": self.temperature,
            "baseline": self.baseline,
            "loaded_from": self.loaded_from,
        }


__all__ = ["GFSLPolicyGuide", "OPERATORS"]
