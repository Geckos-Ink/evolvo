"""Training a genome's module with automatic epoch stopping (requires PyTorch).

:func:`train_module` runs AdamW over (mini)batches, evaluates the validation loss every epoch,
lets a :class:`~evolvo.arch.stopping.PlateauStopper` decide when to stop, and restores the
weights of the best validation epoch. :class:`TorchTask` adapts it to
:class:`~evolvo.arch.search.ArchitectureSearch`: build, count, train, report, and keep the
module of the best candidate so far.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Callable, Dict, Hashable, Mapping, Optional, Sequence, Tuple

from .build import _require_torch, build_module, parameter_count
from .language import ArchGenome, ArchSpace
from .stats import ContextualStats, log_bucket
from .stopping import PlateauStopper

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _as_tuple(x) -> Tuple:
    return tuple(x) if isinstance(x, (tuple, list)) else (x,)


def train_module(
    module,
    train: Tuple[Any, Any],
    val: Tuple[Any, Any],
    *,
    stopper: PlateauStopper,
    lr: float = 3e-3,
    weight_decay: float = 0.0,
    batch_size: Optional[int] = None,
    seed: int = 0,
    loss_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Train until ``stopper`` says stop; restore the best-validation weights.

    ``train``/``val`` are ``(inputs, targets)`` where inputs is a tensor or a tuple of tensors
    (one per architecture input).
    """
    _require_torch()
    loss_fn = loss_fn or torch.nn.functional.mse_loss
    x_train, y_train = _as_tuple(train[0]), train[1]
    x_val, y_val = _as_tuple(val[0]), val[1]
    n = int(y_train.shape[0])
    gen = torch.Generator().manual_seed(int(seed))
    optim = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = copy.deepcopy(module.state_dict())
    best = math.inf
    train_curve = []
    started = time.perf_counter()
    epoch = 0
    while True:
        module.train()
        order = torch.randperm(n, generator=gen) if batch_size and batch_size < n else None
        step = batch_size if order is not None else n
        total = 0.0
        for start in range(0, n, step):
            idx = order[start:start + step] if order is not None else None
            xb = tuple(x[idx] for x in x_train) if idx is not None else x_train
            yb = y_train[idx] if idx is not None else y_train
            optim.zero_grad()
            loss = loss_fn(module(*xb), yb)
            loss.backward()
            optim.step()
            total += float(loss.detach()) * int(yb.shape[0])
        train_curve.append(total / n)
        module.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(module(*x_val), y_val))
        if math.isfinite(val_loss) and val_loss < best:
            best = val_loss
            best_state = copy.deepcopy(module.state_dict())
        if stopper.update(epoch, val_loss):
            break
        epoch += 1
    module.load_state_dict(best_state)
    module.eval()
    return {"val_loss": best, "train_curve": train_curve, "seconds": time.perf_counter() - started,
            **stopper.summary()}


class TorchTask:
    """Evaluate genomes as regressors: build, train with learned stopping, score on validation.

    ``epoch_stats`` holds the learned stopping tendencies, keyed by the task signature and the
    candidate's size bucket, so the patience and epoch cap of each new candidate come from how
    similar candidates trained, not from a constant.
    """

    def __init__(
        self,
        train: Tuple[Any, Any],
        val: Tuple[Any, Any],
        *,
        lr: float = 3e-3,
        weight_decay: float = 0.0,
        batch_size: Optional[int] = None,
        hard_max_epochs: int = 1000,
        min_epochs: int = 5,
        rel_tol: float = 1e-3,
        hopeless_margin: float = 0.5,
        epoch_stats: Optional[ContextualStats] = None,
        loss_fn: Optional[Callable] = None,
    ) -> None:
        _require_torch()
        self.train = train
        self.val = val
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.hard_max_epochs = hard_max_epochs
        self.min_epochs = min_epochs
        self.rel_tol = rel_tol
        self.hopeless_margin = hopeless_margin
        self.epoch_stats = epoch_stats or ContextualStats(pinned=("metric",))
        self.loss_fn = loss_fn
        self.best_loss = math.inf
        self.best_module = None
        self.best_signature: Optional[str] = None
        self.best_curve: Optional[list] = None

    def __call__(self, space: ArchSpace, genome: ArchGenome, info: Mapping[str, Any]) -> Dict[str, Any]:
        module = build_module(space, genome, seed=info.get("seed", 0))
        params = parameter_count(module)
        context: Dict[str, Hashable] = {**space.task, "size": log_bucket(params, 1),
                                        "rows": len(space.effective_rows(genome))}
        stopper = PlateauStopper(
            stats=self.epoch_stats, context=context,
            incumbent=self.best_loss if math.isfinite(self.best_loss) else None,
            reference_curve=self.best_curve,
            hard_max_epochs=self.hard_max_epochs, min_epochs=self.min_epochs, rel_tol=self.rel_tol,
            hopeless_margin=self.hopeless_margin)
        result = train_module(module, self.train, self.val, stopper=stopper, lr=self.lr,
                              weight_decay=self.weight_decay, batch_size=self.batch_size,
                              seed=info.get("seed", 0), loss_fn=self.loss_fn)
        if result["stop_reason"] != "hopeless":
            # A cut-short run did not show how long training pays; do not teach from it.
            stopper.record()
        result["params"] = params
        result["loss"] = result.pop("val_loss")
        if math.isfinite(result["loss"]) and result["loss"] < self.best_loss:
            self.best_loss = result["loss"]
            self.best_module = module
            self.best_signature = info.get("signature")
            self.best_curve = list(stopper.curve)
        return result


__all__ = ["TorchTask", "train_module"]
