"""Supervised guidance model for evolution."""

from __future__ import annotations

import copy
import random
from collections import defaultdict, deque
from typing import List, Optional, Tuple

import numpy as np

_TORCH_IMPORT_ERROR: Optional[ImportError] = None
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
except ImportError as exc:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from .custom_ops import custom_operations
from .enums import Category, DataType, Operation
from .genome import GFSLGenome


def _require_torch(feature: str) -> None:
    """Raise a user-friendly error when PyTorch is unavailable."""
    if torch is None or nn is None or F is None or optim is None:
        raise ModuleNotFoundError(
            f"`torch` is required for {feature}. Install it via `pip install torch`."
        ) from _TORCH_IMPORT_ERROR


_BaseDirectionModule = nn.Module if nn is not None else object


def _torch_has_rocm() -> bool:
    if torch is None:
        return False
    try:
        return bool(getattr(torch.version, "hip", None)) and bool(torch.cuda.is_available())
    except Exception:
        return False


def _torch_has_cuda() -> bool:
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available()) and not _torch_has_rocm()
    except Exception:
        return False


def _torch_has_mps() -> bool:
    if torch is None:
        return False
    try:
        return bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    except Exception:
        return False


def resolve_torch_accelerator(
    preferred_device: Optional[str] = "auto",
) -> Tuple[str, str, bool, bool]:
    """Resolve torch backend/device.

    Returns:
        backend: one of ``none|cpu|cuda|rocm|mps``
        device: torch device name (``cpu|cuda|mps``)
        accelerator_available: whether a GPU-style accelerator is selected
        torch_available: whether torch import/runtime is available
    """
    if torch is None:
        return "none", "cpu", False, False

    preferred = str(preferred_device or "auto").strip().lower()
    if preferred not in {"auto", "cpu", "cuda", "rocm", "mps"}:
        preferred = "auto"

    rocm_available = _torch_has_rocm()
    cuda_available = _torch_has_cuda() or rocm_available
    mps_available = _torch_has_mps()

    if preferred == "cpu":
        return "cpu", "cpu", False, True

    if preferred == "rocm":
        if rocm_available:
            return "rocm", "cuda", True, True
    elif preferred == "cuda":
        if cuda_available:
            if rocm_available:
                return "rocm", "cuda", True, True
            return "cuda", "cuda", True, True
    elif preferred == "mps":
        if mps_available:
            return "mps", "mps", True, True

    if rocm_available:
        return "rocm", "cuda", True, True
    if cuda_available:
        return "cuda", "cuda", True, True
    if mps_available:
        return "mps", "mps", True, True
    return "none", "cpu", False, True


class GFSLFeatureExtractor:
    """Encodes GFSL genomes into fixed feature vectors aligned with GFSL papers."""

    def __init__(self, max_instructions: int = 128, max_outputs: int = 16, max_depth: int = 16):
        _require_torch("GFSLFeatureExtractor")
        self.max_instructions = max_instructions
        self.max_outputs = max_outputs
        self.max_depth = max_depth
        self.operations = list(Operation)
        self.categories = list(Category)
        self.data_types = list(DataType)
        self.base_features = 8
        self.feature_dim = (
            self.base_features
            + len(self.operations)
            + len(self.categories)
            + len(self.data_types) * 2
        )

    def encode(self, genome: GFSLGenome) -> torch.Tensor:
        """Return feature vector capturing structure, control flow, and type usage."""
        features = np.zeros(self.feature_dim, dtype=np.float32)
        instructions = genome.instructions
        instr_count = len(instructions)
        effective = genome.extract_effective_algorithm() if instr_count else []
        effective_ratio = len(effective) / instr_count if instr_count else 0.0
        output_ratio = min(len(genome.outputs) / self.max_outputs, 1.0)

        operation_counts = defaultdict(int)
        target_category_counts = defaultdict(int)
        target_type_counts = defaultdict(int)
        source_type_counts = defaultdict(int)

        control_flow = 0
        set_ops = 0
        value_sources = 0
        depth = 0
        max_depth_seen = 0
        tensor_flag = False
        target_total = 0
        source_total = 0

        for instr in instructions:
            custom_op = custom_operations.get(instr.operation)
            try:
                op = Operation(instr.operation)
            except ValueError:
                op = None

            if op is not None:
                operation_counts[op] += 1

                if op in (Operation.IF, Operation.WHILE, Operation.END, Operation.FUNC):
                    control_flow += 1
                if op in (Operation.IF, Operation.WHILE, Operation.FUNC):
                    depth += 1
                    if depth > max_depth_seen:
                        max_depth_seen = depth
                elif op == Operation.END and depth > 0:
                    depth -= 1
                if op == Operation.SET:
                    set_ops += 1
            elif custom_op:
                if custom_op.target_type == DataType.TENSOR:
                    tensor_flag = True

            target_cat = Category(instr.target_cat)
            target_category_counts[target_cat] += 1
            if target_cat != Category.NONE:
                target_total += 1
                target_dtype = DataType(instr.target_type)
                target_type_counts[target_dtype] += 1
                if target_dtype == DataType.TENSOR or (
                    custom_op and custom_op.target_type == DataType.TENSOR
                ):
                    tensor_flag = True

            for source_cat_int, source_type_int in (
                (instr.source1_cat, instr.source1_type),
                (instr.source2_cat, instr.source2_type),
            ):
                source_cat = Category(source_cat_int)
                if source_cat == Category.NONE:
                    continue
                if source_cat == Category.VALUE:
                    value_sources += 1
                if source_cat in (
                    Category.VARIABLE,
                    Category.CONSTANT,
                    Category.VALUE,
                    Category.LIST,
                    Category.LIST_CONSTANT,
                ):
                    source_dtype = DataType(source_type_int)
                    source_type_counts[source_dtype] += 1
                    source_total += 1
                    if source_dtype == DataType.TENSOR:
                        tensor_flag = True

        instruction_density = min(instr_count / self.max_instructions, 1.0)
        max_depth_ratio = min(max_depth_seen / self.max_depth, 1.0)
        control_flow_ratio = control_flow / instr_count if instr_count else 0.0
        set_ratio = set_ops / instr_count if instr_count else 0.0
        value_ratio = value_sources / source_total if source_total else 0.0
        tensor_feature = 1.0 if tensor_flag else 0.0

        ptr = 0
        features[ptr] = instruction_density
        ptr += 1
        features[ptr] = effective_ratio
        ptr += 1
        features[ptr] = output_ratio
        ptr += 1
        features[ptr] = max_depth_ratio
        ptr += 1
        features[ptr] = control_flow_ratio
        ptr += 1
        features[ptr] = set_ratio
        ptr += 1
        features[ptr] = value_ratio
        ptr += 1
        features[ptr] = tensor_feature
        ptr += 1

        for op in self.operations:
            features[ptr] = operation_counts[op] / instr_count if instr_count else 0.0
            ptr += 1
        for cat in self.categories:
            features[ptr] = target_category_counts[cat] / instr_count if instr_count else 0.0
            ptr += 1
        target_den = target_total if target_total else 1
        for dtype in self.data_types:
            features[ptr] = target_type_counts[dtype] / target_den
            ptr += 1
        source_den = source_total if source_total else 1
        for dtype in self.data_types:
            features[ptr] = source_type_counts[dtype] / source_den
            ptr += 1

        return torch.tensor(features, dtype=torch.float32)


class GFSLSupervisedDirectionModel(_BaseDirectionModule):
    """Small feed-forward network that predicts genome fitness."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: Optional[List[int]] = None,
        *,
        dropout: float = 0.08,
    ):
        _require_torch("GFSLSupervisedDirectionModel")
        super().__init__()
        layers: List[nn.Module] = []
        raw_widths = [int(width) for width in (hidden_layers or [256, 160, 96]) if int(width) > 0]
        if len(raw_widths) < 3:
            fallback = [256, 160, 96]
            for width in fallback:
                if len(raw_widths) >= 3:
                    break
                if width not in raw_widths:
                    raw_widths.append(width)
        widths = raw_widths[:5]
        dropout_p = max(0.0, min(0.30, float(dropout)))
        prev = input_dim
        for idx, width in enumerate(widths):
            if width <= 0:
                continue
            layers.append(nn.Linear(prev, width))
            layers.append(nn.LayerNorm(width))
            layers.append(nn.GELU())
            if idx < len(widths) - 1 and dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


class GFSLSupervisedGuide:
    """Supervised PyTorch model that nudges evolution toward promising genomes."""

    def __init__(
        self,
        hidden_layers: Optional[List[int]] = None,
        buffer_size: int = 512,
        min_buffer: int = 64,
        batch_size: int = 64,
        epochs: int = 3,
        candidate_pool: int = 3,
        max_observations: int = 20,
        device: Optional[str] = None,
        capacity_auto_tune: bool = True,
    ):
        _require_torch("GFSLSupervisedGuide")
        self.feature_extractor = GFSLFeatureExtractor()
        base_layers = [int(width) for width in (hidden_layers or [256, 160, 96]) if int(width) > 0]
        if len(base_layers) < 3:
            fallback = [256, 160, 96]
            for width in fallback:
                if len(base_layers) >= 3:
                    break
                if width not in base_layers:
                    base_layers.append(width)
        self.base_hidden_layers = base_layers[:5]
        requested_device = str(device or "auto")
        (
            self.device_backend,
            resolved_device,
            self.accelerator_available,
            _torch_available,
        ) = resolve_torch_accelerator(requested_device)
        self.device_requested = requested_device
        self.device = torch.device(resolved_device)

        self.buffer = deque(maxlen=buffer_size)
        self.targets = deque(maxlen=buffer_size)
        self.min_buffer = min_buffer
        self.batch_size = batch_size
        self.epochs = epochs
        self.candidate_pool = max(1, candidate_pool)
        self.max_observations = max_observations

        self.target_mean: Optional[float] = None
        self.target_std: Optional[float] = None
        self.trained = False
        self.loss_history: List[float] = []
        self.capacity_auto_tune = bool(capacity_auto_tune)
        self.capacity_bias = 0.0
        self.regularization_bias = 0.0
        self.capacity_rebuild_cooldown = 0
        self.overfit_streak = 0
        self.underfit_streak = 0
        self.last_train_loss: Optional[float] = None
        self.last_val_loss: Optional[float] = None

        self.hidden_layers_current: List[int] = []
        self.dropout_current: float = 0.0
        self.model: GFSLSupervisedDirectionModel
        self.optimizer: optim.Optimizer
        initial_layers = self._scaled_hidden_layers(self.min_buffer)
        initial_dropout = self._effective_dropout(self.min_buffer)
        self._rebuild_model(initial_layers, dropout=initial_dropout)

    def _scaled_hidden_layers(self, dataset_size: int) -> List[int]:
        dataset = max(1, int(dataset_size))
        if not self.capacity_auto_tune:
            return list(self.base_hidden_layers)

        buffer_max = max(self.min_buffer + 1, int(self.buffer.maxlen))
        sample_ratio = max(
            0.0,
            min(
                1.0,
                (float(dataset) - float(self.min_buffer))
                / float(max(1, buffer_max - int(self.min_buffer))),
            ),
        )
        scale = (0.72 + (0.58 * sample_ratio)) * (1.0 + float(self.capacity_bias))
        scale = max(0.60, min(1.55, scale))

        cap_from_samples = max(96, min(768, int(round(float(dataset) * 2.2))))
        cap_from_features = max(
            96,
            min(768, int(round(float(self.feature_extractor.feature_dim) * 10.0))),
        )
        width_cap = min(cap_from_samples, cap_from_features)

        scaled: List[int] = []
        prev = 1_000_000
        for base in self.base_hidden_layers:
            width = int(round(float(base) * scale))
            width = max(64, min(width_cap, width))
            if width > prev:
                width = prev
            scaled.append(width)
            prev = width

        if len(scaled) < 3:
            fallback = [256, 160, 96]
            for width in fallback:
                if len(scaled) >= 3:
                    break
                scaled.append(int(min(width, width_cap)))
        return scaled[:5]

    def _effective_dropout(self, dataset_size: int) -> float:
        dataset = max(1, int(dataset_size))
        if not self.capacity_auto_tune:
            return 0.08
        buffer_max = max(self.min_buffer + 1, int(self.buffer.maxlen))
        sample_ratio = max(
            0.0,
            min(
                1.0,
                (float(dataset) - float(self.min_buffer))
                / float(max(1, buffer_max - int(self.min_buffer))),
            ),
        )
        dropout = 0.14 - (0.07 * sample_ratio) + float(self.regularization_bias)
        return max(0.03, min(0.20, float(dropout)))

    def _rebuild_model(self, hidden_layers: List[int], *, dropout: float) -> None:
        self.model = GFSLSupervisedDirectionModel(
            self.feature_extractor.feature_dim,
            hidden_layers,
            dropout=float(dropout),
        )
        self.model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=2e-4)
        self.hidden_layers_current = [int(width) for width in hidden_layers]
        self.dropout_current = float(dropout)

    def _update_capacity_feedback(self, *, train_loss: float, val_loss: Optional[float]) -> None:
        self.last_train_loss = float(train_loss)
        self.last_val_loss = float(val_loss) if val_loss is not None else None
        if not self.capacity_auto_tune or val_loss is None:
            return

        train = max(1e-6, float(train_loss))
        val = max(1e-6, float(val_loss))
        relative_gap = (val - train) / train
        balanced_high = train > 0.95 and val > 0.95 and abs(relative_gap) < 0.22
        overfit = relative_gap > 0.35 and train < 1.10

        if overfit:
            self.overfit_streak += 1
            self.underfit_streak = max(0, self.underfit_streak - 1)
        elif balanced_high:
            self.underfit_streak += 1
            self.overfit_streak = max(0, self.overfit_streak - 1)
        else:
            self.overfit_streak = max(0, self.overfit_streak - 1)
            self.underfit_streak = max(0, self.underfit_streak - 1)

        if self.overfit_streak >= 2:
            self.capacity_bias = max(-0.30, float(self.capacity_bias) - 0.08)
            self.regularization_bias = min(0.06, float(self.regularization_bias) + 0.015)
            self.capacity_rebuild_cooldown = 0
            self.overfit_streak = 0
        elif self.underfit_streak >= 2:
            self.capacity_bias = min(0.35, float(self.capacity_bias) + 0.08)
            self.regularization_bias = max(-0.03, float(self.regularization_bias) - 0.01)
            self.capacity_rebuild_cooldown = 0
            self.underfit_streak = 0

    def observe_population(self, population: List[GFSLGenome]):
        """Collect labelled data from the current population and train when ready."""
        if not population:
            return

        observed = 0
        for genome in population[: self.max_observations]:
            if genome.fitness is None or not np.isfinite(genome.fitness):
                continue
            self.buffer.append(self.feature_extractor.encode(genome))
            self.targets.append(float(genome.fitness))
            observed += 1

        if observed and len(self.buffer) >= self.min_buffer:
            self._train_model()

    def _train_model(self):
        """Train the PyTorch model on buffered genomes."""
        if len(self.buffer) < self.min_buffer:
            return

        features = torch.stack(list(self.buffer)).to(self.device)
        targets_tensor = torch.tensor(list(self.targets), dtype=torch.float32, device=self.device)
        dataset_size = int(features.size(0))

        target_layers = self._scaled_hidden_layers(dataset_size)
        target_dropout = self._effective_dropout(dataset_size)
        if self.capacity_rebuild_cooldown > 0:
            self.capacity_rebuild_cooldown -= 1
        needs_rebuild = (
            not self.hidden_layers_current
            or len(target_layers) != len(self.hidden_layers_current)
            or any(abs(int(a) - int(b)) >= 48 for a, b in zip(target_layers, self.hidden_layers_current))
            or abs(float(target_dropout) - float(self.dropout_current)) >= 0.03
        )
        if needs_rebuild and self.capacity_rebuild_cooldown <= 0:
            self._rebuild_model(target_layers, dropout=target_dropout)
            self.capacity_rebuild_cooldown = 2

        self.target_mean = float(targets_tensor.mean().item())
        target_std = float(targets_tensor.std().item())
        if target_std < 1e-6:
            target_std = 1.0
        self.target_std = target_std

        normalized_targets = (targets_tensor - self.target_mean) / target_std
        val_size = 0
        if dataset_size >= (int(self.min_buffer) + 8):
            val_size = max(8, int(round(0.15 * float(dataset_size))))
            val_size = min(max(0, dataset_size - 8), val_size)
        permutation_all = torch.randperm(dataset_size, device=self.device)
        if val_size > 0:
            val_idx = permutation_all[:val_size]
            train_idx = permutation_all[val_size:]
        else:
            val_idx = None
            train_idx = permutation_all

        train_x = features[train_idx]
        train_y = normalized_targets[train_idx]
        if train_x.size(0) == 0:
            train_x = features
            train_y = normalized_targets
            val_idx = None

        self.model.train()
        last_loss = torch.tensor(0.0, device=self.device)

        for _ in range(self.epochs):
            permutation = torch.randperm(train_x.size(0), device=self.device)
            for start in range(0, train_x.size(0), self.batch_size):
                end = start + self.batch_size
                batch_idx = permutation[start:end]
                batch_x = train_x[batch_idx]
                batch_y = train_y[batch_idx]

                predictions = self.model(batch_x)
                loss = F.mse_loss(predictions, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                last_loss = loss

            self.loss_history.append(float(last_loss.item()))

        self.model.eval()
        with torch.no_grad():
            train_predictions = self.model(train_x)
            train_loss = float(F.mse_loss(train_predictions, train_y).item())
            val_loss: Optional[float] = None
            if val_idx is not None and val_idx.numel() > 0:
                val_x = features[val_idx]
                val_y = normalized_targets[val_idx]
                val_predictions = self.model(val_x)
                val_loss = float(F.mse_loss(val_predictions, val_y).item())
        self._update_capacity_feedback(train_loss=train_loss, val_loss=val_loss)
        self.trained = True

    def _predict_from_features(self, feature_list: List[torch.Tensor]) -> np.ndarray:
        if not feature_list:
            return np.array([], dtype=np.float32)
        if not self.trained or self.target_mean is None or self.target_std is None:
            return np.full(len(feature_list), np.nan, dtype=np.float32)

        with torch.no_grad():
            stacked = torch.stack(feature_list).to(self.device)
            preds = self.model(stacked)
        preds = preds.cpu().numpy()
        preds = preds * (self.target_std + 1e-6) + self.target_mean
        return preds.astype(np.float32)

    def predict(self, genomes: List[GFSLGenome]) -> np.ndarray:
        """Predict fitness for genomes without mutating them."""
        features = [self.feature_extractor.encode(g) for g in genomes]
        return self._predict_from_features(features)

    def propose_mutation(self, evolver: "GFSLEvolver", genome: GFSLGenome) -> GFSLGenome:
        """Sample candidate mutations and return the model-preferred genome."""
        if not self.trained or len(self.buffer) < self.min_buffer:
            return evolver.mutate(genome)

        candidates: List[GFSLGenome] = []
        candidate_features: List[torch.Tensor] = []

        for _ in range(self.candidate_pool):
            candidate = evolver.mutate(genome)
            candidates.append(candidate)
            candidate_features.append(self.feature_extractor.encode(candidate))

        base_clone = copy.deepcopy(genome)
        candidates.append(base_clone)
        candidate_features.append(self.feature_extractor.encode(base_clone))

        predictions = self._predict_from_features(candidate_features)
        if predictions.size == 0 or not np.isfinite(predictions).any():
            return random.choice(candidates[:-1])

        best_idx = int(np.nanargmax(predictions))
        return candidates[best_idx]


__all__ = [
    "resolve_torch_accelerator",
    "GFSLFeatureExtractor",
    "GFSLSupervisedDirectionModel",
    "GFSLSupervisedGuide",
]
