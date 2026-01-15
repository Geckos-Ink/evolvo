"""Supervised guidance model for evolution."""

from __future__ import annotations

import copy
import random
from collections import defaultdict, deque
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .custom_ops import custom_operations
from .enums import Category, DataType, Operation
from .genome import GFSLGenome


class GFSLFeatureExtractor:
    """Encodes GFSL genomes into fixed feature vectors aligned with GFSL papers."""

    def __init__(self, max_instructions: int = 128, max_outputs: int = 16, max_depth: int = 16):
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

                if op in (Operation.IF, Operation.WHILE, Operation.END):
                    control_flow += 1
                if op in (Operation.IF, Operation.WHILE):
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
                if source_cat in (Category.VARIABLE, Category.CONSTANT, Category.VALUE):
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


class GFSLSupervisedDirectionModel(nn.Module):
    """Small feed-forward network that predicts genome fitness."""

    def __init__(self, input_dim: int, hidden_layers: Optional[List[int]] = None):
        super().__init__()
        layers: List[nn.Module] = []
        widths = hidden_layers or [128, 64]
        prev = input_dim
        for width in widths:
            if width <= 0:
                continue
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
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
    ):
        self.feature_extractor = GFSLFeatureExtractor()
        self.model = GFSLSupervisedDirectionModel(
            self.feature_extractor.feature_dim,
            hidden_layers,
        )
        device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_name)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

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

        self.target_mean = float(targets_tensor.mean().item())
        target_std = float(targets_tensor.std().item())
        if target_std < 1e-6:
            target_std = 1.0
        self.target_std = target_std

        normalized_targets = (targets_tensor - self.target_mean) / target_std

        dataset_size = features.size(0)
        self.model.train()

        for _ in range(self.epochs):
            permutation = torch.randperm(dataset_size, device=self.device)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = permutation[start:end]
                batch_x = features[batch_idx]
                batch_y = normalized_targets[batch_idx]

                predictions = self.model(batch_x)
                loss = F.mse_loss(predictions, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

            self.loss_history.append(float(loss.item()))

        self.model.eval()
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
    "GFSLFeatureExtractor",
    "GFSLSupervisedDirectionModel",
    "GFSLSupervisedGuide",
]
