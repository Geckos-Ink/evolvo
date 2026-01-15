"""Neural model builder for GFSL genomes."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

_TORCH_IMPORT_ERROR: Optional[ImportError] = None
try:
    import torch.nn as nn
except ImportError as exc:
    nn = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc


def _require_torch(feature: str = "RecursiveModelBuilder") -> None:
    """Raise when torch is missing and a torch-based feature is hit."""
    if nn is None:
        raise ModuleNotFoundError(
            f"`torch` is required for {feature}. Install it via `pip install torch`."
        ) from _TORCH_IMPORT_ERROR

from .enums import Category, ConfigProperty, DataType, Operation
from .genome import GFSLGenome
from .instruction import GFSLInstruction
from .values import ValueEnumerations


class RecursiveModelBuilder:
    """
    Builds neural network models using GFSL instructions,
    supporting recursive architecture selection.
    """

    def __init__(self):
        self.layers = []
        self.config_state = {}
        self.current_shape = None

    def build_from_genome(self, genome: GFSLGenome, input_shape: Tuple[int, ...]) -> nn.Module:
        """Build PyTorch model from GFSL genome."""
        _require_torch()
        self.layers = []
        self.config_state = {}
        self.current_shape = input_shape

        for instr in genome.instructions:
            self._process_neural_instruction(instr)

        if not self.layers:
            return nn.Identity()

        return nn.Sequential(*self.layers)

    def _process_neural_instruction(self, instr: GFSLInstruction):
        """Process instruction for neural architecture building."""
        _require_torch()
        try:
            op = Operation(instr.operation)
        except ValueError:
            return

        if op == Operation.SET:
            if instr.source1_cat == Category.CONFIG:
                prop = ConfigProperty(instr.source1_value)
                value_idx = instr.source2_value

                context = (op, prop)
                enum = ValueEnumerations.get_enumeration(context)
                if value_idx < len(enum):
                    self.config_state[prop] = enum[value_idx]

        elif op == Operation.CONV:
            channels = int(self.config_state.get(ConfigProperty.CHANNELS, 32))
            kernel = int(self.config_state.get(ConfigProperty.KERNEL, 3))
            stride = int(self.config_state.get(ConfigProperty.STRIDE, 1))
            padding = int(self.config_state.get(ConfigProperty.PADDING, 1))

            in_channels = self.current_shape[0] if self.current_shape else 3

            layer = nn.Conv2d(in_channels, channels, kernel, stride, padding)
            self.layers.append(layer)

            if self.current_shape and len(self.current_shape) >= 3:
                h, w = self.current_shape[1:3]
                h_out = (h + 2 * padding - kernel) // stride + 1
                w_out = (w + 2 * padding - kernel) // stride + 1
                self.current_shape = (channels, h_out, w_out)
            else:
                self.current_shape = (channels, None, None)

            self.config_state = {}

        elif op == Operation.LINEAR:
            units = int(self.config_state.get(ConfigProperty.UNITS, 128))

            if self.current_shape and len(self.current_shape) > 1:
                self.layers.append(nn.Flatten())
                in_features = np.prod(self.current_shape)
            else:
                in_features = self.current_shape[0] if self.current_shape else 128

            layer = nn.Linear(in_features, units)
            self.layers.append(layer)

            self.current_shape = (units,)
            self.config_state = {}

        elif op == Operation.RELU:
            self.layers.append(nn.ReLU())

        elif op == Operation.DROPOUT:
            rate = self.config_state.get(ConfigProperty.RATE, 0.5)
            self.layers.append(nn.Dropout(rate))
            self.config_state = {}

        elif op == Operation.POOL:
            kernel = int(self.config_state.get(ConfigProperty.KERNEL, 2))
            stride = int(self.config_state.get(ConfigProperty.STRIDE, 2))

            layer = nn.MaxPool2d(kernel, stride)
            self.layers.append(layer)

            if self.current_shape and len(self.current_shape) >= 3:
                h, w = self.current_shape[1:3]
                h_out = (h - kernel) // stride + 1
                w_out = (w - kernel) // stride + 1
                self.current_shape = (self.current_shape[0], h_out, w_out)

            self.config_state = {}

        elif op == Operation.NORM:
            if self.current_shape:
                if len(self.current_shape) >= 3:
                    self.layers.append(nn.BatchNorm2d(self.current_shape[0]))
                else:
                    self.layers.append(nn.BatchNorm1d(self.current_shape[0]))

        elif op == Operation.SOFTMAX:
            self.layers.append(nn.Softmax(dim=-1))


__all__ = ["RecursiveModelBuilder"]
