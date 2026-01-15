#!/usr/bin/env python3
"""
Demonstrate neural SET/CONV flow with consequents.
"""

import sys
from pathlib import Path

_TORCH_IMPORT_ERROR = None
try:
    import torch
except ImportError as exc:
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc


def _require_torch(feature: str) -> None:
    if torch is None:
        raise ModuleNotFoundError(
            f"`torch` is required for {feature}. Install it via `pip install torch`."
        ) from _TORCH_IMPORT_ERROR

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from evolvo import (
    ConfigProperty,
    DataType,
    GFSLGenome,
    Operation,
    RecursiveModelBuilder,
)


def set_config(genome: GFSLGenome, prop: ConfigProperty, value: float) -> None:
    genome.expression().target_none().op(Operation.SET).source1_config(prop).source2_value(value).commit()


def add_conv_with_relu(genome: GFSLGenome, target_idx: int, source_idx: int) -> None:
    def add_relu(genome_ref, instr):
        return (
            genome_ref.expression()
            .target_var(DataType.TENSOR, instr.target_index)
            .op(Operation.RELU)
            .source1_var(DataType.TENSOR, instr.target_index)
            .source2_none()
        )

    (
        genome.expression()
        .target_var(DataType.TENSOR, target_idx)
        .op(Operation.CONV)
        .source1_var(DataType.TENSOR, source_idx)
        .source2_none()
        .then_if({"op": Operation.CONV}, add_relu)
        .commit()
    )


def main() -> None:
    _require_torch("examples/neural_flow.py")
    genome = GFSLGenome("neural")
    genome.validator.activate_type(DataType.TENSOR)
    genome.validator.variable_counts[int(DataType.TENSOR)] = 1  # Expose t$0 as input.

    set_config(genome, ConfigProperty.CHANNELS, 8)
    set_config(genome, ConfigProperty.KERNEL, 3)
    set_config(genome, ConfigProperty.STRIDE, 1)
    set_config(genome, ConfigProperty.PADDING, 1)

    add_conv_with_relu(genome, target_idx=1, source_idx=0)

    builder = RecursiveModelBuilder()
    model = builder.build_from_genome(genome, (3, 32, 32))

    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print("Output shape:", tuple(y.shape))
    print("Human-readable trace:")
    for line in genome.to_human_readable():
        print(" ", line)


if __name__ == "__main__":
    main()
