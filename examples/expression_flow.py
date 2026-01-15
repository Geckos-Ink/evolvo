#!/usr/bin/env python3
"""
Demonstrate slot-wise expression building, option inspection, and consequents.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from evolvo import (
    DataType,
    GFSLExecutor,
    GFSLGenome,
    Operation,
    register_custom_operation,
)


def register_affine_bias() -> int:
    """Register a custom operation that scales and biases the input."""

    def affine_bias(value: float, bias: float, context=None) -> float:
        return float(value) * 1.25 + float(bias)

    try:
        return register_custom_operation(
            "affine_bias",
            target_type=DataType.DECIMAL,
            function=affine_bias,
            arity=2,
            value_enumeration=[-2.0, -1.0, 0.0, 1.0, 2.0],
            doc="Scales the first source then adds a selectable bias.",
        )
    except ValueError:
        from evolvo import custom_operations

        existing = custom_operations.get_code_by_name("affine_bias")
        if existing is None:
            raise
        return existing


def preview_next_options(builder, limit: int = 6) -> None:
    options = builder.next_options()
    preview = [opt.as_dict() for opt in options[:limit]]
    print(f"Next slot: {builder.next_slot_name()} -> {preview}")


def main() -> None:
    opcode = register_affine_bias()

    genome = GFSLGenome("algorithm")
    # Make d$0 available as an input slot without assigning it first.
    genome.validator.variable_counts[int(DataType.DECIMAL)] = 1

    expr = genome.expression()

    preview_next_options(expr)
    expr.target_var(DataType.DECIMAL, 1)

    preview_next_options(expr)
    expr.op(opcode)

    preview_next_options(expr)
    expr.source1_var(DataType.DECIMAL, 0)

    preview_next_options(expr)
    expr.source2_value(2.0)

    def square_output(genome_ref, instr):
        return (
            genome_ref.expression()
            .target_var(DataType.DECIMAL, 2)
            .op(Operation.MUL)
            .source1_var(DataType.DECIMAL, instr.target_index)
            .source2_var(DataType.DECIMAL, instr.target_index)
        )

    expr.then_if({"op": opcode}, square_output).commit()
    genome.mark_output(DataType.DECIMAL, 2)

    executor = GFSLExecutor()
    for x in [-1.0, 0.5, 3.0]:
        outputs = executor.execute(genome, {"d$0": x})
        print(f"x={x:+.1f} -> y={outputs.get('d$2')}")

    print("Human-readable trace:")
    for line in genome.to_human_readable():
        print(" ", line)


if __name__ == "__main__":
    main()
