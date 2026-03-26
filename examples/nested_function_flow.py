#!/usr/bin/env python3
"""
Smoke demo for nested GFSL functions.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from evolvo import (
    Category,
    DataType,
    GFSLExecutor,
    GFSLGenome,
    GFSLInstruction,
    Operation,
    pack_type_index,
)


def build_nested_genome() -> GFSLGenome:
    genome = GFSLGenome("algorithm")
    genome.validator.variable_counts[int(DataType.DECIMAL)] = 1  # expose d$0

    # d&0 FUNC (outer)
    assert genome.add_instruction(
        GFSLInstruction(
            [
                Category.FUNCTION,
                pack_type_index(DataType.DECIMAL, 0),
                Operation.FUNC,
                Category.NONE,
                0,
                Category.NONE,
                0,
            ]
        )
    )

    # d&1 FUNC (nested)
    assert genome.add_instruction(
        GFSLInstruction(
            [
                Category.FUNCTION,
                pack_type_index(DataType.DECIMAL, 1),
                Operation.FUNC,
                Category.NONE,
                0,
                Category.NONE,
                0,
            ]
        )
    )

    # d$1 = MUL(d$0, 2.0)
    assert genome.add_instruction(
        GFSLInstruction(
            [
                Category.VARIABLE,
                pack_type_index(DataType.DECIMAL, 1),
                Operation.MUL,
                Category.VARIABLE,
                pack_type_index(DataType.DECIMAL, 0),
                Category.VALUE,
                3,  # 2.0
            ]
        )
    )

    # END d$1  (close nested)
    assert genome.add_instruction(
        GFSLInstruction(
            [
                Category.NONE,
                0,
                Operation.END,
                Category.VARIABLE,
                pack_type_index(DataType.DECIMAL, 1),
                Category.NONE,
                0,
            ]
        )
    )

    # d$2 = CALL d&1
    assert genome.add_instruction(
        GFSLInstruction(
            [
                Category.VARIABLE,
                pack_type_index(DataType.DECIMAL, 2),
                Operation.CALL,
                Category.FUNCTION,
                pack_type_index(DataType.DECIMAL, 1),
                Category.NONE,
                0,
            ]
        )
    )

    # d$3 = ADD(d$2, 1.0)
    assert genome.add_instruction(
        GFSLInstruction(
            [
                Category.VARIABLE,
                pack_type_index(DataType.DECIMAL, 3),
                Operation.ADD,
                Category.VARIABLE,
                pack_type_index(DataType.DECIMAL, 2),
                Category.VALUE,
                1,  # 1.0
            ]
        )
    )

    # END d$3  (close outer)
    assert genome.add_instruction(
        GFSLInstruction(
            [
                Category.NONE,
                0,
                Operation.END,
                Category.VARIABLE,
                pack_type_index(DataType.DECIMAL, 3),
                Category.NONE,
                0,
            ]
        )
    )

    # d$4 = CALL d&0
    assert genome.add_instruction(
        GFSLInstruction(
            [
                Category.VARIABLE,
                pack_type_index(DataType.DECIMAL, 4),
                Operation.CALL,
                Category.FUNCTION,
                pack_type_index(DataType.DECIMAL, 0),
                Category.NONE,
                0,
            ]
        )
    )

    genome.mark_output(DataType.DECIMAL, 4)
    return genome


def main() -> None:
    genome = build_nested_genome()
    nested_enabled = GFSLExecutor(allow_nested_functions=True)
    nested_disabled = GFSLExecutor(allow_nested_functions=False)

    for x in [1.0, 3.0]:
        inputs = {"d$0": x}
        out_enabled = nested_enabled.execute(genome, inputs).get("d$4")
        out_disabled = nested_disabled.execute(genome, inputs).get("d$4")
        print(
            f"x={x:.1f} | nested=on -> {out_enabled} | nested=off -> {out_disabled}"
        )


if __name__ == "__main__":
    main()
