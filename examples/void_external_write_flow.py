#!/usr/bin/env python3
"""
Smoke demo for void functions with optional external writes.
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


def build_void_genome() -> GFSLGenome:
    genome = GFSLGenome("algorithm")
    genome.validator.variable_counts[int(DataType.DECIMAL)] = 1  # expose d$0

    # n&0 FUNC
    assert genome.add_instruction(
        GFSLInstruction(
            [
                Category.FUNCTION,
                pack_type_index(DataType.NONE, 0),
                Operation.FUNC,
                Category.NONE,
                0,
                Category.NONE,
                0,
            ]
        )
    )

    # d$1 = ADD(d$0, 1.0)
    assert genome.add_instruction(
        GFSLInstruction(
            [
                Category.VARIABLE,
                pack_type_index(DataType.DECIMAL, 1),
                Operation.ADD,
                Category.VARIABLE,
                pack_type_index(DataType.DECIMAL, 0),
                Category.VALUE,
                1,  # 1.0
            ]
        )
    )

    # END n#0
    assert genome.add_instruction(
        GFSLInstruction(
            [
                Category.NONE,
                0,
                Operation.END,
                Category.CONSTANT,
                pack_type_index(DataType.NONE, 0),
                Category.NONE,
                0,
            ]
        )
    )

    # NONE CALL n&0
    assert genome.add_instruction(
        GFSLInstruction(
            [
                Category.NONE,
                0,
                Operation.CALL,
                Category.FUNCTION,
                pack_type_index(DataType.NONE, 0),
                Category.NONE,
                0,
            ]
        )
    )

    genome.mark_output(DataType.DECIMAL, 1)
    return genome


def main() -> None:
    genome = build_void_genome()
    isolated = GFSLExecutor()
    external = GFSLExecutor(allow_function_external_writes=True)
    strict_void = GFSLExecutor(
        allow_function_external_writes=False,
        require_void_external_writes=True,
    )

    for x in [2.0, 4.5]:
        inputs = {"d$0": x}
        out_isolated = isolated.execute(genome, inputs).get("d$1")
        out_external = external.execute(genome, inputs).get("d$1")
        out_strict = strict_void.execute(genome, inputs).get("d$1")
        print(
            f"x={x:.1f} | isolated -> {out_isolated} | external-write -> {out_external} "
            f"| strict-void-policy -> {out_strict}"
        )


if __name__ == "__main__":
    main()
