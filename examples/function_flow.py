#!/usr/bin/env python3
"""
Demonstrate optional GFSL functions and activity-based pruning.
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


def main() -> None:
    genome = GFSLGenome("algorithm")
    genome.validator.variable_counts[int(DataType.DECIMAL)] = 1  # expose d$0 input

    # FUNC d&0
    genome.add_instruction(
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

    # d$1 = ADD(d$0, 2.0)
    genome.add_instruction(
        GFSLInstruction(
            [
                Category.VARIABLE,
                pack_type_index(DataType.DECIMAL, 1),
                Operation.ADD,
                Category.VARIABLE,
                pack_type_index(DataType.DECIMAL, 0),
                Category.VALUE,
                3,  # 2.0 in default math enumeration
            ]
        )
    )

    # END d$1
    genome.add_instruction(
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

    # d$2 = CALL d&0
    genome.add_instruction(
        GFSLInstruction(
            [
                Category.VARIABLE,
                pack_type_index(DataType.DECIMAL, 2),
                Operation.CALL,
                Category.FUNCTION,
                pack_type_index(DataType.DECIMAL, 0),
                Category.NONE,
                0,
            ]
        )
    )

    # Add an unreachable junk instruction to prune later.
    genome.add_instruction(
        GFSLInstruction(
            [
                Category.VARIABLE,
                pack_type_index(DataType.DECIMAL, 3),
                Operation.ADD,
                Category.VALUE,
                1,  # 1.0
                Category.VALUE,
                1,  # 1.0
            ]
        )
    )

    genome.mark_output(DataType.DECIMAL, 2)
    executor = GFSLExecutor()

    for x in [1.0, 3.5]:
        print(f"x={x:.1f} -> {executor.execute(genome, {'d$0': x})}")

    print("Instruction activity (hits):", [meta.hits for meta in genome.instruction_activity])
    removed = genome.prune_stale_instructions(min_hits=1, keep_effective=True)
    print("Pruned stale instruction indices:", removed)
    print("Remaining instruction count:", len(genome.instructions))


if __name__ == "__main__":
    main()
