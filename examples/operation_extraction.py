#!/usr/bin/env python3
"""
Demonstrate operation extraction for specific result references.
"""

from evolvo import (
    Category,
    DataType,
    GFSLGenome,
    Operation,
    resolve_operation_name,
)


def describe_instruction(genome: GFSLGenome, instr) -> str:
    """Return a human-readable description for a single instruction."""
    if instr.target_cat == Category.NONE:
        op = instr.operation
        if op == Operation.IF:
            return f"IF {genome._decode_source(instr, 1)}"
        if op == Operation.WHILE:
            return f"WHILE {genome._decode_source(instr, 1)}"
        if op == Operation.END:
            return "END"
        if op == Operation.RESULT:
            return f"RESULT {genome._decode_source(instr, 1)}"
        return resolve_operation_name(op)

    target = genome._decode_target(instr)
    op_name = resolve_operation_name(instr.operation)
    source1 = genome._decode_source(instr, 1)
    source2 = genome._decode_source(instr, 2)
    if instr.source2_cat == Category.NONE:
        return f"{target} = {op_name}({source1})"
    return f"{target} = {op_name}({source1}, {source2})"


def build_demo_genome() -> GFSLGenome:
    genome = GFSLGenome("algorithm")
    # Expose d$0 and d$1 as inputs.
    genome.validator.variable_counts[int(DataType.DECIMAL)] = 2

    # d#0 = 0.0 + 2.0
    (
        genome.expression()
        .target_const(DataType.DECIMAL, 0)
        .op(Operation.ADD)
        .source1_value(0.0)
        .source2_value(2.0)
        .commit()
    )

    # d$2 = d$0 + d#0
    (
        genome.expression()
        .target_var(DataType.DECIMAL, 2)
        .op(Operation.ADD)
        .source1_var(DataType.DECIMAL, 0)
        .source2_const(DataType.DECIMAL, 0)
        .commit()
    )

    # d$3 = d$2 * d$1
    (
        genome.expression()
        .target_var(DataType.DECIMAL, 3)
        .op(Operation.MUL)
        .source1_var(DataType.DECIMAL, 2)
        .source2_var(DataType.DECIMAL, 1)
        .commit()
    )

    # Junk instruction: d$4 = d$1 - 1.0
    (
        genome.expression()
        .target_var(DataType.DECIMAL, 4)
        .op(Operation.SUB)
        .source1_var(DataType.DECIMAL, 1)
        .source2_value(1.0)
        .commit()
    )

    return genome


def print_instructions(genome: GFSLGenome, indices, title: str) -> None:
    print(title)
    for idx in indices:
        instr = genome.instructions[idx]
        print(f"  [{idx}] {describe_instruction(genome, instr)}")


def main() -> None:
    genome = build_demo_genome()
    print_instructions(genome, range(len(genome.instructions)), "All instructions:")

    result_refs = ["d$3"]
    exec_indices = genome.extract_operation_indices(result_refs, order="execution")
    print_instructions(genome, exec_indices, "Extracted for d$3 (execution order):")

    fixed_indices = genome.extract_operation_indices(result_refs, order="fixed")
    print_instructions(genome, fixed_indices, "Extracted for d$3 (fixed slot order):")

    multi_indices = genome.extract_operation_indices(["d$3", "d$4"], order="execution")
    print_instructions(genome, multi_indices, "Extracted for d$3 + d$4 (execution order):")


if __name__ == "__main__":
    main()
