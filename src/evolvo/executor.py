"""Execution engine for GFSL genomes."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np

from .custom_ops import custom_operations
from .enums import Category, ConfigProperty, DataType, Operation
from .genome import GFSLGenome
from .instruction import GFSLInstruction
from .values import ValueEnumerations


class GFSLExecutor:
    """Executes GFSL genomes."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset execution state."""
        self.variables = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.constants = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.config_state = {}
        self.execution_trace = []

    def _safe_float(self, val) -> float:
        """Safely convert value to float, handling complex numbers and errors."""
        try:
            if isinstance(val, complex):
                return abs(val)
            return float(val)
        except (TypeError, ValueError, OverflowError):
            return 0.0

    def execute(
        self, genome: GFSLGenome, inputs: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Execute genome and return outputs.

        Args:
            genome: The GFSL genome to execute
            inputs: Optional input values as {"d$0": 5.0, "b$0": True, ...}
        Returns:
            Dictionary of output values
        """
        self.reset()

        if inputs:
            for key, value in inputs.items():
                dtype_char = key[0]
                is_const = "#" in key
                idx_str = key.split("$" if "$" in key else "#")[1]
                idx = int(idx_str)

                dtype = {"b": DataType.BOOLEAN, "d": DataType.DECIMAL}.get(
                    dtype_char, DataType.DECIMAL
                )

                if is_const:
                    self.constants[dtype][idx] = value
                else:
                    self.variables[dtype][idx] = value

        effective_indices = genome.extract_effective_algorithm()

        for idx in effective_indices:
            if idx < len(genome.instructions):
                self._execute_instruction(genome.instructions[idx])

        outputs = {}
        for cat, dtype, idx in genome.outputs:
            if cat == Category.VARIABLE:
                key = f"{DataType(dtype).name[0].lower()}${idx}"
                outputs[key] = self.variables[dtype][idx]
            elif cat == Category.CONSTANT:
                key = f"{DataType(dtype).name[0].lower()}#{idx}"
                outputs[key] = self.constants[dtype][idx]

        return outputs

    def _execute_instruction(self, instr: GFSLInstruction):
        """Execute a single instruction."""
        op_code = instr.operation
        custom_op = custom_operations.get(op_code)
        try:
            op = Operation(op_code)
        except ValueError:
            op = None

        source1 = self._get_value(instr, 1)
        source2 = self._get_value(instr, 2)

        result = None

        if custom_op:
            s1 = self._safe_float(source1)
            s2 = self._safe_float(source2)
            args = [s1]
            if custom_op.arity >= 2:
                args.append(s2)
            context = {"executor": self, "instruction": instr}
            try:
                if custom_op.accepts_context:
                    result = custom_op.function(*args, context=context)
                else:
                    result = custom_op.function(*args)
                if custom_op.target_type == DataType.DECIMAL:
                    result = self._safe_float(result)
            except Exception:
                result = 0.0

        elif op is None:
            return
        else:
            if op == Operation.IF:
                pass
            elif op == Operation.WHILE:
                pass
            elif op == Operation.END:
                pass
            elif op == Operation.SET:
                if instr.source1_cat == Category.CONFIG:
                    self.config_state[instr.source1_value] = source2
            elif op == Operation.RESULT:
                pass

            elif op == Operation.GT:
                result = self._safe_float(source1) > self._safe_float(source2)
            elif op == Operation.LT:
                result = self._safe_float(source1) < self._safe_float(source2)
            elif op == Operation.EQ:
                result = (
                    abs(self._safe_float(source1) - self._safe_float(source2)) < 1e-9
                )
            elif op == Operation.AND:
                result = bool(source1) and bool(source2)
            elif op == Operation.OR:
                result = bool(source1) or bool(source2)
            elif op == Operation.NOT:
                result = not bool(source1)

            elif op == Operation.ADD:
                result = self._safe_float(source1) + self._safe_float(source2)
            elif op == Operation.SUB:
                result = self._safe_float(source1) - self._safe_float(source2)
            elif op == Operation.MUL:
                result = self._safe_float(source1) * self._safe_float(source2)
            elif op == Operation.DIV:
                s2 = self._safe_float(source2)
                result = self._safe_float(source1) / s2 if s2 != 0 else 0.0
            elif op == Operation.POW:
                try:
                    b = self._safe_float(source1)
                    e = self._safe_float(source2)
                    if b < 0 and e != int(e):
                        result = 0.0
                    else:
                        result = b**e
                except Exception:
                    result = 0.0
            elif op == Operation.SQRT:
                val = self._safe_float(source1)
                result = val**0.5 if val >= 0 else 0.0
            elif op == Operation.ABS:
                result = abs(self._safe_float(source1))
            elif op == Operation.SIN:
                result = np.sin(self._safe_float(source1))
            elif op == Operation.COS:
                result = np.cos(self._safe_float(source1))
            elif op == Operation.EXP:
                try:
                    result = np.exp(self._safe_float(source1))
                except Exception:
                    result = 0.0
            elif op == Operation.LOG:
                val = self._safe_float(source1)
                result = np.log(val) if val > 1e-9 else -100.0
            elif op == Operation.MOD:
                s2 = self._safe_float(source2)
                result = self._safe_float(source1) % s2 if s2 != 0 else 0.0

        if result is not None and instr.target_cat != Category.NONE:
            if instr.target_cat == Category.VARIABLE:
                self.variables[instr.target_type][instr.target_index] = result
            elif instr.target_cat == Category.CONSTANT:
                self.constants[instr.target_type][instr.target_index] = result

    def _get_value(self, instr: GFSLInstruction, source_num: int) -> Any:
        """Get value for a source."""
        if source_num == 1:
            cat = Category(instr.source1_cat)
            dtype = DataType(instr.source1_type)
            val = instr.source1_value
        else:
            cat = Category(instr.source2_cat)
            dtype = DataType(instr.source2_type)
            val = instr.source2_value

        if cat == Category.NONE:
            return None
        if cat == Category.VARIABLE:
            return self.variables[dtype][val]
        if cat == Category.CONSTANT:
            return self.constants[dtype][val]
        if cat == Category.VALUE:
            context = (instr.operation, None)
            if (
                instr.operation == Operation.SET
                and source_num == 2
                and instr.source1_cat == Category.CONFIG
            ):
                try:
                    prop = ConfigProperty(instr.source1_value)
                except ValueError:
                    prop = None
                context = (instr.operation, prop)
            enum = ValueEnumerations.get_enumeration(context)
            if val < len(enum):
                return enum[val]
            return 0.0
        if cat == Category.CONFIG:
            return val

        return 0.0


__all__ = ["GFSLExecutor"]
