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

    VOID = object()

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset execution state."""
        self.variables = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.constants = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.lists = defaultdict(lambda: defaultdict(list))
        self.constant_lists = defaultdict(lambda: defaultdict(list))
        self.config_state = {}
        self.execution_trace = []

    def _is_void(self, value: Any) -> bool:
        return value is self.VOID

    def _coerce_list_input(self, value: Any) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, np.ndarray):
            converted = value.tolist()
            return converted if isinstance(converted, list) else [converted]
        if isinstance(value, (str, bytes, dict)):
            return [value]
        try:
            return list(value)
        except TypeError:
            return [value]

    def _safe_float(self, val) -> float:
        """Safely convert value to float, handling complex numbers and errors."""
        try:
            if isinstance(val, complex):
                return abs(val)
            return float(val)
        except (TypeError, ValueError, OverflowError):
            return 0.0

    def execute(
        self, genome: GFSLGenome, inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute genome and return outputs.

        Args:
            genome: The GFSL genome to execute
            inputs: Optional input values as {"d$0": 5.0, "b$0": True, "d!0": [1, 2], ...}
        Returns:
            Dictionary of output values
        """
        self.reset()

        if inputs:
            for key, value in inputs.items():
                dtype_char = key[0].lower()
                if "!#" in key:
                    cat = Category.LIST_CONSTANT
                    idx_str = key.split("!#", 1)[1]
                elif "!" in key:
                    cat = Category.LIST
                    idx_str = key.split("!", 1)[1]
                elif "$" in key:
                    cat = Category.VARIABLE
                    idx_str = key.split("$", 1)[1]
                elif "#" in key:
                    cat = Category.CONSTANT
                    idx_str = key.split("#", 1)[1]
                else:
                    continue
                idx = int(idx_str)

                dtype = {"b": DataType.BOOLEAN, "d": DataType.DECIMAL, "t": DataType.TENSOR}.get(
                    dtype_char, DataType.DECIMAL
                )

                if cat == Category.CONSTANT:
                    self.constants[dtype][idx] = value
                elif cat == Category.VARIABLE:
                    self.variables[dtype][idx] = value
                elif cat == Category.LIST:
                    self.lists[dtype][idx] = self._coerce_list_input(value)
                elif cat == Category.LIST_CONSTANT:
                    self.constant_lists[dtype][idx] = self._coerce_list_input(value)

        effective_indices = genome.extract_effective_algorithm()

        for idx in effective_indices:
            if idx < len(genome.instructions):
                self._execute_instruction(genome.instructions[idx])

        outputs = {}
        for cat, dtype, idx in genome.outputs:
            if cat == Category.VARIABLE:
                key = f"{DataType(dtype).name[0].lower()}${idx}"
                value = self.variables[dtype][idx]
                outputs[key] = None if self._is_void(value) else value
            elif cat == Category.CONSTANT:
                key = f"{DataType(dtype).name[0].lower()}#{idx}"
                value = self.constants[dtype][idx]
                outputs[key] = None if self._is_void(value) else value

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

        if instr.source1_cat != Category.NONE and self._is_void(source1):
            return
        if instr.source2_cat != Category.NONE and self._is_void(source2):
            return

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
            elif op == Operation.PREPEND:
                target_list = self.lists[instr.target_type][instr.target_index]
                target_list.insert(0, source1)
                result = list(target_list)
            elif op == Operation.APPEND:
                target_list = self.lists[instr.target_type][instr.target_index]
                target_list.append(source1)
                result = list(target_list)
            elif op == Operation.CLONE:
                if isinstance(source1, list):
                    result = list(source1)
                else:
                    result = []
            elif op == Operation.FIFO:
                if isinstance(source1, list) and source1:
                    result = source1.pop(0)
                else:
                    result = self.VOID
            elif op == Operation.FILO:
                if isinstance(source1, list) and source1:
                    result = source1.pop()
                else:
                    result = self.VOID
            elif op == Operation.LISTCOUNT:
                result = float(len(source1)) if isinstance(source1, list) else 0.0
            elif op == Operation.LISTHASITEMS:
                result = bool(source1) if isinstance(source1, list) else False

        if result is not None and instr.target_cat != Category.NONE:
            if instr.target_cat == Category.VARIABLE:
                self.variables[instr.target_type][instr.target_index] = result
            elif instr.target_cat == Category.CONSTANT:
                self.constants[instr.target_type][instr.target_index] = result
            elif instr.target_cat == Category.LIST:
                self.lists[instr.target_type][instr.target_index] = (
                    list(result) if isinstance(result, list) else [result]
                )
            elif instr.target_cat == Category.LIST_CONSTANT:
                self.constant_lists[instr.target_type][instr.target_index] = (
                    list(result) if isinstance(result, list) else [result]
                )

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
        if cat == Category.LIST:
            return self.lists[dtype][val]
        if cat == Category.LIST_CONSTANT:
            return self.constant_lists[dtype][val]
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
