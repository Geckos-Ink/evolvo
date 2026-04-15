"""Execution engine for GFSL genomes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .custom_ops import custom_operations
from .enums import Category, ConfigProperty, DataType, Operation
from .genome import GFSLGenome
from .instruction import GFSLInstruction
from .values import ValueEnumerations


@dataclass
class FunctionDefinition:
    """Resolved function metadata used during runtime execution."""

    return_dtype: DataType
    start_idx: int
    end_idx: int
    return_source: Optional[Tuple[int, int, int]] = None


class GFSLExecutor:
    """Executes GFSL genomes."""

    VOID = object()

    def __init__(
        self,
        *,
        allow_function_external_writes: bool = False,
        allow_nested_functions: bool = True,
        max_call_depth: int = 32,
        require_void_external_writes: bool = False,
        track_instruction_activity: bool = True,
        compute_backend: str = "auto",
        kompute_warn_on_fallback: bool = True,
        kompute_fail_hard: bool = False,
        kompute_keep_vram_state: bool = True,
    ):
        self.allow_function_external_writes = bool(allow_function_external_writes)
        self.allow_nested_functions = bool(allow_nested_functions)
        self.max_call_depth = max(1, int(max_call_depth))
        self.require_void_external_writes = bool(require_void_external_writes)
        self.track_instruction_activity = bool(track_instruction_activity)
        backend = str(compute_backend).strip().lower()
        if backend not in {"auto", "cpu", "kompute"}:
            backend = "auto"
        self.compute_backend = backend
        self.kompute_warn_on_fallback = bool(kompute_warn_on_fallback)
        self.kompute_fail_hard = bool(kompute_fail_hard)
        self.kompute_keep_vram_state = bool(kompute_keep_vram_state)
        self._kompute_runtime: Any = None
        self._kompute_runtime_checked = False
        self._kompute_warned_keys: Set[str] = set()
        self.reset()

    @staticmethod
    def _new_scalar_store():
        return defaultdict(lambda: defaultdict(lambda: 0.0))

    @staticmethod
    def _new_list_store():
        return defaultdict(lambda: defaultdict(list))

    def _new_call_frame(self) -> Dict[str, Any]:
        return {
            "variables": self._new_scalar_store(),
            "constants": self._new_scalar_store(),
            "lists": self._new_list_store(),
            "constant_lists": self._new_list_store(),
        }

    def reset(self):
        """Reset execution state."""
        self.variables = self._new_scalar_store()
        self.constants = self._new_scalar_store()
        self.lists = self._new_list_store()
        self.constant_lists = self._new_list_store()
        self.config_state: Dict[int, Any] = {}
        self.execution_trace: List[str] = []

        self._call_frames: List[Dict[str, Any]] = []
        self._call_depth = 0
        self._function_defs: Dict[Tuple[int, int], FunctionDefinition] = {}
        self._function_ranges: List[Tuple[int, int]] = []
        self._function_start_to_end: Dict[int, int] = {}
        self._function_range_indices: Set[int] = set()
        self._executed_instruction_indices: Set[int] = set()

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
        self,
        genome: GFSLGenome,
        inputs: Optional[Dict[str, Any]] = None,
        *,
        track_activity: Optional[bool] = None,
        activity_tick: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute genome and return outputs.

        Args:
            genome: The GFSL genome to execute.
            inputs: Optional input values as {"d$0": 5.0, "b$0": True, "d!0": [1, 2], ...}.
            track_activity: Override activity tracking for this run.
            activity_tick: Optional explicit activity tick to record on the genome.
        """
        kompute_outputs = self._attempt_kompute_execution(
            genome,
            inputs=inputs,
            track_activity=track_activity,
            activity_tick=activity_tick,
        )
        if kompute_outputs is not None:
            return kompute_outputs

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

                dtype = {
                    "b": DataType.BOOLEAN,
                    "d": DataType.DECIMAL,
                    "t": DataType.TENSOR,
                    "n": DataType.NONE,
                }.get(dtype_char, DataType.DECIMAL)

                if cat == Category.CONSTANT:
                    self.constants[dtype][idx] = value
                elif cat == Category.VARIABLE:
                    self.variables[dtype][idx] = value
                elif cat == Category.LIST:
                    self.lists[dtype][idx] = self._coerce_list_input(value)
                elif cat == Category.LIST_CONSTANT:
                    self.constant_lists[dtype][idx] = self._coerce_list_input(value)

        self._index_functions(genome)
        effective_indices = genome.extract_effective_algorithm()
        for idx in self._main_execution_indices(effective_indices):
            if idx < len(genome.instructions):
                self._execute_instruction(
                    genome.instructions[idx],
                    genome=genome,
                    instruction_index=idx,
                )

        should_track = self.track_instruction_activity if track_activity is None else bool(track_activity)
        if should_track and self._executed_instruction_indices:
            genome.record_instruction_activity(
                sorted(self._executed_instruction_indices),
                tick=activity_tick,
            )

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

    def _warn_kompute_fallback_once(self, key: str, message: str) -> None:
        if not self.kompute_warn_on_fallback:
            return
        key_norm = str(key).strip().lower() or "kompute"
        if key_norm in self._kompute_warned_keys:
            return
        self._kompute_warned_keys.add(key_norm)
        warnings.warn(
            message,
            RuntimeWarning,
            stacklevel=3,
        )

    def _get_kompute_runtime(self) -> Any:
        if self._kompute_runtime_checked:
            return self._kompute_runtime
        self._kompute_runtime_checked = True
        try:
            from .kompute import GFSLKomputeRuntime
        except Exception:
            self._kompute_runtime = None
            return None
        self._kompute_runtime = GFSLKomputeRuntime()
        return self._kompute_runtime

    def _attempt_kompute_execution(
        self,
        genome: GFSLGenome,
        *,
        inputs: Optional[Dict[str, Any]],
        track_activity: Optional[bool],
        activity_tick: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        backend = str(self.compute_backend).strip().lower()
        if backend == "cpu":
            return None

        runtime = self._get_kompute_runtime()
        if runtime is None:
            if backend == "kompute":
                message = (
                    "GFSLExecutor requested `kompute` backend but runtime is unavailable; "
                    "falling back to CPU execution."
                )
                if self.kompute_fail_hard:
                    raise ModuleNotFoundError(message)
                self._warn_kompute_fallback_once("import", message)
            return None

        if not runtime.is_available():
            if backend == "kompute":
                message = (
                    "GFSLExecutor requested `kompute` backend but `kp` is not installed "
                    "or not usable; falling back to CPU execution."
                )
                if self.kompute_fail_hard:
                    raise ModuleNotFoundError(message)
                self._warn_kompute_fallback_once("kp-unavailable", message)
            return None

        try:
            return runtime.execute(
                genome,
                inputs=inputs,
                track_activity=track_activity,
                activity_tick=activity_tick,
                keep_vram_state=bool(self.kompute_keep_vram_state),
            )
        except Exception as exc:
            message = (
                "Kompute execution failed ({error}); falling back to CPU execution."
            ).format(error=str(exc))
            if self.kompute_fail_hard:
                raise RuntimeError(message) from exc
            self._warn_kompute_fallback_once("runtime-error", message)
            return None

    def _index_functions(self, genome: GFSLGenome) -> None:
        """Build function definitions and executable function ranges."""
        self._function_defs = {}
        self._function_ranges = []
        self._function_start_to_end = {}
        self._function_range_indices = set()

        stack: List[Tuple[str, int, Optional[Tuple[int, int]]]] = []

        for idx, instr in enumerate(genome.instructions):
            op_code = instr.operation
            if op_code == Operation.FUNC and instr.target_cat == Category.FUNCTION:
                if not self.allow_nested_functions and any(kind == "FUNC" for kind, _, _ in stack):
                    stack.append(("SKIP_FUNC", idx, None))
                    continue
                key = (int(instr.target_type), int(instr.target_index))
                stack.append(("FUNC", idx, key))
                continue

            if op_code in (Operation.IF, Operation.WHILE):
                stack.append(("BLOCK", idx, None))
                continue

            if op_code == Operation.END and stack:
                kind, start_idx, key = stack.pop()
                if kind == "SKIP_FUNC":
                    self._function_ranges.append((start_idx, idx))
                    self._function_start_to_end[start_idx] = idx
                    self._function_range_indices.update(range(start_idx, idx + 1))
                    continue
                if kind != "FUNC" or key is None:
                    continue
                return_source = None
                if instr.source1_cat != Category.NONE:
                    return_source = (
                        int(instr.source1_cat),
                        int(instr.source1_type),
                        int(instr.source1_value),
                    )
                try:
                    return_dtype = DataType(key[0])
                except ValueError:
                    return_dtype = DataType.NONE
                self._function_defs[key] = FunctionDefinition(
                    return_dtype=return_dtype,
                    start_idx=start_idx,
                    end_idx=idx,
                    return_source=return_source,
                )
                self._function_ranges.append((start_idx, idx))
                self._function_start_to_end[start_idx] = idx
                self._function_range_indices.update(range(start_idx, idx + 1))

    def _main_execution_indices(self, effective_indices: List[int]) -> List[int]:
        """Return top-level indices, excluding function declaration/body ranges."""
        return [
            int(idx)
            for idx in effective_indices
            if int(idx) not in self._function_range_indices
        ]

    def _select_write_store(self, scope_name: str):
        if not self._call_frames:
            return getattr(self, scope_name)
        if not self.allow_function_external_writes:
            return self._call_frames[-1][scope_name]
        if len(self._call_frames) >= 2:
            return self._call_frames[-2][scope_name]
        return getattr(self, scope_name)

    def _read_scoped_value(self, scope_name: str, dtype: DataType, idx: int) -> Any:
        for frame in reversed(self._call_frames):
            typed = frame[scope_name].get(dtype)
            if typed is not None and idx in typed:
                return typed[idx]
        return getattr(self, scope_name)[dtype][idx]

    def _write_scoped_value(self, scope_name: str, dtype: DataType, idx: int, value: Any) -> None:
        store = self._select_write_store(scope_name)
        store[dtype][idx] = value

    def _list_for_write(self, scope_name: str, dtype: DataType, idx: int) -> List[Any]:
        store = self._select_write_store(scope_name)
        typed = store[dtype]
        if idx in typed:
            return typed[idx]

        if self._call_frames and store is self._call_frames[-1][scope_name]:
            inherited = self._read_scoped_value(scope_name, dtype, idx)
            typed[idx] = list(inherited) if isinstance(inherited, list) else []
            return typed[idx]
        return typed[idx]

    def _write_list_value(self, scope_name: str, dtype: DataType, idx: int, value: Any) -> None:
        store = self._select_write_store(scope_name)
        store[dtype][idx] = list(value) if isinstance(value, list) else [value]

    def _assign_target(self, instr: GFSLInstruction, result: Any) -> None:
        if instr.target_cat == Category.VARIABLE:
            self._write_scoped_value("variables", DataType(instr.target_type), instr.target_index, result)
        elif instr.target_cat == Category.CONSTANT:
            self._write_scoped_value("constants", DataType(instr.target_type), instr.target_index, result)
        elif instr.target_cat == Category.LIST:
            self._write_list_value("lists", DataType(instr.target_type), instr.target_index, result)
        elif instr.target_cat == Category.LIST_CONSTANT:
            self._write_list_value("constant_lists", DataType(instr.target_type), instr.target_index, result)

    def _execute_instruction(
        self,
        instr: GFSLInstruction,
        *,
        genome: Optional[GFSLGenome] = None,
        instruction_index: Optional[int] = None,
    ) -> bool:
        """Execute a single instruction. Returns True when the instruction ran."""
        op_code = instr.operation
        custom_op = custom_operations.get(op_code)
        try:
            op = Operation(op_code)
        except ValueError:
            op = None

        source1 = self._get_value(instr, 1)
        source2 = self._get_value(instr, 2)

        if instr.source1_cat != Category.NONE and self._is_void(source1):
            return False
        if instr.source2_cat != Category.NONE and self._is_void(source2):
            return False

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
            return False
        else:
            if op == Operation.IF:
                pass
            elif op == Operation.WHILE:
                pass
            elif op == Operation.FUNC:
                return False
            elif op == Operation.END:
                pass
            elif op == Operation.SET:
                if instr.source1_cat == Category.CONFIG:
                    self.config_state[instr.source1_value] = source2
            elif op == Operation.RESULT:
                pass
            elif op == Operation.CALL:
                if genome is None:
                    result = self.VOID
                else:
                    result = self._invoke_function(source1, genome)

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
                target_list = self._list_for_write(
                    "lists",
                    DataType(instr.target_type),
                    instr.target_index,
                )
                target_list.insert(0, source1)
                result = list(target_list)
            elif op == Operation.APPEND:
                target_list = self._list_for_write(
                    "lists",
                    DataType(instr.target_type),
                    instr.target_index,
                )
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
            self._assign_target(instr, result)

        if instruction_index is not None:
            self._executed_instruction_indices.add(int(instruction_index))
        return True

    def _invoke_function(
        self,
        function_ref: Any,
        genome: GFSLGenome,
    ) -> Any:
        """Execute a function body and return its value."""
        if (
            not isinstance(function_ref, tuple)
            or len(function_ref) != 2
            or not isinstance(function_ref[0], (int, DataType))
            or not isinstance(function_ref[1], int)
        ):
            return self.VOID

        key = (int(function_ref[0]), int(function_ref[1]))
        definition = self._function_defs.get(key)
        if definition is None:
            return self.VOID
        if self._call_depth >= self.max_call_depth:
            return self.VOID
        if (
            definition.return_dtype == DataType.NONE
            and self.require_void_external_writes
            and not self.allow_function_external_writes
        ):
            return self.VOID

        self._call_depth += 1
        self._call_frames.append(self._new_call_frame())
        self._executed_instruction_indices.add(definition.start_idx)
        try:
            idx = definition.start_idx + 1
            while idx < definition.end_idx:
                nested_end = self._function_start_to_end.get(idx)
                if nested_end is not None and idx != definition.start_idx:
                    idx = nested_end + 1
                    continue
                if 0 <= idx < len(genome.instructions):
                    self._execute_instruction(
                        genome.instructions[idx],
                        genome=genome,
                        instruction_index=idx,
                    )
                idx += 1

            if definition.return_source is None:
                self._executed_instruction_indices.add(definition.end_idx)
                return None if definition.return_dtype == DataType.NONE else self.VOID

            source_cat, source_dtype, source_idx = definition.return_source
            try:
                cat = Category(source_cat)
                dtype = DataType(source_dtype)
            except ValueError:
                return self.VOID

            result = self._get_pointer_value(cat, dtype, source_idx)
            self._executed_instruction_indices.add(definition.end_idx)
            if result is None and definition.return_dtype != DataType.NONE:
                return self.VOID
            return result
        finally:
            self._executed_instruction_indices.add(definition.end_idx)
            if self._call_frames:
                self._call_frames.pop()
            self._call_depth = max(0, self._call_depth - 1)

    def _get_pointer_value(self, cat: Category, dtype: DataType, idx: int) -> Any:
        if cat == Category.NONE:
            return None
        if cat == Category.VARIABLE:
            return self._read_scoped_value("variables", dtype, idx)
        if cat == Category.CONSTANT:
            if dtype == DataType.NONE:
                return None
            return self._read_scoped_value("constants", dtype, idx)
        if cat == Category.LIST:
            return self._read_scoped_value("lists", dtype, idx)
        if cat == Category.LIST_CONSTANT:
            return self._read_scoped_value("constant_lists", dtype, idx)
        if cat == Category.FUNCTION:
            return (int(dtype), int(idx))
        return 0.0

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
            return self._read_scoped_value("variables", dtype, val)
        if cat == Category.CONSTANT:
            if dtype == DataType.NONE:
                return None
            return self._read_scoped_value("constants", dtype, val)
        if cat == Category.LIST:
            return self._read_scoped_value("lists", dtype, val)
        if cat == Category.LIST_CONSTANT:
            return self._read_scoped_value("constant_lists", dtype, val)
        if cat == Category.FUNCTION:
            return (int(dtype), int(val))
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


__all__ = ["FunctionDefinition", "GFSLExecutor"]
