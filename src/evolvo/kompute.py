"""Experimental Kompute planning interfaces for GFSL genomes.

This module focuses on kernel composition and type/buffer planning.
It intentionally avoids hard-coupling to Python Kompute bindings so the
plan can be produced even on systems where Kompute is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from .custom_ops import custom_operations, infer_source_type, resolve_operation_name
from .enums import (
    BOOLEAN_COMPARE_OPS,
    BOOLEAN_LOGIC_OPS,
    DECIMAL_OPS,
    LIST_QUERY_OPS,
    LIST_TARGET_OPS,
    LIST_VALUE_OPS,
    TENSOR_OPS,
    Category,
    DataType,
    Operation,
)
from .genome import GFSLGenome
from .instruction import GFSLInstruction
from .values import ValueEnumerations


_CONTROL_FLOW_OPS = {
    Operation.IF,
    Operation.WHILE,
    Operation.END,
    Operation.SET,
    Operation.RESULT,
    Operation.FUNC,
    Operation.CALL,
}


@dataclass(frozen=True)
class KomputeTypeSpec:
    """Scalar/vector type metadata used to configure Kompute buffers."""

    scalar: str
    components: int = 1
    bytes_per_component: int = 4

    @property
    def byte_width(self) -> int:
        return int(self.components) * int(self.bytes_per_component)


@dataclass(frozen=True)
class KomputeTypeOverride:
    """Optional per-operation type override for target/source operands."""

    target: Optional[KomputeTypeSpec] = None
    source1: Optional[KomputeTypeSpec] = None
    source2: Optional[KomputeTypeSpec] = None


@dataclass(frozen=True)
class KomputeKernelBinding:
    """Maps one GFSL opcode to a logical Kompute kernel key."""

    operation_code: int
    operation_name: str
    shader_key: str
    target_type: DataType
    source_types: Tuple[Optional[DataType], Optional[DataType]]
    persistent_target: bool = True


@dataclass(frozen=True)
class KomputeBufferRef:
    """Logical storage reference used in composed execution plans."""

    key: str
    category: Category
    dtype: DataType
    index: int
    persistent: bool


@dataclass(frozen=True)
class KomputeKernelStage:
    """One composed kernel launch for a GFSL instruction."""

    instruction_index: int
    operation_code: int
    operation_name: str
    shader_key: str
    target: KomputeBufferRef
    source1: Optional[KomputeBufferRef]
    source2: Optional[KomputeBufferRef]
    target_type: KomputeTypeSpec
    source1_type: Optional[KomputeTypeSpec]
    source2_type: Optional[KomputeTypeSpec]
    persistent_target: bool


@dataclass(frozen=True)
class KomputeUnsupportedInstruction:
    """Instruction that cannot currently be mapped to a Kompute stage."""

    instruction_index: int
    operation_code: int
    operation_name: str
    reason: str


@dataclass
class KomputeExecutionPlan:
    """Kernel composition output ready for runtime compilation/execution."""

    stages: List[KomputeKernelStage] = field(default_factory=list)
    persistent_buffers: List[str] = field(default_factory=list)
    transient_buffers: List[str] = field(default_factory=list)
    unsupported: List[KomputeUnsupportedInstruction] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        unsupported_by_operation: Dict[str, int] = {}
        for item in self.unsupported:
            key = str(item.operation_name)
            unsupported_by_operation[key] = int(unsupported_by_operation.get(key, 0)) + 1
        return {
            "stage_count": int(len(self.stages)),
            "persistent_buffers": list(self.persistent_buffers),
            "transient_buffers": list(self.transient_buffers),
            "unsupported": [
                {
                    "instruction_index": int(item.instruction_index),
                    "operation_code": int(item.operation_code),
                    "operation_name": str(item.operation_name),
                    "reason": str(item.reason),
                }
                for item in self.unsupported
            ],
            "unsupported_by_operation": unsupported_by_operation,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class KomputeCompatibilityReport:
    """Compact compatibility summary for one genome/selection."""

    stage_count: int
    unsupported_count: int
    unsupported_by_operation: Dict[str, int]
    notes: Tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        return int(self.stage_count) > 0 and int(self.unsupported_count) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supported": bool(self.supported),
            "stage_count": int(self.stage_count),
            "unsupported_count": int(self.unsupported_count),
            "unsupported_by_operation": {
                str(name): int(count)
                for name, count in self.unsupported_by_operation.items()
            },
            "notes": list(self.notes),
        }


class KomputeInstructionRegistry:
    """Registry for operation->kernel mappings and type resolution."""

    def __init__(self) -> None:
        self._type_specs: Dict[DataType, KomputeTypeSpec] = {
            DataType.NONE: KomputeTypeSpec("u32", components=1, bytes_per_component=4),
            DataType.BOOLEAN: KomputeTypeSpec("u8", components=1, bytes_per_component=1),
            DataType.DECIMAL: KomputeTypeSpec("f32", components=1, bytes_per_component=4),
            DataType.TENSOR: KomputeTypeSpec("f32", components=4, bytes_per_component=4),
        }
        self._bindings: Dict[int, KomputeKernelBinding] = {}
        self._op_type_overrides: Dict[int, KomputeTypeOverride] = {}
        self._register_default_bindings()

    def _register_default_bindings(self) -> None:
        for op in DECIMAL_OPS:
            shader_key = "decimal.unary" if op in {Operation.SQRT, Operation.ABS, Operation.SIN, Operation.COS, Operation.EXP, Operation.LOG} else "decimal.binary"
            self.register_binding(
                op,
                shader_key=shader_key,
                target_type=DataType.DECIMAL,
            )
        for op in BOOLEAN_COMPARE_OPS:
            self.register_binding(
                op,
                shader_key="boolean.compare",
                target_type=DataType.BOOLEAN,
                source_types=(DataType.DECIMAL, DataType.DECIMAL),
            )
        for op in BOOLEAN_LOGIC_OPS:
            source_types = (
                (DataType.BOOLEAN, None)
                if op == Operation.NOT
                else (DataType.BOOLEAN, DataType.BOOLEAN)
            )
            self.register_binding(
                op,
                shader_key="boolean.logic",
                target_type=DataType.BOOLEAN,
                source_types=source_types,
            )
        for op in TENSOR_OPS:
            self.register_binding(
                op,
                shader_key=f"tensor.{op.name.lower()}",
                target_type=DataType.TENSOR,
            )
        for op in LIST_QUERY_OPS:
            self.register_binding(
                op,
                shader_key=f"list.{op.name.lower()}",
                target_type=(
                    DataType.DECIMAL
                    if op == Operation.LISTCOUNT
                    else DataType.BOOLEAN
                ),
            )
        for op in LIST_TARGET_OPS:
            self.register_binding(
                op,
                shader_key=f"list.{op.name.lower()}",
                target_type=DataType.NONE,
            )
        for op in LIST_VALUE_OPS:
            self.register_binding(
                op,
                shader_key=f"list.{op.name.lower()}",
                target_type=DataType.NONE,
            )

    def set_default_type(self, dtype: Union[DataType, int], spec: KomputeTypeSpec) -> None:
        self._type_specs[DataType(dtype)] = spec

    def set_operation_type_override(
        self,
        operation: Union[int, Operation, str],
        *,
        target: Optional[KomputeTypeSpec] = None,
        source1: Optional[KomputeTypeSpec] = None,
        source2: Optional[KomputeTypeSpec] = None,
    ) -> None:
        op_code = self._resolve_operation_code(operation)
        self._op_type_overrides[op_code] = KomputeTypeOverride(
            target=target,
            source1=source1,
            source2=source2,
        )

    def register_binding(
        self,
        operation: Union[int, Operation, str],
        *,
        shader_key: str,
        target_type: Optional[Union[DataType, int]] = None,
        source_types: Optional[Tuple[Optional[Union[DataType, int]], Optional[Union[DataType, int]]]] = None,
        persistent_target: bool = True,
    ) -> int:
        op_code = self._resolve_operation_code(operation)
        op_name = resolve_operation_name(op_code)
        target_dtype = (
            DataType(target_type)
            if target_type is not None
            else self._infer_target_type(op_code)
        )
        if source_types is None:
            s1 = self._infer_source_type(op_code, 1)
            s2 = self._infer_source_type(op_code, 2)
            source_dtype = (s1, s2)
        else:
            source_dtype = (
                None if source_types[0] is None else DataType(source_types[0]),
                None if source_types[1] is None else DataType(source_types[1]),
            )
        self._bindings[op_code] = KomputeKernelBinding(
            operation_code=int(op_code),
            operation_name=str(op_name),
            shader_key=str(shader_key),
            target_type=target_dtype,
            source_types=source_dtype,
            persistent_target=bool(persistent_target),
        )
        return int(op_code)

    def binding_for(self, operation_code: int) -> Optional[KomputeKernelBinding]:
        return self._bindings.get(int(operation_code))

    def ensure_binding_for_opcode(self, operation_code: int) -> Optional[KomputeKernelBinding]:
        code = int(operation_code)
        bound = self.binding_for(code)
        if bound is not None:
            return bound

        custom = custom_operations.get(code)
        if custom is None:
            return None

        target_dtype = DataType(custom.target_type)
        arity = max(1, int(custom.arity))
        source1_dtype = custom.source_types[0] if custom.source_types[0] is not None else target_dtype
        source2_dtype = custom.source_types[1] if custom.source_types[1] is not None else (
            target_dtype if arity >= 2 else None
        )
        shader_key = "custom.{dtype}.arity{arity}".format(
            dtype=target_dtype.name.lower(),
            arity=arity,
        )
        self.register_binding(
            code,
            shader_key=shader_key,
            target_type=target_dtype,
            source_types=(source1_dtype, source2_dtype),
        )
        return self.binding_for(code)

    def resolve_type(
        self,
        dtype: Optional[Union[DataType, int]],
        *,
        operation_code: Optional[int] = None,
        operand: str = "target",
    ) -> Optional[KomputeTypeSpec]:
        if dtype is None:
            return None
        base = self._type_specs.get(DataType(dtype))
        if base is None:
            return None
        if operation_code is None:
            return base
        override = self._op_type_overrides.get(int(operation_code))
        if override is None:
            return base
        if operand == "target" and override.target is not None:
            return override.target
        if operand == "source1" and override.source1 is not None:
            return override.source1
        if operand == "source2" and override.source2 is not None:
            return override.source2
        return base

    def _resolve_operation_code(self, operation: Union[int, Operation, str]) -> int:
        if isinstance(operation, Operation):
            return int(operation)
        if isinstance(operation, str):
            key = operation.strip().upper()
            if not key:
                raise ValueError("Operation name cannot be empty.")
            if key in Operation.__members__:
                return int(Operation[key])
            custom_code = custom_operations.get_code_by_name(key)
            if custom_code is not None:
                return int(custom_code)
            raise ValueError(f"Unknown operation `{operation}`.")
        return int(operation)

    def _infer_source_type(self, operation_code: int, source_index: int) -> Optional[DataType]:
        inferred = int(infer_source_type(int(operation_code), int(source_index)))
        if inferred == int(DataType.NONE):
            return None
        return DataType(inferred)

    def _infer_target_type(self, operation_code: int) -> DataType:
        custom = custom_operations.get(int(operation_code))
        if custom is not None:
            return DataType(custom.target_type)
        try:
            op = Operation(int(operation_code))
        except ValueError:
            return DataType.NONE
        if op in DECIMAL_OPS:
            return DataType.DECIMAL
        if op in BOOLEAN_COMPARE_OPS or op in BOOLEAN_LOGIC_OPS or op in LIST_QUERY_OPS:
            return DataType.BOOLEAN if op in BOOLEAN_LOGIC_OPS else (
                DataType.DECIMAL if op == Operation.LISTCOUNT else DataType.BOOLEAN
            )
        if op in TENSOR_OPS:
            return DataType.TENSOR
        if op in LIST_TARGET_OPS:
            return DataType.NONE
        if op in LIST_VALUE_OPS:
            return DataType.NONE
        if op in _CONTROL_FLOW_OPS:
            return DataType.NONE
        return DataType.NONE


class GFSLKomputePlanner:
    """Compose a GFSL genome into a staged Kompute execution plan."""

    def __init__(self, registry: Optional[KomputeInstructionRegistry] = None) -> None:
        self.registry = registry or KomputeInstructionRegistry()

    def compose(
        self,
        genome: GFSLGenome,
        *,
        order: str = "effective",
        keep_vram_state: bool = True,
        native_dispatch_only: bool = False,
    ) -> KomputeExecutionPlan:
        if str(order).strip().lower() == "execution":
            indices = list(range(len(genome.instructions)))
        else:
            indices = list(genome.extract_effective_algorithm())
        return self.compose_indices(
            genome,
            indices=indices,
            keep_vram_state=keep_vram_state,
            native_dispatch_only=native_dispatch_only,
        )

    def compatibility_report(
        self,
        genome: GFSLGenome,
        *,
        order: str = "effective",
        keep_vram_state: bool = True,
        native_dispatch_only: bool = False,
    ) -> KomputeCompatibilityReport:
        plan = self.compose(
            genome,
            order=order,
            keep_vram_state=keep_vram_state,
            native_dispatch_only=native_dispatch_only,
        )
        unsupported_by_operation: Dict[str, int] = {}
        for item in plan.unsupported:
            op_name = str(item.operation_name)
            unsupported_by_operation[op_name] = int(
                unsupported_by_operation.get(op_name, 0) + 1
            )
        return KomputeCompatibilityReport(
            stage_count=int(len(plan.stages)),
            unsupported_count=int(len(plan.unsupported)),
            unsupported_by_operation=unsupported_by_operation,
            notes=tuple(str(note) for note in plan.notes),
        )

    def compose_indices(
        self,
        genome: GFSLGenome,
        *,
        indices: Iterable[int],
        keep_vram_state: bool = True,
        native_dispatch_only: bool = False,
    ) -> KomputeExecutionPlan:
        plan = KomputeExecutionPlan()
        persistent_keys: List[str] = []
        transient_keys: List[str] = []
        persistent_seen = set()
        transient_seen = set()

        for idx in indices:
            if idx < 0 or idx >= len(genome.instructions):
                continue
            instr = genome.instructions[int(idx)]
            stage = self._compose_instruction(
                instruction=instr,
                instruction_index=int(idx),
                keep_vram_state=bool(keep_vram_state),
                native_dispatch_only=bool(native_dispatch_only),
            )
            if isinstance(stage, KomputeUnsupportedInstruction):
                plan.unsupported.append(stage)
                continue
            plan.stages.append(stage)
            for ref in (stage.target, stage.source1, stage.source2):
                if ref is None:
                    continue
                if ref.persistent:
                    if ref.key not in persistent_seen:
                        persistent_seen.add(ref.key)
                        persistent_keys.append(ref.key)
                else:
                    if ref.key not in transient_seen:
                        transient_seen.add(ref.key)
                        transient_keys.append(ref.key)

        if not plan.stages:
            plan.notes.append("No GPU-compatible stages composed from selected instructions.")
        if plan.unsupported:
            plan.notes.append(
                "Unsupported instructions present; fallback execution path is required."
            )
        plan.persistent_buffers = persistent_keys
        plan.transient_buffers = transient_keys
        return plan

    def _compose_instruction(
        self,
        *,
        instruction: GFSLInstruction,
        instruction_index: int,
        keep_vram_state: bool,
        native_dispatch_only: bool,
    ) -> Union[KomputeKernelStage, KomputeUnsupportedInstruction]:
        op_code = int(instruction.operation)
        binding = self.registry.ensure_binding_for_opcode(op_code)
        if binding is None:
            return KomputeUnsupportedInstruction(
                instruction_index=instruction_index,
                operation_code=op_code,
                operation_name=resolve_operation_name(op_code),
                reason="No Kompute binding registered for this opcode.",
            )
        if self._is_control_flow(op_code):
            return KomputeUnsupportedInstruction(
                instruction_index=instruction_index,
                operation_code=op_code,
                operation_name=binding.operation_name,
                reason="Control-flow instructions are not directly composable into data kernels.",
            )
        if native_dispatch_only and not _is_native_shader_compatible_instruction(instruction):
            return KomputeUnsupportedInstruction(
                instruction_index=instruction_index,
                operation_code=op_code,
                operation_name=binding.operation_name,
                reason=(
                    "Instruction is not compatible with the current native scalar "
                    "Kompute dispatcher."
                ),
            )

        target_ref = self._target_ref(
            instruction,
            keep_vram_state=keep_vram_state and bool(binding.persistent_target),
        )
        if target_ref is None:
            return KomputeUnsupportedInstruction(
                instruction_index=instruction_index,
                operation_code=op_code,
                operation_name=binding.operation_name,
                reason="Instruction has no writable target buffer.",
            )

        source1_ref = self._source_ref(
            category=Category(instruction.source1_cat),
            dtype=binding.source_types[0],
            index=int(instruction.source1_value),
            keep_vram_state=keep_vram_state,
            label="s1",
        )
        source2_ref = self._source_ref(
            category=Category(instruction.source2_cat),
            dtype=binding.source_types[1],
            index=int(instruction.source2_value),
            keep_vram_state=keep_vram_state,
            label="s2",
        )

        return KomputeKernelStage(
            instruction_index=int(instruction_index),
            operation_code=int(binding.operation_code),
            operation_name=str(binding.operation_name),
            shader_key=str(binding.shader_key),
            target=target_ref,
            source1=source1_ref,
            source2=source2_ref,
            target_type=self.registry.resolve_type(
                binding.target_type,
                operation_code=binding.operation_code,
                operand="target",
            )
            or KomputeTypeSpec("f32"),
            source1_type=self.registry.resolve_type(
                binding.source_types[0],
                operation_code=binding.operation_code,
                operand="source1",
            ),
            source2_type=self.registry.resolve_type(
                binding.source_types[1],
                operation_code=binding.operation_code,
                operand="source2",
            ),
            persistent_target=bool(target_ref.persistent),
        )

    def _target_ref(
        self,
        instruction: GFSLInstruction,
        *,
        keep_vram_state: bool,
    ) -> Optional[KomputeBufferRef]:
        category = Category(instruction.target_cat)
        if category == Category.NONE:
            return None
        dtype = DataType(instruction.target_type)
        index = int(instruction.target_index)
        key = self._buffer_key(category=category, dtype=dtype, index=index, label="dst")
        persistent = bool(
            keep_vram_state
            and category in {
                Category.VARIABLE,
                Category.CONSTANT,
                Category.LIST,
                Category.LIST_CONSTANT,
            }
        )
        return KomputeBufferRef(
            key=key,
            category=category,
            dtype=dtype,
            index=index,
            persistent=persistent,
        )

    def _source_ref(
        self,
        *,
        category: Category,
        dtype: Optional[DataType],
        index: int,
        keep_vram_state: bool,
        label: str,
    ) -> Optional[KomputeBufferRef]:
        if category == Category.NONE:
            return None
        resolved_dtype = dtype if dtype is not None else DataType.DECIMAL
        persistent = bool(
            keep_vram_state
            and category in {
                Category.VARIABLE,
                Category.CONSTANT,
                Category.LIST,
                Category.LIST_CONSTANT,
            }
        )
        if category in {Category.VALUE, Category.CONFIG, Category.FUNCTION}:
            persistent = False
        key = self._buffer_key(
            category=category,
            dtype=resolved_dtype,
            index=int(index),
            label=label,
        )
        return KomputeBufferRef(
            key=key,
            category=category,
            dtype=resolved_dtype,
            index=int(index),
            persistent=persistent,
        )

    @staticmethod
    def _buffer_key(
        *,
        category: Category,
        dtype: DataType,
        index: int,
        label: str,
    ) -> str:
        return "{label}:{cat}:{dtype}:{idx}".format(
            label=str(label),
            cat=str(int(category)),
            dtype=str(int(dtype)),
            idx=str(int(index)),
        )

    @staticmethod
    def _is_control_flow(op_code: int) -> bool:
        try:
            return Operation(op_code) in _CONTROL_FLOW_OPS
        except ValueError:
            return False


_SCALAR_POINTER_CATEGORIES = {
    Category.VARIABLE,
    Category.CONSTANT,
}

_DECIMAL_OP_CODES = {int(op) for op in DECIMAL_OPS}
_DECIMAL_UNARY_OP_CODES = {
    int(Operation.SQRT),
    int(Operation.ABS),
    int(Operation.SIN),
    int(Operation.COS),
    int(Operation.EXP),
    int(Operation.LOG),
}
_BOOLEAN_COMPARE_OP_CODES = {int(op) for op in BOOLEAN_COMPARE_OPS}
_BOOLEAN_LOGIC_OP_CODES = {int(op) for op in BOOLEAN_LOGIC_OPS}
_BOOLEAN_UNARY_OP_CODES = {int(Operation.NOT)}


def _native_shader_family(op_code: int) -> Optional[str]:
    code = int(op_code)
    if code in _DECIMAL_OP_CODES:
        return "decimal"
    if code in _BOOLEAN_COMPARE_OP_CODES:
        return "boolean.compare"
    if code in _BOOLEAN_LOGIC_OP_CODES:
        return "boolean.logic"
    return None


def _native_is_unary_op(op_code: int) -> bool:
    code = int(op_code)
    return bool(code in _DECIMAL_UNARY_OP_CODES or code in _BOOLEAN_UNARY_OP_CODES)


def _native_is_valid_source_category(
    category: Category,
    dtype: DataType,
    *,
    allow_none: bool,
) -> bool:
    if allow_none and category == Category.NONE:
        return True
    if category in _SCALAR_POINTER_CATEGORIES:
        return dtype in {DataType.DECIMAL, DataType.BOOLEAN}
    if category == Category.VALUE:
        return True
    return False


def _is_native_shader_compatible_instruction(instruction: GFSLInstruction) -> bool:
    try:
        op_code = int(instruction.operation)
        family = _native_shader_family(op_code)
        if family is None:
            return False

        target_cat = Category(int(instruction.target_cat))
        if target_cat not in _SCALAR_POINTER_CATEGORIES:
            return False
        target_dtype = DataType(int(instruction.target_type))
        if family == "decimal" and target_dtype != DataType.DECIMAL:
            return False
        if family in {"boolean.compare", "boolean.logic"} and target_dtype != DataType.BOOLEAN:
            return False

        source1_cat = Category(int(instruction.source1_cat))
        source1_type = DataType(int(instruction.source1_type))
        if not _native_is_valid_source_category(
            source1_cat,
            source1_type,
            allow_none=False,
        ):
            return False

        source2_cat = Category(int(instruction.source2_cat))
        source2_type = DataType(int(instruction.source2_type))
        if _native_is_unary_op(op_code) and source2_cat == Category.NONE:
            return True
        return _native_is_valid_source_category(
            source2_cat,
            source2_type,
            allow_none=_native_is_unary_op(op_code),
        )
    except Exception:
        return False


_DECIMAL_SHADER_SOURCE = """#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0) buffer Src1Buffer { float src1[]; };
layout(binding = 1) buffer Src2Buffer { float src2[]; };
layout(binding = 2) buffer DstBuffer  { float dst[]; };
layout(push_constant) uniform PushConstants { float op_code; } push_consts;

const float EPS = 1e-9;

void main() {
    uint i = gl_GlobalInvocationID.x;
    float a = src1[i];
    float b = src2[i];
    int op = int(push_consts.op_code + 0.5);
    float out_value = a;

    switch (op) {
        case 30: out_value = a + b; break;
        case 31: out_value = a - b; break;
        case 32: out_value = a * b; break;
        case 33: out_value = (abs(b) > EPS) ? (a / b) : 0.0; break;
        case 34: {
            float rounded = round(b);
            bool exponent_is_integer = abs(b - rounded) <= EPS;
            out_value = (a < 0.0 && !exponent_is_integer) ? 0.0 : pow(a, b);
            break;
        }
        case 35: out_value = (a >= 0.0) ? sqrt(a) : 0.0; break;
        case 36: out_value = abs(a); break;
        case 37: out_value = sin(a); break;
        case 38: out_value = cos(a); break;
        case 39: out_value = exp(clamp(a, -80.0, 80.0)); break;
        case 40: out_value = (a > EPS) ? log(a) : -100.0; break;
        case 41: out_value = (abs(b) > EPS) ? mod(a, b) : 0.0; break;
        default: out_value = a; break;
    }

    dst[i] = out_value;
}
"""

_BOOLEAN_COMPARE_SHADER_SOURCE = """#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0) buffer Src1Buffer { float src1[]; };
layout(binding = 1) buffer Src2Buffer { float src2[]; };
layout(binding = 2) buffer DstBuffer  { float dst[]; };
layout(push_constant) uniform PushConstants { float op_code; } push_consts;

const float EPS = 1e-9;

void main() {
    uint i = gl_GlobalInvocationID.x;
    float a = src1[i];
    float b = src2[i];
    int op = int(push_consts.op_code + 0.5);
    float out_value = 0.0;

    switch (op) {
        case 10: out_value = (a > b) ? 1.0 : 0.0; break;
        case 11: out_value = (a < b) ? 1.0 : 0.0; break;
        case 12: out_value = (abs(a - b) < EPS) ? 1.0 : 0.0; break;
        case 13: out_value = (a >= b) ? 1.0 : 0.0; break;
        case 14: out_value = (a <= b) ? 1.0 : 0.0; break;
        case 15: out_value = (abs(a - b) >= EPS) ? 1.0 : 0.0; break;
        default: out_value = 0.0; break;
    }

    dst[i] = out_value;
}
"""

_BOOLEAN_LOGIC_SHADER_SOURCE = """#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0) buffer Src1Buffer { float src1[]; };
layout(binding = 1) buffer Src2Buffer { float src2[]; };
layout(binding = 2) buffer DstBuffer  { float dst[]; };
layout(push_constant) uniform PushConstants { float op_code; } push_consts;

void main() {
    uint i = gl_GlobalInvocationID.x;
    bool a = src1[i] != 0.0;
    bool b = src2[i] != 0.0;
    int op = int(push_consts.op_code + 0.5);
    float out_value = 0.0;

    switch (op) {
        case 16: out_value = (a && b) ? 1.0 : 0.0; break;
        case 17: out_value = (a || b) ? 1.0 : 0.0; break;
        case 18: out_value = (!a) ? 1.0 : 0.0; break;
        default: out_value = 0.0; break;
    }

    dst[i] = out_value;
}
"""

_GLOBAL_SPIRV_SHADER_CACHE: Dict[str, bytes] = {}


@dataclass
class _NativeTensorState:
    tensor: Any
    category: Category
    dtype: DataType
    index: int
    host_dirty: bool = False
    device_dirty: bool = False
    literal: bool = False


class _NativeKomputeHybridEngine:
    """Hybrid scalar dispatcher: Vulkan for supported ops, CPU fallback otherwise."""

    def __init__(self, executor: Any, *, keep_vram_state: bool) -> None:
        import kp  # type: ignore

        self.kp = kp
        self._sync_device_op, self._sync_local_op = self._sync_ops_for_kp(kp)
        self._has_sync_ops = bool(
            self._sync_device_op is not None and self._sync_local_op is not None
        )
        self._shared_memory_type = self._shared_memory_type_for_kp(kp)
        self._use_shared_memory_tensors = bool(
            (not self._has_sync_ops) and (self._shared_memory_type is not None)
        )
        self.executor = executor
        self.keep_vram_state = bool(keep_vram_state)
        self.manager = self._create_manager_with_fallback()
        self._compiler_path = self._detect_shader_compiler()
        self._tensor_states: Dict[Tuple[int, int, int], _NativeTensorState] = {}
        self._literal_states: Dict[Tuple[str, str], _NativeTensorState] = {}
        self._algorithms: Dict[Tuple[str, int, int, int], Any] = {}
        self._shader_spirv: Dict[str, bytes] = {}
        self.gpu_dispatch_count = 0
        self.cpu_fallback_count = 0
        self.cpu_full_sync_count = 0
        self.cpu_partial_sync_count = 0
        self.cpu_no_sync_count = 0
        self.cpu_synced_tensors = 0
        if (not self._has_sync_ops) and self._shared_memory_type is None:
            raise RuntimeError(
                "kp build does not expose sync ops (OpSyncDevice/OpSyncLocal or "
                "OpTensorSyncDevice/OpTensorSyncLocal) and has no shared memory type "
                "(MemoryTypes.deviceAndHost/host)."
            )

    @staticmethod
    def _detect_shader_compiler() -> Optional[str]:
        env_path = os.environ.get("EVOLVO_GLSL_COMPILER", "").strip()
        if env_path:
            expanded = os.path.expanduser(env_path)
            if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
                return expanded
        for candidate in ("glslangValidator", "glslc"):
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        return None

    @staticmethod
    def _shared_memory_type_for_kp(kp_module: Any) -> Optional[Any]:
        memory_types = getattr(kp_module, "MemoryTypes", None)
        if memory_types is not None and hasattr(memory_types, "deviceAndHost"):
            return memory_types.deviceAndHost
        if memory_types is not None and hasattr(memory_types, "host"):
            return memory_types.host
        if hasattr(kp_module, "deviceAndHost"):
            return getattr(kp_module, "deviceAndHost")
        if hasattr(kp_module, "host"):
            return getattr(kp_module, "host")
        return None

    @staticmethod
    def _sync_ops_for_kp(kp_module: Any) -> Tuple[Optional[Any], Optional[Any]]:
        sync_device = getattr(kp_module, "OpSyncDevice", None)
        sync_local = getattr(kp_module, "OpSyncLocal", None)
        if sync_device is None:
            sync_device = getattr(kp_module, "OpTensorSyncDevice", None)
        if sync_local is None:
            sync_local = getattr(kp_module, "OpTensorSyncLocal", None)
        return sync_device, sync_local

    @staticmethod
    def _vulkan_device_sort_key(index: int, descriptor: Any) -> Tuple[int, int, int]:
        name = ""
        if isinstance(descriptor, dict):
            name = str(descriptor.get("device_name", ""))
        elif descriptor is not None:
            name = str(descriptor)
        name_lc = name.strip().lower()
        is_software = any(
            token in name_lc for token in ("llvmpipe", "lavapipe", "swiftshader", "software")
        )
        looks_amd = any(token in name_lc for token in ("amd", "radeon", "navi", "gfx"))
        # Prefer hardware accelerators over software ICDs; among hardware devices, favor AMD names.
        return (1 if is_software else 0, 0 if looks_amd else 1, int(index))

    @staticmethod
    def _env_int(name: str) -> Optional[int]:
        raw = os.environ.get(name)
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError as exc:
            raise RuntimeError(f"Invalid integer env {name}={raw!r}") from exc

    def _candidate_device_indices(self) -> List[int]:
        configured = self._env_int("EVOLVO_KOMPUTE_DEVICE_INDEX")
        if configured is not None:
            return [max(0, int(configured))]
        try:
            probe = self.kp.Manager(0)
            listed = probe.list_devices()
            if isinstance(listed, list) and listed:
                ranked = sorted(
                    list(enumerate(listed)),
                    key=lambda item: self._vulkan_device_sort_key(int(item[0]), item[1]),
                )
                return [int(idx) for idx, _item in ranked]
        except Exception:
            pass
        return [0, 1, 2, 3]

    def _candidate_queue_families(self) -> List[Optional[int]]:
        configured = self._env_int("EVOLVO_KOMPUTE_QUEUE_FAMILY")
        if configured is not None:
            return [max(0, int(configured))]
        return [None, 0]

    def _probe_manager_sync(self, manager: Any) -> None:
        if self._has_sync_ops:
            tensor = manager.tensor(np.array([1.0], dtype=np.float32))
            sequence = manager.sequence()
            sequence.record(self._sync_device_op([tensor]))  # type: ignore[misc]
            sequence.record(self._sync_local_op([tensor]))  # type: ignore[misc]
            sequence.eval()
            _ = float(tensor.data()[0])
            return
        if self._shared_memory_type is not None:
            tensor = manager.tensor(
                np.array([1.0], dtype=np.float32),
                self._shared_memory_type,
            )
            # Avoid empty sequence eval(): certain AMD Vulkan stacks can stall on this call.
            _ = float(tensor.data()[0])
            return
        # No sync API and no shared tensors: manager creation itself is the lightweight probe.

    def _make_tensor(self, values: np.ndarray) -> Any:
        if self._use_shared_memory_tensors and self._shared_memory_type is not None:
            return self.manager.tensor(values, self._shared_memory_type)
        return self.manager.tensor(values)

    def _build_manager(self, *, device_index: int, queue_family: Optional[int]) -> Any:
        if queue_family is None:
            return self.kp.Manager(int(device_index))
        return self.kp.Manager(
            device=int(device_index),
            family_queue_indices=[int(queue_family)],
            desired_extensions=[],
        )

    def _create_manager_with_fallback(self) -> Any:
        errors: List[str] = []
        for device_index in self._candidate_device_indices():
            for queue_family in self._candidate_queue_families():
                queue_text = "default" if queue_family is None else str(int(queue_family))
                try:
                    manager = self._build_manager(
                        device_index=int(device_index),
                        queue_family=queue_family,
                    )
                    self._probe_manager_sync(manager)
                    return manager
                except Exception as exc:
                    errors.append(
                        "device={device} queue_family={queue} -> {err}".format(
                            device=int(device_index),
                            queue=queue_text,
                            err=str(exc),
                        )
                    )
        detail = "; ".join(errors[-4:]) if errors else "no attempts"
        raise RuntimeError(
            "Failed to initialize Kompute Vulkan manager. "
            "Set EVOLVO_KOMPUTE_DEVICE_INDEX and/or EVOLVO_KOMPUTE_QUEUE_FAMILY "
            "(or run pcpl-evolvo `--kompute-check-libs`). "
            f"Recent attempts: {detail}"
        )

    def _compile_shader_to_spirv(self, source: str, *, shader_name: str) -> bytes:
        if not self._compiler_path:
            raise RuntimeError(
                "No GLSL compiler found. Install `glslangValidator` or `glslc`, "
                "or set `EVOLVO_GLSL_COMPILER`."
            )
        compiler = os.path.basename(self._compiler_path)
        with tempfile.TemporaryDirectory(prefix="evolvo-kompute-") as tmp_dir:
            src_path = os.path.join(tmp_dir, f"{shader_name}.comp")
            spv_path = os.path.join(tmp_dir, f"{shader_name}.spv")
            with open(src_path, "w", encoding="utf-8") as handle:
                handle.write(source)
            if compiler == "glslc":
                cmd = [
                    self._compiler_path,
                    src_path,
                    "-o",
                    spv_path,
                    "-fshader-stage=compute",
                ]
            else:
                cmd = [
                    self._compiler_path,
                    "-V",
                    "-S",
                    "comp",
                    src_path,
                    "-o",
                    spv_path,
                ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                stdout = (proc.stdout or "").strip()
                detail = stderr or stdout or "unknown compiler error"
                raise RuntimeError(
                    f"Kompute shader compilation failed for `{shader_name}`: {detail}"
                )
            with open(spv_path, "rb") as handle:
                return handle.read()

    def _shader_source(self, family: str) -> str:
        if family == "decimal":
            return _DECIMAL_SHADER_SOURCE
        if family == "boolean.compare":
            return _BOOLEAN_COMPARE_SHADER_SOURCE
        if family == "boolean.logic":
            return _BOOLEAN_LOGIC_SHADER_SOURCE
        raise ValueError(f"Unsupported shader family `{family}`.")

    def _shader_spirv_bytes(self, family: str) -> bytes:
        global_cached = _GLOBAL_SPIRV_SHADER_CACHE.get(family)
        if global_cached is not None:
            self._shader_spirv[family] = global_cached
            return global_cached
        cached = self._shader_spirv.get(family)
        if cached is not None:
            return cached
        source = self._shader_source(family)
        compiled = self._compile_shader_to_spirv(source, shader_name=f"gfsl_{family.replace('.', '_')}")
        _GLOBAL_SPIRV_SHADER_CACHE[family] = compiled
        self._shader_spirv[family] = compiled
        return compiled

    @staticmethod
    def _op_shader_family(op_code: int) -> Optional[str]:
        return _native_shader_family(int(op_code))

    @staticmethod
    def _is_unary_op(op_code: int) -> bool:
        return _native_is_unary_op(int(op_code))

    def can_dispatch(self, instruction: GFSLInstruction) -> bool:
        return _is_native_shader_compatible_instruction(instruction)

    @staticmethod
    def _is_valid_source_category(
        category: Category,
        dtype: DataType,
        *,
        allow_none: bool,
    ) -> bool:
        return _native_is_valid_source_category(category, dtype, allow_none=allow_none)

    def _scalar_ref_from_target(
        self,
        instruction: GFSLInstruction,
    ) -> Tuple[int, int, int]:
        return (
            int(instruction.target_cat),
            int(instruction.target_type),
            int(instruction.target_index),
        )

    def _scalar_ref_from_source(
        self,
        instruction: GFSLInstruction,
        source_num: int,
    ) -> Optional[Tuple[int, int, int]]:
        if source_num == 1:
            category = Category(int(instruction.source1_cat))
            dtype = DataType(int(instruction.source1_type))
            index = int(instruction.source1_value)
        else:
            category = Category(int(instruction.source2_cat))
            dtype = DataType(int(instruction.source2_type))
            index = int(instruction.source2_value)
        if category not in _SCALAR_POINTER_CATEGORIES:
            return None
        return (int(category), int(dtype), int(index))

    @staticmethod
    def _float_key(value: float) -> str:
        return f"{float(value):.17g}"

    def _coerce_value_to_float(self, value: Any, *, dtype: DataType) -> float:
        if self.executor._is_void(value):
            return 0.0
        if dtype == DataType.BOOLEAN:
            return 1.0 if bool(value) else 0.0
        try:
            parsed = float(value)
        except Exception:
            return 0.0
        if math.isnan(parsed) or math.isinf(parsed):
            return 0.0
        return float(parsed)

    def _coerce_float_to_host(self, value: float, *, dtype: DataType) -> Any:
        if dtype == DataType.BOOLEAN:
            return bool(value != 0.0)
        return float(value)

    def _read_executor_scalar(self, ref: Tuple[int, int, int]) -> Any:
        category = Category(ref[0])
        dtype = DataType(ref[1])
        index = int(ref[2])
        return self.executor._get_pointer_value(category, dtype, index)

    def _write_executor_scalar(self, ref: Tuple[int, int, int], value: Any) -> None:
        category = Category(ref[0])
        dtype = DataType(ref[1])
        index = int(ref[2])
        if category == Category.VARIABLE:
            self.executor._write_scoped_value("variables", dtype, index, value)
            return
        if category == Category.CONSTANT:
            self.executor._write_scoped_value("constants", dtype, index, value)
            return

    def _ensure_scalar_state(self, ref: Tuple[int, int, int]) -> _NativeTensorState:
        cached = self._tensor_states.get(ref)
        if cached is not None:
            return cached
        category = Category(ref[0])
        dtype = DataType(ref[1])
        index = int(ref[2])
        host_value = self._read_executor_scalar(ref)
        initial_value = self._coerce_value_to_float(host_value, dtype=dtype)
        tensor = self._make_tensor(np.array([initial_value], dtype=np.float32))
        state = _NativeTensorState(
            tensor=tensor,
            category=category,
            dtype=dtype,
            index=index,
            host_dirty=True,
            device_dirty=False,
            literal=False,
        )
        self._tensor_states[ref] = state
        return state

    def _literal_state(self, value: float) -> _NativeTensorState:
        key = ("literal", self._float_key(value))
        cached = self._literal_states.get(key)
        if cached is not None:
            return cached
        tensor = self._make_tensor(np.array([float(value)], dtype=np.float32))
        state = _NativeTensorState(
            tensor=tensor,
            category=Category.VALUE,
            dtype=DataType.DECIMAL,
            index=0,
            host_dirty=True,
            device_dirty=False,
            literal=True,
        )
        self._literal_states[key] = state
        return state

    def _source_state(self, instruction: GFSLInstruction, source_num: int) -> Optional[_NativeTensorState]:
        if source_num == 1:
            category = Category(int(instruction.source1_cat))
        else:
            category = Category(int(instruction.source2_cat))

        if category in _SCALAR_POINTER_CATEGORIES:
            ref = self._scalar_ref_from_source(instruction, source_num)
            if ref is None:
                return None
            return self._ensure_scalar_state(ref)
        if category == Category.VALUE:
            value = self.executor._get_value(instruction, source_num)
            value_float = self._coerce_value_to_float(value, dtype=DataType.DECIMAL)
            return self._literal_state(value_float)
        if category == Category.NONE:
            return self._literal_state(0.0)
        return None

    def _sync_host_to_device(self, states: List[_NativeTensorState]) -> None:
        pending: List[_NativeTensorState] = []
        seen = set()
        for state in states:
            marker = id(state.tensor)
            if marker in seen:
                continue
            if not state.host_dirty:
                continue
            seen.add(marker)
            pending.append(state)
        if not pending:
            return
        if self._has_sync_ops:
            sequence = self.manager.sequence()
            sequence.record(self._sync_device_op([state.tensor for state in pending]))  # type: ignore[misc]
            sequence.eval()
        for state in pending:
            state.host_dirty = False

    def _sync_device_to_host(self, states: List[_NativeTensorState]) -> int:
        pending: List[_NativeTensorState] = []
        seen = set()
        for state in states:
            marker = id(state.tensor)
            if marker in seen:
                continue
            if not state.device_dirty:
                continue
            seen.add(marker)
            pending.append(state)
        if not pending:
            return 0
        if self._has_sync_ops:
            sequence = self.manager.sequence()
            sequence.record(self._sync_local_op([state.tensor for state in pending]))  # type: ignore[misc]
            sequence.eval()
        for state in pending:
            state.device_dirty = False
            if state.literal:
                continue
            host_raw = float(state.tensor.data()[0])
            host_value = self._coerce_float_to_host(host_raw, dtype=state.dtype)
            ref = (int(state.category), int(state.dtype), int(state.index))
            self._write_executor_scalar(ref, host_value)
        return int(len(pending))

    def sync_all_device_to_host(self) -> int:
        dirty = [state for state in self._tensor_states.values() if state.device_dirty]
        return self._sync_device_to_host(dirty)

    def _sync_refs_device_to_host(
        self,
        refs: Iterable[Tuple[int, int, int]],
    ) -> int:
        dirty_states: List[_NativeTensorState] = []
        seen_refs = set()
        for ref in refs:
            ref_key = (int(ref[0]), int(ref[1]), int(ref[2]))
            if ref_key in seen_refs:
                continue
            seen_refs.add(ref_key)
            state = self._tensor_states.get(ref_key)
            if state is None or not state.device_dirty:
                continue
            dirty_states.append(state)
        return self._sync_device_to_host(dirty_states)

    def _requires_full_cpu_sync(self, instruction: GFSLInstruction) -> bool:
        if int(instruction.operation) == int(Operation.CALL):
            return True
        if (
            int(instruction.source1_cat) == int(Category.FUNCTION)
            or int(instruction.source2_cat) == int(Category.FUNCTION)
        ):
            return True
        custom_op = custom_operations.get(int(instruction.operation))
        return bool(custom_op is not None and custom_op.accepts_context)

    def _cpu_source_scalar_refs(
        self,
        instruction: GFSLInstruction,
    ) -> List[Tuple[int, int, int]]:
        refs: List[Tuple[int, int, int]] = []
        source1_ref = self._scalar_ref_from_source(instruction, 1)
        if source1_ref is not None:
            refs.append(source1_ref)
        source2_ref = self._scalar_ref_from_source(instruction, 2)
        if source2_ref is not None:
            refs.append(source2_ref)
        return refs

    def sync_for_cpu_instruction(self, instruction: GFSLInstruction) -> Tuple[str, int]:
        if self._requires_full_cpu_sync(instruction):
            synced = self.sync_all_device_to_host()
            return ("full", int(synced))

        refs = self._cpu_source_scalar_refs(instruction)
        if not refs:
            return ("none", 0)
        synced = self._sync_refs_device_to_host(refs)
        if synced > 0:
            return ("partial", int(synced))
        return ("none", 0)

    def record_cpu_fallback(self, *, sync_mode: str, synced_tensors: int) -> None:
        self.cpu_fallback_count += 1
        self.cpu_synced_tensors += max(0, int(synced_tensors))
        mode = str(sync_mode).strip().lower()
        if mode == "full":
            self.cpu_full_sync_count += 1
        elif mode == "partial":
            self.cpu_partial_sync_count += 1
        else:
            self.cpu_no_sync_count += 1

    def _shader_algorithm(
        self,
        family: str,
        source1: _NativeTensorState,
        source2: _NativeTensorState,
        target: _NativeTensorState,
    ) -> Any:
        key = (family, id(source1.tensor), id(source2.tensor), id(target.tensor))
        cached = self._algorithms.get(key)
        if cached is not None:
            return cached
        spirv = self._shader_spirv_bytes(family)
        algorithm = self.manager.algorithm(
            [source1.tensor, source2.tensor, target.tensor],
            spirv,
            [1, 1, 1],
            [],
            [0.0],
        )
        self._algorithms[key] = algorithm
        return algorithm

    def _source_is_void(self, ref: Tuple[int, int, int]) -> bool:
        state = self._tensor_states.get(ref)
        if state is not None and state.device_dirty:
            return False
        value = self._read_executor_scalar(ref)
        return bool(self.executor._is_void(value))

    def dispatch_instruction(
        self,
        instruction: GFSLInstruction,
        *,
        instruction_index: int,
    ) -> bool:
        if not self.can_dispatch(instruction):
            return False

        source1_ref = self._scalar_ref_from_source(instruction, 1)
        if source1_ref is not None and self._source_is_void(source1_ref):
            return False
        source2_ref = self._scalar_ref_from_source(instruction, 2)
        if source2_ref is not None and self._source_is_void(source2_ref):
            return False

        op_code = int(instruction.operation)
        family = self._op_shader_family(op_code)
        if family is None:
            return False

        source1_state = self._source_state(instruction, 1)
        source2_state = self._source_state(instruction, 2)
        if source1_state is None or source2_state is None:
            return False
        target_ref = self._scalar_ref_from_target(instruction)
        target_state = self._ensure_scalar_state(target_ref)

        self._sync_host_to_device([source1_state, source2_state, target_state])
        algorithm = self._shader_algorithm(family, source1_state, source2_state, target_state)

        sequence = self.manager.sequence()
        sequence.record(self.kp.OpAlgoDispatch(algorithm, [float(op_code)]))
        sequence.eval()

        target_state.device_dirty = True
        target_state.host_dirty = False
        self.gpu_dispatch_count += 1
        self.executor._executed_instruction_indices.add(int(instruction_index))
        return True

    def absorb_cpu_target_update(self, instruction: GFSLInstruction) -> None:
        category = Category(int(instruction.target_cat))
        if category not in _SCALAR_POINTER_CATEGORIES:
            return
        dtype = DataType(int(instruction.target_type))
        ref = (
            int(category),
            int(dtype),
            int(instruction.target_index),
        )
        value = self._read_executor_scalar(ref)
        state = self._tensor_states.get(ref)
        if self.executor._is_void(value):
            if state is not None:
                state.host_dirty = False
                state.device_dirty = False
            return
        if state is None:
            state = self._ensure_scalar_state(ref)
        state.tensor.data()[0] = self._coerce_value_to_float(value, dtype=dtype)
        state.host_dirty = True
        state.device_dirty = False

class GFSLKomputeRuntime:
    """Kompute runtime facade with native or simulated execution modes."""

    def __init__(
        self,
        planner: Optional[GFSLKomputePlanner] = None,
        *,
        execution_mode: str = "native",
        allow_auto_simulation_fallback: bool = True,
    ) -> None:
        self.planner = planner or GFSLKomputePlanner()
        self.execution_mode = self._normalize_execution_mode(execution_mode)
        self.allow_auto_simulation_fallback = bool(allow_auto_simulation_fallback)
        self._last_native_stats: Optional[Dict[str, int]] = None

    @staticmethod
    def _normalize_execution_mode(mode: str) -> str:
        mode_norm = str(mode).strip().lower()
        if mode_norm not in {"native", "simulated", "auto"}:
            return "native"
        return mode_norm

    @staticmethod
    def _has_kp_bindings() -> bool:
        try:
            import kp  # type: ignore
            _ = kp
            return True
        except Exception:
            return False

    def is_available(self) -> bool:
        mode = self._normalize_execution_mode(self.execution_mode)
        if mode == "simulated":
            return True
        if mode == "native":
            return self._has_kp_bindings()
        # auto
        return bool(self._has_kp_bindings() or self.allow_auto_simulation_fallback)

    def _resolve_execute_mode(self) -> str:
        mode = self._normalize_execution_mode(self.execution_mode)
        if mode == "simulated":
            return "simulated"
        if mode == "native":
            return "native"
        # auto
        if self._has_kp_bindings():
            return "native"
        return "simulated" if self.allow_auto_simulation_fallback else "native"

    def compose(
        self,
        genome: GFSLGenome,
        *,
        order: str = "effective",
        keep_vram_state: bool = True,
        native_dispatch_only: Optional[bool] = None,
    ) -> KomputeExecutionPlan:
        native_only = (
            bool(native_dispatch_only)
            if native_dispatch_only is not None
            else self._resolve_execute_mode() == "native"
        )
        return self.planner.compose(
            genome,
            order=order,
            keep_vram_state=keep_vram_state,
            native_dispatch_only=native_only,
        )

    def compatibility_report(
        self,
        genome: GFSLGenome,
        *,
        order: str = "effective",
        keep_vram_state: bool = True,
        native_dispatch_only: Optional[bool] = None,
    ) -> KomputeCompatibilityReport:
        native_only = (
            bool(native_dispatch_only)
            if native_dispatch_only is not None
            else self._resolve_execute_mode() == "native"
        )
        return self.planner.compatibility_report(
            genome,
            order=order,
            keep_vram_state=keep_vram_state,
            native_dispatch_only=native_only,
        )

    def compile(
        self,
        genome: GFSLGenome,
        *,
        order: str = "effective",
        keep_vram_state: bool = True,
    ) -> Dict[str, Any]:
        mode = self._resolve_execute_mode()
        plan = self.compose(
            genome,
            order=order,
            keep_vram_state=keep_vram_state,
            native_dispatch_only=(mode == "native"),
        )
        if mode == "native" and not self._has_kp_bindings():
            raise ModuleNotFoundError(
                "Kompute bindings are not installed (`kp` import failed). "
                "Plan composition is available, but native runtime compilation is unavailable."
            )
        return {
            "status": "planned",
            "plan": plan.to_dict(),
            "execution_mode": mode,
            "kp_available": bool(self._has_kp_bindings()),
            "note": (
                "Native runtime dispatches supported scalar stages via Vulkan shaders. "
                "Unsupported stages execute through CPU fallback with state synchronization."
            ),
        }

    def execute(
        self,
        genome: GFSLGenome,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        track_activity: Optional[bool] = None,
        activity_tick: Optional[int] = None,
        order: str = "effective",
        keep_vram_state: bool = True,
    ) -> Dict[str, Any]:
        """Execute genome through native or simulated Kompute runtime."""
        self._last_native_stats = None
        mode = self._resolve_execute_mode()
        plan = self.compose(
            genome,
            order=order,
            keep_vram_state=keep_vram_state,
            native_dispatch_only=(mode == "native"),
        )
        if not plan.stages:
            raise RuntimeError("Kompute plan produced no executable stages.")
        if mode == "native":
            return self._execute_native(
                genome,
                plan=plan,
                inputs=inputs,
                track_activity=track_activity,
                activity_tick=activity_tick,
                order=order,
                keep_vram_state=keep_vram_state,
            )
        if mode == "simulated":
            return self._execute_simulated(
                genome,
                plan=plan,
                inputs=inputs,
                track_activity=track_activity,
                activity_tick=activity_tick,
            )
        raise RuntimeError(
            f"Unsupported Kompute execution mode `{mode}`."
        )

    def _execute_native(
        self,
        genome: GFSLGenome,
        *,
        plan: KomputeExecutionPlan,
        inputs: Optional[Dict[str, Any]],
        track_activity: Optional[bool],
        activity_tick: Optional[int],
        order: str,
        keep_vram_state: bool,
    ) -> Dict[str, Any]:
        _ = plan
        if not self._has_kp_bindings():
            raise ModuleNotFoundError(
                "Kompute bindings are not installed (`kp` import failed)."
            )

        from .executor import GFSLExecutor

        cpu_executor = GFSLExecutor(compute_backend="cpu")
        self._seed_inputs(cpu_executor, inputs)
        cpu_executor._index_functions(genome)
        selected_indices = self._selected_indices(genome, order=order)
        stage_indices = {int(stage.instruction_index) for stage in plan.stages}

        bridge = _NativeKomputeHybridEngine(
            cpu_executor,
            keep_vram_state=keep_vram_state,
        )
        for idx in cpu_executor._main_execution_indices(selected_indices):
            if idx < 0 or idx >= len(genome.instructions):
                continue
            instruction = genome.instructions[int(idx)]
            ran_on_gpu = False
            if int(idx) in stage_indices and bridge.can_dispatch(instruction):
                try:
                    ran_on_gpu = bridge.dispatch_instruction(
                        instruction,
                        instruction_index=int(idx),
                    )
                except Exception:
                    ran_on_gpu = False
            if ran_on_gpu:
                continue
            sync_mode, synced_tensors = bridge.sync_for_cpu_instruction(instruction)
            bridge.record_cpu_fallback(
                sync_mode=sync_mode,
                synced_tensors=synced_tensors,
            )
            cpu_executor._execute_instruction(
                instruction,
                genome=genome,
                instruction_index=int(idx),
            )
            bridge.absorb_cpu_target_update(instruction)

        final_sync_count = bridge.sync_all_device_to_host()
        self._last_native_stats = {
            "gpu_dispatch_count": int(bridge.gpu_dispatch_count),
            "cpu_fallback_count": int(bridge.cpu_fallback_count),
            "cpu_full_sync_count": int(bridge.cpu_full_sync_count),
            "cpu_partial_sync_count": int(bridge.cpu_partial_sync_count),
            "cpu_no_sync_count": int(bridge.cpu_no_sync_count),
            "cpu_synced_tensors": int(bridge.cpu_synced_tensors),
            "final_sync_count": int(final_sync_count),
        }

        should_track = (
            cpu_executor.track_instruction_activity
            if track_activity is None
            else bool(track_activity)
        )
        if should_track and cpu_executor._executed_instruction_indices:
            genome.record_instruction_activity(
                sorted(cpu_executor._executed_instruction_indices),
                tick=activity_tick,
            )
        return self._collect_outputs(cpu_executor, genome)

    @staticmethod
    def _selected_indices(genome: GFSLGenome, *, order: str) -> List[int]:
        if str(order).strip().lower() == "execution":
            return list(range(len(genome.instructions)))
        return [int(idx) for idx in genome.extract_effective_algorithm()]

    @staticmethod
    def _seed_inputs(cpu_executor: Any, inputs: Optional[Dict[str, Any]]) -> None:
        cpu_executor.reset()
        if not inputs:
            return
        for key, value in inputs.items():
            dtype_char = str(key)[0].lower() if key else ""
            if "!#" in key:
                category = Category.LIST_CONSTANT
                idx_str = key.split("!#", 1)[1]
            elif "!" in key:
                category = Category.LIST
                idx_str = key.split("!", 1)[1]
            elif "$" in key:
                category = Category.VARIABLE
                idx_str = key.split("$", 1)[1]
            elif "#" in key:
                category = Category.CONSTANT
                idx_str = key.split("#", 1)[1]
            else:
                continue
            try:
                idx = int(idx_str)
            except Exception:
                continue

            dtype = {
                "b": DataType.BOOLEAN,
                "d": DataType.DECIMAL,
                "t": DataType.TENSOR,
                "n": DataType.NONE,
            }.get(dtype_char, DataType.DECIMAL)

            if category == Category.CONSTANT:
                cpu_executor.constants[dtype][idx] = value
            elif category == Category.VARIABLE:
                cpu_executor.variables[dtype][idx] = value
            elif category == Category.LIST:
                cpu_executor.lists[dtype][idx] = cpu_executor._coerce_list_input(value)
            elif category == Category.LIST_CONSTANT:
                cpu_executor.constant_lists[dtype][idx] = cpu_executor._coerce_list_input(value)

    @staticmethod
    def _collect_outputs(cpu_executor: Any, genome: GFSLGenome) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {}
        for category, dtype, idx in genome.outputs:
            if category == Category.VARIABLE:
                key = f"{DataType(dtype).name[0].lower()}${idx}"
                value = cpu_executor.variables[dtype][idx]
                outputs[key] = None if cpu_executor._is_void(value) else value
            elif category == Category.CONSTANT:
                key = f"{DataType(dtype).name[0].lower()}#{idx}"
                value = cpu_executor.constants[dtype][idx]
                outputs[key] = None if cpu_executor._is_void(value) else value
        return outputs

    def _execute_simulated(
        self,
        genome: GFSLGenome,
        *,
        plan: KomputeExecutionPlan,
        inputs: Optional[Dict[str, Any]],
        track_activity: Optional[bool],
        activity_tick: Optional[int],
    ) -> Dict[str, Any]:
        _ = plan
        # Reuse trusted CPU executor semantics while forcing Kompute plan compatibility.
        from .executor import GFSLExecutor

        cpu_executor = GFSLExecutor(compute_backend="cpu")
        self._last_native_stats = None
        return cpu_executor.execute(
            genome,
            inputs=inputs,
            track_activity=track_activity,
            activity_tick=activity_tick,
        )


__all__ = [
    "KomputeTypeSpec",
    "KomputeTypeOverride",
    "KomputeKernelBinding",
    "KomputeBufferRef",
    "KomputeKernelStage",
    "KomputeUnsupportedInstruction",
    "KomputeExecutionPlan",
    "KomputeCompatibilityReport",
    "KomputeInstructionRegistry",
    "GFSLKomputePlanner",
    "GFSLKomputeRuntime",
]
