"""Experimental Kompute planning interfaces for GFSL genomes.

This module focuses on kernel composition and type/buffer planning.
It intentionally avoids hard-coupling to Python Kompute bindings so the
plan can be produced even on systems where Kompute is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

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
    ) -> KomputeExecutionPlan:
        if str(order).strip().lower() == "execution":
            indices = list(range(len(genome.instructions)))
        else:
            indices = list(genome.extract_effective_algorithm())
        return self.compose_indices(
            genome,
            indices=indices,
            keep_vram_state=keep_vram_state,
        )

    def compatibility_report(
        self,
        genome: GFSLGenome,
        *,
        order: str = "effective",
        keep_vram_state: bool = True,
    ) -> KomputeCompatibilityReport:
        plan = self.compose(
            genome,
            order=order,
            keep_vram_state=keep_vram_state,
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
    ) -> KomputeExecutionPlan:
        return self.planner.compose(
            genome,
            order=order,
            keep_vram_state=keep_vram_state,
        )

    def compatibility_report(
        self,
        genome: GFSLGenome,
        *,
        order: str = "effective",
        keep_vram_state: bool = True,
    ) -> KomputeCompatibilityReport:
        return self.planner.compatibility_report(
            genome,
            order=order,
            keep_vram_state=keep_vram_state,
        )

    def compile(
        self,
        genome: GFSLGenome,
        *,
        order: str = "effective",
        keep_vram_state: bool = True,
    ) -> Dict[str, Any]:
        plan = self.compose(
            genome,
            order=order,
            keep_vram_state=keep_vram_state,
        )
        mode = self._resolve_execute_mode()
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
                "Native shader compilation/execution is pending; "
                "simulated mode reuses CPU execution after Kompute compatibility checks."
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
        plan = self.compose(
            genome,
            order=order,
            keep_vram_state=keep_vram_state,
        )
        if not plan.stages:
            raise RuntimeError("Kompute plan produced no executable stages.")
        if plan.unsupported:
            raise RuntimeError(
                "Kompute plan has unsupported instructions "
                f"({len(plan.unsupported)} stages unsupported)."
            )
        mode = self._resolve_execute_mode()
        if mode == "native":
            return self._execute_native(
                genome,
                plan=plan,
                inputs=inputs,
                track_activity=track_activity,
                activity_tick=activity_tick,
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
    ) -> Dict[str, Any]:
        _ = (genome, plan, inputs, track_activity, activity_tick)
        if not self._has_kp_bindings():
            raise ModuleNotFoundError(
                "Kompute bindings are not installed (`kp` import failed)."
            )
        raise NotImplementedError(
            "Native Kompute Vulkan dispatch is not implemented yet. "
            "Use `compute_backend='kompute-sim'` (or runtime `execution_mode='simulated'`) "
            "to run plan-checked simulated execution."
        )

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
