"""Execution engine for GFSL genomes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import threading
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .custom_ops import custom_operations
from .enums import Category, ConfigProperty, DataType, Operation
from .genome import GFSLGenome
from .instruction import GFSLInstruction
from .settings import (
    DEFAULT_COMPUTE_BACKEND,
    DEFAULT_KOMPUTE_FAIL_HARD,
    DEFAULT_KOMPUTE_FORCE_CPU_ON_PARTIAL_COVERAGE,
    DEFAULT_KOMPUTE_KEEP_VRAM_STATE,
    DEFAULT_KOMPUTE_MAX_UNSUPPORTED_COUNT,
    DEFAULT_KOMPUTE_MAX_UNSUPPORTED_SHARE,
    DEFAULT_KOMPUTE_MIN_NATIVE_STAGE_COUNT,
    DEFAULT_KOMPUTE_MIN_NATIVE_STAGE_SHARE,
    DEFAULT_KOMPUTE_NATIVE_ENABLE_BOOLEAN_COMPARE,
    DEFAULT_KOMPUTE_NATIVE_ENABLE_BOOLEAN_LOGIC,
    DEFAULT_KOMPUTE_NATIVE_ENABLE_DECIMAL,
    DEFAULT_KOMPUTE_NATIVE_ENABLE_LIST_QUERY,
    DEFAULT_KOMPUTE_RUNTIME_MODE,
    DEFAULT_KOMPUTE_WARN_ON_FALLBACK,
)
from .values import ValueEnumerations

_GLOBAL_AUTO_KOMPUTE_DISABLE_REASON: Optional[str] = None
_GLOBAL_KOMPUTE_WARNING_COUNT_BY_BUCKET: Dict[str, int] = defaultdict(int)
_GLOBAL_KOMPUTE_WARNING_SUPPRESSED_BUCKETS: Set[str] = set()
_GLOBAL_KOMPUTE_WARNING_LOCK = threading.Lock()


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
        compute_backend: Optional[str] = None,
        kompute_runtime_mode: Optional[str] = None,
        kompute_warn_on_fallback: Optional[bool] = None,
        kompute_fail_hard: Optional[bool] = None,
        kompute_keep_vram_state: Optional[bool] = None,
        kompute_min_native_stage_count: Optional[int] = None,
        kompute_min_native_stage_share: Optional[float] = None,
        kompute_max_unsupported_share: Optional[float] = None,
        kompute_max_unsupported_count: Optional[int] = None,
        kompute_force_cpu_on_partial_coverage: Optional[bool] = None,
        kompute_native_enable_decimal: Optional[bool] = None,
        kompute_native_enable_boolean_compare: Optional[bool] = None,
        kompute_native_enable_boolean_logic: Optional[bool] = None,
        kompute_native_enable_list_query: Optional[bool] = None,
    ):
        self.allow_function_external_writes = bool(allow_function_external_writes)
        self.allow_nested_functions = bool(allow_nested_functions)
        self.max_call_depth = max(1, int(max_call_depth))
        self.require_void_external_writes = bool(require_void_external_writes)
        self.track_instruction_activity = bool(track_instruction_activity)
        backend_raw = DEFAULT_COMPUTE_BACKEND if compute_backend is None else compute_backend
        backend = str(backend_raw).strip().lower()
        if backend not in {"auto", "cpu", "kompute", "kompute-sim"}:
            backend = "auto"
        self.compute_backend = backend
        mode_raw = (
            DEFAULT_KOMPUTE_RUNTIME_MODE
            if kompute_runtime_mode is None
            else kompute_runtime_mode
        )
        mode = str(mode_raw).strip().lower()
        if mode not in {"native", "simulated", "auto"}:
            mode = "native"
        self.kompute_runtime_mode = mode
        self.kompute_warn_on_fallback = bool(
            DEFAULT_KOMPUTE_WARN_ON_FALLBACK
            if kompute_warn_on_fallback is None
            else kompute_warn_on_fallback
        )
        self.kompute_fail_hard = bool(
            DEFAULT_KOMPUTE_FAIL_HARD
            if kompute_fail_hard is None
            else kompute_fail_hard
        )
        self.kompute_keep_vram_state = bool(
            DEFAULT_KOMPUTE_KEEP_VRAM_STATE
            if kompute_keep_vram_state is None
            else kompute_keep_vram_state
        )
        min_stage_count = (
            DEFAULT_KOMPUTE_MIN_NATIVE_STAGE_COUNT
            if kompute_min_native_stage_count is None
            else int(kompute_min_native_stage_count)
        )
        self.kompute_min_native_stage_count = max(0, int(min_stage_count))
        min_stage_share = (
            DEFAULT_KOMPUTE_MIN_NATIVE_STAGE_SHARE
            if kompute_min_native_stage_share is None
            else float(kompute_min_native_stage_share)
        )
        self.kompute_min_native_stage_share = max(0.0, min(1.0, float(min_stage_share)))
        max_unsupported_share = (
            DEFAULT_KOMPUTE_MAX_UNSUPPORTED_SHARE
            if kompute_max_unsupported_share is None
            else float(kompute_max_unsupported_share)
        )
        self.kompute_max_unsupported_share = max(
            0.0,
            min(1.0, float(max_unsupported_share)),
        )
        max_unsupported_count = (
            DEFAULT_KOMPUTE_MAX_UNSUPPORTED_COUNT
            if kompute_max_unsupported_count is None
            else int(kompute_max_unsupported_count)
        )
        self.kompute_max_unsupported_count = int(max(-1, int(max_unsupported_count)))
        self.kompute_force_cpu_on_partial_coverage = bool(
            DEFAULT_KOMPUTE_FORCE_CPU_ON_PARTIAL_COVERAGE
            if kompute_force_cpu_on_partial_coverage is None
            else kompute_force_cpu_on_partial_coverage
        )
        self.kompute_native_enable_decimal = bool(
            DEFAULT_KOMPUTE_NATIVE_ENABLE_DECIMAL
            if kompute_native_enable_decimal is None
            else kompute_native_enable_decimal
        )
        self.kompute_native_enable_boolean_compare = bool(
            DEFAULT_KOMPUTE_NATIVE_ENABLE_BOOLEAN_COMPARE
            if kompute_native_enable_boolean_compare is None
            else kompute_native_enable_boolean_compare
        )
        self.kompute_native_enable_boolean_logic = bool(
            DEFAULT_KOMPUTE_NATIVE_ENABLE_BOOLEAN_LOGIC
            if kompute_native_enable_boolean_logic is None
            else kompute_native_enable_boolean_logic
        )
        self.kompute_native_enable_list_query = bool(
            DEFAULT_KOMPUTE_NATIVE_ENABLE_LIST_QUERY
            if kompute_native_enable_list_query is None
            else kompute_native_enable_list_query
        )
        self._kompute_runtime: Any = None
        self._kompute_runtime_checked = False
        self._kompute_warned_keys: Set[str] = set()
        self._kompute_support_cache: Dict[str, Tuple[bool, str]] = {}
        self._kompute_warning_bucket_limit = 24
        self._last_execution_stats: Dict[str, Any] = self._empty_execution_stats()
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

    @staticmethod
    def _empty_execution_stats() -> Dict[str, Any]:
        return {
            "backend": "cpu",
            "used_kompute": False,
            "gpu_dispatch_count": 0,
            "cpu_fallback_count": 0,
            "cpu_full_sync_count": 0,
            "cpu_partial_sync_count": 0,
            "cpu_no_sync_count": 0,
            "cpu_synced_tensors": 0,
            "final_sync_count": 0,
        }

    def last_execution_stats(self) -> Dict[str, Any]:
        stats = getattr(self, "_last_execution_stats", None)
        if isinstance(stats, dict):
            return dict(stats)
        return self._empty_execution_stats()

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
        self._last_execution_stats = self._empty_execution_stats()
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
        bucket = key_norm.split(":", 1)[0]
        if bucket in {"unsupported", "coverage-policy", "hybrid-stats", "partial-coverage"}:
            if key_norm in self._kompute_warned_keys:
                return
            limit = int(self._kompute_warning_bucket_limit)
            emit_message = False
            emit_suppression = False
            with _GLOBAL_KOMPUTE_WARNING_LOCK:
                seen = int(_GLOBAL_KOMPUTE_WARNING_COUNT_BY_BUCKET.get(bucket, 0))
                if seen >= limit:
                    if bucket not in _GLOBAL_KOMPUTE_WARNING_SUPPRESSED_BUCKETS:
                        _GLOBAL_KOMPUTE_WARNING_SUPPRESSED_BUCKETS.add(bucket)
                        emit_suppression = True
                else:
                    _GLOBAL_KOMPUTE_WARNING_COUNT_BY_BUCKET[bucket] = seen + 1
                    emit_message = True
            self._kompute_warned_keys.add(key_norm)
            if emit_message:
                warnings.warn(
                    message,
                    RuntimeWarning,
                    stacklevel=3,
                )
            elif emit_suppression:
                warnings.warn(
                    (
                        "Kompute fallback warnings for `{bucket}` reached the process-wide "
                        "limit ({limit}); suppressing additional unique messages."
                    ).format(bucket=bucket, limit=limit),
                    RuntimeWarning,
                    stacklevel=3,
                )
            return
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
        runtime_mode = self.kompute_runtime_mode
        if self.compute_backend == "kompute-sim":
            runtime_mode = "simulated"
        self._kompute_runtime = GFSLKomputeRuntime(
            execution_mode=runtime_mode,
            native_enable_decimal=bool(self.kompute_native_enable_decimal),
            native_enable_boolean_compare=bool(
                self.kompute_native_enable_boolean_compare
            ),
            native_enable_boolean_logic=bool(self.kompute_native_enable_boolean_logic),
            native_enable_list_query=bool(self.kompute_native_enable_list_query),
        )
        return self._kompute_runtime

    def _kompute_coverage_reject_reason(
        self,
        *,
        stage_count: int,
        unsupported_count: int,
    ) -> Optional[str]:
        stages = max(0, int(stage_count))
        unsupported = max(0, int(unsupported_count))
        total = stages + unsupported
        native_share = (float(stages) / float(total)) if total > 0 else 0.0
        unsupported_share = (float(unsupported) / float(total)) if total > 0 else 0.0

        if self.kompute_force_cpu_on_partial_coverage and unsupported > 0:
            return (
                "partial coverage rejected by policy "
                "(kompute_force_cpu_on_partial_coverage=true)"
            )
        if stages < int(self.kompute_min_native_stage_count):
            return (
                "native stage count {stage} below minimum {minimum}".format(
                    stage=stages,
                    minimum=int(self.kompute_min_native_stage_count),
                )
            )
        if native_share < float(self.kompute_min_native_stage_share):
            return (
                "native stage share {share:.3f} below minimum {minimum:.3f}".format(
                    share=native_share,
                    minimum=float(self.kompute_min_native_stage_share),
                )
            )
        if (
            int(self.kompute_max_unsupported_count) >= 0
            and unsupported > int(self.kompute_max_unsupported_count)
        ):
            return (
                "unsupported stage count {unsupported} exceeds maximum {maximum}".format(
                    unsupported=unsupported,
                    maximum=int(self.kompute_max_unsupported_count),
                )
            )
        if unsupported_share > float(self.kompute_max_unsupported_share):
            return (
                "unsupported share {share:.3f} exceeds maximum {maximum:.3f}".format(
                    share=unsupported_share,
                    maximum=float(self.kompute_max_unsupported_share),
                )
            )
        return None

    def _attempt_kompute_execution(
        self,
        genome: GFSLGenome,
        *,
        inputs: Optional[Dict[str, Any]],
        track_activity: Optional[bool],
        activity_tick: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        global _GLOBAL_AUTO_KOMPUTE_DISABLE_REASON
        backend = str(self.compute_backend).strip().lower()
        forced_backend = backend in {"kompute", "kompute-sim"}
        if backend == "cpu":
            return None
        if backend == "auto" and _GLOBAL_AUTO_KOMPUTE_DISABLE_REASON:
            return None

        runtime = self._get_kompute_runtime()
        if runtime is None:
            if forced_backend:
                message = (
                    "GFSLExecutor requested `kompute` backend but runtime is unavailable; "
                    "falling back to CPU execution."
                )
                if self.kompute_fail_hard:
                    raise ModuleNotFoundError(message)
                self._warn_kompute_fallback_once("import", message)
            return None

        if not runtime.is_available():
            unavailable_reason = ""
            get_unavailable_reason = getattr(runtime, "native_unavailable_reason", None)
            if callable(get_unavailable_reason):
                try:
                    unavailable_reason = str(get_unavailable_reason()).strip()
                except Exception:
                    unavailable_reason = ""
            if backend == "auto" and unavailable_reason:
                _GLOBAL_AUTO_KOMPUTE_DISABLE_REASON = unavailable_reason
            if forced_backend:
                message = "GFSLExecutor requested `kompute` backend but `kp` is not installed or not usable"
                if unavailable_reason:
                    message = f"{message} ({unavailable_reason})"
                message = f"{message}; falling back to CPU execution."
                if self.kompute_fail_hard:
                    raise ModuleNotFoundError(message)
                self._warn_kompute_fallback_once("kp-unavailable", message)
            return None

        signature = ""
        try:
            signature = str(genome.get_signature())
        except Exception:
            signature = f"genome:{id(genome)}"

        cached = self._kompute_support_cache.get(signature)
        if cached is not None:
            supported_cached, reason_cached = cached
            if not supported_cached:
                if forced_backend:
                    if str(reason_cached).startswith("runtime-error:"):
                        runtime_reason = str(reason_cached).split(":", 1)[1].strip()
                        reason_text = runtime_reason or "runtime-error"
                        message = (
                            "Kompute runtime previously failed for genome signature "
                            f"`{signature[:16]}` ({reason_text}); falling back to CPU execution."
                        )
                    else:
                        message = (
                            "Kompute compatibility pre-check failed for genome signature "
                            f"`{signature[:16]}` ({reason_cached}); falling back to CPU execution."
                        )
                    if self.kompute_fail_hard:
                        raise RuntimeError(message)
                    self._warn_kompute_fallback_once(f"unsupported:{signature}", message)
                return None

        try:
            report = runtime.compatibility_report(
                genome,
                keep_vram_state=bool(self.kompute_keep_vram_state),
            )
        except Exception as exc:
            message = (
                "Kompute compatibility pre-check failed ({error}); "
                "falling back to CPU execution."
            ).format(error=str(exc))
            if self.kompute_fail_hard:
                raise RuntimeError(message) from exc
            self._warn_kompute_fallback_once("compatibility-error", message)
            return None

        unsupported_count = int(getattr(report, "unsupported_count", 0))
        stage_count = int(getattr(report, "stage_count", 0))
        unsupported_by_operation = getattr(report, "unsupported_by_operation", {})
        if not isinstance(unsupported_by_operation, dict):
            unsupported_by_operation = {}
        if stage_count <= 0:
            reason = "stages=0"
            self._kompute_support_cache[signature] = (False, reason)
            if forced_backend:
                message = (
                    "Kompute compatibility pre-check found no GPU-dispatchable stages for "
                    f"genome signature `{signature[:16]}`; falling back to CPU execution."
                )
                if self.kompute_fail_hard:
                    raise RuntimeError(message)
                self._warn_kompute_fallback_once(f"unsupported:{signature}", message)
            return None

        top = sorted(
            (
                (str(name), int(count))
                for name, count in unsupported_by_operation.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:6]
        top_text = ", ".join(f"{name}x{count}" for name, count in top) or "none"
        coverage_reject_reason = self._kompute_coverage_reject_reason(
            stage_count=stage_count,
            unsupported_count=unsupported_count,
        )
        if coverage_reject_reason:
            reason = (
                "coverage-policy={policy}; stages={stage_count}, unsupported={unsupported_count}, top=[{top}]"
            ).format(
                policy=coverage_reject_reason,
                stage_count=stage_count,
                unsupported_count=unsupported_count,
                top=top_text,
            )
            self._kompute_support_cache[signature] = (False, reason)
            if forced_backend:
                message = (
                    "Kompute execution policy rejected native hybrid path for genome signature "
                    f"`{signature[:16]}` ({reason}); falling back to CPU execution."
                )
                if self.kompute_fail_hard:
                    raise RuntimeError(message)
                self._warn_kompute_fallback_once(
                    f"coverage-policy:{signature}",
                    message,
                )
            return None

        if unsupported_count > 0:
            reason = (
                f"stages={stage_count}, unsupported={unsupported_count}, top=[{top_text}]"
            )
            self._kompute_support_cache[signature] = (True, f"partial:{reason}")
            if forced_backend:
                message = (
                    "Kompute compatibility pre-check indicates partial native coverage for "
                    f"genome signature `{signature[:16]}` ({reason}); unsupported stages "
                    "will run on CPU fallback."
                )
                self._warn_kompute_fallback_once("partial-coverage", message)
        else:
            self._kompute_support_cache[signature] = (True, "compatible")

        try:
            outputs = runtime.execute(
                genome,
                inputs=inputs,
                track_activity=track_activity,
                activity_tick=activity_tick,
                keep_vram_state=bool(self.kompute_keep_vram_state),
            )
            resolved_mode = "unknown"
            resolve_mode = getattr(runtime, "_resolve_execute_mode", None)
            if callable(resolve_mode):
                try:
                    resolved_mode = str(resolve_mode()).strip().lower() or "unknown"
                except Exception:
                    resolved_mode = "unknown"
            stats = getattr(runtime, "_last_native_stats", None)
            stats_dict: Dict[str, Any] = dict(stats) if isinstance(stats, dict) else {}
            gpu_dispatch = int(stats_dict.get("gpu_dispatch_count", 0))
            cpu_fallback = int(stats_dict.get("cpu_fallback_count", 0))
            full_sync = int(stats_dict.get("cpu_full_sync_count", 0))
            partial_sync = int(stats_dict.get("cpu_partial_sync_count", 0))
            no_sync = int(stats_dict.get("cpu_no_sync_count", 0))
            synced = int(stats_dict.get("cpu_synced_tensors", 0))
            final_sync = int(stats_dict.get("final_sync_count", 0))
            self._last_execution_stats = {
                "backend": f"kompute:{resolved_mode}",
                "used_kompute": True,
                "gpu_dispatch_count": max(0, gpu_dispatch),
                "cpu_fallback_count": max(0, cpu_fallback),
                "cpu_full_sync_count": max(0, full_sync),
                "cpu_partial_sync_count": max(0, partial_sync),
                "cpu_no_sync_count": max(0, no_sync),
                "cpu_synced_tensors": max(0, synced),
                "final_sync_count": max(0, final_sync),
            }
            if forced_backend and isinstance(stats, dict):
                if cpu_fallback > 0:
                    message = (
                        "Kompute hybrid runtime summary for genome signature "
                        f"`{signature[:16]}`: gpu_dispatch={gpu_dispatch}, "
                        f"cpu_fallback={cpu_fallback}, "
                        f"sync(full/partial/none)={full_sync}/{partial_sync}/{no_sync}, "
                        f"synced_tensors={synced}, final_sync={final_sync}."
                    )
                    self._warn_kompute_fallback_once(
                        f"hybrid-stats:{signature}",
                        message,
                    )
            return outputs
        except TimeoutError:
            raise
        except Exception as exc:
            message = (
                "Kompute execution failed ({error}); falling back to CPU execution."
            ).format(error=str(exc))
            self._kompute_support_cache[signature] = (
                False,
                f"runtime-error:{str(exc)}",
            )
            if backend == "auto":
                err_text = str(exc).strip() or "runtime-error"
                _GLOBAL_AUTO_KOMPUTE_DISABLE_REASON = err_text
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
