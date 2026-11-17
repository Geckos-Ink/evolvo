"""GFSL-aligned Evolvo library.

Public names are resolved lazily (PEP 562): ``from evolvo import GFSLGenome`` imports only the
submodule that defines it. Pure-Python subpackages such as ``evolvo.arch`` therefore import
without NumPy, PyTorch or Kompute; the NumPy-backed GFSL runtime loads on first use.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

_LAZY: Dict[str, str] = {
    "Category": "enums",
    "ConfigProperty": "enums",
    "DataType": "enums",
    "Operation": "enums",
    "CONTROL_FLOW_OPS": "enums",
    "BOOLEAN_COMPARE_OPS": "enums",
    "BOOLEAN_LOGIC_OPS": "enums",
    "DECIMAL_OPS": "enums",
    "LIST_TARGET_OPS": "enums",
    "LIST_VALUE_OPS": "enums",
    "LIST_QUERY_OPS": "enums",
    "TENSOR_OPS": "enums",
    "BINARY_OPS": "enums",
    "UNARY_OPS": "enums",
    "ADDRESS_SLOT_COUNT": "slots",
    "OP_SLOT_COUNT": "slots",
    "DEFAULT_SLOT_COUNT": "slots",
    "DEFAULT_PROBABILITY_BRANCHING": "slots",
    "SLOT_TARGET_CAT": "slots",
    "SLOT_TARGET_SPEC": "slots",
    "SLOT_OPERATION": "slots",
    "SLOT_SOURCE1_CAT": "slots",
    "SLOT_SOURCE1_SPEC": "slots",
    "SLOT_SOURCE2_CAT": "slots",
    "SLOT_SOURCE2_SPEC": "slots",
    "SLOT_NAMES": "slots",
    "SLOT_NAME_TO_INDEX": "slots",
    "SPEC_TYPE_SHIFT": "slots",
    "SPEC_INDEX_MASK": "slots",
    "pack_type_index": "slots",
    "unpack_type_index": "slots",
    "slot_name": "slots",
    "resolve_slot_index": "slots",
    "CustomOperation": "custom_ops",
    "CustomOperationManager": "custom_ops",
    "custom_operations": "custom_ops",
    "register_custom_operation": "custom_ops",
    "resolve_operation_name": "custom_ops",
    "infer_source_type": "custom_ops",
    "ValueEnumerations": "values",
    "OperationWeights": "weights",
    "GFSLInstruction": "instruction",
    "SlotOption": "instruction",
    "describe_slot_option": "instruction",
    "SlotValidator": "validator",
    "GFSLExpressionBuilder": "builder",
    "InstructionActivity": "genome",
    "GFSLGenome": "genome",
    "FunctionDefinition": "executor",
    "GFSLExecutor": "executor",
    "RealTimeEvaluator": "evaluator",
    "RecursiveModelBuilder": "model",
    "GFSLEvolver": "evolver",
    "GFSLQLearningGuide": "qlearning",
    "resolve_torch_accelerator": "supervised",
    "GFSLFeatureExtractor": "supervised",
    "GFSLSupervisedDirectionModel": "supervised",
    "GFSLSupervisedGuide": "supervised",
    "KomputeTypeSpec": "kompute",
    "KomputeTypeOverride": "kompute",
    "KomputeKernelBinding": "kompute",
    "KomputeBufferRef": "kompute",
    "KomputeKernelStage": "kompute",
    "KomputeUnsupportedInstruction": "kompute",
    "KomputeExecutionPlan": "kompute",
    "KomputeCompatibilityReport": "kompute",
    "KomputeInstructionRegistry": "kompute",
    "GFSLKomputePlanner": "kompute",
    "GFSLKomputeRuntime": "kompute",
    "DEFAULT_COMPUTE_BACKEND": "settings",
    "DEFAULT_KOMPUTE_RUNTIME_MODE": "settings",
    "DEFAULT_KOMPUTE_WARN_ON_FALLBACK": "settings",
    "DEFAULT_KOMPUTE_FAIL_HARD": "settings",
    "DEFAULT_KOMPUTE_KEEP_VRAM_STATE": "settings",
    "DEFAULT_KOMPUTE_MIN_NATIVE_STAGE_COUNT": "settings",
    "DEFAULT_KOMPUTE_MIN_NATIVE_STAGE_SHARE": "settings",
    "DEFAULT_KOMPUTE_MAX_UNSUPPORTED_SHARE": "settings",
    "DEFAULT_KOMPUTE_MAX_UNSUPPORTED_COUNT": "settings",
    "DEFAULT_KOMPUTE_FORCE_CPU_ON_PARTIAL_COVERAGE": "settings",
    "DEFAULT_KOMPUTE_NATIVE_ENABLE_DECIMAL": "settings",
    "DEFAULT_KOMPUTE_NATIVE_ENABLE_BOOLEAN_COMPARE": "settings",
    "DEFAULT_KOMPUTE_NATIVE_ENABLE_BOOLEAN_LOGIC": "settings",
    "DEFAULT_KOMPUTE_NATIVE_ENABLE_LIST_QUERY": "settings",
    "DEFAULT_KOMPUTE_PLAN_CACHE_ENABLE": "settings",
    "DEFAULT_KOMPUTE_PLAN_CACHE_MAX_ENTRIES": "settings",
    "DEFAULT_KOMPUTE_PLAN_DISK_CACHE_ENABLE": "settings",
    "DEFAULT_KOMPUTE_PLAN_DISK_CACHE_DIR": "settings",
    "DEFAULT_KOMPUTE_PLAN_DISK_CACHE_MAX_FILES": "settings",
    "example_formula_discovery": "demos",
    "example_neural_architecture_search": "demos",
}

__all__: List[str] = list(_LAZY)


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_LAZY))
