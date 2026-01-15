"""GFSL-aligned Evolvo library."""

from .enums import (
    Category,
    ConfigProperty,
    DataType,
    Operation,
    CONTROL_FLOW_OPS,
    BOOLEAN_COMPARE_OPS,
    BOOLEAN_LOGIC_OPS,
    DECIMAL_OPS,
    TENSOR_OPS,
    BINARY_OPS,
    UNARY_OPS,
)
from .slots import (
    ADDRESS_SLOT_COUNT,
    OP_SLOT_COUNT,
    DEFAULT_SLOT_COUNT,
    DEFAULT_PROBABILITY_BRANCHING,
    SLOT_TARGET_CAT,
    SLOT_TARGET_SPEC,
    SLOT_OPERATION,
    SLOT_SOURCE1_CAT,
    SLOT_SOURCE1_SPEC,
    SLOT_SOURCE2_CAT,
    SLOT_SOURCE2_SPEC,
    SLOT_NAMES,
    SLOT_NAME_TO_INDEX,
    SPEC_TYPE_SHIFT,
    SPEC_INDEX_MASK,
    pack_type_index,
    unpack_type_index,
    slot_name,
    resolve_slot_index,
)
from .custom_ops import (
    CustomOperation,
    CustomOperationManager,
    custom_operations,
    register_custom_operation,
    resolve_operation_name,
    infer_source_type,
)
from .values import ValueEnumerations
from .instruction import GFSLInstruction, SlotOption, describe_slot_option
from .weights import OperationWeights
from .validator import SlotValidator
from .builder import GFSLExpressionBuilder
from .genome import GFSLGenome
from .executor import GFSLExecutor
from .evaluator import RealTimeEvaluator
from .model import RecursiveModelBuilder
from .evolver import GFSLEvolver
from .qlearning import GFSLQLearningGuide
from .supervised import (
    GFSLFeatureExtractor,
    GFSLSupervisedDirectionModel,
    GFSLSupervisedGuide,
)
from .demos import example_formula_discovery, example_neural_architecture_search

__all__ = [
    "Category",
    "ConfigProperty",
    "DataType",
    "Operation",
    "CONTROL_FLOW_OPS",
    "BOOLEAN_COMPARE_OPS",
    "BOOLEAN_LOGIC_OPS",
    "DECIMAL_OPS",
    "TENSOR_OPS",
    "BINARY_OPS",
    "UNARY_OPS",
    "ADDRESS_SLOT_COUNT",
    "OP_SLOT_COUNT",
    "DEFAULT_SLOT_COUNT",
    "DEFAULT_PROBABILITY_BRANCHING",
    "SLOT_TARGET_CAT",
    "SLOT_TARGET_SPEC",
    "SLOT_OPERATION",
    "SLOT_SOURCE1_CAT",
    "SLOT_SOURCE1_SPEC",
    "SLOT_SOURCE2_CAT",
    "SLOT_SOURCE2_SPEC",
    "SLOT_NAMES",
    "SLOT_NAME_TO_INDEX",
    "SPEC_TYPE_SHIFT",
    "SPEC_INDEX_MASK",
    "pack_type_index",
    "unpack_type_index",
    "slot_name",
    "resolve_slot_index",
    "CustomOperation",
    "CustomOperationManager",
    "custom_operations",
    "register_custom_operation",
    "resolve_operation_name",
    "infer_source_type",
    "ValueEnumerations",
    "OperationWeights",
    "GFSLInstruction",
    "SlotOption",
    "describe_slot_option",
    "SlotValidator",
    "GFSLExpressionBuilder",
    "GFSLGenome",
    "GFSLExecutor",
    "RealTimeEvaluator",
    "RecursiveModelBuilder",
    "GFSLEvolver",
    "GFSLQLearningGuide",
    "GFSLFeatureExtractor",
    "GFSLSupervisedDirectionModel",
    "GFSLSupervisedGuide",
    "example_formula_discovery",
    "example_neural_architecture_search",
]
