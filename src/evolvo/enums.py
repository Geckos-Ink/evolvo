"""Enumerations and operation groupings for GFSL."""

from enum import IntEnum
from typing import List, Set


class Category(IntEnum):
    """Target/Source categories as integer indices."""

    NONE = 0
    VARIABLE = 1  # $
    CONSTANT = 2  # #
    VALUE = 3  # inline enumerated value
    CONFIG = 4  # &
    LIST = 5  # ! (typed list, e.g. d!0)
    LIST_CONSTANT = 6  # !# (typed constant list, e.g. d!#0)
    FUNCTION = 7  # & (typed function reference, e.g. d&0)


class DataType(IntEnum):
    """Data types as integer indices."""

    NONE = 0
    BOOLEAN = 1  # b
    DECIMAL = 2  # d
    TENSOR = 3  # t


class Operation(IntEnum):
    """Operations as integer indices."""

    # Control Flow (target = NONE)
    IF = 0
    WHILE = 1
    END = 2
    SET = 3
    RESULT = 4
    FUNC = 5
    CALL = 6

    # Boolean operations (target = boolean)
    GT = 10
    LT = 11
    EQ = 12
    GTE = 13
    LTE = 14
    NEQ = 15
    AND = 16
    OR = 17
    NOT = 18

    # Decimal operations (target = decimal)
    ADD = 30
    SUB = 31
    MUL = 32
    DIV = 33
    POW = 34
    SQRT = 35
    ABS = 36
    SIN = 37
    COS = 38
    EXP = 39
    LOG = 40
    MOD = 41

    # List operations (typed queues/stacks)
    PREPEND = 80
    APPEND = 81
    CLONE = 82
    FIFO = 83
    FILO = 84
    LISTCOUNT = 85
    LISTHASITEMS = 86

    # Tensor operations (target = tensor)
    CONV = 60
    LINEAR = 61
    RELU = 62
    POOL = 63
    NORM = 64
    DROPOUT = 65
    SOFTMAX = 66
    RESHAPE = 67
    CONCAT = 68


class ConfigProperty(IntEnum):
    """Configuration properties for neural operations."""

    CHANNELS = 0
    KERNEL = 1
    STRIDE = 2
    PADDING = 3
    UNITS = 4
    RATE = 5
    MOMENTUM = 6
    EPSILON = 7


CONTROL_FLOW_OPS: List[Operation] = [
    Operation.IF,
    Operation.WHILE,
    Operation.END,
    Operation.SET,
    Operation.RESULT,
    Operation.FUNC,
]

BOOLEAN_COMPARE_OPS: List[Operation] = [
    Operation.GT,
    Operation.LT,
    Operation.EQ,
    Operation.GTE,
    Operation.LTE,
    Operation.NEQ,
]

BOOLEAN_LOGIC_OPS: List[Operation] = [
    Operation.AND,
    Operation.OR,
    Operation.NOT,
]

DECIMAL_OPS: List[Operation] = [
    Operation.ADD,
    Operation.SUB,
    Operation.MUL,
    Operation.DIV,
    Operation.POW,
    Operation.SQRT,
    Operation.ABS,
    Operation.SIN,
    Operation.COS,
    Operation.EXP,
    Operation.LOG,
    Operation.MOD,
]

TENSOR_OPS: List[Operation] = [
    Operation.CONV,
    Operation.LINEAR,
    Operation.RELU,
    Operation.POOL,
    Operation.NORM,
    Operation.DROPOUT,
    Operation.SOFTMAX,
    Operation.RESHAPE,
    Operation.CONCAT,
]

LIST_TARGET_OPS: List[Operation] = [
    Operation.PREPEND,
    Operation.APPEND,
    Operation.CLONE,
]

LIST_VALUE_OPS: List[Operation] = [
    Operation.FIFO,
    Operation.FILO,
]

LIST_QUERY_OPS: List[Operation] = [
    Operation.LISTCOUNT,
    Operation.LISTHASITEMS,
]

BINARY_OPS: Set[Operation] = {
    Operation.ADD,
    Operation.SUB,
    Operation.MUL,
    Operation.DIV,
    Operation.POW,
    Operation.MOD,
    Operation.AND,
    Operation.OR,
    Operation.GT,
    Operation.LT,
    Operation.EQ,
    Operation.GTE,
    Operation.LTE,
    Operation.NEQ,
    Operation.SET,
}

UNARY_OPS: Set[Operation] = {
    Operation.NOT,
    Operation.SQRT,
    Operation.ABS,
    Operation.SIN,
    Operation.COS,
    Operation.EXP,
    Operation.LOG,
    Operation.CALL,
    Operation.PREPEND,
    Operation.APPEND,
    Operation.CLONE,
    Operation.FIFO,
    Operation.FILO,
    Operation.LISTCOUNT,
    Operation.LISTHASITEMS,
}


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
    "LIST_TARGET_OPS",
    "LIST_VALUE_OPS",
    "LIST_QUERY_OPS",
    "BINARY_OPS",
    "UNARY_OPS",
]
