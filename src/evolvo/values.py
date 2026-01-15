"""Context-dependent enumerations for VALUE slots."""

from typing import List, Optional, Tuple

from .custom_ops import custom_operations
from .enums import ConfigProperty, Operation


class ValueEnumerations:
    """Context-specific value enumerations."""

    MATH_CONSTANTS = [0.0, 1.0, -1.0, 2.0, 0.5, 3.14159, 2.71828, 10.0]

    CHANNELS = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    KERNELS = [1, 3, 5, 7, 9, 11]

    STRIDES = [1, 2, 3, 4]
    PADDINGS = [0, 1, 2, 3, 4, 5]

    PROBABILITIES = [
        0.0,
        0.1,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.75,
        0.8,
        0.9,
        1.0,
    ]

    LOOP_COUNTS = [1, 2, 3, 5, 10, 20, 50, 100]

    @staticmethod
    def get_enumeration(context: Tuple[Operation, Optional[ConfigProperty]]) -> List[float]:
        """Get enumeration based on operation context."""
        op, prop = context

        custom_def = custom_operations.get(int(op))
        if custom_def and custom_def.value_enumeration is not None:
            return list(custom_def.value_enumeration)

        if op == Operation.SET:
            if prop == ConfigProperty.CHANNELS:
                return ValueEnumerations.CHANNELS
            if prop == ConfigProperty.KERNEL:
                return ValueEnumerations.KERNELS
            if prop in (ConfigProperty.STRIDE, ConfigProperty.PADDING):
                return ValueEnumerations.STRIDES
            if prop == ConfigProperty.RATE:
                return ValueEnumerations.PROBABILITIES

        if op in (Operation.ADD, Operation.SUB, Operation.MUL, Operation.DIV):
            return ValueEnumerations.MATH_CONSTANTS
        if op in (Operation.IF, Operation.WHILE):
            return ValueEnumerations.LOOP_COUNTS
        if op == Operation.DROPOUT:
            return ValueEnumerations.PROBABILITIES

        return ValueEnumerations.MATH_CONSTANTS


__all__ = ["ValueEnumerations"]
