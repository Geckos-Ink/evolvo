"""GFSL instruction representation and slot display helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .custom_ops import infer_source_type, resolve_operation_name
from .enums import Category, ConfigProperty, DataType, Operation
from .slots import (
    DEFAULT_SLOT_COUNT,
    SLOT_OPERATION,
    SLOT_SOURCE1_CAT,
    SLOT_SOURCE1_SPEC,
    SLOT_SOURCE2_CAT,
    SLOT_SOURCE2_SPEC,
    SLOT_TARGET_CAT,
    SLOT_TARGET_SPEC,
    pack_type_index,
    unpack_type_index,
)
from .values import ValueEnumerations


@dataclass
class GFSLInstruction:
    """
    Fixed-slot instruction representation (compressed pointers).
    Each slot is an integer index.
    """

    slots: List[int]

    def __init__(self, slots: Optional[List[int]] = None, slot_count: Optional[int] = None):
        if slot_count is None:
            slot_count = len(slots) if slots is not None else DEFAULT_SLOT_COUNT
        if slots is None:
            self.slots = [0] * slot_count
        else:
            assert len(slots) == slot_count, f"Instruction must have exactly {slot_count} slots"
            self.slots = slots

    def slot_value(self, index: int) -> int:
        """Return slot value or 0 when index is out of range."""
        if index < 0 or index >= len(self.slots):
            return 0
        return self.slots[index]

    def pad_to(self, slot_count: int) -> None:
        """Pad slots with zeros to reach the requested slot count."""
        if slot_count <= len(self.slots):
            return
        self.slots.extend([0] * (slot_count - len(self.slots)))

    @property
    def target_cat(self) -> int:
        return self.slot_value(SLOT_TARGET_CAT)

    @property
    def target_spec(self) -> int:
        return self.slot_value(SLOT_TARGET_SPEC)

    @property
    def target_type(self) -> int:
        if self.target_cat in (Category.VARIABLE, Category.CONSTANT):
            dtype, _ = unpack_type_index(self.target_spec)
            return dtype
        return int(DataType.NONE)

    @property
    def target_index(self) -> int:
        if self.target_cat in (Category.VARIABLE, Category.CONSTANT):
            _, idx = unpack_type_index(self.target_spec)
            return idx
        return 0

    @property
    def operation(self) -> int:
        return self.slot_value(SLOT_OPERATION)

    @property
    def source1_cat(self) -> int:
        return self.slot_value(SLOT_SOURCE1_CAT)

    @property
    def source1_spec(self) -> int:
        return self.slot_value(SLOT_SOURCE1_SPEC)

    @property
    def source1_type(self) -> int:
        if self.source1_cat in (Category.VARIABLE, Category.CONSTANT):
            dtype, _ = unpack_type_index(self.source1_spec)
            return dtype
        if self.source1_cat == Category.VALUE:
            return infer_source_type(self.operation, 1)
        return int(DataType.NONE)

    @property
    def source1_value(self) -> int:
        if self.source1_cat in (Category.VARIABLE, Category.CONSTANT):
            _, idx = unpack_type_index(self.source1_spec)
            return idx
        if self.source1_cat in (Category.VALUE, Category.CONFIG):
            return self.source1_spec
        return 0

    @property
    def source2_cat(self) -> int:
        return self.slot_value(SLOT_SOURCE2_CAT)

    @property
    def source2_spec(self) -> int:
        return self.slot_value(SLOT_SOURCE2_SPEC)

    @property
    def source2_type(self) -> int:
        if self.source2_cat in (Category.VARIABLE, Category.CONSTANT):
            dtype, _ = unpack_type_index(self.source2_spec)
            return dtype
        if self.source2_cat == Category.VALUE:
            return infer_source_type(self.operation, 2)
        return int(DataType.NONE)

    @property
    def source2_value(self) -> int:
        if self.source2_cat in (Category.VARIABLE, Category.CONSTANT):
            _, idx = unpack_type_index(self.source2_spec)
            return idx
        if self.source2_cat in (Category.VALUE, Category.CONFIG):
            return self.source2_spec
        return 0

    @property
    def slot_count(self) -> int:
        return len(self.slots)

    def get_signature(self) -> str:
        """Get unique signature for this instruction."""
        return "|".join(str(s) for s in self.slots)

    def copy(self) -> "GFSLInstruction":
        """Create deep copy."""
        return GFSLInstruction(self.slots.copy(), slot_count=len(self.slots))


@dataclass(frozen=True)
class SlotOption:
    """Represents a readable option for a single slot."""

    value: int
    label: str
    probability: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        data = {"value": self.value, "label": self.label}
        if self.probability is not None:
            data["probability"] = self.probability
        return data


def describe_slot_option(instruction: GFSLInstruction, slot_index: int, value: int) -> str:
    """Return a readable label for a slot value given the current instruction."""
    if slot_index in (SLOT_TARGET_CAT, SLOT_SOURCE1_CAT, SLOT_SOURCE2_CAT):
        try:
            return Category(value).name
        except ValueError:
            return f"CATEGORY[{value}]"

    if slot_index == SLOT_OPERATION:
        return resolve_operation_name(int(value))

    if slot_index in (SLOT_TARGET_SPEC, SLOT_SOURCE1_SPEC, SLOT_SOURCE2_SPEC):
        if slot_index == SLOT_TARGET_SPEC:
            cat_value = instruction.slot_value(SLOT_TARGET_CAT)
        elif slot_index == SLOT_SOURCE1_SPEC:
            cat_value = instruction.slot_value(SLOT_SOURCE1_CAT)
        else:
            cat_value = instruction.slot_value(SLOT_SOURCE2_CAT)

        try:
            cat = Category(cat_value)
        except ValueError:
            return f"SPEC[{value}]"

        if cat == Category.NONE:
            return "NONE"
        if cat in (Category.VARIABLE, Category.CONSTANT):
            dtype, idx = unpack_type_index(value)
            try:
                dtype_enum = DataType(dtype)
                prefix = dtype_enum.name[0].lower()
            except ValueError:
                prefix = f"type{dtype}"
            symbol = "$" if cat == Category.VARIABLE else "#"
            return f"{prefix}{symbol}{idx}"
        if cat == Category.CONFIG:
            try:
                return ConfigProperty(value).name
            except ValueError:
                return f"CONFIG[{value}]"
        if cat == Category.VALUE:
            op = instruction.slot_value(SLOT_OPERATION)
            prop = None
            if slot_index == SLOT_SOURCE2_SPEC and op == Operation.SET:
                if instruction.slot_value(SLOT_SOURCE1_CAT) == Category.CONFIG:
                    try:
                        prop = ConfigProperty(instruction.slot_value(SLOT_SOURCE1_SPEC))
                    except ValueError:
                        prop = None
            enum = ValueEnumerations.get_enumeration((op, prop))
            if value < len(enum):
                return str(enum[value])
            return f"VAL[{value}]"

    return str(value)


__all__ = [
    "GFSLInstruction",
    "SlotOption",
    "describe_slot_option",
]
