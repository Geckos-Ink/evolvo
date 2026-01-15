"""Slot layout constants and slot helper utilities."""

from typing import Tuple, Union


ADDRESS_SLOT_COUNT = 2
OP_SLOT_COUNT = 1
DEFAULT_SLOT_COUNT = ADDRESS_SLOT_COUNT * 3 + OP_SLOT_COUNT
DEFAULT_PROBABILITY_BRANCHING = 128

SLOT_TARGET_CAT = 0
SLOT_TARGET_SPEC = 1
SLOT_OPERATION = 2
SLOT_SOURCE1_CAT = 3
SLOT_SOURCE1_SPEC = 4
SLOT_SOURCE2_CAT = 5
SLOT_SOURCE2_SPEC = 6

SLOT_NAMES = [
    "target_cat",
    "target_spec",
    "op",
    "source1_cat",
    "source1_spec",
    "source2_cat",
    "source2_spec",
]

SLOT_NAME_TO_INDEX = {
    "target_cat": SLOT_TARGET_CAT,
    "target_spec": SLOT_TARGET_SPEC,
    "op": SLOT_OPERATION,
    "operation": SLOT_OPERATION,
    "source1_cat": SLOT_SOURCE1_CAT,
    "source1_spec": SLOT_SOURCE1_SPEC,
    "source2_cat": SLOT_SOURCE2_CAT,
    "source2_spec": SLOT_SOURCE2_SPEC,
}

SPEC_TYPE_SHIFT = 16
SPEC_INDEX_MASK = (1 << SPEC_TYPE_SHIFT) - 1


def pack_type_index(dtype: int, index: int) -> int:
    """Pack a type and index into a single slot value."""
    return (int(dtype) << SPEC_TYPE_SHIFT) | (int(index) & SPEC_INDEX_MASK)


def unpack_type_index(packed: int) -> Tuple[int, int]:
    """Unpack a slot value into (type, index)."""
    return int(packed) >> SPEC_TYPE_SHIFT, int(packed) & SPEC_INDEX_MASK


def slot_name(slot_index: int) -> str:
    """Return a readable slot name for an index."""
    if 0 <= slot_index < len(SLOT_NAMES):
        return SLOT_NAMES[slot_index]
    return f"slot_{slot_index}"


def resolve_slot_index(slot: Union[int, str]) -> int:
    """Resolve slot index from a numeric id or a slot name."""
    if isinstance(slot, int):
        return slot
    if isinstance(slot, str):
        key = slot.strip().lower()
        if key in SLOT_NAME_TO_INDEX:
            return SLOT_NAME_TO_INDEX[key]
        raise ValueError(f"Unknown slot name '{slot}'.")
    raise TypeError("Slot identifier must be an int or str.")


__all__ = [
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
]
