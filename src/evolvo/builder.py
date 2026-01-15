"""Slot-wise expression builder for GFSL instructions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

from .enums import Category, ConfigProperty, DataType, Operation
from .instruction import GFSLInstruction, SlotOption, describe_slot_option
from .slots import (
    DEFAULT_PROBABILITY_BRANCHING,
    DEFAULT_SLOT_COUNT,
    SLOT_SOURCE1_CAT,
    SLOT_SOURCE1_SPEC,
    SLOT_SOURCE2_CAT,
    SLOT_SOURCE2_SPEC,
    SLOT_TARGET_SPEC,
    SLOT_TARGET_CAT,
    SLOT_OPERATION,
    pack_type_index,
    resolve_slot_index,
    slot_name,
)
from .values import ValueEnumerations

if TYPE_CHECKING:
    from .genome import GFSLGenome


class GFSLExpressionBuilder:
    """
    Slot-wise instruction builder that exposes valid next options.
    """

    def __init__(self, genome: "GFSLGenome"):
        self.genome = genome
        self.validator = genome.validator
        if self.validator.slot_count == 0:
            self.validator.slot_count = DEFAULT_SLOT_COUNT
        self.instruction = GFSLInstruction(slot_count=self.validator.slot_count)
        self.cursor = 0
        self._consequents: List[Tuple[Dict[int, int], Any]] = []

    def next_slot_index(self) -> int:
        return self.cursor

    def next_slot_name(self) -> str:
        return slot_name(self.cursor)

    def next_options(
        self,
        slot: Optional[Union[int, str]] = None,
        *,
        viable_only: bool = False,
        include_probabilities: bool = False,
        max_branching: int = DEFAULT_PROBABILITY_BRANCHING,
    ) -> List[SlotOption]:
        slot_index = self.cursor if slot is None else resolve_slot_index(slot)
        if slot_index > self.cursor:
            raise ValueError(
                f"Slot {slot_name(slot_index)} is ahead of the current cursor "
                f"{slot_name(self.cursor)}."
            )
        valid = self.validator.get_valid_options(self.instruction, slot_index)
        options = valid
        probabilities: Dict[int, float] = {}

        if include_probabilities or viable_only:
            probabilities = self.validator.option_success_probabilities(
                self.instruction,
                slot_index,
                max_branching,
            )
            if viable_only:
                options = [opt for opt in valid if probabilities.get(opt, 0.0) > 0.0] or valid

        return [
            SlotOption(
                value=int(opt),
                label=describe_slot_option(self.instruction, slot_index, int(opt)),
                probability=probabilities.get(opt) if include_probabilities else None,
            )
            for opt in options
        ]

    def choose(
        self,
        value: Union[int, SlotOption, Category, DataType, Operation, ConfigProperty, Tuple[Any, Any]],
        *,
        slot: Optional[Union[int, str]] = None,
        require_viable: bool = False,
        max_branching: int = DEFAULT_PROBABILITY_BRANCHING,
    ) -> "GFSLExpressionBuilder":
        slot_index = self.cursor if slot is None else resolve_slot_index(slot)
        if slot_index != self.cursor:
            raise ValueError(
                f"Expected to set {slot_name(self.cursor)} next, got {slot_name(slot_index)}."
            )
        value_int = self._coerce_slot_value(slot_index, value)
        valid = self.validator.get_valid_options(self.instruction, slot_index)
        if value_int not in valid:
            raise ValueError(f"Invalid option {value_int} for {slot_name(slot_index)}.")
        if require_viable:
            viable, _ = self.validator.viable_options(
                self.instruction,
                slot_index,
                max_branching,
            )
            if viable and value_int not in viable:
                raise ValueError(
                    f"Option {value_int} for {slot_name(slot_index)} leads to a dead end."
                )
        self.instruction.slots[slot_index] = value_int
        self.cursor += 1
        return self

    def auto(self, *, max_branching: int = DEFAULT_PROBABILITY_BRANCHING) -> "GFSLExpressionBuilder":
        slot_index = self.cursor
        value = self.validator.choose_option(self.instruction, slot_index, max_branching)
        self.instruction.slots[slot_index] = value
        self.cursor += 1
        return self

    def auto_fill(self, *, max_branching: int = DEFAULT_PROBABILITY_BRANCHING) -> "GFSLExpressionBuilder":
        while self.cursor < self.validator.slot_count:
            self.auto(max_branching=max_branching)
        return self

    def target(
        self,
        category: Category,
        dtype: Optional[Union[DataType, int]] = None,
        index: Optional[int] = None,
    ) -> "GFSLExpressionBuilder":
        cat_value = Category(category)
        self.choose(int(cat_value))
        if cat_value == Category.NONE:
            return self.choose(0)
        if dtype is None or index is None:
            raise ValueError("dtype and index are required for variable/constant targets.")
        spec = pack_type_index(dtype, index)
        return self.choose(spec)

    def target_var(self, dtype: Union[DataType, int], index: int) -> "GFSLExpressionBuilder":
        return self.target(Category.VARIABLE, dtype=dtype, index=index)

    def target_const(self, dtype: Union[DataType, int], index: int) -> "GFSLExpressionBuilder":
        return self.target(Category.CONSTANT, dtype=dtype, index=index)

    def target_none(self) -> "GFSLExpressionBuilder":
        return self.target(Category.NONE)

    def op(self, op_code: Union[Operation, int]) -> "GFSLExpressionBuilder":
        return self.choose(int(op_code))

    def source1(self, category: Category, spec: int) -> "GFSLExpressionBuilder":
        return self._set_pointer(SLOT_SOURCE1_CAT, SLOT_SOURCE1_SPEC, category, spec)

    def source2(self, category: Category, spec: int) -> "GFSLExpressionBuilder":
        return self._set_pointer(SLOT_SOURCE2_CAT, SLOT_SOURCE2_SPEC, category, spec)

    def source1_var(self, dtype: Union[DataType, int], index: int) -> "GFSLExpressionBuilder":
        return self.source1(Category.VARIABLE, pack_type_index(dtype, index))

    def source1_const(self, dtype: Union[DataType, int], index: int) -> "GFSLExpressionBuilder":
        return self.source1(Category.CONSTANT, pack_type_index(dtype, index))

    def source1_value_index(self, index: int) -> "GFSLExpressionBuilder":
        return self.source1(Category.VALUE, int(index))

    def source1_value(self, value: float) -> "GFSLExpressionBuilder":
        return self.source1(Category.VALUE, self._value_index_for_source(1, value))

    def source1_config(self, prop: Union[ConfigProperty, int]) -> "GFSLExpressionBuilder":
        return self.source1(Category.CONFIG, int(prop))

    def source1_none(self) -> "GFSLExpressionBuilder":
        return self.source1(Category.NONE, 0)

    def source2_var(self, dtype: Union[DataType, int], index: int) -> "GFSLExpressionBuilder":
        return self.source2(Category.VARIABLE, pack_type_index(dtype, index))

    def source2_const(self, dtype: Union[DataType, int], index: int) -> "GFSLExpressionBuilder":
        return self.source2(Category.CONSTANT, pack_type_index(dtype, index))

    def source2_value_index(self, index: int) -> "GFSLExpressionBuilder":
        return self.source2(Category.VALUE, int(index))

    def source2_value(self, value: float) -> "GFSLExpressionBuilder":
        return self.source2(Category.VALUE, self._value_index_for_source(2, value))

    def source2_config(self, prop: Union[ConfigProperty, int]) -> "GFSLExpressionBuilder":
        return self.source2(Category.CONFIG, int(prop))

    def source2_none(self) -> "GFSLExpressionBuilder":
        return self.source2(Category.NONE, 0)

    def then(self, consequent: Any) -> "GFSLExpressionBuilder":
        return self.then_if({}, consequent)

    def then_if(self, conditions: Any, consequent: Any) -> "GFSLExpressionBuilder":
        normalized = self._normalize_conditions(conditions)
        self._consequents.append((normalized, consequent))
        return self

    def build(
        self,
        *,
        auto_fill: bool = False,
        max_branching: int = DEFAULT_PROBABILITY_BRANCHING,
    ) -> GFSLInstruction:
        if auto_fill:
            self.auto_fill(max_branching=max_branching)
        if self.cursor < self.validator.slot_count:
            raise ValueError(
                f"Expression incomplete; next slot is {slot_name(self.cursor)}."
            )
        return self.instruction.copy()

    def commit(
        self,
        *,
        auto_fill: bool = False,
        max_branching: int = DEFAULT_PROBABILITY_BRANCHING,
    ) -> GFSLInstruction:
        instruction = self.build(auto_fill=auto_fill, max_branching=max_branching)
        if not self.genome.add_instruction(instruction):
            raise ValueError("Instruction failed validation when added to the genome.")
        for conditions, consequent in self._consequents:
            if self._matches_conditions(instruction, conditions):
                self._apply_consequent(consequent, instruction)
        return instruction

    def reset(self) -> "GFSLExpressionBuilder":
        self.instruction = GFSLInstruction(slot_count=self.validator.slot_count)
        self.cursor = 0
        self._consequents = []
        return self

    def _set_pointer(
        self,
        cat_slot: int,
        spec_slot: int,
        category: Category,
        spec: int,
    ) -> "GFSLExpressionBuilder":
        self.choose(int(category), slot=cat_slot)
        return self.choose(int(spec), slot=spec_slot)

    def _value_index_for_source(self, source_num: int, value: float) -> int:
        op = self.instruction.slot_value(SLOT_OPERATION)
        prop = None
        if source_num == 2 and op == Operation.SET:
            if self.instruction.slot_value(SLOT_SOURCE1_CAT) == Category.CONFIG:
                try:
                    prop = ConfigProperty(self.instruction.slot_value(SLOT_SOURCE1_SPEC))
                except ValueError:
                    prop = None
        enum = ValueEnumerations.get_enumeration((op, prop))
        if not enum:
            raise ValueError("No value enumeration available for the current context.")
        value_float = float(value)
        if value_float not in enum:
            raise ValueError(f"Value {value} not in the current enumeration.")
        return enum.index(value_float)

    def _coerce_slot_value(self, slot_index: int, value: Any) -> int:
        if isinstance(value, SlotOption):
            return int(value.value)
        if isinstance(value, tuple) and slot_index in (
            SLOT_TARGET_SPEC,
            SLOT_SOURCE1_SPEC,
            SLOT_SOURCE2_SPEC,
        ):
            if len(value) != 2:
                raise ValueError("Spec tuple must be (dtype, index).")
            dtype, index = value
            return pack_type_index(dtype, index)
        if isinstance(value, (Category, DataType, Operation, ConfigProperty)):
            return int(value)
        return int(value)

    def _normalize_conditions(self, conditions: Any) -> Dict[int, int]:
        if not conditions:
            return {}
        if isinstance(conditions, dict):
            items = conditions.items()
        else:
            items = conditions
        normalized: Dict[int, int] = {}
        for slot, expected in items:
            slot_index = resolve_slot_index(slot)
            normalized[slot_index] = self._coerce_slot_value(slot_index, expected)
        return normalized

    def _matches_conditions(
        self, instruction: GFSLInstruction, conditions: Dict[int, int]
    ) -> bool:
        for slot_index, expected in conditions.items():
            if instruction.slot_value(slot_index) != expected:
                return False
        return True

    def _apply_consequent(self, consequent: Any, instruction: GFSLInstruction) -> None:
        if consequent is None:
            return
        if isinstance(consequent, GFSLInstruction):
            if not self.genome.add_instruction(consequent):
                raise ValueError("Consequent instruction failed validation.")
            return
        if isinstance(consequent, GFSLExpressionBuilder):
            if consequent.genome is not self.genome:
                raise ValueError("Consequent builder belongs to a different genome.")
            consequent.commit()
            return
        if callable(consequent):
            result = consequent(self.genome, instruction)
            if result is None:
                return
            if isinstance(result, (list, tuple)):
                for item in result:
                    self._apply_consequent(item, instruction)
                return
            self._apply_consequent(result, instruction)
            return
        if isinstance(consequent, (list, tuple)):
            for item in consequent:
                self._apply_consequent(item, instruction)
            return
        raise ValueError("Unsupported consequent type.")


__all__ = ["GFSLExpressionBuilder"]
