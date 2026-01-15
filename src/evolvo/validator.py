"""Cascading slot validator for GFSL instructions."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .custom_ops import custom_operations, infer_source_type
from .enums import (
    Category,
    ConfigProperty,
    DataType,
    Operation,
    BINARY_OPS,
    CONTROL_FLOW_OPS,
    DECIMAL_OPS,
    TENSOR_OPS,
    BOOLEAN_COMPARE_OPS,
    BOOLEAN_LOGIC_OPS,
)
from .instruction import GFSLInstruction
from .slots import (
    DEFAULT_PROBABILITY_BRANCHING,
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


class SlotValidator:
    """
    Enforces cascading validity - each slot's valid options
    depend entirely on all previous slots.
    """

    def __init__(self, slot_count: int = DEFAULT_SLOT_COUNT):
        self.slot_count = int(slot_count)
        self.active_types = {DataType.DECIMAL}  # Start with decimal only
        self.variable_counts = defaultdict(int)
        self.constant_counts = defaultdict(int)
        self.scope_depth = 0
        self.config_state = {}

    def activate_type(self, dtype: DataType):
        """Progressively activate new types."""
        self.active_types.add(dtype)
        if dtype == DataType.BOOLEAN:
            self.active_types.add(DataType.BOOLEAN)
        elif dtype == DataType.TENSOR:
            self.active_types.add(DataType.TENSOR)

    def _coerce_config_property(self, value: int) -> Optional[ConfigProperty]:
        try:
            return ConfigProperty(int(value))
        except ValueError:
            return None

    def _alloc_specifiers(self, category: Category, types: List[DataType]) -> List[int]:
        options: List[int] = []
        for dtype in types:
            if category == Category.VARIABLE:
                count = self.variable_counts[int(dtype)]
            else:
                count = self.constant_counts[int(dtype)]
            for idx in range(count + 1):
                options.append(pack_type_index(dtype, idx))
        return options

    def _existing_specifiers(self, category: Category, dtype: DataType) -> List[int]:
        if category == Category.VARIABLE:
            count = self.variable_counts[int(dtype)]
        else:
            count = self.constant_counts[int(dtype)]
        if count <= 0:
            return []
        return [pack_type_index(dtype, idx) for idx in range(count)]

    def _existing_specifiers_for_types(
        self, category: Category, types: List[DataType]
    ) -> List[int]:
        options: List[int] = []
        for dtype in types:
            options.extend(self._existing_specifiers(category, dtype))
        return options

    def _value_option_indices(
        self, op: int, prop: Optional[ConfigProperty] = None
    ) -> List[int]:
        enum = ValueEnumerations.get_enumeration((op, prop))
        return list(range(len(enum))) if enum else []

    def get_valid_options(self, instruction: GFSLInstruction, slot_index: int) -> List[int]:
        """Get valid options for a specific slot given all previous slots."""
        if slot_index >= self.slot_count:
            return []

        if slot_index == SLOT_TARGET_CAT:
            return [Category.NONE, Category.VARIABLE, Category.CONSTANT]

        if slot_index == SLOT_TARGET_SPEC:
            target_cat = instruction.slot_value(SLOT_TARGET_CAT)
            if target_cat == Category.NONE:
                return [0]
            if target_cat == Category.VARIABLE:
                return self._alloc_specifiers(Category.VARIABLE, sorted(self.active_types))
            if target_cat == Category.CONSTANT:
                return self._alloc_specifiers(
                    Category.CONSTANT,
                    [DataType.BOOLEAN, DataType.DECIMAL],
                )
            return []

        if slot_index == SLOT_OPERATION:
            target_cat = instruction.slot_value(SLOT_TARGET_CAT)
            target_spec = instruction.slot_value(SLOT_TARGET_SPEC)
            target_type = int(DataType.NONE)
            if target_cat in (Category.VARIABLE, Category.CONSTANT):
                target_type, _ = unpack_type_index(target_spec)

            valid_ops: List[Operation] = []
            custom_codes: List[int] = []
            try:
                dtype_enum = DataType(target_type)
            except ValueError:
                dtype_enum = DataType.NONE

            if target_cat != Category.NONE and dtype_enum != DataType.NONE:
                custom_codes = custom_operations.codes_for_target(dtype_enum)

            if target_cat == Category.NONE:
                valid_ops = CONTROL_FLOW_OPS
            elif target_type == DataType.BOOLEAN:
                valid_ops = BOOLEAN_COMPARE_OPS + BOOLEAN_LOGIC_OPS
            elif target_type == DataType.DECIMAL:
                valid_ops = DECIMAL_OPS
            elif target_type == DataType.TENSOR:
                valid_ops = TENSOR_OPS

            result_ops = [int(op) for op in valid_ops]
            result_ops.extend(custom_codes)
            return result_ops

        if slot_index == SLOT_SOURCE1_CAT:
            op = instruction.slot_value(SLOT_OPERATION)
            custom_op = custom_operations.get(op)
            if custom_op:
                allowed = custom_operations.allowed_categories(op, 1)
                if not allowed:
                    return []
                return [int(cat) for cat in allowed]

            if op == Operation.END:
                return [Category.NONE]
            if op == Operation.SET:
                return [Category.CONFIG]
            if op == Operation.RESULT:
                return [Category.VARIABLE]
            if op in (Operation.IF, Operation.WHILE):
                return [Category.VARIABLE, Category.CONSTANT]
            return [Category.VARIABLE, Category.CONSTANT, Category.VALUE]

        if slot_index == SLOT_SOURCE1_SPEC:
            source1_cat = instruction.slot_value(SLOT_SOURCE1_CAT)
            op = instruction.slot_value(SLOT_OPERATION)
            custom_op = custom_operations.get(op)
            if source1_cat == Category.NONE:
                return [0]

            if custom_op:
                if source1_cat == Category.CONFIG:
                    return [int(p) for p in ConfigProperty]
                if source1_cat == Category.VALUE:
                    return self._value_option_indices(op)
                dtype = custom_operations.source_type(op, 1) or custom_op.target_type
                if dtype is None:
                    return []
                return self._existing_specifiers(source1_cat, dtype)

            if source1_cat == Category.CONFIG:
                if op == Operation.SET:
                    return [int(p) for p in ConfigProperty]
                return []

            if source1_cat in (Category.VARIABLE, Category.CONSTANT):
                if op == Operation.RESULT:
                    return self._existing_specifiers_for_types(
                        source1_cat,
                        sorted(self.active_types),
                    )
                dtype = infer_source_type(op, 1)
                if dtype == int(DataType.NONE):
                    return []
                return self._existing_specifiers(source1_cat, DataType(dtype))

            if source1_cat == Category.VALUE:
                return self._value_option_indices(op)

            return []

        if slot_index == SLOT_SOURCE2_CAT:
            op = instruction.slot_value(SLOT_OPERATION)
            custom_op = custom_operations.get(op)
            if custom_op:
                if custom_operations.arity(op) < 2:
                    return [Category.NONE]
                allowed = custom_operations.allowed_categories(op, 2)
                if not allowed:
                    return []
                return [int(cat) for cat in allowed]

            if op == Operation.SET:
                return [Category.VALUE]
            if op in BINARY_OPS:
                return [Category.VARIABLE, Category.CONSTANT, Category.VALUE]
            return [Category.NONE]

        if slot_index == SLOT_SOURCE2_SPEC:
            source2_cat = instruction.slot_value(SLOT_SOURCE2_CAT)
            op = instruction.slot_value(SLOT_OPERATION)
            custom_op = custom_operations.get(op)
            if source2_cat == Category.NONE:
                return [0]

            if custom_op:
                if source2_cat == Category.CONFIG:
                    return [int(p) for p in ConfigProperty]
                if source2_cat == Category.VALUE:
                    return self._value_option_indices(op)
                dtype = custom_operations.source_type(op, 2) or custom_op.target_type
                if dtype is None:
                    return []
                return self._existing_specifiers(source2_cat, dtype)

            if source2_cat == Category.CONFIG:
                if op == Operation.SET:
                    return [int(p) for p in ConfigProperty]
                return []

            if source2_cat in (Category.VARIABLE, Category.CONSTANT):
                dtype = infer_source_type(op, 2)
                if dtype == int(DataType.NONE):
                    return []
                return self._existing_specifiers(source2_cat, DataType(dtype))

            if source2_cat == Category.VALUE:
                prop = None
                if op == Operation.SET and instruction.slot_value(SLOT_SOURCE1_CAT) == Category.CONFIG:
                    prop = self._coerce_config_property(instruction.slot_value(SLOT_SOURCE1_SPEC))
                return self._value_option_indices(op, prop)

            return []

        return []

    def _prob_state_key(
        self, instruction: GFSLInstruction, slot_index: int
    ) -> Tuple[int, Tuple[int, ...]]:
        return slot_index, tuple(instruction.slots[:slot_index])

    def _ensure_instruction_slots(self, instruction: GFSLInstruction) -> None:
        if len(instruction.slots) < self.slot_count:
            instruction.pad_to(self.slot_count)

    def _completion_probability(
        self,
        instruction: GFSLInstruction,
        slot_index: int,
        cache: Dict[Tuple[int, Tuple[int, ...]], float],
        max_branching: int,
    ) -> float:
        if slot_index >= self.slot_count:
            return 1.0
        state_key = self._prob_state_key(instruction, slot_index)
        if state_key in cache:
            return cache[state_key]

        options = self.get_valid_options(instruction, slot_index)
        if not options:
            cache[state_key] = 0.0
            return 0.0
        if len(options) > max_branching:
            options = options[:max_branching]

        total = 0.0
        prior_value = instruction.slots[slot_index]
        for opt in options:
            instruction.slots[slot_index] = opt
            total += self._completion_probability(
                instruction, slot_index + 1, cache, max_branching
            )
        instruction.slots[slot_index] = prior_value

        probability = total / len(options)
        cache[state_key] = probability
        return probability

    def option_success_probabilities(
        self,
        instruction: GFSLInstruction,
        slot_index: int,
        max_branching: int = DEFAULT_PROBABILITY_BRANCHING,
    ) -> Dict[int, float]:
        """Return probability of completing an instruction for each slot option."""
        self._ensure_instruction_slots(instruction)
        options = self.get_valid_options(instruction, slot_index)
        if not options:
            return {}
        cache: Dict[Tuple[int, Tuple[int, ...]], float] = {}
        probabilities: Dict[int, float] = {}
        prior_value = instruction.slots[slot_index]
        for opt in options:
            instruction.slots[slot_index] = opt
            probabilities[opt] = self._completion_probability(
                instruction, slot_index + 1, cache, max_branching
            )
        instruction.slots[slot_index] = prior_value
        return probabilities

    def viable_options(
        self,
        instruction: GFSLInstruction,
        slot_index: int,
        max_branching: int = DEFAULT_PROBABILITY_BRANCHING,
    ) -> Tuple[List[int], Dict[int, float]]:
        """Return options that can still lead to a valid completion."""
        probabilities = self.option_success_probabilities(
            instruction, slot_index, max_branching
        )
        viable = [opt for opt, prob in probabilities.items() if prob > 0.0]
        return viable, probabilities

    def choose_option(
        self,
        instruction: GFSLInstruction,
        slot_index: int,
        max_branching: int = DEFAULT_PROBABILITY_BRANCHING,
    ) -> int:
        """Choose a slot option biased toward successful completion."""
        self._ensure_instruction_slots(instruction)
        valid_options = self.get_valid_options(instruction, slot_index)
        if not valid_options:
            raise ValueError(f"No valid options for slot {slot_index}.")
        if len(valid_options) == 1:
            return valid_options[0]

        viable, probabilities = self.viable_options(
            instruction, slot_index, max_branching
        )
        candidates = viable or valid_options

        weights = [probabilities.get(opt, 0.0) for opt in candidates]
        if viable and sum(weights) > 0:
            return random.choices(candidates, weights=weights, k=1)[0]
        return random.choice(candidates)

    def build_probability_tree(
        self,
        instruction: GFSLInstruction,
        slot_index: int = 0,
        max_branching: int = 32,
    ) -> Dict[int, Dict[str, Any]]:
        """Return a bounded probability tree for instruction completion."""
        self._ensure_instruction_slots(instruction)
        if slot_index >= self.slot_count:
            return {}
        options = self.get_valid_options(instruction, slot_index)
        if len(options) > max_branching:
            options = options[:max_branching]

        cache: Dict[Tuple[int, Tuple[int, ...]], float] = {}
        prior_value = instruction.slots[slot_index]
        tree: Dict[int, Dict[str, Any]] = {}
        for opt in options:
            instruction.slots[slot_index] = opt
            probability = self._completion_probability(
                instruction, slot_index + 1, cache, max_branching
            )
            children = self.build_probability_tree(
                instruction, slot_index + 1, max_branching
            )
            tree[opt] = {"probability": probability, "children": children}
        instruction.slots[slot_index] = prior_value
        return tree

    def update_state(self, instruction: GFSLInstruction):
        """Update validator state after instruction is added."""
        target_cat = instruction.target_cat
        target_type = instruction.target_type
        target_index = instruction.target_index
        op = instruction.operation

        if target_cat == Category.VARIABLE:
            if target_index >= self.variable_counts[target_type]:
                self.variable_counts[target_type] = target_index + 1
        elif target_cat == Category.CONSTANT:
            if target_index >= self.constant_counts[target_type]:
                self.constant_counts[target_type] = target_index + 1

        if op == Operation.IF or op == Operation.WHILE:
            self.scope_depth += 1
        elif op == Operation.END:
            self.scope_depth = max(0, self.scope_depth - 1)

        if op == Operation.SET:
            prop_value = instruction.source1_value
            prop = self._coerce_config_property(prop_value)
            value_idx = instruction.source2_value
            context = (op, prop)
            enum = ValueEnumerations.get_enumeration(context)
            if value_idx < len(enum):
                self.config_state[prop_value] = enum[value_idx]


__all__ = ["SlotValidator"]
