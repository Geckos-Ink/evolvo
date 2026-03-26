"""Cascading slot validator for GFSL instructions."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .custom_ops import custom_operations, infer_source_type
from .enums import (
    BINARY_OPS,
    BOOLEAN_COMPARE_OPS,
    BOOLEAN_LOGIC_OPS,
    Category,
    ConfigProperty,
    CONTROL_FLOW_OPS,
    DataType,
    DECIMAL_OPS,
    LIST_QUERY_OPS,
    LIST_TARGET_OPS,
    LIST_VALUE_OPS,
    Operation,
    TENSOR_OPS,
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
        self.list_counts = defaultdict(int)
        self.constant_list_counts = defaultdict(int)
        self.function_counts = defaultdict(int)
        self.scope_depth = 0
        self.scope_stack: List[Tuple[str, Optional[DataType]]] = []
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
        counters = self._counter_for_category(category)
        if counters is None:
            return options
        for dtype in types:
            count = counters[int(dtype)]
            for idx in range(count + 1):
                options.append(pack_type_index(dtype, idx))
        return options

    def _existing_specifiers(self, category: Category, dtype: DataType) -> List[int]:
        counters = self._counter_for_category(category)
        if counters is None:
            return []
        count = counters[int(dtype)]
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

    def _counter_for_category(self, category: Category):
        if category == Category.VARIABLE:
            return self.variable_counts
        if category == Category.CONSTANT:
            return self.constant_counts
        if category == Category.LIST:
            return self.list_counts
        if category == Category.LIST_CONSTANT:
            return self.constant_list_counts
        if category == Category.FUNCTION:
            return self.function_counts
        return None

    def _target_dtype(self, instruction: GFSLInstruction) -> Optional[DataType]:
        target_cat = instruction.slot_value(SLOT_TARGET_CAT)
        target_spec = instruction.slot_value(SLOT_TARGET_SPEC)
        if target_cat in (
            Category.VARIABLE,
            Category.CONSTANT,
            Category.LIST,
            Category.LIST_CONSTANT,
            Category.FUNCTION,
        ):
            target_type, _ = unpack_type_index(target_spec)
            try:
                return DataType(target_type)
            except ValueError:
                return None
        return None

    def _function_return_types(self) -> List[DataType]:
        return sorted(self.active_types | {DataType.NONE})

    def _has_functions(self, dtype: DataType) -> bool:
        return self.function_counts[int(dtype)] > 0

    def _current_scope_kind(self) -> Optional[str]:
        if not self.scope_stack:
            return None
        return self.scope_stack[-1][0]

    def _current_function_return_dtype(self) -> Optional[DataType]:
        for kind, return_dtype in reversed(self.scope_stack):
            if kind == "FUNC":
                return return_dtype
        return None

    def _has_scalar_source_for_dtype(self, dtype: DataType) -> bool:
        if self.variable_counts[int(dtype)] > 0:
            return True
        if self.constant_counts[int(dtype)] > 0:
            return True
        if dtype == DataType.DECIMAL:
            return bool(self._value_option_indices(Operation.APPEND))
        return False

    def _has_lists(self, dtype: Optional[DataType] = None, *, include_constant: bool = False) -> bool:
        if dtype is not None:
            if self.list_counts[int(dtype)] > 0:
                return True
            if include_constant and self.constant_list_counts[int(dtype)] > 0:
                return True
            return False
        for active_type in self.active_types:
            if self._has_lists(active_type, include_constant=include_constant):
                return True
        return False

    def _list_source_categories_for_dtype(self, dtype: DataType) -> List[int]:
        categories: List[int] = []
        if self.variable_counts[int(dtype)] > 0:
            categories.append(int(Category.VARIABLE))
        if self.constant_counts[int(dtype)] > 0:
            categories.append(int(Category.CONSTANT))
        if dtype == DataType.DECIMAL and self._value_option_indices(Operation.APPEND):
            categories.append(int(Category.VALUE))
        return categories

    def seed_list_count(
        self, dtype: DataType, count: int, *, constant: bool = False
    ) -> None:
        """Seed known list counts (useful for pre-existing runtime list inputs)."""
        dtype_key = int(DataType(dtype))
        target = self.constant_list_counts if constant else self.list_counts
        target[dtype_key] = max(int(target[dtype_key]), int(count))

    def get_valid_options(self, instruction: GFSLInstruction, slot_index: int) -> List[int]:
        """Get valid options for a specific slot given all previous slots."""
        if slot_index >= self.slot_count:
            return []

        if slot_index == SLOT_TARGET_CAT:
            return [
                Category.NONE,
                Category.VARIABLE,
                Category.CONSTANT,
                Category.LIST,
                Category.FUNCTION,
            ]

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
            if target_cat == Category.LIST:
                return self._alloc_specifiers(Category.LIST, sorted(self.active_types))
            if target_cat == Category.FUNCTION:
                return self._alloc_specifiers(Category.FUNCTION, self._function_return_types())
            return []

        if slot_index == SLOT_OPERATION:
            target_cat = instruction.slot_value(SLOT_TARGET_CAT)
            target_spec = instruction.slot_value(SLOT_TARGET_SPEC)
            target_type = int(DataType.NONE)
            if target_cat in (
                Category.VARIABLE,
                Category.CONSTANT,
                Category.LIST,
                Category.LIST_CONSTANT,
                Category.FUNCTION,
            ):
                target_type, _ = unpack_type_index(target_spec)

            valid_ops: List[Operation] = []
            custom_codes: List[int] = []
            try:
                dtype_enum = DataType(target_type)
            except ValueError:
                dtype_enum = DataType.NONE

            if (
                target_cat in (Category.VARIABLE, Category.CONSTANT)
                and dtype_enum != DataType.NONE
            ):
                custom_codes = custom_operations.codes_for_target(dtype_enum)

            if target_cat == Category.NONE:
                valid_ops = [op for op in CONTROL_FLOW_OPS if op != Operation.FUNC]
                if self._has_functions(DataType.NONE):
                    valid_ops.append(Operation.CALL)
            elif target_cat == Category.FUNCTION:
                valid_ops = [Operation.FUNC]
            elif target_cat == Category.LIST and dtype_enum != DataType.NONE:
                if self._has_scalar_source_for_dtype(dtype_enum):
                    valid_ops.extend([Operation.PREPEND, Operation.APPEND])
                if self._has_lists(dtype_enum, include_constant=True):
                    valid_ops.append(Operation.CLONE)
            elif target_type == DataType.BOOLEAN:
                valid_ops = BOOLEAN_COMPARE_OPS + BOOLEAN_LOGIC_OPS
                if self._has_lists(dtype_enum):
                    valid_ops.extend(LIST_VALUE_OPS)
                if self._has_lists():
                    valid_ops.append(Operation.LISTHASITEMS)
            elif target_type == DataType.DECIMAL:
                valid_ops = DECIMAL_OPS
                if self._has_lists(dtype_enum):
                    valid_ops.extend(LIST_VALUE_OPS)
                if self._has_lists():
                    valid_ops.append(Operation.LISTCOUNT)
            elif target_type == DataType.TENSOR:
                valid_ops = TENSOR_OPS
                if self._has_lists(dtype_enum):
                    valid_ops.extend(LIST_VALUE_OPS)

            if (
                target_cat in (Category.VARIABLE, Category.CONSTANT)
                and dtype_enum != DataType.NONE
                and self._has_functions(dtype_enum)
            ):
                valid_ops.append(Operation.CALL)

            result_ops = [int(op) for op in valid_ops]
            result_ops.extend(custom_codes)
            return result_ops

        if slot_index == SLOT_SOURCE1_CAT:
            op = instruction.slot_value(SLOT_OPERATION)
            custom_op = custom_operations.get(op)
            target_dtype = self._target_dtype(instruction)
            if custom_op:
                allowed = custom_operations.allowed_categories(op, 1)
                if not allowed:
                    return []
                return [int(cat) for cat in allowed]

            if op == Operation.END:
                if self._current_scope_kind() != "FUNC":
                    return [Category.NONE]
                return_dtype = self._current_function_return_dtype() or DataType.NONE
                if return_dtype == DataType.NONE:
                    return [Category.NONE, Category.CONSTANT]
                return [Category.VARIABLE]
            if op == Operation.FUNC:
                return [Category.NONE]
            if op == Operation.CALL:
                return [Category.FUNCTION]
            if op == Operation.SET:
                return [Category.CONFIG]
            if op == Operation.RESULT:
                return [Category.VARIABLE]
            if op in (Operation.IF, Operation.WHILE):
                return [Category.VARIABLE, Category.CONSTANT]
            if op in (Operation.PREPEND, Operation.APPEND):
                if target_dtype is None:
                    return []
                return self._list_source_categories_for_dtype(target_dtype)
            if op == Operation.CLONE:
                if target_dtype is None:
                    return []
                options: List[int] = []
                if self._has_lists(target_dtype):
                    options.append(int(Category.LIST))
                if self._has_lists(target_dtype, include_constant=True) and (
                    self.constant_list_counts[int(target_dtype)] > 0
                ):
                    options.append(int(Category.LIST_CONSTANT))
                return options
            if op in (Operation.FIFO, Operation.FILO, Operation.LISTCOUNT, Operation.LISTHASITEMS):
                return [Category.LIST]
            return [Category.VARIABLE, Category.CONSTANT, Category.VALUE]

        if slot_index == SLOT_SOURCE1_SPEC:
            source1_cat = instruction.slot_value(SLOT_SOURCE1_CAT)
            op = instruction.slot_value(SLOT_OPERATION)
            custom_op = custom_operations.get(op)
            target_dtype = self._target_dtype(instruction)
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

            if op == Operation.END:
                if self._current_scope_kind() != "FUNC":
                    return [0] if source1_cat == Category.NONE else []
                return_dtype = self._current_function_return_dtype() or DataType.NONE
                if source1_cat == Category.NONE:
                    return [0]
                if source1_cat == Category.VARIABLE:
                    if return_dtype == DataType.NONE:
                        return []
                    return self._existing_specifiers(Category.VARIABLE, return_dtype)
                if source1_cat == Category.CONSTANT and return_dtype == DataType.NONE:
                    return [pack_type_index(DataType.NONE, 0)]
                return []

            if source1_cat == Category.FUNCTION:
                if op != Operation.CALL:
                    return []
                target_cat = instruction.slot_value(SLOT_TARGET_CAT)
                if target_cat == Category.NONE:
                    return self._existing_specifiers(Category.FUNCTION, DataType.NONE)
                target_dtype = self._target_dtype(instruction)
                if target_dtype is None:
                    return []
                return self._existing_specifiers(Category.FUNCTION, target_dtype)

            if source1_cat == Category.CONFIG:
                if op == Operation.SET:
                    return [int(p) for p in ConfigProperty]
                return []

            if source1_cat in (Category.LIST, Category.LIST_CONSTANT):
                if source1_cat == Category.LIST_CONSTANT and op != Operation.CLONE:
                    return []
                if op in (Operation.CLONE, Operation.FIFO, Operation.FILO) and target_dtype:
                    return self._existing_specifiers(source1_cat, target_dtype)
                if op in (Operation.LISTCOUNT, Operation.LISTHASITEMS):
                    return self._existing_specifiers_for_types(
                        source1_cat, sorted(self.active_types)
                    )
                return []

            if source1_cat in (Category.VARIABLE, Category.CONSTANT):
                if op in (Operation.PREPEND, Operation.APPEND):
                    if target_dtype is None:
                        return []
                    return self._existing_specifiers(source1_cat, target_dtype)
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
                if op in (Operation.PREPEND, Operation.APPEND):
                    if target_dtype != DataType.DECIMAL:
                        return []
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

            if op in (Operation.FUNC, Operation.CALL, Operation.END):
                return [Category.NONE]
            if op in LIST_TARGET_OPS or op in LIST_VALUE_OPS or op in LIST_QUERY_OPS:
                return [Category.NONE]
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
        elif target_cat == Category.LIST:
            if target_index >= self.list_counts[target_type]:
                self.list_counts[target_type] = target_index + 1
        elif target_cat == Category.LIST_CONSTANT:
            if target_index >= self.constant_list_counts[target_type]:
                self.constant_list_counts[target_type] = target_index + 1
        elif target_cat == Category.FUNCTION:
            if target_index >= self.function_counts[target_type]:
                self.function_counts[target_type] = target_index + 1

        if op == Operation.IF or op == Operation.WHILE:
            self.scope_stack.append(("BLOCK", None))
        elif op == Operation.FUNC:
            try:
                return_dtype = DataType(target_type)
            except ValueError:
                return_dtype = DataType.NONE
            self.scope_stack.append(("FUNC", return_dtype))
        elif op == Operation.END:
            if self.scope_stack:
                self.scope_stack.pop()
        self.scope_depth = len(self.scope_stack)

        if op == Operation.SET:
            prop_value = instruction.source1_value
            prop = self._coerce_config_property(prop_value)
            value_idx = instruction.source2_value
            context = (op, prop)
            enum = ValueEnumerations.get_enumeration(context)
            if value_idx < len(enum):
                self.config_state[prop_value] = enum[value_idx]


__all__ = ["SlotValidator"]
