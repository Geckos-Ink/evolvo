"""Custom operation registration and helpers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict

from .enums import (
    Category,
    DataType,
    Operation,
    BOOLEAN_COMPARE_OPS,
    BOOLEAN_LOGIC_OPS,
    CONTROL_FLOW_OPS,
    DECIMAL_OPS,
    TENSOR_OPS,
)


@dataclass
class CustomOperation:
    """Metadata describing a user-registered operation."""

    code: int
    name: str
    target_type: DataType
    arity: int
    function: Callable[..., Any]
    source_types: Tuple[Optional[DataType], Optional[DataType]]
    allowed_source_categories: Tuple[Tuple[Category, ...], Tuple[Category, ...]]
    value_enumeration: Optional[Tuple[float, ...]]
    doc: str = ""
    accepts_context: bool = False


class CustomOperationManager:
    """Registry for ad-hoc operations."""

    def __init__(self, base_code: int = 1000):
        self.base_code = base_code
        self._ops_by_code: Dict[int, CustomOperation] = {}
        self._ops_by_target: Dict[DataType, List[CustomOperation]] = defaultdict(list)
        self._name_to_code: Dict[str, int] = {}
        self._next_code = base_code

    def register(
        self,
        name: str,
        target_type: Union[DataType, int],
        function: Callable[..., Any],
        *,
        arity: int = 2,
        source_types: Optional[
            Tuple[Optional[Union[DataType, int]], Optional[Union[DataType, int]]]
        ] = None,
        allowed_source_categories: Optional[
            Tuple[Tuple[Category, ...], Tuple[Category, ...]]
        ] = None,
        value_enumeration: Optional[Union[List[float], Tuple[float, ...]]] = None,
        code: Optional[int] = None,
        doc: str = "",
    ) -> int:
        """Register a new custom operation and return its opcode."""
        if arity not in (1, 2):
            raise ValueError("Custom operations currently support arity 1 or 2.")

        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Operation name must be a non-empty string.")
        name_key = clean_name.upper()
        if name_key in self._name_to_code:
            raise ValueError(f"Operation '{clean_name}' is already registered.")

        if code is None:
            code = self._next_code
            self._next_code += 1
        else:
            code = int(code)
            if code < self.base_code:
                raise ValueError(
                    f"Custom operation codes must be >= {self.base_code}; received {code}."
                )
            if code >= self._next_code:
                self._next_code = code + 1
        if code in self._ops_by_code:
            raise ValueError(f"Opcode {code} is already in use.")

        dtype_target = DataType(target_type)

        if source_types is None:
            if arity == 1:
                source_types = (dtype_target, None)
            else:
                source_types = (dtype_target, dtype_target)
        if len(source_types) != 2:
            raise ValueError("source_types must be a tuple of length 2.")
        resolved_source_types: Tuple[Optional[DataType], Optional[DataType]] = (
            None if source_types[0] is None else DataType(source_types[0]),
            None if source_types[1] is None else DataType(source_types[1]),
        )

        if allowed_source_categories is None:
            if arity == 1:
                allowed_source_categories = (
                    (Category.VARIABLE, Category.CONSTANT, Category.VALUE),
                    (Category.NONE,),
                )
            else:
                allowed_source_categories = (
                    (Category.VARIABLE, Category.CONSTANT, Category.VALUE),
                    (Category.VARIABLE, Category.CONSTANT, Category.VALUE),
                )
        if len(allowed_source_categories) != 2:
            raise ValueError("allowed_source_categories must contain two tuples.")
        allowed_source_categories = (
            tuple(allowed_source_categories[0]),
            tuple(allowed_source_categories[1]),
        )

        signature = inspect.signature(function)
        positional_params = [
            p
            for p in signature.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        required_positionals = [
            p for p in positional_params if p.default is inspect._empty
        ]
        if len(positional_params) < arity:
            raise ValueError(
                f"Function '{clean_name}' accepts fewer positional arguments than required arity {arity}."
            )
        if len(required_positionals) > arity:
            raise ValueError(
                f"Function '{clean_name}' requires more positional arguments than arity {arity}."
            )
        accepts_context = any(
            p.name == "context"
            and p.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            for p in signature.parameters.values()
        )

        value_enum_tuple = None
        if value_enumeration is not None:
            value_enum_tuple = tuple(float(v) for v in value_enumeration)

        op = CustomOperation(
            code=code,
            name=clean_name,
            target_type=dtype_target,
            arity=arity,
            function=function,
            source_types=resolved_source_types,
            allowed_source_categories=allowed_source_categories,
            value_enumeration=value_enum_tuple,
            doc=doc,
            accepts_context=accepts_context,
        )

        self._ops_by_code[code] = op
        self._ops_by_target[dtype_target].append(op)
        self._name_to_code[name_key] = code

        self._ops_by_target[dtype_target].sort(key=lambda item: item.code)
        return code

    def get(self, code: Union[int, Operation]) -> Optional[CustomOperation]:
        """Retrieve metadata for a custom operation code."""
        try:
            return self._ops_by_code[int(code)]
        except (KeyError, ValueError, TypeError):
            return None

    def get_code_by_name(self, name: str) -> Optional[int]:
        """Return opcode for a registered operation name if available."""
        return self._name_to_code.get(name.strip().upper())

    def get_by_name(self, name: str) -> Optional[CustomOperation]:
        code = self.get_code_by_name(name)
        return self._ops_by_code.get(code) if code is not None else None

    def codes_for_target(self, dtype: Union[DataType, int]) -> List[int]:
        """Return sorted custom opcodes that produce the requested data type."""
        dtype_enum = DataType(dtype)
        return [op.code for op in self._ops_by_target.get(dtype_enum, [])]

    def allowed_categories(self, code: int, source_index: int) -> Tuple[Category, ...]:
        op = self.get(code)
        if not op:
            return ()
        idx = source_index - 1
        if idx < 0 or idx >= len(op.allowed_source_categories):
            return ()
        return op.allowed_source_categories[idx]

    def source_type(self, code: int, source_index: int) -> Optional[DataType]:
        op = self.get(code)
        if not op:
            return None
        idx = source_index - 1
        if idx < 0 or idx >= len(op.source_types):
            return None
        dtype = op.source_types[idx]
        if dtype is None and source_index == 1:
            return op.target_type
        return dtype

    def value_options(self, code: int) -> Optional[List[float]]:
        op = self.get(code)
        if not op or op.value_enumeration is None:
            return None
        return list(op.value_enumeration)

    def arity(self, code: int) -> int:
        op = self.get(code)
        return op.arity if op else 0

    def accepts_context(self, code: int) -> bool:
        op = self.get(code)
        return bool(op and op.accepts_context)


custom_operations = CustomOperationManager()


def resolve_operation_name(op_code: int) -> str:
    """Return a readable name for built-in or custom operations."""
    custom_op = custom_operations.get(op_code)
    if custom_op:
        return custom_op.name.upper()
    try:
        return Operation(op_code).name
    except ValueError:
        return f"CUSTOM_{op_code}"


def register_custom_operation(
    name: str,
    target_type: Union[DataType, int],
    function: Callable[..., Any],
    *,
    arity: int = 2,
    source_types: Optional[
        Tuple[Optional[Union[DataType, int]], Optional[Union[DataType, int]]]
    ] = None,
    allowed_source_categories: Optional[
        Tuple[Tuple[Category, ...], Tuple[Category, ...]]
    ] = None,
    value_enumeration: Optional[Union[List[float], Tuple[float, ...]]] = None,
    code: Optional[int] = None,
    doc: str = "",
) -> int:
    """
    Public helper for registering ad-hoc operations.

    The callable should accept one or two positional arguments (matching ``arity``)
    and may optionally include a ``context`` keyword-only parameter that will receive
    a dictionary with the current executor and instruction.
    """
    return custom_operations.register(
        name,
        target_type,
        function,
        arity=arity,
        source_types=source_types,
        allowed_source_categories=allowed_source_categories,
        value_enumeration=value_enumeration,
        code=code,
        doc=doc,
    )


def infer_source_type(op_code: int, source_index: int) -> int:
    """Infer the expected data type for a source position given an opcode."""
    custom_op = custom_operations.get(op_code)
    if custom_op:
        dtype = custom_operations.source_type(op_code, source_index) or custom_op.target_type
        return int(dtype) if dtype is not None else int(DataType.NONE)
    try:
        op = Operation(op_code)
    except ValueError:
        return int(DataType.NONE)
    if op in BOOLEAN_COMPARE_OPS:
        return int(DataType.DECIMAL)
    if op in (Operation.AND, Operation.OR, Operation.NOT):
        return int(DataType.BOOLEAN)
    if op in (Operation.IF, Operation.WHILE):
        return int(DataType.BOOLEAN)
    if op in DECIMAL_OPS:
        return int(DataType.DECIMAL)
    if op in TENSOR_OPS:
        return int(DataType.TENSOR)
    if op in CONTROL_FLOW_OPS:
        return int(DataType.NONE)
    return int(DataType.NONE)


__all__ = [
    "CustomOperation",
    "CustomOperationManager",
    "custom_operations",
    "register_custom_operation",
    "resolve_operation_name",
    "infer_source_type",
]
