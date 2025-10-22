# -*- coding: utf-8 -*-
"""
GFSL-Aligned Evolvo Library
===========================

An implementation of the Genetic Fixed Structure Language (GFSL) inspired by the
"GFSL Basic" and "GFSL Extensibility" papers. The library keeps the fixed-slot
instruction contract, cascading validity system, and progressive type activation
while adding a supervised PyTorch guidance model that learns to steer evolution
toward promising regions of the search space.

Core ideas carried over from the papers:

1. **Fixed 10-Slot Instructions**: Every instruction is exactly 10 integer indices
2. **Cascading Validity**: Each slot's options depend entirely on previous slots
3. **Progressive Type System**: Types activate based on problem complexity
4. **Context-Dependent Enumerations**: Values selected from operation-specific lists
5. **Effective Algorithm Extraction**: Backward dependency tracing from outputs
6. **Unified Evolution**: Common framework for algorithms and neural architectures

Additional enhancements:
- GFSLInstruction: Fixed 10-slot instruction representation
- SlotValidator: Enforces cascading validity constraints
- GFSLGenome: Algorithm/Neural genome with slot-based instructions
- EffectiveAlgorithmExtractor: Removes junk genome
- RealTimeEvaluator: Multiple execution with different inputs
- RecursiveModelBuilder: Neural architecture using GFSL instructions
- GFSLSupervisedGuide: PyTorch learner that provides a dynamic evolution bias
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import hashlib
import json
import copy
import inspect
from typing import List, Dict, Any, Tuple, Optional, Union, Set, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum
from abc import ABC, abstractmethod
import traceback
from datetime import datetime
from pathlib import Path


# ============================================================================
# SLOT DEFINITIONS AND ENUMERATIONS
# ============================================================================

class Category(IntEnum):
    """Target/Source categories as integer indices"""
    NONE = 0
    VARIABLE = 1  # $
    CONSTANT = 2  # #
    VALUE = 3     # !
    CONFIG = 4    # &

class DataType(IntEnum):
    """Data types as integer indices"""
    NONE = 0
    BOOLEAN = 1   # b
    DECIMAL = 2   # d
    TENSOR = 3    # t

class Operation(IntEnum):
    """Operations as integer indices"""
    # Control Flow (target = NONE)
    IF = 0
    WHILE = 1
    END = 2
    SET = 3
    RESULT = 4
    
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
    """Configuration properties for neural operations"""
    CHANNELS = 0
    KERNEL = 1
    STRIDE = 2
    PADDING = 3
    UNITS = 4
    RATE = 5
    MOMENTUM = 6
    EPSILON = 7

# ============================================================================
# CONTEXT-DEPENDENT ENUMERATIONS
# ============================================================================

class ValueEnumerations:
    """Context-specific value enumerations"""
    
    # Common mathematical values
    MATH_CONSTANTS = [0.0, 1.0, -1.0, 2.0, 0.5, 3.14159, 2.71828, 10.0]
    
    # Neural network channels (powers of 2)
    CHANNELS = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    # Kernel sizes
    KERNELS = [1, 3, 5, 7, 9, 11]
    
    # Strides and padding
    STRIDES = [1, 2, 3, 4]
    PADDINGS = [0, 1, 2, 3, 4, 5]
    
    # Probabilities (for dropout, etc.)
    PROBABILITIES = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
    
    # Loop counts
    LOOP_COUNTS = [1, 2, 3, 5, 10, 20, 50, 100]
    
    @staticmethod
    def get_enumeration(context: Tuple[Operation, Optional[ConfigProperty]]) -> List[float]:
        """Get enumeration based on operation context"""
        op, prop = context

        # Custom operations may contribute their own enumerations
        custom_def = None
        try:
            custom_def = custom_operations.get(int(op))
        except NameError:
            # Manager not yet initialised; fall back to defaults
            custom_def = None
        if custom_def and custom_def.value_enumeration is not None:
            return list(custom_def.value_enumeration)
        
        # Configuration contexts
        if op == Operation.SET:
            if prop == ConfigProperty.CHANNELS:
                return ValueEnumerations.CHANNELS
            elif prop == ConfigProperty.KERNEL:
                return ValueEnumerations.KERNELS
            elif prop in [ConfigProperty.STRIDE, ConfigProperty.PADDING]:
                return ValueEnumerations.STRIDES
            elif prop == ConfigProperty.RATE:
                return ValueEnumerations.PROBABILITIES
                
        # Operation contexts
        if op in [Operation.ADD, Operation.SUB, Operation.MUL, Operation.DIV]:
            return ValueEnumerations.MATH_CONSTANTS
        elif op in [Operation.IF, Operation.WHILE]:
            return ValueEnumerations.LOOP_COUNTS
        elif op == Operation.DROPOUT:
            return ValueEnumerations.PROBABILITIES
            
        return ValueEnumerations.MATH_CONSTANTS

# ============================================================================
# CUSTOM OPERATIONS
# ============================================================================


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
        source_types: Optional[Tuple[Optional[Union[DataType, int]], Optional[Union[DataType, int]]]] = None,
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
            p for p in signature.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
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
            p.name == "context" and p.kind in (
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
        
        # Keep per-target lists ordered for stable behaviour
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

# ============================================================================
# GFSL INSTRUCTION
# ============================================================================

@dataclass
class GFSLInstruction:
    """
    Fixed 10-slot instruction representation.
    Each slot is an integer index.
    """
    slots: List[int]
    
    def __init__(self, slots: Optional[List[int]] = None):
        if slots is None:
            # Initialize with NONE instruction
            self.slots = [0] * 10
        else:
            assert len(slots) == 10, "Instruction must have exactly 10 slots"
            self.slots = slots
    
    @property
    def target_cat(self) -> int: return self.slots[0]
    
    @property
    def target_type(self) -> int: return self.slots[1]
    
    @property
    def target_index(self) -> int: return self.slots[2]
    
    @property
    def operation(self) -> int: return self.slots[3]
    
    @property
    def source1_cat(self) -> int: return self.slots[4]
    
    @property
    def source1_type(self) -> int: return self.slots[5]
    
    @property
    def source1_value(self) -> int: return self.slots[6]
    
    @property
    def source2_cat(self) -> int: return self.slots[7]
    
    @property
    def source2_type(self) -> int: return self.slots[8]
    
    @property
    def source2_value(self) -> int: return self.slots[9]
    
    def get_signature(self) -> str:
        """Get unique signature for this instruction"""
        return '|'.join(str(s) for s in self.slots)
    
    def copy(self) -> 'GFSLInstruction':
        """Create deep copy"""
        return GFSLInstruction(self.slots.copy())

# ============================================================================
# SLOT VALIDATOR
# ============================================================================

class SlotValidator:
    """
    Enforces cascading validity - each slot's valid options
    depend entirely on all previous slots.
    """
    
    def __init__(self):
        self.active_types = {DataType.DECIMAL}  # Start with decimal only
        self.variable_counts = defaultdict(int)  # Track allocated variables
        self.constant_counts = defaultdict(int)  # Track defined constants
        self.scope_depth = 0
        self.config_state = {}  # Current configuration state for SET operations
    
    def activate_type(self, dtype: DataType):
        """Progressively activate new types"""
        self.active_types.add(dtype)
        if dtype == DataType.BOOLEAN:
            # Boolean enables control flow
            self.active_types.add(DataType.BOOLEAN)
        elif dtype == DataType.TENSOR:
            # Tensor enables neural operations
            self.active_types.add(DataType.TENSOR)
    
    def get_valid_options(self, instruction: GFSLInstruction, slot_index: int) -> List[int]:
        """Get valid options for a specific slot given all previous slots"""
        
        # Slot 0: Target Category
        if slot_index == 0:
            return [Category.NONE, Category.VARIABLE, Category.CONSTANT]
        
        # Slot 1: Target Type
        elif slot_index == 1:
            target_cat = instruction.slots[0]
            if target_cat == Category.NONE:
                return [DataType.NONE]
            elif target_cat == Category.VARIABLE:
                return list(self.active_types)
            elif target_cat == Category.CONSTANT:
                # Only primitives can be constants
                return [DataType.BOOLEAN, DataType.DECIMAL]
            return [DataType.NONE]
        
        # Slot 2: Target Index
        elif slot_index == 2:
            target_cat = instruction.slots[0]
            target_type = instruction.slots[1]
            
            if target_cat == Category.NONE:
                return [0]
            elif target_cat == Category.VARIABLE:
                # Existing variables + option to allocate new
                count = self.variable_counts[target_type]
                return list(range(count + 1))
            elif target_cat == Category.CONSTANT:
                count = self.constant_counts[target_type]
                return list(range(count + 1))
            return [0]
        
        # Slot 3: Operation
        elif slot_index == 3:
            target_cat = instruction.slots[0]
            target_type = instruction.slots[1]
            
            valid_ops = []
            custom_codes: List[int] = []
            try:
                dtype_enum = DataType(target_type)
            except ValueError:
                dtype_enum = DataType.NONE
            
            if target_cat != Category.NONE and dtype_enum != DataType.NONE:
                custom_codes = custom_operations.codes_for_target(dtype_enum)
            
            if target_cat == Category.NONE:
                # Control flow operations
                valid_ops = [Operation.IF, Operation.WHILE, Operation.END, 
                           Operation.SET, Operation.RESULT]
                
            elif target_type == DataType.BOOLEAN:
                # Boolean operations
                valid_ops = [Operation.GT, Operation.LT, Operation.EQ,
                           Operation.GTE, Operation.LTE, Operation.NEQ,
                           Operation.AND, Operation.OR, Operation.NOT]
                
            elif target_type == DataType.DECIMAL:
                # Decimal operations
                valid_ops = [Operation.ADD, Operation.SUB, Operation.MUL,
                           Operation.DIV, Operation.POW, Operation.SQRT,
                           Operation.ABS, Operation.SIN, Operation.COS,
                           Operation.EXP, Operation.LOG, Operation.MOD]
                
            elif target_type == DataType.TENSOR:
                # Tensor operations
                valid_ops = [Operation.CONV, Operation.LINEAR, Operation.RELU,
                           Operation.POOL, Operation.NORM, Operation.DROPOUT,
                           Operation.SOFTMAX, Operation.RESHAPE, Operation.CONCAT]
            
            result_ops = [int(op) for op in valid_ops]
            result_ops.extend(custom_codes)
            return result_ops
        
        # Slot 4: Source1 Category
        elif slot_index == 4:
            op = instruction.slots[3]
            custom_op = custom_operations.get(op)
            if custom_op:
                allowed = custom_operations.allowed_categories(op, 1)
                if not allowed:
                    return [Category.NONE]
                return [int(cat) for cat in allowed]
            
            if op == Operation.END:
                return [Category.NONE]
            elif op == Operation.SET:
                return [Category.CONFIG]  # Configuration property
            elif op == Operation.RESULT:
                return [Category.VARIABLE]
            elif op in [Operation.IF, Operation.WHILE]:
                return [Category.VARIABLE, Category.CONSTANT]  # Boolean source
            elif op in [Operation.NOT, Operation.SQRT, Operation.ABS, 
                       Operation.SIN, Operation.COS, Operation.EXP, Operation.LOG]:
                # Unary operations
                return [Category.VARIABLE, Category.CONSTANT, Category.VALUE]
            else:
                # Binary operations
                return [Category.VARIABLE, Category.CONSTANT, Category.VALUE]
        
        # Slot 5: Source1 Type
        elif slot_index == 5:
            source1_cat = instruction.slots[4]
            op = instruction.slots[3]
            custom_op = custom_operations.get(op)
            
            if custom_op:
                if source1_cat == Category.NONE:
                    return [DataType.NONE]
                elif source1_cat == Category.CONFIG:
                    return [0]
                dtype = custom_operations.source_type(op, 1) or custom_op.target_type
                if dtype is None:
                    return [DataType.NONE]
                return [int(dtype)]
            
            if source1_cat == Category.NONE:
                return [DataType.NONE]
            elif source1_cat == Category.CONFIG:
                # Config properties don't have types
                return [0]  # Placeholder
            elif source1_cat in [Category.VARIABLE, Category.CONSTANT]:
                # Type depends on operation requirements
                if op in [Operation.GT, Operation.LT, Operation.EQ, 
                         Operation.GTE, Operation.LTE, Operation.NEQ]:
                    return [DataType.DECIMAL]
                elif op in [Operation.AND, Operation.OR, Operation.NOT]:
                    return [DataType.BOOLEAN]
                elif op in [Operation.IF, Operation.WHILE]:
                    return [DataType.BOOLEAN]
                elif op >= Operation.ADD and op <= Operation.MOD:
                    return [DataType.DECIMAL]
                elif op >= Operation.CONV and op <= Operation.CONCAT:
                    return [DataType.TENSOR]
                    
            elif source1_cat == Category.VALUE:
                # Inline values - type implicit from operation
                return [DataType.DECIMAL]  # Most common
                
            return [DataType.NONE]
        
        # Slot 6: Source1 Value/Index
        elif slot_index == 6:
            source1_cat = instruction.slots[4]
            source1_type = instruction.slots[5]
            op = instruction.slots[3]
            
            if source1_cat == Category.NONE:
                return [0]
            elif source1_cat == Category.CONFIG:
                # Configuration properties
                if op == Operation.SET:
                    return [int(p) for p in ConfigProperty]
            elif source1_cat == Category.VARIABLE:
                count = self.variable_counts[source1_type]
                return list(range(count)) if count > 0 else [0]
            elif source1_cat == Category.CONSTANT:
                count = self.constant_counts[source1_type]
                return list(range(count)) if count > 0 else [0]
            elif source1_cat == Category.VALUE:
                # Index into context-specific enumeration
                context = (op, None)
                enum = ValueEnumerations.get_enumeration(context)
                return list(range(len(enum)))
                
            return [0]
        
        # Slots 7-9: Source2 (similar logic to Source1)
        elif slot_index == 7:
            op = instruction.slots[3]
            custom_op = custom_operations.get(op)
            if custom_op:
                if custom_operations.arity(op) < 2:
                    return [Category.NONE]
                allowed = custom_operations.allowed_categories(op, 2)
                if not allowed:
                    return [Category.NONE]
                return [int(cat) for cat in allowed]
            # Check if operation is binary
            if op in [Operation.ADD, Operation.SUB, Operation.MUL, Operation.DIV,
                     Operation.POW, Operation.MOD, Operation.AND, Operation.OR,
                     Operation.GT, Operation.LT, Operation.EQ, Operation.GTE,
                     Operation.LTE, Operation.NEQ]:
                return [Category.VARIABLE, Category.CONSTANT, Category.VALUE]
            else:
                return [Category.NONE]
        
        elif slot_index == 8:
            source2_cat = instruction.slots[7]
            op = instruction.slots[3]
            custom_op = custom_operations.get(op)
            if custom_op:
                if source2_cat == Category.NONE:
                    return [DataType.NONE]
                elif source2_cat == Category.CONFIG:
                    return [0]
                dtype = custom_operations.source_type(op, 2) or custom_op.target_type
                if dtype is None:
                    return [DataType.NONE]
                return [int(dtype)]
            if source2_cat == Category.NONE:
                return [DataType.NONE]
            # Similar logic to source1_type
            return self.get_valid_options(instruction, 5)  # Reuse source1 logic
        
        elif slot_index == 9:
            source2_cat = instruction.slots[7]
            op = instruction.slots[3]
            custom_op = custom_operations.get(op)
            if custom_op:
                if source2_cat == Category.NONE:
                    return [0]
                elif source2_cat == Category.CONFIG:
                    return [0]
                elif source2_cat == Category.VARIABLE:
                    dtype = custom_operations.source_type(op, 2) or custom_op.target_type
                    dtype_idx = int(dtype) if dtype is not None else 0
                    count = self.variable_counts[dtype_idx]
                    return list(range(count)) if count > 0 else [0]
                elif source2_cat == Category.CONSTANT:
                    dtype = custom_operations.source_type(op, 2) or custom_op.target_type
                    dtype_idx = int(dtype) if dtype is not None else 0
                    count = self.constant_counts[dtype_idx]
                    return list(range(count)) if count > 0 else [0]
                elif source2_cat == Category.VALUE:
                    options = custom_operations.value_options(op)
                    if options is not None and len(options) > 0:
                        return list(range(len(options)))
                    context = (op, None)
                    enum = ValueEnumerations.get_enumeration(context)
                    return list(range(len(enum))) if enum else [0]
                return [0]
            if source2_cat == Category.NONE:
                return [0]
            # Similar logic to source1_value
            return self.get_valid_options(instruction, 6)  # Reuse source1 logic
        
        return [0]
    
    def update_state(self, instruction: GFSLInstruction):
        """Update validator state after instruction is added"""
        target_cat = instruction.target_cat
        target_type = instruction.target_type
        target_index = instruction.target_index
        op = instruction.operation
        
        # Track variable/constant allocation
        if target_cat == Category.VARIABLE:
            if target_index >= self.variable_counts[target_type]:
                self.variable_counts[target_type] = target_index + 1
        elif target_cat == Category.CONSTANT:
            if target_index >= self.constant_counts[target_type]:
                self.constant_counts[target_type] = target_index + 1
        
        # Track scope depth
        if op == Operation.IF or op == Operation.WHILE:
            self.scope_depth += 1
        elif op == Operation.END:
            self.scope_depth = max(0, self.scope_depth - 1)
        
        # Handle SET operations
        if op == Operation.SET:
            prop = instruction.source1_value
            value_idx = instruction.source2_value
            context = (op, prop)
            enum = ValueEnumerations.get_enumeration(context)
            if value_idx < len(enum):
                self.config_state[prop] = enum[value_idx]

# ============================================================================
# GFSL GENOME
# ============================================================================

class GFSLGenome:
    """
    Genome using fixed-structure GFSL instructions.
    Supports both algorithmic and neural network representations.
    """
    
    def __init__(self, genome_type: str = "algorithm"):
        self.genome_type = genome_type
        self.instructions: List[GFSLInstruction] = []
        self.validator = SlotValidator()
        self.outputs: List[Tuple[int, int, int]] = []  # (category, type, index)
        self.fitness: Optional[float] = None
        self.generation: int = 0
        self._signature: Optional[str] = None
        self._effective_instructions: Optional[List[int]] = None
    
    def add_instruction_interactive(self) -> GFSLInstruction:
        """
        Build instruction slot-by-slot with cascading validity.
        This is the key method for Q-learning integration.
        """
        instruction = GFSLInstruction()
        
        for slot_idx in range(10):
            valid_options = self.validator.get_valid_options(instruction, slot_idx)
            if not valid_options:
                valid_options = [0]
            
            # For now, random selection (Q-learning would choose here)
            instruction.slots[slot_idx] = random.choice(valid_options)
        
        self.instructions.append(instruction)
        self.validator.update_state(instruction)
        self._signature = None
        self._effective_instructions = None
        
        return instruction
    
    def add_instruction(self, instruction: GFSLInstruction) -> bool:
        """Add a complete instruction with validation"""
        # Validate each slot against current state
        test_instr = GFSLInstruction()
        for slot_idx in range(10):
            valid_options = self.validator.get_valid_options(test_instr, slot_idx)
            if instruction.slots[slot_idx] not in valid_options:
                return False
            test_instr.slots[slot_idx] = instruction.slots[slot_idx]
        
        self.instructions.append(instruction.copy())
        self.validator.update_state(instruction)
        self._signature = None
        self._effective_instructions = None
        
        return True
    
    def mark_output(self, var_type: DataType, var_index: int):
        """Mark a variable as output"""
        self.outputs.append((Category.VARIABLE, var_type, var_index))
        self._effective_instructions = None
    
    def extract_effective_algorithm(self) -> List[int]:
        """
        Extract effective algorithm by tracing dependencies backward from outputs.
        Returns indices of instructions that contribute to outputs.
        """
        if self._effective_instructions is not None:
            return self._effective_instructions
        
        if not self.outputs:
            # No outputs defined, consider all instructions effective
            self._effective_instructions = list(range(len(self.instructions)))
            return self._effective_instructions
        
        # Build dependency graph
        dependencies = defaultdict(set)  # instruction_idx -> set of instruction_idx it depends on
        producers = {}  # (category, type, index) -> instruction_idx that produces it
        
        for idx, instr in enumerate(self.instructions):
            # Record what this instruction produces
            if instr.target_cat != Category.NONE:
                target_key = (instr.target_cat, instr.target_type, instr.target_index)
                producers[target_key] = idx
            
            # Record dependencies
            if instr.source1_cat in [Category.VARIABLE, Category.CONSTANT]:
                source_key = (instr.source1_cat, instr.source1_type, instr.source1_value)
                if source_key in producers:
                    dependencies[idx].add(producers[source_key])
            
            if instr.source2_cat in [Category.VARIABLE, Category.CONSTANT]:
                source_key = (instr.source2_cat, instr.source2_type, instr.source2_value)
                if source_key in producers:
                    dependencies[idx].add(producers[source_key])
        
        # Backward trace from outputs
        effective = set()
        to_check = []
        
        # Find instructions that produce outputs
        for output in self.outputs:
            if output in producers:
                to_check.append(producers[output])
        
        # Trace dependencies
        while to_check:
            idx = to_check.pop()
            if idx not in effective:
                effective.add(idx)
                for dep_idx in dependencies[idx]:
                    to_check.append(dep_idx)
        
        self._effective_instructions = sorted(effective)
        return self._effective_instructions
    
    def get_signature(self) -> str:
        """Generate unique signature for this genome"""
        if self._signature is None:
            effective = self.extract_effective_algorithm()
            sig_parts = []
            for idx in effective:
                sig_parts.append(self.instructions[idx].get_signature())
            self._signature = hashlib.md5('|'.join(sig_parts).encode()).hexdigest()
        return self._signature
    
    def to_human_readable(self) -> List[str]:
        """Convert to human-readable format"""
        readable = []
        for idx, instr in enumerate(self.instructions):
            is_effective = idx in self.extract_effective_algorithm()
            prefix = "✓" if is_effective else "✗"
            
            # Decode instruction
            if instr.target_cat == Category.NONE:
                if instr.operation == Operation.IF:
                    readable.append(f"{prefix} IF {self._decode_source(instr, 1)}")
                elif instr.operation == Operation.WHILE:
                    readable.append(f"{prefix} WHILE {self._decode_source(instr, 1)}")
                elif instr.operation == Operation.END:
                    readable.append(f"{prefix} END")
                elif instr.operation == Operation.RESULT:
                    readable.append(f"{prefix} RESULT {self._decode_source(instr, 1)}")
            else:
                target = self._decode_target(instr)
                op_name = resolve_operation_name(instr.operation)
                source1 = self._decode_source(instr, 1)
                source2 = self._decode_source(instr, 2)
                
                if instr.source2_cat == Category.NONE:
                    readable.append(f"{prefix} {target} = {op_name}({source1})")
                else:
                    readable.append(f"{prefix} {target} = {op_name}({source1}, {source2})")
        
        return readable
    
    def _decode_target(self, instr: GFSLInstruction) -> str:
        """Decode target to readable format"""
        cat = Category(instr.target_cat)
        dtype = DataType(instr.target_type)
        idx = instr.target_index
        
        if cat == Category.VARIABLE:
            return f"{dtype.name[0].lower()}${idx}"
        elif cat == Category.CONSTANT:
            return f"{dtype.name[0].lower()}#{idx}"
        return "NONE"
    
    def _decode_source(self, instr: GFSLInstruction, source_num: int) -> str:
        """Decode source to readable format"""
        if source_num == 1:
            cat = Category(instr.source1_cat)
            dtype = DataType(instr.source1_type)
            val = instr.source1_value
        else:
            cat = Category(instr.source2_cat)
            dtype = DataType(instr.source2_type)
            val = instr.source2_value
        
        if cat == Category.NONE:
            return "NONE"
        elif cat == Category.VARIABLE:
            return f"{dtype.name[0].lower()}${val}"
        elif cat == Category.CONSTANT:
            return f"{dtype.name[0].lower()}#{val}"
        elif cat == Category.VALUE:
            # Get actual value from enumeration
            context = (instr.operation, None)
            enum = ValueEnumerations.get_enumeration(context)
            if val < len(enum):
                return str(enum[val])
            return f"VAL[{val}]"
        elif cat == Category.CONFIG:
            return ConfigProperty(val).name
        
        return f"?{cat}:{dtype}:{val}"

# ============================================================================
# EXECUTION ENGINE
# ============================================================================

class GFSLExecutor:
    """Executes GFSL genomes"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset execution state"""
        self.variables = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.constants = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.config_state = {}
        self.execution_trace = []
    
    def execute(self, genome: GFSLGenome, inputs: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Execute genome and return outputs.
        
        Args:
            genome: The GFSL genome to execute
            inputs: Optional input values as {"d$0": 5.0, "b$0": True, ...}
        
        Returns:
            Dictionary of output values
        """
        self.reset()
        
        # Set input values
        if inputs:
            for key, value in inputs.items():
                # Parse key like "d$0" or "b#1"
                dtype_char = key[0]
                is_const = '#' in key
                idx = int(key.split('$' if '$' in key else '#')[1])
                
                dtype = {'b': DataType.BOOLEAN, 'd': DataType.DECIMAL}.get(dtype_char, DataType.DECIMAL)
                
                if is_const:
                    self.constants[dtype][idx] = value
                else:
                    self.variables[dtype][idx] = value
        
        # Execute only effective instructions
        effective_indices = genome.extract_effective_algorithm()
        
        for idx in effective_indices:
            if idx < len(genome.instructions):
                self._execute_instruction(genome.instructions[idx])
        
        # Collect outputs
        outputs = {}
        for cat, dtype, idx in genome.outputs:
            if cat == Category.VARIABLE:
                key = f"{DataType(dtype).name[0].lower()}${idx}"
                outputs[key] = self.variables[dtype][idx]
            elif cat == Category.CONSTANT:
                key = f"{DataType(dtype).name[0].lower()}#{idx}"
                outputs[key] = self.constants[dtype][idx]
        
        return outputs
    
    def _execute_instruction(self, instr: GFSLInstruction):
        """Execute a single instruction"""
        op_code = instr.operation
        custom_op = custom_operations.get(op_code)
        try:
            op = Operation(op_code)
        except ValueError:
            op = None
        
        # Get source values
        source1 = self._get_value(instr, 1)
        source2 = self._get_value(instr, 2)
        
        # Execute operation
        result = None
        
        if custom_op:
            args = [source1]
            if custom_op.arity >= 2:
                args.append(source2)
            context = {"executor": self, "instruction": instr}
            try:
                if custom_op.accepts_context:
                    result = custom_op.function(*args, context=context)
                else:
                    result = custom_op.function(*args)
            except Exception as exc:
                raise RuntimeError(
                    f"Error executing custom operation '{custom_op.name}': {exc}"
                ) from exc
        elif op is None:
            # Unknown opcode - ignore gracefully
            return
        else:
            # Control flow
            if op == Operation.IF:
                # Would need stack-based scope management for full implementation
                pass
            elif op == Operation.WHILE:
                pass
            elif op == Operation.END:
                pass
            elif op == Operation.SET:
                # Store configuration
                if instr.source1_cat == Category.CONFIG:
                    self.config_state[instr.source1_value] = source2
            elif op == Operation.RESULT:
                # Mark as output (handled elsewhere)
                pass
            
            # Boolean operations
            elif op == Operation.GT:
                result = float(source1) > float(source2)
            elif op == Operation.LT:
                result = float(source1) < float(source2)
            elif op == Operation.EQ:
                result = abs(float(source1) - float(source2)) < 1e-9
            elif op == Operation.AND:
                result = bool(source1) and bool(source2)
            elif op == Operation.OR:
                result = bool(source1) or bool(source2)
            elif op == Operation.NOT:
                result = not bool(source1)
            
            # Decimal operations
            elif op == Operation.ADD:
                result = float(source1) + float(source2)
            elif op == Operation.SUB:
                result = float(source1) - float(source2)
            elif op == Operation.MUL:
                result = float(source1) * float(source2)
            elif op == Operation.DIV:
                result = float(source1) / float(source2) if source2 != 0 else 0.0
            elif op == Operation.POW:
                try:
                    result = float(source1) ** float(source2)
                except:
                    result = 0.0
            elif op == Operation.SQRT:
                result = float(source1) ** 0.5 if source1 >= 0 else 0.0
            elif op == Operation.ABS:
                result = abs(float(source1))
            elif op == Operation.SIN:
                result = np.sin(float(source1))
            elif op == Operation.COS:
                result = np.cos(float(source1))
            elif op == Operation.EXP:
                try:
                    result = np.exp(float(source1))
                except:
                    result = 0.0
            elif op == Operation.LOG:
                result = np.log(float(source1)) if source1 > 0 else -float('inf')
            elif op == Operation.MOD:
                result = float(source1) % float(source2) if source2 != 0 else 0.0
        
        # Store result
        if result is not None and instr.target_cat != Category.NONE:
            if instr.target_cat == Category.VARIABLE:
                self.variables[instr.target_type][instr.target_index] = result
            elif instr.target_cat == Category.CONSTANT:
                self.constants[instr.target_type][instr.target_index] = result
    
    def _get_value(self, instr: GFSLInstruction, source_num: int) -> Any:
        """Get value for a source"""
        if source_num == 1:
            cat = Category(instr.source1_cat)
            dtype = DataType(instr.source1_type)
            val = instr.source1_value
        else:
            cat = Category(instr.source2_cat)
            dtype = DataType(instr.source2_type)
            val = instr.source2_value
        
        if cat == Category.NONE:
            return None
        elif cat == Category.VARIABLE:
            return self.variables[dtype][val]
        elif cat == Category.CONSTANT:
            return self.constants[dtype][val]
        elif cat == Category.VALUE:
            # Get from enumeration
            context = (instr.operation, None)
            enum = ValueEnumerations.get_enumeration(context)
            if val < len(enum):
                return enum[val]
            return 0.0
        elif cat == Category.CONFIG:
            return val  # Config property index
        
        return 0.0

# ============================================================================
# REAL-TIME EVALUATOR
# ============================================================================

class RealTimeEvaluator:
    """
    Evaluates genomes multiple times with different inputs,
    supporting real-time scoring and aggregation.
    """
    
    def __init__(self, test_cases: List[Dict[str, float]], 
                 expected_outputs: Optional[List[Dict[str, float]]] = None,
                 score_aggregator: Optional[Callable] = None):
        """
        Args:
            test_cases: List of input dictionaries for each test
            expected_outputs: Optional expected outputs for each test
            score_aggregator: Custom function to aggregate scores across tests
        """
        self.test_cases = test_cases
        self.expected_outputs = expected_outputs or [{}] * len(test_cases)
        self.score_aggregator = score_aggregator or self._default_aggregator
        self.executor = GFSLExecutor()
    
    def _default_aggregator(self, scores: List[float]) -> float:
        """Default score aggregation: mean with penalty for failures"""
        if not scores:
            return -float('inf')
        
        valid_scores = [s for s in scores if s != -float('inf')]
        if not valid_scores:
            return -float('inf')
        
        # Penalize for failed test cases
        success_rate = len(valid_scores) / len(scores)
        mean_score = sum(valid_scores) / len(valid_scores)
        
        return mean_score * success_rate
    
    def evaluate(self, genome: GFSLGenome, 
                callback: Optional[Callable[[int, Dict, float], None]] = None) -> float:
        """
        Evaluate genome on all test cases.
        
        Args:
            genome: The genome to evaluate
            callback: Optional callback(test_idx, output, score) for real-time feedback
        
        Returns:
            Aggregated fitness score
        """
        scores = []
        
        for idx, (inputs, expected) in enumerate(zip(self.test_cases, self.expected_outputs)):
            try:
                # Execute genome
                outputs = self.executor.execute(genome, inputs)
                
                # Calculate score for this test case
                score = self._calculate_score(outputs, expected)
                scores.append(score)
                
                # Real-time callback
                if callback:
                    callback(idx, outputs, score)
                
            except Exception as e:
                scores.append(-float('inf'))
                if callback:
                    callback(idx, {}, -float('inf'))
        
        # Aggregate scores
        return self.score_aggregator(scores)
    
    def _calculate_score(self, outputs: Dict[str, float], 
                        expected: Dict[str, float]) -> float:
        """Calculate score for a single test case"""
        if not expected:
            # No expected output, just check if execution succeeded
            return 1.0 if outputs else 0.0
        
        # Calculate error
        total_error = 0.0
        for key, expected_val in expected.items():
            if key in outputs:
                error = abs(outputs[key] - expected_val)
                total_error += error
            else:
                total_error += abs(expected_val)  # Missing output penalty
        
        # Convert error to score (lower error = higher score)
        return 1.0 / (1.0 + total_error)

# ============================================================================
# RECURSIVE MODEL BUILDER
# ============================================================================

class RecursiveModelBuilder:
    """
    Builds neural network models using GFSL instructions,
    supporting recursive architecture selection.
    """
    
    def __init__(self):
        self.layers = []
        self.config_state = {}
        self.current_shape = None
    
    def build_from_genome(self, genome: GFSLGenome, input_shape: Tuple[int, ...]) -> nn.Module:
        """Build PyTorch model from GFSL genome"""
        self.layers = []
        self.config_state = {}
        self.current_shape = input_shape
        
        # Process instructions
        for instr in genome.instructions:
            self._process_neural_instruction(instr)
        
        # Create sequential model
        if not self.layers:
            # Empty model - just identity
            return nn.Identity()
        
        return nn.Sequential(*self.layers)
    
    def _process_neural_instruction(self, instr: GFSLInstruction):
        """Process instruction for neural architecture building"""
        try:
            op = Operation(instr.operation)
        except ValueError:
            # Custom operations are ignored by the neural builder
            return
        
        # Configuration setting
        if op == Operation.SET:
            if instr.source1_cat == Category.CONFIG:
                prop = ConfigProperty(instr.source1_value)
                value_idx = instr.source2_value
                
                # Get actual value from enumeration
                context = (op, prop)
                enum = ValueEnumerations.get_enumeration(context)
                if value_idx < len(enum):
                    self.config_state[prop] = enum[value_idx]
        
        # Layer operations
        elif op == Operation.CONV:
            channels = int(self.config_state.get(ConfigProperty.CHANNELS, 32))
            kernel = int(self.config_state.get(ConfigProperty.KERNEL, 3))
            stride = int(self.config_state.get(ConfigProperty.STRIDE, 1))
            padding = int(self.config_state.get(ConfigProperty.PADDING, 1))
            
            # Infer input channels from current shape
            in_channels = self.current_shape[0] if self.current_shape else 3
            
            layer = nn.Conv2d(in_channels, channels, kernel, stride, padding)
            self.layers.append(layer)
            
            # Update shape
            if self.current_shape and len(self.current_shape) >= 3:
                h, w = self.current_shape[1:3]
                h_out = (h + 2 * padding - kernel) // stride + 1
                w_out = (w + 2 * padding - kernel) // stride + 1
                self.current_shape = (channels, h_out, w_out)
            else:
                self.current_shape = (channels, None, None)
            
            # Clear config for next layer
            self.config_state = {}
        
        elif op == Operation.LINEAR:
            units = int(self.config_state.get(ConfigProperty.UNITS, 128))
            
            # Add flatten if needed
            if self.current_shape and len(self.current_shape) > 1:
                self.layers.append(nn.Flatten())
                in_features = np.prod(self.current_shape)
            else:
                in_features = self.current_shape[0] if self.current_shape else 128
            
            layer = nn.Linear(in_features, units)
            self.layers.append(layer)
            
            self.current_shape = (units,)
            self.config_state = {}
        
        elif op == Operation.RELU:
            self.layers.append(nn.ReLU())
        
        elif op == Operation.DROPOUT:
            rate = self.config_state.get(ConfigProperty.RATE, 0.5)
            self.layers.append(nn.Dropout(rate))
            self.config_state = {}
        
        elif op == Operation.POOL:
            kernel = int(self.config_state.get(ConfigProperty.KERNEL, 2))
            stride = int(self.config_state.get(ConfigProperty.STRIDE, 2))
            
            layer = nn.MaxPool2d(kernel, stride)
            self.layers.append(layer)
            
            # Update shape
            if self.current_shape and len(self.current_shape) >= 3:
                h, w = self.current_shape[1:3]
                h_out = (h - kernel) // stride + 1
                w_out = (w - kernel) // stride + 1
                self.current_shape = (self.current_shape[0], h_out, w_out)
            
            self.config_state = {}
        
        elif op == Operation.NORM:
            if self.current_shape:
                if len(self.current_shape) >= 3:
                    # BatchNorm2d for conv layers
                    self.layers.append(nn.BatchNorm2d(self.current_shape[0]))
                else:
                    # BatchNorm1d for linear layers
                    self.layers.append(nn.BatchNorm1d(self.current_shape[0]))
        
        elif op == Operation.SOFTMAX:
            self.layers.append(nn.Softmax(dim=-1))

# ============================================================================
# EVOLUTIONARY SYSTEM
# ============================================================================

class GFSLEvolver:
    """Evolution engine for GFSL genomes"""
    
    def __init__(self, population_size: int = 50, supervised_guide: Optional['GFSLSupervisedGuide'] = None):
        self.population_size = population_size
        self.population: List[GFSLGenome] = []
        self.generation = 0
        self.mutation_rate = 0.3
        self.crossover_rate = 0.7
        self.elite_ratio = 0.1
        self.diversity_cache: Set[str] = set()
        self.supervised_guide = supervised_guide

    def set_supervised_guide(self, guide: 'GFSLSupervisedGuide'):
        """Attach or replace the supervised guidance model."""
        self.supervised_guide = guide
    
    def initialize_population(self, genome_type: str = "algorithm", 
                            initial_instructions: int = 10):
        """Create initial random population"""
        self.population = []
        self.diversity_cache = set()
        attempts = 0
        max_attempts = self.population_size * 10
        
        while len(self.population) < self.population_size and attempts < max_attempts:
            genome = GFSLGenome(genome_type)
            
            # Add random instructions
            for _ in range(random.randint(1, initial_instructions)):
                genome.add_instruction_interactive()
            
            # Add to population if unique
            sig = genome.get_signature()
            if sig not in self.diversity_cache:
                self.diversity_cache.add(sig)
                self.population.append(genome)
            
            attempts += 1
    
    def mutate(self, genome: GFSLGenome) -> GFSLGenome:
        """Mutate genome using slot-level mutations"""
        mutated = copy.deepcopy(genome)
        mutation_type = random.choice(['slot', 'add', 'remove'])
        
        if mutation_type == 'slot' and mutated.instructions:
            # Mutate a random slot in a random instruction
            instr_idx = random.randint(0, len(mutated.instructions) - 1)
            slot_idx = random.randint(0, 9)
            
            # Get valid options for this slot
            test_instr = mutated.instructions[instr_idx].copy()
            valid_options = mutated.validator.get_valid_options(test_instr, slot_idx)
            
            if valid_options:
                # Choose different value
                current_val = test_instr.slots[slot_idx]
                other_options = [v for v in valid_options if v != current_val]
                if other_options:
                    mutated.instructions[instr_idx].slots[slot_idx] = random.choice(other_options)
                    
                    # Cascade updates to dependent slots
                    for next_slot in range(slot_idx + 1, 10):
                        next_valid = mutated.validator.get_valid_options(
                            mutated.instructions[instr_idx], next_slot)
                        if next_valid:
                            mutated.instructions[instr_idx].slots[next_slot] = random.choice(next_valid)
        
        elif mutation_type == 'add':
            # Add new instruction
            mutated.add_instruction_interactive()
        
        elif mutation_type == 'remove' and len(mutated.instructions) > 1:
            # Remove random instruction
            idx = random.randint(0, len(mutated.instructions) - 1)
            mutated.instructions.pop(idx)
        
        mutated._signature = None
        mutated._effective_instructions = None
        mutated.fitness = None
        mutated.generation = genome.generation
        return mutated
    
    def crossover(self, parent1: GFSLGenome, parent2: GFSLGenome) -> GFSLGenome:
        """Crossover two genomes"""
        child = GFSLGenome(parent1.genome_type)
        
        # Single-point crossover
        if parent1.instructions and parent2.instructions:
            point1 = random.randint(0, len(parent1.instructions))
            point2 = random.randint(0, len(parent2.instructions))
            
            # Take instructions from both parents
            for instr in parent1.instructions[:point1]:
                child.add_instruction(instr)
            for instr in parent2.instructions[point2:]:
                child.add_instruction(instr)
        
        # Inherit outputs from random parent
        parent_outputs = random.choice([parent1.outputs, parent2.outputs])
        child.outputs = parent_outputs.copy()
        
        return child
    
    def evolve(self, generations: int, evaluator: Callable[[GFSLGenome], float],
              progress_callback: Optional[Callable[[int, GFSLGenome, float], None]] = None):
        """
        Main evolution loop.
        
        Args:
            generations: Number of generations to evolve
            evaluator: Fitness function
            progress_callback: Optional callback(generation, best_genome, best_fitness)
        """
        for gen in range(generations):
            self.generation = gen
            
            # Evaluate population
            for genome in self.population:
                if genome.fitness is None:
                    try:
                        genome.fitness = evaluator(genome)
                    except Exception:
                        genome.fitness = -float('inf')
            
            # Sort by fitness
            self.population.sort(key=lambda g: g.fitness or -float('inf'), reverse=True)

            if self.supervised_guide:
                self.supervised_guide.observe_population(self.population)
            
            # Progress callback
            if progress_callback and self.population:
                best = self.population[0]
                progress_callback(gen, best, best.fitness or -float('inf'))
            
            # Print progress
            if self.population and gen % 10 == 0:
                best = self.population[0]
                effective_size = len(best.extract_effective_algorithm())
                print(f"Gen {gen:03d}: Best Fitness={best.fitness:.4f}, "
                      f"Effective Size={effective_size}/{len(best.instructions)}")
            
            # Create next generation
            elite_size = int(self.population_size * self.elite_ratio)
            new_population = self.population[:elite_size]
            
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(random.choice([parent1, parent2]))
                
                child.fitness = None
                child.generation = gen + 1
                
                # Mutation
                if random.random() < self.mutation_rate:
                    if self.supervised_guide:
                        child = self.supervised_guide.propose_mutation(self, child)
                    else:
                        child = self.mutate(child)
                    child.fitness = None
                    child.generation = gen + 1
                
                # Add if unique
                sig = child.get_signature()
                if sig not in self.diversity_cache:
                    self.diversity_cache.add(sig)
                    new_population.append(child)
            
            self.population = new_population[:self.population_size]
    
    def _tournament_select(self, tournament_size: int = 3) -> GFSLGenome:
        """Tournament selection"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda g: g.fitness or -float('inf'))

# ============================================================================
# Q-LEARNING INTEGRATION
# ============================================================================

class GFSLQLearningGuide:
    """
    Q-Learning agent for guiding GFSL instruction construction.
    Learns which slot values work best given previous slots.
    """
    
    def __init__(self, learning_rate: float = 0.1, epsilon: float = 0.1):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount_factor = 0.95
        
        # Q-table: (slot_index, previous_slots_hash, action) -> Q-value
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.experience_buffer = deque(maxlen=1000)
    
    def get_state_key(self, instruction: GFSLInstruction, slot_index: int) -> str:
        """Generate state key from previous slots"""
        prev_slots = instruction.slots[:slot_index]
        return f"{slot_index}:{':'.join(str(s) for s in prev_slots)}"
    
    def choose_action(self, instruction: GFSLInstruction, slot_index: int,
                     valid_options: List[int]) -> int:
        """Choose slot value using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.choice(valid_options)
        
        state_key = self.get_state_key(instruction, slot_index)
        q_values = self.q_table[state_key]
        
        # Get Q-values for valid options
        valid_q = {opt: q_values.get(opt, 0.0) for opt in valid_options}
        
        if not valid_q:
            return random.choice(valid_options)
        
        # Choose action with highest Q-value
        max_q = max(valid_q.values())
        best_actions = [a for a, q in valid_q.items() if q == max_q]
        return random.choice(best_actions)
    
    def build_instruction(self, validator: SlotValidator) -> GFSLInstruction:
        """Build instruction using Q-learning guidance"""
        instruction = GFSLInstruction()
        states_actions = []
        
        for slot_idx in range(10):
            valid_options = validator.get_valid_options(instruction, slot_idx)
            if not valid_options:
                valid_options = [0]
            
            state_key = self.get_state_key(instruction, slot_idx)
            action = self.choose_action(instruction, slot_idx, valid_options)
            
            instruction.slots[slot_idx] = action
            states_actions.append((state_key, action))
        
        return instruction, states_actions
    
    def update_from_fitness(self, states_actions: List[Tuple[str, int]], 
                           fitness: float):
        """Update Q-values based on fitness reward"""
        # Use fitness as terminal reward
        reward = fitness
        
        # Update Q-values in reverse order (backward through slots)
        for i in range(len(states_actions) - 1, -1, -1):
            state_key, action = states_actions[i]
            
            current_q = self.q_table[state_key][action]
            
            if i == len(states_actions) - 1:
                # Terminal state
                new_q = current_q + self.learning_rate * (reward - current_q)
            else:
                # Non-terminal state
                next_state, next_action = states_actions[i + 1]
                next_q = self.q_table[next_state][next_action]
                new_q = current_q + self.learning_rate * (
                    reward + self.discount_factor * next_q - current_q)
            
            self.q_table[state_key][action] = new_q

# ============================================================================
# SUPERVISED EVOLUTION GUIDE
# ============================================================================

class GFSLFeatureExtractor:
    """Encodes GFSL genomes into fixed feature vectors aligned with GFSL papers."""

    def __init__(self, max_instructions: int = 128, max_outputs: int = 16, max_depth: int = 16):
        self.max_instructions = max_instructions
        self.max_outputs = max_outputs
        self.max_depth = max_depth
        self.operations = list(Operation)
        self.categories = list(Category)
        self.data_types = list(DataType)
        self.base_features = 8
        self.feature_dim = (
            self.base_features
            + len(self.operations)
            + len(self.categories)
            + len(self.data_types) * 2
        )

    def encode(self, genome: GFSLGenome) -> torch.Tensor:
        """Return feature vector capturing structure, control flow, and type usage."""
        features = np.zeros(self.feature_dim, dtype=np.float32)
        instructions = genome.instructions
        instr_count = len(instructions)
        effective = genome.extract_effective_algorithm() if instr_count else []
        effective_ratio = len(effective) / instr_count if instr_count else 0.0
        output_ratio = min(len(genome.outputs) / self.max_outputs, 1.0)

        operation_counts = defaultdict(int)
        target_category_counts = defaultdict(int)
        target_type_counts = defaultdict(int)
        source_type_counts = defaultdict(int)

        control_flow = 0
        set_ops = 0
        value_sources = 0
        depth = 0
        max_depth_seen = 0
        tensor_flag = False
        target_total = 0
        source_total = 0

        for instr in instructions:
            custom_op = custom_operations.get(instr.operation)
            try:
                op = Operation(instr.operation)
            except ValueError:
                op = None

            if op is not None:
                operation_counts[op] += 1

                if op in (Operation.IF, Operation.WHILE, Operation.END):
                    control_flow += 1
                if op in (Operation.IF, Operation.WHILE):
                    depth += 1
                    if depth > max_depth_seen:
                        max_depth_seen = depth
                elif op == Operation.END and depth > 0:
                    depth -= 1
                if op == Operation.SET:
                    set_ops += 1
            elif custom_op:
                if custom_op.target_type == DataType.TENSOR:
                    tensor_flag = True

            target_cat = Category(instr.target_cat)
            target_category_counts[target_cat] += 1
            if target_cat != Category.NONE:
                target_total += 1
                target_dtype = DataType(instr.target_type)
                target_type_counts[target_dtype] += 1
                if target_dtype == DataType.TENSOR or (
                    custom_op and custom_op.target_type == DataType.TENSOR
                ):
                    tensor_flag = True

            for source_cat_int, source_type_int in (
                (instr.source1_cat, instr.source1_type),
                (instr.source2_cat, instr.source2_type),
            ):
                source_cat = Category(source_cat_int)
                if source_cat == Category.NONE:
                    continue
                if source_cat == Category.VALUE:
                    value_sources += 1
                if source_cat in (Category.VARIABLE, Category.CONSTANT, Category.VALUE):
                    source_dtype = DataType(source_type_int)
                    source_type_counts[source_dtype] += 1
                    source_total += 1
                    if source_dtype == DataType.TENSOR:
                        tensor_flag = True

        instruction_density = min(instr_count / self.max_instructions, 1.0)
        max_depth_ratio = min(max_depth_seen / self.max_depth, 1.0)
        control_flow_ratio = control_flow / instr_count if instr_count else 0.0
        set_ratio = set_ops / instr_count if instr_count else 0.0
        value_ratio = value_sources / source_total if source_total else 0.0
        tensor_feature = 1.0 if tensor_flag else 0.0

        ptr = 0
        features[ptr] = instruction_density
        ptr += 1
        features[ptr] = effective_ratio
        ptr += 1
        features[ptr] = output_ratio
        ptr += 1
        features[ptr] = max_depth_ratio
        ptr += 1
        features[ptr] = control_flow_ratio
        ptr += 1
        features[ptr] = set_ratio
        ptr += 1
        features[ptr] = value_ratio
        ptr += 1
        features[ptr] = tensor_feature
        ptr += 1

        for op in self.operations:
            features[ptr] = operation_counts[op] / instr_count if instr_count else 0.0
            ptr += 1
        for cat in self.categories:
            features[ptr] = target_category_counts[cat] / instr_count if instr_count else 0.0
            ptr += 1
        target_den = target_total if target_total else 1
        for dtype in self.data_types:
            features[ptr] = target_type_counts[dtype] / target_den
            ptr += 1
        source_den = source_total if source_total else 1
        for dtype in self.data_types:
            features[ptr] = source_type_counts[dtype] / source_den
            ptr += 1

        return torch.tensor(features, dtype=torch.float32)


class GFSLSupervisedDirectionModel(nn.Module):
    """Small feed-forward network that predicts genome fitness."""

    def __init__(self, input_dim: int, hidden_layers: Optional[List[int]] = None):
        super().__init__()
        layers: List[nn.Module] = []
        widths = hidden_layers or [128, 64]
        prev = input_dim
        for width in widths:
            if width <= 0:
                continue
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


class GFSLSupervisedGuide:
    """Supervised PyTorch model that nudges evolution toward promising genomes."""

    def __init__(
        self,
        hidden_layers: Optional[List[int]] = None,
        buffer_size: int = 512,
        min_buffer: int = 64,
        batch_size: int = 64,
        epochs: int = 3,
        candidate_pool: int = 3,
        max_observations: int = 20,
        device: Optional[str] = None,
    ):
        self.feature_extractor = GFSLFeatureExtractor()
        self.model = GFSLSupervisedDirectionModel(
            self.feature_extractor.feature_dim,
            hidden_layers,
        )
        device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_name)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.buffer = deque(maxlen=buffer_size)
        self.targets = deque(maxlen=buffer_size)
        self.min_buffer = min_buffer
        self.batch_size = batch_size
        self.epochs = epochs
        self.candidate_pool = max(1, candidate_pool)
        self.max_observations = max_observations

        self.target_mean: Optional[float] = None
        self.target_std: Optional[float] = None
        self.trained = False
        self.loss_history: List[float] = []

    def observe_population(self, population: List[GFSLGenome]):
        """Collect labelled data from the current population and train when ready."""
        if not population:
            return

        observed = 0
        for genome in population[: self.max_observations]:
            if genome.fitness is None or not np.isfinite(genome.fitness):
                continue
            self.buffer.append(self.feature_extractor.encode(genome))
            self.targets.append(float(genome.fitness))
            observed += 1

        if observed and len(self.buffer) >= self.min_buffer:
            self._train_model()

    def _train_model(self):
        """Train the PyTorch model on buffered genomes."""
        if len(self.buffer) < self.min_buffer:
            return

        features = torch.stack(list(self.buffer)).to(self.device)
        targets_tensor = torch.tensor(list(self.targets), dtype=torch.float32, device=self.device)

        self.target_mean = float(targets_tensor.mean().item())
        target_std = float(targets_tensor.std().item())
        if target_std < 1e-6:
            target_std = 1.0
        self.target_std = target_std

        normalized_targets = (targets_tensor - self.target_mean) / target_std

        dataset_size = features.size(0)
        self.model.train()

        for _ in range(self.epochs):
            permutation = torch.randperm(dataset_size, device=self.device)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = permutation[start:end]
                batch_x = features[batch_idx]
                batch_y = normalized_targets[batch_idx]

                predictions = self.model(batch_x)
                loss = F.mse_loss(predictions, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

            self.loss_history.append(float(loss.item()))

        self.model.eval()
        self.trained = True

    def _predict_from_features(self, feature_list: List[torch.Tensor]) -> np.ndarray:
        if not feature_list:
            return np.array([], dtype=np.float32)
        if not self.trained or self.target_mean is None or self.target_std is None:
            return np.full(len(feature_list), np.nan, dtype=np.float32)

        with torch.no_grad():
            stacked = torch.stack(feature_list).to(self.device)
            preds = self.model(stacked)
        preds = preds.cpu().numpy()
        preds = preds * (self.target_std + 1e-6) + self.target_mean
        return preds.astype(np.float32)

    def predict(self, genomes: List[GFSLGenome]) -> np.ndarray:
        """Predict fitness for genomes without mutating them."""
        features = [self.feature_extractor.encode(g) for g in genomes]
        return self._predict_from_features(features)

    def propose_mutation(self, evolver: 'GFSLEvolver', genome: GFSLGenome) -> GFSLGenome:
        """Sample candidate mutations and return the model-preferred genome."""
        if not self.trained or len(self.buffer) < self.min_buffer:
            return evolver.mutate(genome)

        candidates: List[GFSLGenome] = []
        candidate_features: List[torch.Tensor] = []

        for _ in range(self.candidate_pool):
            candidate = evolver.mutate(genome)
            candidates.append(candidate)
            candidate_features.append(self.feature_extractor.encode(candidate))

        base_clone = copy.deepcopy(genome)
        candidates.append(base_clone)
        candidate_features.append(self.feature_extractor.encode(base_clone))

        predictions = self._predict_from_features(candidate_features)
        if predictions.size == 0 or not np.isfinite(predictions).any():
            return random.choice(candidates[:-1])

        best_idx = int(np.nanargmax(predictions))
        return candidates[best_idx]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_formula_discovery():
    """Example: Discover mathematical formulas"""
    print("=== GFSL Formula Discovery ===\n")
    
    # Create test cases for formula: y = x^2 + 2x + 1
    test_cases = []
    expected_outputs = []
    
    for x in range(-5, 6):
        test_cases.append({"d$0": float(x)})  # Input: x
        expected_outputs.append({"d$1": float(x**2 + 2*x + 1)})  # Output: y
    
    # Create evaluator
    evaluator_obj = RealTimeEvaluator(test_cases, expected_outputs)
    
    def fitness_func(genome):
        return evaluator_obj.evaluate(genome)
    
    # Initialize evolver
    evolver = GFSLEvolver(population_size=30)
    evolver.initialize_population("algorithm", initial_instructions=15)
    
    # Mark d$1 as output for all genomes
    for genome in evolver.population:
        genome.mark_output(DataType.DECIMAL, 1)
    
    # Evolve
    def progress_callback(gen, best_genome, best_fitness):
        if gen % 20 == 0:
            print(f"\nGeneration {gen}:")
            print(f"  Best fitness: {best_fitness:.6f}")
            print(f"  Effective algorithm:")
            for line in best_genome.to_human_readable():
                if line.startswith("✓"):
                    print(f"    {line}")
    
    evolver.evolve(100, fitness_func, progress_callback)
    
    # Show best solution
    best = evolver.population[0]
    print("\n=== Best Formula Found ===")
    print(f"Fitness: {best.fitness:.6f}")
    print(f"Signature: {best.get_signature()[:16]}...")
    print("\nFull Algorithm:")
    for line in best.to_human_readable():
        print(f"  {line}")
    
    # Test the formula
    print("\n=== Testing Formula ===")
    executor = GFSLExecutor()
    for x in [-2, 0, 1, 3]:
        result = executor.execute(best, {"d$0": float(x)})
        expected = x**2 + 2*x + 1
        print(f"  x={x}: Result={result.get('d$1', 0):.2f}, Expected={expected:.2f}")

def example_neural_architecture_search():
    """Example: Evolve neural network architecture"""
    print("\n=== GFSL Neural Architecture Search ===\n")
    
    # Create model builder
    builder = RecursiveModelBuilder()
    
    # Create a genome with neural instructions
    genome = GFSLGenome("neural")
    
    # Activate tensor type
    genome.validator.activate_type(DataType.TENSOR)
    
    # Build a simple CNN using GFSL instructions
    # SET CHANNELS = 32
    set_channels = GFSLInstruction([
        Category.NONE, DataType.NONE, 0, Operation.SET,
        Category.CONFIG, 0, ConfigProperty.CHANNELS,
        Category.VALUE, DataType.DECIMAL, 5  # Index 5 -> 64 channels
    ])
    genome.add_instruction(set_channels)
    
    # SET KERNEL = 3
    set_kernel = GFSLInstruction([
        Category.NONE, DataType.NONE, 0, Operation.SET,
        Category.CONFIG, 0, ConfigProperty.KERNEL,
        Category.VALUE, DataType.DECIMAL, 1  # Index 1 -> kernel size 3
    ])
    genome.add_instruction(set_kernel)
    
    # t$0 = CONV(t$0)
    conv = GFSLInstruction([
        Category.VARIABLE, DataType.TENSOR, 0, Operation.CONV,
        Category.VARIABLE, DataType.TENSOR, 0,
        Category.NONE, DataType.NONE, 0
    ])
    genome.add_instruction(conv)
    
    # t$0 = RELU(t$0)
    relu = GFSLInstruction([
        Category.VARIABLE, DataType.TENSOR, 0, Operation.RELU,
        Category.VARIABLE, DataType.TENSOR, 0,
        Category.NONE, DataType.NONE, 0
    ])
    genome.add_instruction(relu)
    
    # Build PyTorch model
    model = builder.build_from_genome(genome, (3, 32, 32))
    
    print("Generated Model Architecture:")
    print(model)
    
    # Test with random input
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Show genome in readable format
    print("\nGFSL Instructions:")
    for line in genome.to_human_readable():
        print(f"  {line}")

if __name__ == "__main__":
    # Run examples
    example_formula_discovery()
    example_neural_architecture_search()
