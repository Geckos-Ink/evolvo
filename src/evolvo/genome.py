"""GFSL genome and extraction utilities."""

from __future__ import annotations

import hashlib
import heapq
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from .builder import GFSLExpressionBuilder
from .custom_ops import resolve_operation_name
from .enums import Category, ConfigProperty, DataType, Operation
from .instruction import GFSLInstruction
from .slots import DEFAULT_SLOT_COUNT
from .validator import SlotValidator
from .values import ValueEnumerations


class GFSLGenome:
    """
    Genome using fixed-structure GFSL instructions.
    Supports both algorithmic and neural network representations.
    """

    def __init__(
        self,
        genome_type: str = "algorithm",
        slot_count: Optional[int] = None,
        auto_slot_count: bool = True,
    ):
        self.genome_type = genome_type
        self.instructions: List[GFSLInstruction] = []
        self.auto_slot_count = auto_slot_count
        if slot_count is None:
            slot_count = DEFAULT_SLOT_COUNT
        self.validator = SlotValidator(slot_count=slot_count)
        self.outputs: List[Tuple[int, int, int]] = []
        self.fitness: Optional[float] = None
        self.generation: int = 0
        self._signature: Optional[str] = None
        self._effective_instructions: Optional[List[int]] = None

    def expression(self) -> GFSLExpressionBuilder:
        """Create a slot-wise instruction builder bound to this genome."""
        return GFSLExpressionBuilder(self)

    def _ensure_slot_count(self, slot_count: int) -> None:
        if slot_count <= self.validator.slot_count:
            return
        for instr in self.instructions:
            instr.pad_to(slot_count)
        self.validator.slot_count = slot_count

    def add_instruction_interactive(self, max_attempts: int = 25) -> GFSLInstruction:
        """
        Build instruction slot-by-slot with cascading validity.
        This is the key method for Q-learning integration.
        """
        if self.validator.slot_count == 0:
            self._ensure_slot_count(DEFAULT_SLOT_COUNT)
        last_error: Optional[Exception] = None
        for _ in range(max_attempts):
            instruction = GFSLInstruction(slot_count=self.validator.slot_count)
            try:
                for slot_idx in range(self.validator.slot_count):
                    instruction.slots[slot_idx] = self.validator.choose_option(
                        instruction, slot_idx
                    )
            except ValueError as exc:
                last_error = exc
                continue

            self.instructions.append(instruction)
            self.validator.update_state(instruction)
            self._signature = None
            self._effective_instructions = None

            return instruction

        raise RuntimeError(
            f"Unable to build a valid instruction after {max_attempts} attempts."
        ) from last_error

    def add_instruction(self, instruction: GFSLInstruction) -> bool:
        """Add a complete instruction with validation."""
        instr_copy = instruction.copy()
        if self.auto_slot_count and not self.instructions:
            if len(instr_copy.slots) > 0:
                self.validator.slot_count = len(instr_copy.slots)
        if len(instr_copy.slots) > self.validator.slot_count:
            if self.auto_slot_count:
                self._ensure_slot_count(len(instr_copy.slots))
            else:
                return False
        if len(instr_copy.slots) < self.validator.slot_count:
            instr_copy.pad_to(self.validator.slot_count)
        test_instr = GFSLInstruction(slot_count=self.validator.slot_count)
        for slot_idx in range(self.validator.slot_count):
            valid_options = self.validator.get_valid_options(test_instr, slot_idx)
            if instr_copy.slots[slot_idx] not in valid_options:
                return False
            test_instr.slots[slot_idx] = instr_copy.slots[slot_idx]

        self.instructions.append(instr_copy)
        self.validator.update_state(instr_copy)
        self._signature = None
        self._effective_instructions = None

        return True

    def mark_output(self, var_type: DataType, var_index: int):
        """Mark a variable as output."""
        self.outputs.append((Category.VARIABLE, var_type, var_index))
        self._effective_instructions = None

    def _build_dependency_graph(
        self,
    ) -> Tuple[Dict[int, Set[int]], Dict[Tuple[int, int, int], int]]:
        """Build data dependency graph and latest producers by target address."""
        dependencies: Dict[int, Set[int]] = defaultdict(set)
        producers: Dict[Tuple[int, int, int], int] = {}

        for idx, instr in enumerate(self.instructions):
            if instr.target_cat != Category.NONE:
                target_key = (
                    int(instr.target_cat),
                    int(instr.target_type),
                    int(instr.target_index),
                )
                producers[target_key] = idx

            if instr.source1_cat in (Category.VARIABLE, Category.CONSTANT):
                source_key = (
                    int(instr.source1_cat),
                    int(instr.source1_type),
                    int(instr.source1_value),
                )
                if source_key in producers:
                    dependencies[idx].add(producers[source_key])

            if instr.source2_cat in (Category.VARIABLE, Category.CONSTANT):
                source_key = (
                    int(instr.source2_cat),
                    int(instr.source2_type),
                    int(instr.source2_value),
                )
                if source_key in producers:
                    dependencies[idx].add(producers[source_key])

        return dependencies, producers

    @staticmethod
    def _looks_like_reference(ref: Any) -> bool:
        if not isinstance(ref, (tuple, list)):
            return False
        if len(ref) == 2:
            return isinstance(ref[0], (int, DataType)) and isinstance(ref[1], int)
        if len(ref) == 3:
            return (
                isinstance(ref[0], (int, Category))
                and isinstance(ref[1], (int, DataType))
                and isinstance(ref[2], int)
            )
        return False

    @staticmethod
    def _coerce_result_reference(ref: Any) -> Tuple[int, int, int]:
        if isinstance(ref, str):
            text = ref.strip()
            if not text:
                raise ValueError("Result reference string is empty.")
            if "$" in text:
                dtype_char, idx_str = text.split("$", 1)
                cat = Category.VARIABLE
            elif "#" in text:
                dtype_char, idx_str = text.split("#", 1)
                cat = Category.CONSTANT
            else:
                raise ValueError(
                    f"Result reference '{ref}' must include '$' or '#'."
                )
            dtype_map = {
                "b": DataType.BOOLEAN,
                "d": DataType.DECIMAL,
                "t": DataType.TENSOR,
            }
            dtype = dtype_map.get(dtype_char.lower())
            if dtype is None:
                raise ValueError(f"Unknown dtype prefix '{dtype_char}' in '{ref}'.")
            return int(cat), int(dtype), int(idx_str)

        if isinstance(ref, dict):
            cat = ref.get("category", ref.get("cat", Category.VARIABLE))
            dtype = ref.get("dtype", ref.get("type"))
            index = ref.get("index", ref.get("idx"))
        elif isinstance(ref, (tuple, list)):
            if len(ref) == 2:
                cat = Category.VARIABLE
                dtype, index = ref
            elif len(ref) == 3:
                cat, dtype, index = ref
            else:
                raise ValueError(
                    "Tuple/list result references must be (dtype, index) or (category, dtype, index)."
                )
        else:
            raise ValueError("Result references must be a string, tuple/list, or dict.")

        if dtype is None or index is None:
            raise ValueError("Result references must include dtype and index.")

        cat_enum = Category(cat)
        if cat_enum not in (Category.VARIABLE, Category.CONSTANT):
            raise ValueError("Result references must target a variable or constant.")
        return int(cat_enum), int(DataType(dtype)), int(index)

    def _normalize_result_references(self, result_refs: Any) -> List[Tuple[int, int, int]]:
        if result_refs is None:
            return []

        if (
            isinstance(result_refs, (str, dict))
            or self._looks_like_reference(result_refs)
        ):
            refs = [result_refs]
        elif isinstance(result_refs, set):
            refs = list(result_refs)
        elif isinstance(result_refs, (list, tuple)):
            refs = list(result_refs)
        else:
            refs = [result_refs]

        normalized: List[Tuple[int, int, int]] = []
        for ref in refs:
            normalized.append(self._coerce_result_reference(ref))
        return normalized

    @staticmethod
    def _collect_required_indices(
        output_refs: List[Tuple[int, int, int]],
        dependencies: Dict[int, Set[int]],
        producers: Dict[Tuple[int, int, int], int],
    ) -> Set[int]:
        effective: Set[int] = set()
        to_check: List[int] = []

        for output in output_refs:
            key = (int(output[0]), int(output[1]), int(output[2]))
            if key in producers:
                to_check.append(producers[key])

        while to_check:
            idx = to_check.pop()
            if idx in effective:
                continue
            effective.add(idx)
            for dep_idx in dependencies.get(idx, ()): 
                to_check.append(dep_idx)

        return effective

    def _instruction_sort_key(self, instr: GFSLInstruction) -> Tuple[int, ...]:
        return tuple(int(s) for s in instr.slots)

    def _order_operation_indices(
        self,
        indices: Set[int],
        dependencies: Dict[int, Set[int]],
        order: str,
    ) -> List[int]:
        if not indices:
            return []

        order_key = order.lower().strip()
        if order_key in {"execution", "original", "index"}:
            return sorted(indices)

        key_map = {idx: self._instruction_sort_key(self.instructions[idx]) for idx in indices}

        if order_key in {"canonical", "fixed", "slots", "slot"}:
            return sorted(indices, key=lambda i: (key_map[i], i))

        if order_key in {"topological", "dependency", "deps"}:
            indegree = {idx: 0 for idx in indices}
            dependents: Dict[int, Set[int]] = defaultdict(set)

            for idx in indices:
                for dep_idx in dependencies.get(idx, ()):  
                    if dep_idx in indices:
                        indegree[idx] += 1
                        dependents[dep_idx].add(idx)

            heap: List[Tuple[Tuple[int, ...], int]] = []
            for idx, deg in indegree.items():
                if deg == 0:
                    heapq.heappush(heap, (key_map[idx], idx))

            ordered: List[int] = []
            while heap:
                _, idx = heapq.heappop(heap)
                ordered.append(idx)
                for child in dependents.get(idx, ()):  
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        heapq.heappush(heap, (key_map[child], child))

            if len(ordered) != len(indices):
                return sorted(indices, key=lambda i: (key_map[i], i))
            return ordered

        raise ValueError(
            f"Unknown order '{order}'. Use 'execution', 'topological', or 'fixed'."
        )

    def extract_operation_indices(
        self,
        result_refs: Optional[Any] = None,
        *,
        order: str = "fixed",
    ) -> List[int]:
        """
        Extract instruction indices required to compute the requested result references.
        Order 'fixed' sorts by slot values (enumerator order) for stable comparisons.
        """
        dependencies, producers = self._build_dependency_graph()

        if result_refs is None:
            if not self.outputs:
                indices = set(range(len(self.instructions)))
                return self._order_operation_indices(indices, dependencies, order)
            output_refs = self._normalize_result_references(self.outputs)
        else:
            output_refs = self._normalize_result_references(result_refs)
            if not output_refs:
                return []

        effective = self._collect_required_indices(output_refs, dependencies, producers)
        return self._order_operation_indices(effective, dependencies, order)

    def extract_operations(
        self,
        result_refs: Optional[Any] = None,
        *,
        order: str = "fixed",
    ) -> List[GFSLInstruction]:
        """Return the instructions needed for the requested result references."""
        indices = self.extract_operation_indices(result_refs, order=order)
        return [self.instructions[idx] for idx in indices]

    def extract_effective_algorithm(self) -> List[int]:
        """
        Extract effective algorithm by tracing dependencies backward from outputs.
        Returns indices of instructions that contribute to outputs.
        """
        if self._effective_instructions is not None:
            return self._effective_instructions

        if not self.outputs:
            self._effective_instructions = list(range(len(self.instructions)))
            return self._effective_instructions

        dependencies, producers = self._build_dependency_graph()
        output_refs = self._normalize_result_references(self.outputs)
        effective = self._collect_required_indices(output_refs, dependencies, producers)
        self._effective_instructions = sorted(effective)
        return self._effective_instructions

    def get_signature(self) -> str:
        """Generate unique signature for this genome."""
        if self._signature is None:
            effective = self.extract_effective_algorithm()
            sig_parts = []
            for idx in effective:
                sig_parts.append(self.instructions[idx].get_signature())
            self._signature = hashlib.md5("|".join(sig_parts).encode()).hexdigest()
        return self._signature

    def to_human_readable(self) -> List[str]:
        """Convert to human-readable format."""
        readable = []
        for idx, instr in enumerate(self.instructions):
            is_effective = idx in self.extract_effective_algorithm()
            prefix = "✓" if is_effective else "✗"

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
        """Decode target to readable format."""
        cat = Category(instr.target_cat)
        dtype = DataType(instr.target_type)
        idx = instr.target_index

        if cat == Category.VARIABLE:
            return f"{dtype.name[0].lower()}${idx}"
        if cat == Category.CONSTANT:
            return f"{dtype.name[0].lower()}#{idx}"
        return "NONE"

    def _decode_source(self, instr: GFSLInstruction, source_num: int) -> str:
        """Decode source to readable format."""
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
        if cat == Category.VARIABLE:
            return f"{dtype.name[0].lower()}${val}"
        if cat == Category.CONSTANT:
            return f"{dtype.name[0].lower()}#{val}"
        if cat == Category.VALUE:
            context = (instr.operation, None)
            if (
                instr.operation == Operation.SET
                and source_num == 2
                and instr.source1_cat == Category.CONFIG
            ):
                try:
                    prop = ConfigProperty(instr.source1_value)
                except ValueError:
                    prop = None
                context = (instr.operation, prop)
            enum = ValueEnumerations.get_enumeration(context)
            if val < len(enum):
                return str(enum[val])
            return f"VAL[{val}]"
        if cat == Category.CONFIG:
            return ConfigProperty(val).name

        return f"?{cat}:{dtype}:{val}"


__all__ = ["GFSLGenome"]
