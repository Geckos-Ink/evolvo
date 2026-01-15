"""Operation weight profiles for GFSL instructions."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Set, Union

from .enums import Operation


class OperationWeights:
    """Optional weights for operations and named groups of operations."""

    def __init__(self, default_weight: Optional[float] = None):
        self.default_weight = default_weight
        self._op_weights: Dict[int, float] = {}
        self._groups: Dict[str, Set[int]] = {}
        self._group_weights: Dict[str, float] = {}

    @staticmethod
    def _normalize_op(op_code: Union[int, Operation]) -> int:
        return int(op_code)

    @staticmethod
    def _normalize_group(name: str) -> str:
        clean = name.strip()
        if not clean:
            raise ValueError("Group name must be a non-empty string.")
        return clean

    def set_operation_weight(self, op_code: Union[int, Operation], weight: Optional[float]) -> None:
        """Assign or clear an operation-specific weight."""
        code = self._normalize_op(op_code)
        if weight is None:
            self._op_weights.pop(code, None)
            return
        self._op_weights[code] = float(weight)

    def get_operation_weight(
        self, op_code: Union[int, Operation], default: Optional[float] = None
    ) -> Optional[float]:
        """Fetch an explicit operation weight if present."""
        code = self._normalize_op(op_code)
        if code in self._op_weights:
            return self._op_weights[code]
        return default

    def set_group(
        self,
        name: str,
        op_codes: Iterable[Union[int, Operation]],
        *,
        weight: Optional[float] = None,
    ) -> None:
        """Define or replace a group of operations, optionally setting its weight."""
        key = self._normalize_group(name)
        self._groups[key] = {self._normalize_op(op) for op in op_codes}
        if weight is not None:
            self._group_weights[key] = float(weight)

    def add_to_group(
        self, name: str, op_codes: Iterable[Union[int, Operation]]
    ) -> None:
        """Add operations to an existing or new group."""
        key = self._normalize_group(name)
        members = self._groups.setdefault(key, set())
        members.update(self._normalize_op(op) for op in op_codes)

    def remove_from_group(
        self, name: str, op_codes: Iterable[Union[int, Operation]]
    ) -> None:
        """Remove operations from a group if present."""
        key = self._normalize_group(name)
        members = self._groups.get(key)
        if not members:
            return
        for op in op_codes:
            members.discard(self._normalize_op(op))

    def group_members(self, name: str) -> Set[int]:
        """Return the operation codes registered under a group name."""
        key = self._normalize_group(name)
        return set(self._groups.get(key, set()))

    def set_group_weight(self, name: str, weight: Optional[float]) -> None:
        """Assign or clear a group weight."""
        key = self._normalize_group(name)
        if weight is None:
            self._group_weights.pop(key, None)
            return
        self._group_weights[key] = float(weight)

    def get_group_weight(self, name: str, default: Optional[float] = None) -> Optional[float]:
        """Fetch a group weight if present."""
        key = self._normalize_group(name)
        if key in self._group_weights:
            return self._group_weights[key]
        return default

    @staticmethod
    def _reduce_group_weights(weights: Iterable[float], mode: str) -> float:
        items = list(weights)
        if not items:
            raise ValueError("No group weights provided for reduction.")
        key = mode.strip().lower()
        if key == "mean":
            return sum(items) / len(items)
        if key == "min":
            return min(items)
        if key == "max":
            return max(items)
        if key == "sum":
            return sum(items)
        raise ValueError(
            "Unknown group_reduce mode. Use 'mean', 'min', 'max', or 'sum'."
        )

    def resolve_weight(
        self,
        op_code: Union[int, Operation],
        *,
        default: Optional[float] = None,
        group_reduce: str = "mean",
    ) -> Optional[float]:
        """
        Resolve a weight for an operation code, falling back to group weights or defaults.
        """
        code = self._normalize_op(op_code)
        if code in self._op_weights:
            return self._op_weights[code]

        group_weights = [
            weight
            for name, members in self._groups.items()
            if code in members and (weight := self._group_weights.get(name)) is not None
        ]
        if group_weights:
            return self._reduce_group_weights(group_weights, group_reduce)

        if default is not None:
            return default
        return self.default_weight


__all__ = ["OperationWeights"]
