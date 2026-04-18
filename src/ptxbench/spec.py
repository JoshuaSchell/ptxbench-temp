from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal


ArgType = Literal[
    "tensor",
    "pointer",
    "bool",
    "float32",
    "float64",
    "int32",
    "int64",
    "uint32",
    "uint64",
]
LaunchValue = tuple[int, int, int] | Callable[..., tuple[int, int, int]]
SharedMemValue = int | Callable[..., int]


def _normalize_dim(dim: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(dim) != 3:
        raise ValueError(f"Launch dimensions must be 3-tuples, got {dim}")
    normalized = tuple(int(x) for x in dim)
    if any(x <= 0 for x in normalized):
        raise ValueError(f"Launch dimensions must be positive, got {dim}")
    return normalized


@dataclass(frozen=True)
class PTXKernelSpec:
    entry: str
    grid: LaunchValue
    block: LaunchValue
    arg_types: tuple[ArgType, ...]
    shared_mem: SharedMemValue = 0

    def resolve_grid(self, *args: Any, **kwargs: Any) -> tuple[int, int, int]:
        value = self.grid(*args, **kwargs) if callable(self.grid) else self.grid
        return _normalize_dim(value)

    def resolve_block(self, *args: Any, **kwargs: Any) -> tuple[int, int, int]:
        value = self.block(*args, **kwargs) if callable(self.block) else self.block
        return _normalize_dim(value)

    def resolve_shared_mem(self, *args: Any, **kwargs: Any) -> int:
        value = self.shared_mem(*args, **kwargs) if callable(self.shared_mem) else self.shared_mem
        return int(value)
