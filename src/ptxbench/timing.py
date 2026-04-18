from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import gc
import numpy as np


@dataclass(frozen=True)
class TimingStats:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    num_trials: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def clear_l2_cache(device: Any = "cuda") -> None:
    import torch

    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=device)
    dummy.fill_(1)
    del dummy


def summarize_timings(samples_ms: list[float]) -> TimingStats:
    return TimingStats(
        mean_ms=float(np.mean(samples_ms)),
        std_ms=float(np.std(samples_ms)),
        min_ms=float(np.min(samples_ms)),
        max_ms=float(np.max(samples_ms)),
        num_trials=len(samples_ms),
    )


def time_callable_cuda_events(
    fn,
    *,
    num_warmup: int,
    num_trials: int,
    device: Any,
    clear_cache: bool = True,
) -> list[float]:
    import torch

    with torch.cuda.device(device):
        for _ in range(num_warmup):
            output = fn()
            del output
            torch.cuda.synchronize(device=device)

        if clear_cache:
            torch.cuda.empty_cache()
            gc.collect()

        samples_ms: list[float] = []
        for _ in range(num_trials):
            torch.cuda.synchronize(device=device)
            clear_l2_cache(device=device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = fn()
            end.record()
            torch.cuda.synchronize(device=device)
            del output
            samples_ms.append(float(start.elapsed_time(end)))
            torch.cuda.empty_cache()

    return samples_ms
