from __future__ import annotations

from collections.abc import Callable
import math
import random
import statistics


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n < 0 or successes < 0 or successes > n:
        raise ValueError("successes must satisfy 0 <= successes <= n")
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1.0 + (z * z / n)
    center = (phat + (z * z) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + ((z * z) / (4 * n * n)))
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_ci(
    values: list[float],
    statistic: str | Callable[[list[float]], float],
    n_resamples: int = 10000,
    seed: int = 0,
) -> tuple[float, float]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return (0.0, 0.0)
    stat_fn = _resolve_statistic(statistic)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        resample = [clean[rng.randrange(len(clean))] for _ in range(len(clean))]
        samples.append(float(stat_fn(resample)))
    samples.sort()
    return (_percentile(samples, 0.025), _percentile(samples, 0.975))


def paired_bootstrap_ci(
    left: list[float],
    right: list[float],
    statistic: str | Callable[[list[float]], float] = "mean",
    n_resamples: int = 10000,
    seed: int = 0,
) -> tuple[float, float]:
    if len(left) != len(right):
        raise ValueError("paired bootstrap inputs must have the same length")
    differences = [float(a) - float(b) for a, b in zip(left, right, strict=True)]
    return bootstrap_ci(differences, statistic=statistic, n_resamples=n_resamples, seed=seed)


def _resolve_statistic(statistic: str | Callable[[list[float]], float]) -> Callable[[list[float]], float]:
    if callable(statistic):
        return statistic
    if statistic == "mean":
        return lambda values: sum(values) / len(values)
    if statistic == "median":
        return lambda values: float(statistics.median(values))
    if statistic == "geomean":
        return _geomean
    raise ValueError(f"Unsupported bootstrap statistic: {statistic}")


def _geomean(values: list[float]) -> float:
    positives = [value for value in values if value > 0]
    if not positives:
        return 0.0
    return math.exp(sum(math.log(value) for value in positives) / len(positives))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight
