from __future__ import annotations

import pandas as pd

from ptxbench.profiler import ProfileRequest, profile_callable_with_nsight


class _FakeResult:
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"Metric": "gpu__time_duration.sum", "AvgValue": 123.0},
                {"Metric": "sm__throughput.avg.pct_of_peak_sustained_elapsed", "AvgValue": 45.0},
            ]
        )


class _FakeAnnotate:
    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeAnalyze:
    def __init__(self) -> None:
        self.decorator_kwargs: dict | None = None

    def kernel(self, **kwargs):
        self.decorator_kwargs = kwargs

        def decorator(func):
            def wrapper(*args, **inner_kwargs):
                assert args == ()
                assert inner_kwargs == {}
                func()
                return _FakeResult()

            return wrapper

        return decorator


class _FakeNsight:
    def __init__(self) -> None:
        self.analyze = _FakeAnalyze()

    def annotate(self, label: str) -> _FakeAnnotate:
        return _FakeAnnotate(label)


def test_profile_callable_with_nsight_invokes_wrapped_function_without_configs(monkeypatch) -> None:
    fake_nsight = _FakeNsight()

    monkeypatch.setattr("ptxbench.profiler.nsight", fake_nsight)
    monkeypatch.setattr("ptxbench.profiler.NSIGHT_PYTHON_AVAILABLE", True)
    monkeypatch.setattr("ptxbench.profiler.check_ncu_available", lambda: True)
    monkeypatch.setattr("torch.cuda.synchronize", lambda: None)

    called = {"count": 0}

    def workload() -> None:
        called["count"] += 1

    result = profile_callable_with_nsight(
        workload,
        request=ProfileRequest(
            enabled=True,
            metrics=(
                "gpu__time_duration.sum",
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            ),
            num_trials=1,
        ),
    )

    assert called["count"] == 1
    assert result.status == "collected"
    assert result.metrics["gpu__time_duration.sum"] == 123.0
    assert result.metrics["sm__throughput.avg.pct_of_peak_sustained_elapsed"] == 45.0
    assert fake_nsight.analyze.decorator_kwargs is not None
    assert fake_nsight.analyze.decorator_kwargs["output"] == "quiet"
