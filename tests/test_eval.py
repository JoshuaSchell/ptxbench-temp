import pytest

from ptxbench.eval import _compare_outputs, _measure_compile_default_baseline

torch = pytest.importorskip("torch")


def test_compare_outputs_reports_small_tensor_mismatch_summary() -> None:
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    candidate = torch.tensor([[1.0, 2.0], [3.0, 5.0]])

    ok, message, details = _compare_outputs(reference, candidate, atol=1e-4, rtol=1e-4)

    assert not ok
    assert message is not None
    assert "tensor mismatch at output[1, 1]" in message
    assert "max_abs_diff=1" in message
    assert "max_rel_diff=0.25" in message
    assert "nan(ref=False,cand=False)" in message
    assert "inf(ref=False,cand=False)" in message
    assert details is not None
    assert details["first_bad_index"] == [1, 1]
    assert details["max_abs_diff"] == 1.0
    assert details["max_rel_diff"] == 0.25
    assert details["reference_has_nan"] is False
    assert details["candidate_has_inf"] is False


def test_compare_outputs_reports_shape_mismatch_details() -> None:
    reference = torch.ones((2, 2))
    candidate = torch.ones((2, 3))

    ok, message, details = _compare_outputs(reference, candidate, atol=1e-4, rtol=1e-4)

    assert not ok
    assert message is not None
    assert "shape mismatch at output" in message
    assert "expected (2, 2)" in message
    assert "got (2, 3)" in message
    assert details is not None
    assert details["shape_mismatch"] is True
    assert details["reference_shape"] == [2, 2]
    assert details["candidate_shape"] == [2, 3]


def test_compare_outputs_reports_dtype_mismatch() -> None:
    reference = torch.tensor([1.0, 2.0], dtype=torch.float32)
    candidate = torch.tensor([1.0, 2.0], dtype=torch.float16)

    ok, message, details = _compare_outputs(reference, candidate, atol=1e-4, rtol=1e-4)

    assert not ok
    assert message is not None
    assert "dtype mismatch at output" in message
    assert details is not None
    assert details["dtype_mismatch"] is True
    assert details["reference_dtype"] == "torch.float32"
    assert details["candidate_dtype"] == "torch.float16"


def test_compare_outputs_chunked_large_tensor_keeps_bad_index(monkeypatch) -> None:
    monkeypatch.setattr("ptxbench.eval._tensor_chunk_numel", lambda *values: 4)
    reference = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    candidate = reference.clone()
    candidate[2, 1] += 1.0

    ok, message, details = _compare_outputs(reference, candidate, atol=1e-4, rtol=1e-4)

    assert not ok
    assert message is not None
    assert "output[2, 1]" in message
    assert details is not None
    assert details["first_bad_index"] == [2, 1]


def test_measure_compile_default_baseline_excludes_compile_step(monkeypatch) -> None:
    model = torch.nn.Identity()
    perf_inputs = [torch.tensor([1.0])]
    call_log: list[str] = []

    def fake_compile(module):
        call_log.append("compile")

        def compiled(*args):
            call_log.append("compiled_call")
            return module(*args)

        return compiled

    def fake_time(fn, *, num_warmup, num_trials, device, clear_cache=True):
        call_log.append(f"time:{num_warmup}:{num_trials}:{device}")
        fn()
        return [1.0, 1.2]

    monkeypatch.setattr(torch, "compile", fake_compile, raising=False)
    monkeypatch.setattr("ptxbench.eval.time_callable_cuda_events", fake_time)
    monkeypatch.setattr("ptxbench.eval._cleanup_cuda", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None, raising=False)

    stats, error, oom = _measure_compile_default_baseline(
        model,
        perf_inputs,
        num_warmup=2,
        num_trials=3,
        device="cuda:0",
    )

    assert error is None
    assert oom is False
    assert stats is not None
    assert stats.mean_ms == pytest.approx(1.1)
    assert call_log[0] == "compile"
    assert call_log[1] == "compiled_call"
    assert call_log[2] == "time:2:3:cuda:0"
