import pytest

from ptxbench.eval import _compare_outputs

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
