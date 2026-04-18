from ptxbench.config import EXPECTED_KERNELBENCH_COMMIT, detect_vendor_commit
from ptxbench.dataset import construct_dataset


def test_dataset_level_1_loads() -> None:
    dataset = construct_dataset(level=1)
    assert len(dataset) == 100
    assert dataset.get_problem(19).name.endswith("ReLU.py")


def test_dataset_level_2_loads() -> None:
    dataset = construct_dataset(level=2)
    assert len(dataset) == 100
    assert dataset.get_problem(40).name.startswith("40_")


def test_dataset_level_3_loads() -> None:
    dataset = construct_dataset(level=3)
    assert len(dataset) == 50
    assert dataset.get_problem(43).name.startswith("43_")


def test_dataset_level_4_loads() -> None:
    dataset = construct_dataset(level=4)
    assert len(dataset) == 20
    assert dataset.get_problem(20).name.startswith("20_")


def test_vendor_commit_matches_expected_snapshot() -> None:
    assert detect_vendor_commit() == EXPECTED_KERNELBENCH_COMMIT


def test_problem_task_family_tags_cover_known_examples() -> None:
    dataset = construct_dataset(level=1)
    assert dataset.get_problem(1).task_family_tags == ("matmul_or_conv",)
    assert dataset.get_problem(19).task_family_tags == ("elementwise",)
    assert dataset.get_problem(23).task_family_tags == ("attention_or_loss",)
    assert dataset.get_problem(40).task_family_tags == ("norm",)
