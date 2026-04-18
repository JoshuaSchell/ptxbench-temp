from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import random
import re

from .config import DEFAULT_LEVELS, KERNELBENCH_TASK_ROOT, ensure_vendor_snapshot


LEVEL_REPRESENTATIVE_IDS: dict[int, list[int]] = {
    1: [1, 3, 6, 18, 23, 26, 33, 36, 40, 42, 48, 54, 57, 65, 77, 82, 86, 87],
    2: [1, 2, 8, 18, 23, 28, 33, 43],
    3: [1, 5, 9, 15, 20, 28, 35, 43, 50],
    4: [1, 5, 10, 15, 20],
}

TASK_FAMILY_ORDER = (
    "matmul_or_conv",
    "elementwise",
    "reduction",
    "norm",
    "pooling",
    "attention_or_loss",
)


def infer_task_family_tags(name: str, code: str = "") -> tuple[str, ...]:
    haystack = f"{name}\n{code}".lower()
    if any(token in haystack for token in ("layernorm", "rmsnorm", "batchnorm", "instancenorm", "groupnorm", "norm")):
        return ("norm",)
    if "pool" in haystack:
        return ("pooling",)
    if any(token in haystack for token in ("attention", "softmax", "logsoftmax", "loss", "margin", "crossentropy", "nll")):
        return ("attention_or_loss",)
    if any(token in haystack for token in ("argmax", "argmin", "reduction", "sum_", "mean_", "max_", "min_", "sum ", "mean ", "reduce")):
        return ("reduction",)
    if any(token in haystack for token in ("matmul", "matrix", "conv", "gemm", "bmm")):
        return ("matmul_or_conv",)
    return ("elementwise",)


def get_code_hash(code: str) -> str:
    code = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', "", code)
    cleaned = re.sub(r"#.*$|\s+", "", code, flags=re.MULTILINE)
    return hashlib.md5(cleaned.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Problem:
    problem_id: int
    level: int
    name: str
    path: Path
    code: str

    @property
    def code_hash(self) -> str:
        return get_code_hash(self.code)

    @property
    def hash(self) -> str:
        return self.code_hash

    @property
    def task_family_tags(self) -> tuple[str, ...]:
        return infer_task_family_tags(self.name, self.code)

    @property
    def primary_task_family(self) -> str:
        return self.task_family_tags[0]


class ProblemDataset:
    def __init__(
        self,
        level: int,
        task_root: Path | None = None,
        problem_ids: list[int] | None = None,
        id_range: tuple[int, int] | None = None,
    ) -> None:
        if level not in DEFAULT_LEVELS:
            raise ValueError(f"PTXBench v1 supports levels {DEFAULT_LEVELS}, got {level}")
        self.level = level
        self.task_root = task_root or ensure_vendor_snapshot().task_root
        self.problem_dir = self.task_root / f"level{level}"
        if not self.problem_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {self.problem_dir}")
        requested_ids = self._build_filter_ids(problem_ids=problem_ids, id_range=id_range)
        self._problems: dict[int, Problem] = {}
        for path in sorted(self.problem_dir.glob("*.py")):
            try:
                problem_id = int(path.name.split("_", 1)[0])
            except (IndexError, ValueError):
                continue
            if requested_ids and problem_id not in requested_ids:
                continue
            self._problems[problem_id] = Problem(
                problem_id=problem_id,
                level=level,
                name=path.name,
                path=path,
                code=path.read_text(encoding="utf-8"),
            )

    @staticmethod
    def _build_filter_ids(
        problem_ids: list[int] | None,
        id_range: tuple[int, int] | None,
    ) -> set[int]:
        requested_ids = set(problem_ids or [])
        if id_range is not None:
            start_id, end_id = id_range
            if start_id > end_id:
                start_id, end_id = end_id, start_id
            requested_ids.update(range(start_id, end_id + 1))
        return requested_ids

    def __len__(self) -> int:
        return len(self._problems)

    def __iter__(self):
        for problem_id in self.problem_ids():
            yield self._problems[problem_id]

    def problem_ids(self) -> list[int]:
        return sorted(self._problems)

    def get_problem(self, problem_id: int) -> Problem:
        try:
            return self._problems[problem_id]
        except KeyError as exc:
            raise KeyError(f"Problem {problem_id} not found in level {self.level}") from exc

    def get_problem_by_id(self, problem_id: int) -> Problem:
        try:
            return self.get_problem(problem_id)
        except KeyError as exc:
            raise ValueError(f"Problem {problem_id} not found in level {self.level}") from exc

    def get_problem_ids(self) -> list[int]:
        return self.problem_ids()

    def subset(
        self,
        problem_ids: list[int] | None = None,
        id_range: tuple[int, int] | None = None,
    ) -> "ProblemDataset":
        return ProblemDataset(level=self.level, task_root=self.task_root, problem_ids=problem_ids, id_range=id_range)

    def sample(self, n: int, seed: int = 42) -> "ProblemDataset":
        all_ids = self.problem_ids()
        n = min(n, len(all_ids))
        generator = random.Random(seed)
        return self.subset(problem_ids=sorted(generator.sample(all_ids, n)))

    def representative_subset(self) -> "ProblemDataset":
        return self.subset(problem_ids=LEVEL_REPRESENTATIVE_IDS[self.level])

    def get_representative_subset(self) -> "ProblemDataset":
        return self.representative_subset()


def fetch_ref_arch_from_dataset(dataset: ProblemDataset, problem_id: int) -> tuple[Path, str, str]:
    problem = dataset.get_problem_by_id(problem_id)
    return problem.path, problem.name, problem.code


def construct_dataset(level: int, problem_ids: list[int] | None = None) -> ProblemDataset:
    return ProblemDataset(level=level, problem_ids=problem_ids)
