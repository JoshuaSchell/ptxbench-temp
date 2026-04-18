from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import textwrap

from .config import (
    DEFAULT_AGENTIC_MAX_STEPS,
    DEFAULT_ARCH,
    REPO_ROOT,
)
from .dataset import Problem
from .run_metadata import sha256_text


PTX_ONE_SHOT_EXAMPLE = textwrap.dedent(
    """
    # Reference
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, a, b):
            return a + b

    def get_inputs():
        a = torch.randn(1024, device="cuda")
        b = torch.randn(1024, device="cuda")
        return [a, b]

    def get_init_inputs():
        return []

    # Submission
    import torch
    import torch.nn as nn
    from ptxbench.runtime import PTXModuleRunner
    from ptxbench.spec import PTXKernelSpec

    PTX_SOURCES = {
        "add": r\"\"\"
    .version 8.0
    .target sm_89
    .address_size 64

    .visible .entry add_kernel(
        .param .u64 a_ptr,
        .param .u64 b_ptr,
        .param .u64 out_ptr,
        .param .u32 n
    )
    {
        .reg .pred %p<2>;
        .reg .b32 %r<6>;
        .reg .b64 %rd<7>;
        .reg .f32 %f<4>;

        ld.param.u64 %rd1, [a_ptr];
        ld.param.u64 %rd2, [b_ptr];
        ld.param.u64 %rd3, [out_ptr];
        ld.param.u32 %r1, [n];

        mov.u32 %r2, %tid.x;
        mov.u32 %r3, %ctaid.x;
        mov.u32 %r4, %ntid.x;
        mad.lo.s32 %r5, %r3, %r4, %r2;
        setp.ge.u32 %p1, %r5, %r1;
        @%p1 bra DONE;

        mul.wide.u32 %rd4, %r5, 4;
        add.s64 %rd5, %rd1, %rd4;
        add.s64 %rd6, %rd2, %rd4;
        add.s64 %rd4, %rd3, %rd4;
        ld.global.f32 %f1, [%rd5];
        ld.global.f32 %f2, [%rd6];
        add.f32 %f3, %f1, %f2;
        st.global.f32 [%rd4], %f3;

    DONE:
        ret;
    }
    \"\"\"
    }

    PTX_KERNELS = {
        "add": PTXKernelSpec(
            entry="add_kernel",
            grid=lambda a, b, out, n: ((int((n + 255) // 256), 1, 1)),
            block=(256, 1, 1),
            arg_types=("tensor", "tensor", "tensor", "uint32"),
        ),
    }

    class ModelNew(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.runner = PTXModuleRunner(PTX_SOURCES, PTX_KERNELS)

        def forward(self, a, b):
            out = torch.empty_like(a)
            self.runner.launch("add", a, b, out, a.numel())
            return out
    """
).strip()


CUDA_ONE_SHOT_EXAMPLE = textwrap.dedent(
    """
    import torch
    import torch.nn as nn
    from torch.utils.cpp_extension import load_inline

    CUDA_SRC = r\"\"\"
    #include <torch/extension.h>
    #include <cuda_runtime.h>

    __global__ void add_kernel(const float* a, const float* b, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = a[idx] + b[idx];
        }
    }

    torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
        auto out = torch::zeros_like(a);
        int n = static_cast<int>(a.numel());
        int block = 256;
        int grid = (n + block - 1) / block;
        add_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
        return out;
    }
    \"\"\"

    module = load_inline(
        name="ptxbench_cuda_add",
        cpp_sources="torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);",
        cuda_sources=CUDA_SRC,
        functions=["add_cuda"],
        verbose=False,
    )

    class ModelNew(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, a, b):
            return module.add_cuda(a, b)
    """
).strip()


@dataclass(frozen=True)
class GenerationRequest:
    backend: str
    model: str
    run_name: str
    temperature: float
    max_tokens: int


def build_generation_prompt(
    problem: Problem,
    backend: str,
    arch: str = DEFAULT_ARCH,
    *,
    track: str = "oneshot",
) -> str:
    if track == "agentic":
        return build_agentic_step_prompt(
            problem,
            backend=backend,
            arch=arch,
            step_index=1,
            max_steps=DEFAULT_AGENTIC_MAX_STEPS,
            max_tool_calls=0,
        )
    return build_oneshot_generation_prompt(problem, backend=backend, arch=arch)


def build_oneshot_generation_prompt(problem: Problem, backend: str, arch: str = DEFAULT_ARCH) -> str:
    backend = backend.lower()
    if backend not in {"ptx", "cuda"}:
        raise ValueError(f"Unsupported backend: {backend}")

    example = PTX_ONE_SHOT_EXAMPLE if backend == "ptx" else CUDA_ONE_SHOT_EXAMPLE
    contract = (
        "Output only valid Python source code for a module that defines ModelNew."
        if backend == "cuda"
        else (
            "Output only valid Python source code for a module that defines ModelNew, "
            "PTX_SOURCES, and PTX_KERNELS. Use PTXModuleRunner and PTXKernelSpec only."
        )
    )
    optimization_hint = (
        f"Target NVIDIA {arch}. You may use architecture-specific PTX instructions when beneficial."
        if backend == "ptx"
        else f"Target NVIDIA {arch} with CUDA C++ kernels."
    )
    return textwrap.dedent(
        f"""
        You are participating in PTXBench.

        Goal:
        - Optimize the provided KernelBench-style PyTorch model.
        - Preserve functional correctness on fp32 inputs.
        - Make the generated implementation faster than the PyTorch eager baseline.
        - {optimization_hint}

        Rules:
        - {contract}
        - Do not use try/except fallbacks.
        - Do not call high-level torch compute ops inside ModelNew as a fallback path.
        - Do not modify timers, streams, or thread scheduling to game the benchmark.

        One-shot example:
        {example}

        Problem to optimize:
        {problem.code}
        """
    ).strip()


def build_agentic_step_prompt(
    problem: Problem,
    *,
    backend: str,
    arch: str = DEFAULT_ARCH,
    step_index: int,
    max_steps: int,
    max_tool_calls: int,
    previous_source: str | None = None,
    previous_observation: str | None = None,
) -> str:
    backend = backend.lower()
    if backend not in {"ptx", "cuda"}:
        raise ValueError(f"Unsupported backend: {backend}")
    contract = (
        "Output only valid Python source code for a module that defines ModelNew."
        if backend == "cuda"
        else (
            "Output only valid Python source code for a module that defines ModelNew, "
            "PTX_SOURCES, and PTX_KERNELS. Use PTXModuleRunner and PTXKernelSpec only."
        )
    )
    benchmark_rules = [
        "You are participating in the PTXBench agentic track.",
        f"This is step {step_index} of {max_steps}.",
        f"You have at most {max_tool_calls} benchmark validation rounds before the final step.",
        "Always return a complete replacement Python module only.",
        "Do not include Markdown fences or explanations.",
        contract,
        "Do not use try/except fallbacks.",
        "Do not call high-level torch compute ops inside ModelNew as a fallback path.",
        "Do not modify timers, streams, or thread scheduling to game the benchmark.",
    ]
    if backend == "ptx":
        benchmark_rules.append(f"Target pure PTX for NVIDIA {arch}. Do not emit CUDA, Triton, or inline fallback kernels.")
    else:
        benchmark_rules.append(f"Target matched CUDA C++ kernels for NVIDIA {arch}.")

    sections = [
        "You are participating in PTXBench.",
        "",
        "Agentic track rules:",
        "\n".join(f"- {rule}" for rule in benchmark_rules),
        "",
        "Optimization goals:",
        "- Preserve functional correctness on fp32 inputs.",
        "- Beat the PyTorch eager baseline when possible.",
        "",
        "Reference problem:",
        problem.code,
    ]
    if previous_source:
        sections.extend(
            [
                "",
                "Previous submission:",
                previous_source,
            ]
        )
    if previous_observation:
        sections.extend(
            [
                "",
                "Benchmark feedback from the previous submission:",
                previous_observation,
            ]
        )
    return "\n".join(sections).strip()


def prompt_template_hash(backend: str, arch: str = DEFAULT_ARCH, *, track: str = "oneshot") -> str:
    placeholder_problem = Problem(
        problem_id=0,
        level=1,
        name="placeholder.py",
        path=Path("placeholder.py"),
        code="class Model:\n    pass\n\ndef get_inputs():\n    return []\n\ndef get_init_inputs():\n    return []\n",
    )
    if track == "agentic":
        template = build_agentic_step_prompt(
            placeholder_problem,
            backend=backend,
            arch=arch,
            step_index=1,
            max_steps=DEFAULT_AGENTIC_MAX_STEPS,
            max_tool_calls=0,
            previous_source="class ModelNew:\n    pass\n",
            previous_observation="static_check: pass",
        )
    else:
        template = build_oneshot_generation_prompt(placeholder_problem, backend=backend, arch=arch)
    return sha256_text(template)


def extract_python_source(response_text: str) -> str:
    fenced = re.findall(r"```(?:python)?\s*([\s\S]*?)```", response_text)
    if fenced:
        return fenced[-1].strip()
    return response_text.strip()


def write_generation_artifacts(
    output_path: Path,
    prompt: str,
    response_text: str,
    extracted_source: str,
    metadata: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(extracted_source, encoding="utf-8")
    artifact_path = output_path.with_suffix(".meta.json")
    artifact_path.write_text(
        json.dumps(
            {
                "prompt": prompt,
                "raw_response": response_text,
                "metadata": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def generation_failure_path(output_path: Path) -> Path:
    return output_path.with_suffix(".failure.json")


def write_generation_failure(
    output_path: Path,
    *,
    prompt: str,
    metadata: dict,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path = generation_failure_path(output_path)
    failure_path.write_text(
        json.dumps(
            {
                "prompt": prompt,
                "metadata": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return failure_path


def clear_generation_failure(output_path: Path) -> None:
    failure_path = generation_failure_path(output_path)
    if failure_path.exists():
        failure_path.unlink()


def default_run_dir(run_name: str, backend: str, level: int) -> Path:
    return REPO_ROOT / "runs" / run_name / backend / f"level{level}"


def default_episode_dir(run_name: str, backend: str, level: int, problem: Problem) -> Path:
    return default_run_dir(run_name, backend, level) / "_episodes" / f"{problem.problem_id:03d}_{problem.path.stem}"
