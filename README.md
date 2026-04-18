# PTXBench

PTXBench is a PTX-first benchmark scaffold for measuring how well models and agents can author NVIDIA GPU kernels across the KernelBench task family. It supports both a frozen one-shot control track and an agentic iterative track, and compares PTX-authored submissions against both PyTorch eager and matched CUDA-authored submissions.

## What is included

- Pinned local KernelBench snapshot for Levels 1-4
- PTX submission contract built around `ModelNew`, `PTX_SOURCES`, and `PTX_KERNELS`
- PTX assembler + CUDA driver runtime using `ptxas` and `cuda-python`
- Static guardrails against fallback and timing hacks
- One-shot and agentic generation tracks
- Optional Nsight Compute profiler feedback during agentic dev eval
- Single-sample, batch-eval, and analysis scripts with paired PTX-vs-CUDA reporting

Current vendored task counts:

- Level 1: 100 tasks
- Level 2: 100 tasks
- Level 3: 50 tasks
- Level 4: 20 tasks

## Quick start

```powershell
uv sync --extra dev
uv add torch --index pytorch=https://download.pytorch.org/whl/cu128
uv run python scripts\bootstrap_kernelbench.py
uv run pytest -q
uv run python scripts\run_and_check.py --backend ptx --level 1 --problem-id 19 --submission tests\fixtures\submissions\ptx\relu_submission.py --num-correct-trials 1 --num-perf-trials 2
```

`torch` needs a CUDA wheel from the official PyTorch index for local GPU evaluation. A CPU-only wheel will import, but `torch.cuda.is_available()` will stay false and PTXBench evaluation will not run.

The vendored KernelBench task snapshot is pinned to a single upstream commit. PTXBench will fail closed if `vendor/KernelBench-upstream/KernelBench` is missing or if `vendor/KernelBench-upstream` is checked out at a different commit. Re-run `uv run python scripts\bootstrap_kernelbench.py` to clone or reset the pinned snapshot.

For the matched CUDA track on Windows, install the Visual Studio `Desktop development with C++` workload, including the MSVC v143 build tools and a Windows SDK. PTXBench attempts to bootstrap the MSVC environment automatically through `VsDevCmd.bat`; if your install lives in a non-standard location, set `PTXBENCH_VSDEVCMD` to that batch file and `PTXBENCH_MSVC_ROOT` to the corresponding `VC\Tools\MSVC\<version>` directory. PTX-only submissions do not require the MSVC host compiler, but `torch.utils.cpp_extension.load_inline` does.

Level 4 also pulls in Hugging Face transformer models from the vendored KernelBench tasks. That means Level 4 runs need:

- `transformers` installed
- network access or a pre-populated Hugging Face cache
- enough disk and GPU memory for large pretrained checkpoints

## Generation providers

`scripts/generate_samples.py` can now use either:

- `--provider litellm` to call a normal LiteLLM backend
- `--provider codex` to shell out to a locally installed `codex` CLI

Example:

```powershell
uv run python scripts\generate_samples.py --provider codex --model gpt-5.4 --backend ptx --level 1 --run-name codex-ptx
```

When using the local Codex path, PTXBench invokes `codex exec` non-interactively and captures the last assistant message as the generated submission.

## Tracks

PTXBench supports two generation tracks:

- `--track oneshot` keeps the KernelBench-style control path.
- `--track agentic` runs a bounded iterative episode with per-step benchmark feedback, episode artifacts, and held-back final evaluation seeds.

Example agentic generation:

```powershell
uv run python scripts\generate_samples.py --provider codex --model gpt-5.4 --track agentic --backend ptx --level 1 --run-name codex-ptx-agentic
```

The default agentic budget is:

- `max_steps=5`
- `max_wall_clock_minutes=20`
- `max_tool_calls=4`
- `dev_eval_correct_trials=2`
- `dev_eval_perf_trials=5`

Each agentic episode writes per-step prompts, responses, submissions, observations, and an `episode_manifest.json` under `runs/<run_name>/<backend>/level<level>/_episodes/`.

## Optional profiler feedback

PTXBench can optionally attach Nsight Compute metrics to the agentic dev-eval feedback loop. This keeps the normal benchmark timing path unchanged while giving the model profiler-style hardware feedback when the local machine supports it.

Requirements:

- NVIDIA Nsight Compute CLI (`ncu`) on `PATH`
- `nsight-python` installed in the active environment
- local permission to access the requested hardware counters

Example agentic run with profiler feedback enabled:

```powershell
uv run python scripts\generate_samples.py --provider codex --model gpt-5.4 --track agentic --backend ptx --level 1 --run-name codex-ptx-agentic --dev-eval-profile --dev-eval-profile-metric gpu__time_duration.sum --dev-eval-profile-metric sm__cycles_active.avg
```

You can also profile a single checked-in submission:

```powershell
uv run python scripts\run_and_check.py --backend ptx --level 1 --problem-id 19 --submission tests\fixtures\submissions\ptx\relu_submission.py --profile --profile-metric gpu__time_duration.sum
```

## Locked experiments

Checked-in experiment specs live under [experiments/](./experiments/README.md). Use these to lock the model, track, budgets, seeds, trials, and WSL target before starting a paper-facing rerun.

Inspect a spec without running it:

```powershell
uv run python scripts\run_experiment_wsl.py --spec experiments\level1_matched_agentic_gpt54.toml --dry-run
```

Run a locked experiment:

```powershell
uv run python scripts\run_experiment_wsl.py --spec experiments\level1_matched_agentic_gpt54.toml
```

Current checked-in specs:

- `experiments/level1_matched_oneshot_gpt54.toml`
- `experiments/level1_matched_agentic_gpt54.toml`
- `experiments/level1_pilot_oneshot_gpt54.toml`
- `experiments/level1_pilot_agentic_gpt54.toml`

The two `level1_matched_*` files are the current canonical Level 1 experiment contracts for this machine. The two `level1_pilot_*` files are locked pilot rehearsals on the fixed pilot subset. Each spec declares whether it is locked and canonical, the claim scope it supports, the official and dev eval seeds, and the required outputs that make up the evidence bundle.

## Core scripts

- `scripts/generate_samples.py`
- `scripts/run_and_check.py`
- `scripts/bootstrap_kernelbench.py`
- `scripts/eval_from_generations.py`
- `scripts/benchmark_eval_analysis.py`
- `scripts/validate_evidence_bundle.py`
- `scripts/sync_to_wsl.py`
- `scripts/run_level_paired.py`
- `scripts/run_level1_paired.py`
- `scripts/run_level1_paired_wsl.py`

`scripts/run_level_paired.py` is the preferred generic Linux entrypoint for Levels 1-4. `scripts/run_level1_paired.py` remains as a backward-compatible alias.

`scripts/eval_from_generations.py` now supports resumable evaluation by reusing existing per-problem result JSON files unless `--overwrite-existing` is passed.

By default, batch evaluation runs each submission in its own subprocess and enforces a wall-clock timeout per problem:

```powershell
uv run python scripts\eval_from_generations.py --run-name codex-ptx --backend ptx --level 1 --per-problem-timeout-seconds 300
```

Useful flags:

- `--per-problem-timeout-seconds 300` sets the per-problem wall-clock limit.
- `--torch-compile-baseline` also measures the reference model with default `torch.compile` and records `ref_runtime_compile_default_ms` plus `speedup_vs_compile_default`, while keeping compile time out of the timed runtime window.
- `--in-process` disables subprocess isolation and uses the legacy in-process path.

## Paper-readiness gate

For a claim-safe KernelBench-style paper bundle, the expected gate is now:

1. Run paired generation and official evaluation.
2. Produce paired analysis JSON plus Markdown.
3. Validate the artifact bundle before making claims:

```powershell
uv run python scripts\validate_evidence_bundle.py --run-name pilot-gpt54 --level 1 --track oneshot
```

The validator checks the paper run manifest, backend run manifests, per-problem result JSON files, timing summaries, and paired analysis outputs. It also verifies the standardized evidence fields carried in each result record, including submission hash, failure category, evaluation seeds/trials, hardware/software provenance, and the eager-vs-compile baseline aliases.

## Vendored KernelBench snapshot

Bootstrap the pinned upstream task set into `vendor/KernelBench-upstream`:

```powershell
uv run python scripts\bootstrap_kernelbench.py
```

Verify an existing vendored checkout without fetching:

```powershell
uv run python scripts\bootstrap_kernelbench.py --verify-only
```

PTXBench expects the vendored snapshot at commit `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`. Dataset construction and paper-run metadata collection both refuse to proceed if the vendored task set is missing or at any other commit.

## WSL2 paper-run workflow

For paper-grade runs on this machine, use Ubuntu WSL2 as the primary experiment environment and keep native Windows as a secondary/dev path.

1. Sync the repo into the Linux filesystem and include any existing generated runs you want to reevaluate:

```powershell
uv run python scripts\sync_to_wsl.py --include-run pilot-gpt54
```

2. Bootstrap the Linux environment inside the synced repo:

```powershell
wsl bash -lc "cd ~/ptxbench/PTXBench && bash scripts/setup_wsl_benchmark_env.sh"
```

The bootstrap script now installs a Linux `node` if needed so WSL can run the local `codex` CLI shim, and `scripts/run_level1_paired_wsl.py` will reuse your Windows `C:\Users\Josh\.codex` auth state by default.

3. Run the paired workflow for the level you want inside WSL2:

```powershell
uv run python scripts\run_level1_paired_wsl.py --phase pilot --level 3 --run-name pilot-gpt54-l3 --skip-generation
```

The paired analysis command now writes both JSON and Markdown outputs under `results/analysis/`, including failure breakdowns, correctness rates, and a paper-facing methodology/context section.

For a full paired run in WSL2, use `--level 1`, `--level 2`, `--level 3`, or `--level 4` on the same wrapper. Level 4 is the heaviest tier and will generally be the slowest to bootstrap because it may download large pretrained models before evaluation.
