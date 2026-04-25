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

Generated package metadata such as `src/*.egg-info/`, plus build artifacts under `build/` and `dist/`, is ignored and should not be committed.

Current vendored task counts:

- Level 1: 100 tasks
- Level 2: 100 tasks
- Level 3: 50 tasks
- Level 4: 20 tasks

## Quick start

```bash
uv sync --extra dev
uv add torch --index pytorch=https://download.pytorch.org/whl/cu128
uv run python scripts/bootstrap_kernelbench.py
uv run pytest -q
uv run python scripts/run_and_check.py --backend ptx --level 1 --problem-id 19 --submission tests/fixtures/submissions/ptx/relu_submission.py --num-correct-trials 1 --num-perf-trials 2
```

`torch` needs a CUDA wheel from the official PyTorch index for local GPU evaluation. A CPU-only wheel will import, but `torch.cuda.is_available()` will stay false and PTXBench evaluation will not run.

The vendored KernelBench task snapshot is pinned to a single upstream commit. PTXBench will fail closed if `vendor/KernelBench-upstream/KernelBench` is missing or if `vendor/KernelBench-upstream` is checked out at a different commit. Re-run `uv run python scripts/bootstrap_kernelbench.py` to clone or reset the pinned snapshot.

Level 4 also pulls in Hugging Face transformer models from the vendored KernelBench tasks. That means Level 4 runs need:

- `transformers` installed
- network access or a pre-populated Hugging Face cache
- enough disk and GPU memory for large pretrained checkpoints

## Generation providers

`scripts/generate_samples.py` can now use either:

- `--provider litellm` to call a normal LiteLLM backend
- `--provider codex` to shell out to a locally installed `codex` CLI
- `--provider claude-code` to shell out to a locally installed Claude Code CLI in print mode

Example:

```bash
uv run python scripts/generate_samples.py --provider codex --model gpt-5.4 --backend ptx --level 1 --run-name codex-ptx
uv run python scripts/generate_samples.py --provider claude-code --claude-bin claude --model claude-sonnet-4-6 --backend ptx --level 1 --run-name claude-ptx
```

When using the local Codex path, PTXBench invokes `codex exec` non-interactively and captures the last assistant message as the generated submission.
GPT-5.5 medium paper specs pass `model_reasoning_effort=medium` through Codex config.
When using Claude Code, PTXBench invokes `claude --print` non-interactively, passes `--model` when provided, and records stdout/stderr plus a secret-free command shape in generation metadata.

## Tracks

PTXBench supports two generation tracks:

- `--track oneshot` keeps the KernelBench-style control path.
- `--track agentic` runs a bounded iterative episode with per-step benchmark feedback, episode artifacts, and held-back final evaluation seeds.

Example agentic generation:

```bash
uv run python scripts/generate_samples.py --provider codex --model gpt-5.4 --track agentic --backend ptx --level 1 --run-name codex-ptx-agentic
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

```bash
uv run python scripts/generate_samples.py --provider codex --model gpt-5.4 --track agentic --backend ptx --level 1 --run-name codex-ptx-agentic --dev-eval-profile --dev-eval-profile-metric gpu__time_duration.sum --dev-eval-profile-metric sm__cycles_active.avg
```

You can also profile a single checked-in submission:

```bash
uv run python scripts/run_and_check.py --backend ptx --level 1 --problem-id 19 --submission tests/fixtures/submissions/ptx/relu_submission.py --profile --profile-metric gpu__time_duration.sum
```

## Locked experiments

Checked-in experiment specs live under [experiments/](./experiments/README.md). Use these to lock the model, track, budgets, seeds, and trials before starting a paper-facing rerun.

Inspect a spec without running it:

```bash
uv run python scripts/run_experiment.py --spec experiments/level1_matched_agentic_gpt54.toml --dry-run
```

Run a locked experiment:

```bash
uv run python scripts/run_experiment.py --spec experiments/level1_matched_agentic_gpt54.toml
```

The experiment directory now contains a paper model matrix for GPT-5.4, GPT-5.5 medium, Claude Sonnet 4.6, and Claude Opus 4.7. Batch files live under `experiments/batches/`:

```bash
uv run python scripts/check_experiment_specs.py
uv run python scripts/run_experiment_batch.py --batch-file experiments/batches/pilot_matrix.txt --dry-run
uv run python scripts/run_experiment_batch.py --batch-file experiments/batches/paper_core_matrix.txt --dry-run
```

Each spec declares whether it is locked and canonical, the claim scope it supports, the official and dev eval seeds, model metadata, reasoning effort, and required outputs.

## Core scripts

- `scripts/generate_samples.py`
- `scripts/run_and_check.py`
- `scripts/bootstrap_kernelbench.py`
- `scripts/eval_from_generations.py`
- `scripts/benchmark_eval_analysis.py`
- `scripts/validate_evidence_bundle.py`
- `scripts/run_experiment.py`
- `scripts/run_level_paired.py`
- `scripts/run_level1_paired.py`

`scripts/run_level_paired.py` is the preferred generic Linux entrypoint for Levels 1-4. `scripts/run_level1_paired.py` remains as a backward-compatible alias.

`scripts/eval_from_generations.py` now supports resumable evaluation by reusing existing per-problem result JSON files unless `--overwrite-existing` is passed.

By default, batch evaluation runs each submission in its own subprocess and enforces a wall-clock timeout per problem:

```bash
uv run python scripts/eval_from_generations.py --run-name codex-ptx --backend ptx --level 1 --per-problem-timeout-seconds 300
```

Useful flags:

- `--per-problem-timeout-seconds 300` sets the per-problem wall-clock limit.
- `--torch-compile-baseline` also measures the reference model with default `torch.compile` and records `ref_runtime_compile_default_ms` plus `speedup_vs_compile_default`, while keeping compile time out of the timed runtime window.
- `--in-process` disables subprocess isolation and uses the legacy in-process path.

## Paper-readiness gate

For a claim-safe KernelBench-style paper bundle, the expected run order is now:

1. Run hygiene and spec checks.
2. Run the pilot matrix.
3. Run the Level 1 matched one-shot matrix.
4. Run the Level 2 spread matrix.
5. Optionally run Level 3 spread.
6. Optionally run agentic pilots.

Then produce paired analysis, deterministic paper report artifacts, and validate the artifact bundle before making claims:

```bash
uv run python scripts/check_experiment_specs.py
uv run python scripts/run_experiment.py --spec experiments/level2_pilot_oneshot_gpt54.toml
uv run python scripts/benchmark_eval_analysis.py --run-name level2-pilot-oneshot-gpt54 --level 2
uv run python scripts/make_paper_report.py --run-name level2-pilot-oneshot-gpt54 --levels 2
uv run python scripts/validate_evidence_bundle.py --run-name level2-pilot-oneshot-gpt54 --level 2 --track oneshot --require-paper-report
```

The validator checks the paper run manifest, backend run manifests, per-problem result JSON files, timing summaries, and paired analysis outputs. It also verifies the standardized evidence fields carried in each result record, including submission hash, failure category, evaluation seeds/trials, hardware/software provenance, and the eager-vs-compile baseline aliases.
For strict paper claims, keep the default torch.compile and PTX-resource requirements enabled. Use `--allow-missing-compile-baseline` or `--allow-missing-ptx-resources` only for legacy or non-paper bundles.

## Vendored KernelBench snapshot

Bootstrap the pinned upstream task set into `vendor/KernelBench-upstream`:

```bash
uv run python scripts/bootstrap_kernelbench.py
```

Verify an existing vendored checkout without fetching:

```bash
uv run python scripts/bootstrap_kernelbench.py --verify-only
```

PTXBench expects the vendored snapshot at commit `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`. Dataset construction and paper-run metadata collection both refuse to proceed if the vendored task set is missing or at any other commit.

## Native paper-run workflow

For paper-grade runs on this machine, use the native Linux environment directly.

1. Run a locked experiment spec:

```bash
uv run python scripts/run_experiment.py --spec experiments/level1_matched_agentic_gpt54.toml
```

2. Or run the paired workflow directly for the level you want:

```bash
uv run python scripts/run_level_paired.py --phase pilot --level 3 --run-name pilot-gpt54-l3 --skip-generation
```

The paired analysis command now writes both JSON and Markdown outputs under `results/analysis/`, including failure breakdowns, correctness rates, and a paper-facing methodology/context section.

For a full paired run, use `--level 1`, `--level 2`, `--level 3`, or `--level 4` on the same wrapper. Level 4 is the heaviest tier and will generally be the slowest to bootstrap because it may download large pretrained models before evaluation.
