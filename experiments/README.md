# Experiment Specs

These TOML files are the checked-in experiment contracts for PTXBench on this machine.

## Current locked specs

- `level1_matched_oneshot_gpt54.toml`
  - Same-machine one-shot PTX vs CUDA control run.
  - Use the CUDA side of this run as the KernelBench-style same-machine one-shot control.
- `level1_matched_agentic_gpt54.toml`
  - Same-machine agentic PTX vs CUDA run under a fixed budget.

These two specs are the current **canonical Level 1 experiments** for this machine.

## Locked pilot specs

- `level1_pilot_oneshot_gpt54.toml`
  - Fixed-subset one-shot rehearsal for the canonical Level 1 one-shot run.
- `level1_pilot_agentic_gpt54.toml`
  - Fixed-subset agentic rehearsal for the canonical Level 1 agentic run.

These pilot specs are locked but not canonical. They exist to validate the full evidence path before the full Level 1 runs.

## How to inspect a spec without running it

```powershell
uv run python scripts\run_experiment_wsl.py --spec experiments\level1_matched_agentic_gpt54.toml --dry-run
```

## How to run a locked spec

```powershell
uv run python scripts\run_experiment_wsl.py --spec experiments\level1_matched_agentic_gpt54.toml
```

## Rules

- If a run configuration changes materially, create a new spec file instead of editing an old one in place.
- Treat the spec file, the generated manifests, and the final analysis outputs as the experiment evidence bundle.
- Use the one-shot spec before making one-shot parity claims.
- Use the agentic spec before making claims about whether agents should write PTX or CUDA.
- Use the pilot specs to validate the benchmark machinery, not to make final benchmark claims.
- Canonical specs should set `locked = true` and `canonical = true`, and should declare both claim scope, fixed seeds, and required outputs.
