# Experiment Specs

These TOML files are checked-in experiment contracts for paper-facing PTXBench runs on this machine. They lock model/provider, track, level, seeds, trials, subsets, and required evidence paths.

## Baseline GPT-5.4

- Canonical full: `level1_matched_oneshot_gpt54.toml`, `level1_matched_agentic_gpt54.toml`, `level2_matched_oneshot_gpt54.toml`
- Locked pilots: `level1_pilot_oneshot_gpt54.toml`, `level1_pilot_agentic_gpt54.toml`, `level2_pilot_oneshot_gpt54.toml`, `level2_pilot_agentic_gpt54.toml`
- Locked exploratory spread: `level2_spread_oneshot_gpt54.toml`, `level3_spread_oneshot_gpt54.toml`

## GPT-5.5 Medium

- Canonical full: `level1_matched_oneshot_gpt55_medium.toml`
- Locked pilot: `level1_pilot_oneshot_gpt55_medium.toml`
- Locked exploratory spread: `level2_spread_oneshot_gpt55_medium.toml`, `level3_spread_oneshot_gpt55_medium.toml`
- Exploratory reasoning ablation: `level1_pilot_oneshot_gpt55_high.toml`

GPT-5.5 specs use provider `codex` and record `reasoning_effort`. The runner translates that to `--codex-config model_reasoning_effort=<value>`.

## Claude Sonnet 4.6

- Canonical full: `level1_matched_oneshot_claude_sonnet46.toml`
- Locked pilot: `level1_pilot_oneshot_claude_sonnet46.toml`
- Locked exploratory spread: `level2_spread_oneshot_claude_sonnet46.toml`, `level3_spread_oneshot_claude_sonnet46.toml`

## Claude Opus 4.7

- Canonical full: `level1_matched_oneshot_claude_opus47.toml`
- Locked pilot: `level1_pilot_oneshot_claude_opus47.toml`
- Locked exploratory spread: `level2_spread_oneshot_claude_opus47.toml`, `level3_spread_oneshot_claude_opus47.toml`

Claude specs use provider `claude-code`, default `claude_bin = "claude"`, and pass `--effort medium --bare --no-session-persistence` through `claude_extra_args`.

## Batches

- `batches/pilot_matrix.txt`: quick Level 1 pilot pass across GPT-5.4, GPT-5.5 medium, Claude Sonnet 4.6, and Claude Opus 4.7.
- `batches/paper_core_matrix.txt`: Level 1 matched plus Level 2 spread one-shot matrix.

Inspect without running:

```bash
uv run python scripts/run_experiment.py --spec experiments/level1_matched_oneshot_gpt55_medium.toml --dry-run
uv run python scripts/run_experiment_batch.py --batch-file experiments/batches/pilot_matrix.txt --dry-run
```

Check all specs:

```bash
uv run python scripts/check_experiment_specs.py
```

## Rules

- Create a new spec file when model, provider, phase, track, subset, seeds, trials, arch, timeout, reasoning effort, or budgets change.
- Use pilot specs to validate machinery, not to make final claims.
- `canonical = true` is reserved for full claim-supporting specs with clear claim scope.
- Spread specs are locked and exploratory, not canonical.
