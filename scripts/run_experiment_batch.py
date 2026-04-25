from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ptxbench.experiment_specs import (
    build_experiment_command,
    load_experiment_spec,
    render_experiment_summary,
    resolve_experiment_spec_path,
    shell_render_command,
)


def _read_batch_file(path: Path) -> list[str]:
    specs: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        specs.append(line)
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple checked-in PTXBench experiment specs sequentially through the native Linux workflow."
    )
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        help="Experiment spec path or basename under experiments/. Repeat for multiple specs.",
    )
    parser.add_argument(
        "--batch-file",
        action="append",
        default=[],
        help="Text file containing one experiment spec path or basename per non-comment line.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved commands and exit without running them.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running later specs if an earlier spec fails.",
    )
    args = parser.parse_args()

    raw_specs = list(args.spec)
    for batch_file in args.batch_file:
        raw_specs.extend(_read_batch_file(Path(batch_file)))
    if not raw_specs:
        raise ValueError("Provide at least one --spec or --batch-file")

    failed_specs: list[str] = []
    for raw_spec in raw_specs:
        spec_path = resolve_experiment_spec_path(raw_spec)
        spec = load_experiment_spec(spec_path)
        command = build_experiment_command(spec, python_exe=sys.executable)

        print(render_experiment_summary(spec))
        print("")
        print("command:")
        print(shell_render_command(command))
        print("")

        if args.dry_run:
            continue

        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            continue

        failed_specs.append(str(spec_path))
        if not args.continue_on_error:
            raise subprocess.CalledProcessError(result.returncode, command)

    if failed_specs:
        raise SystemExit("failed specs:\n" + "\n".join(failed_specs))


if __name__ == "__main__":
    main()
