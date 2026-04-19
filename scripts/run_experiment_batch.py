from __future__ import annotations

import argparse
import subprocess
import sys

from ptxbench.experiment_specs import (
    build_experiment_command,
    load_experiment_spec,
    render_experiment_summary,
    resolve_experiment_spec_path,
    shell_render_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple checked-in PTXBench experiment specs sequentially through the native Linux workflow."
    )
    parser.add_argument(
        "--spec",
        action="append",
        required=True,
        help="Experiment spec path or basename under experiments/. Repeat for multiple specs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved commands and exit without running them.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running later specs if an earlier spec fails.",
    )
    args = parser.parse_args()

    failed_specs: list[str] = []
    for raw_spec in args.spec:
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
