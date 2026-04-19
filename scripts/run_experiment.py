from __future__ import annotations

import argparse
import subprocess
import sys

from ptxbench.experiment_specs import (
    available_experiment_specs,
    build_experiment_command,
    load_experiment_spec,
    render_experiment_summary,
    resolve_experiment_spec_path,
    shell_render_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a checked-in PTXBench experiment spec through the native Linux paired workflow."
    )
    parser.add_argument("--spec", help="Experiment spec path or basename under experiments/.")
    parser.add_argument("--list-specs", action="store_true", help="List available checked-in experiment specs.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command and exit without running it.")
    args = parser.parse_args()

    if args.list_specs:
        for spec_path in available_experiment_specs():
            print(spec_path)
        return

    if not args.spec:
        raise ValueError("--spec is required unless --list-specs is used")

    spec_path = resolve_experiment_spec_path(args.spec)
    spec = load_experiment_spec(spec_path)
    command = build_experiment_command(spec, python_exe=sys.executable)

    print(render_experiment_summary(spec))
    print("")
    print("command:")
    print(shell_render_command(command))

    if args.dry_run:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
