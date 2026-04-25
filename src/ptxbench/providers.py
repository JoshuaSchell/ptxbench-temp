from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess
import tempfile


class GenerationProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderResponse:
    content: str
    metadata: dict[str, object]


def _decode_process_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _resolve_codex_bin(codex_bin: str) -> str:
    if os.name != "nt":
        return codex_bin
    if Path(codex_bin).suffix.lower() in {".exe", ".cmd", ".bat"}:
        return codex_bin
    candidates = [
        shutil.which(f"{codex_bin}.cmd"),
        shutil.which(f"{codex_bin}.exe"),
        shutil.which(codex_bin),
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    return codex_bin


def _safe_command_shape(command: list[str]) -> list[str]:
    """Return argv shape only; prompts and environment variables are never included."""
    shape: list[str] = []
    previous_sensitive = False
    sensitive_tokens = ("key", "token", "secret", "password")
    for index, arg in enumerate(command):
        lowered = arg.lower()
        is_sensitive_flag = arg.startswith("-") and any(token in lowered for token in sensitive_tokens)
        if index == 0:
            shape.append(arg)
        elif previous_sensitive:
            shape.append("<redacted>")
        elif "=" in arg and any(token in lowered.split("=", 1)[0] for token in sensitive_tokens):
            shape.append(arg.split("=", 1)[0] + "=<redacted>")
        elif arg.startswith("-"):
            shape.append(arg)
        else:
            shape.append("<arg>")
        previous_sensitive = is_sensitive_flag
    return shape


def generate_with_litellm(
    *,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int | None = None,
) -> ProviderResponse:
    from litellm import completion

    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout_seconds,
    )
    content = response.choices[0].message.content or ""
    return ProviderResponse(
        content=content,
        metadata={
            "provider": "litellm",
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout_seconds": timeout_seconds,
        },
    )


def generate_with_codex_cli(
    *,
    prompt: str,
    model: str,
    working_dir: Path,
    codex_bin: str = "codex",
    sandbox: str = "read-only",
    codex_home: Path | None = None,
    config_overrides: list[str] | None = None,
    extra_writable_dirs: list[Path] | None = None,
    timeout_seconds: int | None = None,
) -> ProviderResponse:
    extra_writable_dirs = extra_writable_dirs or []
    config_overrides = config_overrides or []
    command = [
        _resolve_codex_bin(codex_bin),
        "exec",
        "-m",
        model,
        "--cd",
        str(working_dir),
        "--sandbox",
        sandbox,
        "--skip-git-repo-check",
        "--ephemeral",
        "--color",
        "never",
    ]
    for config_override in config_overrides:
        command.extend(["-c", config_override])
    for writable_dir in extra_writable_dirs:
        command.extend(["--add-dir", str(writable_dir)])

    env = os.environ.copy()
    if codex_home is not None:
        env["CODEX_HOME"] = str(codex_home)

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False, encoding="utf-8") as output_file:
        output_path = Path(output_file.name)

    command.extend(["--output-last-message", str(output_path), "-"])
    try:
        process = subprocess.run(
            command,
            input=prompt.encode("utf-8"),
            capture_output=True,
            cwd=str(working_dir),
            env=env,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise GenerationProviderError(
            "codex exec timed out"
            f"\ntimeout_seconds={timeout_seconds}"
            f"\nstdout={_decode_process_output(exc.stdout).strip()}"
            f"\nstderr={_decode_process_output(exc.stderr).strip()}"
        ) from exc

    try:
        content = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
    finally:
        if output_path.exists():
            output_path.unlink()

    if process.returncode != 0:
        raise GenerationProviderError(
            "codex exec failed"
            f"\nexit_code={process.returncode}"
            f"\nstdout={_decode_process_output(process.stdout).strip()}"
            f"\nstderr={_decode_process_output(process.stderr).strip()}"
        )

    return ProviderResponse(
        content=content,
        metadata={
            "provider": "codex",
            "model": model,
            "sandbox": sandbox,
            "codex_bin": codex_bin,
            "codex_home": str(codex_home) if codex_home is not None else None,
            "timeout_seconds": timeout_seconds,
            "stdout": _decode_process_output(process.stdout).strip(),
            "stderr": _decode_process_output(process.stderr).strip(),
        },
    )


def generate_with_claude_code_cli(
    *,
    prompt: str,
    model: str | None = None,
    working_dir: Path | None = None,
    claude_bin: str = "claude",
    extra_args: list[str] | None = None,
    timeout_seconds: int | None = None,
) -> ProviderResponse:
    extra_args = extra_args or []
    command = [claude_bin, "--print"]
    if model:
        command.extend(["--model", model])
    command.extend(extra_args)

    try:
        process = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            cwd=str(working_dir) if working_dir is not None else None,
            check=False,
            timeout=timeout_seconds,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise GenerationProviderError(
            "claude-code CLI timed out"
            f"\ntimeout_seconds={timeout_seconds}"
            f"\nstdout={_decode_process_output(exc.stdout).strip()}"
            f"\nstderr={_decode_process_output(exc.stderr).strip()}"
        ) from exc

    stdout = _decode_process_output(process.stdout).strip()
    stderr = _decode_process_output(process.stderr).strip()
    if process.returncode != 0:
        raise GenerationProviderError(
            "claude-code CLI failed"
            f"\nexit_code={process.returncode}"
            f"\nstdout={stdout}"
            f"\nstderr={stderr}"
        )

    return ProviderResponse(
        content=stdout,
        metadata={
            "provider": "claude-code",
            "model": model,
            "claude_bin": claude_bin,
            "timeout_seconds": timeout_seconds,
            "stdout": stdout,
            "stderr": stderr,
            "command_shape": _safe_command_shape(command),
        },
    )
