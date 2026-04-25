from pathlib import Path
import shutil
from unittest.mock import patch

import pytest

from ptxbench.providers import (
    GenerationProviderError,
    ProviderResponse,
    generate_with_claude_code_cli,
    generate_with_codex_cli,
    generate_with_litellm,
)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


@patch("litellm.completion", return_value=_FakeResponse("```python\nprint('ok')\n```"))
def test_generate_with_litellm_uses_completion(mock_completion) -> None:
    response = generate_with_litellm(
        prompt="hello",
        model="gpt-5.4",
        temperature=0.0,
        max_tokens=10,
    )
    assert isinstance(response, ProviderResponse)
    assert response.content.startswith("```python")
    assert response.metadata["provider"] == "litellm"
    mock_completion.assert_called_once()


@patch("subprocess.run")
def test_generate_with_codex_cli_reads_output_file(mock_run) -> None:
    def _fake_run(command, **kwargs):
        output_index = command.index("--output-last-message") + 1
        Path(command[output_index]).write_text("```python\nprint('codex')\n```", encoding="utf-8")

        class _Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return _Result()

    mock_run.side_effect = _fake_run
    scratch = Path("tests") / ".tmp_provider"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        response = generate_with_codex_cli(
            prompt="hello",
            model="gpt-5.4",
            working_dir=scratch.resolve(),
            codex_bin="codex",
        )
        assert response.content.startswith("```python")
        assert response.metadata["provider"] == "codex"
        invoked = mock_run.call_args.args[0]
        assert invoked[0].endswith("codex.cmd") or invoked[0] == "codex"
        assert invoked[1:3] == ["exec", "-m"]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


@patch("subprocess.run")
def test_generate_with_claude_code_cli_extracts_stdout_and_metadata(mock_run) -> None:
    class _Result:
        returncode = 0
        stdout = "```python\nprint('claude')\n```"
        stderr = "note"

    mock_run.return_value = _Result()

    response = generate_with_claude_code_cli(
        prompt="hello",
        model="claude-sonnet-4-5",
        claude_bin="claude-test",
        extra_args=["--dangerously-skip-permissions"],
        timeout_seconds=30,
    )

    assert response.content.startswith("```python")
    assert response.metadata["provider"] == "claude-code"
    assert response.metadata["model"] == "claude-sonnet-4-5"
    assert response.metadata["claude_bin"] == "claude-test"
    assert response.metadata["stderr"] == "note"
    assert response.metadata["command_shape"][:4] == ["claude-test", "--print", "--model", "<arg>"]
    assert "hello" not in response.metadata["command_shape"]


@patch("subprocess.run")
def test_generate_with_claude_code_cli_nonzero_raises_useful_stderr(mock_run) -> None:
    class _Result:
        returncode = 2
        stdout = ""
        stderr = "auth failed"

    mock_run.return_value = _Result()

    with pytest.raises(GenerationProviderError, match="auth failed"):
        generate_with_claude_code_cli(prompt="hello", model="claude-sonnet-4-5")
