from pathlib import Path
import shutil
from unittest.mock import patch

from ptxbench.providers import ProviderResponse, generate_with_codex_cli, generate_with_litellm


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
