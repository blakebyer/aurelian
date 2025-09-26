"""
Tests for CLI commands execution.

This file contains tests for CLI command execution, including direct query mode.
These tests mock out dependencies to avoid making actual API calls.
"""

from pathlib import Path
import sys
import types

import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from aurelian.cli import main
from aurelian.agents.hpoa import hpoa_agent


@pytest.fixture
def mock_agent_runner():
    """Mock the agent Runner to avoid actual API calls."""
    with patch("aurelian.cli.run_agent") as mock_run:
        mock_run.return_value = None
        yield mock_run


def test_agent_ui_mode(mock_agent_runner):
    """Test running an agent in UI mode."""
    runner = CliRunner()
    result = runner.invoke(main, ["diagnosis", "--ui"])
    assert result.exit_code == 0
    mock_agent_runner.assert_called_once()
    args, kwargs = mock_agent_runner.call_args
    assert kwargs["ui"] is True
    assert kwargs["query"] == ()


def test_agent_direct_query_mode(mock_agent_runner):
    """Test running an agent in direct query mode."""
    runner = CliRunner()
    result = runner.invoke(main, ["diagnosis", "test query"])
    assert result.exit_code == 0
    mock_agent_runner.assert_called_once()
    args, kwargs = mock_agent_runner.call_args
    assert kwargs["ui"] is False
    # Click passes this as a tuple with one string containing the whole query
    assert kwargs["query"] == ("test query",)


def test_chemistry_command(mock_agent_runner):
    """Test the chemistry command specifically since we just fixed it."""
    runner = CliRunner()
    result = runner.invoke(main, ["chemistry", "what is aspirin"])
    assert result.exit_code == 0
    mock_agent_runner.assert_called_once()
    args, kwargs = mock_agent_runner.call_args
    # Check correct parameters are passed 
    assert "chemistry" == kwargs.get("agent_name", args[0])
    # Validate query format
    assert kwargs["query"] == ("what is aspirin",)


def test_datasheets_help():
    """Test the datasheets help, which has URL instead of QUERY."""
    runner = CliRunner()
    result = runner.invoke(main, ["datasheets", "--help"])
    assert result.exit_code == 0
    # Different wording for URL-based command
    assert "Run with a URL for direct mode" in result.output


def test_all_agent_commands_help():
    """Test that all agent commands display help correctly."""
    runner = CliRunner()
    commands = [
        "amigo", "biblio", "checklist", "chemistry", 
        "diagnosis", "gocam", "linkml", "literature", "mapper", 
        "monarch", "phenopackets", "rag", "robot", "ubergraph"
    ]
    
    for command in commands:
        result = runner.invoke(main, [command, "--help"])
        assert result.exit_code == 0, f"Help for {command} failed with {result.output}"
        assert "Run with a query for direct mode" in result.output or "Run with a URL for direct mode" in result.output, \
            f"Missing mode info in {command} help"

def test_hpoa_direct_query_uses_retry(monkeypatch):
    """Ensure the HPOA CLI path routes through call_agent by default."""
    runner = CliRunner()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    fake_pdfminer = types.ModuleType("pdfminer")
    fake_pdfminer_high_level = types.ModuleType("pdfminer.high_level")
    fake_pdfminer_high_level.extract_text = lambda *a, **k: ""
    fake_pdfminer.high_level = fake_pdfminer_high_level
    monkeypatch.setitem(sys.modules, "pdfminer", fake_pdfminer)
    monkeypatch.setitem(sys.modules, "pdfminer.high_level", fake_pdfminer_high_level)

    fake_gradio = types.ModuleType("gradio")

    def _fake_chat_interface(*args, **kwargs):
        return types.SimpleNamespace(launch=lambda **kw: None, fn=kwargs.get("fn"))

    class _FakeChatbot:
        def __init__(self, *args, **kwargs):
            pass

    fake_gradio.ChatInterface = _fake_chat_interface
    fake_gradio.Chatbot = _FakeChatbot
    monkeypatch.setitem(sys.modules, "gradio", fake_gradio)

    fake_logfire = types.ModuleType("logfire")
    fake_logfire.configure = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "logfire", fake_logfire)

    fake_logfire_api = types.ModuleType("logfire_api")
    fake_logfire_api.LogfireSpan = object
    monkeypatch.setitem(sys.modules, "logfire_api", fake_logfire_api)

    calls = {}

    class _DummyResult:
        def __init__(self):
            self.output = "ok"

        def all_messages_json(self):
            return "[]"

        def new_messages(self):
            return []

    async def _fake_call(input, agent=None, deps=None, **kwargs):
        calls["input"] = input
        payload = dict(kwargs)
        payload.setdefault("use_history", None)
        payload.setdefault("agent_variant", None)
        payload["agent"] = agent
        payload["deps"] = deps
        calls["kwargs"] = payload
        return _DummyResult()

    monkeypatch.setattr("aurelian.agents.hpoa.hpoa_agent.call_agent", _fake_call, raising=False)

    result = runner.invoke(main, ["hpoa", "test case"])
    assert calls.get("input") == "test case"
    assert result.exit_code == 0
    kwargs = calls["kwargs"]
    assert kwargs.get("deps") is not None
    assert "use_retry" not in kwargs
    assert kwargs.get("use_history") is None
    assert kwargs.get("agent_variant") == "standard"
    assert kwargs.get("output_path") is None


def test_hpoa_agent_variant_reasoning(monkeypatch):
    runner = CliRunner()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    capture = {}

    class _DummyResult:
        def __init__(self):
            self.output = "ok"

        def all_messages_json(self):
            return "[]"

        def new_messages(self):
            return []

    async def _fake_call(*args, **kwargs):
        payload = dict(kwargs)
        payload.setdefault("use_history", None)
        payload.setdefault("agent_variant", None)
        capture["kwargs"] = payload
        return _DummyResult()

    monkeypatch.setattr("aurelian.agents.hpoa.hpoa_agent.call_agent", _fake_call, raising=False)

    result = runner.invoke(main, ["hpoa", "--agent", "reasoning", "lookup"])
    assert result.exit_code == 0
    assert capture["kwargs"].get("agent_variant") == "reasoning"



def test_hpoa_retry_flag_uses_retry(monkeypatch):
    runner = CliRunner()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    called = {}

    class _DummyResult:
        def __init__(self):
            self.output = "ok"

        def all_messages_json(self):
            return "[]"

        def new_messages(self):
            return []

    async def _fake_retry(*args, **kwargs):
        called["used"] = True
        return _DummyResult()

    async def _should_not_run(*args, **kwargs):
        raise AssertionError("call_agent should not be used when --retry is set")

    monkeypatch.setattr("aurelian.agents.hpoa.hpoa_agent.call_agent_with_retry", _fake_retry, raising=False)
    monkeypatch.setattr("aurelian.agents.hpoa.hpoa_agent.call_agent", _should_not_run, raising=False)

    result = runner.invoke(main, ["hpoa", "--retry", "lookup"])
    assert result.exit_code == 0
    assert called.get("used") is True


def test_hpoa_output_must_be_json(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def _raise(*args, **kwargs):
        raise ValueError('output path must end with .json')

    monkeypatch.setattr(hpoa_agent, '_write_output_to_file', _raise, raising=True)

    result = runner.invoke(main, ["hpoa", "demo query", "--output", "results/out.txt"])
    assert result.exit_code != 0
    assert "must end with .json" in result.output

def test_hpoa_output_option_forwarded(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    capture = {}

    class _DummyResult:
        def __init__(self):
            self.output = {"explanation": "ok", "annotations": []}

        def all_messages_json(self):
            return "[]"

        def new_messages(self):
            return []

    async def _fake_call(*args, **kwargs):
        payload = dict(kwargs)
        payload.setdefault("use_history", None)
        payload.setdefault("agent_variant", None)
        capture["kwargs"] = payload
        return _DummyResult()

    monkeypatch.setattr("aurelian.agents.hpoa.hpoa_agent.call_agent", _fake_call, raising=False)

    output_file = tmp_path / "out.json"
    result = runner.invoke(main, ["hpoa", "demo query", "--output", str(output_file)])
    assert result.exit_code == 0
    assert 'output_path' not in capture['kwargs']
    assert output_file.exists()

