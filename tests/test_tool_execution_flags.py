from __future__ import annotations

from subllm.providers.claude_code import ClaudeCodeProvider
from subllm.providers.codex import CodexProvider
from subllm.providers.gemini_cli import GeminiCLIProvider


def test_codex_cli_args_do_not_enable_full_auto() -> None:
    provider = CodexProvider(cli_path="codex")

    args = provider._build_cli_args("hello", "gpt-5.2")

    assert "--full-auto" not in args
    assert args == ["codex", "exec", "hello", "--model", "gpt-5.2", "--json"]


def test_gemini_cli_args_do_not_enable_yolo_mode() -> None:
    provider = GeminiCLIProvider(cli_path="gemini")

    args = provider._build_cli_args("hello", "gemini-3-flash-preview")

    assert "--yolo" not in args
    assert args == [
        "gemini",
        "-p",
        "hello",
        "--model",
        "gemini-3-flash-preview",
        "--output-format",
        "json",
    ]


def test_claude_sdk_defaults_to_permission_prompts() -> None:
    provider = ClaudeCodeProvider(cli_path="claude")

    options = provider._build_sdk_options("sonnet-4-5")

    assert options.permission_mode == "default"
