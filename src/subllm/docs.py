"""Generated documentation support for SubLLM metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from subllm.model_registry import _PROVIDERS
from subllm.types import CompletionRequest

README_PATH = Path(__file__).resolve().parents[2] / "README.md"
README_MANAGED_START = "<!-- BEGIN GENERATED SECTION: registry-docs -->"
README_MANAGED_END = "<!-- END GENERATED SECTION: registry-docs -->"


@dataclass(frozen=True)
class DocsValidationResult:
    valid: bool
    target: Path


def render_readme_managed_section() -> str:
    supported_fields = ", ".join(
        f"`{field}`" for field in sorted(CompletionRequest.supported_fields)
    )
    unsupported_fields = ", ".join(
        f"`{field}`"
        for field in (
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "response_format",
            "logprobs",
            "top_logprobs",
            "n",
            "metadata",
            "modalities",
            "audio",
            "store",
            "user",
            "reasoning",
        )
    )
    lines = [
        "## Supported Chat Completions Subset",
        "",
        "SubLLM implements a strict subset of the OpenAI chat completions contract.",
        "",
        f"- Accepted request fields: {supported_fields}",
        "- Supported message roles: `system`, `user`, `assistant`",
        "- Supported endpoints: `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`",
        (
            "- Unsupported chat-completions fields are rejected explicitly. "
            f"Common examples: {unsupported_fields}"
        ),
        "",
        "## Available Models",
        "",
        "| Model ID | Backend | Auth |",
        "| --- | --- | --- |",
    ]
    for provider in _PROVIDERS:
        for model in provider.models:
            lines.append(
                f"| `{model.id}` | {model.backend_name} via `{provider.backend}` | "
                f"{model.auth_description} |"
            )
    lines.extend(
        [
            "",
            "## Provider Capabilities",
            "",
            (
                "| Provider | Streaming | Sessions | System Prompt | Vision | "
                "Context Window | Auth Modes | Backend |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for provider in _PROVIDERS:
        capabilities = provider.capabilities
        auth_modes: list[str] = []
        if capabilities.subscription_auth:
            auth_modes.append("subscription")
        if capabilities.api_key_auth:
            auth_modes.append("api_key")
        lines.append(
            "| "
            + " | ".join(
                [
                    provider.name,
                    _render_bool(capabilities.supports_streaming),
                    _render_bool(capabilities.supports_sessions),
                    _render_bool(capabilities.supports_system_prompt),
                    _render_bool(capabilities.supports_vision),
                    f"{capabilities.max_context_tokens:,}",
                    ", ".join(auth_modes),
                    f"`{provider.backend}`",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_readme() -> str:
    readme = README_PATH.read_text()
    generated = f"{README_MANAGED_START}\n{render_readme_managed_section()}\n{README_MANAGED_END}"
    if README_MANAGED_START not in readme or README_MANAGED_END not in readme:
        raise ValueError("README generated section markers are missing")
    before, remainder = readme.split(README_MANAGED_START, 1)
    _, after = remainder.split(README_MANAGED_END, 1)
    return f"{before}{generated}\n\n{after.lstrip()}"


def validate_readme() -> DocsValidationResult:
    current = README_PATH.read_text()
    return DocsValidationResult(valid=current == render_readme(), target=README_PATH)


def write_readme() -> Path:
    updated = render_readme()
    README_PATH.write_text(updated)
    return README_PATH


def _render_bool(value: bool) -> str:
    return "yes" if value else "no"
