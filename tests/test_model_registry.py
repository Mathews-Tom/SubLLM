from __future__ import annotations

from subllm.docs import render_readme, render_readme_managed_section, validate_readme
from subllm.model_registry import (
    all_model_descriptors,
    provider_capabilities,
    provider_model_aliases,
    resolve_provider_model,
)


def test_registry_exposes_registered_models_and_capabilities() -> None:
    assert "claude-code/sonnet-4-5" in [model["id"] for model in all_model_descriptors()]
    assert provider_model_aliases("codex") == [
        "gpt-5.2",
        "gpt-5.2-codex",
        "gpt-4.1",
        "gpt-5-mini",
    ]
    assert resolve_provider_model("claude-code", "sonnet-4-5") == "claude-sonnet-4-5"
    capabilities = provider_capabilities("gemini")
    assert capabilities is not None
    assert capabilities.supports_vision is True
    assert capabilities.supports_sessions is False


def test_readme_generated_section_matches_registry() -> None:
    managed = render_readme_managed_section()

    assert "## Supported Chat Completions Subset" in managed
    assert "## Available Models" in managed
    assert "## Provider Capabilities" in managed
    assert "`codex/gpt-5.2`" in managed


def test_validate_readme_accepts_checked_in_registry_docs() -> None:
    assert validate_readme().valid is True
    rendered = render_readme()
    assert "BEGIN GENERATED SECTION: registry-docs" in rendered
