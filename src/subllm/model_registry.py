"""Authoritative provider and model metadata for SubLLM."""

from __future__ import annotations

from dataclasses import dataclass

from subllm.providers.base import ProviderCapabilities
from subllm.types import ModelDescriptor


@dataclass(frozen=True)
class ModelRegistryEntry:
    provider: str
    alias: str
    provider_model: str
    backend_name: str
    auth_description: str

    @property
    def id(self) -> str:
        return f"{self.provider}/{self.alias}"

    def to_descriptor(self) -> ModelDescriptor:
        return {"id": self.id, "provider": self.provider}


@dataclass(frozen=True)
class ProviderRegistryEntry:
    name: str
    backend: str
    auth_description: str
    capabilities: ProviderCapabilities
    models: tuple[ModelRegistryEntry, ...]


_PROVIDERS: tuple[ProviderRegistryEntry, ...] = (
    ProviderRegistryEntry(
        name="claude-code",
        backend="claude-agent-sdk",
        auth_description="Claude Pro ($20) / Max ($100-200) or Anthropic API key",
        capabilities=ProviderCapabilities(
            supports_streaming=True,
            supports_sessions=True,
            supports_system_prompt=True,
            supports_vision=True,
            max_context_tokens=200_000,
            subscription_auth=True,
            api_key_auth=True,
        ),
        models=(
            ModelRegistryEntry(
                provider="claude-code",
                alias="opus-4-6",
                provider_model="claude-opus-4-6",
                backend_name="Claude Opus 4.6",
                auth_description="Claude Max ($200) or Anthropic API key",
            ),
            ModelRegistryEntry(
                provider="claude-code",
                alias="sonnet-4-5",
                provider_model="claude-sonnet-4-5",
                backend_name="Claude Sonnet 4.5",
                auth_description="Claude Pro ($20) / Max ($100-200) or Anthropic API key",
            ),
            ModelRegistryEntry(
                provider="claude-code",
                alias="haiku-4-5",
                provider_model="claude-haiku-4-5",
                backend_name="Claude Haiku 4.5",
                auth_description="Claude Pro ($20) / Max ($100-200) or Anthropic API key",
            ),
        ),
    ),
    ProviderRegistryEntry(
        name="codex",
        backend="codex exec",
        auth_description="ChatGPT Plus ($20) / Pro ($200) or OPENAI_API_KEY",
        capabilities=ProviderCapabilities(
            supports_streaming=True,
            supports_sessions=True,
            supports_system_prompt=True,
            supports_vision=False,
            max_context_tokens=200_000,
            subscription_auth=True,
            api_key_auth=True,
        ),
        models=(
            ModelRegistryEntry(
                provider="codex",
                alias="gpt-5.2",
                provider_model="gpt-5.2",
                backend_name="GPT-5.2",
                auth_description="ChatGPT Plus ($20) / Pro ($200) or OPENAI_API_KEY",
            ),
            ModelRegistryEntry(
                provider="codex",
                alias="gpt-5.2-codex",
                provider_model="gpt-5.2-codex",
                backend_name="GPT-5.2-Codex",
                auth_description="ChatGPT Plus ($20) / Pro ($200) or OPENAI_API_KEY",
            ),
            ModelRegistryEntry(
                provider="codex",
                alias="gpt-4.1",
                provider_model="gpt-4.1",
                backend_name="GPT-4.1",
                auth_description="ChatGPT Plus ($20) / Pro ($200) or OPENAI_API_KEY",
            ),
            ModelRegistryEntry(
                provider="codex",
                alias="gpt-5-mini",
                provider_model="gpt-5-mini",
                backend_name="GPT-5 Mini",
                auth_description="ChatGPT Plus ($20) / Pro ($200) or OPENAI_API_KEY",
            ),
        ),
    ),
    ProviderRegistryEntry(
        name="gemini",
        backend="gemini -p",
        auth_description="GEMINI_API_KEY, GOOGLE_API_KEY, Google AI Pro, or Google AI Ultra",
        capabilities=ProviderCapabilities(
            supports_streaming=True,
            supports_sessions=False,
            supports_system_prompt=True,
            supports_vision=True,
            max_context_tokens=1_000_000,
            subscription_auth=True,
            api_key_auth=True,
        ),
        models=(
            ModelRegistryEntry(
                provider="gemini",
                alias="gemini-3-pro-preview",
                provider_model="gemini-3-pro-preview",
                backend_name="Gemini 3 Pro Preview",
                auth_description="GEMINI_API_KEY, GOOGLE_API_KEY, Google AI Pro, or Google AI Ultra",
            ),
            ModelRegistryEntry(
                provider="gemini",
                alias="gemini-3-flash-preview",
                provider_model="gemini-3-flash-preview",
                backend_name="Gemini 3 Flash Preview",
                auth_description="GEMINI_API_KEY, GOOGLE_API_KEY, Google AI Pro, or Google AI Ultra",
            ),
        ),
    ),
)

_PROVIDER_INDEX = {provider.name: provider for provider in _PROVIDERS}
_MODEL_INDEX = {
    (model.provider, model.alias): model for provider in _PROVIDERS for model in provider.models
}


def registered_provider_names() -> list[str]:
    return [provider.name for provider in _PROVIDERS]


def provider_registry_entry(provider_name: str) -> ProviderRegistryEntry | None:
    return _PROVIDER_INDEX.get(provider_name)


def provider_capabilities(provider_name: str) -> ProviderCapabilities | None:
    provider = provider_registry_entry(provider_name)
    return provider.capabilities if provider is not None else None


def provider_model_aliases(provider_name: str) -> list[str]:
    provider = provider_registry_entry(provider_name)
    if provider is None:
        return []
    return [model.alias for model in provider.models]


def provider_model_entries(provider_name: str) -> list[ModelRegistryEntry]:
    provider = provider_registry_entry(provider_name)
    if provider is None:
        return []
    return list(provider.models)


def all_model_entries() -> list[ModelRegistryEntry]:
    return [model for provider in _PROVIDERS for model in provider.models]


def all_model_descriptors() -> list[ModelDescriptor]:
    return [model.to_descriptor() for model in all_model_entries()]


def resolve_provider_model(provider_name: str, model_alias: str) -> str | None:
    model = _MODEL_INDEX.get((provider_name, model_alias))
    return model.provider_model if model is not None else None
