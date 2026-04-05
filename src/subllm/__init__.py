"""SubLLM — Route LLM API calls through subscription-authenticated coding agents.

Supports Claude Code, OpenAI Codex, and Gemini CLI as backends.

Usage:
    import subllm

    # Non-streaming
    response = await subllm.completion(
        model="claude-code/sonnet-4-5",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.choices[0].message.content)

    # Streaming
    async for chunk in await subllm.completion(
        model="gemini/gemini-3-flash-preview",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    ):
        print(chunk.choices[0].delta.content, end="")

    # Batch (parallel across providers)
    results = await subllm.batch([
        {"model": "claude-code/sonnet-4-5", "messages": [...]},
        {"model": "gemini/gemini-3-flash-preview", "messages": [...]},
        {"model": "codex/gpt-5.2", "messages": [...]},
    ], concurrency=5)
"""

from __future__ import annotations

from subllm.cache import CacheConfig, ResponseCache
from subllm.errors import (
    AuthenticationError,
    InvalidAttachmentError,
    MalformedRequestError,
    PromptRenderError,
    ProviderFailureError,
    ProviderTimeoutError,
    RateLimitExceededError,
    RequestTimeoutError,
    RequestTooLargeError,
    SubLLMError,
    UnknownModelError,
    UnknownPromptError,
    UnsupportedFeatureError,
)
from subllm.evals import (
    ContractCase,
    ContractResult,
    ContractSuiteResult,
    default_contract_fixture_dir,
    load_contract_cases,
    run_contract_case,
    run_contract_suite,
)
from subllm.model_registry import (
    all_model_descriptors,
    all_model_entries,
    provider_capabilities,
    provider_model_aliases,
    provider_model_entries,
    provider_registry_entry,
    registered_provider_names,
)
from subllm.prompts import (
    PromptRegistry,
    RegisteredPrompt,
    ResolvedPrompt,
    get_prompt_registry,
    list_registered_prompts,
    resolve_prompt,
)
from subllm.providers.base import Provider, ProviderCapabilities
from subllm.router import (
    Router,
    batch,
    check_auth,
    check_auth_provider,
    close,
    completion,
    get_router,
    list_models,
)
from subllm.types import (
    AuthStatus,
    ChatCompletionChunk,
    ChatCompletionResponse,
    CompletionRequest,
    FileContentPart,
    ImageContentPart,
    ImageUrlSpec,
    MessageContentPart,
    PromptReference,
    RequestMessage,
    ResolvedImageInput,
    TextContentPart,
)

__version__ = "0.4.0"

__all__ = [
    "Provider",
    "ProviderCapabilities",
    "CacheConfig",
    "ResponseCache",
    "PromptRegistry",
    "RegisteredPrompt",
    "ResolvedPrompt",
    "ContractCase",
    "ContractResult",
    "ContractSuiteResult",
    "Router",
    "batch",
    "check_auth",
    "check_auth_provider",
    "close",
    "completion",
    "default_contract_fixture_dir",
    "all_model_descriptors",
    "all_model_entries",
    "get_router",
    "get_prompt_registry",
    "load_contract_cases",
    "list_models",
    "list_registered_prompts",
    "provider_capabilities",
    "provider_model_aliases",
    "provider_model_entries",
    "provider_registry_entry",
    "registered_provider_names",
    "resolve_prompt",
    "run_contract_case",
    "run_contract_suite",
    "AuthStatus",
    "ChatCompletionChunk",
    "ChatCompletionResponse",
    "CompletionRequest",
    "AuthenticationError",
    "FileContentPart",
    "ImageContentPart",
    "ImageUrlSpec",
    "InvalidAttachmentError",
    "MessageContentPart",
    "MalformedRequestError",
    "PromptReference",
    "PromptRenderError",
    "ProviderFailureError",
    "ProviderTimeoutError",
    "RateLimitExceededError",
    "RequestMessage",
    "ResolvedImageInput",
    "RequestTimeoutError",
    "RequestTooLargeError",
    "SubLLMError",
    "TextContentPart",
    "UnknownModelError",
    "UnknownPromptError",
    "UnsupportedFeatureError",
]
