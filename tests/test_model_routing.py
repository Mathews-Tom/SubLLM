from __future__ import annotations

import pytest

from subllm.errors import UnknownModelError
from subllm.router import Router


def test_router_rejects_unprefixed_model_ids() -> None:
    router = Router()

    with pytest.raises(UnknownModelError) as exc_info:
        router._resolve("sonnet-4-5")

    assert exc_info.value.param == "model"
    assert "provider prefix" in exc_info.value.message


def test_router_rejects_unknown_provider_model_pair() -> None:
    router = Router()

    with pytest.raises(UnknownModelError) as exc_info:
        router._resolve("codex/not-a-real-model")

    assert exc_info.value.param == "model"
    assert "Supported models for provider 'codex'" in exc_info.value.message


def test_router_resolves_supported_prefixed_model_ids() -> None:
    router = Router()

    provider, model_alias = router._resolve("codex/gpt-5.2")

    assert provider.name == "codex"
    assert model_alias == "gpt-5.2"
