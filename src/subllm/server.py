"""OpenAI-compatible HTTP server entrypoint for SubLLM."""

from subllm.router import Router
from subllm.server_api.app import create_app

__all__ = ["Router", "create_app"]
