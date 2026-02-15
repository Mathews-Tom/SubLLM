"""Example: Using SubLLM proxy server with any OpenAI-compatible client.

First, start the server:
    subllm serve --port 8080

Then run this script. Works with Langchain, LlamaIndex, Cursor, or
any tool that speaks the OpenAI protocol.
"""

from __future__ import annotations

from openai import OpenAI

# Point OpenAI client at SubLLM proxy
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="unused",  # SubLLM uses CLI auth, not API keys
)

# ── List models ────────────────────────────────────────────
print("Available models:")
for model in client.models.list():
    print(f"  {model.id}")
print()

# ── Non-streaming ──────────────────────────────────────────
print("=== Claude Code via proxy ===")
response = client.chat.completions.create(
    model="claude-code/sonnet",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
)
print(response.choices[0].message.content)
print()

# ── Streaming ──────────────────────────────────────────────
print("=== Gemini via proxy (streaming) ===")
stream = client.chat.completions.create(
    model="gemini/flash",
    messages=[{"role": "user", "content": "Count from 1 to 5."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
