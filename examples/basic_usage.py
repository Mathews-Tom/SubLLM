"""Example: Basic SubLLM usage across all three providers."""

from __future__ import annotations

import asyncio

import subllm


async def main() -> None:
    # ── Check auth status ──────────────────────────────────
    print("=== Auth Status ===")
    statuses = await subllm.check_auth()
    for s in statuses:
        icon = "✓" if s.authenticated else "✗"
        print(f"  {icon} {s.provider}: {s.method or s.error}")
    print()

    # ── List available models ──────────────────────────────
    print("=== Available Models ===")
    for m in subllm.list_models():
        print(f"  {m['id']}")
    print()

    # ── Non-streaming completion ───────────────────────────
    print("=== Non-streaming (Claude Code) ===")
    response = await subllm.completion(
        model="claude-code/sonnet",
        messages=[{"role": "user", "content": "What is 2+2? Reply in one sentence."}],
    )
    print(response.choices[0].message.content)
    print()

    # ── Streaming completion ───────────────────────────────
    print("=== Streaming (Gemini) ===")
    stream = await subllm.completion(
        model="gemini/flash",
        messages=[{"role": "user", "content": "Write a haiku about Python."}],
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
    print("\n")

    # ── Multi-turn conversation ────────────────────────────
    # SubLLM replays the full message history each turn (stateless).
    print("=== Multi-turn (Codex) ===")
    conversation = [
        {"role": "user", "content": "Remember the number 42."},
    ]
    r1 = await subllm.completion(model="codex/gpt-5.2-codex", messages=conversation)
    print(f"Turn 1: {r1.choices[0].message.content}")

    conversation.append({"role": "assistant", "content": r1.choices[0].message.content})
    conversation.append({"role": "user", "content": "What number did I ask you to remember?"})
    r2 = await subllm.completion(model="codex/gpt-5.2-codex", messages=conversation)
    print(f"Turn 2: {r2.choices[0].message.content}")
    print()

    # ── Batch processing ───────────────────────────────────
    print("=== Batch (3 providers in parallel) ===")
    results = await subllm.batch(
        [
            {
                "model": "claude-code/haiku",
                "messages": [{"role": "user", "content": "Capital of France? One word."}],
            },
            {
                "model": "gemini/flash",
                "messages": [{"role": "user", "content": "Capital of Japan? One word."}],
            },
            {
                "model": "codex/gpt-5.2-codex",
                "messages": [{"role": "user", "content": "Capital of Brazil? One word."}],
            },
        ],
        concurrency=3,
    )
    for r in results:
        if isinstance(r, Exception):
            print(f"  Error: {r}")
        else:
            print(f"  {r.model}: {r.choices[0].message.content}")


if __name__ == "__main__":
    asyncio.run(main())
