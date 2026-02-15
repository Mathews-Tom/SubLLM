# SubLLM

> Route standard LLM API calls through subscription-authenticated coding agents instead of API keys.

Use your Claude Pro/Max, ChatGPT Plus/Pro, or Google AI Pro/Ultra subscription as the backend for programmatic LLM calls. SubLLM provides a **LiteLLM-style unified interface** that abstracts Claude Code, Codex, and Gemini CLIs behind a standard `completion()` API.

## Why?

| Approach                  | Cost (heavy usage) | Overhead  | Flexibility        |
| ------------------------- | ------------------ | --------- | ------------------ |
| Direct API (per-token)    | $50-500+/mo        | ~0s       | Full control       |
| **SubLLM (subscription)** | **$0-200/mo flat** | ~0.5-1.5s | Good for batch/dev |
| LiteLLM (API keys)        | $50-500+/mo        | ~0s       | Multi-provider     |

SubLLM is ideal for: development, prototyping, batch jobs, CI/CD, personal automation — anywhere the sub-2s CLI overhead is acceptable and cost matters more than latency.

## Quick Start

### 1. Install

```bash
pip install subllm              # Core (CLI subprocess mode)
pip install subllm[server]      # + OpenAI-compatible proxy server
pip install subllm[sdk]         # + Claude Agent SDK integration
```

### 2. Authenticate Your CLIs

**Claude Code** (subscription auth):

```bash
curl -fsSL https://claude.ai/install.sh | bash
claude login
unset ANTHROPIC_API_KEY          # Force subscription auth
```

For headless/CI:

```bash
claude setup-token
export CLAUDE_CODE_OAUTH_TOKEN="your-token-here"
```

**Codex** (subscription auth):

```bash
npm install -g @openai/codex
codex login
```

**Gemini CLI** (free tier or subscription):

```bash
npm install -g @anthropic-ai/gemini-cli
gemini                           # Complete Google login
# Or use free tier with API key:
export GEMINI_API_KEY="your-key"
```

### 3. Use

**Python API:**

```python
import asyncio
import subllm

async def main():
    # Non-streaming
    response = await subllm.completion(
        model="claude-code/sonnet",
        messages=[{"role": "user", "content": "Explain monads in one sentence"}],
    )
    print(response.choices[0].message.content)

    # Streaming
    stream = await subllm.completion(
        model="gemini/flash",
        messages=[{"role": "user", "content": "Write a haiku about Rust"}],
        stream=True,
    )
    async for chunk in stream:
        print(chunk.choices[0].delta.content, end="")

asyncio.run(main())
```

**CLI:**

```bash
subllm auth                                        # Check all providers
subllm models                                      # List available models
subllm complete "What is 2+2?" -m claude-code/sonnet
subllm complete "Write a haiku" -m gemini/flash --stream
```

**OpenAI-compatible proxy:**

```bash
subllm serve --port 8080

# Then use ANY OpenAI-compatible client
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
response = client.chat.completions.create(
    model="claude-code/sonnet",
    messages=[{"role": "user", "content": "hello"}],
)
```

## Available Models

| Model ID              | Backend                  | Auth                              |
| --------------------- | ------------------------ | --------------------------------- |
| `claude-code/sonnet`  | Claude Sonnet 4.5        | Claude Pro ($20) / Max ($100-200) |
| `claude-code/opus`    | Claude Opus 4.5          | Claude Max ($200)                 |
| `claude-code/haiku`   | Claude Haiku 4.5         | Claude Pro ($20) / Max ($100-200) |
| `codex/gpt-5.3`       | GPT-5.3-Codex            | ChatGPT Plus ($20) / Pro ($200)   |
| `codex/gpt-5.3-spark` | GPT-5.3-Codex-Spark      | ChatGPT Pro ($200)                |
| `gemini/2.5-pro`      | Gemini 2.5 Pro (1M ctx)  | Free tier / AI Pro / AI Ultra     |
| `gemini/2.5-flash`    | Gemini 2.5 Flash         | Free tier / AI Pro / AI Ultra     |
| `gemini/flash`        | Gemini 2.5 Flash (alias) | Free tier / AI Pro / AI Ultra     |
| `gemini/pro`          | Gemini 2.5 Pro (alias)   | Free tier / AI Pro / AI Ultra     |

## Architecture

```
User Code ──→ subllm.completion() ──→ Router
                                       ├── ClaudeCodeProvider
                                       │     └── claude --print (subprocess)
                                       │         or claude-agent-sdk (async)
                                       ├── CodexProvider
                                       │     └── codex exec (subprocess)
                                       └── GeminiCLIProvider
                                             └── gemini -p (subprocess)
```

All providers delegate auth entirely to the underlying CLIs. SubLLM never stores or manages tokens directly. Multi-turn conversations use stateless message replay — the full conversation history is sent each turn.

## Batch Processing

```python
results = await subllm.batch([
    {"model": "claude-code/sonnet", "messages": [...]},
    {"model": "gemini/flash", "messages": [...]},
    {"model": "codex/gpt-5.3", "messages": [...]},
], concurrency=5)
```

Runs completions in parallel with a concurrency semaphore. Each provider's CLI handles its own rate limiting internally.

## ToS Notes

- **Anthropic**: Officially prohibits third-party developers from offering claude.ai login for their products. However, the pattern of "user brings their own authenticated CLI" is established by Cline, Zed, and Repo Prompt. **Safe for personal/team use. Don't ship as a SaaS.**
- **OpenAI**: Codex CLI explicitly supports ChatGPT subscription auth and programmatic `exec` mode. **Officially supported pattern.**
- **Google**: Gemini CLI supports Google OAuth and API key auth. Free tier available (60 req/min, 1000 req/day). **Low risk.**

## What SubLLM Is NOT

- A drop-in replacement for direct API calls (structural overhead, no tool use / function calling / prompt caching / logprobs / stop sequences — these features execute inside CLI sandboxes)
- Suitable for real-time chat UIs or latency-sensitive production services
- A multi-tenant SaaS (ToS constraints)

## License

MIT
