# SubLLM v2 — Research Analysis & Recommendations

## Executive Summary

After deep-diving into the internals of all three CLI backends (Claude Code, Codex, Gemini CLI), this document maps the **real constraints and opportunities** across three axes: latency reduction, tri-provider support from day one, and feature parity with direct API calls. The conclusion is that SubLLM is viable but needs to be honest about what it is: a **cost-optimized proxy for batch/dev workloads**, not a drop-in replacement for direct API calls. The design should lean into that identity rather than fighting it.

---

## 1. Latency Analysis

### 1.1 Where the Time Actually Goes

The 3-7s figure from v1 was a rough estimate. Here's what the research reveals about actual latency breakdown:

| Component | Claude Code | Codex | Gemini CLI |
|-----------|-------------|-------|------------|
| **CLI cold start (Node.js/binary boot)** | ~600ms ([benchmarked](https://github.com/anthropics/claude-code/issues/8164)) | ~55ms (Rust binary) | ~200-400ms (Node.js) |
| **Auth handshake** | ~100-500ms (OAuth token validation) | ~50-100ms | ~100-300ms |
| **Prompt serialization + IPC** | ~50-100ms | ~50ms | ~50-100ms |
| **Model inference (TTFT)** | 500ms-3s (model-dependent) | 500ms-2s | 300ms-1.5s |
| **Response streaming** | Token-rate dependent | Token-rate dependent | Token-rate dependent |
| **Total cold path** | **~1.5-4.5s** | **~0.7-2.2s** | **~0.7-2.3s** |

**Key finding**: Claude Code's 600ms CLI boot is the outlier. Codex (Rust binary) is 10x faster at startup. Gemini CLI sits in between. The dominant latency is actually **model inference TTFT**, which is the same regardless of whether you're calling via CLI or direct API.

### 1.2 Latency Reduction Strategies

#### Strategy A: Process Pool (Warm CLI Instances)

Keep pre-spawned CLI processes alive and route requests to them, eliminating cold start entirely.

| Dimension | Assessment |
|-----------|------------|
| Cold start elimination | Yes — removes ~600ms (Claude), ~55ms (Codex), ~300ms (Gemini) |
| Implementation complexity | Medium — need process lifecycle management, health checks, stdin/stdout multiplexing |
| Auth persistence | Good — OAuth tokens remain cached in the running process |
| Failure modes | Process crashes require restart; leaked context across requests if not isolated |
| **Net latency improvement** | **~0.3-0.8s saved per request** |

**Verdict**: Meaningful for Claude Code, marginal for Codex. Worth implementing as an optional optimization, not a default.

#### Strategy B: Agent SDK Direct Integration (Claude-specific)

The `claude-agent-sdk` wraps the CLI but maintains an internal connection. Using it avoids subprocess overhead entirely for the Python path.

| Dimension | Assessment |
|-----------|------------|
| Cold start | Still ~2s first call (Node.js boot happens inside SDK), then warm for subsequent calls in same process |
| Streaming granularity | AsyncIterator — better than line-buffered stdout |
| Session persistence | Native `resume` parameter with session IDs |
| **Net latency improvement** | **~0.3-0.6s saved after first call** |

**Verdict**: Best option for Claude Code in long-running Python processes (servers, background workers). Not useful for one-shot scripts.

#### Strategy C: Codex SDK (TypeScript, Bridge Required)

Codex has a first-party TypeScript SDK (`@openai/codex-sdk`) with `thread.run()` that maintains persistent sessions. No Python SDK exists yet (open feature request, issue #2772).

| Dimension | Assessment |
|-----------|------------|
| Option 1 (subprocess bridge) | Works, adds ~100ms IPC overhead, complex to maintain |
| Option 2 (HTTP bridge) | Cleaner interface, ~50ms localhost overhead, but another process to manage |
| Option 3 (wait) | Unknown timeline, but issue #2772 has community momentum |
| **Net latency improvement** | **~0.1-0.3s over raw CLI exec** |

**Verdict**: For v1, stick with `codex exec`. The 55ms Codex cold start means there's almost nothing to optimize. Revisit when the Python SDK ships.

#### Strategy D: Response Caching Layer

Cache identical prompt→response mappings to eliminate inference entirely for repeated queries.

| Dimension | Assessment |
|-----------|------------|
| Hit rate for dev workflows | Low — most prompts are unique |
| Hit rate for batch/template jobs | High — e.g., "summarize this PR" templates |
| Implementation | Redis/SQLite + hash of (model, messages, temperature) |
| **Net latency improvement** | **~0ms for misses, eliminates all latency for hits** |

**Verdict**: High value for batch/CI workflows. Low value for interactive use. Implement as opt-in with TTL.

#### Strategy E: Speculative Prefetch

**Verdict**: Over-engineered for v1. Revisit in v3.

### 1.3 Latency Recommendation Summary

| Strategy | Impact | Effort | Priority |
|----------|--------|--------|----------|
| **B: Agent SDK for Claude** | -0.3-0.6s warm | Medium | **P1 — Phase 2** |
| **A: Process Pool** | -0.3-0.8s | High | **P2 — Phase 3** |
| **D: Response Cache** | Eliminates repeat calls | Low | **P1 — Phase 2** |
| **C: Codex SDK bridge** | -0.1-0.3s | High | **P3 — When Python SDK ships** |
| **E: Speculative Prefetch** | Variable | Very High | **P4 — Future** |

**Honest framing**: Even with all optimizations, SubLLM adds **~0.5-1.5s overhead** beyond raw model inference. This is structural — CLI processes do real work (auth, tool registration, context loading). The right move is to **own this constraint** and position for batch/dev/background workloads, while making streaming smooth enough that interactive use feels responsive.

---

## 2. Tri-Provider Support: Claude Code + Codex + Gemini CLI

### 2.1 Provider Capability Matrix

| Capability | Claude Code CLI | Codex CLI | Gemini CLI |
|------------|----------------|-----------|------------|
| **Headless/non-interactive mode** | `claude --print -p "..."` | `codex exec "..."` | `gemini -p "..."` |
| **Structured output** | `--output-format stream-json` | `--output-format jsonl` | `--output-format json` or `stream-json` |
| **Streaming** | Line-buffered stdout + stream-json | JSONL stdout | JSONL stdout (`stream-json`) |
| **Subscription auth** | OAuth via `claude login` or `CLAUDE_CODE_OAUTH_TOKEN` | ChatGPT OAuth via `codex login` | Google OAuth via `gemini` login + cached credentials |
| **API key auth** | `ANTHROPIC_API_KEY` | `OPENAI_API_KEY` | `GEMINI_API_KEY` |
| **Session resume** | `--resume <session_id>` / `--continue` | `codex resume <session_id>` | `/resume` (interactive) / session files in `~/.gemini/` |
| **Headless session resume** | `--resume <id> --print` ✓ | `codex exec --resume <id>` ✓ | Not natively supported in `-p` mode ✗ |
| **System prompt** | `--system-prompt "..."` | Custom instructions in config | `--system-prompt` / GEMINI.md |
| **Model selection** | `--model <model>` | `--model <model>` | `--model <model>` |
| **Max tokens** | `--max-tokens <n>` | Config-based | Config-based / `--max-output-tokens` |
| **Tool use passthrough** | `--allowed-tools` | `--approval-mode` | Automatic with permissions |
| **SDK availability** | Python (`claude-agent-sdk`) + TS | TypeScript only (`@openai/codex-sdk`) | None (CLI only) |
| **CLI language** | Node.js (JavaScript) | Rust | Node.js (TypeScript) |
| **Free tier** | No | No (limited free trial period) | Yes — 60 req/min, 1000 req/day |
| **Subscription tiers** | Pro $20, Max $100/$200 | Plus $20, Pro $200 | Free, AI Pro, AI Ultra |

### 2.2 Gemini CLI Integration Design

Gemini CLI is the most straightforward to integrate — it has a clean headless mode, structured JSON output, and Google OAuth subscription auth that caches credentials locally.

**Key differences from Claude Code / Codex**:

1. **Auth model**: Google OAuth cached in `~/.gemini/` — simpler than Claude's keychain approach. Free tier available with API key. Google AI Pro/Ultra subscribers use Google account login.

2. **Streaming**: Uses `--output-format stream-json` which emits JSONL events similar to Claude Code's format.

3. **Session resume in headless mode**: This is the gap. Gemini CLI saves sessions automatically but the `-p` headless mode starts a fresh session each time. Multi-turn requires either stateless replay or session file workarounds (fragile, undocumented).

4. **Model availability**: Gemini 2.5 Pro with 1M context, 2.5 Flash — extremely competitive. The free tier (60 req/min) makes this provider uniquely useful for cost-sensitive batch work.

**Gemini model mapping**:
```
gemini/2.5-pro        → gemini-2.5-pro
gemini/2.5-flash      → gemini-2.5-flash
gemini/2.5-pro-exp    → gemini-2.5-pro-exp-03-25
```

### 2.3 Plugin Architecture for Future Providers

The key design principle: **providers declare their capabilities, the router adapts behavior accordingly**. If a provider doesn't support sessions, the router handles stateless multi-turn by replaying message history. If a provider doesn't support streaming, the router wraps the blocking call in an async iterator that yields a single chunk.

For future providers (Cursor CLI, Aider, Windsurf, etc.):
1. Implement `Provider` abstract class
2. Register via `router.register(MyProvider())`
3. Capabilities auto-discovered via `provider.capabilities`

---

## 3. Feature Parity with Direct API Calls

### 3.1 Feature Parity Matrix

| Feature | Direct API | Claude Code CLI | Codex CLI | Gemini CLI | SubLLM Approach |
|---------|-----------|----------------|-----------|------------|-----------------|
| **Single completion** | ✅ Native | ✅ `--print` | ✅ `exec` | ✅ `-p` | ✅ Direct mapping |
| **Streaming** | ✅ SSE | ⚠️ Line-buffered / stream-json | ⚠️ JSONL stdout | ⚠️ stream-json | ⚠️ Normalized to OpenAI SSE, but chunkier |
| **Multi-turn** | ✅ Messages array | ⚠️ Session resume (buggy) | ⚠️ Thread resume | ⚠️ Interactive only | ⚠️ Stateless replay (v1) |
| **System prompt** | ✅ `system` role | ✅ `--system-prompt` | ✅ Config | ✅ `--system-prompt` | ✅ Passthrough |
| **Temperature** | ✅ 0.0-2.0 | ⚠️ Limited | ⚠️ Limited | ⚠️ Config-based | ⚠️ Best-effort |
| **Max tokens** | ✅ Precise | ✅ `--max-tokens` | ⚠️ Approximate | ⚠️ Config-based | ⚠️ Passthrough |
| **Tool use** | ✅ Native | ✅ Built-in + MCP | ✅ Built-in | ✅ Built-in + MCP | ❌ Not feasible |
| **JSON mode** | ✅ Native | ❌ | ❌ | ❌ | ❌ Prompt engineering |
| **Vision** | ✅ Image input | ⚠️ File-based | ⚠️ File-based | ⚠️ File-based | ⚠️ base64→temp file (P2) |
| **Token usage** | ✅ Exact | ⚠️ Approximate | ⚠️ Approximate | ✅ JSON stats | ⚠️ Best-effort |
| **Batch API** | ✅ Batches API | ❌ | ❌ | ❌ | ⚠️ Client-side gather |
| **Rate limits** | ✅ Headers | ❌ Opaque | ❌ Opaque | ❌ Opaque | ❌ No visibility |
| **Prompt caching** | ✅ cache_control | ❌ | ❌ | ⚠️ API key only | ❌ Not accessible |
| **Stop sequences** | ✅ Native | ❌ | ❌ | ❌ | ❌ Not supported |
| **Logprobs** | ✅ Some providers | ❌ | ❌ | ❌ | ❌ Not supported |

### 3.2 Multi-Turn Conversations — The Hard Problem

**Approach 1: Stateless Replay (Recommended for v1)** — Reconstruct the full conversation in each prompt. Universal support, simple, correct. O(n²) token cost acceptable for 3-10 turn conversations SubLLM targets.

**Approach 2: Native Session Resume** — Use each CLI's built-in session management. Claude Code `--resume <id> --print` (buggy per issue #5012), Codex `exec --resume <id>` (untested at scale), Gemini ❌ in headless mode.

**Approach 3: Hybrid (Recommended for v2+)** — Native sessions where available and reliable, stateless replay fallback.

### 3.3 Vision / Multimodal

All three CLIs support file-based input. SubLLM approach: Accept base64/URL in OpenAI-compatible format, write temp files to disk, pass file paths to CLI. Feasible but P2 priority.

### 3.4 Batch Processing

Client-side batch layer via `asyncio.gather` with semaphore. CLI processes run in parallel. Subscription rate limits are opaque — implement exponential backoff and surface warnings.

---

## 4. What SubLLM Is (and Isn't)

### SubLLM IS:
- A cost-optimized proxy for development, prototyping, batch processing, and CI/CD
- A unified interface across Claude Code, Codex, and Gemini CLI with subscription auth
- A LiteLLM-style library that works without API keys
- A way to use $0-200/mo subscriptions for programmatic LLM access
- An OpenAI-compatible local proxy for any tool that speaks the OpenAI protocol

### SubLLM IS NOT:
- A drop-in replacement for direct API calls (structural latency, feature gaps)
- Suitable for real-time chat UIs or latency-sensitive production services
- A way to get tool use / function calling through subscription auth
- A way to access prompt caching, logprobs, or stop sequences
- A multi-tenant SaaS (ToS constraints, especially Anthropic)

### The Honest Pitch:
> "Pay $0-200/mo instead of $200-500/mo in API costs. Trades ~1s extra latency and some advanced features for 5-10x cost reduction. Perfect for dev workflows, batch jobs, and personal automation."

---

## 5. Updated Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [x] Claude Code CLI provider
- [x] Codex CLI provider
- [x] **Gemini CLI provider**
- [x] OpenAI-compatible response format
- [x] Basic streaming
- [ ] Stateless multi-turn via message replay
- [x] Auth status checking
- [x] CLI tool
- [x] **Provider capabilities declaration**

### Phase 2: Optimization (Weeks 3-4)
- [ ] Claude Agent SDK integration (warm process, better streaming)
- [ ] Response cache (SQLite-based, opt-in)
- [ ] Client-side batch executor with concurrency control
- [ ] Vision/multimodal support (base64 → temp file → CLI)
- [ ] Token usage estimation improvements

### Phase 3: Server & Ecosystem (Weeks 5-6)
- [ ] FastAPI proxy server with SSE streaming
- [ ] Process pool for warm CLI instances
- [ ] Hybrid session management (native + replay fallback)
- [ ] LiteLLM custom provider plugin
- [ ] Provider plugin system (`subllm.register_provider()`)

### Phase 4: Integrations (Weeks 7+)
- [ ] Langchain `ChatSubLLM` wrapper
- [ ] LlamaIndex integration
- [ ] Multi-provider fallback chains (try Gemini free → Claude sub → Codex sub)
- [ ] Usage tracking dashboard
- [ ] Codex Python SDK integration (when available)
