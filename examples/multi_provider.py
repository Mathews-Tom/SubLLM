"""Example: SubLLM multi-provider — mix-match Claude Code, Codex, and Gemini in one script."""

from __future__ import annotations

import asyncio
import time

import subllm
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

CLAUDE = "claude-code/sonnet-4-5"
CODEX = "codex/gpt-5.2"
GEMINI = "gemini/gemini-3-flash-preview"
ALL_MODELS = [CLAUDE, CODEX, GEMINI]


def _fmt(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms" if seconds < 1 else f"{seconds:.1f}s"


async def main() -> None:
    total_start = time.perf_counter()
    timings: list[tuple[str, str, float]] = []

    # ── Auth (all providers) ──────────────────────────────
    t0 = time.perf_counter()
    with console.status("[bold cyan]Checking all providers…"):
        statuses = await subllm.check_auth()
    t_auth = time.perf_counter() - t0
    timings.append(("Auth check (all)", "all", t_auth))

    auth_table = Table(title="Auth Status", show_header=True, header_style="bold")
    auth_table.add_column("", width=2)
    auth_table.add_column("Provider")
    auth_table.add_column("Method / Error")
    for s in statuses:
        icon = "[green]✓[/]" if s.authenticated else "[red]✗[/]"
        detail = s.method or s.error or "unknown"
        style = "" if s.authenticated else "dim"
        auth_table.add_row(icon, s.provider, detail, style=style)
    console.print(auth_table)
    console.print(f"[dim]Auth completed in {_fmt(t_auth)}[/]")
    console.print()

    # ── All models ────────────────────────────────────────
    model_table = Table(title="All Available Models", show_header=True, header_style="bold")
    model_table.add_column("Model ID")
    model_table.add_column("Provider", style="dim")
    for m in subllm.list_models():
        model_table.add_row(m["id"], m["provider"])
    console.print(model_table)
    console.print()

    # ── Same prompt, three providers (non-streaming) ──────
    prompt = "What is 2+2? Reply in one sentence."
    result_table = Table(
        title="Same Prompt \u2192 Three Providers",
        show_header=True,
        header_style="bold",
    )
    result_table.add_column("Model")
    result_table.add_column("Response")
    result_table.add_column("Time", style="dim", justify="right")

    for model in ALL_MODELS:
        t0 = time.perf_counter()
        with console.status(f"[bold cyan]{model} — non-streaming…"):
            response = await subllm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
        elapsed = time.perf_counter() - t0
        timings.append(("Non-streaming", model, elapsed))
        result_table.add_row(model, response.choices[0].message.content, _fmt(elapsed))

    console.print(result_table)
    console.print()

    # ── Streaming from each provider ──────────────────────
    for model in ALL_MODELS:
        t0 = time.perf_counter()
        stream = await subllm.completion(
            model=model,
            messages=[{"role": "user", "content": "Write a haiku about Python."}],
            stream=True,
        )

        streamed = Text()
        with Live(
            Panel(streamed, title=f"[bold]Streaming — {model}[/]", border_style="green"),
            console=console,
            refresh_per_second=15,
        ) as live:
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    streamed.append(delta.content)
                    live.update(Panel(
                        streamed,
                        title=f"[bold]Streaming — {model}[/]",
                        subtitle=f"[dim]{_fmt(time.perf_counter() - t0)}[/]",
                        border_style="green",
                    ))
        t_stream = time.perf_counter() - t0
        timings.append(("Streaming", model, t_stream))
        console.print()

    # ── Multi-turn with provider handoff ──────────────────
    conversation: list[dict] = [
        {"role": "user", "content": "Remember the number 42."},
    ]

    t0 = time.perf_counter()
    with console.status(f"[bold cyan]{CLAUDE} — turn 1 (remember)…"):
        r1 = await subllm.completion(model=CLAUDE, messages=conversation)
    t1 = time.perf_counter() - t0
    timings.append(("Multi-turn (turn 1)", CLAUDE, t1))

    conversation.append({"role": "assistant", "content": r1.choices[0].message.content})
    conversation.append({"role": "user", "content": "What number did I ask you to remember?"})

    t0 = time.perf_counter()
    with console.status(f"[bold cyan]{CODEX} — turn 2 (recall)…"):
        r2 = await subllm.completion(model=CODEX, messages=conversation)
    t2 = time.perf_counter() - t0
    timings.append(("Multi-turn (turn 2)", CODEX, t2))

    conversation.append({"role": "assistant", "content": r2.choices[0].message.content})
    conversation.append({"role": "user", "content": "Is that correct? Reply yes or no."})

    t0 = time.perf_counter()
    with console.status(f"[bold cyan]{GEMINI} — turn 3 (verify)…"):
        r3 = await subllm.completion(model=GEMINI, messages=conversation)
    t3 = time.perf_counter() - t0
    timings.append(("Multi-turn (turn 3)", GEMINI, t3))

    turn_table = Table(title="Multi-turn Provider Handoff", show_header=True, header_style="bold")
    turn_table.add_column("Turn")
    turn_table.add_column("Model")
    turn_table.add_column("Response")
    turn_table.add_column("Time", style="dim", justify="right")
    turn_table.add_row("1 (remember)", CLAUDE, r1.choices[0].message.content, _fmt(t1))
    turn_table.add_row("2 (recall)", CODEX, r2.choices[0].message.content, _fmt(t2))
    turn_table.add_row("3 (verify)", GEMINI, r3.choices[0].message.content, _fmt(t3))
    console.print(turn_table)
    console.print()

    # ── Cross-provider batch ──────────────────────────────
    batch_requests = [
        {"model": CLAUDE, "messages": [{"role": "user", "content": "Capital of France? One word."}]},
        {"model": CODEX, "messages": [{"role": "user", "content": "Capital of Japan? One word."}]},
        {"model": GEMINI, "messages": [{"role": "user", "content": "Capital of Brazil? One word."}]},
    ]

    t0 = time.perf_counter()
    with console.status("[bold cyan]Cross-provider batch (3 parallel)…"):
        results = await subllm.batch(batch_requests, concurrency=3)
    t_batch = time.perf_counter() - t0
    timings.append(("Batch (3 parallel)", "cross-provider", t_batch))

    batch_table = Table(title="Cross-provider Batch", show_header=True, header_style="bold")
    batch_table.add_column("Model")
    batch_table.add_column("Response")
    for r in results:
        if isinstance(r, Exception):
            batch_table.add_row("[red]error[/]", str(r))
        else:
            batch_table.add_row(r.model, r.choices[0].message.content)
    console.print(batch_table)
    console.print()

    # ── Timing Summary ────────────────────────────────────
    total_elapsed = time.perf_counter() - total_start

    summary = Table(title="Timing Summary", show_header=True, header_style="bold")
    summary.add_column("Operation")
    summary.add_column("Model")
    summary.add_column("Duration", justify="right")
    for label, model, duration in timings:
        summary.add_row(label, model, _fmt(duration))
    summary.add_section()
    summary.add_row("[bold]Total[/]", "", f"[bold]{_fmt(total_elapsed)}[/]")
    console.print(summary)


if __name__ == "__main__":
    asyncio.run(main())
