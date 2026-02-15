"""Example: SubLLM with Gemini CLI provider — single-turn, streaming, multi-turn, batch."""

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

PROVIDER = "gemini"
MODEL = f"{PROVIDER}/gemini-3-flash-preview"
MODEL_ALT = f"{PROVIDER}/gemini-3-pro-preview"


def _fmt(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms" if seconds < 1 else f"{seconds:.1f}s"


async def main() -> None:
    total_start = time.perf_counter()
    timings: list[tuple[str, str, float]] = []

    # ── Auth ──────────────────────────────────────────────
    t0 = time.perf_counter()
    with console.status(f"[bold cyan]Checking {PROVIDER} auth…"):
        status = await subllm.check_auth_provider(PROVIDER)
    t_auth = time.perf_counter() - t0
    timings.append(("Auth check", PROVIDER, t_auth))

    icon = "[green]✓[/]" if status.authenticated else "[red]✗[/]"
    console.print(f"  {icon} {status.provider}: {status.method or status.error} [dim]({_fmt(t_auth)})[/]")
    if not status.authenticated:
        console.print("[red]Auth failed. Run `gemini` and complete Google login first.[/]")
        return
    console.print()

    # ── Models ────────────────────────────────────────────
    model_table = Table(title=f"{PROVIDER} Models", show_header=True, header_style="bold")
    model_table.add_column("Model ID")
    for m in subllm.list_models():
        if m["provider"] == PROVIDER:
            model_table.add_row(m["id"])
    console.print(model_table)
    console.print()

    # ── Single-turn (non-streaming) ───────────────────────
    t0 = time.perf_counter()
    with console.status(f"[bold cyan]{MODEL} — non-streaming…"):
        response = await subllm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "What is 2+2? Reply in one sentence."}],
        )
    t_single = time.perf_counter() - t0
    timings.append(("Non-streaming", MODEL, t_single))

    console.print(Panel(
        response.choices[0].message.content,
        title="[bold]Single-turn (non-streaming)[/]",
        subtitle=f"[dim]{MODEL} · {_fmt(t_single)}[/]",
        border_style="blue",
    ))
    console.print()

    # ── Single-turn (streaming) ───────────────────────────
    t0 = time.perf_counter()
    stream = await subllm.completion(
        model=MODEL,
        messages=[{"role": "user", "content": "Write a haiku about Python."}],
        stream=True,
    )

    streamed = Text()
    with Live(
        Panel(streamed, title="[bold]Single-turn (streaming)[/]", border_style="green"),
        console=console,
        refresh_per_second=15,
    ) as live:
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                streamed.append(delta.content)
                live.update(Panel(
                    streamed,
                    title="[bold]Single-turn (streaming)[/]",
                    subtitle=f"[dim]{MODEL} · {_fmt(time.perf_counter() - t0)}[/]",
                    border_style="green",
                ))
    t_stream = time.perf_counter() - t0
    timings.append(("Streaming", MODEL, t_stream))
    console.print()

    # ── Multi-turn ────────────────────────────────────────
    conversation: list[dict] = [
        {"role": "user", "content": "Remember the number 42."},
    ]

    t0 = time.perf_counter()
    with console.status(f"[bold cyan]{MODEL} — multi-turn (turn 1)…"):
        r1 = await subllm.completion(model=MODEL, messages=conversation)
    t1 = time.perf_counter() - t0
    timings.append(("Multi-turn (turn 1)", MODEL, t1))

    conversation.append({"role": "assistant", "content": r1.choices[0].message.content})
    conversation.append({"role": "user", "content": "What number did I ask you to remember?"})

    t0 = time.perf_counter()
    with console.status(f"[bold cyan]{MODEL} — multi-turn (turn 2)…"):
        r2 = await subllm.completion(model=MODEL, messages=conversation)
    t2 = time.perf_counter() - t0
    timings.append(("Multi-turn (turn 2)", MODEL, t2))

    turn_table = Table(title="Multi-turn", show_header=True, header_style="bold")
    turn_table.add_column("Turn")
    turn_table.add_column("Response")
    turn_table.add_column("Time", style="dim", justify="right")
    turn_table.add_row("1", r1.choices[0].message.content, _fmt(t1))
    turn_table.add_row("2", r2.choices[0].message.content, _fmt(t2))
    console.print(turn_table)
    console.print()

    # ── Batch ─────────────────────────────────────────────
    batch_requests = [
        {"model": MODEL, "messages": [{"role": "user", "content": "Capital of France? One word."}]},
        {"model": MODEL, "messages": [{"role": "user", "content": "Capital of Japan? One word."}]},
        {"model": MODEL_ALT, "messages": [{"role": "user", "content": "Capital of Brazil? One word."}]},
    ]

    t0 = time.perf_counter()
    with console.status(f"[bold cyan]{PROVIDER} — batch (3 parallel)…"):
        results = await subllm.batch(batch_requests, concurrency=3)
    t_batch = time.perf_counter() - t0
    timings.append(("Batch (3 parallel)", PROVIDER, t_batch))

    batch_table = Table(title="Batch (3 parallel)", show_header=True, header_style="bold")
    batch_table.add_column("Model")
    batch_table.add_column("Response")
    batch_table.add_column("Time", style="dim", justify="right")
    for r in results:
        if isinstance(r, Exception):
            batch_table.add_row("[red]error[/]", str(r), "")
        else:
            batch_table.add_row(r.model, r.choices[0].message.content, "")
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
