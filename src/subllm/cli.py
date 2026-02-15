"""SubLLM CLI — check auth, list models, run completions, or start proxy server."""

from __future__ import annotations

import argparse
import asyncio
import sys


def _run_auth_check() -> None:
    from subllm.router import check_auth

    results = asyncio.run(check_auth())
    for status in results:
        icon = "✓" if status.authenticated else "✗"
        method = f" ({status.method})" if status.method else ""
        error = f" — {status.error}" if status.error else ""
        print(f"  {icon} {status.provider}{method}{error}")


def _run_models() -> None:
    from subllm.router import list_models

    for m in list_models():
        print(f"  {m['id']}")


def _run_completion(model: str, prompt: str, stream: bool) -> None:
    from subllm.router import completion

    messages = [{"role": "user", "content": prompt}]

    async def _do():
        if stream:
            result = await completion(model=model, messages=messages, stream=True)
            async for chunk in result:
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end="", flush=True)
            print()
        else:
            result = await completion(model=model, messages=messages)
            print(result.choices[0].message.content)

    asyncio.run(_do())


def _run_server(host: str, port: int) -> None:
    try:
        from subllm.server import create_app
        import uvicorn
    except ImportError:
        print("Server dependencies not installed. Run: uv add subllm[server]")
        sys.exit(1)

    app = create_app()
    print(f"SubLLM proxy server starting on http://{host}:{port}")
    print(f"Use as OpenAI base_url: http://{host}:{port}/v1")
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="subllm",
        description="Route LLM calls through subscription-authenticated coding agents",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("auth", help="Check authentication status for all providers")
    sub.add_parser("models", help="List available models")

    p_complete = sub.add_parser("complete", help="Run a completion")
    p_complete.add_argument("prompt", help="Prompt text")
    p_complete.add_argument("-m", "--model", default="claude-code/sonnet", help="Model to use")
    p_complete.add_argument("-s", "--stream", action="store_true", help="Stream output")

    p_serve = sub.add_parser("serve", help="Start OpenAI-compatible proxy server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()

    if args.command == "auth":
        _run_auth_check()
    elif args.command == "models":
        _run_models()
    elif args.command == "complete":
        _run_completion(args.model, args.prompt, args.stream)
    elif args.command == "serve":
        _run_server(args.host, args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
