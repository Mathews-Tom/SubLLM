"""SubLLM CLI — check auth, list models, run completions, or start proxy server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from ipaddress import ip_address


def _is_local_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


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


def _run_docs(check: bool, write: bool) -> None:
    from subllm.docs import validate_readme, write_readme

    if write:
        path = write_readme()
        print(f"updated {path}")
        return

    result = validate_readme()
    if result.valid:
        print(f"validated {result.target}")
        return

    print(f"out of date: {result.target}")
    if check:
        raise SystemExit(1)


def _run_completion(model: str, prompt: str, stream: bool) -> None:
    from subllm.router import complete_completion, stream_completion

    messages = [{"role": "user", "content": prompt}]

    async def _do() -> None:
        if stream:
            stream_result = stream_completion(model=model, messages=messages)
            async for chunk in stream_result:
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end="", flush=True)
            print()
        else:
            completion_result = await complete_completion(model=model, messages=messages)
            print(completion_result.choices[0].message.content)

    asyncio.run(_do())


def _run_eval_contracts(fixture_dir: str | None, emit_json: bool) -> None:
    from subllm.evals import default_contract_fixture_dir, run_contract_suite

    async def _do() -> None:
        suite = await run_contract_suite(fixture_dir or default_contract_fixture_dir())
        if emit_json:
            print(json.dumps(suite.to_dict(), indent=2, sort_keys=True))
            return

        print(f"passed: {suite.passed}")
        print(f"failed: {suite.failed}")
        for case in suite.cases:
            status = "PASS" if case.passed else "FAIL"
            print(f"{status} {case.provider} {case.mode} {case.name} - {case.detail}")

        if suite.failed:
            raise SystemExit(1)

    asyncio.run(_do())


def _run_server(
    host: str,
    port: int,
    *,
    auth_token: str | None,
    max_request_bytes: int | None,
    request_timeout_seconds: float | None,
    rate_limit_per_minute: int | None,
    trace_export_path: str | None,
    trace_service_name: str | None,
) -> None:
    try:
        from subllm.server import create_app
        from subllm.server_api.settings import ServerSettings
        import uvicorn
    except ImportError:
        print("Server dependencies not installed. Run: uv add subllm[server]")
        sys.exit(1)

    settings = ServerSettings.from_inputs(
        auth_token=auth_token,
        max_request_bytes=max_request_bytes,
        request_timeout_seconds=request_timeout_seconds,
        rate_limit_per_minute=rate_limit_per_minute,
        trace_export_path=trace_export_path or ".subllm/traces.jsonl",
        trace_service_name=trace_service_name,
    )
    if not _is_local_host(host) and settings.auth_token is None:
        print("Refusing to bind a non-local host without a bearer auth token.")
        sys.exit(1)

    app = create_app(settings=settings)
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
    p_complete.add_argument("-m", "--model", default="claude-code/sonnet-4-5", help="Model to use")
    p_complete.add_argument("-s", "--stream", action="store_true", help="Stream output")

    p_eval = sub.add_parser("eval-contracts", help="Run transcript-based provider contract checks")
    p_eval.add_argument("--fixture-dir")
    p_eval.add_argument("--json", action="store_true", dest="emit_json")

    p_docs = sub.add_parser("docs", help="Validate or update generated registry docs")
    p_docs.add_argument("--check", action="store_true")
    p_docs.add_argument("--write", action="store_true")

    p_serve = sub.add_parser("serve", help="Start OpenAI-compatible proxy server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--auth-token")
    p_serve.add_argument("--max-request-bytes", type=int)
    p_serve.add_argument("--request-timeout-seconds", type=float)
    p_serve.add_argument("--rate-limit-per-minute", type=int)
    p_serve.add_argument("--trace-export-path")
    p_serve.add_argument("--trace-service-name")

    args = parser.parse_args()

    if args.command == "auth":
        _run_auth_check()
    elif args.command == "models":
        _run_models()
    elif args.command == "complete":
        _run_completion(args.model, args.prompt, args.stream)
    elif args.command == "eval-contracts":
        _run_eval_contracts(args.fixture_dir, args.emit_json)
    elif args.command == "docs":
        _run_docs(args.check, args.write)
    elif args.command == "serve":
        _run_server(
            args.host,
            args.port,
            auth_token=args.auth_token,
            max_request_bytes=args.max_request_bytes,
            request_timeout_seconds=args.request_timeout_seconds,
            rate_limit_per_minute=args.rate_limit_per_minute,
            trace_export_path=args.trace_export_path,
            trace_service_name=args.trace_service_name,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
