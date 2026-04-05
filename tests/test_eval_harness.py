from __future__ import annotations

import json

import pytest

from subllm.cli import _run_eval_contracts
from subllm.evals import default_contract_fixture_dir, run_contract_suite


@pytest.mark.asyncio
async def test_contract_suite_passes_fixture_cases() -> None:
    result = await run_contract_suite(default_contract_fixture_dir())

    assert result.failed == 0
    assert result.passed == 5


def test_eval_contracts_cli_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    _run_eval_contracts(str(default_contract_fixture_dir()), True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["failed"] == 0
    assert payload["passed"] == 5
    assert {case["name"] for case in payload["cases"]} == {
        "claude-stream-failure",
        "codex-completion-success",
        "codex-stream-failure",
        "gemini-completion-success",
        "gemini-stream-failure",
    }
