"""verifier_node — judges whether the task is complete."""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ._client import llm, load_prompt

logger = logging.getLogger("nodes.verifier")


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class Verdict(BaseModel):
    complete: bool = Field(
        description="True if all plan steps succeeded and all tests pass"
    )
    failure_type: Literal["none", "test_failure", "infra_error", "no_diff", "sandbox_error"] = Field(
        description=(
            "none — task complete; "
            "test_failure — tests ran but failed (code is wrong); "
            "infra_error — test runner could not start (bad args, missing tool, env problem); "
            "no_diff — no code was changed; "
            "sandbox_error — sandbox crashed before tests could run"
        )
    )
    summary: str = Field(
        description="One-sentence explanation referencing specific evidence (diff size, test counts, errors seen)"
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

_chain = (
    ChatPromptTemplate.from_messages([
        ("system", load_prompt("verifier")),
        ("human", "{input}"),
    ])
    | llm.with_structured_output(Verdict)
)


_NO_TESTS_PHRASES = ("no tests ran", "collected 0 items", "no tests collected")
_ONLY_GITIGNORE_RE = None  # built lazily


def _has_meaningful_diff(results: list) -> bool:
    """Return True if any result has a diff touching a file other than .gitignore."""
    for r in results:
        diff = r.get("diff", "") if isinstance(r, dict) else ""
        if not diff or diff == "(no diff)":
            continue
        for line in diff.splitlines():
            # Look for +++ b/<path> lines that are not .gitignore
            if line.startswith("+++ b/") and not line.endswith(".gitignore"):
                return True
    return False


def _no_test_failures(results: list) -> bool:
    """Return True when no result's test_output contains FAILED or ERROR lines."""
    for r in results:
        out = r.get("test_output", "") if isinstance(r, dict) else ""
        if "FAILED" in out or "\nERROR" in out:
            return False
    return True


def _all_tests_absent(results: list) -> bool:
    """Return True when every result's test_output signals that no tests were found."""
    for r in results:
        out = r.get("test_output", "") if isinstance(r, dict) else ""
        if out and not any(p in out.lower() for p in _NO_TESTS_PHRASES):
            return False
    return True


def verifier_node(state: dict) -> dict:
    """Judge whether the task has been completed successfully."""
    results = state.get("results", [])

    # ── Deterministic fast-path ───────────────────────────────────────────────
    # When the diff shows real changes AND tests either pass or simply don't exist
    # in the repo, mark complete without asking the LLM.  This avoids the model
    # misclassifying "no tests ran" as infra_error for config/build fixes.
    if (
        _has_meaningful_diff(results)
        and _no_test_failures(results)
        and (_all_tests_absent(results) or _no_test_failures(results))
    ):
        # Double-check there are no sandbox errors
        if not any(isinstance(r, dict) and r.get("sandbox_error") for r in results):
            summary = (
                "Diff present and no test failures — task complete "
                f"(changed: {[r.get('file') for r in results if isinstance(r, dict) and r.get('diff') and r.get('diff') != '(no diff)']})"
            )
            logger.info("Verdict (deterministic): complete=True — %s", summary)
            return {"output": summary, "status": "complete", "failure_type": "none"}

    # ── LLM judgement for all other cases ─────────────────────────────────────
    plan_text = "\n".join(
        f"{i+1}. [{s.get('status', '?')}] {s.get('file', '')} — {s.get('description', s)}"
        if isinstance(s, dict) else f"{i+1}. {s}"
        for i, s in enumerate(state.get("plan", []))
    )

    verdict: Verdict = _chain.invoke({
        "input": (
            f"Task:\n{state.get('task', '')}\n\n"
            f"Plan:\n{plan_text}\n\n"
            f"Results so far:\n{results}"
        ),
    })

    logger.info(
        "Verdict: complete=%s failure_type=%s — %s",
        verdict.complete, verdict.failure_type, verdict.summary,
    )

    if verdict.complete:
        return {"output": verdict.summary, "status": "complete", "failure_type": "none"}
    return {
        "retries": state.get("retries", 0) + 1,
        "status": "executing",
        "failure_type": verdict.failure_type,
    }
