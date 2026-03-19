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


def verifier_node(state: dict) -> dict:
    """Judge whether the task has been completed successfully."""
    plan_text = "\n".join(
        f"{i+1}. [{s.get('status', '?')}] {s.get('file', '')} — {s.get('description', s)}"
        if isinstance(s, dict) else f"{i+1}. {s}"
        for i, s in enumerate(state.get("plan", []))
    )

    verdict: Verdict = _chain.invoke({
        "input": (
            f"Task:\n{state.get('task', '')}\n\n"
            f"Plan:\n{plan_text}\n\n"
            f"Results so far:\n{state.get('results', [])}"
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
