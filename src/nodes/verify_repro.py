"""verify_repro_node — confirms the reproduction test exists and is actually failing."""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ._client import llm, load_prompt

logger = logging.getLogger("nodes.verify_repro")


class ReproVerdict(BaseModel):
    reproduced: bool = Field(
        description="True if a failing test was written and it fails (bug is reproduced)"
    )
    failure_reason: Literal["none", "no_diff", "infra_error", "test_passed", "sandbox_error"] = Field(
        description=(
            "none — reproduction confirmed; "
            "no_diff — no test file was written; "
            "infra_error — test runner could not start; "
            "test_passed — test ran but passed (bug not exposed); "
            "sandbox_error — sandbox crashed"
        )
    )
    summary: str = Field(
        description="One-sentence explanation citing specific evidence"
    )


_chain = (
    ChatPromptTemplate.from_messages([
        ("system", load_prompt("verify_repro")),
        ("human", "{input}"),
    ])
    | llm.with_structured_output(ReproVerdict)
)


def verify_repro_node(state: dict) -> dict:
    """Judge whether the reproduction test exists and is genuinely failing."""
    repro_result = state.get("repro_result", {})
    issue_desc = state.get("issue_desc", "") or state.get("task", "")

    verdict: ReproVerdict = _chain.invoke({
        "input": (
            f"Bug description:\n{issue_desc}\n\n"
            f"Reproduce step result:\n{repro_result}"
        ),
    })

    logger.info(
        "ReproVerdict: reproduced=%s reason=%s — %s",
        verdict.reproduced, verdict.failure_reason, verdict.summary,
    )

    if verdict.reproduced:
        return {"repro_verified": True, "status": "fixing"}

    return {
        "repro_verified": False,
        "repro_retries": state.get("repro_retries", 0) + 1,
        "status": "reproducing",
        "repro_failure_reason": verdict.failure_reason,
    }
