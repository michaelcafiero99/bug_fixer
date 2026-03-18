"""verifier_node — judges whether the task is complete."""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ._client import llm, load_prompt

logger = logging.getLogger("nodes.verifier")


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class Verdict(BaseModel):
    complete: bool = Field(
        description="True if all plan steps succeeded and the task is fully done"
    )
    summary: str = Field(
        description="One-sentence explanation of the verdict"
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

    logger.info("Verdict: complete=%s — %s", verdict.complete, verdict.summary)

    if verdict.complete:
        return {"output": verdict.summary, "status": "complete"}
    return {"retries": state.get("retries", 0) + 1, "status": "executing"}
