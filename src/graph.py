"""
graph.py — LangGraph orchestration for the Plan-Act-Verify SWE agent.

State machine:
  planner  →  actor  →  verifier
                ↑____________|  (loop back on failure)
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from nodes import actor_node, planner_node, reproduce_node, verify_repro_node, verifier_node

# ---------------------------------------------------------------------------
# Shared state schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # The original task description supplied by the caller
    task: str
    # Raw issue body text (populated for GitHub issue events)
    issue_desc: str
    # GitHub clone URL of the target repo
    repo_url: str
    # Structured plan produced by the planner node (list of PlanStep dicts)
    plan: list[Any]

    # ── Reproduce phase ────────────────────────────────────────────────────
    # Result dict from reproduce_node (diff, test_output, aider_output, etc.)
    repro_result: dict
    # Whether verify_repro_node confirmed the test exists and fails
    repro_verified: bool
    # How many reproduce-retry cycles have been attempted
    repro_retries: int
    # Last failure reason from verify_repro_node
    repro_failure_reason: str

    # ── Fix phase ──────────────────────────────────────────────────────────
    # Accumulated results from actor_node steps
    results: Annotated[list[Any], operator.add]
    # How many fix-retry cycles have been attempted
    retries: int
    # Failure classification from the last verifier run
    # none | test_failure | infra_error | no_diff | sandbox_error
    failure_type: str

    # ── Terminal ───────────────────────────────────────────────────────────
    # Final answer / summary once the workflow is complete
    output: str
    # Lifecycle status: planning | reproducing | fixing | complete | error
    status: str


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

MAX_RETRIES = 3


def route_after_repro(state: AgentState) -> str:
    """After verify_repro: proceed to fix phase or retry reproduction."""
    if state.get("repro_verified"):
        return "actor"
    if state.get("repro_retries", 0) >= MAX_RETRIES:
        return END
    return "reproduce"


def route_after_verify(state: AgentState) -> str:
    """After verifier: end if done/give-up, or retry actor."""
    if state.get("output"):
        return END
    if state.get("retries", 0) >= MAX_RETRIES:
        return END
    # Infra errors won't be fixed by retrying the same code
    if state.get("failure_type") == "infra_error":
        return END
    return "actor"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("reproduce", reproduce_node)
    g.add_node("verify_repro", verify_repro_node)
    g.add_node("actor", actor_node)
    g.add_node("verifier", verifier_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "reproduce")
    g.add_edge("reproduce", "verify_repro")
    g.add_conditional_edges(
        "verify_repro",
        route_after_repro,
        {"reproduce": "reproduce", "actor": "actor", END: END},
    )
    g.add_edge("actor", "verifier")
    g.add_conditional_edges(
        "verifier",
        route_after_verify,
        {"actor": "actor", END: END},
    )

    return g.compile()


# Singleton — imported by bridge.py
graph = build_graph()
