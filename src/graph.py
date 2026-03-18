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

from nodes import actor_node, planner_node, verifier_node

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
    # Accumulated stdout / artefacts from the actor node
    results: Annotated[list[Any], operator.add]
    # How many verify-retry cycles have been attempted
    retries: int
    # Final answer / summary once the workflow is complete
    output: str
    # Lifecycle status: planning | executing | verifying | complete | error
    status: str


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

MAX_RETRIES = 3


def route_after_verify(state: AgentState) -> str:
    """Return the next node name based on verification outcome."""
    if state.get("output"):
        return END
    if state["retries"] >= MAX_RETRIES:
        # Give up gracefully — surface whatever we have
        return END
    return "actor"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("actor", actor_node)
    g.add_node("verifier", verifier_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "actor")
    g.add_edge("actor", "verifier")
    g.add_conditional_edges(
        "verifier",
        route_after_verify,
        {"actor": "actor", END: END},
    )

    return g.compile()


# Singleton — imported by bridge.py
graph = build_graph()
