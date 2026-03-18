"""planner_node — analyses the issue, inspects the repo, returns a PlanStep list."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ._client import llm, load_prompt

logger = logging.getLogger("nodes.planner")


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class PlanStep(BaseModel):
    file: str = Field(description="Relative path of the file to create or modify")
    description: str = Field(description="One-sentence description of the change for that file")
    status: str = Field(default="pending", description="Execution status: pending | in_progress | done | failed")


class Plan(BaseModel):
    steps: list[PlanStep] = Field(description="Ordered list of implementation steps")


# ---------------------------------------------------------------------------
# Repo structure utility
# ---------------------------------------------------------------------------

_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache"}


def _repo_map(repo_path: str) -> str:
    """Build an aider RepoMap; fall back to ls -R if aider is unavailable."""
    try:
        from aider.repomap import RepoMap
        from aider.io import InputOutput

        io = InputOutput(pretty=False, yes=True)
        rm = RepoMap(root=repo_path, io=io, map_tokens=2048)

        all_files: list[str] = []
        for dirpath, dirs, filenames in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
            for f in filenames:
                all_files.append(os.path.join(dirpath, f))

        return rm.get_repo_map([], all_files) or "(empty repo map)"

    except Exception as exc:
        logger.warning("aider RepoMap failed, falling back to ls -R: %s", exc)
        result = subprocess.run(
            ["ls", "-R", repo_path], capture_output=True, text=True, timeout=10
        )
        output = result.stdout.replace(repo_path, ".")
        lines = [line for line in output.splitlines() if ".git" not in line]
        return "\n".join(lines)


def _get_repo_structure(repo_url: str) -> str:
    """Clone repo (depth 1) into a temp dir and return its aider RepoMap."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--quiet", repo_url, tmpdir],
                capture_output=True,
                check=True,
                timeout=30,
            )
            return _repo_map(tmpdir)
    except Exception as exc:
        logger.warning("Could not clone repo: %s", exc)
        return "(repo structure unavailable)"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

_chain = (
    ChatPromptTemplate.from_messages([
        ("system", load_prompt("planner")),
        ("human", "{input}"),
    ])
    | llm.with_structured_output(Plan)
)


def planner_node(state: dict) -> dict:
    """Analyse the issue + repo structure, return a structured PlanStep list."""
    issue_desc = state.get("issue_desc", "") or state.get("task", "")
    repo_url = state.get("repo_url", "")

    repo_context = _get_repo_structure(repo_url) if repo_url else "(no repo URL provided)"
    logger.info("Repo map (%d chars):\n%s", len(repo_context), repo_context[:1000])

    result: Plan = _chain.invoke({
        "input": f"Issue / task:\n{issue_desc}\n\nRepository map:\n{repo_context}",
    })

    plan = [step.model_dump() for step in result.steps]
    logger.info("Planner produced %d step(s): %s", len(plan), [s["file"] for s in plan])
    return {"plan": plan, "retries": 0, "status": "executing"}
