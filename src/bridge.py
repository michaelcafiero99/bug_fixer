"""
bridge.py — FastAPI listener that accepts tasks and drives the LangGraph agent.

Endpoints:
  POST /         — GitHub webhook (push events trigger the agent)
  POST /run      — submit a task manually, receive a run_id
  GET  /status/{run_id} — poll for result
  GET  /health   — liveness check
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# force=True replaces any handlers uvicorn already attached to the root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
# Ensure our node loggers are always visible (they run in a thread executor)
logging.getLogger("nodes").setLevel(logging.INFO)
logger = logging.getLogger("bridge")

# Lazy import so graph compilation errors surface clearly
from graph import graph

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SWE Agent Orchestrator",
    description="LangGraph + E2B autonomous software engineering agent",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory run store (replace with Redis / DB for production)
_runs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    task: str


class RunResponse(BaseModel):
    run_id: str
    status: str = "queued"


class StatusResponse(BaseModel):
    run_id: str
    status: str  # queued | running | complete | error
    output: str | None = None
    results: list[Any] | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

async def _execute_run(
    run_id: str, task: str, issue_desc: str = "", repo_url: str = ""
) -> None:
    _runs[run_id]["status"] = "running"
    try:
        initial_state = {
            "task": task,
            "issue_desc": issue_desc,
            "repo_url": repo_url,
            "plan": [],
            # reproduce phase
            "repro_result": {},
            "repro_verified": False,
            "repro_retries": 0,
            "repro_failure_reason": "",
            # fix phase
            "results": [],
            "retries": 0,
            "failure_type": "",
            # terminal
            "output": "",
            "status": "planning",
        }
        # LangGraph invoke is synchronous — run in a thread so we don't block
        loop = asyncio.get_event_loop()
        final_state = await loop.run_in_executor(
            None, lambda: graph.invoke(initial_state)
        )
        _runs[run_id].update(
            {
                "status": "complete",
                "output": final_state.get("output", ""),
                "results": final_state.get("results", []),
            }
        )
    except Exception as exc:
        _runs[run_id].update({"status": "error", "error": str(exc)})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/", status_code=202)
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receive GitHub webhooks and trigger the agent."""
    payload = await request.json()
    event_type = request.headers.get("X-GitHub-Event", "")

    # GitHub sends a ping event on webhook creation — acknowledge it
    if "zen" in payload:
        logger.info("GitHub ping received: %s", payload["zen"])
        return {"msg": "pong", "zen": payload["zen"]}

    repo = payload.get("repository", {}).get("full_name", "unknown/repo")
    logger.info("Received GitHub event '%s' for repo '%s'", event_type, repo)

    # ── Issues event ────────────────────────────────────────────────────────
    if event_type == "issues":
        issue = payload.get("issue", {})
        action = payload.get("action", "unknown")
        number = issue.get("number")
        title = issue.get("title", "")
        body = issue.get("body", "") or ""
        user = issue.get("user", {}).get("login", "unknown")
        url = issue.get("html_url", "")

        logger.info(
            "Issue #%s %s by %s — %s\n  URL : %s\n  Body: %s",
            number, action, user, title, url,
            body[:500] + ("..." if len(body) > 500 else ""),
        )

        _ACTIONABLE = {"opened", "edited", "reopened"}
        if action not in _ACTIONABLE:
            logger.info("Issue action '%s' ignored (not in %s)", action, _ACTIONABLE)
            return {"msg": f"issue action '{action}' ignored"}

        task = (
            f"GitHub issue #{number} was {action} on {repo}.\n"
            f"Title: {title}\n"
            f"Author: {user}\n"
            f"URL: {url}\n\n"
            f"Issue body:\n{body}\n\n"
            f"Analyse the issue, suggest a fix or implementation plan, "
            f"and produce any relevant code or steps."
        )

        repo_url = payload.get("repository", {}).get("clone_url", "")
        run_id = str(uuid.uuid4())
        _runs[run_id] = {"status": "queued", "task": task, "repo": repo, "issue": number}
        background_tasks.add_task(_execute_run, run_id, task, body, repo_url)
        logger.info("Queued run %s for issue #%s (repo_url=%s)", run_id, number, repo_url)
        return {"run_id": run_id, "status": "queued", "repo": repo, "issue": number}

    # ── Push event ──────────────────────────────────────────────────────────
    if event_type == "push":
        ref = payload.get("ref", "refs/heads/unknown")
        branch = ref.replace("refs/heads/", "")
        commits = payload.get("commits", [])
        commit_msgs = "\n".join(f"- {c['message']}" for c in commits[:5])

        logger.info("Push to %s/%s — %d commit(s):\n%s", repo, branch, len(commits), commit_msgs)

        task = (
            f"A push was made to {repo} on branch '{branch}'.\n"
            f"Commits:\n{commit_msgs}\n\n"
            f"Review the changes, identify any issues or improvements, "
            f"and produce a short engineering report."
        )

        repo_url = payload.get("repository", {}).get("clone_url", "")
        run_id = str(uuid.uuid4())
        _runs[run_id] = {"status": "queued", "task": task, "repo": repo, "branch": branch}
        background_tasks.add_task(_execute_run, run_id, task, "", repo_url)
        logger.info("Queued run %s for push to %s/%s", run_id, repo, branch)
        return {"run_id": run_id, "status": "queued", "repo": repo, "branch": branch}

    logger.info("Event '%s' ignored", event_type)
    return {"msg": f"event '{event_type}' ignored"}


@app.post("/run", response_model=RunResponse, status_code=202)
async def submit_run(body: RunRequest, background_tasks: BackgroundTasks):
    """Submit a new engineering task to the agent manually."""
    run_id = str(uuid.uuid4())
    _runs[run_id] = {"status": "queued", "task": body.task}
    background_tasks.add_task(_execute_run, run_id, body.task)
    return RunResponse(run_id=run_id)


@app.get("/status/{run_id}", response_model=StatusResponse)
async def get_status(run_id: str):
    """Poll the status of a previously submitted run."""
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return StatusResponse(run_id=run_id, **{k: v for k, v in run.items() if k != "task"})


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("bridge:app", host="0.0.0.0", port=8000, reload=True)
