"""reproduce_node — clones the repo and uses Aider to write a failing regression test."""

from __future__ import annotations

import logging
import os

from e2b_code_interpreter import Sandbox

from ._client import load_prompt
from .actor import _clone_repo, _aider_step, _git_diff, _run_tests

logger = logging.getLogger("nodes.reproduce")

_MESSAGE_TEMPLATE = load_prompt("reproduce")


def reproduce_node(state: dict) -> dict:
    """Write a failing test that reproduces the bug described in issue_desc."""
    issue_desc = state.get("issue_desc", "") or state.get("task", "")
    repo_url = state.get("repo_url", "")
    api_key = os.environ.get("GEMINI_API_KEY", "")

    # Build the full Aider message: system instruction + bug description
    message = f"{_MESSAGE_TEMPLATE}\n\nBug to reproduce:\n{issue_desc}"

    logger.info("─── Reproduce node ─── writing failing test for: %s", issue_desc[:120])

    result: dict = {"step": "write_failing_test"}

    sbx = None
    try:
        sbx = Sandbox.create()

        # ── 1. Clone ───────────────────────────────────────────────────────
        if repo_url:
            clone_out = _clone_repo(sbx, repo_url)
            logger.info("Clone: %s", clone_out)
            result["clone_output"] = clone_out
        else:
            logger.warning("No repo_url — skipping clone")

        # ── 2. Aider writes the test (no target_file; it decides where) ────
        aider_out = _aider_step(sbx, message, target_file="", api_key=api_key)
        logger.info("Aider output (%d chars):\n%s", len(aider_out), aider_out[:1000])
        result["aider_output"] = aider_out

        # ── 3. Diff to confirm a test file was written ────────────────────
        diff = _git_diff(sbx)
        result["diff"] = diff
        logger.info("Git diff (%d chars):\n%s", len(diff), diff[:500])

        # ── 4. Run tests — we EXPECT them to fail here ────────────────────
        test_output = _run_tests(sbx)
        result["test_output"] = test_output
        logger.info("Test output (expecting failure):\n%s", test_output[:800])

    except Exception as exc:
        logger.warning("Sandbox error in reproduce node: %s", exc)
        result["sandbox_error"] = str(exc)
    finally:
        if sbx is not None:
            try:
                sbx.kill()
            except Exception:
                pass

    return {"repro_result": result}
