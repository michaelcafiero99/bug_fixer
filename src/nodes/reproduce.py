"""reproduce_node — clones the repo and uses Aider to write a failing regression test."""

from __future__ import annotations

import logging
import os

from e2b_code_interpreter import Sandbox

from ._client import load_prompt
from .actor import _clone_repo, _aider_step, _ensure_aider, _git_diff, _run_tests

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
        template = os.environ.get("E2B_TEMPLATE_ID") or None
        sbx = Sandbox.create(template=template) if template else Sandbox.create()

        # ── 0. Ensure aider is installed ──────────────────────────────────
        aider_status = _ensure_aider(sbx)
        logger.info("Aider ready: %s", aider_status[:100])

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

        # Extract the new test file path from the diff (first +++ b/... line)
        test_file_path = ""
        for line in diff.splitlines():
            if line.startswith("+++ b/") and "test" in line.lower():
                test_file_path = line[6:].strip()  # strip '+++ b/'
                break
        if test_file_path:
            logger.info("Reproduce: running specific test file: %s", test_file_path)
        result["test_file"] = test_file_path

        # ── 4. Run tests — we EXPECT them to fail here ────────────────────
        test_output = _run_tests(sbx, test_path=test_file_path)
        result["test_output"] = test_output
        logger.info("Test output (expecting failure):\n%s", test_output[:800])

        # ── 5. Read test file content so actor can include it in the PR ───
        test_content = ""
        if test_file_path:
            read_r = sbx.run_code(
                "try:\n"
                f"    with open('/repo/{test_file_path}') as f: print(f.read())\n"
                "except Exception as e: print('ERROR:', e)\n"
            )
            raw = "\n".join(read_r.logs.stdout).strip()
            if not raw.startswith("ERROR:"):
                test_content = raw
        result["repro_test_content"] = test_content

    except Exception as exc:
        logger.warning("Sandbox error in reproduce node: %s", exc)
        result["sandbox_error"] = str(exc)
    finally:
        if sbx is not None:
            try:
                sbx.kill()
            except Exception:
                pass

    # Surface repro test as a top-level state key so actor can write it into the
    # fresh sandbox before applying the fix (the test file only lives in the
    # reproduce sandbox's working tree — it is never committed to the repo).
    repro_test: dict = {}
    _path = result.get("test_file", "")
    _content = result.get("repro_test_content", "")
    if _path and _content:
        repro_test = {"path": _path, "content": _content}
        logger.info("Captured repro test: %s (%d chars)", _path, len(_content))

    return {"repro_result": result, "repro_test": repro_test}
