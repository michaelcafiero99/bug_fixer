"""actor_node — clones the repo into an E2B sandbox, applies one plan step, returns diff + test output."""

from __future__ import annotations

import base64
import logging
from typing import Any

from e2b_code_interpreter import Sandbox
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ._client import llm, load_prompt

logger = logging.getLogger("nodes.actor")

_SYSTEM = load_prompt("actor")


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class FileEdit(BaseModel):
    new_content: str = Field(description="Complete new content for the target file")
    explanation: str = Field(description="One-sentence summary of the change made")


_chain = (
    ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human", "{input}"),
    ])
    | llm.with_structured_output(FileEdit)
)


# ---------------------------------------------------------------------------
# Sandbox helpers
# ---------------------------------------------------------------------------

def _stdout(result) -> str:
    """Extract printed output from an E2B Execution (logs.stdout, not .text)."""
    return "\n".join(result.logs.stdout).strip()


def _clone_repo(sbx: Sandbox, repo_url: str) -> str:
    """Clone repo at depth 1 into /repo inside the sandbox. Returns stdout/stderr."""
    result = sbx.run_code(
        f"import subprocess\n"
        f"r = subprocess.run(['git', 'clone', '--depth', '1', '--quiet', {repo_url!r}, '/repo'],\n"
        f"                   capture_output=True, text=True, timeout=60)\n"
        f"print(r.stdout or r.stderr or 'cloned ok')\n"
    )
    return _stdout(result)


def _read_file(sbx: Sandbox, rel_path: str) -> str:
    """Read /repo/{rel_path} from the sandbox; return empty string if missing."""
    result = sbx.run_code(
        f"import pathlib\n"
        f"p = pathlib.Path('/repo/{rel_path}')\n"
        f"print(p.read_text() if p.exists() else '')\n"
    )
    return _stdout(result)


def _write_file(sbx: Sandbox, rel_path: str, content: str) -> None:
    """Write content to /repo/{rel_path} inside the sandbox via base64 to avoid shell escaping."""
    encoded = base64.b64encode(content.encode()).decode()
    sbx.run_code(
        f"import base64, pathlib\n"
        f"p = pathlib.Path('/repo/{rel_path}')\n"
        f"p.parent.mkdir(parents=True, exist_ok=True)\n"
        f"p.write_bytes(base64.b64decode({encoded!r}))\n"
    )


def _git_diff(sbx: Sandbox) -> str:
    """Stage all changes then return the unified diff inside /repo."""
    result = sbx.run_code(
        "import subprocess\n"
        "# Stage everything so new/deleted files appear in the diff\n"
        "subprocess.run(['git', 'add', '-A'], cwd='/repo', capture_output=True)\n"
        "r = subprocess.run(['git', 'diff', '--cached', '--unified=3'],\n"
        "                   capture_output=True, text=True, cwd='/repo')\n"
        "print(r.stdout or '(no diff)')\n"
    )
    return _stdout(result) or "(no diff)"


def _run_tests(sbx: Sandbox) -> str:
    """Detect and run the test suite inside /repo. Returns combined output."""
    result = sbx.run_code(
        "import subprocess, pathlib\n"
        "\n"
        "repo = pathlib.Path('/repo')\n"
        "\n"
        "# Install dependencies if present\n"
        "for req in ['requirements.txt', 'requirements-dev.txt']:\n"
        "    rp = repo / req\n"
        "    if rp.exists():\n"
        "        subprocess.run(['pip', 'install', '-q', '-r', str(rp)], timeout=120)\n"
        "\n"
        "# Try pytest first, then unittest discovery\n"
        "has_pytest = subprocess.run(['python', '-m', 'pytest', '--version'],\n"
        "                            capture_output=True).returncode == 0\n"
        "if has_pytest:\n"
        "    cmd = ['python', '-m', 'pytest', '-v', '--tb=short']\n"
        "else:\n"
        "    cmd = ['python', '-m', 'unittest', 'discover', '-v']\n"
        "\n"
        "r = subprocess.run(cmd, capture_output=True, text=True, cwd='/repo', timeout=120)\n"
        "print(r.stdout)\n"
        "if r.stderr:\n"
        "    print('--- stderr ---')\n"
        "    print(r.stderr)\n"
    )
    return _stdout(result) or "(no test output)"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def actor_node(state: dict) -> dict:
    """Clone the repo, apply one plan step, return diff and test results."""
    plan = state.get("plan", [])
    results = state.get("results", [])
    repo_url = state.get("repo_url", "")
    step_index = len(results)

    if step_index >= len(plan):
        return {"results": []}

    raw_step = plan[step_index]
    if isinstance(raw_step, dict):
        current_step = raw_step.get("description", str(raw_step))
        target_file = raw_step.get("file", "")
    else:
        current_step = str(raw_step)
        target_file = ""

    logger.info(
        "Actor step %d/%d — file: %s | %s",
        step_index + 1, len(plan), target_file or "(none)", current_step,
    )

    result: dict[str, Any] = {
        "step": current_step,
        "file": target_file,
    }

    sbx = None
    try:
        sbx = Sandbox.create()

        # ── 1. Clone the repo ──────────────────────────────────────────────
        if repo_url:
            clone_out = _clone_repo(sbx, repo_url)
            logger.info("Clone output: %s", clone_out.strip())
            result["clone_output"] = clone_out
        else:
            logger.warning("No repo_url in state — skipping clone")

        # ── 2. Read existing file content ──────────────────────────────────
        existing_content = ""
        if target_file and repo_url:
            existing_content = _read_file(sbx, target_file)
            logger.info(
                "Read %d chars from %s",
                len(existing_content), target_file,
            )

        # ── 3. Ask LLM to produce the new file content ────────────────────
        human_msg = (
            f"Plan step {step_index + 1}: {current_step}\n\n"
            f"Target file: {target_file or '(new file)'}\n\n"
        )
        if existing_content:
            human_msg += f"Current file content:\n```\n{existing_content}\n```\n"
        else:
            human_msg += "Current file content: (file does not exist yet — create it)\n"

        edit: FileEdit = _chain.invoke({"input": human_msg})
        result["explanation"] = edit.explanation
        logger.info("LLM edit explanation: %s", edit.explanation)

        # ── 4. Write the new content back into the sandbox ─────────────────
        if target_file:
            _write_file(sbx, target_file, edit.new_content)
            logger.info("Wrote new content to /repo/%s", target_file)
            result["new_content"] = edit.new_content

        # ── 5. Capture git diff ────────────────────────────────────────────
        diff = _git_diff(sbx)
        result["diff"] = diff
        logger.info("Git diff (%d chars):\n%s", len(diff), diff[:500])

        # ── 6. Run test suite ──────────────────────────────────────────────
        test_output = _run_tests(sbx)
        result["test_output"] = test_output
        logger.info("Test output (%d chars):\n%s", len(test_output), test_output[:800])

    except Exception as exc:
        logger.warning("Sandbox error on step %d: %s", step_index + 1, exc)
        result["sandbox_error"] = str(exc)
    finally:
        if sbx is not None:
            try:
                sbx.kill()
            except Exception:
                pass

    return {"results": [result]}
