"""actor_node — clones the repo into an E2B sandbox and runs Aider to apply one plan step."""

from __future__ import annotations

import logging
import os
from typing import Any

from e2b_code_interpreter import Sandbox

logger = logging.getLogger("nodes.actor")


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


def _aider_step(sbx: Sandbox, message: str, target_file: str, api_key: str) -> str:
    """Find a compatible Python (3.10-3.12), install aider-chat, and run it.

    Split into two run_code calls so that each gets its own explicit timeout:
    - Cell 1 (timeout=300s): locate python3.12 and pip-install aider-chat
    - Cell 2 (timeout=360s): run aider with the given message
    aider-chat requires Python <3.13; the E2B sandbox ships 3.13.
    """
    # ── Cell 1: install ──────────────────────────────────────────────────────
    install_result = sbx.run_code(
        "import subprocess, shutil\n"
        "\n"
        "py = None\n"
        "for candidate in ['python3.12', 'python3.11', 'python3.10']:\n"
        "    if shutil.which(candidate):\n"
        "        py = candidate\n"
        "        break\n"
        "\n"
        "if py is None:\n"
        "    subprocess.run(['apt-get', 'install', '-y', '-q', 'python3.12'],\n"
        "                   capture_output=True, timeout=120)\n"
        "    py = 'python3.12'\n"
        "\n"
        "print('Using Python:', py)\n"
        "\n"
        "pip = subprocess.run(\n"
        "    [py, '-m', 'pip', 'install', '-q', 'aider-chat'],\n"
        "    capture_output=True, text=True, timeout=240)\n"
        "if pip.returncode != 0:\n"
        "    print('pip install FAILED:', pip.stderr[:800])\n"
        "else:\n"
        "    print('aider-chat installed ok')\n"
        "\n"
        "# Expose py for the next cell via a sentinel file\n"
        "import pathlib\n"
        "pathlib.Path('/tmp/aider_py').write_text(py)\n",
        timeout=300,
    )
    install_out = _stdout(install_result)
    logger.info("Aider install: %s", install_out)

    if "FAILED" in install_out:
        return install_out  # surface the error immediately

    # ── Cell 2: run aider ────────────────────────────────────────────────────
    extra_file = f", '--file', {target_file!r}" if target_file else ""
    run_result = sbx.run_code(
        "import subprocess, os, pathlib\n"
        f"os.environ['GEMINI_API_KEY'] = {api_key!r}\n"
        "py = pathlib.Path('/tmp/aider_py').read_text().strip()\n"
        f"cmd = [py, '-m', 'aider',\n"
        f"       '--model', 'gemini/gemini-2.5-flash',\n"
        f"       '--message', {message!r},\n"
        f"       '--yes-always', '--no-show-model-warnings', '--no-auto-commits'"
        f"{extra_file}]\n"
        "r = subprocess.run(cmd, capture_output=True, text=True, cwd='/repo', timeout=300)\n"
        "print(r.stdout)\n"
        "if r.stderr:\n"
        "    print('--- aider stderr ---')\n"
        "    print(r.stderr[:2000])\n",
        timeout=360,
    )
    return install_out + "\n" + _stdout(run_result)


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
    """Clone the repo, run Aider to apply one plan step, return diff and test results."""
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

    api_key = os.environ.get("GEMINI_API_KEY", "")

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

        # ── 2. Run Aider ───────────────────────────────────────────────────
        aider_out = _aider_step(sbx, current_step, target_file, api_key)
        logger.info("Aider output (%d chars):\n%s", len(aider_out), aider_out[:1000])
        result["aider_output"] = aider_out

        # ── 3. Capture git diff ────────────────────────────────────────────
        diff = _git_diff(sbx)
        result["diff"] = diff
        logger.info("Git diff (%d chars):\n%s", len(diff), diff[:500])

        # ── 4. Run test suite ──────────────────────────────────────────────
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
