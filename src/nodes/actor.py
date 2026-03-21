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
    """Run aider inside the sandbox.

    Resolves the aider binary path explicitly — uv installs to ~/.local/bin which
    is not in the Jupyter kernel's default PATH, so bare 'aider' would raise
    FileNotFoundError (swallowed by the kernel, producing 0 chars of output).
    """
    extra_file = f", '--file', {target_file!r}" if target_file else ""
    result = sbx.run_code(
        "import subprocess, os, shutil, pathlib\n"
        "home = pathlib.Path.home()\n"
        "search_roots = [str(home / '.local'), '/home/user/.local']\n"
        "def find_aider():\n"
        "    if w := shutil.which('aider'): return w\n"
        "    for root in search_roots:\n"
        "        r = subprocess.run(\n"
        "            ['find', root, '-maxdepth', '6', '-name', 'aider',\n"
        "             '-not', '-path', '*/site-packages/*'],\n"
        "            capture_output=True, text=True, timeout=10)\n"
        "        for line in r.stdout.splitlines():\n"
        "            if line.strip(): return line.strip()\n"
        "    return None\n"
        "aider_bin = find_aider()\n"
        "if aider_bin:\n"
        "    parent = str(pathlib.Path(aider_bin).parent)\n"
        "    if parent not in os.environ.get('PATH', ''):\n"
        "        os.environ['PATH'] = parent + ':' + os.environ.get('PATH', '')\n"
        "if not aider_bin:\n"
        "    print('ERROR: aider binary not found — run _ensure_aider first')\n"
        "else:\n"
        f"    os.environ['GEMINI_API_KEY'] = {api_key!r}\n"
        f"    cmd = [aider_bin,\n"
        f"           '--model', 'gemini/gemini-2.5-flash',\n"
        f"           '--message', {message!r},\n"
        f"           '--yes-always', '--no-show-model-warnings', '--no-auto-commits'"
        f"{extra_file}]\n"
        "    r = subprocess.run(cmd, capture_output=True, text=True, cwd='/repo', timeout=300)\n"
        "    print(r.stdout)\n"
        "    if r.stderr:\n"
        "        print('--- aider stderr ---')\n"
        "        print(r.stderr[:2000])\n",
        timeout=360,
    )
    out = _stdout(result)
    # Surface kernel-level execution errors (e.g. FileNotFoundError, timeout)
    if not out and hasattr(result, "error") and result.error:
        out = f"[kernel error] {result.error}"
    return out


def _direct_delete_file(sbx: Sandbox, target_file: str) -> str:
    """Delete a tracked file using git rm inside /repo.

    Used instead of aider for deletion tasks — aider's diff-fenced edit format
    cannot express file deletions (it only patches file content).  git rm stages
    the deletion so _git_diff() captures it correctly.
    """
    result = sbx.run_code(
        "import subprocess, pathlib\n"
        f"target = {target_file!r}\n"
        "p = pathlib.Path('/repo') / target\n"
        "if p.exists():\n"
        "    r = subprocess.run(\n"
        "        ['git', 'rm', '-f', '--', target],\n"
        "        capture_output=True, text=True, cwd='/repo'\n"
        "    )\n"
        "    print(r.stdout or r.stderr or 'git rm ok')\n"
        "else:\n"
        "    # File not tracked by git — unlink if it exists on disk\n"
        "    if p.is_file():\n"
        "        p.unlink()\n"
        "        print('unlinked (untracked):', target)\n"
        "    else:\n"
        "        print('file not found:', target)\n"
    )
    return _stdout(result) or "(no output)"


# Deletion-intent keywords: if the step description contains one of these and
# target_file is set, we skip aider and use git rm directly.
_DELETION_KEYWORDS = ("remove", "delete", "rm ", "get rid of", "eliminate")


def _ensure_aider(sbx: Sandbox) -> str:
    """Locate aider (pre-baked in template) and patch PATH for the kernel session.

    uv drops the aider shim in ~/.local/bin, which is NOT in the Jupyter kernel's
    default PATH.  We look for the binary directly by path first so the pre-baked
    template binary is found without re-running aider-install every time.
    """
    result = sbx.run_code(
        "import subprocess, shutil, os, pathlib\n"
        "home = pathlib.Path.home()\n"
        "# Template build runs as 'user'; Jupyter kernel runs as 'root' — scan both\n"
        "search_roots = [str(home / '.local'), '/home/user/.local']\n"
        "def find_aider():\n"
        "    if w := shutil.which('aider'):\n"
        "        return w\n"
        "    for root in search_roots:\n"
        "        r = subprocess.run(\n"
        "            ['find', root, '-maxdepth', '6', '-name', 'aider',\n"
        "             '-not', '-path', '*/site-packages/*'],\n"
        "            capture_output=True, text=True, timeout=10\n"
        "        )\n"
        "        for line in r.stdout.splitlines():\n"
        "            if line.strip():\n"
        "                return line.strip()\n"
        "    return None\n"
        "aider_bin = find_aider()\n"
        "if aider_bin:\n"
        "    # Add its parent to PATH so subprocess can find it too\n"
        "    parent = str(pathlib.Path(aider_bin).parent)\n"
        "    if parent not in os.environ.get('PATH', ''):\n"
        "        os.environ['PATH'] = parent + ':' + os.environ.get('PATH', '')\n"
        "    print('aider ready:', aider_bin)\n"
        "else:\n"
        "    print('aider not in template — installing...')\n"
        "    r1 = subprocess.run(\n"
        "        ['pip', 'install', '-q', 'aider-install'],\n"
        "        capture_output=True, text=True, timeout=120\n"
        "    )\n"
        "    if r1.returncode != 0:\n"
        "        print('pip stderr:', r1.stderr[-300:])\n"
        "    r2 = subprocess.run(\n"
        "        ['aider-install'],\n"
        "        capture_output=True, text=True, timeout=180\n"
        "    )\n"
        "    print((r2.stdout + r2.stderr)[-300:])\n"
        "    aider_bin = find_aider()\n"
        "    if aider_bin:\n"
        "        parent = str(pathlib.Path(aider_bin).parent)\n"
        "        if parent not in os.environ.get('PATH', ''):\n"
        "            os.environ['PATH'] = parent + ':' + os.environ.get('PATH', '')\n"
        "    print('aider path after install:', aider_bin or 'NOT FOUND')\n",
        timeout=360,
    )
    return _stdout(result)


def _write_file(sbx: Sandbox, rel_path: str, content: str) -> None:
    """Write *content* to /repo/<rel_path>, creating parent directories as needed.

    Content is passed via an environment variable so that quotes, backslashes,
    and multi-line strings in the test file don't require any escaping.
    """
    sbx.run_code(
        "import os, pathlib\n"
        f"p = pathlib.Path('/repo/{rel_path}')\n"
        "p.parent.mkdir(parents=True, exist_ok=True)\n"
        "p.write_text(os.environ['_FILE_CONTENT'])\n",
        environment={"_FILE_CONTENT": content},
    )


def _commit_and_push(sbx: Sandbox, branch: str, repo_url: str, token: str,
                     message: str) -> str:
    """Stage all changes, commit, and force-push to *branch* on the remote.

    Returns 'pushed ok' on success or a short error string.
    """
    # Inject the token into the HTTPS clone URL so git can authenticate:
    # https://github.com/owner/repo.git → https://x-access-token:TOKEN@github.com/...
    push_url = repo_url
    if token and "github.com" in repo_url:
        push_url = repo_url.replace(
            "https://github.com",
            f"https://x-access-token:{token}@github.com",
        )

    result = sbx.run_code(
        "import subprocess\n"
        f"branch   = {branch!r}\n"
        f"push_url = {push_url!r}\n"
        f"msg      = {message!r}\n"
        "cwd = '/repo'\n"
        # Identity required for git commit
        "subprocess.run(['git', 'config', 'user.email', 'gh-agent@users.noreply.github.com'],"
        "               cwd=cwd, capture_output=True)\n"
        "subprocess.run(['git', 'config', 'user.name', 'GH Agent'],"
        "               cwd=cwd, capture_output=True)\n"
        # Stage everything (fix + repro test)
        "subprocess.run(['git', 'add', '-A'], cwd=cwd, capture_output=True)\n"
        # Switch to (or force-create) the fix branch
        "subprocess.run(['git', 'checkout', '-B', branch], cwd=cwd, capture_output=True)\n"
        # Commit
        "r_commit = subprocess.run(\n"
        "    ['git', 'commit', '-m', msg, '--allow-empty'],\n"
        "    capture_output=True, text=True, cwd=cwd\n"
        ")\n"
        # Force-push so retries overwrite the previous attempt
        "r_push = subprocess.run(\n"
        "    ['git', 'push', '--force', push_url, f'{branch}:{branch}'],\n"
        "    capture_output=True, text=True, cwd=cwd, timeout=60\n"
        ")\n"
        "if r_push.returncode == 0:\n"
        "    print('pushed ok')\n"
        "else:\n"
        "    print('push error:', r_push.stderr[-400:])\n",
        timeout=90,
    )
    return "\n".join(result.logs.stdout).strip()


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


def _run_tests(sbx: Sandbox, test_path: str = "") -> str:
    """Detect and run the test suite inside /repo. Returns combined output.

    If test_path is given (e.g. 'tests/test_config.py'), run only that file so
    collection errors are surfaced explicitly rather than silently yielding 0 items.

    Always uses pytest (installed on demand) because the E2B Jupyter kernel PATH
    may not expose the system pytest, and unittest can't discover plain def test_*()
    functions written by aider.
    """
    result = sbx.run_code(
        "import subprocess, pathlib\n"
        "\n"
        "repo = pathlib.Path('/repo')\n"
        f"test_path = {test_path!r}\n"
        "\n"
        "# Install dependencies if present\n"
        "for req in ['requirements.txt', 'requirements-dev.txt']:\n"
        "    rp = repo / req\n"
        "    if rp.exists():\n"
        "        subprocess.run(['pip', 'install', '-q', '-r', str(rp)], timeout=120)\n"
        "\n"
        "# Ensure pytest is available (the Jupyter kernel PATH may not expose system pytest)\n"
        "chk = subprocess.run(['python', '-m', 'pytest', '--version'],\n"
        "                     capture_output=True)\n"
        "if chk.returncode != 0:\n"
        "    subprocess.run(['pip', 'install', '-q', 'pytest'], timeout=60)\n"
        "\n"
        "# Use explicit path when provided — surfaces import/collection errors that\n"
        "# would otherwise produce a silent 'collected 0 items' when scanning '.'.\n"
        "target = test_path if test_path else '.'\n"
        "cmd = ['python', '-m', 'pytest', '-v', '--tb=short', '-p', 'no:cacheprovider', target]\n"
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
    github_token = os.environ.get("GITHUB_TOKEN", "")
    fix_branch = state.get("fix_branch", "")
    repro_test = state.get("repro_test", {})  # {"path": str, "content": str}

    sbx = None
    try:
        template = os.environ.get("E2B_TEMPLATE_ID") or None
        sbx = Sandbox.create(template=template) if template else Sandbox.create()

        # Detect deletion tasks upfront — skip aider entirely for these since
        # aider's diff-fenced format cannot express file deletions.
        step_lower = current_step.lower()
        is_deletion = bool(
            target_file
            and any(kw in step_lower for kw in _DELETION_KEYWORDS)
        )

        # ── 0. Ensure aider is installed (skip for pure deletions) ────────
        if not is_deletion:
            aider_status = _ensure_aider(sbx)
            logger.info("Aider ready: %s", aider_status[:100])

        # ── 1. Clone ───────────────────────────────────────────────────────
        if repo_url:
            clone_out = _clone_repo(sbx, repo_url)
            logger.info("Clone: %s", clone_out)
            result["clone_output"] = clone_out
        else:
            logger.warning("No repo_url — skipping clone")

        # ── 1b. Write repro test from reproduce phase ─────────────────────
        # The test file was created in the reproduce sandbox (never committed).
        # We write it here so it's included in the PR alongside the fix.
        if repro_test.get("path") and repro_test.get("content"):
            _write_file(sbx, repro_test["path"], repro_test["content"])
            logger.info("Wrote repro test to /repo/%s", repro_test["path"])

        # ── 2. Run Aider (or direct git rm for deletion tasks) ────────────
        if is_deletion:
            logger.info("Deletion task — using git rm directly for: %s", target_file)
            aider_out = _direct_delete_file(sbx, target_file)
        else:
            aider_out = _aider_step(sbx, current_step, target_file, api_key)
        logger.info("Aider output (%d chars):\n%s", len(aider_out), aider_out[:1000])
        result["aider_output"] = aider_out

        # ── 3. Capture git diff ────────────────────────────────────────────
        diff = _git_diff(sbx)
        result["diff"] = diff
        logger.info("Git diff (%d chars):\n%s", len(diff), diff[:500])

        # ── 4. Run test suite (using repro test if available) ─────────────
        test_path = repro_test.get("path", "") if repro_test else ""
        test_output = _run_tests(sbx, test_path=test_path)
        result["test_output"] = test_output
        logger.info("Test output (%d chars):\n%s", len(test_output), test_output[:800])

        # ── 5. Commit + push to fix branch (if token + branch configured) ─
        if fix_branch and github_token and repo_url:
            commit_msg = f"fix: {current_step[:72]}"
            push_status = _commit_and_push(sbx, fix_branch, repo_url,
                                           github_token, commit_msg)
            result["push_status"] = push_status
            result["fix_branch"] = fix_branch
            logger.info("Push to %s: %s", fix_branch, push_status)
        else:
            logger.info("Skipping push — fix_branch=%r token=%s",
                        fix_branch, "set" if github_token else "missing")

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
