"""
Build the GH-agent E2B sandbox template (Build System v2 — no Docker required).

Usage:
    pip install e2b          # if not already installed alongside e2b-code-interpreter
    python sandbox/build.py

After it completes, copy the printed template ID into .env:
    E2B_TEMPLATE_ID=<id>

The template extends E2B's code-interpreter base (which runs Jupyter on port 49999)
and adds aider-chat so sandboxes start in ~2s with aider pre-installed.

IMPORTANT: Must extend "code-interpreter-v1" (not the bare base image) so that
e2b_code_interpreter.Sandbox can connect to Jupyter on port 49999.

Note: .run_cmd() bakes steps into the image at build time.
      .set_start_cmd() runs on every sandbox start — don't use that for installs.
"""

from e2b import Template, default_build_logger  # pip install e2b

template = (
    # from_template("code-interpreter-v1") extends E2B's code-interpreter base,
    # which keeps Jupyter running on port 49999 (required by e2b_code_interpreter.Sandbox)
    Template()
    .from_template("code-interpreter-v1")
    # aider-install uses uv to pick a Python <3.13 (aider-chat requires <3.13)
    .run_cmd("pip install -q aider-install")
    .run_cmd("aider-install")
    # Smoke-test: fail the build early if aider isn't on PATH
    .run_cmd("aider --version")
)
if __name__ == "__main__":
    result = Template.build(
        template,
        alias="gh-agent",
        on_build_logs=default_build_logger(),
    )
    print(f"\n✅ Template built successfully.")
    print(f"   Add to .env:  E2B_TEMPLATE_ID={result.template_id}")
