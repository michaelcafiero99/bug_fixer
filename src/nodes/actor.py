"""actor_node — executes one plan step inside an E2B sandbox."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from e2b_code_interpreter import Sandbox
from langchain_core.messages import HumanMessage, SystemMessage

from ._client import llm, load_prompt

logger = logging.getLogger("nodes.actor")

_SYSTEM = load_prompt("actor")


def actor_node(state: dict) -> dict:
    """Execute the next pending plan step inside an E2B sandbox."""
    plan = state.get("plan", [])
    results = state.get("results", [])
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

    step_prompt = f"Plan step {step_index + 1}"
    if target_file:
        step_prompt += f" (file: {target_file})"
    step_prompt += f": {current_step}"

    logger.info("Actor executing: %s", step_prompt)

    response = llm.invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=step_prompt),
    ])
    code_instructions: str = response.content

    result: dict[str, Any] = {
        "step": current_step,
        "file": target_file,
        "output": code_instructions,
    }
    try:
        with Sandbox(api_key=os.environ["E2B_API_KEY"]) as sbx:
            code_blocks = re.findall(r"```python\s*(.*?)```", code_instructions, re.DOTALL)
            for code in code_blocks:
                exec_result = sbx.run_code(code)
                result["sandbox_stdout"] = exec_result.text
                result["sandbox_logs"] = [str(log) for log in exec_result.logs.stdout]
    except Exception as exc:
        logger.warning("Sandbox error on step %d: %s", step_index + 1, exc)
        result["sandbox_error"] = str(exc)

    return {"results": [result]}
