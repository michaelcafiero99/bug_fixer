"""Shared LangChain LLM client and prompt loader for all nodes."""

from __future__ import annotations

import os
import pathlib

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

MODEL = "gemini-2.5-flash"

llm = ChatGoogleGenerativeAI(
    model=MODEL,
    google_api_key=os.environ["GEMINI_API_KEY"],
)

_PROMPTS_DIR = pathlib.Path(__file__).parent.parent / "prompts"


def load_prompt(name: str) -> str:
    """Read src/prompts/{name}.txt and return its contents."""
    return (_PROMPTS_DIR / f"{name}.txt").read_text()
