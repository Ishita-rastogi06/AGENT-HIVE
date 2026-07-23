"""Shared state passed through the LangGraph workflow."""
from typing import Literal, TypedDict

class AgentHiveState(TypedDict, total=False):
    task: str
    selected_agent: Literal["research", "coder"]
    specialist_output: str
    final_answer: str
    trace: list[str]
