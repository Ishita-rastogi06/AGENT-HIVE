"""Shared state passed through the AgentHive LangGraph workflow."""

from typing import Literal, TypedDict


class AgentHiveState(TypedDict, total=False):
    """Data contract shared by every agent in the AgentHive workflow."""

    task: str
    memory_context: str
    memory_id: str
    selected_agent: Literal["research", "coder"]
    specialist_output: str
    final_answer: str
    trace: list[str]