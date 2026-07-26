"""LangGraph node that recalls relevant long-term AgentHive memories."""

from src.orchestration.state import AgentHiveState
from src.tools.memory_tool import AgentMemoryTool


class MemoryRecallAgent:
    """Retrieve relevant past task memories before AgentHive starts work."""

    name = "memory_recall"

    def __init__(self) -> None:
        self.memory_tool = AgentMemoryTool()

    def run(self, state: AgentHiveState) -> dict:
        """Add relevant long-term memory context to the workflow state."""
        memory_context = self.memory_tool.format_recalled_memories(
            task=state["task"],
            limit=3,
        )

        trace_message = (
            "Long-term memory recalled"
            if memory_context
            else "Long-term memory checked — no relevant memories found"
        )

        return {
            "memory_context": memory_context,
            "trace": [*state.get("trace", []), trace_message],
        }