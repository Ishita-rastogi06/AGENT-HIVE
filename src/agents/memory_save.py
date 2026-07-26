"""LangGraph node that saves completed AgentHive tasks to long-term memory."""

from src.orchestration.state import AgentHiveState
from src.tools.memory_tool import AgentMemoryTool


class MemorySaveAgent:
    """Persist a completed task after the reviewer produces its final answer."""

    name = "memory_save"

    def __init__(self) -> None:
        self.memory_tool = AgentMemoryTool()

    def run(self, state: AgentHiveState) -> dict:
        """
        Save the completed task without interrupting a successful workflow
        if memory storage is temporarily unavailable.
        """
        try:
            memory_id = self.memory_tool.save_completed_task(
                task=state["task"],
                final_answer=state["final_answer"],
                selected_agent=state.get("selected_agent", "unknown"),
                trace=state.get("trace", []),
            )

            return {
                "memory_id": memory_id,
                "trace": [
                    *state.get("trace", []),
                    "Completed task saved to long-term memory",
                ],
            }

        except Exception:
            return {
                "trace": [
                    *state.get("trace", []),
                    "Completed task could not be saved to long-term memory",
                ],
            }