"""LangGraph node that recalls relevant long-term AgentHive memories."""

from src.orchestration.state import AgentHiveState


class MemoryRecallAgent:
    """Retrieve memories when available without blocking the workflow."""

    name = "memory_recall"

    def __init__(self) -> None:
        self.memory_tool = None

        try:
            # Lazy import prevents optional ChromaDB errors from crashing Phase 1.
            from src.tools.memory_tool import AgentMemoryTool

            self.memory_tool = AgentMemoryTool()
        except Exception:
            self.memory_tool = None

    def run(self, state: AgentHiveState) -> dict:
        """Recall memory if available; otherwise continue normally."""

        current_trace = state.get("trace", [])
        user_id = state.get("user_id") or "default"
        task_id = state.get("task_id", "")

        if self.memory_tool is None:
            return {
                "memory_context": "",
                "trace": [
                    *current_trace,
                    "Memory unavailable — continuing without previous context",
                ],
            }

        try:
            memory_context = self.memory_tool.format_planning_context(
                task=state["task"],
                user_id=user_id,
                limit=3,
            )

            # Check for short-term working memory context if present
            stm_context = ""
            if task_id:
                try:
                    from src.memory.short_term import ShortTermStore
                    stm = ShortTermStore()
                    stm_context = stm.get_context_summary(user_id=user_id, task_id=task_id)
                except Exception:
                    pass

            if stm_context:
                memory_context = f"{memory_context}\n\n{stm_context}".strip() if memory_context else stm_context

            message = (
                "Long-term memory recalled"
                if memory_context
                else "Long-term memory checked — no relevant memories found"
            )

            return {
                "user_id": user_id,
                "memory_context": memory_context,
                "trace": [*current_trace, message],
            }

        except Exception:
            return {
                "user_id": user_id,
                "memory_context": "",
                "trace": [
                    *current_trace,
                    "Memory recall failed — continuing without previous context",
                ],
            }