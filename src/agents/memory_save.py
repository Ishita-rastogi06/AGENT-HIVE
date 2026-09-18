"""LangGraph node that saves completed tasks to long-term memory."""

from src.orchestration.state import AgentHiveState


class MemorySaveAgent:
    """Save completed tasks when memory storage is available."""

    name = "memory_save"

    def __init__(self) -> None:
        self.memory_tool = None

        try:
            # Lazy import keeps ChromaDB optional during Phase 1.
            from src.tools.memory_tool import AgentMemoryTool

            self.memory_tool = AgentMemoryTool()
        except Exception:
            self.memory_tool = None

    def run(self, state: AgentHiveState) -> dict:
        """Save the result if possible without interrupting the workflow."""

        current_trace = state.get("trace", [])
        user_id = state.get("user_id") or "default"
        task_id = state.get("task_id", "")
        selected_agent = state.get("selected_agent", "research")

        # Determine task outcome
        status = state.get("workflow_status", "completed")
        if state.get("needs_human_review") or status == "escalated":
            outcome = "escalated"
        elif status in ("failed", "error"):
            outcome = "failure"
        else:
            outcome = "success"

        # Determine non-empty final answer content
        final_answer = (
            state.get("final_answer")
            or state.get("specialist_output")
            or (state.get("current_subtask") or {}).get("output")
            or (state.get("review") or {}).get("feedback")
            or f"Task {outcome} — no explicit answer string recorded."
        ).strip()

        # Extract used tool names from tool_logs in state
        tool_logs = state.get("tool_logs") or []
        used_tools: list[str] = []
        for log in tool_logs:
            if isinstance(log, dict) and log.get("tool_name"):
                t_name = str(log["tool_name"])
                if t_name not in used_tools:
                    used_tools.append(t_name)

        if not used_tools:
            if "```" in final_answer or selected_agent == "coder":
                used_tools = ["python_executor"]
            elif selected_agent == "data":
                used_tools = ["sqlite_db"]

        # Clear short-term task memory on completion
        if task_id:
            try:
                from src.memory.short_term import ShortTermStore
                ShortTermStore().clear(user_id=user_id, task_id=task_id)
            except Exception:
                pass

        if self.memory_tool is None:
            return {
                "trace": [
                    *current_trace,
                    "Memory unavailable — completed task was not saved",
                ],
            }

        try:
            memory_id = self.memory_tool.save_completed_task(
                task=state["task"],
                final_answer=final_answer,
                selected_agent=selected_agent,
                trace=current_trace,
                user_id=user_id,
                task_type=selected_agent,
                outcome=outcome,
                tools_used=used_tools,
                final_confidence=state.get("specialist_confidence", 0.85),
            )
            return {
                "memory_id": memory_id,
                "final_answer": final_answer,
                "tools_used": used_tools,
                "trace": [
                    *current_trace,
                    "Completed task saved to long-term memory",
                ],
            }
        except Exception:
            return {
                "final_answer": final_answer,
                "tools_used": used_tools,
                "trace": [
                    *current_trace,
                    "Completed task could not be saved to long-term memory",
                ],
            }