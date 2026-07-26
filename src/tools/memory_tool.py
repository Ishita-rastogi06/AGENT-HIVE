"""Helpers for saving and recalling AgentHive task memories."""

from typing import Any

from src.memory.vector_db import MemoryVectorStore


class AgentMemoryTool:
    """Create useful long-term memories from completed AgentHive tasks."""

    MAX_MEMORY_DISTANCE = 0.85
    MAX_MEMORY_CHARACTERS = 600

    def __init__(self) -> None:
        self.store = MemoryVectorStore()

    def save_completed_task(
        self,
        task: str,
        final_answer: str,
        selected_agent: str,
        trace: list[str],
    ) -> str:
        """
        Save a completed AgentHive task as a semantic long-term memory.

        Returns:
            The ID of the newly stored memory.
        """
        memory_text = (
            "Previous AgentHive task\n"
            f"Task: {task.strip()}\n\n"
            f"Selected agent: {selected_agent}\n\n"
            f"Final answer: {final_answer.strip()}"
        )

        return self.store.save_memory(
            text=memory_text,
            metadata={
                "memory_type": "completed_task",
                "selected_agent": selected_agent,
                "workflow_trace": " → ".join(trace),
            },
        )

    def recall_for_task(
        self,
        task: str,
        limit: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Find only close, relevant earlier completed tasks.

        A smaller distance means a closer semantic match. Weak matches are
        discarded so unrelated memories are not sent to the AI model.
        """
        memories = self.store.search_memories(task, limit=limit)

        return [
            memory
            for memory in memories
            if memory["distance"] <= self.MAX_MEMORY_DISTANCE
        ]

    def format_recalled_memories(
        self,
        task: str,
        limit: int = 1,
    ) -> str:
        """
        Format one short, relevant memory for use inside an agent prompt.

        Returns an empty string when no close memory exists.
        """
        memories = self.recall_for_task(task, limit=limit)

        if not memories:
            return ""

        memory_text = memories[0]["text"]

        if len(memory_text) > self.MAX_MEMORY_CHARACTERS:
            memory_text = (
                f"{memory_text[:self.MAX_MEMORY_CHARACTERS].rstrip()}..."
            )

        return (
            "Relevant past AgentHive context (use only if helpful):\n"
            f"{memory_text}"
        )