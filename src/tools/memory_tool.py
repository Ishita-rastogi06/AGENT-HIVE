"""
High-level memory API used by MemoryRecallAgent and MemorySaveAgent.

This module bridges the orchestration layer to the two storage backends:
  - LongTermStore (ChromaDB via MemoryVectorStore)
  - All public methods accept and enforce user_id

Planning prompt injection
─────────────────────────
format_planning_context() returns a structured prompt block that the
supervisor's planning call receives.  The exact template is:

    ── Prior context (top-{k} similar tasks) ──────────────────────────
    [1] {task_type} task — outcome: {outcome} — agent: {selected_agent}
        Tools used: {tools_used}
        What worked: {answer_summary}
        Confidence: {confidence:.2f}

    [2] …

    Guidance: prefer approaches that worked in similar past tasks.
    ────────────────────────────────────────────────────────────────────

If no memories pass the relevance threshold the block is omitted entirely
(empty string returned) so the supervisor prompt is not padded with noise.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import settings
from src.memory.schemas import MemoryQuery, MemoryStats

logger = logging.getLogger(__name__)

# Char budget per memory snippet inside the planning context block.
_MAX_SNIPPET_CHARS = 400
_SEPARATOR = "─" * 68


class AgentMemoryTool:
    """
    Facade over MemoryVectorStore that adds:
      • structured planning-prompt injection (format_planning_context)
      • list / delete / stats API for the UI dashboard
      • importance feedback endpoint (apply_feedback)
    """

    def __init__(self, storage_path: Any | None = None) -> None:
        from src.memory.vector_db import MemoryVectorStore
        self.store = MemoryVectorStore(storage_path=storage_path)

    # ── save ──────────────────────────────────────────────────────────────────

    def save_completed_task(
        self,
        task: str,
        final_answer: str,
        selected_agent: str,
        trace: list[str],
        user_id: str = "default",
        task_type: str = "research",
        outcome: str = "success",
        tools_used: list[str] | None = None,
        final_confidence: float = 0.5,
    ) -> str:
        """
        Persist one completed task as a long-term memory entry.

        Returns the UUID of the saved memory.
        """
        tool_names = ", ".join(tools_used or []) or "none"

        # Store full answer text without artificial length caps so memory content is complete
        answer_preview = (final_answer or "").strip() or f"Task outcome: {outcome}."

        memory_text = (
            f"Task: {task.strip()}\n\n"
            f"Agent: {selected_agent} | Type: {task_type} | Outcome: {outcome}\n"
            f"Tools used: {tool_names}\n"
            f"Confidence: {final_confidence:.2f}\n\n"
            f"Answer summary:\n{answer_preview}"
        )

        metadata: dict[str, Any] = {
            "memory_type": "task_outcome",
            "task_type": task_type,
            "selected_agent": selected_agent,
            "outcome": outcome,
            "tools_used": tool_names,
            "final_confidence": final_confidence,
            "workflow_trace": " → ".join(trace[-10:]),  # last 10 steps
        }

        return self.store.save_memory(
            text=memory_text,
            user_id=user_id,
            metadata=metadata,
        )

    def save_domain_fact(
        self,
        user_id: str,
        fact: str,
        source: str = "user_input",
    ) -> str:
        """
        Save a standalone domain fact for user_id.

        Tagged distinctly (memory_type="fact") so retrieval can distinguish facts from task outcomes.
        Returns the assigned memory UUID.
        """
        clean_fact = fact.strip()
        if not clean_fact:
            raise ValueError("Fact text cannot be empty.")

        memory_text = f"Domain Fact: {clean_fact}\nSource: {source}"
        metadata: dict[str, Any] = {
            "memory_type": "fact",
            "source": source,
            "task_type": "domain_fact",
            "outcome": "success",
            "selected_agent": "system",
            "tools_used": "none",
            "final_confidence": 1.0,
        }
        return self.store.save_memory(
            text=memory_text,
            user_id=user_id,
            metadata=metadata,
        )

    def save_user_preference(
        self,
        user_id: str,
        preference: str,
    ) -> str:
        """
        Save a standalone user preference for user_id.

        Tagged distinctly (memory_type="preference") so retrieval can distinguish preferences from task outcomes.
        Returns the assigned memory UUID.
        """
        clean_pref = preference.strip()
        if not clean_pref:
            raise ValueError("Preference text cannot be empty.")

        memory_text = f"User Preference: {clean_pref}"
        metadata: dict[str, Any] = {
            "memory_type": "preference",
            "task_type": "user_preference",
            "outcome": "success",
            "selected_agent": "system",
            "tools_used": "none",
            "final_confidence": 1.0,
        }
        return self.store.save_memory(
            text=memory_text,
            user_id=user_id,
            metadata=metadata,
        )

    # ── recall & planning prompt ──────────────────────────────────────────────

    def recall_for_task(
        self,
        task: str,
        user_id: str = "default",
        limit: int | None = None,
        task_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return up to ``limit`` relevant past memories for this user.

        Applies the configured max_distance filter so weak matches are
        excluded before they reach the prompt.
        """
        return self.store.search_memories(
            query=task,
            user_id=user_id,
            limit=limit or settings.memory_top_k,
            task_type=task_type,
        )

    def format_recalled_memories(
        self,
        task: str,
        user_id: str = "default",
        limit: int | None = None,
        task_type: str | None = None,
    ) -> str:
        """Alias for format_planning_context."""
        return self.format_planning_context(task=task, user_id=user_id, limit=limit, task_type=task_type)

    def format_planning_context(
        self,
        task: str,
        user_id: str = "default",
        limit: int | None = None,
        task_type: str | None = None,
    ) -> str:
        """
        Build the structured prior-context block injected into the supervisor's
        planning prompt.

        Returns an empty string when no relevant memories exist so the caller
        can safely skip the injection.

        Template
        ────────
        ── Prior context (top-{k} similar tasks) ──────────────────────────
        [1] {task_type} task — outcome: {outcome} — agent: {selected_agent}
            Tools used: {tools_used}
            What worked: {answer_summary}
            Confidence: {confidence:.2f}

        [2] …

        Guidance: prefer approaches that worked in similar past tasks.
        ────────────────────────────────────────────────────────────────────
        """
        memories = self.recall_for_task(
            task=task,
            user_id=user_id,
            limit=limit,
            task_type=task_type,
        )
        if not memories:
            return ""

        effective_k = len(memories)
        lines: list[str] = [
            f"── Prior context (top-{effective_k} similar task{'s' if effective_k != 1 else ''}) "
            + "─" * max(0, 68 - 30 - len(str(effective_k))),
        ]

        for idx, mem in enumerate(memories, 1):
            meta = mem.get("metadata") or {}
            text = mem.get("text", "")

            task_type_label = meta.get("task_type", "unknown")
            outcome = meta.get("outcome", "unknown")
            agent = meta.get("selected_agent", "unknown")
            tools = meta.get("tools_used", "none")
            confidence = float(meta.get("final_confidence", 0.5))
            distance = mem.get("distance", 1.0)

            # Extract the "Answer summary:" section from the stored text.
            answer_summary = _extract_answer_summary(text, _MAX_SNIPPET_CHARS)

            lines.append(
                f"[{idx}] {task_type_label} task — outcome: {outcome} "
                f"— agent: {agent} (similarity: {1 - distance:.2f})"
            )
            lines.append(f"    Tools used: {tools}")
            lines.append(f"    What worked: {answer_summary}")
            lines.append(f"    Confidence: {confidence:.2f}")
            if idx < effective_k:
                lines.append("")  # blank line between entries

        lines.append("")
        lines.append("Guidance: prefer approaches that worked in similar past tasks.")
        lines.append(_SEPARATOR)

        return "\n".join(lines)

    # ── list / delete / stats for UI ──────────────────────────────────────────

    def list_memories(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return paginated memories for user_id, sorted by importance desc."""
        return self.store.list_memories(user_id=user_id, limit=limit, offset=offset)

    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Delete one memory owned by user_id. Returns True on success."""
        return self.store.delete_memory(memory_id=memory_id, user_id=user_id)

    def delete_all(self, user_id: str) -> int:
        """Delete all memories for user_id. Returns deleted count."""
        return self.store.delete_all_for_user(user_id=user_id)

    def apply_feedback(
        self,
        memory_id: str,
        user_id: str,
        feedback: float,
    ) -> bool:
        """Record explicit user feedback (−1 / 0 / +1) for a memory."""
        return self.store.apply_user_feedback(
            memory_id=memory_id,
            user_id=user_id,
            feedback=feedback,
        )

    def get_stats(self, user_id: str) -> MemoryStats:
        """Return aggregate memory statistics for the dashboard."""
        return self.store.get_stats(user_id=user_id)

    def expire_old_memories(self, user_id: str) -> int:
        """Expire low-importance old memories for user_id."""
        return self.store.expire_low_importance(user_id=user_id)

    def consolidate(self, user_id: str) -> int:
        """Merge near-duplicate memories for user_id."""
        return self.store.consolidate_duplicates(user_id=user_id)

    def count(self, user_id: str) -> int:
        """Return the total number of memories for user_id."""
        return self.store.count_for_user(user_id=user_id)


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_answer_summary(text: str, max_chars: int) -> str:
    """
    Pull the content after "Answer summary:" in the stored memory text.
    Falls back to the last ``max_chars`` chars of the whole text.
    """
    marker = "Answer summary:"
    idx = text.find(marker)
    if idx != -1:
        snippet = text[idx + len(marker):].strip()
    else:
        snippet = text.strip()

    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + "…"
    # Collapse newlines to single spaces for inline display.
    return " ".join(snippet.split())
