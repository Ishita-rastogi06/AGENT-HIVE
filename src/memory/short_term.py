"""
Short-term (Redis) memory for AgentHive.

Each task gets its own key: ``agenthive:stm:<user_id>:<task_id>``

The value is a JSON-serialised ShortTermMemory dict.  The key expires
automatically after ``settings.short_term_ttl_seconds``.

Graceful degradation
────────────────────
If Redis is unreachable (no REDIS_URL, or connection refused), every method
returns a safe no-op result and logs a WARNING.  The workflow continues
without short-term memory rather than crashing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from src.config import settings
from src.memory.schemas import ShortTermMemory, SubtaskResult

logger = logging.getLogger(__name__)

_PREFIX = "agenthive:stm"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redis_client():
    """
    Return a connected Redis client or None if unavailable.

    Deliberately instantiated lazily per-call so the import of this module
    never crashes even when redis-py is not installed.
    """
    url = settings.redis_url
    if not url:
        return None
    try:
        import redis  # type: ignore
        client = redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        return client
    except ImportError:
        logger.warning(
            "ShortTermMemory: redis package not installed — "
            "run `pip install redis` to enable short-term memory."
        )
        return None
    except Exception as exc:
        logger.warning(
            "ShortTermMemory: Redis unavailable at %s — %s. "
            "Short-term memory is disabled for this session.",
            settings.redis_url,
            exc,
        )
        return None


def _key(user_id: str, task_id: str) -> str:
    return f"{_PREFIX}:{user_id}:{task_id}"


class ShortTermStore:
    """
    Task-scoped working memory backed by Redis.

    All methods are safe to call when Redis is unavailable — they return empty
    results and log a warning instead of raising.

    Usage pattern in a workflow:
        store = ShortTermStore()
        task_id = store.create(user_id, task, task_type)   # start of workflow
        store.add_subtask_result(user_id, task_id, result) # after each subtask
        context = store.get_context_summary(user_id, task_id)  # for recall
        store.clear(user_id, task_id)                      # on completion
    """

    def __init__(self) -> None:
        self._ttl = settings.short_term_ttl_seconds

    # ── create / read / clear ─────────────────────────────────────────────────

    def create(
        self,
        user_id: str,
        task: str,
        task_id: str | None = None,
        task_type: str = "research",
        plan_objective: str = "",
        plan_reasoning: str = "",
    ) -> str:
        """
        Initialise a new short-term memory slot and return the task_id.

        Returns a UUID task_id even when Redis is unavailable — the caller can
        still track the task; context will just come from long-term memory only.
        """
        effective_task_id = task_id or str(uuid4())
        record: ShortTermMemory = {
            "task_id": effective_task_id,
            "user_id": user_id,
            "task": task,
            "task_type": task_type,
            "plan_objective": plan_objective,
            "plan_reasoning": plan_reasoning,
            "subtask_results": [],
            "intermediate_notes": [],
            "errors": [],
            "workflow_status": "planning",
            "started_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        self._write(user_id, effective_task_id, record)
        return effective_task_id

    def get(self, user_id: str, task_id: str) -> ShortTermMemory | None:
        """Return the current short-term record or None."""
        raw = self._read(user_id, task_id)
        return raw  # type: ignore[return-value]

    def clear(self, user_id: str, task_id: str) -> None:
        """Delete the short-term key immediately (e.g. on task completion)."""
        client = _redis_client()
        if client is None:
            return
        try:
            client.delete(_key(user_id, task_id))
        except Exception as exc:
            logger.warning("ShortTermStore.clear failed: %s", exc)

    # ── mutation helpers ──────────────────────────────────────────────────────

    def add_subtask_result(
        self,
        user_id: str,
        task_id: str,
        result: SubtaskResult,
    ) -> None:
        """Append one completed subtask result to the working memory."""
        record = self._read(user_id, task_id)
        if record is None:
            return
        existing: list = record.get("subtask_results") or []  # type: ignore[assignment]
        existing.append(result)
        record["subtask_results"] = existing
        record["updated_at"] = _now_iso()
        self._write(user_id, task_id, record)

    def add_note(self, user_id: str, task_id: str, note: str) -> None:
        """Append a free-form intermediate note."""
        record = self._read(user_id, task_id)
        if record is None:
            return
        notes: list = record.get("intermediate_notes") or []  # type: ignore[assignment]
        notes.append(note)
        record["intermediate_notes"] = notes
        record["updated_at"] = _now_iso()
        self._write(user_id, task_id, record)

    def add_error(self, user_id: str, task_id: str, error: str) -> None:
        """Record a non-fatal error encountered during the task."""
        record = self._read(user_id, task_id)
        if record is None:
            return
        errors: list = record.get("errors") or []  # type: ignore[assignment]
        errors.append(error)
        record["errors"] = errors
        record["updated_at"] = _now_iso()
        self._write(user_id, task_id, record)

    def update_status(
        self,
        user_id: str,
        task_id: str,
        status: str,
    ) -> None:
        """Update the workflow_status field."""
        record = self._read(user_id, task_id)
        if record is None:
            return
        record["workflow_status"] = status
        record["updated_at"] = _now_iso()
        self._write(user_id, task_id, record)

    # ── context summary for supervisor ───────────────────────────────────────

    def get_context_summary(
        self,
        user_id: str,
        task_id: str,
        max_chars: int = 800,
    ) -> str:
        """
        Return a concise summary of in-progress work to inject into the
        supervisor's planning prompt.

        Returns an empty string when nothing is available.
        """
        record = self._read(user_id, task_id)
        if not record:
            return ""

        parts: list[str] = ["Current task working memory:"]

        objective = record.get("plan_objective", "")
        if objective:
            parts.append(f"  Objective: {objective}")

        subtask_results: list[dict] = record.get("subtask_results") or []  # type: ignore[assignment]
        if subtask_results:
            parts.append(f"  Completed subtasks: {len(subtask_results)}")
            for sr in subtask_results[-3:]:  # last 3 only to stay concise
                title = sr.get("title", sr.get("subtask_id", "unknown"))
                confidence = sr.get("confidence", 0.0)
                parts.append(f"    • {title} (confidence {confidence:.2f})")

        notes: list[str] = record.get("intermediate_notes") or []  # type: ignore[assignment]
        if notes:
            parts.append("  Notes: " + "; ".join(notes[-2:]))

        errors: list[str] = record.get("errors") or []  # type: ignore[assignment]
        if errors:
            parts.append("  Errors encountered: " + "; ".join(errors[-2:]))

        summary = "\n".join(parts)
        if len(summary) > max_chars:
            summary = summary[:max_chars].rstrip() + "..."
        return summary

    # ── low-level Redis I/O ───────────────────────────────────────────────────

    def _write(self, user_id: str, task_id: str, record: dict) -> None:
        client = _redis_client()
        if client is None:
            return
        try:
            client.setex(
                _key(user_id, task_id),
                self._ttl,
                json.dumps(record, ensure_ascii=False, default=str),
            )
        except Exception as exc:
            logger.warning("ShortTermStore._write failed: %s", exc)

    def _read(self, user_id: str, task_id: str) -> dict | None:
        client = _redis_client()
        if client is None:
            return None
        try:
            raw = client.get(_key(user_id, task_id))
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning("ShortTermStore._read failed: %s", exc)
            return None

    # ── availability check ────────────────────────────────────────────────────

    @staticmethod
    def is_available() -> bool:
        """Return True if Redis can be reached right now."""
        return _redis_client() is not None
