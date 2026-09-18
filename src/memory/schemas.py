"""
Memory record schemas for AgentHive's two-tier memory system.

Tier 1 — Short-term (Redis)
    ShortTermMemory  : task-scoped working memory stored in Redis with TTL.
                       Holds the current plan, subtask outputs, and errors.

Tier 2 — Long-term (ChromaDB)
    LongTermMemoryRecord : a single embedded memory entry with full provenance.
    ImportanceScore      : composite score used for expiry and ranking.
    MemoryQuery          : parameters for a semantic recall operation.
    MemoryStats          : aggregate statistics shown on the memory dashboard.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from typing_extensions import TypedDict


# ── Short-term (Redis) ────────────────────────────────────────────────────────

class SubtaskResult(TypedDict, total=False):
    """Output captured from one completed subtask."""
    subtask_id: str
    title: str
    output: str
    confidence: float
    tools_used: list[str]
    error: str


class ShortTermMemory(TypedDict, total=False):
    """
    Task-scoped working memory stored in Redis under key:
        agenthive:stm:<user_id>:<task_id>

    Written by MemorySaveAgent on each subtask completion and cleared when
    the task finishes or after SHORT_TERM_TTL_SECONDS.
    """
    task_id: str
    user_id: str
    task: str                            # original user request
    task_type: str                       # research / coder / data / writer
    plan_objective: str                  # supervisor's task objective
    plan_reasoning: str
    subtask_results: list[SubtaskResult] # accumulated completed outputs
    intermediate_notes: list[str]        # agent-generated notes mid-task
    errors: list[str]                    # non-fatal errors encountered
    workflow_status: str
    started_at: str                      # ISO-8601
    updated_at: str                      # ISO-8601


# ── Long-term (ChromaDB) ──────────────────────────────────────────────────────

class LongTermMemoryRecord(TypedDict, total=False):
    """
    A single entry in ChromaDB's agenthive_memories collection.

    The ``text`` field is what gets embedded.  Everything else lives in the
    ChromaDB ``metadata`` dict alongside ``text``.
    """
    # ChromaDB document fields
    id: str                              # UUID
    text: str                            # embedded document text

    # Required provenance fields (every save must include these)
    user_id: str
    task_type: str                       # research | coder | data | writer
    outcome: Literal["success", "failure", "escalated"]
    timestamp: str                       # ISO-8601 UTC

    # Enrichment fields
    selected_agent: str
    tools_used: str                      # comma-joined tool names
    workflow_trace: str                  # " → "-joined trace steps
    final_confidence: float

    # Importance scoring (updated on each recall hit)
    importance_score: float              # 0.0–1.0
    retrieval_count: int                 # how many times this memory was recalled
    user_feedback: float                 # explicit user rating −1 / 0 / +1


class ImportanceScore(TypedDict):
    """
    Composite importance score for one memory entry.

    Formula (all terms normalised to [0, 1]):
        importance = 0.40 * recency
                   + 0.35 * retrieval_frequency
                   + 0.25 * user_feedback_normalised
    """
    memory_id: str
    recency: float            # 1.0 = today,  decays toward 0 over expiry_days
    retrieval_frequency: float# normalised retrieval_count
    user_feedback: float      # mapped: +1 → 1.0,  0 → 0.5,  −1 → 0.0
    composite: float          # weighted sum


# ── Query / stats ─────────────────────────────────────────────────────────────

class MemoryQuery(TypedDict, total=False):
    """Parameters for a semantic recall operation."""
    query_text: str
    user_id: str
    task_type: str           # optional filter
    outcome: str             # optional filter: success | failure | escalated
    top_k: int               # max results to return
    max_distance: float      # max cosine distance


class MemoryStats(TypedDict, total=False):
    """Aggregate statistics shown on the memory dashboard."""
    user_id: str
    total_memories: int
    success_count: int
    failure_count: int
    most_used_agent: str
    avg_importance: float
    oldest_memory_date: str
    newest_memory_date: str
