"""
Comprehensive unit and integration test suite for AgentHive Phase 2 Memory System:
- Short-term task working memory (Redis)
- Long-term semantic memory (ChromaDB)
- Per-user scoping and isolation
- Memory recall & Supervisor planning prompt injection
- Importance scoring, consolidation, expiration
- Graceful degradation when services are offline
- Repeated task memory improvement test
- History migration script test
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.memory_recall import MemoryRecallAgent
from src.agents.memory_save import MemorySaveAgent
from src.agents.supervisor import Supervisor
from src.llm.ollama_client import OllamaClient
from src.memory.schemas import ImportanceScore
from src.memory.short_term import ShortTermStore
from src.memory.vector_db import MemoryVectorStore, _now_iso
from src.orchestration.state import AgentHiveState
from src.tools.memory_tool import AgentMemoryTool
from scripts.migrate_history_to_chroma import migrate_history_to_chroma


# ── Short-Term Working Memory (Redis) Tests ─────────────────────────────────

def test_short_term_store_fallback():
    """Verify ShortTermStore degrades gracefully without raising when Redis is missing."""
    with patch("src.memory.short_term._redis_client", return_value=None):
        store = ShortTermStore()
        task_id = store.create(user_id="user_test", task="Test request")
        assert task_id is not None
        assert len(task_id) > 0

        # All operations should be no-ops and return empty/safe results
        assert store.get("user_test", task_id) is None
        store.add_subtask_result("user_test", task_id, {"subtask_id": "t1", "output": "ok"})
        store.add_note("user_test", task_id, "test note")
        store.add_error("user_test", task_id, "test error")
        store.update_status("user_test", task_id, "executing")
        assert store.get_context_summary("user_test", task_id) == ""
        store.clear("user_test", task_id)


def test_short_term_store_mock_redis():
    """Test full ShortTermStore CRUD cycle when Redis client is available."""
    storage: dict[str, str] = {}

    mock_client = MagicMock()
    def mock_setex(key, ttl, value):
        storage[key] = value
    def mock_get(key):
        return storage.get(key)
    def mock_delete(key):
        storage.pop(key, None)

    mock_client.setex.side_effect = mock_setex
    mock_client.get.side_effect = mock_get
    mock_client.delete.side_effect = mock_delete

    with patch("src.memory.short_term._redis_client", return_value=mock_client):
        store = ShortTermStore()
        user_id = "user_123"
        task_id = "task_abc"

        # Create
        created_id = store.create(
            user_id=user_id,
            task="Compute fibonacci",
            task_id=task_id,
            task_type="coder",
            plan_objective="Calculate fibonacci sequence",
        )
        assert created_id == task_id
        assert f"agenthive:stm:{user_id}:{task_id}" in storage

        # Get
        record = store.get(user_id, task_id)
        assert record is not None
        assert record["task"] == "Compute fibonacci"

        # Add subtask result
        store.add_subtask_result(user_id, task_id, {
            "subtask_id": "t1",
            "title": "Implement fibonacci function",
            "output": "def fib(n): ...",
            "confidence": 0.95,
        })
        record = store.get(user_id, task_id)
        assert len(record["subtask_results"]) == 1

        # Summary
        summary = store.get_context_summary(user_id, task_id)
        assert "Calculate fibonacci sequence" in summary
        assert "Implement fibonacci function" in summary

        # Clear
        store.clear(user_id, task_id)
        assert store.get(user_id, task_id) is None


# ── Long-Term Memory (ChromaDB) & Scoping Tests ──────────────────────────────

def test_long_term_user_scoping(tmp_path):
    """Confirm memories saved for user A are invisible to user B."""
    mock_embeddings = [[0.1] * 768]
    with patch("src.memory.embedder.OllamaEmbedder.embed", return_value=mock_embeddings):
        store = MemoryVectorStore(storage_path=tmp_path / "chroma_scoping")

        # Save for User A
        id_a = store.save_memory(
            text="user_A secret preference: prefers Python 3.13",
            user_id="user_A",
            metadata={"task_type": "coder", "outcome": "success"},
        )

        # Save for User B
        id_b = store.save_memory(
            text="user_B secret preference: prefers Rust",
            user_id="user_B",
            metadata={"task_type": "coder", "outcome": "success"},
        )

        # Search for User A
        memories_a = store.search_memories(query="Python preference", user_id="user_A")
        assert len(memories_a) == 1
        assert "user_A" in memories_a[0]["text"]

        # Search for User B
        memories_b = store.search_memories(query="Python preference", user_id="user_B")
        # User B should NOT receive User A's memory
        for m in memories_b:
            assert m["metadata"]["user_id"] == "user_B"

        # Count for User A & User B
        assert store.count_for_user("user_A") == 1
        assert store.count_for_user("user_B") == 1

        # Delete all for User A
        store.delete_all_for_user("user_A")
        assert store.count_for_user("user_A") == 0
        assert store.count_for_user("user_B") == 1


def test_importance_scoring():
    """Verify composite importance score formula: 0.40*recency + 0.35*retrieval_freq + 0.25*user_feedback."""
    store = MemoryVectorStore.__new__(MemoryVectorStore)

    # Today's memory, 0 retrievals, neutral feedback (0.0 -> 0.5 norm)
    meta = {
        "timestamp": _now_iso(),
        "retrieval_count": 0,
        "user_feedback": 0.0,
    }
    score: ImportanceScore = store.compute_importance(meta, expiry_days=30)
    # recency=1.0, freq=0.0, feedback_norm=0.5 -> 0.40(1) + 0.35(0) + 0.25(0.5) = 0.525
    assert abs(score["composite"] - 0.525) < 0.01

    # High retrieval, positive feedback (+1.0 -> 1.0 norm)
    meta_high = {
        "timestamp": _now_iso(),
        "retrieval_count": 100,
        "user_feedback": 1.0,
    }
    score_high = store.compute_importance(meta_high, expiry_days=30)
    # recency=1.0, freq=1.0, feedback_norm=1.0 -> 0.40(1) + 0.35(1) + 0.25(1) = 1.0
    assert abs(score_high["composite"] - 1.0) < 0.01


# ── Supervisor Planning Injection & Repeated Task Test ───────────────────────

def test_repeated_task_memory_injection(tmp_path):
    """
    Run a task, save it to memory, and confirm the second run's supervisor
    planning step injects prior memory context into the prompt template.
    """
    mock_embeddings = [[0.1] * 768]
    test_db_path = tmp_path / "chroma_repeat"
    with patch("src.memory.embedder.OllamaEmbedder.embed", return_value=mock_embeddings):

        tool = AgentMemoryTool(storage_path=test_db_path)
        user_id = "test_developer"
        task_text = "Write a python function to print a palindrome"

        # 1. Run 1: Save completed task to memory
        mem_id = tool.save_completed_task(
            task=task_text,
            final_answer="def is_palindrome(s):\n    return s == s[::-1]\nassert is_palindrome('madam')",
            selected_agent="coder",
            trace=["Supervisor → coder", "Coding specialist completed", "Reviewer completed"],
            user_id=user_id,
            task_type="coder",
            outcome="success",
            tools_used=["sandbox_code_execution"],
            final_confidence=0.95,
        )
        assert mem_id is not None

        # 2. Run 2: Recall memory for the exact same task
        memory_recall_node = MemoryRecallAgent()
        memory_recall_node.memory_tool = tool  # use test tool instance

        state_1: AgentHiveState = {
            "task": task_text,
            "user_id": user_id,
            "trace": [],
        }
        recall_result = memory_recall_node.run(state_1)

        memory_context = recall_result.get("memory_context", "")
        assert memory_context != ""
        assert "Prior context" in memory_context
        assert "coder task — outcome: success" in memory_context
        assert "sandbox_code_execution" in memory_context

        # 3. Test Supervisor planning prompt injection
        mock_llm = MagicMock(spec=OllamaClient)
        mock_llm.ask.return_value = json.dumps({
            "objective": "Write palindrome function based on prior pattern",
            "reasoning": "Using s == s[::-1] strategy from recalled memory",
            "subtasks": [
                {
                    "id": "t1",
                    "title": "Implement palindrome function",
                    "description": task_text,
                    "assigned_agent": "coder",
                    "dependencies": [],
                    "instruction": "Implement is_palindrome using reverse slice",
                    "expected_output": "Palindrome function with test cases",
                    "status": "pending",
                }
            ],
            "execution_order": ["t1"],
        })

        supervisor = Supervisor(mock_llm)
        state_2: AgentHiveState = {
            "task": task_text,
            "user_id": user_id,
            "memory_context": memory_context,
            "trace": recall_result.get("trace", []),
        }

        sup_result = supervisor.run(state_2)

        # Assert Supervisor received memory context and called LLM with injected prompt
        assert mock_llm.ask.called
        called_prompt, system_prompt = mock_llm.ask.call_args[0]
        assert "Prior context" in called_prompt
        assert "Guidance: prefer approaches that worked in similar past tasks" in called_prompt
        assert sup_result["task_plan"]["objective"] == "Write palindrome function based on prior pattern"


# ── Graceful Degradation Test ────────────────────────────────────────────────

def test_graceful_degradation_on_memory_failure():
    """Ensure MemoryRecallAgent and MemorySaveAgent degrade gracefully if storage raises errors."""
    with patch("src.tools.memory_tool.AgentMemoryTool.format_planning_context", side_effect=RuntimeError("DB Connection Error")), \
         patch("src.tools.memory_tool.AgentMemoryTool.save_completed_task", side_effect=RuntimeError("DB Connection Error")):

        recall_agent = MemoryRecallAgent()
        save_agent = MemorySaveAgent()

        state: AgentHiveState = {
            "task": "Test fallback task",
            "user_id": "fallback_user",
            "final_answer": "Some answer",
            "trace": [],
        }

        # Memory recall failure shouldn't raise exception
        recall_res = recall_agent.run(state)
        assert recall_res["memory_context"] == ""
        assert any("failed" in t for t in recall_res["trace"])

        # Memory save failure shouldn't raise exception
        save_res = save_agent.run(state)
        assert any("could not be saved" in t for t in save_res["trace"])


# ── Migration Script Test ────────────────────────────────────────────────────

def test_migration_script(tmp_path):
    """Test backfilling task_history.json into ChromaDB."""
    sample_history = [
        {
            "timestamp": "27 Jul 2026 · 01:28 AM",
            "task": "Write a python code to print a palindrome",
            "selected_agent": "coder",
            "trace": ["Supervisor → coder", "Completed task saved to long-term memory"],
            "final_answer": "def is_palindrome(s): return s == s[::-1]",
        }
    ]
    history_file = tmp_path / "task_history.json"
    history_file.write_text(json.dumps(sample_history), encoding="utf-8")

    mock_embeddings = [[0.1] * 768]
    test_db_path = tmp_path / "chroma_migrate"
    with patch("src.memory.embedder.OllamaEmbedder.embed", return_value=mock_embeddings), \
         patch("scripts.migrate_history_to_chroma.AgentMemoryTool", lambda: AgentMemoryTool(storage_path=test_db_path)):

        migrated_count = migrate_history_to_chroma(user_id="migration_user", history_file=history_file)
        assert migrated_count == 1

        tool = AgentMemoryTool(storage_path=test_db_path)
        memories = tool.list_memories(user_id="migration_user")
        assert len(memories) == 1
        assert "palindrome" in memories[0]["text"]


# ── Standalone Domain Fact & User Preference Tests ───────────────────────────

def test_save_domain_fact_and_user_preference(tmp_path):
    """Confirm save_domain_fact and save_user_preference store distinct memory_type entries."""
    mock_embeddings = [[0.1] * 768]
    test_db_path = tmp_path / "chroma_facts_prefs"
    with patch("src.memory.embedder.OllamaEmbedder.embed", return_value=mock_embeddings):
        tool = AgentMemoryTool(storage_path=test_db_path)
        user_id = "user_facts_test"

        # 1. Save Domain Fact
        fact_id = tool.save_domain_fact(
            user_id=user_id,
            fact="PostgreSQL 16 supports pgvector for vector search.",
            source="documentation",
        )
        assert fact_id is not None

        # 2. Save User Preference
        pref_id = tool.save_user_preference(
            user_id=user_id,
            preference="Prefers concise code without unnecessary comments.",
        )
        assert pref_id is not None

        # 3. Retrieve and verify metadata tagging
        memories = tool.list_memories(user_id=user_id)
        assert len(memories) == 2

        fact_entry = next(m for m in memories if m["metadata"].get("memory_type") == "fact")
        pref_entry = next(m for m in memories if m["metadata"].get("memory_type") == "preference")

        assert "pgvector" in fact_entry["text"]
        assert fact_entry["metadata"]["source"] == "documentation"
        assert fact_entry["metadata"]["task_type"] == "domain_fact"

        assert "concise code" in pref_entry["text"]
        assert pref_entry["metadata"]["task_type"] == "user_preference"


def test_domain_fact_and_preference_scoping(tmp_path):
    """Confirm standalone facts and preferences are strictly isolated per user_id."""
    mock_embeddings = [[0.1] * 768]
    test_db_path = tmp_path / "chroma_facts_scoping"
    with patch("src.memory.embedder.OllamaEmbedder.embed", return_value=mock_embeddings):
        tool = AgentMemoryTool(storage_path=test_db_path)

        tool.save_domain_fact(user_id="user_alice", fact="Alice's team uses Kubernetes.", source="survey")
        tool.save_user_preference(user_id="user_bob", preference="Bob prefers tab indentation.")

        alice_memories = tool.list_memories(user_id="user_alice")
        bob_memories = tool.list_memories(user_id="user_bob")

        assert len(alice_memories) == 1
        assert "Kubernetes" in alice_memories[0]["text"]
        assert alice_memories[0]["metadata"]["user_id"] == "user_alice"

        assert len(bob_memories) == 1
        assert "tab indentation" in bob_memories[0]["text"]
        assert bob_memories[0]["metadata"]["user_id"] == "user_bob"


def test_empty_fact_and_preference_validation(tmp_path):
    """Verify ValueError is raised when attempting to save empty fact or preference text."""
    tool = AgentMemoryTool.__new__(AgentMemoryTool)
    with pytest.raises(ValueError, match="Fact text cannot be empty"):
        tool.save_domain_fact(user_id="u1", fact="   ")

    with pytest.raises(ValueError, match="Preference text cannot be empty"):
        tool.save_user_preference(user_id="u1", preference="")


def test_memory_save_agent_extracts_content_and_tools():
    """Verify MemorySaveAgent extracts content and tools correctly even on escalated tasks."""
    mock_memory_tool = MagicMock()
    agent = MemorySaveAgent()
    agent.memory_tool = mock_memory_tool

    state = {
        "task": "Build python web scraper",
        "selected_agent": "coder",
        "workflow_status": "escalated",
        "needs_human_review": True,
        "specialist_output": "```python\nimport requests\nprint('hello')\n```",
        "tool_logs": [
            {"tool_name": "python_executor", "success": True},
            {"tool_name": "sqlite_tool", "success": True},
        ],
        "trace": ["Task escalated"],
        "user_id": "test_u",
    }
    agent.run(state)
    mock_memory_tool.save_completed_task.assert_called_once()
    kwargs = mock_memory_tool.save_completed_task.call_args[1]
    assert "import requests" in kwargs["final_answer"]
    assert kwargs["outcome"] == "escalated"
    assert "python_executor" in kwargs["tools_used"]
    assert "sqlite_tool" in kwargs["tools_used"]


def test_save_completed_task_preserves_long_final_answer_without_truncation(tmp_path):
    """Verify that long specialist answers (e.g. >1500 chars) are preserved in full without silent truncation."""
    from src.tools.memory_tool import AgentMemoryTool

    tool = AgentMemoryTool(storage_path=str(tmp_path / "test_long_mem_db"))
    long_heading_and_code = "Print a formatted summary table:\n```python\nprint(df.to_string())\n```"
    long_answer = ("# 1. Setup sqlite3\n" + "x = 100\n" * 120 + f"\n# 5. {long_heading_and_code}")

    mem_id = tool.save_completed_task(
        task="Sales monthly revenue task",
        final_answer=long_answer,
        selected_agent="coder",
        trace=["start"],
        user_id="test_user_long",
    )
    record = tool.store.collection.get(ids=[mem_id], include=["documents"])
    assert record["documents"] and len(record["documents"]) > 0
    saved_text = record["documents"][0]
    assert "Print a formatted summary table" in saved_text
    assert "print(df.to_string())" in saved_text


