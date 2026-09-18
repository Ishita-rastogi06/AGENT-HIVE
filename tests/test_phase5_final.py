"""
Phase 5 End-to-End Test Suite for AgentHive.

Test Coverage:
1. Plan Validity: Supervisor produces parseable TaskPlan across diverse input tasks.
2. Reviewer Catches Bad Outputs: Deliberately flawed specialist output triggers retry/escalate with low confidence.
3. Memory Injection for Repeated Tasks: Recalled long-term context is injected into supervisor prompt.
4. Graceful Mid-Task Failure Recovery: Redis and ChromaDB mid-task exceptions fall back gracefully without crashing the execution workflow.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from src.agents.reviewer import Reviewer
from src.agents.supervisor import Supervisor
from src.config import settings
from src.llm.ollama_client import OllamaClient
from src.orchestration.graph import build_graph
from src.orchestration.state import AgentHiveState


def test_supervisor_plan_validity_across_diverse_inputs():
    """Verify that Supervisor produces a valid, parseable TaskPlan across diverse input tasks."""
    supervisor = Supervisor(llm=OllamaClient())

    test_tasks = [
        "Create a Python function to calculate Fibonacci sequence.",
        "Analyze SQL database performance metrics and suggest indexing strategy.",
        "Research transformer architectures and write a summary report.",
        "Execute a Python calculation and format the results as a Markdown table.",
    ]

    mock_valid_plan = json.dumps({
        "objective": "Test Task",
        "reasoning": "Standard plan",
        "subtasks": [
            {
                "id": "t1",
                "title": "Subtask 1",
                "instruction": "Perform step 1",
                "assigned_agent": "coder",
                "expected_output": "Code solution",
                "status": "pending",
            }
        ],
        "execution_order": ["t1"],
    })

    with patch("src.llm.ollama_client.OllamaClient.ask", return_value=mock_valid_plan):
        for task_prompt in test_tasks:
            state: AgentHiveState = {
                "task": task_prompt,
                "user_id": "test_user",
                "task_id": "t_plan_val",
                "trace": [],
            }
            updates = supervisor.run(state)
            plan = updates.get("task_plan")
            assert plan is not None
            assert plan.get("objective") is not None
            assert len(plan.get("subtasks", [])) >= 1


def test_reviewer_catches_bad_specialist_output():
    """Verify that Reviewer returns 'retry' or 'escalate' and low confidence when specialist output is bad."""
    reviewer = Reviewer(llm=OllamaClient())

    state: AgentHiveState = {
        "task": "Write a robust SQL query to select active users.",
        "task_id": "t_bad_out",
        "user_id": "u1",
        "selected_agent": "coder",
        "specialist_output": "Error: SyntaxError: invalid syntax near 'SELECT'",
        "current_subtask": {"id": "t1", "instruction": "Write SQL query", "assigned_agent": "coder"},
        "retry_count": 0,
        "max_retries": 2,
        "trace": [],
    }

    mock_bad_review = json.dumps({
        "decision": "retry",
        "confidence": 0.35,
        "feedback": "The output contains syntax errors and is incomplete.",
        "issues": ["Syntax error in generated SQL"],
    })

    with patch("src.llm.ollama_client.OllamaClient.ask", return_value=mock_bad_review):
        updates = reviewer.run(state)
        review = updates.get("review", {})
        assert review.get("decision") == "retry"
        assert review.get("confidence", 1.0) < 0.60
        assert len(review.get("issues", [])) >= 1


def test_repeated_task_memory_injection():
    """Verify that long-term semantic memory from prior runs is recalled and injected into supervisor state."""
    from src.agents.memory_recall import MemoryRecallAgent

    recall_agent = MemoryRecallAgent()
    state: AgentHiveState = {
        "task": "Configure Redis persistence settings",
        "user_id": "u_repeat",
        "task_id": "t_repeat_1",
        "trace": [],
    }

    with patch("src.memory.vector_db.MemoryVectorStore.search_memories") as mock_search:
        mock_search.return_value = [
            {
                "memory_id": "m1",
                "text": "For Redis persistence, set appendonly yes in redis.conf.",
                "distance": 0.1,
                "metadata": {"task_type": "coding", "outcome": "success"},
            }
        ]

        updates = recall_agent.run(state)
        memory_ctx = updates.get("memory_context", "")
        assert "appendonly yes" in memory_ctx


def test_graceful_mid_task_failure_recovery(tmp_path):
    """
    Verify that if Redis or ChromaDB throw exceptions mid-task execution,
    the graph workflow degrades gracefully without crashing.
    """
    db_file = tmp_path / "graceful_fail.db"
    graph = build_graph()
    config = {"configurable": {"thread_id": "task_graceful_fail"}}

    mock_plan = json.dumps({
        "objective": "Graceful failure test",
        "reasoning": "Resilience test",
        "subtasks": [
            {
                "id": "t1",
                "title": "Subtask 1",
                "instruction": "Say hello",
                "assigned_agent": "research",
                "expected_output": "Hello",
                "status": "pending",
            }
        ],
        "execution_order": ["t1"],
    })
    mock_approve = json.dumps({"decision": "approve", "confidence": 0.9, "feedback": "OK", "issues": []})

    def mock_ask(prompt, system=""):
        s_lower = system.lower()
        p_lower = prompt.lower()
        if "decompose" in p_lower or "subtask" in p_lower:
            return mock_plan
        if "reviewer" in s_lower or "quality" in s_lower or "decision" in p_lower or "review" in p_lower:
            return mock_approve
        return "Resilient response."

    # Simulate ChromaDB and Redis raising exceptions mid-task
    with patch("src.memory.vector_db.MemoryVectorStore.search_memories", side_effect=RuntimeError("ChromaDB disconnected")), \
         patch("src.memory.short_term.ShortTermStore.add_subtask_result", side_effect=ConnectionError("Redis connection lost")), \
         patch("src.llm.ollama_client.OllamaClient.ask", side_effect=mock_ask):

        res = graph.invoke(
            {
                "task": "Test graceful failure recovery",
                "user_id": "user_resilience",
                "task_id": "task_graceful_fail",
                "db_path": str(db_file),
                "trace": [],
            },
            config=config,
        )

        assert res is not None
        assert res.get("workflow_status") in ("completed", "executing", "escalated")
