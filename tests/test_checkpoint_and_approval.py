"""
Comprehensive unit and integration test suite for AgentHive Phase 3 Human-in-the-Loop Orchestration:
- Pre-execution pause for sensitive operations (before tool call execution).
- Checkpointed state persistence surviving process restart (SqliteSaver).
- Full pause -> approve -> resume lifecycle with Command(resume=...).
- Granular approval levels (Approve Action, Approve Plan, Take Over).
- Approval queue persistence and audit logging.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from src.config import settings
from src.llm.ollama_client import OllamaClient
from src.orchestration.approval import (
    add_pending_approval,
    get_approval,
    get_pending_approvals,
    is_sensitive_operation,
    log_approval_event,
)
from src.orchestration.graph import build_graph
from src.orchestration.state import AgentHiveState


# ── Sensitive Operation Detection Tests ────────────────────────────────────────

def test_is_sensitive_operation_detection():
    """Verify pre-execution detection for python_sandbox, api_call, and db_query tools."""
    # 1. Python sandbox file write
    sens1, reason1 = is_sensitive_operation("python_sandbox", {"code": "with open('out.txt', 'w') as f: f.write('data')"})
    assert sens1 is True
    assert "file-write" in reason1

    safe_py, _ = is_sensitive_operation("python_sandbox", {"code": "print(2 + 2)"})
    assert safe_py is False

    # 2. HTTP API call POST / PUT / DELETE
    sens2, reason2 = is_sensitive_operation("api_call", {"method": "POST", "url": "https://api.github.com/repos"})
    assert sens2 is True
    assert "POST" in reason2

    safe_api, _ = is_sensitive_operation("api_call", {"method": "GET", "url": "https://api.github.com/repos"})
    assert safe_api is False

    # 3. Database query non-read-only or SQL mutation
    sens3, reason3 = is_sensitive_operation("db_query", {"query": "INSERT INTO users (name) VALUES ('Alice');", "read_only": True})
    assert sens3 is True
    assert "INSERT" in reason3

    sens3_b, _ = is_sensitive_operation("db_query", {"query": "SELECT * FROM users;", "read_only": False})
    assert sens3_b is True

    safe_db, _ = is_sensitive_operation("db_query", {"query": "SELECT count(*) FROM users;", "read_only": True})
    assert safe_db is False


# ── Pre-Execution Pause Test ──────────────────────────────────────────────────

def test_sensitive_operation_pauses_before_execution(tmp_path):
    """Confirm a sensitive operation pauses execution BEFORE the specialist or tool executes."""
    sqlite_path = tmp_path / "test_pause_before.db"
    conn = sqlite3.connect(str(sqlite_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_graph(checkpointer=checkpointer)

    task_id = "task_sensitive_test"
    user_id = "sec_user"
    config = {"configurable": {"thread_id": task_id}}

    mock_llm_response = json.dumps({
        "objective": "Insert new user record into database",
        "reasoning": "Database mutation subtask required",
        "subtasks": [
            {
                "id": "t1",
                "title": "Insert user into database",
                "description": "INSERT INTO users (name) VALUES ('Bob');",
                "assigned_agent": "data",
                "dependencies": [],
                "instruction": "INSERT INTO users (name) VALUES ('Bob');",
                "expected_output": "Row inserted.",
                "status": "pending",
            }
        ],
        "execution_order": ["t1"],
    })

    with patch("src.llm.ollama_client.OllamaClient.ask", return_value=mock_llm_response):
        result = graph.invoke(
            {
                "task": "INSERT INTO users (name) VALUES ('Bob');",
                "user_id": user_id,
                "task_id": task_id,
                "db_path": str(sqlite_path),
                "trace": [],
            },
            config=config,
        )

        assert "__interrupt__" in result or result.get("needs_human_review")
        pending = get_pending_approvals(user_id=user_id, db_path=sqlite_path)
        assert len(pending) == 1
        assert pending[0]["task_id"] == task_id
        assert "sensitive_operation" in pending[0]["trigger_reason"] or "INSERT" in pending[0]["trigger_reason"]

    conn.close()


# ── Checkpointed State Persistence & Process Restart Test ─────────────────────

def test_checkpointed_pause_resume_process_restart(tmp_path):
    """
    Test full pause -> checkpoint to disk -> simulated process restart (fresh graph compile) -> resume.
    Proves state persistence survives across process restarts and Streamlit reruns.
    """
    db_file = tmp_path / "agent_hive_process_restart.db"
    task_id = "task_restart_123"
    user_id = "restart_user"
    config = {"configurable": {"thread_id": task_id}}

    # Phase A: Initial process execution that pauses
    conn1 = sqlite3.connect(str(db_file), check_same_thread=False)
    checkpointer1 = SqliteSaver(conn1)
    graph1 = build_graph(checkpointer=checkpointer1)

    mock_llm_plan = json.dumps({
        "objective": "Send POST request to external API",
        "reasoning": "API subtask",
        "subtasks": [
            {
                "id": "t1",
                "title": "Post payload to webhook",
                "description": "Send POST request",
                "assigned_agent": "research",
                "dependencies": [],
                "instruction": "Send POST request to https://api.github.com/repos",
                "expected_output": "Webhook response",
                "status": "pending",
            }
        ],
        "execution_order": ["t1"],
    })

    with patch("src.llm.ollama_client.OllamaClient.ask", return_value=mock_llm_plan):
        res_a = graph1.invoke(
            {
                "task": "Send POST request to https://api.github.com/repos",
                "user_id": user_id,
                "task_id": task_id,
                "db_path": str(db_file),
                "pause_mode": "step_by_step",
                "trace": [],
            },
            config=config,
        )
        assert "__interrupt__" in res_a

    conn1.close()  # Simulate process shutdown / end of request

    # Phase B: Simulated Process Restart
    # Re-open database connection and re-compile graph fresh from SQLite file on disk
    conn2 = sqlite3.connect(str(db_file), check_same_thread=False)
    checkpointer2 = SqliteSaver(conn2)
    fresh_graph = build_graph(checkpointer=checkpointer2)

    # Inspect state snapshot reloaded from disk
    snapshot = fresh_graph.get_state(config)
    assert snapshot.next == ("approval_check",)

    # Phase C: Resume execution via Command(resume=...)
    mock_reviewer_approve = json.dumps({"decision": "approve", "confidence": 0.95, "feedback": "Approved POST request", "issues": []})
    with patch("src.llm.ollama_client.OllamaClient.ask", return_value=mock_reviewer_approve):
        res_b = fresh_graph.invoke(
            Command(resume={"action": "approve_action", "decision": "approve"}),
            config=config,
        )
        assert res_b["workflow_status"] == "completed"
        assert any("RESUMED BY HUMAN" in t for t in res_b.get("trace", []))

    conn2.close()


# ── Granular Approval Levels & Take Over Test ──────────────────────────────────

def test_take_over_approval_level(tmp_path):
    """Test 'Take Over' approval level where human reviewer supplies custom output text."""
    db_file = tmp_path / "takeover_test.db"

    conn = sqlite3.connect(str(db_file), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_graph(checkpointer=checkpointer)

    task_id = "task_takeover_999"
    user_id = "reviewer_user"
    config = {"configurable": {"thread_id": task_id}}

    mock_plan = json.dumps({
        "objective": "Write report",
        "reasoning": "Single task",
        "subtasks": [
            {
                "id": "t1",
                "title": "Write summary report",
                "description": "Write report",
                "assigned_agent": "writer",
                "dependencies": [],
                "instruction": "Write report",
                "expected_output": "Report",
                "status": "pending",
            }
        ],
        "execution_order": ["t1"],
    })

    with patch("src.llm.ollama_client.OllamaClient.ask", return_value=mock_plan):
        graph.invoke(
            {
                "task": "Write report",
                "user_id": user_id,
                "task_id": task_id,
                "db_path": str(db_file),
                "pause_mode": "step_by_step",
                "trace": [],
            },
            config=config,
        )

    # Resume with 'Take Over' action and custom human text
    human_text = "Executive Summary: Human approved manual response."
    mock_reviewer_approve = json.dumps({"decision": "approve", "confidence": 1.0, "feedback": "Human override accepted", "issues": []})

    with patch("src.llm.ollama_client.OllamaClient.ask", return_value=mock_reviewer_approve):
        res = graph.invoke(
            Command(resume={"action": "take_over", "decision": "approve", "custom_output": human_text}),
            config=config,
        )

    assert res["specialist_output"] == human_text
    assert any("take_over" in t for t in res.get("trace", []))
    conn.close()


# ── Audit Logging Test ────────────────────────────────────────────────────────

def test_approval_event_audit_logging(tmp_path):
    """Confirm every approval decision is logged to SQLite and JSONL audit files."""
    db_file = tmp_path / "audit_test.db"
    jsonl_dir = tmp_path / "data"
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    log_approval_event(
        task_id="task_audit_1",
        user_id="user_admin",
        trigger_type="sensitive_operation",
        approval_level="Approve Action",
        decision="approved",
        user_response="Action approved by admin",
        db_path=db_file,
        data_dir=jsonl_dir,
    )

    # Verify SQLite audit log
    conn = sqlite3.connect(str(db_file))
    cursor = conn.execute("SELECT * FROM approval_audit_log WHERE task_id='task_audit_1';")
    row = cursor.fetchone()
    assert row is not None
    assert row[3] == "sensitive_operation"
    assert row[5] == "approved"
    conn.close()

    # Verify JSONL log file
    jsonl_file = jsonl_dir / "approval_events.jsonl"
    assert jsonl_file.exists()
    lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["task_id"] == "task_audit_1"
    assert entry["decision"] == "approved"
