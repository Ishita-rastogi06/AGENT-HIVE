"""
Comprehensive test suite for Phase 4 Execution Tracing, Log Merging, Trace Diffing, and Replay System.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from unittest.mock import patch

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from src.orchestration.approval import add_pending_approval, log_approval_event
from src.orchestration.graph import build_graph
from src.orchestration.trace import (
    build_execution_trace,
    compare_traces,
    diff_state_snapshots,
    full_replay_task,
    partial_replay_task,
)


def test_diff_state_snapshots():
    """Verify dictionary diffing between state snapshots."""
    prev = {"task": "test", "status": "planning", "retry_count": 0}
    curr = {"task": "test", "status": "executing", "retry_count": 0, "specialist_output": "done"}

    diff = diff_state_snapshots(prev, curr)
    assert diff == {"status": "executing", "specialist_output": "done"}


def test_unified_trace_building_and_multi_log_merging(tmp_path):
    """
    Test building a unified trace tree for a task including a retry and an approval pause.
    Confirms all three log sources (LangGraph checkpoints, tool JSONL log, SQLite approval audit log) merge correctly.
    """
    db_file = tmp_path / "trace_merge_test.db"
    jsonl_dir = tmp_path / "data"
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    task_id = "task_trace_merge_101"
    user_id = "trace_user"
    config = {"configurable": {"thread_id": task_id}}

    # 1. Setup persistent graph run that pauses for sensitive operation
    conn = sqlite3.connect(str(db_file), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_graph(checkpointer=checkpointer)

    mock_plan = json.dumps({
        "objective": "Query database and review",
        "reasoning": "Database subtask",
        "subtasks": [
            {
                "id": "t1",
                "title": "Query DB",
                "instruction": "INSERT INTO records VALUES (1);",
                "assigned_agent": "data",
                "expected_output": "Inserted",
                "status": "pending",
            }
        ],
        "execution_order": ["t1"],
    })

    with patch("src.llm.ollama_client.OllamaClient.ask", return_value=mock_plan):
        res_pause = graph.invoke(
            {
                "task": "INSERT INTO records VALUES (1);",
                "user_id": user_id,
                "task_id": task_id,
                "db_path": str(db_file),
                "trace": [],
            },
            config=config,
        )
        assert "__interrupt__" in res_pause

    # 2. Write mock tool-call audit entry in tool_calls.jsonl
    tool_log_file = jsonl_dir / "tool_calls.jsonl"
    tool_entry = {
        "timestamp": "2026-08-30T23:00:00Z",
        "task_id": task_id,
        "tool_name": "db_query",
        "arguments": {"query": "INSERT INTO records VALUES (1);", "read_only": False},
        "success": True,
        "result_summary": "1 row inserted",
        "duration_seconds": 0.12,
    }
    tool_log_file.write_text(json.dumps(tool_entry) + "\n", encoding="utf-8")

    # 3. Resume workflow with robust mock function
    def mock_ask_resume(prompt, system=""):
        if "review" in system.lower() or "quality" in prompt.lower():
            return json.dumps({"decision": "approve", "confidence": 0.95, "feedback": "Approved", "issues": []})
        return "Query result: 1 row inserted."

    with patch("src.llm.ollama_client.OllamaClient.ask", side_effect=mock_ask_resume):
        res_done = graph.invoke(
            Command(resume={"action": "approve_action", "decision": "approve"}),
            config=config,
        )

    conn.close()

    # 4. Build unified execution trace
    trace_data = build_execution_trace(
        task_id,
        user_id=user_id,
        db_path=db_file,
        data_dir=jsonl_dir,
        tool_log_file=tool_log_file,
    )

    assert trace_data["task_id"] == task_id
    assert trace_data["total_steps"] >= 3

    nodes = trace_data["nodes"]
    step_names = [n["step_name"] for n in nodes]
    assert "supervisor" in step_names
    assert "approval_check" in step_names

    # Check child approval event attachment
    app_nodes = [n for n in nodes if any(c["node_type"] == "approval_event" for c in n.get("children", []))]
    assert len(app_nodes) >= 1

    # Check child tool call attachment
    tool_nodes = [n for n in nodes if any(c["node_type"] == "tool_call" for c in n.get("children", []))]
    assert len(tool_nodes) >= 1


def test_partial_replay_and_trace_diffing(tmp_path):
    """
    Test a partial replay that diverges from the original at a chosen step.
    Confirms the trace diff viewer correctly identifies the exact divergence point.
    """
    db_file = tmp_path / "replay_diff_test.db"
    conn = sqlite3.connect(str(db_file), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_graph(checkpointer=checkpointer)

    orig_task_id = "task_orig_555"
    user_id = "replay_user"
    config = {"configurable": {"thread_id": orig_task_id}}

    mock_plan = json.dumps({
        "objective": "Compare frameworks",
        "reasoning": "Research comparison",
        "subtasks": [
            {
                "id": "t1",
                "title": "Compare REST vs GraphQL",
                "instruction": "Compare REST vs GraphQL",
                "assigned_agent": "research",
                "expected_output": "Comparison table",
                "status": "pending",
            }
        ],
        "execution_order": ["t1"],
    })

    def mock_ask_orig(prompt, system=""):
        if "decompose" in prompt.lower() or "subtask" in prompt.lower():
            return mock_plan
        if "review" in system.lower() or "quality" in prompt.lower():
            return json.dumps({"decision": "approve", "confidence": 0.95, "feedback": "Good job", "issues": []})
        return "REST vs GraphQL comparison text."

    with patch("src.llm.ollama_client.OllamaClient.ask", side_effect=mock_ask_orig):
        res_orig = graph.invoke(
            {
                "task": "Compare REST vs GraphQL",
                "user_id": user_id,
                "task_id": orig_task_id,
                "db_path": str(db_file),
                "trace": [],
            },
            config=config,
        )

    orig_trace = build_execution_trace(orig_task_id, user_id=user_id, db_path=db_file)
    target_node = next(n for n in orig_trace["nodes"] if n["step_name"] in ("research", "approval_check"))

    # Perform partial replay with edited specialist output
    new_output = "MODIFIED REPLAY OUTPUT: REST uses endpoints, GraphQL uses query schemas."

    def mock_ask_replay(prompt, system=""):
        if "review" in system.lower() or "quality" in prompt.lower():
            return json.dumps({"decision": "approve", "confidence": 0.95, "feedback": "Good job", "issues": []})
        return new_output

    with patch("src.llm.ollama_client.OllamaClient.ask", side_effect=mock_ask_replay):
        replay_task_id, res_replay = partial_replay_task(
            original_task_id=orig_task_id,
            checkpoint_id=target_node["node_id"],
            override_updates={"specialist_output": new_output},
            user_id=user_id,
            db_path=db_file,
        )

    replay_trace = build_execution_trace(replay_task_id, user_id=user_id, db_path=db_file)
    diff_result = compare_traces(orig_trace, replay_trace)

    assert diff_result["diverged"] is True
    assert diff_result["divergence_point"] is not None
    conn.close()


def test_reviewer_node_details_populated(tmp_path):
    """Verify that expanding a reviewer node in build_execution_trace populates decision, confidence, and feedback."""
    db_file = tmp_path / "reviewer_details_test.db"
    conn = sqlite3.connect(str(db_file), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_graph(checkpointer=checkpointer)

    task_id = "task_reviewer_details_777"
    user_id = "reviewer_user"
    config = {"configurable": {"thread_id": task_id}}

    mock_plan = json.dumps({
        "objective": "Research REST API principles",
        "reasoning": "Single conceptual subtask",
        "subtasks": [
            {
                "id": "t1",
                "title": "Research REST principles",
                "instruction": "Explain REST principles",
                "assigned_agent": "research",
                "expected_output": "Principles explained",
                "status": "pending",
            }
        ],
        "execution_order": ["t1"],
    })

    def mock_ask(prompt, system="", *args, **kwargs):
        if "subtask title:" in prompt.lower() or "quality" in system.lower():
            return json.dumps({
                "decision": "approve",
                "confidence": 0.94,
                "feedback": "Clear explanation of REST principles",
                "issues": []
            })
        if "decompose" in prompt.lower() or "subtask" in prompt.lower():
            return mock_plan
        return "REST principles include statelessness and resource-based URIs."

    with patch("src.llm.ollama_client.OllamaClient.ask", side_effect=mock_ask):
        graph.invoke(
            {
                "task": "Explain REST principles",
                "user_id": user_id,
                "task_id": task_id,
                "db_path": str(db_file),
                "trace": [],
            },
            config=config,
        )

    trace = build_execution_trace(task_id, user_id=user_id, db_path=db_file)
    rev_nodes = [n for n in trace["nodes"] if n["step_name"] == "reviewer"]
    assert len(rev_nodes) >= 1

    rev_details = rev_nodes[0].get("details", {}).get("review", {})
    assert rev_details.get("decision") == "approve"
    assert rev_details.get("confidence") == 0.94
    assert rev_details.get("feedback") == "Clear explanation of REST principles"
    assert rev_details.get("issues") == []
    conn.close()


def test_tool_calls_strictly_scoped_by_task_id(tmp_path):
    """Confirm that tool call log entries are strictly matched by task_id and never cross-contaminate other tasks."""
    db_file = tmp_path / "strict_scope_test.db"
    jsonl_file = tmp_path / "tool_calls.jsonl"

    conn = sqlite3.connect(str(db_file), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_graph(checkpointer=checkpointer)

    task_a = "task_alpha_111"
    task_b = "task_beta_222"
    user_id = "scope_user"

    mock_plan = json.dumps({
        "objective": "Task execution",
        "reasoning": "Single subtask",
        "subtasks": [
            {
                "id": "t1",
                "title": "Subtask 1",
                "instruction": "Do work",
                "assigned_agent": "research",
                "expected_output": "Done",
                "status": "pending",
            }
        ],
        "execution_order": ["t1"],
    })

    def mock_ask(prompt, system="", *args, **kwargs):
        if "subtask title:" in prompt.lower() or "quality" in system.lower():
            return json.dumps({"decision": "approve", "confidence": 0.95, "feedback": "Good", "issues": []})
        if "decompose" in prompt.lower() or "subtask" in prompt.lower():
            return mock_plan
        return "Work output."

    # Run Task A
    with patch("src.llm.ollama_client.OllamaClient.ask", side_effect=mock_ask):
        graph.invoke(
            {"task": "Task A prompt", "user_id": user_id, "task_id": task_a, "db_path": str(db_file), "trace": []},
            config={"configurable": {"thread_id": task_a}},
        )

    # Run Task B
    with patch("src.llm.ollama_client.OllamaClient.ask", side_effect=mock_ask):
        graph.invoke(
            {"task": "Task B prompt", "user_id": user_id, "task_id": task_b, "db_path": str(db_file), "trace": []},
            config={"configurable": {"thread_id": task_b}},
        )

    conn.close()

    # Write distinct tool call entries for Task A, Task B, and an untagged/None task entry
    tool_entry_a = {
        "timestamp": "2026-09-15T12:00:00Z",
        "task_id": task_a,
        "agent_name": "research",
        "tool_name": "web_search",
        "arguments": {"query": "Task A search query"},
        "success": True,
        "result_summary": "Task A search output",
        "duration_seconds": 0.5,
    }
    tool_entry_b = {
        "timestamp": "2026-09-15T12:01:00Z",
        "task_id": task_b,
        "agent_name": "research",
        "tool_name": "web_search",
        "arguments": {"query": "Task B battery query"},
        "success": True,
        "result_summary": "Task B battery output",
        "duration_seconds": 0.8,
    }
    tool_entry_untagged = {
        "timestamp": "2026-09-15T12:02:00Z",
        "task_id": None,
        "agent_name": "research",
        "tool_name": "web_search",
        "arguments": {"query": "Untagged query"},
        "success": True,
        "result_summary": "Untagged output",
        "duration_seconds": 0.2,
    }

    with jsonl_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(tool_entry_a) + "\n")
        f.write(json.dumps(tool_entry_b) + "\n")
        f.write(json.dumps(tool_entry_untagged) + "\n")

    # Trace for Task A MUST only contain tool_entry_a
    trace_a = build_execution_trace(task_a, user_id=user_id, db_path=db_file, tool_log_file=jsonl_file)
    res_node_a = next(n for n in trace_a["nodes"] if n["step_name"] == "research")
    child_tools_a = [c for c in res_node_a.get("children", []) if c.get("node_type") == "tool_call"]
    assert len(child_tools_a) == 1
    assert child_tools_a[0]["details"]["arguments"] == {"query": "Task A search query"}

    # Trace for Task B MUST only contain tool_entry_b
    trace_b = build_execution_trace(task_b, user_id=user_id, db_path=db_file, tool_log_file=jsonl_file)
    res_node_b = next(n for n in trace_b["nodes"] if n["step_name"] == "research")
    child_tools_b = [c for c in res_node_b.get("children", []) if c.get("node_type") == "tool_call"]
    assert len(child_tools_b) == 1
    assert child_tools_b[0]["details"]["arguments"] == {"query": "Task B battery query"}
