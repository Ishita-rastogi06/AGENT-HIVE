"""
Tests for Reviewer grounding refusal escalation and Approval Queue escalation persistence.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.agents.reviewer import Reviewer
from src.orchestration.approval import get_pending_approvals, clear_pending_approvals
from src.orchestration.graph import _combined_finalise


def test_reviewer_escalates_on_grounding_refusal():
    """Verify that Reviewer immediately outputs decision='escalate' when specialist output contains a grounding refusal."""
    mock_llm = MagicMock()
    mock_llm.ask.return_value = '{"decision": "retry", "confidence": 0.4, "feedback": "Fails to list top 3 breakthroughs", "issues": ["missing 3 breakthroughs"]}'

    reviewer = Reviewer(mock_llm)

    test_outputs = [
        "I must state that the web search data returned was insufficient or limited for this query.",
        "search data was insufficient to identify specific top 3 breakthroughs without hallucinating.",
        "Web search returned no relevant snippets for this query.",
        "Search results were empty, so data is limited.",
        "Unable to verify top 3 breakthroughs from search findings.",
    ]

    subtask = {
        "title": "Battery Technology Breakthroughs",
        "instruction": "Find the top 3 breakthroughs in electric car battery technology",
        "expected_output": "List of top 3 battery breakthroughs",
        "tool_logs": [{"tool_name": "web_search", "query": "battery breakthroughs", "success": True}],
    }

    for spec_out in test_outputs:
        result = reviewer.review(
            specialist_output=spec_out,
            instruction=subtask["instruction"],
            expected_output=subtask["expected_output"],
            state={"current_subtask": subtask, "low_confidence_threshold": 0.6},
        )
        assert result["decision"] == "escalate"
        assert result["approved"] is False
        assert "grounding mandate refusal due to insufficient search data" in result["issues"]


def test_db_query_task_does_not_trigger_grounding_refusal_escalation():
    """Verify that a db_query-based task does NOT get the grounding mandate refusal escalation label even if phrase contains 'insufficient data'."""
    mock_llm = MagicMock()
    mock_llm.ask.return_value = '{"decision": "retry", "confidence": 0.35, "feedback": "Query returned insufficient rows and missing GROUP BY status.", "issues": ["missing GROUP BY status in SQL query"]}'

    reviewer = Reviewer(mock_llm)

    subtask = {
        "title": "Query customers balance",
        "instruction": "Query the customers table and show the total account balance grouped by status.",
        "expected_output": "Grouped total account balance table",
        "tool_logs": [{"tool_name": "db_query", "query": "SELECT * FROM customers", "success": True}],
    }

    spec_output = "Execution returned insufficient data: query results lack status grouping."

    result = reviewer.review(
        specialist_output=spec_output,
        instruction=subtask["instruction"],
        expected_output=subtask["expected_output"],
        state={"current_subtask": subtask, "low_confidence_threshold": 0.6, "selected_agent": "data"},
    )

    # Must NOT get the grounding mandate refusal search issue
    assert "grounding mandate refusal due to insufficient search data" not in result.get("issues", [])
    assert result["decision"] == "retry"
    assert "missing GROUP BY status in SQL query" in result["issues"]


def test_retry_exhausted_escalation_appears_in_approval_queue(tmp_path):
    """Verify that any workflow escalation via finalise node reliably persists a pending entry in the approval queue."""
    db_file = tmp_path / "test_approval_queue.db"
    user_id = "test_user_esc"
    task_id = "test_task_esc_123"

    clear_pending_approvals(user_id=user_id, db_path=db_file)

    initial_state = {
        "task_id": task_id,
        "user_id": user_id,
        "task": "Research battery tech breakthroughs",
        "current_subtask": {
            "id": "t1",
            "title": "Research breakthroughs",
            "instruction": "Find top 3 battery breakthroughs",
            "status": "pending",
        },
        "task_plan": {
            "subtasks": [
                {
                    "id": "t1",
                    "title": "Research breakthroughs",
                    "instruction": "Find top 3 battery breakthroughs",
                    "status": "pending",
                }
            ],
            "execution_order": ["t1"],
        },
        "selected_agent": "research",
        "specialist_output": "I must state that the web search data returned was insufficient or limited for this query.",
        "review": {
            "decision": "escalate",
            "approved": False,
            "confidence": 0.0,
            "feedback": "Specialist reported search data was insufficient.",
            "issues": ["grounding mandate refusal due to insufficient search data"],
        },
        "db_path": db_file,
    }

    finalise_node = _combined_finalise(None)
    result = finalise_node(initial_state)

    assert result.get("workflow_status") == "escalated"

    pending = get_pending_approvals(user_id=user_id, db_path=db_file)
    assert len(pending) == 1
    assert pending[0]["task_id"] == task_id
    assert "escalated" in pending[0]["trigger_reason"].lower()
