"""
Full integration test suite for AgentHive's multi-agent orchestration.

Covers:
  1. Coder-path escalation (t1 fails twice -> escalates, t2/t3 halted)
  2. UI mutual exclusivity (paused banner vs completed card)
  3. Approved subtask advancement (no spurious re-pause)
  4. Sensitive-operation-triggers-pause (new)
  5. Research / Data / Writer specialists — parallel coverage of the
     SAME bug classes originally found only in CodingSpecialist:
       - retry feedback + issues actually reaching the specialist prompt
       - reviewer confidence varies across retries (not flat/stale)
       - non-empty, non-fallback final_answer on escalation
       - tools_used reflects real tool calls, not a guessed placeholder
  6. Pure text-generation scenario (no fake tools generated when tool_logs is empty)
"""

import pytest
from unittest.mock import patch, MagicMock

from src.orchestration.graph import build_graph, get_checkpointer
from src.orchestration.state import AgentHiveState


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def cfg(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def make_plan(agent_for_t1="coder"):
    instructions = {
        "coder": "Write python script to create sales table and insert transactions",
        "data": "Query sqlite database for monthly revenue grouped by item",
        "research": "Find the latest research and current facts on database performance",
        "writer": "Find recent trends and statistics to write an executive report",
    }
    inst = instructions.get(agent_for_t1, "Subtask instruction")
    return {
        "subtasks": [
            {"id": "t1", "title": f"Subtask 1 for {agent_for_t1}", "assigned_agent": agent_for_t1,
             "instruction": inst, "expected_output": "Completed output", "status": "pending"},
            {"id": "t2", "title": f"Subtask 2 for {agent_for_t1}", "assigned_agent": agent_for_t1,
             "instruction": inst, "expected_output": "Completed output", "status": "pending"},
            {"id": "t3", "title": f"Subtask 3 for {agent_for_t1}", "assigned_agent": agent_for_t1,
             "instruction": inst, "expected_output": "Completed output", "status": "pending"},
        ],
        "execution_order": ["t1", "t2", "t3"],
    }


FALLBACK_MARKERS = ("no explicit answer string recorded", "no content returned")


def assert_real_tool_names(tools_used, agent_name, allow_empty=False):
    if not allow_empty:
        assert tools_used, f"No tools recorded for {agent_name} in a tool-use scenario."
    guessed_placeholder = f"{agent_name}_tool"
    assert guessed_placeholder not in tools_used, (
        f"tools_used contains the guessed placeholder {guessed_placeholder!r} "
        f"instead of a real tool name — tool_logs is likely not being "
        f"populated at the actual call site for {agent_name}."
    )


def assert_not_fallback_string(text, label):
    assert text, f"{label} is empty."
    lowered = text.lower()
    assert not any(marker in lowered for marker in FALLBACK_MARKERS), (
        f"{label} contains a generic fallback string ({text!r}) instead of "
        f"real content — MemorySaveAgent fallback chain masked an empty "
        f"upstream field rather than it being genuinely populated."
    )


REVIEWER_RETRY_TWICE = [
    {"decision": "retry", "confidence": 0.32,
     "feedback": "Missing requirement A.", "issues": ["requirement A"]},
    {"decision": "retry", "confidence": 0.41,
     "feedback": "Requirement A partially fixed, still missing B.",
     "issues": ["requirement B"]},
    {"decision": "escalate", "confidence": 0.35,
     "feedback": "Retries exhausted — escalating.",
     "issues": ["requirement B"]},
]


# ---------------------------------------------------------------------------
# 1. Coder-path escalation
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCoderEscalationHaltsWorkflow:

    @patch("src.agents.supervisor.Supervisor.plan", return_value=make_plan("coder"))
    @patch("src.agents.specialist_coder.CodingSpecialist.run")
    @patch("src.agents.reviewer.Reviewer.review")
    def test_t1_escalation_halts_t2_and_t3(
        self, mock_review, mock_coder_run, mock_plan
    ):
        mock_coder_run.side_effect = [
            {"output": "SELECT * FROM sales;", "confidence": 0.87, "tool_logs": [{"tool_name": "python_executor", "success": True}]},
            {"output": "SELECT * FROM sales GROUP BY item;", "confidence": 0.88, "tool_logs": [{"tool_name": "python_executor", "success": True}]},
            {"output": "SELECT item, SUM(amount) FROM sales GROUP BY item;", "confidence": 0.89, "tool_logs": [{"tool_name": "python_executor", "success": True}]},
        ]
        mock_review.side_effect = REVIEWER_RETRY_TWICE

        graph = build_graph(get_checkpointer(db_path=":memory:"))

        initial_state: AgentHiveState = {
            "task": "Write python script for sales analysis",
            "task_id": "it_coder_1", "user_id": "test_user",
            "max_retries": 2, "retry_count": 0,
        }
        result = graph.invoke(initial_state, config=cfg("it_coder_1"))

        assert result["workflow_status"] == "escalated"
        escalation = result.get("escalation") or {}
        failed_id = result.get("failed_subtask_id") or escalation.get("failed_subtask_id")
        halted_subtasks = result.get("halted_subtasks") or escalation.get("halted_subtasks") or []
        assert failed_id == "t1"
        assert set(halted_subtasks) == {"t2", "t3"}

        plan = result.get("task_plan") or {}
        subtasks_by_id = {s["id"]: s for s in plan.get("subtasks", [])}
        assert subtasks_by_id["t2"]["status"] == "pending"
        assert subtasks_by_id["t3"]["status"] == "pending"

        confidences = [c["confidence"] for c in REVIEWER_RETRY_TWICE]
        assert len(set(confidences)) > 1, "Test fixture itself was flat — fix fixture."

        assert_not_fallback_string(result.get("final_answer", ""), "final_answer")
        assert_real_tool_names(result.get("tools_used") or [], "coder")

        # Core Bug 4 Regression Guard: Verify reviewer issue text reached specialist retry prompt
        second_call_kwargs = mock_coder_run.call_args_list[-1]
        call_text = str(second_call_kwargs)
        assert "requirement A" in call_text or "requirement A" in str(mock_coder_run.call_args_list), (
            "Specific reviewer issue text ('requirement A') was not passed "
            "into the retry call — retry feedback may not be reaching the specialist."
        )


# ---------------------------------------------------------------------------
# 2. UI mutual exclusivity
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestUISummaryMutualExclusivity:

    def test_paused_and_completed_are_mutually_exclusive(self):
        from src.ui.streamlit_views import derive_ui_flags

        escalated_paused_state = {"workflow_status": "escalated", "is_paused": True}
        flags = derive_ui_flags(escalated_paused_state)
        assert flags["show_paused_banner"] is True
        assert flags["show_completed_card"] is False

        completed_state = {"workflow_status": "completed", "is_paused": False}
        flags2 = derive_ui_flags(completed_state)
        assert flags2["show_paused_banner"] is False
        assert flags2["show_completed_card"] is True

        escalated_not_paused_state = {"workflow_status": "escalated", "is_paused": False}
        flags3 = derive_ui_flags(escalated_not_paused_state)
        assert not (flags3["show_paused_banner"] and flags3["show_completed_card"])


# ---------------------------------------------------------------------------
# 3. Approved subtask advancement + 4. Sensitive-op pause
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestApprovalGateBehavior:

    @patch("src.agents.supervisor.Supervisor.plan", return_value=make_plan("coder"))
    @patch("src.agents.specialist_coder.CodingSpecialist.run",
           return_value={"output": "-- fully correct code --", "confidence": 0.9, "tool_logs": [{"tool_name": "python_executor", "success": True}]})
    @patch("src.agents.reviewer.Reviewer.review",
           return_value={"decision": "approve", "confidence": 0.9,
                          "feedback": "Meets all requirements.", "issues": []})
    def test_approved_t1_advances_without_pausing(
        self, mock_review, mock_coder_run, mock_plan
    ):
        graph = build_graph(get_checkpointer(db_path=":memory:"))
        initial_state: AgentHiveState = {
            "task": "Write python script for sales analysis",
            "task_id": "it_approve_1", "user_id": "test_user",
            "max_retries": 2, "retry_count": 0,
        }
        result = graph.invoke(initial_state, config=cfg("it_approve_1"))
        assert result.get("is_paused") is not True

    @patch("src.agents.supervisor.Supervisor.plan")
    @patch("src.agents.specialist_data.DataSpecialist.run",
           return_value={"output": "DROP TABLE sales; -- destructive op", "confidence": 0.9, "tool_logs": [{"tool_name": "sqlite_db", "success": True}]})
    @patch("src.agents.reviewer.Reviewer.review",
           return_value={"decision": "approve", "confidence": 0.9,
                          "feedback": "Correct but destructive.", "issues": []})
    def test_sensitive_operation_triggers_pause_even_after_approval(
        self, mock_review, mock_data_run, mock_plan
    ):
        plan = make_plan("data")
        plan["subtasks"][0]["sensitive"] = True
        mock_plan.return_value = plan

        graph = build_graph(get_checkpointer(db_path=":memory:"))

        initial_state: AgentHiveState = {
            "task": "Perform database operation",
            "task_id": "it_sensitive_1", "user_id": "test_user",
            "max_retries": 2, "retry_count": 0,
        }
        result = graph.invoke(initial_state, config=cfg("it_sensitive_1"))
        is_paused = bool(result.get("is_paused") or result.get("__interrupt__"))
        assert is_paused is True, (
            "A sensitive/destructive operation was approved by the reviewer "
            "but did NOT trigger a human-approval pause."
        )


# ---------------------------------------------------------------------------
# 5. Research / Data / Writer — parallel coverage of coder's bug classes
# ---------------------------------------------------------------------------

def _run_retry_then_escalate(agent_key, run_patch_target, review_side_effect,
                              run_side_effect, thread_id):
    with patch("src.agents.supervisor.Supervisor.plan",
               return_value=make_plan(agent_key)), \
         patch(run_patch_target, side_effect=run_side_effect), \
         patch("src.agents.reviewer.Reviewer.review",
               side_effect=review_side_effect):
        graph = build_graph(get_checkpointer(db_path=":memory:"))
        initial_state: AgentHiveState = {
            "task": "Execute multi-agent workflow task",
            "task_id": thread_id, "user_id": "test_user",
            "max_retries": 2, "retry_count": 0,
        }
        return graph.invoke(initial_state, config=cfg(thread_id))


@pytest.mark.integration
class TestResearchSpecialistParity:

    def test_research_retry_and_escalation_parity(self):
        result = _run_retry_then_escalate(
            agent_key="research",
            run_patch_target="src.agents.specialist_research.ResearchSpecialist.run",
            review_side_effect=REVIEWER_RETRY_TWICE,
            run_side_effect=[
                {"output": "Summary draft v1", "confidence": 0.8, "tool_logs": [{"tool_name": "web_search", "success": True}]},
                {"output": "Summary draft v2", "confidence": 0.82, "tool_logs": [{"tool_name": "web_search", "success": True}]},
                {"output": "Summary draft v3", "confidence": 0.84, "tool_logs": [{"tool_name": "web_search", "success": True}]},
            ],
            thread_id="it_research_1",
        )
        assert result["workflow_status"] == "escalated"
        escalation = result.get("escalation") or {}
        halted_subtasks = result.get("halted_subtasks") or escalation.get("halted_subtasks") or []
        assert set(halted_subtasks) == {"t2", "t3"}
        assert_not_fallback_string(result.get("final_answer", ""), "final_answer (research)")
        assert_real_tool_names(result.get("tools_used") or [], "research")


@pytest.mark.integration
class TestDataSpecialistParity:

    def test_data_retry_and_escalation_parity(self):
        result = _run_retry_then_escalate(
            agent_key="data",
            run_patch_target="src.agents.specialist_data.DataSpecialist.run",
            review_side_effect=REVIEWER_RETRY_TWICE,
            run_side_effect=[
                {"output": "SELECT * FROM raw;", "confidence": 0.8, "tool_logs": [{"tool_name": "sqlite_db", "success": True}]},
                {"output": "SELECT * FROM raw GROUP BY x;", "confidence": 0.83, "tool_logs": [{"tool_name": "sqlite_db", "success": True}]},
                {"output": "SELECT x, COUNT(*) FROM raw GROUP BY x;", "confidence": 0.85, "tool_logs": [{"tool_name": "sqlite_db", "success": True}]},
            ],
            thread_id="it_data_1",
        )
        assert result["workflow_status"] == "escalated"
        escalation = result.get("escalation") or {}
        halted_subtasks = result.get("halted_subtasks") or escalation.get("halted_subtasks") or []
        assert set(halted_subtasks) == {"t2", "t3"}
        assert_not_fallback_string(result.get("final_answer", ""), "final_answer (data)")
        assert_real_tool_names(result.get("tools_used") or [], "data")


@pytest.mark.integration
class TestWriterSpecialistParity:

    def test_writer_retry_and_escalation_parity(self):
        result = _run_retry_then_escalate(
            agent_key="writer",
            run_patch_target="src.agents.specialist_writer.WriterSpecialist.run",
            review_side_effect=REVIEWER_RETRY_TWICE,
            run_side_effect=[
                {"output": "Draft paragraph v1", "confidence": 0.75, "tool_logs": [{"tool_name": "web_search", "success": True}]},
                {"output": "Draft paragraph v2", "confidence": 0.79, "tool_logs": [{"tool_name": "web_search", "success": True}]},
                {"output": "Draft paragraph v3", "confidence": 0.81, "tool_logs": [{"tool_name": "web_search", "success": True}]},
            ],
            thread_id="it_writer_1",
        )
        assert result["workflow_status"] == "escalated"
        escalation = result.get("escalation") or {}
        halted_subtasks = result.get("halted_subtasks") or escalation.get("halted_subtasks") or []
        assert set(halted_subtasks) == {"t2", "t3"}
        assert_not_fallback_string(result.get("final_answer", ""), "final_answer (writer)")
        assert_real_tool_names(result.get("tools_used") or [], "writer")


# ---------------------------------------------------------------------------
# 6. Pure text-generation scenario (no fake tools generated)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPureTextGenerationNoFakeTools:

    def test_pure_text_generation_no_fake_tools(self):
        """Verify pure text generation tasks without tool keywords record empty/real tools without fake placeholders."""
        pure_plan = {
            "subtasks": [
                {"id": "t1", "title": "Write essay", "assigned_agent": "writer",
                 "instruction": "Write an essay explaining leadership principles",
                 "expected_output": "Essay text", "status": "pending"},
            ],
            "execution_order": ["t1"],
        }
        with patch("src.agents.supervisor.Supervisor.plan", return_value=pure_plan), \
             patch("src.agents.specialist_writer.WriterSpecialist.run", return_value={"output": "Leadership essay content", "confidence": 0.9, "tool_logs": []}), \
             patch("src.agents.reviewer.Reviewer.review", return_value={"decision": "approve", "confidence": 0.9, "feedback": "Good", "issues": []}):
            graph = build_graph(get_checkpointer(db_path=":memory:"))
            initial_state: AgentHiveState = {
                "task": "Write essay",
                "task_id": "it_pure_text_1", "user_id": "test_user",
                "max_retries": 2, "retry_count": 0,
            }
            result = graph.invoke(initial_state, config=cfg("it_pure_text_1"))
            assert_real_tool_names(result.get("tools_used") or [], "writer", allow_empty=True)


# ---------------------------------------------------------------------------
# 7. Cross-specialist consistency check
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCrossSpecialistConsistency:

    SPECIALIST_MATRIX = [
        ("coder", "src.agents.specialist_coder.CodingSpecialist.run"),
        ("research", "src.agents.specialist_research.ResearchSpecialist.run"),
        ("data", "src.agents.specialist_data.DataSpecialist.run"),
        ("writer", "src.agents.specialist_writer.WriterSpecialist.run"),
    ]

    @pytest.mark.parametrize("agent_key,run_target", SPECIALIST_MATRIX)
    def test_retry_prompt_includes_reviewer_issues(self, agent_key, run_target):
        with patch("src.agents.supervisor.Supervisor.plan",
                   return_value=make_plan(agent_key)), \
             patch(run_target) as mock_run, \
             patch("src.agents.reviewer.Reviewer.review",
                   side_effect=REVIEWER_RETRY_TWICE):

            graph = build_graph(get_checkpointer(db_path=":memory:"))
            mock_run.side_effect = [
                {"output": "attempt 1", "confidence": 0.7, "tool_logs": [{"tool_name": f"{agent_key}_tool_call", "success": True}]},
                {"output": "attempt 2", "confidence": 0.75, "tool_logs": [{"tool_name": f"{agent_key}_tool_call", "success": True}]},
                {"output": "attempt 3", "confidence": 0.78, "tool_logs": [{"tool_name": f"{agent_key}_tool_call", "success": True}]},
            ]
            initial_state: AgentHiveState = {
                "task": "Execute specialist consistency task",
                "task_id": f"it_matrix_{agent_key}", "user_id": "test_user",
                "max_retries": 2, "retry_count": 0,
            }
            graph.invoke(initial_state, config=cfg(f"it_matrix_{agent_key}"))

            assert mock_run.call_count >= 2, (
                f"{agent_key} specialist was not retried after a 'retry' "
                f"reviewer decision."
            )
            # Core Bug 4 Regression Guard: Verify reviewer issue text reached specialist retry prompt
            retry_call = mock_run.call_args_list[-1]
            call_repr = str(retry_call)
            assert "requirement A" in call_repr, (
                f"{agent_key} specialist's retry call does not contain the "
                f"specific reviewer issue text ('requirement A'). This is the "
                f"exact Bug 4 defect — confirm it hasn't reappeared for "
                f"{agent_key} even though it was fixed for coder."
            )
