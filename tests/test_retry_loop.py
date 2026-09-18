"""
Tests for the retry loop and escalation graph edges (Item 4 requirement).

Verifies:
  - Reviewer rejects twice then approves → specialist called 3 times total,
    reviewer called 3 times, final state is completed with final_answer set
  - Reviewer always rejects → retry budget exhausted → escalated, needs_human_review=True
  - Reviewer immediately escalates → needs_human_review=True on first pass
  - workflow_status transitions: planning → executing/reviewing → retrying → completed
  - retry_count increments correctly on each loop
  - Reviewer feedback is passed to state for the specialist to read on retry
"""
import sys
import unittest

sys.path.insert(0, __file__.split("tests")[0].rstrip("\\/"))

from src.orchestration.graph import (
    _route_after_review,
    _pre_retry_node,
    _combined_finalise,
    _ROUTE_FINALISE,
    _ROUTE_RETRY_RESEARCH,
    _ROUTE_RETRY_CODER,
)
from src.orchestration.state import AgentHiveState


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_state(**overrides) -> AgentHiveState:
    base: AgentHiveState = {
        "task": "write fibonacci",
        "selected_agent": "research",
        "retry_count": 0,
        "max_retries": 2,
        "low_confidence_threshold": 0.60,
        "workflow_status": "reviewing",
        "trace": [],
        "review": {"decision": "approve", "approved": True, "confidence": 0.85,
                   "feedback": "Good.", "issues": []},
        "specialist_output": "```python\ndef fib(n): pass\n```\nExplanation.",
        "needs_human_review": False,
    }
    base.update(overrides)
    return base


def _make_reviewer(decisions):
    """
    Return a callable that mimics reviewer.run() using a list of decisions.
    Each call pops the next (decision, confidence, feedback) tuple.
    decisions = [("retry", 0.30, "Too short"), ("approve", 0.90, "Great")]
    """
    calls = list(decisions)
    call_log = []

    def _run(state: AgentHiveState) -> dict:
        decision, confidence, feedback = calls.pop(0)
        call_log.append(decision)
        approved = decision == "approve"
        review = {
            "decision": decision,
            "approved": approved,
            "confidence": confidence,
            "feedback": feedback,
            "issues": [] if approved else ["needs improvement"],
        }
        return {
            "review": review,
            "final_answer": state.get("specialist_output", "") if approved else "",
            "trace": [*state.get("trace", []), f"reviewer: {decision}"],
        }

    _run.call_log = call_log
    return _run


def _make_specialist(output_fn=None):
    """Return a specialist that logs calls and returns configurable output."""
    call_log = []

    def _run(state: AgentHiveState) -> dict:
        retry_count = state.get("retry_count", 0)
        call_log.append(retry_count)
        output = output_fn(state) if output_fn else f"output at retry={retry_count}"
        return {
            "specialist_output": output,
            "specialist_confidence": 0.9 if retry_count > 0 else 0.3,
            "workflow_status": "reviewing",
            "trace": [*state.get("trace", []), f"specialist retry={retry_count}"],
        }

    _run.call_log = call_log
    return _run


def _run_full_loop(
    initial_state: AgentHiveState,
    specialist_fn,
    reviewer_fn,
    max_iterations: int = 10,
) -> AgentHiveState:
    """
    Drive the graph nodes manually to simulate LangGraph execution.
    Returns the final state after loop completion.
    """
    state = dict(initial_state)
    pre_retry = _pre_retry_node(state.get("selected_agent", "research"))
    finalise = _combined_finalise(None)

    for _ in range(max_iterations):
        state.update(specialist_fn(state))
        state.update(reviewer_fn(state))
        route = _route_after_review(state)

        if route == _ROUTE_FINALISE:
            state.update(finalise(state))
            return state

        # pre_retry then loop back
        state.update(pre_retry(state))

    raise RuntimeError("Loop did not terminate within max_iterations")


# ── tests ─────────────────────────────────────────────────────────────────────

class TestRouteAfterReview(unittest.TestCase):
    """Unit-test the routing function in isolation."""

    def test_approve_routes_to_finalise(self):
        s = _make_state(review={"decision": "approve"}, retry_count=0)
        self.assertEqual(_route_after_review(s), _ROUTE_FINALISE)

    def test_escalate_routes_to_finalise(self):
        s = _make_state(review={"decision": "escalate"}, retry_count=0)
        self.assertEqual(_route_after_review(s), _ROUTE_FINALISE)

    def test_retry_research_under_budget(self):
        s = _make_state(review={"decision": "retry"}, retry_count=0,
                        selected_agent="research")
        self.assertEqual(_route_after_review(s), _ROUTE_RETRY_RESEARCH)

    def test_retry_coder_under_budget(self):
        s = _make_state(review={"decision": "retry"}, retry_count=1,
                        selected_agent="coder")
        self.assertEqual(_route_after_review(s), _ROUTE_RETRY_CODER)

    def test_retry_exhausted_routes_to_finalise(self):
        s = _make_state(review={"decision": "retry"}, retry_count=2,
                        max_retries=2, selected_agent="research")
        self.assertEqual(_route_after_review(s), _ROUTE_FINALISE)

    def test_retry_at_exactly_max_routes_to_finalise(self):
        for max_r in (1, 2, 3):
            s = _make_state(review={"decision": "retry"}, retry_count=max_r,
                            max_retries=max_r)
            self.assertEqual(
                _route_after_review(s), _ROUTE_FINALISE,
                f"Expected finalise at retry_count={max_r} max_retries={max_r}"
            )


class TestPreRetryNode(unittest.TestCase):
    """pre_retry_node increments counter and stamps workflow_status."""

    def setUp(self):
        self.fn = _pre_retry_node("research")

    def test_increments_retry_count(self):
        s = _make_state(retry_count=0)
        result = self.fn(s)
        self.assertEqual(result["retry_count"], 1)

    def test_stamps_retrying(self):
        s = _make_state(retry_count=0)
        result = self.fn(s)
        self.assertEqual(result["workflow_status"], "retrying")

    def test_trace_contains_retry_info(self):
        s = _make_state(retry_count=0, review={"feedback": "Too short."})
        result = self.fn(s)
        self.assertTrue(any("Retry 1" in t for t in result["trace"]))
        self.assertTrue(any("Too short" in t for t in result["trace"]))

    def test_escalates_when_over_budget(self):
        s = _make_state(retry_count=2, max_retries=2)
        result = self.fn(s)
        self.assertTrue(result.get("needs_human_review"))
        self.assertEqual(result["workflow_status"], "escalated")


class TestFinaliseNode(unittest.TestCase):
    """_combined_finalise stamps the right workflow_status."""

    def setUp(self):
        self.fn = _combined_finalise(None)

    def test_approve_stamps_completed(self):
        s = _make_state(review={"decision": "approve"})
        result = self.fn(s)
        self.assertEqual(result["workflow_status"], "completed")

    def test_escalate_stamps_escalated(self):
        s = _make_state(
            review={"decision": "escalate", "feedback": "Dangerous.", "issues": []},
            retry_count=0,
        )
        result = self.fn(s)
        self.assertEqual(result["workflow_status"], "escalated")
        self.assertTrue(result.get("needs_human_review"))
        self.assertTrue(result["escalation"]["required"])

    def test_retry_exhausted_escalates(self):
        s = _make_state(
            review={"decision": "retry", "feedback": "Still bad.", "issues": []},
            retry_count=2, max_retries=2,
        )
        result = self.fn(s)
        self.assertEqual(result["workflow_status"], "escalated")
        self.assertTrue(result.get("needs_human_review"))

    def test_escalation_reason_from_feedback(self):
        s = _make_state(
            review={"decision": "escalate", "feedback": "Dangerous output.", "issues": []},
            retry_count=0,
        )
        result = self.fn(s)
        self.assertIn("Dangerous output", result["escalation"]["reason"])


class TestRetryTwiceThenApprove(unittest.TestCase):
    """
    Core retry-loop test: reviewer rejects twice then approves.

    Expected: specialist called 3 times, reviewer called 3 times,
    final workflow_status=completed, final_answer non-empty.
    """

    def test_rejects_twice_then_approves(self):
        reviewer = _make_reviewer([
            ("retry",   0.35, "Too short, add examples."),
            ("retry",   0.45, "Still missing usage example."),
            ("approve", 0.88, "Good answer with examples."),
        ])
        specialist = _make_specialist()

        initial = _make_state(retry_count=0, max_retries=2, selected_agent="research")
        final = _run_full_loop(initial, specialist, reviewer)

        self.assertEqual(final["workflow_status"], "completed")
        self.assertTrue(final.get("final_answer"), "final_answer should be non-empty")
        self.assertEqual(specialist.call_log, [0, 1, 2],
                         "Specialist should be called at retry counts 0, 1, 2")
        self.assertEqual(reviewer.call_log, ["retry", "retry", "approve"])

    def test_retry_count_increments_correctly(self):
        counts_seen = []
        reviewer = _make_reviewer([
            ("retry",   0.30, "needs work"),
            ("approve", 0.85, "great"),
        ])

        def _tracking_specialist(state):
            counts_seen.append(state.get("retry_count", 0))
            return {
                "specialist_output": "output",
                "specialist_confidence": 0.9,
                "workflow_status": "reviewing",
                "trace": state.get("trace", []),
            }

        initial = _make_state(retry_count=0, max_retries=2)
        _run_full_loop(initial, _tracking_specialist, reviewer)
        self.assertEqual(counts_seen, [0, 1])

    def test_feedback_in_state_on_retry(self):
        """Specialist must be able to read reviewer feedback in state on retry."""
        feedback_seen = []

        def _feedback_specialist(state):
            review = state.get("review") or {}
            if state.get("retry_count", 0) > 0:
                feedback_seen.append(review.get("feedback", ""))
            return {
                "specialist_output": "improved output with examples " + "x" * 200,
                "specialist_confidence": 0.9,
                "workflow_status": "reviewing",
                "trace": state.get("trace", []),
            }

        reviewer = _make_reviewer([
            ("retry",   0.30, "Add a usage example."),
            ("approve", 0.88, "Good."),
        ])
        initial = _make_state(retry_count=0, max_retries=2)
        _run_full_loop(initial, _feedback_specialist, reviewer)
        self.assertTrue(feedback_seen, "Specialist never received feedback on retry")
        self.assertIn("Add a usage example.", feedback_seen[0])


class TestRetryBudgetExhausted(unittest.TestCase):
    """Reviewer always rejects → escalated after max_retries."""

    def test_always_retry_escalates(self):
        reviewer = _make_reviewer([
            ("retry", 0.20, "Bad output."),
            ("retry", 0.20, "Still bad."),
            ("retry", 0.20, "Still bad after retry 2."),  # shouldn't reach here
        ])
        specialist = _make_specialist()
        initial = _make_state(retry_count=0, max_retries=2, selected_agent="research")
        final = _run_full_loop(initial, specialist, reviewer)

        self.assertEqual(final["workflow_status"], "escalated")
        self.assertTrue(final.get("needs_human_review"))

    def test_escalated_has_escalation_details(self):
        reviewer = _make_reviewer([
            ("retry", 0.20, "Bad output."),
            ("retry", 0.20, "Still bad."),
            ("retry", 0.20, "placeholder"),
        ])
        specialist = _make_specialist()
        initial = _make_state(retry_count=0, max_retries=2)
        final = _run_full_loop(initial, specialist, reviewer)

        self.assertIn("escalation", final)
        self.assertTrue(final["escalation"]["required"])

    def test_max_retries_one_calls_specialist_twice(self):
        """max_retries=1 means one retry pass, so specialist is called twice."""
        specialist = _make_specialist()
        reviewer = _make_reviewer([
            ("retry",   0.30, "Bad."),
            ("approve", 0.85, "Good."),
        ])
        initial = _make_state(retry_count=0, max_retries=1, selected_agent="research")
        final = _run_full_loop(initial, specialist, reviewer)
        self.assertEqual(final["workflow_status"], "completed")
        self.assertEqual(len(specialist.call_log), 2)


class TestImmediateEscalation(unittest.TestCase):
    """Reviewer escalates on first call → needs_human_review immediately."""

    def test_immediate_escalate(self):
        reviewer = _make_reviewer([
            ("escalate", 0.05, "Dangerous content detected."),
        ])
        specialist = _make_specialist()
        initial = _make_state(retry_count=0, max_retries=2)
        final = _run_full_loop(initial, specialist, reviewer)

        self.assertEqual(final["workflow_status"], "escalated")
        self.assertTrue(final.get("needs_human_review"))
        # Specialist should only be called once.
        self.assertEqual(len(specialist.call_log), 1)


class TestWorkflowStatusTransitions(unittest.TestCase):
    """workflow_status is correctly stamped at each stage."""

    def test_retrying_status_set_during_loop(self):
        statuses_seen = []
        reviewer = _make_reviewer([
            ("retry",   0.30, "Needs work."),
            ("approve", 0.85, "Good."),
        ])

        def _status_tracking_specialist(state):
            statuses_seen.append(state.get("workflow_status"))
            return {
                "specialist_output": "output",
                "specialist_confidence": 0.9,
                "workflow_status": "reviewing",
                "trace": state.get("trace", []),
            }

        initial = _make_state(retry_count=0, max_retries=2, workflow_status="planning")
        _run_full_loop(initial, _status_tracking_specialist, reviewer)

        # First specialist call sees planning (initial), second sees retrying.
        self.assertIn("retrying", statuses_seen,
                      f"Expected 'retrying' in status transitions: {statuses_seen}")

    def test_completed_status_final(self):
        reviewer = _make_reviewer([("approve", 0.90, "Great.")])
        specialist = _make_specialist()
        initial = _make_state()
        final = _run_full_loop(initial, specialist, reviewer)
        self.assertEqual(final["workflow_status"], "completed")


class TestReviewerCorrectnessAudit(unittest.TestCase):
    """Verify Reviewer flags missing requirement keywords (e.g. monthly) for retry."""

    def test_reviewer_flags_missing_monthly_requirement(self):
        from src.agents.reviewer import Reviewer
        from src.llm.ollama_client import OllamaClient

        reviewer = Reviewer(llm=OllamaClient())
        subtask = {
            "title": "Query monthly revenue",
            "instruction": "Write a Python script using sqlite3 and pandas that creates a sales table, inserts 5 sample transactions, queries the total monthly revenue grouped by item, and prints a formatted summary table.",
            "expected_output": "Python script returning monthly revenue grouped by item",
        }
        # Code that only groups by item without monthly date truncation
        flawed_code_output = (
            "```python\n"
            "import sqlite3, pandas as pd\n"
            "conn = sqlite3.connect('sales.db')\n"
            "df = pd.read_sql_query('SELECT item, SUM(quantity * price) AS total_revenue FROM sales GROUP BY item', conn)\n"
            "print(df)\n"
            "```"
        )
        res = reviewer._llm_review(flawed_code_output, subtask, threshold=0.60)
        if res is not None:
            self.assertEqual(res["decision"], "retry", f"Reviewer should flag missing monthly aggregation, got {res}")
            self.assertLess(res["confidence"], 0.60)

    def test_retry_prompt_includes_specific_reviewer_issues_and_fresh_scoring(self):
        from src.agents.specialist_coder import CodingSpecialist
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        mock_llm.ask.return_value = "```python\nprint('hello')\n```"
        coder = CodingSpecialist(llm=mock_llm)
        subtask = {
            "id": "t1",
            "title": "Query monthly revenue",
            "instruction": "Write a script querying monthly revenue",
            "expected_output": "Monthly summary",
        }
        state = {
            "current_subtask": subtask,
            "retry_count": 1,
            "review": {
                "decision": "retry",
                "feedback": "Missing strftime date truncation in SQL query.",
                "issues": ["strftime date truncation", "monthly grouping"],
            },
            "specialist_output": "old output",
            "trace": [],
        }
        coder.run(state)
        call_args = mock_llm.ask.call_args[0]
        prompt = call_args[0]
        self.assertIn("Missing strftime date truncation", prompt)
        self.assertIn("strftime date truncation, monthly grouping", prompt)
        self.assertIn("[RETRY PASS #1 REVIEWER FEEDBACK", prompt)

    def test_retry_prompt_includes_full_original_task_requirements(self):
        """
        Verify that on retry passes, the specialist prompt ALWAYS includes FULL ORIGINAL TASK REQUIREMENTS
        from state['task'] so multi-dimension requirements (e.g. monthly + item grouping) are never lost.
        """
        from src.agents.specialist_coder import CodingSpecialist
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        mock_llm.ask.return_value = "```python\nprint('code')\n```"
        coder = CodingSpecialist(llm=mock_llm)

        state = {
            "task": "Write a Python script for total monthly revenue grouped by item",
            "current_subtask": {
                "id": "t1",
                "title": "Subtask title",
                "instruction": "Fix grouping issue",
                "expected_output": "Code snippet",
            },
            "retry_count": 1,
            "review": {
                "decision": "retry",
                "feedback": "Add strftime('%Y-%m', date)",
                "issues": ["strftime date truncation"],
            },
            "specialist_output": "old code",
            "trace": ["Reviewer: RETRY — missing strftime"],
        }
        coder.run(state)
        call_args = mock_llm.ask.call_args[0]
        prompt = call_args[0]
        self.assertIn("FULL ORIGINAL TASK REQUIREMENTS: Write a Python script for total monthly revenue grouped by item", prompt)
        self.assertIn("Cumulative Review Audit History:", prompt)
        self.assertIn("CRITICAL MANDATE: You MUST fix all specific issues identified above in this new code attempt while strictly satisfying ALL FULL ORIGINAL TASK REQUIREMENTS.", prompt)

    def test_reviewer_high_confidence_with_substantive_issues_remains_retry(self):
        """
        Verify that if the Reviewer JSON contains a high confidence score (e.g. 0.85)
        but also contains non-empty substantive issues in `issues`, the decision remains 'retry'
        and is NOT force-approved.
        """
        from src.agents.reviewer import Reviewer
        from src.llm.ollama_client import OllamaClient

        reviewer = Reviewer(llm=OllamaClient())
        raw_llm_json = '{"decision": "approve", "confidence": 0.85, "feedback": "Code is mostly fine but lacks monthly grouping.", "issues": ["missing monthly date truncation"]}'
        subtask = {"instruction": "Calculate monthly revenue grouped by item", "expected_output": "Monthly summary"}

        res = reviewer._parse_review_response(raw_llm_json, threshold=0.60, specialist_output="```python\n# test code\n```", subtask=subtask)
        self.assertIsNotNone(res)
        self.assertEqual(res["decision"], "retry", f"Decision must remain 'retry' when issues are non-empty despite confidence 0.85, got {res['decision']}")
        self.assertFalse(res["approved"])

    def test_monthly_grouping_task_retries_when_date_truncation_missing(self):
        """Task explicitly requiring monthly grouping MUST retry when strftime/date truncation is missing."""
        from src.agents.reviewer import Reviewer
        from src.llm.ollama_client import OllamaClient

        reviewer = Reviewer(llm=OllamaClient())
        subtask = {
            "title": "Monthly revenue",
            "instruction": "Calculate total monthly revenue grouped by item",
            "expected_output": "Monthly summary",
        }
        flawed_code = "```sql\nSELECT item, SUM(price) FROM sales GROUP BY item\n```"
        raw_json = '{"decision": "approve", "confidence": 0.90, "feedback": "Looks good", "issues": []}'

        res = reviewer._parse_review_response(raw_json, threshold=0.60, specialist_output=flawed_code, subtask=subtask)
        self.assertIsNotNone(res)
        self.assertEqual(res["decision"], "retry", "Task requiring monthly grouping MUST retry if strftime/date truncation is missing")
        self.assertIn("missing monthly date truncation in SQL query", res["issues"])

    def test_non_monthly_grouping_task_approves_without_monthly_check(self):
        """Task NOT requiring monthly grouping (e.g. customers/status) MUST NOT trigger monthly check and approves normally."""
        from src.agents.reviewer import Reviewer
        from src.llm.ollama_client import OllamaClient

        reviewer = Reviewer(llm=OllamaClient())
        subtask = {
            "title": "Customer balance by status",
            "instruction": "Query the customers table and show the total account balance grouped by status.",
            "expected_output": "Total balance grouped by status",
        }
        valid_sql_code = (
            "```sql\n"
            "SELECT status, SUM(account_balance) AS total_balance\n"
            "FROM customers\n"
            "GROUP BY status;\n"
            "```"
        )
        raw_json_with_hallucinated_issue = (
            '{"decision": "retry", "confidence": 0.40, '
            '"feedback": "Code performs SQL aggregation but lacks monthly date truncation", '
            '"issues": ["missing monthly date truncation in SQL query"]}'
        )

        res = reviewer._parse_review_response(raw_json_with_hallucinated_issue, threshold=0.60, specialist_output=valid_sql_code, subtask=subtask)
        self.assertIsNotNone(res)
        self.assertEqual(res["decision"], "approve", "Non-monthly task MUST NOT fail due to hallucinated date truncation issue")
        self.assertTrue(res["approved"])
        self.assertGreaterEqual(res["confidence"], 0.75)
        self.assertNotIn("missing monthly date truncation in SQL query", res["issues"])

    def test_reviewer_catches_factual_hallucination_acronym_expansion(self):
        """
        Verify that if specialist research output contains a factual hallucination
        (e.g., expanding 'FAME' as 'Faradere Incentives for Advanced Technology Vehicles'),
        the Reviewer deterministically flags it with decision='retry', issues=['factual hallucination...'],
        and confidence < 0.50.
        """
        from src.agents.reviewer import Reviewer
        from src.llm.ollama_client import OllamaClient

        reviewer = Reviewer(llm=OllamaClient())
        hallucinated_output = (
            "The FAME (Faradere Incentives for Advanced Technology Vehicles) scheme "
            "was launched in 2015 to promote EV adoption."
        )
        subtask = {
            "id": "t1",
            "title": "Research FAME scheme",
            "instruction": "Research the FAME scheme and list acronyms",
            "expected_output": "FAME scheme details",
            "tool_logs": [{"tool_name": "web_search", "success": False, "error": "No useful result"}],
        }
        # Canned LLM response that erroneously approved the hallucinated text
        raw_llm_json = '{"decision": "approve", "confidence": 0.83, "feedback": "Good summary", "issues": []}'

        res = reviewer._parse_review_response(raw_llm_json, threshold=0.60, specialist_output=hallucinated_output, subtask=subtask)
        self.assertIsNotNone(res)
        self.assertEqual(res["decision"], "retry")
        self.assertFalse(res["approved"])
        self.assertLess(res["confidence"], 0.50)
        self.assertTrue(any("ungrounded" in issue.lower() or "factual" in issue.lower() for issue in res["issues"]))


if __name__ == "__main__":
    unittest.main()

