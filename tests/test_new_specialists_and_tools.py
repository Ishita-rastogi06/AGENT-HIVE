"""
Tests for the new domain specialists (data, writer) and tools (db_query,
api_call) added in the Phase 1 gap-closure pass.

Coverage:
  TestAgentNameFourWay     – AgentName literal has all 4 agents
  TestSupervisorRouting    – keyword routing for all 4 agents
  TestDataSpecialist       – shape, confidence, tool-call paths, retry feedback
  TestWriterSpecialist     – shape, confidence, tool-call paths, retry feedback
  TestDatabaseQueryTool    – validation, read-only guard, graceful degradation
  TestApiCallTool          – domain allowlist, method validation, HTTP errors
  TestGraphFourSpecialists – graph has 13 nodes, all retry routes present
"""

import sys
import typing
import unittest
import urllib.error
import urllib.request as _ureq
from unittest.mock import MagicMock, patch

sys.path.insert(0, __file__.split("tests")[0].rstrip("\\/"))

from src.orchestration.state import AgentHiveState, AgentName
from src.agents.supervisor import Supervisor, _pick_agent, _SIGNAL_PRIORITY
from src.agents.specialist_data import DataSpecialist
from src.agents.specialist_writer import WriterSpecialist
from src.tools.db_query_tool import DatabaseQueryTool
from src.tools.api_call_tool import ApiCallTool
from src.tools.base import ToolContext, tool_registry
from src.config import Settings


# ── shared helpers ────────────────────────────────────────────────────────────

class _NullLLM:
    def ask(self, p, s=None, *args, **kwargs): return None
    def is_available(self): return False


class _GoodLLM:
    """Returns a long, substantive answer."""
    def ask(self, p, s=None):
        return "Detailed answer: " + "x" * 300
    def is_available(self): return False


def _make_subtask(agent: str = "data") -> dict:
    return {
        "id": "t1", "title": "Test subtask",
        "instruction": "Do the task", "expected_output": "Correct output",
        "assigned_agent": agent, "dependencies": [], "status": "pending",
    }


def _make_state(agent: str = "data", extra: dict | None = None) -> AgentHiveState:
    base: AgentHiveState = {
        "task": "test task",
        "current_subtask": _make_subtask(agent),
        "retry_count": 0,
        "tool_logs": [],
        "trace": [],
    }
    if extra:
        base.update(extra)
    return base


CTX = ToolContext(agent_name="test", subtask_id="t0")


# ── AgentName literal ─────────────────────────────────────────────────────────

class TestAgentNameFourWay(unittest.TestCase):

    def test_all_four_present(self):
        args = typing.get_args(AgentName)
        for name in ("research", "coder", "data", "writer"):
            self.assertIn(name, args, f"AgentName missing '{name}'")

    def test_no_extras(self):
        args = typing.get_args(AgentName)
        self.assertEqual(set(args), {"research", "coder", "data", "writer"})

    def test_route_label_all_four_agents(self):
        from src.ui.streamlit_views import route_label
        self.assertEqual(route_label("coder")[0], "Coding Specialist")
        self.assertEqual(route_label("data")[0], "Data Specialist")
        self.assertEqual(route_label("writer")[0], "Writer Specialist")
        self.assertEqual(route_label("research")[0], "Research Specialist")


# ── Supervisor routing ────────────────────────────────────────────────────────

class TestSupervisorRouting(unittest.TestCase):

    def setUp(self):
        self.llm = _NullLLM()

    # data
    def test_sql_routes_to_data(self):
        self.assertEqual(_pick_agent("write a sql query to find top customers", self.llm), "data")

    def test_database_routes_to_data(self):
        self.assertEqual(_pick_agent("query the database for monthly revenue", self.llm), "data")

    def test_data_analysis_routes_to_data(self):
        self.assertEqual(_pick_agent("analyse data from the sales dataset", self.llm), "data")

    def test_calculate_routes_to_data(self):
        self.assertEqual(_pick_agent("calculate the average revenue per customer", self.llm), "data")

    # writer
    def test_blog_routes_to_writer(self):
        self.assertEqual(_pick_agent("write a blog post about AI", self.llm), "writer")

    def test_exact_500_word_blog_post_routes_to_writer(self):
        task = "Write a 500-word blog post about X with an engaging introduction and clear conclusion"
        self.assertEqual(_pick_agent(task, self.llm), "writer")

    def test_supervisor_run_full_flow_routes_to_writer(self):
        class _SubtaskResearchLLM:
            def ask(self, p, s=None, **kw):
                return '''{
                    "objective": "Write blog post",
                    "reasoning": "Plan",
                    "subtasks": [{
                        "id": "t1", "title": "Write post", "description": "Write post",
                        "assigned_agent": "research", "dependencies": [], "instruction": "Write post",
                        "expected_output": "Post", "status": "pending"
                    }],
                    "execution_order": ["t1"]
                }'''
            def is_available(self): return False

        sup = Supervisor(_SubtaskResearchLLM())
        task = "Write a 500-word blog post about why remote work improves developer productivity, with an engaging introduction and a clear conclusion."
        state = {"task": task, "trace": [], "tool_logs": []}
        res = sup.run(state)
        self.assertEqual(res["selected_agent"], "writer")
        self.assertEqual(res["current_subtask"]["assigned_agent"], "writer")
        self.assertIn("Supervisor → writer", res["trace"][-1])

    def test_email_draft_for_audience_routes_to_writer(self):
        task = "draft an email summary for stakeholders highlighting key milestones and next steps"
        self.assertEqual(_pick_agent(task, self.llm), "writer")

    def test_report_routes_to_writer(self):
        self.assertEqual(_pick_agent("draft a report on quarterly earnings", self.llm), "writer")

    def test_executive_summary_routes_to_writer(self):
        self.assertEqual(_pick_agent("write an executive summary of the project", self.llm), "writer")

    # coder — code-specific language, not data language
    def test_write_python_routes_to_coder(self):
        self.assertEqual(_pick_agent("write a python script to reverse a string", self.llm), "coder")

    def test_write_java_code_routes_to_coder(self):
        task = "Write a java code to print a palindrome"
        self.assertEqual(_pick_agent(task, self.llm), "coder")

    def test_algorithm_routes_to_coder(self):
        self.assertEqual(_pick_agent("implement a binary search algorithm", self.llm), "coder")

    def test_debug_routes_to_coder(self):
        self.assertEqual(_pick_agent("debug this python error in my script", self.llm), "coder")

    # research
    def test_explain_routes_to_research(self):
        self.assertEqual(_pick_agent("explain how neural networks work", self.llm), "research")

    def test_what_is_routes_to_research(self):
        self.assertEqual(_pick_agent("what is the difference between REST and GraphQL", self.llm), "research")

    def test_factual_lookup_latest_news_routes_to_research(self):
        task = "what is the latest research on fusion energy efficiency"
        self.assertEqual(_pick_agent(task, self.llm), "research")

    def test_compare_topics_routes_to_research(self):
        task = "compare quantum computing and classical computing paradigms"
        self.assertEqual(_pick_agent(task, self.llm), "research")

    # data beats coder on data-specific compound phrase
    def test_data_beats_coder_on_data_phrase(self):
        result = _pick_agent("calculate the average of the dataset columns", self.llm)
        self.assertEqual(result, "data")

    # LLM fallback for unrecognised task
    def test_llm_fallback_respected(self):
        class _DataLLM:
            def ask(self, p, s=None): return "data"
            def is_available(self): return False
        result = _pick_agent("some completely ambiguous request xyz", _DataLLM())
        self.assertEqual(result, "data")

    def test_llm_fallback_unknown_returns_research(self):
        class _JunkLLM:
            def ask(self, p, s=None): return "blahblah"
            def is_available(self): return False
        result = _pick_agent("some completely ambiguous request xyz", _JunkLLM())
        self.assertEqual(result, "research")

    # _validate_and_build_plan clamps invalid agents
    def test_invalid_agent_clamped(self):
        sup = Supervisor(_NullLLM())
        data = {
            "objective": "test", "reasoning": "",
            "subtasks": [{"id": "t1", "title": "x", "description": "x",
                          "assigned_agent": "INVALID", "dependencies": [],
                          "instruction": "x", "expected_output": "x",
                          "status": "pending"}],
            "execution_order": ["t1"],
        }
        plan = sup._validate_and_build_plan(data, "test", "research")
        self.assertEqual(plan["subtasks"][0]["assigned_agent"], "research")

    def test_all_four_agents_valid_in_plan(self):
        sup = Supervisor(_NullLLM())
        for agent in ("research", "coder", "data", "writer"):
            data = {
                "objective": "test", "reasoning": "",
                "subtasks": [{"id": "t1", "title": "x", "description": "x",
                              "assigned_agent": agent, "dependencies": [],
                              "instruction": "x", "expected_output": "x",
                              "status": "pending"}],
                "execution_order": ["t1"],
            }
            plan = sup._validate_and_build_plan(data, "test", agent)
            self.assertEqual(plan["subtasks"][0]["assigned_agent"], agent)


# ── DataSpecialist ────────────────────────────────────────────────────────────

class TestDataSpecialist(unittest.TestCase):

    def test_output_non_empty(self):
        ds = DataSpecialist(_GoodLLM())
        result = ds.run(_make_state("data"))
        self.assertTrue(result["specialist_output"])

    def test_confidence_in_range(self):
        ds = DataSpecialist(_GoodLLM())
        result = ds.run(_make_state("data"))
        self.assertGreaterEqual(result["specialist_confidence"], 0.0)
        self.assertLessEqual(result["specialist_confidence"], 1.0)

    def test_workflow_status_reviewing(self):
        ds = DataSpecialist(_GoodLLM())
        result = ds.run(_make_state("data"))
        self.assertEqual(result["workflow_status"], "reviewing")

    def test_subtask_status_completed(self):
        ds = DataSpecialist(_GoodLLM())
        result = ds.run(_make_state("data"))
        self.assertEqual(result["current_subtask"]["status"], "completed")

    def test_tool_logs_list(self):
        ds = DataSpecialist(_GoodLLM())
        result = ds.run(_make_state("data"))
        self.assertIsInstance(result["tool_logs"], list)

    def test_no_subtask_returns_failed(self):
        ds = DataSpecialist(_GoodLLM())
        state: AgentHiveState = {"task": "test", "trace": [], "tool_logs": []}
        result = ds.run(state)
        self.assertEqual(result["workflow_status"], "failed")
        self.assertEqual(result["specialist_confidence"], 0.0)

    def test_llm_unavailable_returns_failed(self):
        ds = DataSpecialist(_NullLLM())
        result = ds.run(_make_state("data"))
        self.assertEqual(result["workflow_status"], "failed")

    def test_retry_feedback_in_prompt_context(self):
        """On retry, reviewer feedback should be visible in state."""
        ds = DataSpecialist(_GoodLLM())
        state = _make_state("data", {
            "retry_count": 1,
            "review": {"feedback": "Add statistical context."},
        })
        result = ds.run(state)
        # Specialist ran without crashing and still produced output.
        self.assertTrue(result["specialist_output"])

    def test_confidence_higher_with_structured_output(self):
        class _TableLLM:
            def ask(self, p, s=None):
                return "| Column | Value |\n|--------|-------|\n| Revenue | $1000 |\n" + "x" * 200
            def is_available(self): return False
        ds = DataSpecialist(_TableLLM())
        result = ds.run(_make_state("data"))
        self.assertGreater(result["specialist_confidence"], 0.50)

    def test_db_query_tool_invoked_for_sql_instruction(self):
        """Data specialist tries db_query when instruction contains 'query'."""
        invoked = []
        original_invoke = tool_registry.invoke

        def _mock_invoke(tool_name, arguments=None, context=None):
            invoked.append(tool_name)
            from src.tools.base import ToolResult
            return ToolResult(success=True, output="42 rows"), {"call_id": "x"}

        tool_registry.invoke = _mock_invoke
        try:
            ds = DataSpecialist(_GoodLLM())
            state = _make_state("data", {
                "current_subtask": {
                    "id": "t1", "title": "Count rows",
                    "instruction": "SELECT COUNT(*) FROM table;",
                    "expected_output": "Row count", "assigned_agent": "data",
                    "dependencies": [], "status": "pending",
                },
            })
            result = ds.run(state)
            self.assertIn("db_query", invoked)
        finally:
            tool_registry.invoke = original_invoke

    def test_db_query_tool_invoked_with_llm_generated_query(self):
        """Verify db_query is invoked with the EXACT SELECT query generated by the LLM, not a hardcoded probe query."""
        invoked_args = []
        original_invoke = tool_registry.invoke

        def _mock_invoke(tool_name, arguments=None, context=None):
            if tool_name == "db_query":
                invoked_args.append(arguments)
            from src.tools.base import ToolResult
            return ToolResult(success=True, output="[('Active', 1000.0)]"), {"tool": "db_query"}

        tool_registry.invoke = _mock_invoke
        try:
            expected_sql = "SELECT status, SUM(balance) AS total_balance FROM customers GROUP BY status;"
            class _SqlLLM:
                def ask(self, p, s=None):
                    return f"```sql\n{expected_sql}\n```"
                def is_available(self): return True

            ds = DataSpecialist(_SqlLLM())
            state = _make_state("data", {
                "current_subtask": {
                    "id": "t_sql", "title": "Customer balance by status",
                    "instruction": "Query the customers table and show the total account balance grouped by status.",
                    "expected_output": "Total balance grouped by status", "assigned_agent": "data",
                    "dependencies": [], "status": "pending",
                },
            })
            result = ds.run(state)
            self.assertGreaterEqual(len(invoked_args), 1, "db_query should be invoked for target data query")
            actual_query = invoked_args[-1].get("query")
            self.assertEqual(actual_query, expected_sql, f"Expected db_query to be invoked with '{expected_sql}', but got '{actual_query}'")
            self.assertNotIn("information_schema.tables", actual_query)
        finally:
            tool_registry.invoke = original_invoke


# ── WriterSpecialist ──────────────────────────────────────────────────────────

class TestWriterSpecialist(unittest.TestCase):

    def test_output_non_empty(self):
        ws = WriterSpecialist(_GoodLLM())
        result = ws.run(_make_state("writer"))
        self.assertTrue(result["specialist_output"])

    def test_confidence_in_range(self):
        ws = WriterSpecialist(_GoodLLM())
        result = ws.run(_make_state("writer"))
        self.assertGreaterEqual(result["specialist_confidence"], 0.0)
        self.assertLessEqual(result["specialist_confidence"], 1.0)

    def test_workflow_status_reviewing(self):
        ws = WriterSpecialist(_GoodLLM())
        result = ws.run(_make_state("writer"))
        self.assertEqual(result["workflow_status"], "reviewing")

    def test_subtask_status_completed(self):
        ws = WriterSpecialist(_GoodLLM())
        result = ws.run(_make_state("writer"))
        self.assertEqual(result["current_subtask"]["status"], "completed")

    def test_no_subtask_returns_failed(self):
        ws = WriterSpecialist(_GoodLLM())
        state: AgentHiveState = {"task": "test", "trace": [], "tool_logs": []}
        result = ws.run(state)
        self.assertEqual(result["workflow_status"], "failed")

    def test_llm_unavailable_returns_failed(self):
        ws = WriterSpecialist(_NullLLM())
        result = ws.run(_make_state("writer"))
        self.assertEqual(result["workflow_status"], "failed")

    def test_long_output_higher_confidence(self):
        class _LongLLM:
            def ask(self, p, s=None):
                return "# Report\n\n" + "Paragraph content. " * 50
            def is_available(self): return False
        ws = WriterSpecialist(_LongLLM())
        result = ws.run(_make_state("writer"))
        self.assertGreater(result["specialist_confidence"], 0.60)

    def test_retry_feedback_does_not_crash(self):
        ws = WriterSpecialist(_GoodLLM())
        state = _make_state("writer", {
            "retry_count": 1,
            "review": {"feedback": "Make the introduction more engaging."},
        })
        result = ws.run(state)
        self.assertTrue(result["specialist_output"])

    def test_tool_logs_accumulated(self):
        ws = WriterSpecialist(_GoodLLM())
        existing_log = [{"call_id": "existing", "tool_name": "web_search"}]
        state = _make_state("writer", {"tool_logs": existing_log})
        result = ws.run(state)
        # Existing log entry should still be present.
        self.assertTrue(any(
            log.get("call_id") == "existing" for log in result["tool_logs"]
        ))


# ── DatabaseQueryTool ─────────────────────────────────────────────────────────

class TestDatabaseQueryTool(unittest.TestCase):

    def setUp(self):
        self.tool = DatabaseQueryTool()

    # validate_arguments
    def test_empty_query_raises(self):
        with self.assertRaises(ValueError):
            self.tool.validate_arguments({"query": ""})

    def test_non_string_query_raises(self):
        with self.assertRaises(ValueError):
            self.tool.validate_arguments({"query": 123})

    def test_select_allowed_in_read_only(self):
        self.tool.validate_arguments({"query": "SELECT * FROM users LIMIT 10", "read_only": True})

    def test_insert_blocked_in_read_only(self):
        with self.assertRaises(PermissionError):
            self.tool.validate_arguments({"query": "INSERT INTO t VALUES (1)", "read_only": True})

    def test_update_blocked_in_read_only(self):
        with self.assertRaises(PermissionError):
            self.tool.validate_arguments({"query": "UPDATE t SET x=1", "read_only": True})

    def test_delete_blocked_in_read_only(self):
        with self.assertRaises(PermissionError):
            self.tool.validate_arguments({"query": "DELETE FROM t WHERE id=1", "read_only": True})

    def test_drop_blocked_in_read_only(self):
        with self.assertRaises(PermissionError):
            self.tool.validate_arguments({"query": "DROP TABLE users", "read_only": True})

    def test_create_blocked_in_read_only(self):
        with self.assertRaises(PermissionError):
            self.tool.validate_arguments({"query": "CREATE TABLE x (id INT)", "read_only": True})

    def test_truncate_blocked_in_read_only(self):
        with self.assertRaises(PermissionError):
            self.tool.validate_arguments({"query": "TRUNCATE TABLE logs", "read_only": True})

    def test_invalid_params_type_raises(self):
        with self.assertRaises(TypeError):
            self.tool.validate_arguments({"query": "SELECT 1", "parameters": "bad"})

    def test_list_params_allowed(self):
        self.tool.validate_arguments({"query": "SELECT * FROM t WHERE id=%s", "parameters": [1]})

    # execute — graceful degradation
    def test_no_db_url_returns_failure(self):
        import os
        orig_db_url = os.environ.pop("DB_URL", None)
        try:
            result = self.tool.execute({"query": "SELECT 1", "read_only": True}, CTX)
            self.assertFalse(result.success)
            self.assertIn("DB_URL", result.error)
        finally:
            if orig_db_url is not None:
                os.environ["DB_URL"] = orig_db_url

    def test_psycopg2_missing_returns_failure(self):
        import os
        orig_db_url = os.environ.get("DB_URL")
        os.environ["DB_URL"] = "postgresql://fake:fake@localhost/fake"
        import sys
        # Temporarily hide psycopg2
        orig = sys.modules.get("psycopg2")
        sys.modules["psycopg2"] = None  # type: ignore
        try:
            result = self.tool.execute({"query": "SELECT 1"}, CTX)
            self.assertFalse(result.success)
            self.assertIn("psycopg2", result.error)
        finally:
            if orig is None:
                del sys.modules["psycopg2"]
            else:
                sys.modules["psycopg2"] = orig
            if orig_db_url is not None:
                os.environ["DB_URL"] = orig_db_url
            else:
                os.environ.pop("DB_URL", None)

    def test_connection_failure_returns_failure(self):
        import os
        orig_db_url = os.environ.get("DB_URL")
        os.environ["DB_URL"] = "postgresql://bad:bad@127.0.0.1:1/nonexistent"
        try:
            # psycopg2 may or may not be installed; either path should return failure.
            result = self.tool.execute({"query": "SELECT 1"}, CTX)
            self.assertFalse(result.success)
        finally:
            if orig_db_url is not None:
                os.environ["DB_URL"] = orig_db_url
            else:
                os.environ.pop("DB_URL", None)

    def test_audit_metadata_in_registry_invoke(self):
        """tool_registry.invoke returns a valid audit record for db_query."""
        import os
        orig_db_url = os.environ.pop("DB_URL", None)
        try:
            result, audit = tool_registry.invoke(
                "db_query",
                {"query": "SELECT 1", "read_only": True},
                CTX,
            )
            self.assertIn("call_id", audit)
            self.assertIn("duration_ms", audit)
            self.assertFalse(result.success)  # no DB_URL, but audit still created
        finally:
            if orig_db_url is not None:
                os.environ["DB_URL"] = orig_db_url


# ── ApiCallTool ───────────────────────────────────────────────────────────────

class TestApiCallTool(unittest.TestCase):

    def setUp(self):
        self.tool = ApiCallTool()

    # validate_arguments — domain allowlist
    def test_disallowed_domain_blocked(self):
        with self.assertRaises(PermissionError):
            self.tool.validate_arguments({"url": "https://evil.com/steal"})

    def test_http_disallowed_domain_blocked(self):
        with self.assertRaises(PermissionError):
            self.tool.validate_arguments({"url": "http://evil.com/steal"})

    def test_allowed_domain_passes(self):
        self.tool.validate_arguments({
            "url": "https://jsonplaceholder.typicode.com/posts/1",
            "method": "GET",
        })

    def test_subdomain_of_allowed_domain_passes(self):
        # api.github.com is in the default allowlist.
        self.tool.validate_arguments({
            "url": "https://api.github.com/repos/octocat/Hello-World",
        })

    # validate_arguments — method
    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            self.tool.validate_arguments({
                "url": "https://jsonplaceholder.typicode.com/posts",
                "method": "HEAD",
            })

    def test_valid_methods_accepted(self):
        for method in ("GET", "POST", "PUT", "PATCH", "DELETE"):
            self.tool.validate_arguments({
                "url": "https://jsonplaceholder.typicode.com/posts",
                "method": method,
            })

    # validate_arguments — body
    def test_dict_body_accepted(self):
        self.tool.validate_arguments({
            "url": "https://jsonplaceholder.typicode.com/posts",
            "method": "POST",
            "body": {"title": "test"},
        })

    def test_string_body_accepted(self):
        self.tool.validate_arguments({
            "url": "https://jsonplaceholder.typicode.com/posts",
            "method": "POST",
            "body": '{"title": "test"}',
        })

    def test_invalid_headers_type_raises(self):
        with self.assertRaises(TypeError):
            self.tool.validate_arguments({
                "url": "https://jsonplaceholder.typicode.com/posts",
                "headers": "not-a-dict",
            })

    # execute — network errors degrade gracefully
    def _offline_execute(self, **kwargs):
        _orig = _ureq.urlopen
        def _raise(*a, **kw): raise urllib.error.URLError("offline")
        _ureq.urlopen = _raise
        try:
            return self.tool.execute(
                {"url": "https://jsonplaceholder.typicode.com/posts/1", **kwargs},
                CTX,
            )
        finally:
            _ureq.urlopen = _orig

    def test_network_error_returns_failure(self):
        result = self._offline_execute()
        self.assertFalse(result.success)
        self.assertIn("Network error", result.error)

    def test_http_error_returns_failure(self):
        _orig = _ureq.urlopen
        def _raise(*a, **kw):
            raise urllib.error.HTTPError(
                "https://jsonplaceholder.typicode.com/posts/999",
                404, "Not Found", {}, None,
            )
        _ureq.urlopen = _raise
        try:
            result = self.tool.execute(
                {"url": "https://jsonplaceholder.typicode.com/posts/999"},
                CTX,
            )
        finally:
            _ureq.urlopen = _orig
        self.assertFalse(result.success)
        self.assertIn("404", result.error)

    def test_successful_json_response_parsed(self):
        """Mock a successful JSON response and verify parsing."""
        import io
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.read.return_value = b'{"id": 1, "title": "Test"}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch.object(_ureq, "urlopen", return_value=mock_resp):
            result = self.tool.execute(
                {"url": "https://jsonplaceholder.typicode.com/posts/1"},
                CTX,
            )
        self.assertTrue(result.success)
        self.assertEqual(result.output["id"], 1)
        self.assertEqual(result.output["title"], "Test")

    def test_successful_text_response_returned_as_string(self):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.headers = {"Content-Type": "text/plain"}
        mock_resp.read.return_value = b"Hello, world"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch.object(_ureq, "urlopen", return_value=mock_resp):
            result = self.tool.execute(
                {"url": "https://jsonplaceholder.typicode.com/posts/1"},
                CTX,
            )
        self.assertTrue(result.success)
        self.assertEqual(result.output, "Hello, world")

    def test_binary_response_rejected(self):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.headers = {"Content-Type": "image/png"}
        mock_resp.read.return_value = b"\x89PNG\r\n"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch.object(_ureq, "urlopen", return_value=mock_resp):
            result = self.tool.execute(
                {"url": "https://jsonplaceholder.typicode.com/image.png"},
                CTX,
            )
        self.assertFalse(result.success)
        self.assertIn("Binary", result.error)

    def test_metadata_contains_url_and_status(self):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.read.return_value = b'{"ok": true}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch.object(_ureq, "urlopen", return_value=mock_resp):
            result = self.tool.execute(
                {"url": "https://jsonplaceholder.typicode.com/posts/1"},
                CTX,
            )
        self.assertIn("url", result.metadata)
        self.assertIn("status_code", result.metadata)
        self.assertEqual(result.metadata["status_code"], 200)

    def test_allowed_domain_allowlist_logic(self):
        """is_api_domain_allowed covers subdomains."""
        ts = Settings.__new__(Settings)
        object.__setattr__(ts, "allowed_api_domains",
                           ("api.github.com", "jsonplaceholder.typicode.com"))
        self.assertTrue(ts.is_api_domain_allowed("api.github.com"))
        self.assertTrue(ts.is_api_domain_allowed("sub.api.github.com"))
        self.assertFalse(ts.is_api_domain_allowed("evil.com"))
        self.assertFalse(ts.is_api_domain_allowed("notgithub.com"))

    def test_audit_record_from_registry_invoke(self):
        result, audit = tool_registry.invoke(
            "api_call",
            {"url": "https://evil-not-allowed.example.com/"},
            CTX,
        )
        # Domain not allowed → should fail but audit record still created.
        self.assertFalse(result.success)
        self.assertIn("call_id", audit)
        self.assertIn("duration_ms", audit)


# ── Graph topology ────────────────────────────────────────────────────────────

class TestGraphFourSpecialists(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.orchestration.graph import build_graph
        cls.graph = build_graph()
        cls.nodes = set(cls.graph.nodes)

    def _assert_node(self, name):
        self.assertIn(name, self.nodes, f"Missing graph node: '{name}'")

    def test_core_nodes_present(self):
        for n in ("memory_recall", "supervisor", "reviewer", "finalise", "memory_save"):
            self._assert_node(n)

    def test_all_four_specialist_nodes(self):
        for n in ("research", "coder", "data", "writer"):
            self._assert_node(n)

    def test_all_four_retry_nodes(self):
        for n in ("pre_retry_research", "pre_retry_coder",
                  "pre_retry_data", "pre_retry_writer"):
            self._assert_node(n)

    def test_total_node_count(self):
        # 5 core + 4 specialists + 4 retry = 13 + __start__
        self.assertGreaterEqual(len(self.nodes), 13)

    def test_route_after_review_data_retry(self):
        from src.orchestration.graph import _route_after_review, _ROUTE_RETRY_DATA
        state = {
            "review": {"decision": "retry"},
            "retry_count": 0,
            "max_retries": 2,
            "selected_agent": "data",
        }
        self.assertEqual(_route_after_review(state), _ROUTE_RETRY_DATA)

    def test_route_after_review_writer_retry(self):
        from src.orchestration.graph import _route_after_review, _ROUTE_RETRY_WRITER
        state = {
            "review": {"decision": "retry"},
            "retry_count": 1,
            "max_retries": 2,
            "selected_agent": "writer",
        }
        self.assertEqual(_route_after_review(state), _ROUTE_RETRY_WRITER)

    def test_route_after_review_data_approve(self):
        from src.orchestration.graph import _route_after_review, _ROUTE_FINALISE
        state = {
            "review": {"decision": "approve"},
            "retry_count": 0,
            "max_retries": 2,
            "selected_agent": "data",
        }
        self.assertEqual(_route_after_review(state), _ROUTE_FINALISE)


# ── 8. CodingSpecialist Language Gating ──────────────────────────────────────

class TestCoderLanguageGating(unittest.TestCase):
    """Verifies that python_sandbox is only invoked for Python tasks, and skipped for Java/other languages."""

    def test_python_task_invokes_sandbox(self):
        from src.agents.specialist_coder import CodingSpecialist
        mock_llm = MagicMock()
        mock_llm.ask.return_value = "def palindrome(s):\n    return s == s[::-1]"

        coder = CodingSpecialist(mock_llm)
        subtask = {
            "id": "t_py",
            "title": "Python palindrome",
            "instruction": "Write a python function to check if a string is a palindrome and execute it",
            "expected_output": "Working code",
            "assigned_agent": "coder",
            "dependencies": [],
            "status": "pending",
        }
        state: AgentHiveState = {
            "task": "Python task",
            "current_subtask": subtask,
            "subtasks": [subtask],
            "outputs": {},
            "subtask_outputs": {},
            "subtask_logs": {},
            "execution_history": [],
            "retry_count": 0,
            "max_retries": 2,
            "selected_agent": "coder",
            "grounding_passed": True,
        }

        with patch("src.tools.base.tool_registry.invoke") as mock_invoke:
            mock_invoke.return_value = (MagicMock(success=True, output="True", error=None), {"tool": "python_sandbox"})
            result_state = coder.run(state)
            
            tool_logs = result_state.get("tool_logs", [])
            sandbox_calls = [l for l in tool_logs if l.get("tool") == "python_sandbox"]
            self.assertEqual(len(sandbox_calls), 1, "Python code tasks MUST invoke python_sandbox when executable")

    def test_java_task_skips_sandbox(self):
        from src.agents.specialist_coder import CodingSpecialist
        mock_llm = MagicMock()
        mock_llm.ask.return_value = "public class Palindrome { public static void main(String[] args) {} }"

        coder = CodingSpecialist(mock_llm)
        subtask = {
            "id": "t_java",
            "title": "Java palindrome",
            "instruction": "Write a java program to print a palindrome and execute algorithm",
            "expected_output": "Working java code",
            "assigned_agent": "coder",
            "dependencies": [],
            "status": "pending",
        }
        state: AgentHiveState = {
            "task": "Java task",
            "current_subtask": subtask,
            "subtasks": [subtask],
            "outputs": {},
            "subtask_outputs": {},
            "subtask_logs": {},
            "execution_history": [],
            "retry_count": 0,
            "max_retries": 2,
            "selected_agent": "coder",
            "grounding_passed": True,
        }

        with patch("src.tools.base.tool_registry.invoke") as mock_invoke:
            result_state = coder.run(state)
            
            tool_logs = result_state.get("tool_logs", [])
            sandbox_calls = [l for l in tool_logs if l.get("tool") == "python_sandbox"]
            self.assertEqual(len(sandbox_calls), 0, "Java/other-language tasks MUST NOT invoke python_sandbox")
            mock_invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()

