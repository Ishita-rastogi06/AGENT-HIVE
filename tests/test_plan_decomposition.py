"""
Tests for supervisor plan decomposition (Item 1 requirement).

Verifies:
  - _fallback_plan always returns a valid TaskPlan with correct shape
  - _validate_and_build_plan produces ≥2 subtasks when given multi-part input
  - Dependencies between subtasks are preserved exactly
  - execution_order only contains ids that exist in subtasks
  - supervisor.run() stamps workflow_status=planning and correct fields
  - AgentName literal only contains "research" and "coder"
"""
import sys
import unittest

sys.path.insert(0, __file__.split("tests")[0].rstrip("\\/"))

from src.agents.supervisor import Supervisor
from src.orchestration.state import AgentHiveState, AgentName


class _MockLLM:
    """LLM that always returns None — forces fallback paths."""
    def ask(self, prompt, system=None, *args, **kwargs):
        return None
    def is_available(self):
        return False


class _JsonLLM:
    """LLM that returns a canned multi-subtask JSON response."""
    RESPONSE = """{
        "objective": "Compare REST and GraphQL, then write a Python client",
        "reasoning": "Two distinct concerns: research first, then code.",
        "subtasks": [
            {
                "id": "t1",
                "title": "Compare REST and GraphQL",
                "description": "Research the two approaches",
                "assigned_agent": "research",
                "dependencies": [],
                "instruction": "Compare REST and GraphQL for mobile APIs",
                "expected_output": "A concise comparison with pros/cons",
                "status": "pending"
            },
            {
                "id": "t2",
                "title": "Write a Python REST client example",
                "description": "Implement a minimal example",
                "assigned_agent": "coder",
                "dependencies": ["t1"],
                "instruction": "Write a Python REST client using urllib",
                "expected_output": "Working Python code with usage example",
                "status": "pending"
            }
        ],
        "execution_order": ["t1", "t2"]
    }"""

    def ask(self, prompt, system=None, *args, **kwargs):
        return self.RESPONSE

    def is_available(self):
        return False


class TestFallbackPlan(unittest.TestCase):
    """_fallback_plan always produces a valid single-subtask TaskPlan."""

    def setUp(self):
        self.sup = Supervisor(_MockLLM())

    def test_fallback_has_objective(self):
        plan = self.sup._fallback_plan("explain quicksort", "research")
        self.assertTrue(plan["objective"])

    def test_fallback_has_one_subtask(self):
        plan = self.sup._fallback_plan("explain quicksort", "research")
        self.assertEqual(len(plan["subtasks"]), 1)

    def test_fallback_subtask_shape(self):
        plan = self.sup._fallback_plan("explain quicksort", "research")
        st = plan["subtasks"][0]
        for key in ("id", "title", "description", "assigned_agent",
                    "dependencies", "instruction", "expected_output", "status"):
            self.assertIn(key, st, f"Missing key '{key}' in fallback subtask")

    def test_fallback_subtask_status_pending(self):
        plan = self.sup._fallback_plan("explain quicksort", "research")
        self.assertEqual(plan["subtasks"][0]["status"], "pending")

    def test_fallback_execution_order_matches_subtask_id(self):
        plan = self.sup._fallback_plan("explain quicksort", "research")
        subtask_ids = {st["id"] for st in plan["subtasks"]}
        for eid in plan["execution_order"]:
            self.assertIn(eid, subtask_ids)

    def test_fallback_plan_status_pending(self):
        plan = self.sup._fallback_plan("explain quicksort", "research")
        self.assertEqual(plan["status"], "pending")

    def test_fallback_title_truncates_at_word_boundary(self):
        long_task = "Write a comprehensive Python script that handles very long tasks without cutting words in half unexpectedly"
        plan = self.sup._fallback_plan(long_task, "coder")
        title = plan["subtasks"][0]["title"]
        self.assertLessEqual(len(title), 65)
        self.assertTrue(title.endswith("..."))
        # Ensure no word is split awkwardly mid-character
        self.assertNotIn("unexpecte...", title)

    def test_normalise_plan_truncates_long_titles_at_word_boundary(self):
        long_title = "Implement advanced data aggregation and analytics algorithm with pandas and sqlite3 database"
        raw_data = {
            "objective": "Task objective",
            "subtasks": [{
                "id": "t1",
                "title": long_title,
                "assigned_agent": "coder",
                "instruction": "Do coding",
            }]
        }
        plan = self.sup._validate_and_build_plan(raw_data, "Do coding", "coder")
        st_title = plan["subtasks"][0]["title"]
        self.assertLessEqual(len(st_title), 65)
        self.assertTrue(st_title.endswith("..."))
        self.assertNotIn("dat...", st_title)

    def test_fallback_assigned_agent_propagated(self):
        plan = self.sup._fallback_plan("write fibonacci", "coder")
        self.assertEqual(plan["subtasks"][0]["assigned_agent"], "coder")


class TestMultiSubtaskDecomposition(unittest.TestCase):
    """_validate_and_build_plan produces ≥2 subtasks with correct dependencies."""

    def setUp(self):
        self.sup = Supervisor(_MockLLM())
        self.data = {
            "objective": "Compare REST and GraphQL, then write code",
            "reasoning": "Two concerns",
            "subtasks": [
                {
                    "id": "t1", "title": "Research REST vs GraphQL",
                    "description": "Compare APIs", "assigned_agent": "research",
                    "dependencies": [], "instruction": "Compare REST and GraphQL",
                    "expected_output": "Comparison", "status": "pending",
                },
                {
                    "id": "t2", "title": "Write Python REST client",
                    "description": "Code it", "assigned_agent": "coder",
                    "dependencies": ["t1"], "instruction": "Write Python code",
                    "expected_output": "Working code", "status": "pending",
                },
            ],
            "execution_order": ["t1", "t2"],
        }

    def test_produces_two_subtasks(self):
        plan = self.sup._validate_and_build_plan(self.data, "multi-part task", "research")
        self.assertEqual(len(plan["subtasks"]), 2)

    def test_execution_order_preserved(self):
        plan = self.sup._validate_and_build_plan(self.data, "multi-part task", "research")
        self.assertEqual(plan["execution_order"], ["t1", "t2"])

    def test_dependency_preserved(self):
        plan = self.sup._validate_and_build_plan(self.data, "multi-part task", "research")
        t2 = next(st for st in plan["subtasks"] if st["id"] == "t2")
        self.assertEqual(t2["dependencies"], ["t1"])

    def test_first_subtask_has_no_dependencies(self):
        plan = self.sup._validate_and_build_plan(self.data, "multi-part task", "research")
        t1 = next(st for st in plan["subtasks"] if st["id"] == "t1")
        self.assertEqual(t1["dependencies"], [])

    def test_execution_order_only_valid_ids(self):
        plan = self.sup._validate_and_build_plan(self.data, "multi-part task", "research")
        subtask_ids = {st["id"] for st in plan["subtasks"]}
        for eid in plan["execution_order"]:
            self.assertIn(eid, subtask_ids)

    def test_all_subtask_fields_present(self):
        plan = self.sup._validate_and_build_plan(self.data, "multi-part task", "research")
        for st in plan["subtasks"]:
            for key in ("id", "title", "description", "assigned_agent",
                        "dependencies", "instruction", "expected_output", "status"):
                self.assertIn(key, st)

    def test_objective_set(self):
        plan = self.sup._validate_and_build_plan(self.data, "multi-part task", "research")
        self.assertEqual(plan["objective"], "Compare REST and GraphQL, then write code")

    def test_reasoning_set(self):
        plan = self.sup._validate_and_build_plan(self.data, "multi-part task", "research")
        self.assertEqual(plan["reasoning"], "Two concerns")

    def test_empty_subtasks_falls_back(self):
        bad_data = {"objective": "test", "subtasks": [], "execution_order": []}
        plan = self.sup._validate_and_build_plan(bad_data, "fallback task", "research")
        self.assertGreaterEqual(len(plan["subtasks"]), 1)


class TestLLMDecomposition(unittest.TestCase):
    """_create_plan uses LLM JSON when available."""

    def test_json_llm_produces_two_subtasks(self):
        sup = Supervisor(_JsonLLM())
        plan = sup._create_plan("Compare REST and GraphQL then write a Python client", "research")
        self.assertGreaterEqual(len(plan["subtasks"]), 2)

    def test_json_llm_deps_wired(self):
        sup = Supervisor(_JsonLLM())
        plan = sup._create_plan("Compare REST and GraphQL then write a Python client", "research")
        t2 = next((st for st in plan["subtasks"] if st["id"] == "t2"), None)
        self.assertIsNotNone(t2)
        self.assertIn("t1", t2["dependencies"])


class TestConceptualQuestionSingleSubtask(unittest.TestCase):
    """Simple conceptual questions must remain as a single subtask."""

    def test_simple_conceptual_question_collapses_to_single_subtask(self):
        """Confirm a simple conceptual question without sequential steps collapses to 1 subtask."""
        raw_data = {
            "objective": "Explain key differences between REST and GraphQL APIs",
            "reasoning": "Simple conceptual comparison",
            "subtasks": [
                {
                    "id": "t1", "title": "Define REST",
                    "description": "Define REST API", "assigned_agent": "research",
                    "dependencies": [], "instruction": "Define REST API",
                    "expected_output": "REST overview", "status": "pending",
                },
                {
                    "id": "t2", "title": "Compare REST and GraphQL",
                    "description": "Compare query mechanisms", "assigned_agent": "research",
                    "dependencies": ["t1"], "instruction": "Compare query mechanisms",
                    "expected_output": "Comparison", "status": "pending",
                },
                {
                    "id": "t3", "title": "Summarize pros and cons",
                    "description": "Provide pros and cons", "assigned_agent": "research",
                    "dependencies": ["t2"], "instruction": "Summarize pros and cons",
                    "expected_output": "Summary", "status": "pending",
                },
            ],
            "execution_order": ["t1", "t2", "t3"],
        }
        sup = Supervisor(_MockLLM())
        plan = sup._validate_and_build_plan(
            raw_data,
            "Explain the key differences between REST and GraphQL APIs",
            "research",
        )
        self.assertEqual(len(plan["subtasks"]), 1)
        self.assertEqual(
            plan["subtasks"][0]["instruction"],
            "Explain the key differences between REST and GraphQL APIs"
        )

    def test_multi_step_sequential_task_retains_multiple_subtasks(self):
        """Confirm tasks with explicit sequential dependency ('then') preserve multiple subtasks."""
        raw_data = {
            "objective": "Research REST vs GraphQL then write code",
            "reasoning": "Sequential research and coding",
            "subtasks": [
                {
                    "id": "t1", "title": "Research REST vs GraphQL",
                    "description": "Research", "assigned_agent": "research",
                    "dependencies": [], "instruction": "Research REST vs GraphQL",
                    "expected_output": "Research doc", "status": "pending",
                },
                {
                    "id": "t2", "title": "Write Python client",
                    "description": "Code", "assigned_agent": "coder",
                    "dependencies": ["t1"], "instruction": "Write client code",
                    "expected_output": "Python file", "status": "pending",
                },
            ],
            "execution_order": ["t1", "t2"],
        }
        sup = Supervisor(_MockLLM())
        plan = sup._validate_and_build_plan(
            raw_data,
            "Research REST vs GraphQL then write code",
            "research",
        )
        self.assertEqual(len(plan["subtasks"]), 2)


class TestSupervisorRun(unittest.TestCase):
    """supervisor.run() returns a correctly-shaped state update."""

    def setUp(self):
        self.sup = Supervisor(_MockLLM())

    def test_run_stamps_planning(self):
        state: AgentHiveState = {"task": "explain quicksort", "trace": []}
        result = self.sup.run(state)
        self.assertEqual(result["workflow_status"], "planning")

    def test_run_returns_task_plan(self):
        state: AgentHiveState = {"task": "explain quicksort", "trace": []}
        result = self.sup.run(state)
        self.assertIn("task_plan", result)
        self.assertIn("subtasks", result["task_plan"])

    def test_run_sets_retry_count_zero(self):
        state: AgentHiveState = {"task": "explain quicksort", "trace": []}
        result = self.sup.run(state)
        self.assertEqual(result["retry_count"], 0)

    def test_run_selected_agent_valid(self):
        state: AgentHiveState = {"task": "explain quicksort", "trace": []}
        result = self.sup.run(state)
        self.assertIn(result["selected_agent"], ("research", "coder"))

    def test_run_sets_pending_and_completed_ids(self):
        state: AgentHiveState = {"task": "explain quicksort", "trace": []}
        result = self.sup.run(state)
        self.assertIn("pending_subtask_ids", result)
        self.assertIn("completed_subtask_ids", result)
        self.assertIsInstance(result["completed_subtask_ids"], list)
        self.assertEqual(result["completed_subtask_ids"], [])

    def test_run_appends_to_trace(self):
        state: AgentHiveState = {"task": "explain quicksort", "trace": ["existing"]}
        result = self.sup.run(state)
        self.assertIn("existing", result["trace"])
        self.assertGreater(len(result["trace"]), 1)


class TestAgentNameLiteral(unittest.TestCase):
    """AgentName only contains agents that actually exist."""

    def test_only_valid_agents(self):
        # The Literal type args aren't directly inspectable at runtime without
        # typing.get_args, but we verify indirectly by checking the supervisor
        # only routes to known agents.
        sup = Supervisor(_MockLLM())
        for task in [
            "explain quicksort",
            "write fibonacci",
            "compare REST and GraphQL",
            "debug this python error",
        ]:
            state: AgentHiveState = {"task": task, "trace": []}
            result = sup.run(state)
            self.assertIn(
                result["selected_agent"], ("research", "coder"),
                f"Unexpected agent for task '{task}': {result['selected_agent']}"
            )

    def test_all_four_agents_in_literal(self):
        """All four domain specialists must be present in the AgentName literal."""
        import typing
        from src.orchestration.state import AgentName
        args = typing.get_args(AgentName)
        for expected in ("research", "coder", "data", "writer"):
            self.assertIn(expected, args, f"AgentName is missing '{expected}'")


if __name__ == "__main__":
    unittest.main()
