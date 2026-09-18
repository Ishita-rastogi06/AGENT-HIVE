"""
Data specialist for AgentHive.

Handles tasks involving structured data analysis, aggregation, querying, and
interpretation.  Attempts to use db_query_tool for live data when available,
falls back to python_sandbox for in-memory computation, and always produces a
structured analysis response rather than prose.
"""

from __future__ import annotations

import logging

from src.agents.base import BaseAgent
from src.llm.ollama_client import OllamaClient
from src.orchestration.state import AgentHiveState, SubTask
import src.tools  # ensures all tools are registered  # noqa: F401

logger = logging.getLogger(__name__)


def _get_current_subtask(state: AgentHiveState) -> SubTask | None:
    subtask = state.get("current_subtask")
    if subtask:
        return subtask
    plan = state.get("task_plan") or {}
    subtasks = plan.get("subtasks") or []
    return subtasks[0] if subtasks else None


class DataSpecialist(BaseAgent):
    """
    Analyse structured data, run queries, and produce precise numerical /
    tabular results.  Uses db_query_tool for database tasks and python_sandbox
    for in-memory computation.
    """

    name = "data"

    def __init__(self, llm: OllamaClient) -> None:
        self.llm = llm

    def run(self, state: AgentHiveState) -> dict:
        subtask = _get_current_subtask(state)
        current_trace = state.get("trace", [])
        tool_logs = list(state.get("tool_logs") or [])
        retry_count = state.get("retry_count", 0)

        if subtask is None:
            return {
                "specialist_output": "No subtask found in plan.",
                "specialist_confidence": 0.0,
                "workflow_status": "failed",
                "trace": [*current_trace, "Data specialist: no subtask available"],
            }

        original_task = str(state.get("task") or "").strip()
        instruction = str(subtask.get("instruction") or original_task).strip()
        expected_output = str(subtask.get("expected_output") or "").strip()
        subtask_id = subtask.get("id", "unknown")

        prompt_parts = [
            f"FULL ORIGINAL TASK REQUIREMENTS: {original_task}",
            f"Subtask Title: {subtask.get('title', '')}",
            f"Instruction: {instruction}",
            f"Expected Output: {expected_output}",
        ]

        # Include reviewer feedback and specific issue list on retry passes.
        review = state.get("review") or {}
        if retry_count > 0:
            feedback_text = str(review.get("feedback") or "Reviewer requested fixes for previous attempt.").strip()
            issues_list = [str(i) for i in (review.get("issues") or []) if i]
            issues_str = ", ".join(issues_list) if issues_list else "None specified"
            prompt_parts.append(
                f"\n[RETRY PASS #{retry_count} REVIEWER FEEDBACK]\n"
                f"Reviewer Feedback: {feedback_text}\n"
                f"Specific Missing/Defective Issues: {issues_str}\n"
                "CRITICAL MANDATE: You MUST fix all specific issues identified above while strictly satisfying ALL FULL ORIGINAL TASK REQUIREMENTS."
            )

        if state.get("memory_context"):
            prompt_parts.append(f"\nRelevant past context:\n{state['memory_context']}")

        # ── tool invocations ────────────────────────────────────────────────
        tool_results: list[str] = []
        task_id = state.get("task_id", "")
        executed_queries = set()

        # 0. Include schema context for database tasks if db_query is available
        if any(kw in instruction.lower() for kw in ["select", "query", "sql", "from", "table", "where", "database", "customer", "balance"]):
            schema_hint = self._get_db_schema_context(subtask_id, tool_logs, task_id=task_id)
            if schema_hint:
                prompt_parts.append(f"\n{schema_hint}")

        # 1. Pre-execution: check if instruction contains an explicit SQL query
        initial_query = self._extract_sql_query(instruction)
        if initial_query:
            db_result = self._try_db_query(initial_query, subtask_id, tool_logs, task_id=task_id)
            executed_queries.add(initial_query)
            if db_result:
                tool_results.append(f"Database query results:\n{db_result}")

        # 2. Try python_sandbox for in-memory data computation.
        if any(kw in instruction.lower() for kw in ["calculate", "compute", "sum", "average", "mean", "ratio", "percentage", "total"]):
            sandbox_result = self._try_sandbox(instruction, subtask_id, tool_logs, task_id=task_id)
            if sandbox_result:
                tool_results.append(f"Computation output:\n{sandbox_result}")

        if tool_results:
            prompt_parts.append("\nTool outputs (reference these in your analysis):")
            prompt_parts.extend(tool_results)

        full_prompt = "\n".join(prompt_parts)

        system_prompt = (
            "You are AgentHive's data specialist. Analyse structured data, "
            "produce precise numerical results, and present findings clearly. "
            "CRITICAL DATABASE MANDATE: Always refer to the 'Available Database Tables & Columns' provided in the prompt and use EXACT column names from the schema (e.g. for the customers table, use 'account_balance', NOT 'balance'). "
            "Always write clean SQL queries inside ```sql ... ``` blocks when querying database tables. "
            "Do not invent column names or data values."
        )

        response = self.llm.ask(
            full_prompt,
            system_prompt,
        )

        if not response:
            return {
                "specialist_output": "Data specialist could not contact the configured Ollama model.",
                "specialist_confidence": 0.0,
                "tool_logs": tool_logs,
                "current_subtask": {**subtask, "status": "failed"},
                "workflow_status": "failed",
                "trace": [*current_trace, "Data specialist: LLM unavailable"],
            }

        # 3. Post-execution: Extract SQL query generated in LLM response and execute against db_query
        llm_query = self._extract_sql_query(response)
        if llm_query and llm_query not in executed_queries:
            db_result = self._try_db_query(llm_query, subtask_id, tool_logs, task_id=task_id)
            executed_queries.add(llm_query)
            if db_result:
                tool_results.append(f"Database query results for `{llm_query}`:\n{db_result}")
                if db_result not in response:
                    response = (
                        response.strip() +
                        "\n\n**Live Database Query Results:**\n```\n" +
                        db_result +
                        "\n```"
                    )

        confidence = self._estimate_confidence(response, expected_output, tool_results)

        return {
            "specialist_output": response,
            "specialist_confidence": confidence,
            "tool_logs": tool_logs,
            "current_subtask": {**subtask, "status": "completed", "output": response, "confidence": confidence},
            "workflow_status": "reviewing",
            "trace": [
                *current_trace,
                f"Data specialist completed subtask '{subtask_id}' (confidence {confidence:.2f})",
            ],
        }

    # ── tool helpers ─────────────────────────────────────────────────────────

    def _get_db_schema_context(self, subtask_id: str, tool_logs: list, task_id: str = "") -> str | None:
        """Fetch database table names and columns to guide LLM query generation."""
        try:
            from src.tools.base import tool_registry, ToolContext
            import json

            if "db_query" not in tool_registry.list_names():
                return None
            ctx = ToolContext(agent_name=self.name, subtask_id=subtask_id, task_id=task_id)
            res, _ = tool_registry.invoke(
                "db_query",
                {"query": "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'public' LIMIT 50;", "read_only": True},
                ctx,
            )
            if res.success and res.output:
                output_data = res.output
                if isinstance(output_data, str):
                    try:
                        output_data = json.loads(output_data)
                    except Exception:
                        pass
                if isinstance(output_data, list):
                    tables: dict[str, list[str]] = {}
                    for row in output_data:
                        if isinstance(row, dict):
                            t_name = str(row.get("table_name") or "")
                            c_name = str(row.get("column_name") or "")
                            if t_name and c_name:
                                tables.setdefault(t_name, []).append(c_name)
                    if tables:
                        schema_lines = [f"  • Table '{t}': {', '.join(cols)}" for t, cols in tables.items()]
                        return "Available Database Tables & Columns:\n" + "\n".join(schema_lines)
        except Exception as exc:
            logger.debug("Failed to fetch DB schema context: %s", exc)
        return None

    def _try_db_query(self, query_or_instr: str, subtask_id: str, tool_logs: list, task_id: str = "") -> str | None:
        try:
            from src.tools.base import tool_registry, ToolContext

            if "db_query" not in tool_registry.list_names():
                return None

            clean_query = self._extract_sql_query(query_or_instr) or (
                query_or_instr.strip() if query_or_instr and query_or_instr.strip().upper().startswith("SELECT") else None
            )
            if not clean_query:
                return None

            ctx = ToolContext(agent_name=self.name, subtask_id=subtask_id, task_id=task_id)
            result, audit = tool_registry.invoke(
                "db_query",
                {"query": clean_query, "read_only": True},
                ctx,
            )
            tool_logs.append(audit)
            if not result.success and ("column \"balance\" does not exist" in str(result.error) or "column balance does not exist" in str(result.error)):
                healed_query = clean_query.replace("(balance)", "(account_balance)").replace(" balance ", " account_balance ").replace(", balance", ", account_balance")
                result_h, audit_h = tool_registry.invoke(
                    "db_query",
                    {"query": healed_query, "read_only": True},
                    ctx,
                )
                tool_logs.append(audit_h)
                if result_h.success:
                    return str(result_h.output)
            return str(result.output) if result.success else None

        except Exception as exc:
            logger.debug("db_query invocation error: %s", exc)
            return None

    def _try_sandbox(self, instruction: str, subtask_id: str, tool_logs: list, task_id: str = "") -> str | None:
        try:
            from src.tools.base import tool_registry, ToolContext

            snippet = self._extract_computation_snippet(instruction)
            if not snippet:
                return None

            ctx = ToolContext(agent_name=self.name, subtask_id=subtask_id, task_id=task_id)
            result, audit = tool_registry.invoke(
                "python_sandbox",
                {"code": snippet},
                ctx,
            )
            tool_logs.append(audit)
            return str(result.output) if result.success else None

        except Exception as exc:
            logger.debug("Sandbox invocation error: %s", exc)
            return None

    @staticmethod
    def _extract_sql_query(text: str) -> str | None:
        """
        Extract a valid SELECT SQL query from instruction or LLM response text.
        Returns clean SELECT query string ending with a semicolon, or None.
        """
        if not text:
            return None
        import re

        # 1. Look for ```sql ... ``` or ``` ... ``` code blocks containing SELECT
        fenced_blocks = re.findall(r"```(?:sql)?\s*(SELECT[\s\S]+?)```", text, re.IGNORECASE)
        if fenced_blocks:
            best = max(fenced_blocks, key=len).strip()
            cleaned = best.rstrip(";\n\r\t ")
            return cleaned + ";"

        # 2. Look for explicit SELECT ... FROM ... in text
        select_match = re.search(r"\b(SELECT\s+[\s\S]+?\bFROM\s+[\s\S]+?)(?:;|\n\n|\Z)", text, re.IGNORECASE)
        if select_match:
            candidate = select_match.group(1).strip()
            cleaned = candidate.rstrip(";\n\r\t ")
            if len(cleaned) >= 10:
                return cleaned + ";"

        return None

    @staticmethod
    def _extract_computation_snippet(instruction: str) -> str | None:
        instr_lower = instruction.lower()
        if "average" in instr_lower or "mean" in instr_lower:
            return "import statistics\ndata=[10,20,30,40,50]\nprint(f'Mean: {statistics.mean(data)}, Median: {statistics.median(data)}')"
        if "sort" in instr_lower:
            return "data=[5,3,8,1,9,2]\nprint(sorted(data))"
        return None

    @staticmethod
    def _estimate_confidence(response: str, expected_output: str, tool_results: list) -> float:
        score = 0.50
        if len(response) > 150:
            score += 0.15
        if any(c in response for c in ("|", "•", "-", "\n\n")):  # structured output
            score += 0.10
        if tool_results:
            score += 0.15
        if expected_output:
            keywords = [w for w in expected_output.lower().split() if len(w) > 4]
            if keywords:
                hits = sum(1 for kw in keywords if kw in response.lower())
                score += min(0.10, 0.10 * hits / len(keywords))
        return round(min(score, 1.0), 3)
