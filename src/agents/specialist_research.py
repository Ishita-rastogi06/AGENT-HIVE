"""Research specialist for AgentHive."""

from __future__ import annotations

import logging

from src.agents.base import BaseAgent
from src.llm.ollama_client import OllamaClient
from src.orchestration.state import AgentHiveState, SubTask
import src.tools  # ensures all tools are registered before first invocation  # noqa: F401

logger = logging.getLogger(__name__)


def _get_current_subtask(state: AgentHiveState) -> SubTask | None:
    """Return the subtask the specialist should work on right now."""
    subtask = state.get("current_subtask")
    if subtask:
        return subtask
    plan = state.get("task_plan") or {}
    subtasks = plan.get("subtasks") or []
    return subtasks[0] if subtasks else None


class ResearchSpecialist(BaseAgent):
    """Answer research-oriented subtasks using memory and web search."""

    name = "research"

    def __init__(self, llm: OllamaClient) -> None:
        self.llm = llm

    def run(self, state: AgentHiveState) -> dict:
        """Execute the current SubTask from the TaskPlan."""
        subtask = _get_current_subtask(state)
        current_trace = state.get("trace", [])
        tool_logs = list(state.get("tool_logs") or [])
        retry_count = state.get("retry_count", 0)

        if subtask is None:
            return {
                "specialist_output": "No subtask found in plan.",
                "specialist_confidence": 0.0,
                "workflow_status": "failed",
                "trace": [*current_trace, "Research specialist: no subtask available"],
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

        # ── tool invocations ─────────────────────────────────────────────────
        web_results: list[str] = []

        # Attempt a web search for factual / current-info queries.
        if any(kw in instruction.lower() for kw in ["latest", "current", "recent", "today", "search", "find", "look up", "trend", "ev"]):
            web_result = self._try_web_search(instruction, subtask_id, tool_logs, task_id=state.get("task_id", ""))
            if web_result:
                web_results.append(f"Web search results:\n{web_result}")
            else:
                prompt_parts.append(
                    "\n[SEARCH DATA NOTICE]: Web search returned no useful snippets or insufficient search data for this query.\n"
                    "STRICT GROUNDING MANDATE: State explicitly that web search data was insufficient or limited. "
                    "Do NOT invent, fabricate, or guess specific figures, monetary amounts (e.g. ₹ amounts), dates, or acronym expansions not grounded in verified data."
                )

        if web_results:
            prompt_parts.append("\nSearch results (reference these in your answer; do NOT fabricate unmentioned figures or acronyms):")
            prompt_parts.extend(web_results)

        full_prompt = "\n".join(prompt_parts)

        response = self.llm.ask(
            full_prompt,
            (
                "You are AgentHive's research specialist. Give a strictly factual, grounded answer.\n"
                "STRICT GROUNDING RULES:\n"
                "1. Base all specific claims, figures, dates, and acronym expansions strictly on provided search results or recalled memory.\n"
                "2. If search results are empty, thin, or lack specific figures/dates/acronyms, state explicitly that search results were limited or insufficient rather than filling gaps with guessed or fabricated details.\n"
                "3. NEVER fabricate numbers, monetary amounts, dates, or acronym expansions (e.g. do NOT invent acronym expansions like 'Faraday Fellowship' or 'Faradere Incentives'). State uncertainty plainly."
            ),
        ) or (
            "Ollama is not reachable. Start Ollama, pull the configured model, "
            "and try again."
        )

        confidence = self._estimate_confidence(response, expected_output, web_results)

        return {
            "specialist_output": response,
            "specialist_confidence": confidence,
            "tool_logs": tool_logs,
            "current_subtask": {**subtask, "status": "completed", "output": response, "confidence": confidence},
            "workflow_status": "reviewing",
            "trace": [
                *current_trace,
                f"Research specialist completed subtask '{subtask_id}' (confidence {confidence:.2f})",
            ],
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _try_web_search(
        self,
        query: str,
        subtask_id: str,
        tool_logs: list,
        task_id: str = "",
    ) -> str | None:
        """Invoke the web_search tool if registered; degrade gracefully."""
        try:
            from src.tools.base import tool_registry, ToolContext

            if "web_search" not in tool_registry.list_names():
                return None

            ctx = ToolContext(agent_name=self.name, subtask_id=subtask_id, task_id=task_id)
            result, audit = tool_registry.invoke(
                "web_search",
                {"query": query[:300]},
                ctx,
            )
            tool_logs.append(audit)

            if result.success:
                return str(result.output)
            logger.debug("Web search failed: %s", result.error)
            return None

        except Exception as exc:
            logger.debug("Web search invocation error: %s", exc)
            return None

    @staticmethod
    def _estimate_confidence(
        response: str,
        expected_output: str,
        web_results: list[str],
    ) -> float:
        """Heuristic confidence score (0.0–1.0)."""
        score = 0.5

        if len(response) > 200:
            score += 0.15
        # Penalise short or error responses.
        if len(response) < 80:
            score -= 0.20
        if web_results:
            score += 0.10
        if expected_output:
            keywords = [w for w in expected_output.lower().split() if len(w) > 4]
            if keywords:
                hits = sum(1 for kw in keywords if kw in response.lower())
                score += min(0.10, 0.10 * hits / len(keywords))
        # Penalise "Ollama is not reachable" fallback text.
        if "ollama is not reachable" in response.lower():
            score = 0.1

        return round(max(0.0, min(score, 1.0)), 3)
