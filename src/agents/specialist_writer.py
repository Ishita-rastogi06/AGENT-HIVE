"""
Writing specialist for AgentHive.

Handles long-form prose tasks: blog posts, reports, executive summaries,
documentation, and structured narratives.  Uses web_search for factual
grounding and api_call_tool for fetching reference material.  Output is
tuned for readability, structure, and tone — not code generation.
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


class WriterSpecialist(BaseAgent):
    """
    Produce structured long-form prose: reports, blog posts, summaries, and
    documentation.  Grounds factual claims with web search results when the
    task requires current or external information.
    """

    name = "writer"

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
                "trace": [*current_trace, "Writer specialist: no subtask available"],
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
        reference_material: list[str] = []

        task_id = state.get("task_id", "")
        # Attempt web search for current events / factual reference requests.
        if any(kw in instruction.lower() for kw in
               ["latest", "current", "recent", "today", "trends", "statistics",
                "according to", "based on", "cite", "source"]):
            web_result = self._try_web_search(instruction, subtask_id, tool_logs, task_id=task_id)
            if web_result:
                reference_material.append(f"Web search reference:\n{web_result}")

        # API call for structured reference data.
        if any(kw in instruction.lower() for kw in
               ["api", "fetch", "retrieve", "pull from", "endpoint"]):
            api_result = self._try_api_call(instruction, subtask_id, tool_logs, task_id=task_id)
            if api_result:
                reference_material.append(f"API reference data:\n{api_result}")

        if reference_material:
            prompt_parts.append("\nReference material (incorporate where relevant):")
            prompt_parts.extend(reference_material)

        full_prompt = "\n".join(prompt_parts)

        response = self.llm.ask(
            full_prompt,
            (
                "You are AgentHive's writing specialist. Produce well-structured, "
                "engaging long-form prose. Use clear headings, logical flow, and "
                "an appropriate tone for the audience. Cite facts from provided "
                "reference material; do not invent statistics or sources. "
                "Do not write code unless it is an inline illustrative snippet."
            ),
        )

        if not response:
            return {
                "specialist_output": "Writer specialist could not contact the configured Ollama model.",
                "specialist_confidence": 0.0,
                "tool_logs": tool_logs,
                "current_subtask": {**subtask, "status": "failed"},
                "workflow_status": "failed",
                "trace": [*current_trace, "Writer specialist: LLM unavailable"],
            }

        confidence = self._estimate_confidence(response, expected_output, reference_material)

        return {
            "specialist_output": response,
            "specialist_confidence": confidence,
            "tool_logs": tool_logs,
            "current_subtask": {**subtask, "status": "completed", "output": response, "confidence": confidence},
            "workflow_status": "reviewing",
            "trace": [
                *current_trace,
                f"Writer specialist completed subtask '{subtask_id}' (confidence {confidence:.2f})",
            ],
        }

    # ── tool helpers ─────────────────────────────────────────────────────────

    def _try_web_search(self, query: str, subtask_id: str, tool_logs: list, task_id: str = "") -> str | None:
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
            return str(result.output) if result.success else None

        except Exception as exc:
            logger.debug("web_search invocation error: %s", exc)
            return None

    def _try_api_call(self, instruction: str, subtask_id: str, tool_logs: list, task_id: str = "") -> str | None:
        try:
            from src.tools.base import tool_registry, ToolContext

            if "api_call" not in tool_registry.list_names():
                return None

            # Extract a URL hint from the instruction if present.
            import re
            url_match = re.search(r"https?://[^\s\"']+", instruction)
            if not url_match:
                return None

            ctx = ToolContext(agent_name=self.name, subtask_id=subtask_id, task_id=task_id)
            result, audit = tool_registry.invoke(
                "api_call",
                {"url": url_match.group(0), "method": "GET"},
                ctx,
            )
            tool_logs.append(audit)
            return str(result.output) if result.success else None

        except Exception as exc:
            logger.debug("api_call invocation error: %s", exc)
            return None

    @staticmethod
    def _estimate_confidence(response: str, expected_output: str, references: list) -> float:
        score = 0.50
        if len(response) > 300:
            score += 0.15
        if len(response) > 600:
            score += 0.05
        # Structured prose signals: headings, paragraphs.
        if "#" in response or "\n\n" in response:
            score += 0.10
        if references:
            score += 0.10
        if expected_output:
            keywords = [w for w in expected_output.lower().split() if len(w) > 4]
            if keywords:
                hits = sum(1 for kw in keywords if kw in response.lower())
                score += min(0.10, 0.10 * hits / len(keywords))
        return round(min(score, 1.0), 3)
