"""Supervisor agent — plans, decomposes, and routes AgentHive tasks."""

from __future__ import annotations

import json
import logging
import re
import textwrap
from uuid import uuid4

from src.agents.base import BaseAgent
from src.config import settings
from src.llm.ollama_client import OllamaClient
from src.orchestration.state import AgentHiveState, SubTask, TaskPlan

logger = logging.getLogger(__name__)


# ── routing keyword lists ────────────────────────────────────────────────────

_CODING_SIGNALS = [
    "write code", "write a python", "write a java", "write a script", "python script", "java script",
    "script using", "create a function", "create an app", "build an app", "debug", "error", "exception",
    "fix this code", "implement", "program", "algorithm", "refactor", "palindrome", "fibonacci",
    "unit test", "class definition", "write python", "python code", "java code", "c++ code", "cpp code",
    "js code", "javascript code", "typescript code", "code to print", "code to calculate", "code to check",
    "code to find", "sqlite3 and pandas", "java program", "python program", "c++ program",
]

_STRONG_CODING_SIGNALS = [
    "write a python script", "python script", "write a script", "write code",
    "create a script", "write a python", "write python", "python program",
    "script using", "write a function", "write code for", "write a python script using",
    "write a java", "write java", "java code", "java program", "write a java code",
    "write c++", "write cpp", "c++ code", "cpp code", "write javascript", "js code",
    "write typescript", "ts code", "write a program", "write a code", "code to print",
    "code to check", "code to calculate", "code to find", "code to implement",
]

_RESEARCH_SIGNALS = [
    "explain", "what is", "what are", "why", "compare",
    "difference between", "advantages", "limitations",
    "plan to learn", "research", "summarize", "analyse", "analyze",
    "overview", "history of", "how does",
]

_DATA_SIGNALS = [
    "query", "sql", "database", "table", "schema", "column",
    "insert", "insert into", "update", "delete from", "drop table",
    "aggregate", "count rows", "sum of", "average of", "mean of",
    "pivot", "group by", "join table", "data analysis", "analyse data",
    "analyze data", "filter data", "sort data", "dataset",
    "csv", "dataframe", "statistics on", "distribution of",
    "calculate", "compute", "average revenue", "total sales",
]

_WRITER_SIGNALS = [
    "blog post", "blog", "article", "essay", "newsletter", "press release",
    "editorial", "speech", "story", "narrative", "copywriting",
    "email draft", "executive summary", "long-form", "documentation",
    "proposal", "report", "whitepaper",
    "write a blog", "write an article", "write a report", "write an essay",
    "write a story", "write a summary", "write a proposal", "write a newsletter",
    "write an email", "draft a", "draft an", "compose a", "compose an",
    "create a report", "create an article", "create a blog", "create a post",
    "summary for", "writing-focused", "write about", "drafting",
    "engaging introduction", "clear conclusion", "introduction and conclusion",
    "word count", "500-word", "300-word", "1000-word", "word blog", "word article",
    "word report", "tone", "for an audience", "target audience", "structure",
]

# Ordered by specificity — explicit code instructions checked first.
_SIGNAL_PRIORITY: list[tuple[str, list[str]]] = [
    ("coder",    _CODING_SIGNALS),
    ("data",     _DATA_SIGNALS),
    ("writer",   _WRITER_SIGNALS),
    ("research", _RESEARCH_SIGNALS),
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _pick_agent(task: str, llm: OllamaClient) -> str:
    """
    Return the best-matching specialist name for the task.

    Directly routes explicit code creation requests to 'coder'.
    Scores all four agent keyword lists and returns the agent with the
    highest score.
    """
    normalized = task.lower()

    if any(sig in normalized for sig in _STRONG_CODING_SIGNALS):
        return "coder"

    scores: dict[str, int] = {
        agent: sum(1 for s in signals if s in normalized)
        for agent, signals in _SIGNAL_PRIORITY
    }

    best_score = max(scores.values())
    if best_score > 0:
        for agent, _ in _SIGNAL_PRIORITY:
            if scores[agent] == best_score:
                return agent

    # No keyword hit — ask the LLM.
    decision = llm.ask(
        (
            f"User request:\n{task}\n\n"
            "Which specialist should handle this? Choose exactly one: "
            "research, coder, data, or writer."
        ),
        (
            "You are AgentHive's routing supervisor. "
            "Reply with exactly one word: research, coder, data, or writer."
        ),
    )
    normalized_decision = (decision or "").strip().lower()
    if normalized_decision in ("research", "coder", "data", "writer"):
        return normalized_decision
    return "research"  # safe default


def _extract_json_block(text: str) -> str | None:
    """Pull the first JSON object or array out of an LLM response."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    bare = re.search(r"(\{.*\})", text, re.DOTALL)
    if bare:
        return bare.group(1)
    return None


# ── supervisor ───────────────────────────────────────────────────────────────

class Supervisor(BaseAgent):
    """
    Analyse a request, decompose it into 1-3 ordered SubTasks, and return a
    full TaskPlan.  Routes to one of four specialists: research, coder, data,
    writer.  For multi-part requests, produces ≥2 subtasks with explicit
    dependency links between them.
    """

    name = "supervisor"

    def __init__(self, llm: OllamaClient) -> None:
        self.llm = llm

    # ── planning ─────────────────────────────────────────────────────────────

    def plan(self, task: str = "", primary_agent: str = "coder", memory_context: str = "") -> TaskPlan:
        """Wrapper method for external calls and unit test mocks."""
        return self._create_plan(task, primary_agent, memory_context)

    def _create_plan(self, task: str, primary_agent: str, memory_context: str = "") -> TaskPlan:
        """
        Build a TaskPlan by asking the LLM to decompose the task.

        Falls back to a safe single-subtask plan if the LLM response cannot
        be parsed so the workflow never hard-crashes here.
        """
        context_block = f"{memory_context.strip()}\n\n" if memory_context and memory_context.strip() else ""

        decompose_prompt = (
            f"User request: {task}\n\n"
            f"{context_block}"
            "Decompose this request cleanly:\n"
            "- For single conceptual questions, writing requests (e.g. 'Write a 500-word blog post'), explanations, or single-domain tasks, keep it as EXACTLY 1 subtask assigned to the appropriate specialist.\n"
            "- ONLY produce 2-3 subtasks if the request explicitly requires distinct sequential steps or multi-agent handoffs.\n\n"
            "Each subtask must be a JSON object with these fields:\n"
            '  "id": unique short string (e.g. "t1"),\n'
            '  "title": one short sentence,\n'
            '  "description": what needs to be done,\n'
            f'  "assigned_agent": one of "research", "coder", "data", "writer",\n'
            '  "dependencies": list of ids that must finish before this one ([] for first),\n'
            '  "instruction": specific instruction for the specialist,\n'
            '  "expected_output": what a correct, complete result looks like,\n'
            '  "status": "pending"\n\n'
            "Return ONLY a JSON object like:\n"
            '{"objective": "...", "reasoning": "...", "subtasks": [...], '
            '"execution_order": ["t1"]}'
        )

        system_prompt = (
            "You are AgentHive's planning supervisor. "
            "Available specialists: research (factual Q&A/explanations), coder (code generation), "
            "data (structured data analysis and SQL queries), "
            "writer (long-form prose, blog posts, articles, and reports).\n"
            "DECOMPOSITION RULES:\n"
            "1. SINGLE SUBTASK: Single writing tasks (blog posts, articles, reports), coding tasks, explanations, or single-pass requests MUST stay as 1 single subtask.\n"
            "2. MULTI-SUBTASK: Create 2-3 subtasks ONLY for requests with explicit sequential dependencies or multi-agent handoffs.\n"
            "Return valid JSON only — no extra text."
        )

        try:
            raw = self.llm.ask(decompose_prompt, system_prompt, num_predict=600)
        except TypeError:
            raw = self.llm.ask(decompose_prompt, system_prompt)

        if raw:
            json_str = _extract_json_block(raw) or raw.strip()
            try:
                data = json.loads(json_str)
                return self._validate_and_build_plan(data, task, primary_agent)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Supervisor plan parse failed (%s) — using fallback", exc)

        return self._fallback_plan(task, primary_agent)

    def _validate_and_build_plan(
        self,
        data: dict,
        task: str,
        primary_agent: str,
    ) -> TaskPlan:
        """
        Validate a parsed LLM response and normalise it into a TaskPlan.

        Any subtask missing required keys gets sensible defaults so the rest
        of the workflow can proceed.  Unknown assigned_agent values are
        clamped to a valid agent.
        """
        raw_subtasks: list[dict] = data.get("subtasks", [])

        if not raw_subtasks:
            return self._fallback_plan(task, primary_agent)

        _valid_agents = {"research", "coder", "data", "writer"}

        # Heuristic: Discourage decomposition into multiple subtasks for single, answerable-in-one-pass conceptual questions.
        # Check for explicit sequential dependency signals ("then", "after", "step 1", etc.)
        sequential_patterns = [
            r"\bthen\b", r"\band then\b", r"\bafter that\b", r"\bfollowed by\b",
            r"\bfirst\b.*\bsecond\b", r"\bstep 1\b", r"\bphase 1\b", r"\bstage 1\b"
        ]
        task_text = f"{task} {data.get('objective', '')}"
        has_sequential_dependency = any(re.search(pat, task_text, re.IGNORECASE) for pat in sequential_patterns)

        # Check for multi-agent handoffs across different specialist domains
        unique_agents = {raw.get("assigned_agent") for raw in raw_subtasks if raw.get("assigned_agent") in _valid_agents}
        is_multi_agent = len(unique_agents) > 1

        # If LLM returned multiple subtasks without explicit sequential dependency keywords in task_text,
        # collapse into a single subtask assigned to primary_agent to prevent unnecessary multi-agent splits.
        if len(raw_subtasks) > 1 and not has_sequential_dependency:
            logger.info("Collapsing multi-subtask plan without explicit sequential keywords into 1 single subtask: '%s'", task)
            return self._fallback_plan(task, primary_agent)

        # Deduplicate identical subtask instructions to avoid duplicate execution loops (59s -> 31s)
        seen_instructions = set()
        deduped_subtasks = []
        for raw in raw_subtasks:
            instr = str(raw.get("instruction") or task).strip()
            if instr not in seen_instructions:
                seen_instructions.add(instr)
                deduped_subtasks.append(raw)
        raw_subtasks = deduped_subtasks

        subtasks: list[SubTask] = []
        for raw in raw_subtasks:
            assigned = raw.get("assigned_agent", primary_agent)
            if len(raw_subtasks) == 1 or primary_agent == "coder" or assigned not in _valid_agents:
                assigned = primary_agent

            raw_title = str(raw.get("title") or "Untitled subtask").strip()
            subtask_title = textwrap.shorten(raw_title, width=65, placeholder="...") if len(raw_title) > 65 else raw_title

            st: SubTask = {
                "id": str(raw.get("id") or uuid4().hex[:6]),
                "title": subtask_title,
                "description": str(raw.get("description") or task),
                "assigned_agent": assigned,
                "dependencies": list(raw.get("dependencies") or []),
                "instruction": task if primary_agent == "coder" else str(raw.get("instruction") or task),
                "expected_output": str(
                    raw.get("expected_output") or "A correct, complete response."
                ),
                "status": "pending",
                "retry_count": 0,
            }
            subtasks.append(st)

        ids = [st["id"] for st in subtasks]
        execution_order: list[str] = data.get("execution_order") or ids
        execution_order = [eid for eid in execution_order if eid in ids]
        if not execution_order:
            execution_order = ids

        return TaskPlan(
            objective=str(data.get("objective") or task),
            reasoning=str(data.get("reasoning") or ""),
            subtasks=subtasks,
            execution_order=execution_order,
            status="pending",
        )

    def _fallback_plan(self, task: str, primary_agent: str) -> TaskPlan:
        """Return a single-subtask plan when decomposition fails."""
        subtask_id = "t1"
        first_line = task.strip().split("\n")[0]
        clean_task_heading = textwrap.shorten(first_line, width=65, placeholder="...") if len(first_line) > 65 else first_line
        task_title = clean_task_heading if len(clean_task_heading) > 10 else f"Execute task: {clean_task_heading}"
        subtask: SubTask = {
            "id": subtask_id,
            "title": task_title,
            "description": task,
            "assigned_agent": primary_agent,
            "dependencies": [],
            "instruction": task,
            "expected_output": "A correct, complete response that fully addresses the request.",
            "status": "pending",
            "retry_count": 0,
        }
        return TaskPlan(
            objective=task,
            reasoning="Single-step task — direct execution.",
            subtasks=[subtask],
            execution_order=[subtask_id],
            status="pending",
        )

    # ── LangGraph node ───────────────────────────────────────────────────────

    def run(self, state: AgentHiveState) -> dict:
        """Create the TaskPlan and set up execution state."""
        task = state["task"].strip()
        user_id = state.get("user_id") or "default"
        task_id = state.get("task_id") or str(uuid4())
        memory_context = state.get("memory_context") or ""

        primary_agent = _pick_agent(task, self.llm)
        task_plan = self.plan(task, primary_agent, memory_context=memory_context)

        # Store initial task plan in short-term Redis working memory
        try:
            from src.memory.short_term import ShortTermStore
            stm = ShortTermStore()
            stm.create(
                user_id=user_id,
                task=task,
                task_id=task_id,
                task_type=primary_agent,
                plan_objective=task_plan.get("objective", task),
                plan_reasoning=task_plan.get("reasoning", ""),
            )
        except Exception:
            pass

        execution_order: list[str] = task_plan.get("execution_order", [])
        first_id = execution_order[0] if execution_order else ""

        subtasks: list[SubTask] = task_plan.get("subtasks", [])
        first_subtask = next(
            (st for st in subtasks if st["id"] == first_id),
            subtasks[0] if subtasks else None,
        )

        pending_ids = execution_order[1:] if len(execution_order) > 1 else []

        trace = [
            *state.get("trace", []),
            f"Supervisor created plan: {len(subtasks)} subtask(s)",
            f"Execution order: {' → '.join(execution_order)}",
            f"First subtask: '{first_subtask['title']}'" if first_subtask else "No subtasks",
            f"Supervisor → {first_subtask['assigned_agent'] if first_subtask else primary_agent}",
        ]

        # Compute plan confidence score (valid single-step and multi-step plans get high confidence)
        plan_confidence = 0.90
        reasoning = task_plan.get("reasoning", "")
        if "using fallback" in reasoning.lower():
            plan_confidence = 0.80

        low_threshold = state.get("low_confidence_threshold") or settings.low_confidence_threshold
        needs_review = plan_confidence < low_threshold

        esc_payload = {}
        if needs_review:
            esc_payload = {
                "needs_human_review": True,
                "escalation": {
                    "required": True,
                    "reason": f"Plan confidence ({plan_confidence:.2f}) below threshold ({low_threshold:.2f}).",
                    "severity": "medium",
                    "requested_action": "Review supervisor proposed task decomposition",
                    "approval_level": "Approve Plan",
                    "trigger_type": "low_plan_confidence",
                    "context": {
                        "task_id": task_id,
                        "plan_confidence": plan_confidence,
                        "low_threshold": low_threshold,
                    },
                },
            }

        return {
            "user_id": user_id,
            "task_id": task_id,
            "selected_agent": first_subtask["assigned_agent"] if first_subtask else primary_agent,
            "task_plan": task_plan,
            "current_subtask_id": first_id,
            "current_subtask": first_subtask,
            "pending_subtask_ids": pending_ids,
            "completed_subtask_ids": [],
            "retry_count": 0,
            "max_retries": state.get("max_retries", 2),
            "low_confidence_threshold": low_threshold,
            "tool_logs": [],
            "workflow_status": "planning",
            "trace": trace,
            **esc_payload,
        }
