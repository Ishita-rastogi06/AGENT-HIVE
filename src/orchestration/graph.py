"""
LangGraph wiring for AgentHive's multi-agent workflow.

Graph topology (all four specialists with persistent checkpointer & interrupt approvals)
──────────────────────────────────────────────────────────────────────────────────────
                                               ┌──────────────────────────────────────────┐
                                               │   retry (retry_count < max_retries)      │
                                               ▼                                           │
memory_recall → supervisor → approval_check → [research|coder|data|writer] → reviewer ────┤
                                                                                          │ approve
                                                                                          ▼
                                                                                   finalise → memory_save → END
                                                                                          │ escalate
                                                                                          ▼
                                                                                   finalise → memory_save → END
                                                                                    (needs_human_review=True)
"""

from __future__ import annotations

import logging
from pathlib import Path
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from src.agents.memory_recall import MemoryRecallAgent
from src.agents.memory_save import MemorySaveAgent
from src.agents.reviewer import Reviewer
from src.agents.specialist_coder import CodingSpecialist
from src.agents.specialist_data import DataSpecialist
from src.agents.specialist_research import ResearchSpecialist
from src.agents.specialist_writer import WriterSpecialist
from src.agents.supervisor import Supervisor
from src.config import settings
from src.llm.ollama_client import OllamaClient
from src.orchestration.approval import (
    add_pending_approval,
    is_sensitive_operation,
    log_approval_event,
    update_approval_status,
)
from src.orchestration.state import AgentHiveState, EscalationDetails

logger = logging.getLogger(__name__)

# ── routing constants ─────────────────────────────────────────────────────────
_ROUTE_FINALISE        = "finalise"
_ROUTE_RETRY_RESEARCH  = "retry_research"
_ROUTE_RETRY_CODER     = "retry_coder"
_ROUTE_RETRY_DATA      = "retry_data"
_ROUTE_RETRY_WRITER    = "retry_writer"

_ALL_AGENTS = {"research", "coder", "data", "writer"}

_RETRY_ROUTE: dict[str, str] = {
    "research": _ROUTE_RETRY_RESEARCH,
    "coder":    _ROUTE_RETRY_CODER,
    "data":     _ROUTE_RETRY_DATA,
    "writer":   _ROUTE_RETRY_WRITER,
}


# ── checkpointer helper ───────────────────────────────────────────────────────

def get_checkpointer(db_path: Path | str | None = None) -> SqliteSaver:
    """Return a thread-safe persistent SqliteSaver checkpointer for AgentHive."""
    target_path = Path(db_path) if db_path else settings.sqlite_db_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path), check_same_thread=False)
    return SqliteSaver(conn)


# ── node wrappers & approval check ─────────────────────────────────────────────

def _wrap_specialist(specialist_run, agent_name: str):
    """Stamp workflow_status=executing on entry, then run the specialist."""
    def _node(state: AgentHiveState) -> dict:
        result = specialist_run(state)
        result.setdefault("workflow_status", "executing")
        existing_logs = list(state.get("tool_logs") or [])
        new_logs = list(result.get("tool_logs") or [])
        merged_logs = existing_logs + [l for l in new_logs if l not in existing_logs]
        result["tool_logs"] = merged_logs
        return result
    _node.__name__ = agent_name
    return _node


def _pre_execution_approval_check(state: AgentHiveState) -> dict:
    """
    Check for human review escalation conditions before specialist execution:
      1. Low plan confidence (from supervisor step)
      2. Sensitive operations (pre-execution check on tool/subtask)
      3. Explicit user request (pause_mode == 'step_by_step')

    If a condition triggers and auto_approve_remaining is False, pauses execution via interrupt().
    Resumes when Command(resume=...) is passed by human reviewer.
    """
    if state.get("auto_approve_remaining"):
        return {"workflow_status": "executing"}

    task_id = state.get("task_id") or "default_task"
    user_id = state.get("user_id") or "default"
    db_path = state.get("db_path")
    task = state.get("task", "")
    selected_agent = state.get("selected_agent", "research")
    current_subtask = state.get("current_subtask") or {}

    trigger_type = None
    reason = ""

    # Check 1: Low plan confidence
    if state.get("needs_human_review") and (state.get("escalation") or {}).get("trigger_type") == "low_plan_confidence":
        trigger_type = "low_plan_confidence"
        reason = (state.get("escalation") or {}).get("reason", "Supervisor reported low plan confidence.")

    # Check 2: Sensitive operations (pre-execution check)
    if not trigger_type:
        is_sensitive, sens_reason = is_sensitive_operation(
            tool_name=selected_agent,
            subtask=current_subtask,
        )
        if is_sensitive:
            trigger_type = "sensitive_operation"
            reason = sens_reason

    # Check 3: Explicit user request ("pause before each step")
    if not trigger_type and state.get("pause_mode") == "step_by_step":
        trigger_type = "explicit_user_request"
        reason = "Explicit user request: pause before each step enabled."

    if not trigger_type:
        return {"workflow_status": "executing"}

    # Map trigger to default approval level
    default_level = settings.default_approval_levels.get(trigger_type, "Approve Action")

    proposed_action = f"Execute subtask '{current_subtask.get('title', 'Task execution')}' via {selected_agent}"
    reasoning = current_subtask.get("instruction") or task

    esc_details: EscalationDetails = {
        "required": True,
        "reason": reason,
        "severity": "high" if trigger_type in ("sensitive_operation", "retries_exhausted") else "medium",
        "requested_action": proposed_action,
        "approval_level": default_level,  # type: ignore[typeddict-item]
        "trigger_type": trigger_type,    # type: ignore[typeddict-item]
        "context": {
            "task_id": task_id,
            "subtask_id": current_subtask.get("id", ""),
            "proposed_action": proposed_action,
            "reasoning": reasoning,
        },
    }

    # Persist pending approval to SQLite Approval Queue database
    add_pending_approval(
        task_id=task_id,
        user_id=user_id,
        task=task,
        trigger_reason=reason,
        approval_level=default_level,
        proposed_action=proposed_action,
        reasoning=reasoning,
        recalled_memories=state.get("memory_context", ""),
        plan_so_far=state.get("task_plan") or {},
        db_path=db_path,
    )

    trace = [
        *state.get("trace", []),
        f"PAUSED FOR HUMAN APPROVAL ({trigger_type}): {reason}",
    ]

    # Pause graph execution via LangGraph 1.2.11 interrupt() API
    resume_data = interrupt(esc_details)

    # Execution resumes here when Command(resume=...) is passed!
    action = "approve_action"
    decision = "approved"
    custom_output = None

    if isinstance(resume_data, dict):
        action = resume_data.get("action", "approve_action")
        decision = resume_data.get("decision", "approved")
        custom_output = resume_data.get("custom_output")
    elif isinstance(resume_data, str):
        action = resume_data

    # Log approval event
    log_approval_event(
        task_id=task_id,
        user_id=user_id,
        trigger_type=trigger_type,
        approval_level=action,
        decision=decision,
        user_response=custom_output or action,
        db_path=db_path,
    )
    update_approval_status(
        task_id,
        status="approved",
        decision=decision,
        approval_level=action,
        custom_output=custom_output,
        db_path=db_path,
    )

    updates: dict[str, Any] = {
        "needs_human_review": False,
        "workflow_status": "executing",
        "trace": [*trace, f"RESUMED BY HUMAN ({action}): {decision}"],
    }

    if action == "approve_plan":
        updates["auto_approve_remaining"] = True

    if action == "take_over" and custom_output:
        updates["specialist_output"] = custom_output
        updates["specialist_confidence"] = 1.0

    return updates


# ── routing functions ─────────────────────────────────────────────────────────

def _route_after_approval_check(state: AgentHiveState) -> str:
    """
    Route after pre-execution approval check.

    If 'take_over' was selected and human provided custom output text, route directly
    to reviewer (bypassing specialist execution).
    """
    trace = state.get("trace") or []
    if state.get("specialist_output") and any("take_over" in str(t) for t in trace):
        return "reviewer"
    return state.get("selected_agent", "research")

def _route_after_review(state: AgentHiveState) -> str:
    review        = state.get("review") or {}
    decision      = review.get("decision", "approve")
    retry_count   = int(state.get("retry_count") or 0)
    max_retries   = int(state.get("max_retries") or settings.max_retries)
    selected      = state.get("selected_agent", "research")

    if decision == "approve":
        return _ROUTE_FINALISE

    if decision == "escalate":
        logger.info("Reviewer escalated (retry_count=%d)", retry_count)
        return _ROUTE_FINALISE

    if retry_count < max_retries:
        route = _RETRY_ROUTE.get(selected, _ROUTE_RETRY_RESEARCH)
        logger.info("Reviewer retry %d/%d → %s", retry_count + 1, max_retries, selected)
        return route

    logger.info("Retry budget exhausted (%d/%d) — escalating", retry_count, max_retries)
    return _ROUTE_FINALISE


def _pre_retry_node(agent_name: str):
    def _node(state: AgentHiveState) -> dict:
        retry_count = int(state.get("retry_count") or 0) + 1
        max_retries = int(state.get("max_retries") or settings.max_retries)
        feedback    = (state.get("review") or {}).get("feedback", "")

        trace = [
            *state.get("trace", []),
            f"Retry {retry_count}/{max_retries} — feedback: {feedback[:120]}",
        ]

        if retry_count > max_retries:
            return {
                "retry_count":       retry_count,
                "needs_human_review": True,
                "workflow_status":   "escalated",
                "trace": [*trace, f"Max retries ({max_retries}) reached — escalating."],
            }

        return {
            "retry_count":     retry_count,
            "workflow_status": "retrying",
            "trace":           trace,
        }

    _node.__name__ = f"pre_retry_{agent_name}"
    return _node


def _escalate_state(state: AgentHiveState) -> dict:
    review = state.get("review") or {}
    trigger_type = "reviewer_escalation" if review.get("decision") == "escalate" else "retries_exhausted"
    default_level = settings.default_approval_levels.get(trigger_type, "Take Over")

    return {
        "needs_human_review": True,
        "escalation": {
            "required":         True,
            "reason":           review.get("feedback", "Reviewer requested escalation."),
            "severity":         "high",
            "requested_action": "Human review of specialist output required.",
            "approval_level":   default_level,
            "trigger_type":     trigger_type,
            "context": {
                "retry_count": state.get("retry_count", 0),
                "issues":      review.get("issues", []),
            },
        },
    }


def _route_after_finalise(state: AgentHiveState) -> str:
    """
    Route after finalise node.
    If there are remaining pending subtasks (workflow_status == 'executing'), route to approval_check to run next subtask.
    Otherwise (workflow_status in 'completed', 'escalated', 'failed'), route to memory_save.
    """
    wf_status = state.get("workflow_status")
    if wf_status == "executing":
        return "approval_check"
    return "memory_save"


def _combined_finalise(_workflow_ref):
    def _node(state: AgentHiveState) -> dict:
        review       = state.get("review") or {}
        decision     = review.get("decision", "approve")
        retry_count  = int(state.get("retry_count") or 0)
        max_retries  = int(state.get("max_retries") or settings.max_retries)
        curr_st      = state.get("current_subtask") or {}
        st_id        = curr_st.get("id", "t1")
        st_title     = curr_st.get("title", st_id)

        needs_escalation = (
            decision == "escalate"
            or (decision == "retry" and retry_count >= max_retries)
            or state.get("needs_human_review", False)
        )

        if needs_escalation:
            esc = _escalate_state(state)
            plan = dict(state.get("task_plan") or {})
            subtasks = list(plan.get("subtasks") or [])
            halted_subtasks = [st.get("id") for st in subtasks if st.get("id") != st_id and st.get("status") == "pending"]

            reason_msg = f"Subtask '{st_id}' ({st_title}) escalated: {esc['escalation']['reason']}"
            if halted_subtasks:
                reason_msg += f". Workflow halted — remaining pending subtasks: {', '.join(halted_subtasks)}."

            esc["escalation"]["reason"] = reason_msg
            esc["escalation"]["failed_subtask_id"] = st_id
            esc["escalation"]["halted_subtasks"] = halted_subtasks

            # Unify escalation queue persistence for SQLite Approval Queue
            task_id = state.get("task_id") or "default_task"
            user_id = state.get("user_id") or "default"
            db_path = state.get("db_path")
            task = state.get("task") or ""

            add_pending_approval(
                task_id=task_id,
                user_id=user_id,
                task=task,
                trigger_reason=reason_msg,
                approval_level=esc["escalation"].get("approval_level", "Take Over"),
                proposed_action=esc["escalation"].get("requested_action", f"Review escalated subtask '{st_id}'"),
                reasoning=curr_st.get("instruction") or task,
                recalled_memories=state.get("memory_context", ""),
                plan_so_far=plan,
                db_path=db_path,
            )

            return {
                **esc,
                "failed_subtask_id": st_id,
                "halted_subtasks": halted_subtasks,
                "final_answer": state.get("specialist_output") or state.get("final_answer") or (state.get("review") or {}).get("feedback") or "Escalated",
                "workflow_status": "escalated",
                "trace": [
                    *state.get("trace", []),
                    f"Workflow ESCALATED on Subtask '{st_id}' — {reason_msg}",
                ],
            }

        # Multi-subtask plan progression
        plan = dict(state.get("task_plan") or {})
        subtasks = [dict(st) for st in (plan.get("subtasks") or [])]
        exec_order = list(plan.get("execution_order") or [st.get("id") for st in subtasks])

        # Mark current subtask as completed
        for st_item in subtasks:
            if st_item.get("id") == st_id:
                st_item["status"] = "completed"
                st_item["output"] = state.get("specialist_output", "")

        # Find next pending subtask
        next_subtask = None
        if exec_order and subtasks:
            for next_id in exec_order:
                matching = next((st for st in subtasks if st.get("id") == next_id), None)
                if matching and matching.get("status") != "completed":
                    next_subtask = matching
                    break

        if next_subtask:
            plan["subtasks"] = subtasks
            return {
                "task_plan": plan,
                "current_subtask": next_subtask,
                "selected_agent": next_subtask.get("assigned_agent", state.get("selected_agent", "research")),
                "retry_count": 0,
                "review": None,
                "workflow_status": "executing",
                "trace": [
                    *state.get("trace", []),
                    f"Subtask '{st_id}' completed. Advancing to next subtask '{next_subtask.get('id')}' ({next_subtask.get('title')}).",
                ],
            }

        final_ans = state.get("final_answer") or state.get("specialist_output") or ""
        return {
            "task_plan": plan,
            "final_answer": final_ans,
            "workflow_status": "completed",
            "trace": [*state.get("trace", []), "Workflow completed successfully."],
        }

    return _node


# ── graph builder ─────────────────────────────────────────────────────────────

def build_graph(checkpointer: SqliteSaver | None = None):
    """Build and compile the AgentHive task workflow backed by SQLite checkpointer."""
    llm = OllamaClient()

    specialists = {
        "research": ResearchSpecialist(llm),
        "coder":    CodingSpecialist(llm),
        "data":     DataSpecialist(llm),
        "writer":   WriterSpecialist(llm),
    }
    reviewer = Reviewer(llm)

    workflow = StateGraph(AgentHiveState)

    # ── core nodes ────────────────────────────────────────────────────────────
    workflow.add_node("memory_recall", MemoryRecallAgent().run)
    workflow.add_node("supervisor",    Supervisor(llm).run)
    workflow.add_node("approval_check", _pre_execution_approval_check)
    workflow.add_node("reviewer",      reviewer.run)
    workflow.add_node("finalise",      _combined_finalise(workflow))
    workflow.add_node("memory_save",   MemorySaveAgent().run)

    for name, spec in specialists.items():
        workflow.add_node(name, _wrap_specialist(spec.run, name))

    # ── retry nodes ───────────────────────────────────────────────────────────
    for name in specialists:
        workflow.add_node(f"pre_retry_{name}", _pre_retry_node(name))

    # ── linear edges ──────────────────────────────────────────────────────────
    workflow.add_edge(START, "memory_recall")
    workflow.add_edge("memory_recall", "supervisor")
    workflow.add_edge("supervisor", "approval_check")

    # approval_check → specialist (or directly to reviewer on take_over)
    workflow.add_conditional_edges(
        "approval_check",
        _route_after_approval_check,
        {**{name: name for name in specialists}, "reviewer": "reviewer"},
    )

    # specialist → reviewer
    for name in specialists:
        workflow.add_edge(name, "reviewer")

    # reviewer → route (approve / escalate / retry per-agent)
    workflow.add_conditional_edges(
        "reviewer",
        _route_after_review,
        {
            _ROUTE_FINALISE:       "finalise",
            _ROUTE_RETRY_RESEARCH: "pre_retry_research",
            _ROUTE_RETRY_CODER:    "pre_retry_coder",
            _ROUTE_RETRY_DATA:     "pre_retry_data",
            _ROUTE_RETRY_WRITER:   "pre_retry_writer",
        },
    )

    # pre_retry_<agent> → specialist (loop back)
    for name in specialists:
        workflow.add_edge(f"pre_retry_{name}", name)

    # finalise → memory_save (or loop back to approval_check if more subtasks remain)
    workflow.add_conditional_edges(
        "finalise",
        _route_after_finalise,
        {
            "approval_check": "approval_check",
            "memory_save":    "memory_save",
        },
    )
    workflow.add_edge("memory_save", END)

    effective_checkpointer = checkpointer if checkpointer is not None else get_checkpointer()
    return workflow.compile(checkpointer=effective_checkpointer)

