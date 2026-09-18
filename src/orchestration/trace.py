"""
Unified Execution Trace, Diff Engine, and Replay System for AgentHive.

Capabilities:
1. Builds a unified execution trace tree for a task by combining:
   - LangGraph checkpoint history (via graph.get_state_history())
   - Tool-call audit logs (data/tool_calls.jsonl)
   - Human approval audit logs (SQLite approval_audit_log table & approval_events.jsonl)
2. State snapshot diffing (diff_state_snapshots) between consecutive checkpoints.
3. Trace diffing (compare_traces) between an original task run and a replay task run.
4. Full replay and partial replay (resuming with step output overrides).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

from langgraph.types import Command

from src.config import settings

logger = logging.getLogger(__name__)


def _parse_iso_ts(ts_str: str | None) -> datetime | None:
    if not ts_str:
        return None
    try:
        # Handle ISO strings with Z or timezone offsets
        clean_str = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(clean_str)
    except Exception:
        return None


def diff_state_snapshots(
    prev_values: dict[str, Any] | None,
    curr_values: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Perform a dictionary diff between two consecutive state snapshots.
    Returns a dict containing newly added or modified state keys.
    """
    if not curr_values:
        return {}
    if not prev_values:
        return dict(curr_values)

    diff: dict[str, Any] = {}
    for key, val in curr_values.items():
        # Exclude internal metadata keys from plain diff
        if key in ("__interrupt__",):
            continue
        if key not in prev_values or prev_values[key] != val:
            diff[key] = val
    return diff


def build_execution_trace(
    task_id: str,
    user_id: str | None = None,
    db_path: Path | str | None = None,
    data_dir: Path | str | None = None,
    tool_log_file: Path | str | None = None,
) -> dict[str, Any]:
    """
    Build a unified execution trace tree for task_id by joining:
      - LangGraph checkpoint state history
      - Tool call audit logs (data/tool_calls.jsonl)
      - Approval audit logs (SQLite approval_audit_log & approval_events.jsonl)

    Returns a structured dictionary containing nodes, metrics, and metadata.
    """
    from src.orchestration.graph import build_graph, get_checkpointer

    effective_db_path = Path(db_path) if db_path else settings.sqlite_db_path
    effective_data_dir = Path(data_dir) if data_dir else settings.data_dir
    effective_tool_log = Path(tool_log_file) if tool_log_file else settings.tool_log_file

    checkpointer = get_checkpointer(db_path=effective_db_path)
    graph = build_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": task_id}}

    # 1. Read LangGraph checkpoint history (newest to oldest) and reverse to chronological order
    try:
        raw_history = list(graph.get_state_history(config))
    except Exception as exc:
        logger.warning("Could not read state history for task_id %s: %s", task_id, exc)
        raw_history = []

    history = list(reversed(raw_history))

    # 2. Read tool call audit entries for task_id (STRICT match by task_id)
    tool_events: list[dict[str, Any]] = []
    if effective_tool_log.exists() and task_id:
        try:
            with effective_tool_log.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        if rec.get("task_id") and rec.get("task_id") == task_id:
                            tool_events.append(rec)
        except Exception as exc:
            logger.warning("Failed to read tool log file: %s", exc)

    # 3. Read approval audit log entries for task_id
    approval_events: list[dict[str, Any]] = []
    try:
        conn = sqlite3.connect(str(effective_db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM approval_audit_log WHERE task_id=? ORDER BY timestamp ASC;",
            (task_id,),
        )
        approval_events = [dict(row) for row in cursor.fetchall()]
        conn.close()
    except Exception as exc:
        logger.warning("Failed to read SQLite approval audit log: %s", exc)

    # Fallback to jsonl if SQLite entries not found
    if not approval_events:
        app_jsonl = effective_data_dir / "approval_events.jsonl"
        if app_jsonl.exists():
            try:
                with app_jsonl.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            rec = json.loads(line)
                            if rec.get("task_id") == task_id:
                                approval_events.append(rec)
            except Exception:
                pass

    # 4. Construct trace nodes from checkpoints
    nodes: list[dict[str, Any]] = []
    prev_values: dict[str, Any] | None = None
    prev_ts: datetime | None = None
    total_duration = 0.0

    for idx, snapshot in enumerate(history):
        curr_values = snapshot.values or {}
        tasks = getattr(snapshot, "tasks", ())
        step_name = tasks[0].name if tasks else "checkpoint"
        if step_name == "__start__":
            step_name = "start"

        ckpt_id = (snapshot.config or {}).get("configurable", {}).get("checkpoint_id", f"ckpt_{idx}")
        ts_str = getattr(snapshot, "created_at", None)
        curr_ts = _parse_iso_ts(ts_str)

        duration = 0.0
        next_snapshot = history[idx + 1] if idx + 1 < len(history) else None
        next_ts = _parse_iso_ts(getattr(next_snapshot, "created_at", None)) if next_snapshot else None
        if curr_ts and next_ts:
            delta = (next_ts - curr_ts).total_seconds()
            duration = max(0.0, delta)
        elif prev_ts and curr_ts:
            delta = (curr_ts - prev_ts).total_seconds()
            duration = max(0.0, delta)

        children: list[dict[str, Any]] = []

        # Attach matching tool calls strictly to relevant specialist step
        if step_name in ("research", "coder", "data", "writer", "executing"):
            matched_events = []
            remaining_events = []
            curr_subtask_id = (curr_values.get("current_subtask") or {}).get("id", "")

            for t_evt in tool_events:
                evt_agent = (t_evt.get("agent_name") or "").strip().lower()
                evt_subtask = (t_evt.get("subtask_id") or "").strip()

                if evt_agent == step_name or (curr_subtask_id and evt_subtask == curr_subtask_id) or not evt_agent:
                    matched_events.append(t_evt)
                else:
                    remaining_events.append(t_evt)

            for t_evt in matched_events:
                dur_sec = t_evt.get("duration_seconds")
                if dur_sec is None and t_evt.get("duration_ms") is not None:
                    dur_sec = round(float(t_evt["duration_ms"]) / 1000.0, 3)

                t_child = {
                    "node_id": f"tool_{t_evt.get('timestamp', idx)}",
                    "step_name": f"tool:{t_evt.get('tool_name', 'tool')}",
                    "node_type": "tool_call",
                    "timestamp": t_evt.get("timestamp"),
                    "duration_seconds": dur_sec or 0.0,
                    "details": {
                        "arguments": t_evt.get("arguments"),
                        "result_summary": t_evt.get("result_summary") or t_evt.get("result_preview"),
                        "success": t_evt.get("success"),
                        "error": t_evt.get("error"),
                    },
                }
                children.append(t_child)
            tool_events = remaining_events

        # Attach matching approval events to approval_check steps
        if step_name in ("approval_check", "supervisor", "finalise"):
            for a_evt in approval_events:
                a_child = {
                    "node_id": f"approval_{a_evt.get('event_id', idx)}",
                    "step_name": f"approval:{a_evt.get('approval_level', 'action')}",
                    "node_type": "approval_event",
                    "timestamp": a_evt.get("timestamp"),
                    "duration_seconds": 0.0,
                    "details": {
                        "trigger_type": a_evt.get("trigger_type"),
                        "approval_level": a_evt.get("approval_level"),
                        "decision": a_evt.get("decision"),
                        "user_response": a_evt.get("user_response"),
                    },
                }
                children.append(a_child)
            approval_events = []

        # Ensure parent node duration reflects tool children duration
        child_duration_sum = sum(float(c.get("duration_seconds") or 0.0) for c in children)
        if child_duration_sum > duration:
            duration = child_duration_sum

        total_duration += duration

        diff = diff_state_snapshots(prev_values, curr_values)

        node_details: dict[str, Any] = {}
        if step_name == "supervisor":
            next_snap_values = history[idx + 1].values if (idx + 1 < len(history) and history[idx + 1].values) else {}
            plan_obj = curr_values.get("task_plan") or next_snap_values.get("task_plan") or curr_values.get("plan")
            if hasattr(plan_obj, "dict"):
                plan_obj = plan_obj.dict()
            elif hasattr(plan_obj, "__dict__"):
                plan_obj = dict(plan_obj.__dict__)
            node_details["plan"] = plan_obj if isinstance(plan_obj, dict) else (plan_obj or {})
            node_details["memory_context"] = curr_values.get("memory_context") or next_snap_values.get("memory_context")
        elif step_name in ("research", "coder", "data", "writer"):
            next_snap_values = history[idx + 1].values if (idx + 1 < len(history) and history[idx + 1].values) else {}
            node_details["specialist_output"] = curr_values.get("specialist_output") or next_snap_values.get("specialist_output")
            node_details["subtask"] = next_snap_values.get("current_subtask") or curr_values.get("current_subtask")
        elif step_name == "reviewer":
            next_snap_values = history[idx + 1].values if (idx + 1 < len(history) and history[idx + 1].values) else {}
            rev_obj = next_snap_values.get("review") or curr_values.get("review")
            if hasattr(rev_obj, "dict"):
                rev_obj = rev_obj.dict()
            elif hasattr(rev_obj, "__dict__"):
                rev_obj = dict(rev_obj.__dict__)
            node_details["review"] = rev_obj if isinstance(rev_obj, dict) else (rev_obj or {})

        trace_node = {
            "node_id": ckpt_id,
            "step_index": idx,
            "step_name": step_name,
            "node_type": "checkpoint",
            "timestamp": ts_str,
            "duration_seconds": round(duration, 3),
            "state_snapshot": curr_values,
            "state_diff": diff,
            "details": node_details,
            "children": children,
        }

        nodes.append(trace_node)
        prev_values = curr_values
        if curr_ts:
            prev_ts = curr_ts

    # Overall task summary metadata
    latest_values = history[-1].values if history else {}
    selected_agent = (latest_values or {}).get("selected_agent")
    task_text = (latest_values or {}).get("task", "")

    # Fallback to local task history if checkpoints are not available
    if not selected_agent or not history:
        try:
            hist_file = effective_data_dir / "task_history.json"
            if hist_file.exists():
                with hist_file.open("r", encoding="utf-8") as f:
                    hist_data = json.load(f)
                    for item in hist_data:
                        if item.get("task_id") == task_id or item.get("task", "")[:40] in task_text[:40]:
                            selected_agent = item.get("selected_agent", "coder")
                            task_text = task_text or item.get("task", "")
                            if total_duration == 0.0:
                                total_duration = float(item.get("runtime_seconds", 0.0))
                            break
        except Exception:
            pass

    selected_agent = selected_agent or "coder"

    return {
        "task_id": task_id,
        "user_id": user_id or (latest_values or {}).get("user_id", "default"),
        "task": task_text,
        "selected_agent": selected_agent,
        "total_duration_seconds": round(total_duration, 2),
        "total_steps": len(nodes),
        "nodes": nodes,
    }


# ── Trace Diff Engine ─────────────────────────────────────────────────────────

def compare_traces(
    original_trace: dict[str, Any],
    replay_trace: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare two execution traces step-by-step.
    Identifies matching steps, state diffs, and the exact divergence point.
    """
    orig_nodes = original_trace.get("nodes", [])
    replay_nodes = replay_trace.get("nodes", [])

    comparisons: list[dict[str, Any]] = []
    divergence_point = None

    max_len = max(len(orig_nodes), len(replay_nodes))

    for i in range(max_len):
        o_node = orig_nodes[i] if i < len(orig_nodes) else None
        r_node = replay_nodes[i] if i < len(replay_nodes) else None

        matched = True
        reason = "Matched"

        if not o_node:
            matched = False
            reason = "Replay introduced extra step"
        elif not r_node:
            matched = False
            reason = "Original contained extra step"
        elif o_node["step_name"] != r_node["step_name"]:
            matched = False
            reason = f"Graph routing sequence diverged at step: '{o_node['step_name']}' vs '{r_node['step_name']}'"
        else:
            # Compare key domain state outputs (ignoring volatile task metadata keys)
            o_diff = o_node.get("state_diff", {})
            r_diff = r_node.get("state_diff", {})
            meta_keys = {"task_id", "db_path", "pause_mode", "trace", "user_id", "created_at", "timestamp"}
            o_domain = {k: v for k, v in o_diff.items() if k not in meta_keys}
            r_domain = {k: v for k, v in r_diff.items() if k not in meta_keys}
            if o_domain != r_domain:
                matched = False
                changed_keys = [k for k in set(o_domain.keys()) | set(r_domain.keys()) if o_domain.get(k) != r_domain.get(k)]
                reason = f"State output diverged on keys: {', '.join(changed_keys)}"

        if not matched and divergence_point is None:
            divergence_point = {
                "step_index": i,
                "step_name": (r_node or o_node or {}).get("step_name", "unknown"),
                "reason": reason,
            }

        comparisons.append({
            "step_index": i,
            "original_step": o_node["step_name"] if o_node else None,
            "replay_step": r_node["step_name"] if r_node else None,
            "matched": matched,
            "divergence_reason": reason,
            "original_node": o_node,
            "replay_node": r_node,
        })

    return {
        "original_task_id": original_trace.get("task_id"),
        "replay_task_id": replay_trace.get("task_id"),
        "total_steps_original": len(orig_nodes),
        "total_steps_replay": len(replay_nodes),
        "diverged": divergence_point is not None,
        "divergence_point": divergence_point,
        "comparisons": comparisons,
    }


# ── Replay System Functions ───────────────────────────────────────────────────

def full_replay_task(
    original_task_id: str,
    user_id: str = "default",
    db_path: Path | str | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Re-run an original task from scratch with the same input objective under a new task_id.
    Returns (new_task_id, execution_result).
    """
    from src.orchestration.graph import build_graph, get_checkpointer

    trace = build_execution_trace(original_task_id, db_path=db_path)
    original_task = trace.get("task") or "Replay task"

    new_task_id = f"replay_full_{str(uuid4())[:8]}"
    checkpointer = get_checkpointer(db_path=db_path)
    graph = build_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": new_task_id}}
    input_state = {
        "task": original_task,
        "user_id": user_id,
        "task_id": new_task_id,
        "db_path": str(db_path) if db_path else "",
        "trace": [f"Full replay of original task {original_task_id[:8]}"],
    }

    result = graph.invoke(input_state, config=config)
    return new_task_id, result


def partial_replay_task(
    original_task_id: str,
    checkpoint_id: str,
    override_updates: dict[str, Any],
    user_id: str = "default",
    db_path: Path | str | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Partial replay: load checkpoint state snapshot, apply override_updates (e.g. modified specialist output),
    and resume graph execution downstream from target_step_name under a new replay_task_id.
    """
    from src.orchestration.graph import build_graph, get_checkpointer

    checkpointer = get_checkpointer(db_path=db_path)
    graph = build_graph(checkpointer=checkpointer)

    orig_config = {"configurable": {"thread_id": original_task_id}}
    history = list(graph.get_state_history(orig_config))

    if not history:
        return full_replay_task(original_task_id, user_id=user_id, db_path=db_path)

    # Re-order history chronologically (start -> latest)
    chronological = list(reversed(history))

    target_idx = len(chronological) - 1
    for i, snap in enumerate(chronological):
        cid = (snap.config or {}).get("configurable", {}).get("checkpoint_id")
        if cid == checkpoint_id:
            target_idx = i
            break

    target_snapshot = chronological[target_idx]
    new_task_id = f"replay_part_{str(uuid4())[:8]}"
    effective_db = Path(db_path) if db_path else settings.sqlite_db_path

    # 1. Try official checkpointer.copy_thread() API first; fall back to raw SQLite checkpoint copying if NotImplementedError
    try:
        if hasattr(checkpointer, "copy_thread"):
            checkpointer.copy_thread(original_task_id, new_task_id)
        else:
            raise NotImplementedError
    except NotImplementedError:
        logger.info("Partial replay: using raw SQLite checkpoint duplication (copy_thread not implemented by this checkpointer)")
        # NOTE: SqliteSaver in current LangGraph versions raises NotImplementedError for copy_thread().
        # We manually duplicate checkpoints and writes up to target_idx to preserve DeltaChannel parent chains.
        try:
            conn = sqlite3.connect(str(effective_db))
            cur = conn.cursor()
            copied_ckpt_ids = [
                (snap.config or {}).get("configurable", {}).get("checkpoint_id")
                for snap in chronological[:target_idx + 1]
                if (snap.config or {}).get("configurable", {}).get("checkpoint_id")
            ]
            for cid in copied_ckpt_ids:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata)
                    SELECT ?, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
                    FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_id = ?
                    """,
                    (new_task_id, original_task_id, cid),
                )
                cur.execute(
                    """
                    INSERT OR IGNORE INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value)
                    SELECT ?, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value
                    FROM writes
                    WHERE thread_id = ? AND checkpoint_id = ?
                    """,
                    (new_task_id, original_task_id, cid),
                )
            conn.commit()
            conn.close()
        except Exception as copy_err:
            logger.warning("Partial replay: could not copy prior checkpoints: %s", copy_err)

    meta = target_snapshot.metadata or {}
    target_step_name = meta.get("langgraph_node") or meta.get("step_name")
    if not target_step_name and getattr(target_snapshot, "tasks", None):
        target_step_name = target_snapshot.tasks[0].name

    target_values = dict(target_snapshot.values) if target_snapshot and target_snapshot.values else {}
    target_ckpt_id = (target_snapshot.config or {}).get("configurable", {}).get("checkpoint_id")
    target_ckpt_ns = (target_snapshot.config or {}).get("configurable", {}).get("checkpoint_ns", "")

    replay_state = {
        **target_values,
        "task_id": new_task_id,
        "user_id": user_id,
        "db_path": str(db_path) if db_path else "",
        "workflow_status": "executing",
        **override_updates,
        "trace": [
            *(target_values.get("trace") or []),
            f"PARTIAL REPLAY from step {checkpoint_id[:8] if checkpoint_id else 'checkpoint'} with updates: {list(override_updates.keys())}",
        ],
    }

    new_ckpt_config = {
        "configurable": {
            "thread_id": new_task_id,
            "checkpoint_ns": target_ckpt_ns,
            "checkpoint_id": target_ckpt_id,
        }
    }

    try:
        if target_step_name:
            logger.info("Partial replay: updating state as_node='%s' on checkpoint '%s'", target_step_name, target_ckpt_id)
            graph.update_state(new_ckpt_config, replay_state, as_node=target_step_name)
        else:
            logger.info("Partial replay: updating state on checkpoint '%s'", target_ckpt_id)
            graph.update_state(new_ckpt_config, replay_state)
        result = graph.invoke(None, config={"configurable": {"thread_id": new_task_id}})
        logger.info("Partial replay: successfully resumed execution downstream from checkpoint.")
    except Exception as exc:
        logger.error("Partial replay update_state failed (%s) — falling back to full graph invoke", exc)
        result = graph.invoke(replay_state, config={"configurable": {"thread_id": new_task_id}})

    if isinstance(result, dict) and result.get("__interrupt__"):
        try:
            from langgraph.types import Command
            result = graph.invoke(
                Command(resume={"action": "approve_action", "decision": "approve"}),
                config=new_config,
            )
        except Exception:
            pass

    return new_task_id, result
