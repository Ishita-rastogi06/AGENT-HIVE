"""
Approval Queue Manager, Sensitive Operation Detector, and Audit Logger for AgentHive.

Handles:
1. Sensitive operation detection before tool calls execute.
2. Persisted SQLite Approval Queue database that survives Streamlit script reruns and process restarts.
3. Structured logging of human approval events to JSONL and SQLite audit log tables.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sqlite3
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Sensitive Operation Detector ──────────────────────────────────────────────

_WRITE_SQL_RE = re.compile(
    r"\b(INSERT\s+INTO|UPDATE\s+\w+|DELETE\s+FROM|DROP\s+\b(?:TABLE|DATABASE|VIEW|INDEX)\b|\bCREATE\b\s+\b(?:TABLE|DATABASE|VIEW|INDEX)\b|ALTER\s+TABLE|TRUNCATE\s+TABLE|GRANT\s+\w+|REVOKE\s+\w+)\b",
    re.IGNORECASE,
)

_FILE_WRITE_CODE_RE = re.compile(
    r"(\bopen\s*\([^)]*['\"]w['\"]|\bopen\s*\([^)]*['\"]a['\"]|\.write\s*\(|"
    r"\bos\.remove\b|\bos\.unlink\b|\bshutil\.rmtree\b|\bPath\([^)]*\)\.write_|"
    r"\bos\.mkdir\b|\bos\.makedirs\b|\bos\.rmdir\b|\bsubprocess\b)",
    re.IGNORECASE,
)


def is_sensitive_operation(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    subtask: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """
    Determine whether a proposed tool invocation or subtask constitutes a sensitive operation.

    Returns (is_sensitive, reason_explanation).
    Must be checked BEFORE the tool call executes.
    """
    args = arguments or {}
    clean_tool = tool_name.strip().lower()

    # Check direct SQL write statements in tool query arguments or explicit raw SQL instructions
    sql_text = str(args.get("query") or args.get("code") or "")
    if subtask and not args:
        if subtask.get("sensitive"):
            return True, f"Subtask '{subtask.get('title', 'execution')}' is explicitly marked as sensitive operation."
        if subtask.get("assigned_agent") == "coder":
            return False, ""
        instruction = str(subtask.get("instruction") or "").strip()
        if instruction.upper().startswith(("INSERT INTO", "UPDATE ", "DELETE FROM", "DROP TABLE", "DROP DATABASE", "CREATE TABLE", "CREATE DATABASE", "ALTER TABLE", "TRUNCATE TABLE")):
            sql_text = instruction
        else:
            return False, ""

    # 1. Python sandbox execution with system command execution or explicit destruct file ops
    if clean_tool in ("python_sandbox", "sandbox_code_execution", "coder"):
        code = str(args.get("code") or "")
        if code:
            match = _FILE_WRITE_CODE_RE.search(code)
            if match:
                return True, f"Python sandbox contains file-write side-effects or system operations ('{match.group(1)}')"

    # 2. HTTP API call tool with non-idempotent HTTP methods (POST, PUT, DELETE, PATCH)
    if clean_tool in ("api_call", "api_call_tool", "api"):
        method = str(args.get("method") or "GET").strip().upper()
        if method in ("POST", "PUT", "DELETE", "PATCH"):
            return True, f"HTTP API call uses non-idempotent method '{method}'"

    # 3. Database query tool in non-read-only mode or running write SQL statements on live DB
    if clean_tool in ("db_query", "db_query_tool", "database_query", "data"):
        read_only = args.get("read_only", True)
        if not read_only:
            return True, "Database query is configured with read_only=False"
        if sql_text:
            match = _WRITE_SQL_RE.search(sql_text)
            if match:
                return True, f"Database query contains non-read-only SQL statement ('{match.group(1).upper()}')"

    return False, ""


_SCHEMA_INITIALIZED_PATHS: set[str] = set()


# ── Approval Database & Queue Persistence ──────────────────────────────────────

def get_approval_db_conn(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Return a thread-safe connection to the AgentHive SQLite database."""
    target_path = Path(db_path) if db_path else settings.sqlite_db_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _init_db_schema(conn, str(target_path))
    return conn


def _init_db_schema(conn: sqlite3.Connection, target_path_str: str) -> None:
    """Create the approval_queue and approval_audit_log tables if they don't exist (cached per process)."""
    if target_path_str in _SCHEMA_INITIALIZED_PATHS and Path(target_path_str).exists():
        return
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS approval_queue (
                task_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                task TEXT NOT NULL,
                trigger_reason TEXT NOT NULL,
                approval_level TEXT NOT NULL,
                proposed_action TEXT NOT NULL,
                reasoning TEXT,
                recalled_memories TEXT,
                plan_so_far TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                decision TEXT,
                custom_output TEXT,
                timestamp TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS approval_audit_log (
                event_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                trigger_type TEXT NOT NULL,
                approval_level TEXT NOT NULL,
                decision TEXT NOT NULL,
                user_response TEXT,
                timestamp TEXT NOT NULL
            );
            """
        )
    _SCHEMA_INITIALIZED_PATHS.add(target_path_str)


def add_pending_approval(
    task_id: str,
    user_id: str,
    task: str,
    trigger_reason: str,
    approval_level: str,
    proposed_action: str,
    reasoning: str = "",
    recalled_memories: str = "",
    plan_so_far: dict[str, Any] | str = "",
    db_path: Path | str | None = None,
) -> None:
    """Insert or update a pending approval in the SQLite queue."""
    conn = get_approval_db_conn(db_path=db_path)
    plan_str = json.dumps(plan_so_far, ensure_ascii=False) if isinstance(plan_so_far, dict) else str(plan_so_far)
    clean_user = (user_id or "default").strip().lower()

    logger.info(
        "[DB_LOG] INSERT/UPDATE approval_queue: task_id=%s, user_id=%s, trigger_reason=%s, proposed_action=%s, status=pending",
        task_id, clean_user, trigger_reason, proposed_action
    )

    with conn:
        conn.execute(
            """
            INSERT INTO approval_queue (
                task_id, user_id, task, trigger_reason, approval_level,
                proposed_action, reasoning, recalled_memories, plan_so_far, status, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            ON CONFLICT(task_id) DO UPDATE SET
                user_id=excluded.user_id,
                trigger_reason=excluded.trigger_reason,
                approval_level=excluded.approval_level,
                proposed_action=excluded.proposed_action,
                reasoning=excluded.reasoning,
                recalled_memories=excluded.recalled_memories,
                plan_so_far=excluded.plan_so_far,
                status='pending',
                timestamp=excluded.timestamp;
            """,
            (
                task_id, clean_user, task, trigger_reason, approval_level,
                proposed_action, reasoning, recalled_memories, plan_str, _now_iso(),
            ),
        )
    conn.close()


_INTERNAL_TEST_PREFIXES = (
    "_internal_verify_",
    "db_test_",
    "live_test_",
    "task_sensitive_",
    "task_restart_",
    "task_takeover_",
)


def _is_internal_test_task(task_id: str) -> bool:
    if not task_id:
        return False
    return any(task_id.startswith(prefix) for prefix in _INTERNAL_TEST_PREFIXES)


def get_pending_approvals(
    user_id: str | None = None,
    db_path: Path | str | None = None,
    include_internal: bool = True,
) -> list[dict[str, Any]]:
    """Fetch pending approval records for a specific user or all users."""
    conn = get_approval_db_conn(db_path=db_path)
    clean_user = user_id.strip().lower() if user_id else None

    if clean_user:
        cursor = conn.execute(
            "SELECT * FROM approval_queue WHERE status='pending' AND LOWER(user_id)=? ORDER BY timestamp DESC;",
            (clean_user,),
        )
        rows = [dict(row) for row in cursor.fetchall()]
        if not rows:
            # Fallback: if user-specific query returns empty, fetch all pending approvals
            cursor = conn.execute("SELECT * FROM approval_queue WHERE status='pending' ORDER BY timestamp DESC;")
            rows = [dict(row) for row in cursor.fetchall()]
    else:
        cursor = conn.execute(
            "SELECT * FROM approval_queue WHERE status='pending' ORDER BY timestamp DESC;"
        )
        rows = [dict(row) for row in cursor.fetchall()]

    conn.close()

    # Filter out internal test/verification script rows unless explicitly included
    if not include_internal:
        rows = [r for r in rows if not _is_internal_test_task(r.get("task_id", ""))]

    logger.info(
        "[DB_LOG] SELECT approval_queue: queried status='pending' for user_id=%s. Found %d item(s).",
        clean_user, len(rows)
    )
    return rows


def get_approval(task_id: str, db_path: Path | str | None = None) -> dict[str, Any] | None:
    """Fetch approval queue record for a specific task_id."""
    conn = get_approval_db_conn(db_path=db_path)
    cursor = conn.execute("SELECT * FROM approval_queue WHERE task_id=?;", (task_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def update_approval_status(
    task_id: str,
    status: str,
    decision: str = "approved",
    approval_level: str = "",
    custom_output: str | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Update approval record status to 'approved', 'rejected', or 'completed'."""
    conn = get_approval_db_conn(db_path=db_path)
    with conn:
        conn.execute(
            """
            UPDATE approval_queue
            SET status=?, decision=?, custom_output=?, timestamp=?
            WHERE task_id=?;
            """,
            (status, decision, custom_output or "", _now_iso(), task_id),
        )
    conn.close()


def clear_pending_approvals(user_id: str | None = None, db_path: Path | str | None = None) -> int:
    """Clear pending approval queue records for a user or all users."""
    conn = get_approval_db_conn(db_path=db_path)
    with conn:
        if user_id:
            cursor = conn.execute("UPDATE approval_queue SET status='cleared' WHERE status='pending' AND user_id=?;", (user_id,))
        else:
            cursor = conn.execute("UPDATE approval_queue SET status='cleared' WHERE status='pending';")
        count = cursor.rowcount
    conn.close()
    return count


# ── Audit Logger for Approval Events ──────────────────────────────────────────

def log_approval_event(
    task_id: str,
    user_id: str,
    trigger_type: str,
    approval_level: str,
    decision: str,
    user_response: str = "",
    db_path: Path | str | None = None,
    data_dir: Path | str | None = None,
) -> None:
    """
    Log every human approval decision to both the SQLite audit database table
    and the jsonl tool_calls audit file.
    """
    from uuid import uuid4
    event_id = str(uuid4())
    now = _now_iso()

    # 1. Save to SQLite database table
    try:
        conn = get_approval_db_conn(db_path=db_path)
        with conn:
            conn.execute(
                """
                INSERT INTO approval_audit_log (
                    event_id, task_id, user_id, trigger_type, approval_level, decision, user_response, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (event_id, task_id, user_id, trigger_type, approval_level, decision, user_response, now),
            )
        conn.close()
    except Exception as exc:
        logger.warning("Failed to log approval event to SQLite: %s", exc)

    # 2. Append to jsonl file
    target_data_dir = Path(data_dir) if data_dir else settings.data_dir
    target_data_dir.mkdir(parents=True, exist_ok=True)
    log_file = target_data_dir / "approval_events.jsonl"
    entry = {
        "event_id": event_id,
        "task_id": task_id,
        "user_id": user_id,
        "trigger_type": trigger_type,
        "approval_level": approval_level,
        "decision": decision,
        "user_response": user_response,
        "timestamp": now,
    }
    try:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Failed to write approval event to jsonl: %s", exc)
