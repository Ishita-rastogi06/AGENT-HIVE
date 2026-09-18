"""
Integration and unit tests for DatabaseQueryTool with real PostgreSQL.

Skipped automatically when DB_URL is not configured in the test environment.
"""

from __future__ import annotations

import os
import pytest

from src.config import settings
from src.tools.base import ToolContext
from src.tools.db_query_tool import DatabaseQueryTool, db_query_tool

DB_URL = os.getenv("DB_URL") or settings.db_url


def _is_postgres_reachable(db_url: str) -> bool:
    if not db_url:
        return False
    try:
        import psycopg2
        conn = psycopg2.connect(db_url, connect_timeout=2)
        conn.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _is_postgres_reachable(DB_URL), reason="PostgreSQL database is not reachable at DB_URL")
def test_postgres_real_insert_and_select_roundtrip():
    """
    Connect to real Postgres, execute an INSERT (simulating post-approval write),
    commit, and confirm row persistence via a follow-up SELECT.
    """
    ctx = ToolContext(agent_name="data", subtask_id="test_subtask_pg")

    # 1. Execute INSERT statement with read_only=False (simulates post-approval execution)
    insert_sql = "INSERT INTO audit_logs (event_type, description) VALUES (%s, %s)"
    insert_args = {
        "query": insert_sql,
        "read_only": False,
        "parameters": ["integration_test", "Real Postgres INSERT test"],
    }

    # Validate argument rules
    db_query_tool.validate_arguments(insert_args)
    insert_res = db_query_tool.execute(insert_args, ctx)

    assert insert_res.success is True, f"INSERT failed: {insert_res.error}"

    # 2. Execute SELECT query with read_only=True to confirm row retrieval
    select_sql = "SELECT id, event_type, description FROM audit_logs WHERE event_type = %s ORDER BY id DESC LIMIT 1"
    select_args = {
        "query": select_sql,
        "read_only": True,
        "parameters": ["integration_test"],
    }

    db_query_tool.validate_arguments(select_args)
    select_res = db_query_tool.execute(select_args, ctx)

    assert select_res.success is True, f"SELECT failed: {select_res.error}"
    assert isinstance(select_res.output, list)
    assert len(select_res.output) > 0, "Inserted row was not found in audit_logs"

    row = select_res.output[0]
    assert row["event_type"] == "integration_test"
    assert "Real Postgres INSERT test" in row["description"]


def test_db_query_tool_validation_rules():
    """Verify tool argument validation for write queries in read_only mode."""
    tool = DatabaseQueryTool()

    # Write statement in read_only mode must raise PermissionError
    with pytest.raises(PermissionError):
        tool.validate_arguments({
            "query": "INSERT INTO audit_logs (event_type, description) VALUES ('test', 'fail')",
            "read_only": True,
        })

    # Empty query must raise ValueError
    with pytest.raises(ValueError):
        tool.validate_arguments({"query": "   ", "read_only": True})

    # Valid SELECT query in read_only mode passes validation
    tool.validate_arguments({
        "query": "SELECT * FROM audit_logs",
        "read_only": True,
    })
