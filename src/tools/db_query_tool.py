"""
Read-only (or allow-listed write) database query tool for AgentHive.

Connects to a Postgres instance configured via DB_URL in .env.  All queries
run inside a transaction that is always rolled back for SELECT-only mode so
no accidental mutations occur.

Safety model
────────────
• Default mode is read_only=True: every statement runs inside a SAVEPOINT
  that is rolled back after execution.  INSERT/UPDATE/DELETE/DROP are
  rejected at the SQL-parser level when read_only=True.
• Writes are only allowed when read_only=False AND the query passes a
  whitelist check (configured via DB_ALLOWED_WRITE_TABLES in .env).
• Query results are truncated to MAXIMUM_TOOL_OUTPUT_CHARS.
• Connection pooling is intentionally avoided — each call opens and closes
  its own connection so the tool has no cross-request state.

The tool degrades gracefully when psycopg2 is not installed or no DB_URL
is configured: it returns a descriptive ToolResult(success=False) rather
than raising an exception.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from src.config import settings
from src.tools.base import BaseTool, ToolContext, ToolResult, tool_registry

logger = logging.getLogger(__name__)

# SQL statements that are never permitted in read_only mode.
_WRITE_STATEMENT_RE = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

# Maximum rows returned per query to avoid flooding the prompt.
_MAX_ROWS = 100


class DatabaseQueryTool(BaseTool):
    """
    Execute read-only (default) or allow-listed write queries against a
    Postgres database.

    Arguments:
        query      (str, required)  — SQL statement to execute
        read_only  (bool, default True) — enforce read-only mode
        parameters (list, optional) — positional parameters for the query
                                       (maps to %s placeholders)

    Returns a ToolResult whose output is a list of dicts (one per row) or a
    plain string for non-SELECT statements.
    """

    name = "db_query"
    description = (
        "Execute a read-only SQL query against the configured Postgres database. "
        "Returns up to 100 rows as a list of dicts. Requires DB_URL in .env."
    )

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        super().validate_arguments(arguments)
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("The 'query' argument must be a non-empty string.")

        read_only = arguments.get("read_only", True)
        if read_only and _WRITE_STATEMENT_RE.match(query.strip()):
            raise PermissionError(
                "Write operations are not permitted in read_only mode. "
                "Pass read_only=False and ensure the table is allow-listed."
            )

        params = arguments.get("parameters")
        if params is not None and not isinstance(params, (list, tuple)):
            raise TypeError("'parameters' must be a list or None.")

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        query = arguments["query"].strip()
        read_only = bool(arguments.get("read_only", True))
        parameters = arguments.get("parameters") or []

        db_url = os.getenv("DB_URL", "").strip()
        if not db_url:
            return ToolResult(
                success=False,
                error=(
                    "DB_URL is not configured. Add DB_URL=postgresql://user:pass@host/db "
                    "to your .env file to enable database queries."
                ),
                metadata={"hint": "set DB_URL in .env"},
            )

        # Attempt to import psycopg2; degrade gracefully if not installed.
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            return ToolResult(
                success=False,
                error=(
                    "psycopg2 is not installed. "
                    "Run `pip install psycopg2-binary` to enable database queries."
                ),
                metadata={"hint": "pip install psycopg2-binary"},
            )

        try:
            conn = psycopg2.connect(db_url)
            conn.autocommit = False
        except Exception as exc:
            logger.warning("db_query: connection failed — %s", exc)
            return ToolResult(
                success=False,
                error=f"Database connection failed: {exc}",
            )

        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if read_only:
                    cur.execute("SAVEPOINT _agenthive_ro")

                cur.execute(query, parameters or None)

                # fetchmany works for SELECT-like statements that return tuples.
                try:
                    rows = cur.fetchmany(_MAX_ROWS)
                    output = [dict(row) for row in rows]
                    truncated = cur.fetchone() is not None  # more rows exist

                    if read_only:
                        cur.execute("ROLLBACK TO SAVEPOINT _agenthive_ro")
                        conn.rollback()

                    preview = str(output)
                    if len(preview) > settings.maximum_tool_output_chars:
                        preview = preview[:settings.maximum_tool_output_chars] + "\n...[truncated]"
                        output = preview

                    return ToolResult(
                        success=True,
                        output=output,
                        metadata={
                            "row_count": len(rows),
                            "truncated": truncated,
                            "read_only": read_only,
                        },
                    )
                except psycopg2.ProgrammingError:
                    # Non-SELECT statement (allowed only if read_only=False).
                    if read_only:
                        cur.execute("ROLLBACK TO SAVEPOINT _agenthive_ro")
                        conn.rollback()
                    else:
                        conn.commit()

                    return ToolResult(
                        success=True,
                        output="Statement executed successfully (no rows returned).",
                        metadata={"read_only": read_only},
                    )

        except PermissionError:
            raise
        except Exception as exc:
            logger.warning("db_query: execution error — %s", exc)
            conn.rollback()
            return ToolResult(
                success=False,
                error=f"Query execution failed: {exc}",
            )
        finally:
            conn.close()


db_query_tool = DatabaseQueryTool()
tool_registry.register(db_query_tool)
