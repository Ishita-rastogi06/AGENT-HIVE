"""
AgentHive tool package.

Importing this package ensures every tool module is loaded so their
module-level ``tool_registry.register(...)`` calls fire exactly once.
The ToolRegistry.register() method is idempotent (skip-on-duplicate), so
re-importing this package or any individual tool module is always safe.

Registered tools (5 total, matching the Phase 1 spec):
  workspace_file  – path-restricted file I/O in the workspace directory
  python_sandbox  – sandboxed code execution with AST pre-validation
  web_search      – DuckDuckGo Instant Answer API (no key)
  db_query        – read-only Postgres query tool (requires DB_URL in .env)
  api_call        – generic HTTP call restricted to the allowed-domains list
"""

from src.tools.base import BaseTool, ToolContext, ToolResult, ToolRegistry, tool_registry  # noqa: F401
from src.tools.file_tools import workspace_file_tool  # noqa: F401
from src.tools.sandbox_tools import python_sandbox_tool  # noqa: F401
from src.tools.web_search_tool import web_search_tool  # noqa: F401
from src.tools.db_query_tool import db_query_tool  # noqa: F401
from src.tools.api_call_tool import api_call_tool  # noqa: F401

__all__ = [
    "BaseTool",
    "ToolContext",
    "ToolResult",
    "ToolRegistry",
    "tool_registry",
    "workspace_file_tool",
    "python_sandbox_tool",
    "web_search_tool",
    "db_query_tool",
    "api_call_tool",
]
