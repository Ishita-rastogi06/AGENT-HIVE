"""Base contracts, registry, validation, and logging for AgentHive tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any
from uuid import uuid4

from src.config import settings


@dataclass
class ToolContext:
    """Metadata describing the agent, subtask, and task invoking a tool."""

    agent_name: str = "unknown"
    subtask_id: str = "unknown"
    task_id: str = ""


@dataclass
class ToolResult:
    """Standard response returned by every AgentHive tool."""

    success: bool
    output: Any = None
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary where possible."""

        return asdict(self)


class BaseTool(ABC):
    """Abstract contract that every registered tool must follow."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the tool and return a standard ToolResult."""

        raise NotImplementedError

    def validate_arguments(
        self,
        arguments: dict[str, Any],
    ) -> None:
        """Perform tool-specific argument validation when required."""

        if not isinstance(arguments, dict):
            raise TypeError("Tool arguments must be provided as a dictionary.")


class ToolRegistry:
    """
    Register and invoke tools through one controlled entry point.

    Calling tools through this registry ensures that every invocation is
    timed, normalized, and appended to the JSONL audit log.
    """

    def __init__(self, log_file: Path | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._explicit_log_file = log_file
        self._log_lock = Lock()

    @property
    def log_file(self) -> Path:
        """Return the effective tool log file path (dynamic to support test isolation)."""
        return self._explicit_log_file if self._explicit_log_file is not None else settings.tool_log_file

    def register(self, tool: BaseTool, *, allow_reregister: bool = False) -> BaseTool:
        """
        Register a tool instance and return it.

        By default, re-importing a module that calls ``tool_registry.register``
        at module level is safe: if the same tool name is already registered
        (same object *or* a new instance of the same class), the call is a
        no-op and the existing registration is returned.  Pass
        ``allow_reregister=True`` to force replacement of an existing tool.
        """

        tool_name = tool.name.strip()

        if not tool_name:
            raise ValueError(
                "A registered tool must define a non-empty name."
            )

        if not tool.description.strip():
            raise ValueError(
                f"Tool '{tool_name}' must define a description."
            )

        if tool_name in self._tools:
            if allow_reregister:
                self._tools[tool_name] = tool
            # Otherwise silently skip — idempotent re-import is safe.
            return self._tools[tool_name]

        self._tools[tool_name] = tool
        return tool

    def unregister(self, tool_name: str) -> None:
        """Remove a registered tool when it exists."""

        self._tools.pop(tool_name, None)

    def get(self, tool_name: str) -> BaseTool:
        """Return a registered tool or raise a helpful error."""

        try:
            return self._tools[tool_name]
        except KeyError as error:
            available = ", ".join(self.list_names()) or "none"

            raise KeyError(
                f"Unknown tool '{tool_name}'. "
                f"Available tools: {available}."
            ) from error

    def list_names(self) -> list[str]:
        """Return registered tool names in stable order."""

        return sorted(self._tools)

    def describe_tools(self) -> list[dict[str, str]]:
        """Return tool names and descriptions for supervisor prompts."""

        return [
            {
                "name": name,
                "description": self._tools[name].description,
            }
            for name in self.list_names()
        ]

    def invoke(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        context: ToolContext | None = None,
    ) -> tuple[ToolResult, dict[str, Any]]:
        """
        Invoke a registered tool and return its result plus audit record.

        Tool failures are converted into ToolResult objects instead of
        crashing the complete workflow.
        """

        call_id = str(uuid4())
        started_at = datetime.now(timezone.utc)
        timer_started = perf_counter()

        normalized_arguments = arguments or {}
        normalized_context = context or ToolContext()

        result = ToolResult(
            success=False,
            error="Tool execution did not start.",
        )

        try:
            tool = self.get(tool_name)
            tool.validate_arguments(normalized_arguments)

            raw_result = tool.execute(
                arguments=normalized_arguments,
                context=normalized_context,
            )

            if not isinstance(raw_result, ToolResult):
                raise TypeError(
                    f"Tool '{tool_name}' returned an invalid result type."
                )

            result = raw_result

        except Exception as error:
            result = ToolResult(
                success=False,
                output=None,
                error=f"{type(error).__name__}: {error}",
            )

        duration_ms = round(
            (perf_counter() - timer_started) * 1000,
            2,
        )

        audit_record = {
            "call_id": call_id,
            "timestamp": started_at.isoformat(),
            "tool_name": tool_name,
            "agent_name": normalized_context.agent_name,
            "subtask_id": normalized_context.subtask_id,
            "task_id": normalized_context.task_id or None,
            "arguments": self._make_json_safe(normalized_arguments),
            "result_preview": self._build_result_preview(result),
            "success": result.success,
            "error": result.error,
            "duration_ms": duration_ms,
        }

        self._append_log(audit_record)

        return result, audit_record

    def _build_result_preview(self, result: ToolResult) -> str:
        """Create a bounded preview suitable for logs and workflow state."""

        if result.success:
            value = result.output
        else:
            value = result.error

        if isinstance(value, str):
            preview = value
        else:
            preview = json.dumps(
                self._make_json_safe(value),
                ensure_ascii=False,
                sort_keys=True,
            )

        maximum_length = settings.maximum_tool_output_chars

        if len(preview) > maximum_length:
            return (
                preview[:maximum_length].rstrip()
                + "\n...[tool output truncated]"
            )

        return preview

    def _append_log(
        self,
        audit_record: dict[str, Any],
    ) -> None:
        """Append one audit record to the JSONL log safely."""

        serialized_record = json.dumps(
            self._make_json_safe(audit_record),
            ensure_ascii=False,
            sort_keys=True,
        )

        target_log = self.log_file
        target_log.parent.mkdir(parents=True, exist_ok=True)

        with self._log_lock:
            with target_log.open(
                "a",
                encoding="utf-8",
            ) as log_handle:
                log_handle.write(serialized_record + "\n")

    @staticmethod
    def _make_json_safe(value: Any) -> Any:
        """Convert common Python objects into JSON-safe values."""

        if value is None or isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, dict):
            return {
                str(key): ToolRegistry._make_json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [
                ToolRegistry._make_json_safe(item)
                for item in value
            ]

        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return repr(value)


tool_registry = ToolRegistry()