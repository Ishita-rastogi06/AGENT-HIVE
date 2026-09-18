"""Path-restricted file operations for the AgentHive workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import settings
from src.tools.base import (
    BaseTool,
    ToolContext,
    ToolResult,
    tool_registry,
)


class WorkspaceFileTool(BaseTool):
    """
    Read and write text files only inside the configured workspace.

    Supported actions:
    - read
    - write
    - append
    - list
    - exists
    """

    name = "workspace_file"
    description = (
        "Read, write, append, list, or check text files inside the "
        "restricted AgentHive workspace directory."
    )

    supported_actions = {
        "read",
        "write",
        "append",
        "list",
        "exists",
    }

    maximum_read_bytes = 1_000_000
    maximum_write_characters = 500_000

    def validate_arguments(
        self,
        arguments: dict[str, Any],
    ) -> None:
        """Validate common file-tool arguments."""

        super().validate_arguments(arguments)

        action = arguments.get("action")

        if not isinstance(action, str) or not action.strip():
            raise ValueError(
                "The 'action' argument must be a non-empty string."
            )

        normalized_action = action.strip().lower()

        if normalized_action not in self.supported_actions:
            supported = ", ".join(sorted(self.supported_actions))

            raise ValueError(
                f"Unsupported file action '{normalized_action}'. "
                f"Supported actions: {supported}."
            )

        path = arguments.get("path", ".")

        if not isinstance(path, str):
            raise TypeError(
                "The 'path' argument must be a string."
            )

        if normalized_action in {"write", "append"}:
            content = arguments.get("content")

            if not isinstance(content, str):
                raise TypeError(
                    "The 'content' argument must be a string "
                    "for write and append actions."
                )

            if len(content) > self.maximum_write_characters:
                raise ValueError(
                    "File content is too large. Maximum allowed size is "
                    f"{self.maximum_write_characters} characters."
                )

    def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute a path-restricted workspace file operation."""

        action = str(arguments["action"]).strip().lower()
        requested_path = str(arguments.get("path", ".")).strip() or "."

        target_path = self._resolve_safe_path(requested_path)

        if action == "read":
            return self._read_file(target_path)

        if action == "write":
            return self._write_file(
                target_path=target_path,
                content=str(arguments["content"]),
                append=False,
            )

        if action == "append":
            return self._write_file(
                target_path=target_path,
                content=str(arguments["content"]),
                append=True,
            )

        if action == "list":
            recursive = bool(arguments.get("recursive", False))

            return self._list_directory(
                target_path=target_path,
                recursive=recursive,
            )

        if action == "exists":
            return ToolResult(
                success=True,
                output={
                    "exists": target_path.exists(),
                    "is_file": target_path.is_file(),
                    "is_directory": target_path.is_dir(),
                    "path": self._relative_display_path(target_path),
                },
            )

        return ToolResult(
            success=False,
            error=f"Action '{action}' was not executed.",
        )

    def _resolve_safe_path(self, requested_path: str) -> Path:
        """
        Resolve a path and ensure it remains inside the workspace.

        Absolute paths, parent-directory traversal, and symlink escapes are
        rejected automatically.
        """

        workspace_root = settings.workspace_dir.resolve()

        path_object = Path(requested_path)

        if path_object.is_absolute():
            raise PermissionError(
                "Absolute paths are not allowed. "
                "Use a path relative to the workspace."
            )

        target_path = (workspace_root / path_object).resolve()

        try:
            target_path.relative_to(workspace_root)
        except ValueError as error:
            raise PermissionError(
                "Requested path is outside the allowed workspace."
            ) from error

        return target_path

    def _read_file(self, target_path: Path) -> ToolResult:
        """Read a UTF-8 text file within the workspace."""

        if not target_path.exists():
            return ToolResult(
                success=False,
                error=(
                    "File does not exist: "
                    f"{self._relative_display_path(target_path)}"
                ),
            )

        if not target_path.is_file():
            return ToolResult(
                success=False,
                error=(
                    "Requested path is not a file: "
                    f"{self._relative_display_path(target_path)}"
                ),
            )

        file_size = target_path.stat().st_size

        if file_size > self.maximum_read_bytes:
            return ToolResult(
                success=False,
                error=(
                    "File is too large to read safely. "
                    f"Maximum allowed size is {self.maximum_read_bytes} bytes."
                ),
                metadata={
                    "size_bytes": file_size,
                },
            )

        try:
            content = target_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                error=(
                    "Only UTF-8 text files can be read by this tool."
                ),
            )

        return ToolResult(
            success=True,
            output=content,
            metadata={
                "path": self._relative_display_path(target_path),
                "size_bytes": file_size,
                "character_count": len(content),
            },
        )

    def _write_file(
        self,
        target_path: Path,
        content: str,
        append: bool,
    ) -> ToolResult:
        """Write or append UTF-8 text inside the workspace."""

        if target_path.exists() and target_path.is_dir():
            return ToolResult(
                success=False,
                error=(
                    "Cannot write content to a directory: "
                    f"{self._relative_display_path(target_path)}"
                ),
            )

        target_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        mode = "a" if append else "w"

        with target_path.open(
            mode,
            encoding="utf-8",
        ) as file_handle:
            file_handle.write(content)

        operation = "appended" if append else "written"

        return ToolResult(
            success=True,
            output=(
                f"Successfully {operation}: "
                f"{self._relative_display_path(target_path)}"
            ),
            metadata={
                "path": self._relative_display_path(target_path),
                "characters_processed": len(content),
                "operation": operation,
            },
        )

    def _list_directory(
        self,
        target_path: Path,
        recursive: bool,
    ) -> ToolResult:
        """List files and directories within a workspace folder."""

        if not target_path.exists():
            return ToolResult(
                success=False,
                error=(
                    "Directory does not exist: "
                    f"{self._relative_display_path(target_path)}"
                ),
            )

        if not target_path.is_dir():
            return ToolResult(
                success=False,
                error=(
                    "Requested path is not a directory: "
                    f"{self._relative_display_path(target_path)}"
                ),
            )

        iterator = (
            target_path.rglob("*")
            if recursive
            else target_path.iterdir()
        )

        entries: list[dict[str, Any]] = []

        for entry in sorted(iterator, key=lambda item: str(item).lower()):
            entries.append(
                {
                    "path": self._relative_display_path(entry),
                    "type": "directory" if entry.is_dir() else "file",
                    "size_bytes": (
                        entry.stat().st_size if entry.is_file() else None
                    ),
                }
            )

        return ToolResult(
            success=True,
            output=entries,
            metadata={
                "directory": self._relative_display_path(target_path),
                "recursive": recursive,
                "entry_count": len(entries),
            },
        )

    @staticmethod
    def _relative_display_path(target_path: Path) -> str:
        """Return a readable path relative to the workspace."""

        workspace_root = settings.workspace_dir.resolve()
        relative_path = target_path.resolve().relative_to(workspace_root)

        if str(relative_path) == ".":
            return "."

        return relative_path.as_posix()


workspace_file_tool = WorkspaceFileTool()
tool_registry.register(workspace_file_tool)