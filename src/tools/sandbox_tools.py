"""Restricted Python execution tool for AgentHive coder tasks."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from src.config import settings
from src.tools.base import (
    BaseTool,
    ToolContext,
    ToolResult,
    tool_registry,
)


ALLOWED_MODULES = {
    "collections",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "itertools",
    "json",
    "math",
    "random",
    "re",
    "statistics",
    "string",
}

BLOCKED_NAMES = {
    "__builtins__",
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "exit",
    # getattr and setattr are ALLOWED — they are needed for legitimate code
    # (e.g. dynamic dispatch, dataclass helpers).  A runtime wrapper in
    # _SAFE_BUILTINS blocks dunder-attribute access instead (see below).
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "quit",
    # setattr allowed — same reasoning as getattr above.
    "vars",
}

BLOCKED_MODULES = {
    "asyncio",
    "ctypes",
    "http",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "urllib",
}


def _apply_posix_resource_limits() -> None:
    """
    Limit child-process resources on Linux and macOS.

    Windows still receives process isolation, an empty temporary working
    directory, bounded output, and a strict execution timeout.
    """

    try:
        import resource

        timeout = max(settings.sandbox_timeout_seconds, 1)

        resource.setrlimit(
            resource.RLIMIT_CPU,
            (timeout, timeout + 1),
        )
        resource.setrlimit(
            resource.RLIMIT_AS,
            (256 * 1024 * 1024, 256 * 1024 * 1024),
        )
        resource.setrlimit(
            resource.RLIMIT_FSIZE,
            (10 * 1024 * 1024, 10 * 1024 * 1024),
        )

        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(
                resource.RLIMIT_NPROC,
                (1, 1),
            )

    except (ImportError, OSError, ValueError):
        # Timeout and subprocess isolation still remain active.
        pass


class PythonSandboxTool(BaseTool):
    """Execute small Python snippets in a restricted child process."""

    name = "python_sandbox"
    description = (
        "Execute calculation and data-processing Python code in a "
        "restricted temporary process with import checks and a timeout."
    )

    maximum_code_characters = 20_000
    maximum_output_characters = 20_000

    def validate_arguments(
        self,
        arguments: dict[str, Any],
    ) -> None:
        """Validate and statically inspect the submitted Python code."""

        super().validate_arguments(arguments)

        code = arguments.get("code")

        if not isinstance(code, str) or not code.strip():
            raise ValueError(
                "The 'code' argument must be a non-empty string."
            )

        if len(code) > self.maximum_code_characters:
            raise ValueError(
                "Submitted code is too large. Maximum allowed size is "
                f"{self.maximum_code_characters} characters."
            )

        try:
            syntax_tree = ast.parse(
                code,
                mode="exec",
            )
        except SyntaxError as error:
            raise ValueError(
                f"Python syntax error on line {error.lineno}: "
                f"{error.msg}"
            ) from error

        self._validate_syntax_tree(syntax_tree)

    def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Run validated Python code inside a temporary child process."""

        code = str(arguments["code"])
        execution_wrapper = self._build_execution_wrapper(code)

        settings.temp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        process_environment = {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }

        process_kwargs: dict[str, Any] = {
            "args": [
                sys.executable,
                "-I",
                "-c",
                execution_wrapper,
            ],
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "timeout": settings.sandbox_timeout_seconds,
            "env": process_environment,
        }

        if os.name == "posix":
            process_kwargs["preexec_fn"] = _apply_posix_resource_limits

        if os.name == "nt":
            process_kwargs["creationflags"] = getattr(
                subprocess,
                "CREATE_NO_WINDOW",
                0,
            )

        try:
            with tempfile.TemporaryDirectory(
                prefix="agenthive_",
                dir=settings.temp_dir,
            ) as temporary_directory:
                process_kwargs["cwd"] = temporary_directory

                completed_process = subprocess.run(
                    **process_kwargs,
                )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=(
                    "Python execution exceeded the "
                    f"{settings.sandbox_timeout_seconds}-second timeout."
                ),
                metadata={
                    "timed_out": True,
                },
            )

        except OSError as error:
            return ToolResult(
                success=False,
                error=f"Could not start Python sandbox: {error}",
            )

        standard_output = self._bound_output(
            completed_process.stdout.strip()
        )
        standard_error = self._bound_output(
            completed_process.stderr.strip()
        )

        if completed_process.returncode != 0:
            return ToolResult(
                success=False,
                output=standard_output or None,
                error=standard_error or (
                    "Python execution failed without an error message."
                ),
                metadata={
                    "return_code": completed_process.returncode,
                },
            )

        return ToolResult(
            success=True,
            output=standard_output or "(code completed with no output)",
            metadata={
                "return_code": completed_process.returncode,
                "stderr": standard_error,
            },
        )

    def _validate_syntax_tree(
        self,
        syntax_tree: ast.AST,
    ) -> None:
        """Reject imports and language features that can escape isolation."""

        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_module = alias.name.split(".")[0]

                    if root_module not in ALLOWED_MODULES:
                        raise PermissionError(
                            f"Import '{alias.name}' is not allowed."
                        )

            elif isinstance(node, ast.ImportFrom):
                root_module = (
                    (node.module or "").split(".")[0]
                )

                if (
                    not root_module
                    or root_module not in ALLOWED_MODULES
                ):
                    raise PermissionError(
                        f"Import from '{node.module}' is not allowed."
                    )

            elif isinstance(node, ast.Name):
                if node.id in BLOCKED_NAMES:
                    raise PermissionError(
                        f"Use of '{node.id}' is not allowed."
                    )

                if node.id in BLOCKED_MODULES:
                    raise PermissionError(
                        f"Use of module '{node.id}' is not allowed."
                    )

            elif isinstance(node, ast.Attribute):
                # Block explicit access to private / dunder attributes written
                # directly in source (e.g. obj.__class__, obj._secret).
                # Normal method calls like list.append() or x.__len__() via
                # Python's implicit operator dispatch are NOT blocked here —
                # the runtime wrapper for getattr/setattr handles that path.
                if node.attr.startswith("_"):
                    raise PermissionError(
                        f"Direct access to private/dunder attribute "
                        f"'{node.attr}' is not allowed. "
                        "Use public methods instead."
                    )

            elif isinstance(node, ast.Call):
                # Block calls to hard-blocked builtins (eval, exec, etc.).
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id in BLOCKED_NAMES
                ):
                    raise PermissionError(
                        f"Calling '{node.func.id}' is not allowed."
                    )

                # Narrow check: block getattr(x, "__dunder__") and
                # setattr(x, "__dunder__", value) at the AST level.
                # Legitimate getattr(obj, "public_attr") is fine.
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id in ("getattr", "setattr")
                    and len(node.args) >= 2
                ):
                    attr_arg = node.args[1]
                    if (
                        isinstance(attr_arg, ast.Constant)
                        and isinstance(attr_arg.value, str)
                        and attr_arg.value.startswith("_")
                    ):
                        raise PermissionError(
                            f"Calling {node.func.id}() with a private/dunder "
                            f"attribute name '{attr_arg.value}' is not allowed."
                        )

    @staticmethod
    def _build_execution_wrapper(code: str) -> str:
        """
        Build the isolated interpreter payload.

        A reduced built-in namespace is supplied to the submitted code.
        """

        allowed_modules_literal = repr(sorted(ALLOWED_MODULES))
        code_literal = repr(code)

        return f"""
import builtins as _builtins

_ALLOWED_MODULES = set({allowed_modules_literal})
_original_import = _builtins.__import__


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root_name = name.split(".")[0]

    if root_name not in _ALLOWED_MODULES:
        raise ImportError(
            "Import is not allowed in the AgentHive sandbox: " + name
        )

    return _original_import(
        name,
        globals,
        locals,
        fromlist,
        level,
    )


def _safe_getattr(obj, name, *args):
    if isinstance(name, str) and name.startswith("_"):
        raise AttributeError(
            "getattr() with a private/dunder attribute name '"
            + name + "' is not allowed in the sandbox."
        )
    return getattr(obj, name, *args)


def _safe_setattr(obj, name, value):
    if isinstance(name, str) and name.startswith("_"):
        raise AttributeError(
            "setattr() with a private/dunder attribute name '"
            + name + "' is not allowed in the sandbox."
        )
    return setattr(obj, name, value)


_SAFE_BUILTINS = {{
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "getattr": _safe_getattr,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "object": object,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "setattr": _safe_setattr,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "Exception": Exception,
    "ArithmeticError": ArithmeticError,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "ZeroDivisionError": ZeroDivisionError,
    "__build_class__": __build_class__,
    "__import__": _safe_import,
}}

execution_globals = {{
    "__builtins__": _SAFE_BUILTINS,
    "__name__": "__sandbox__",
}}

submitted_code = {code_literal}
compiled_code = compile(
    submitted_code,
    "<agenthive-sandbox>",
    "exec",
)
exec(
    compiled_code,
    execution_globals,
    execution_globals,
)
"""

    def _bound_output(self, output: str) -> str:
        """Prevent unusually large child-process output."""

        if len(output) <= self.maximum_output_characters:
            return output

        return (
            output[:self.maximum_output_characters].rstrip()
            + "\n...[sandbox output truncated]"
        )


python_sandbox_tool = PythonSandboxTool()
tool_registry.register(python_sandbox_tool)