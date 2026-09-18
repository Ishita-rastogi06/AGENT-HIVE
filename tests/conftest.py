"""
Global pytest configuration and fixtures for AgentHive test isolation.

Ensures that running unit and integration tests NEVER writes to production
data files in data/ (such as data/tool_calls.jsonl, data/agent_hive.db,
or data/task_history.json).
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from src.config import settings


@pytest.fixture(autouse=True, scope="function")
def isolate_test_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Automatically isolate every pytest run to a temporary directory.
    Redirects settings.data_dir, tool_log_file, sqlite_db_path, temp_dir,
    workspace_dir, and chroma_memory_path to tmp_path.
    """
    test_data_dir = tmp_path / "data"
    test_temp_dir = tmp_path / "temp"
    test_workspace_dir = tmp_path / "workspace"
    test_tool_log = test_data_dir / "tool_calls.jsonl"
    test_sqlite_db = test_data_dir / "agent_hive.db"
    test_chroma_dir = tmp_path / "chroma_memory"

    test_data_dir.mkdir(parents=True, exist_ok=True)
    test_temp_dir.mkdir(parents=True, exist_ok=True)
    test_workspace_dir.mkdir(parents=True, exist_ok=True)
    test_chroma_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "data_dir", test_data_dir)
    monkeypatch.setattr(settings, "temp_dir", test_temp_dir)
    monkeypatch.setattr(settings, "workspace_dir", test_workspace_dir)
    monkeypatch.setattr(settings, "tool_log_file", test_tool_log)
    monkeypatch.setattr(settings, "sqlite_db_path", test_sqlite_db)
    monkeypatch.setattr(settings, "chroma_memory_path", str(test_chroma_dir))
    monkeypatch.setenv("CHROMA_MEMORY_PATH", str(test_chroma_dir))

    settings.ensure_runtime_directories()

    yield
