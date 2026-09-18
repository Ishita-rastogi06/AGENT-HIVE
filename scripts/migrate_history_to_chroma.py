"""
Migration script to backfill ChromaDB long-term memory from task_history.json.

Usage:
    py scripts/migrate_history_to_chroma.py [--user-id DEFAULT_USER_ID]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.tools.memory_tool import AgentMemoryTool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate")


def migrate_history_to_chroma(user_id: str = "default", history_file: Path | None = None) -> int:
    """Read task_history.json and store entries as long-term memories in ChromaDB."""
    target_file = history_file or (settings.data_dir / "task_history.json")
    if not target_file.exists():
        logger.warning("History file %s does not exist. Nothing to migrate.", target_file)
        return 0

    try:
        raw = target_file.read_text(encoding="utf-8")
        items = json.loads(raw)
    except Exception as exc:
        logger.error("Failed to read %s: %s", target_file, exc)
        return 0

    if not isinstance(items, list) or not items:
        logger.info("No task history items found in %s.", target_file)
        return 0

    tool = AgentMemoryTool()
    migrated_count = 0

    for idx, item in enumerate(items, 1):
        task = item.get("task", "").strip()
        final_answer = item.get("final_answer") or item.get("specialist_output") or ""
        if not task or not final_answer:
            continue

        selected_agent = item.get("selected_agent", "research")
        trace = item.get("trace") or []
        if not isinstance(trace, list):
            trace = [str(trace)]

        # Determine outcome from answer or trace
        is_error = "Ollama is not reachable" in final_answer or "could not be saved" in " ".join(trace)
        outcome = "failure" if is_error else "success"

        try:
            mem_id = tool.save_completed_task(
                task=task,
                final_answer=final_answer,
                selected_agent=selected_agent,
                trace=trace,
                user_id=user_id,
                task_type=selected_agent,
                outcome=outcome,
                final_confidence=0.85 if outcome == "success" else 0.30,
            )
            migrated_count += 1
            logger.info("Migrated [%d/%d] task memory ID: %s", idx, len(items), mem_id[:8])
        except Exception as exc:
            logger.warning("Failed to migrate item %d: %s", idx, exc)

    logger.info("Migration complete: %d / %d items backfilled to ChromaDB for user '%s'.", migrated_count, len(items), user_id)
    return migrated_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill ChromaDB from task_history.json")
    parser.add_argument("--user-id", default="default", help="User ID to tag memories with")
    args = parser.parse_args()

    migrate_history_to_chroma(user_id=args.user_id)
