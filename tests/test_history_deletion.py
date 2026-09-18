import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ui.streamlit_views import load_task_history, save_task_history


class TestTaskHistorySingleDeletion(unittest.TestCase):
    """Test single task history deletion and persistence."""

    def test_save_and_load_task_history(self):
        sample_history = [
            {"task_id": "t1", "task": "Task 1", "timestamp": "01 Jan 2026"},
            {"task_id": "t2", "task": "Task 2", "timestamp": "02 Jan 2026"},
        ]
        tmp_path = Path("scratch/test_history.json")
        with patch("src.ui.streamlit_views._get_history_file", return_value=tmp_path):
            try:
                save_task_history(sample_history)
                loaded = load_task_history()
                self.assertEqual(len(loaded), 2)
                self.assertEqual(loaded[0]["task_id"], "t1")

                # Perform single item deletion
                target_id = "t1"
                filtered = [h for h in loaded if h.get("task_id") != target_id]
                save_task_history(filtered)

                reloaded = load_task_history()
                self.assertEqual(len(reloaded), 1)
                self.assertEqual(reloaded[0]["task_id"], "t2")
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
