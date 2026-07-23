"""Application settings loaded from environment variables."""
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

@dataclass(frozen=True)
class Settings:
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    data_dir: Path = PROJECT_ROOT / "data"
    temp_dir: Path = PROJECT_ROOT / "temp"

    def ensure_runtime_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

settings = Settings()
