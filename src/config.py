"""Application configuration loaded from environment variables."""

from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _get_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable safely."""

    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _get_int(name: str, default: int) -> int:
    """Read an integer environment variable safely."""

    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    """Read a float environment variable safely."""

    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_allowed_domains() -> tuple[str, ...]:
    """Return API domains that tools are allowed to contact."""

    raw_value = os.getenv(
        "ALLOWED_API_DOMAINS",
        "api.github.com,jsonplaceholder.typicode.com",
    )

    return tuple(
        domain.strip().lower()
        for domain in raw_value.split(",")
        if domain.strip()
    )


@dataclass
class Settings:
    """Central configuration for the AgentHive application."""

    # Ollama configuration
    ollama_host: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_HOST",
            "http://localhost:11434",
        )
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_MODEL",
            "llama3.2:3b",
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_EMBEDDING_MODEL",
            "nomic-embed-text",
        )
    )

    # When Ollama is unavailable, deterministic local responses are used.
    enable_offline_fallback: bool = field(
        default_factory=lambda: _get_bool(
            "ENABLE_OFFLINE_FALLBACK",
            True,
        )
    )

    llm_timeout_seconds: int = field(
        default_factory=lambda: _get_int(
            "LLM_TIMEOUT_SECONDS",
            60,
        )
    )
    llm_max_tokens: int = field(
        default_factory=lambda: _get_int(
            "LLM_MAX_TOKENS",
            1200,
        )
    )
    llm_context_window: int = field(
        default_factory=lambda: _get_int(
            "LLM_CONTEXT_WINDOW",
            4096,
        )
    )
    llm_temperature: float = field(
        default_factory=lambda: _get_float(
            "LLM_TEMPERATURE",
            0.2,
        )
    )

    # Workflow configuration
    max_retries: int = field(
        default_factory=lambda: _get_int(
            "MAX_RETRIES",
            2,
        )
    )
    low_confidence_threshold: float = field(
        default_factory=lambda: _get_float(
            "LOW_CONFIDENCE_THRESHOLD",
            0.60,
        )
    )

    # Granular escalation default approval levels
    default_approval_levels: dict[str, str] = field(
        default_factory=lambda: {
            "low_plan_confidence": "Approve Plan",
            "sensitive_operation": "Approve Action",
            "explicit_user_request": "Approve Action",
            "reviewer_escalation": "Approve Plan",
            "retries_exhausted": "Take Over",
        }
    )

    # Tool safety configuration
    sandbox_timeout_seconds: int = field(
        default_factory=lambda: _get_int(
            "SANDBOX_TIMEOUT_SECONDS",
            10,
        )
    )
    api_timeout_seconds: int = field(
        default_factory=lambda: _get_int(
            "API_TIMEOUT_SECONDS",
            15,
        )
    )
    maximum_tool_output_chars: int = field(
        default_factory=lambda: _get_int(
            "MAXIMUM_TOOL_OUTPUT_CHARS",
            8_000,
        )
    )
    allowed_api_domains: tuple[str, ...] = field(
        default_factory=_get_allowed_domains
    )

    # ── Database — PostgreSQL ──────────────────────────────────────────────────
    db_url: str = field(
        default_factory=lambda: os.getenv("DB_URL", "")
    )

    # ── Memory — Redis (short-term) ───────────────────────────────────────────
    # Connection URL for task-scoped working memory.
    # When absent the app degrades gracefully (short-term memory skipped).
    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "")
    )
    # Seconds before a task's short-term memory key expires automatically.
    short_term_ttl_seconds: int = field(
        default_factory=lambda: _get_int("SHORT_TERM_TTL_SECONDS", 3600)
    )

    # ── Memory — ChromaDB (long-term) ─────────────────────────────────────────
    # Path to the persistent ChromaDB storage directory.
    chroma_memory_path: str = field(
        default_factory=lambda: os.getenv("CHROMA_MEMORY_PATH", "")
    )
    # How many past memories to retrieve for each planning call.
    memory_top_k: int = field(
        default_factory=lambda: _get_int("MEMORY_TOP_K", 3)
    )
    # Maximum cosine distance for a memory to be considered relevant (0–2).
    memory_max_distance: float = field(
        default_factory=lambda: _get_float("MEMORY_MAX_DISTANCE", 0.85)
    )
    # Importance score below which old memories are expired.
    memory_min_importance: float = field(
        default_factory=lambda: _get_float("MEMORY_MIN_IMPORTANCE", 0.10)
    )
    # Days after which an unimportant memory is eligible for expiry.
    memory_expiry_days: int = field(
        default_factory=lambda: _get_int("MEMORY_EXPIRY_DAYS", 90)
    )
    # Cosine-distance threshold below which two memories are near-duplicates.
    memory_consolidation_threshold: float = field(
        default_factory=lambda: _get_float("MEMORY_CONSOLIDATION_THRESHOLD", 0.15)
    )

    # Runtime directories and files
    data_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data"
    )
    temp_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "temp"
    )
    workspace_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "workspace"
    )
    tool_log_file: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "tool_calls.jsonl"
    )
    sqlite_db_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "agent_hive.db"
    )

    def ensure_runtime_directories(self) -> None:
        """Create all writable application directories."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.tool_log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.sqlite_db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def is_api_domain_allowed(self, hostname: str) -> bool:
        """
        Check whether a hostname is included in the API allowlist.

        Subdomains are accepted when their parent domain is allowlisted.
        """

        normalized_hostname = hostname.strip().lower()

        return any(
            normalized_hostname == allowed_domain
            or normalized_hostname.endswith(f".{allowed_domain}")
            for allowed_domain in self.allowed_api_domains
        )


settings = Settings()
settings.ensure_runtime_directories()