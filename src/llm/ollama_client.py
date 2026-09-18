"""
Application-specific wrapper around the local Ollama client (ollama 0.6.x).

Error classification
────────────────────
ConnectionError  – Ollama process not running or wrong host
ModelNotFoundError – model name is not pulled / does not exist
TimeoutError     – LLM took longer than llm_timeout_seconds
OllamaClientError – any other API-level error
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from ollama import Client
from ollama._types import RequestError, ResponseError

from src.config import settings

logger = logging.getLogger(__name__)


# ── custom exception hierarchy ────────────────────────────────────────────────

class OllamaClientError(RuntimeError):
    """Base class for all OllamaClient errors."""


class OllamaConnectionError(OllamaClientError):
    """Raised when the Ollama process cannot be reached."""


class OllamaModelNotFoundError(OllamaClientError):
    """Raised when the requested model has not been pulled."""


class OllamaTimeoutError(OllamaClientError):
    """Raised when the LLM takes longer than the configured timeout."""


# ── client ────────────────────────────────────────────────────────────────────

class OllamaClient:
    """
    Send requests to the local Ollama service.

    Response parsing uses the Pydantic model returned by ollama 0.6.x:
        ChatResponse.message  →  Message object
        Message.content       →  str | None

    Config values (all overridable via .env / Settings):
        OLLAMA_HOST, OLLAMA_MODEL, LLM_TIMEOUT_SECONDS,
        LLM_MAX_TOKENS, LLM_CONTEXT_WINDOW, LLM_TEMPERATURE
    """

    def __init__(self) -> None:
        self.client = Client(
            host=settings.ollama_host,
            timeout=settings.llm_timeout_seconds,
        )
        self.model = settings.ollama_model

    # ── public API ────────────────────────────────────────────────────────────

    def ask(
        self,
        prompt: str,
        system: str = "You are a helpful AI assistant.",
        num_predict: Optional[int] = None,
    ) -> Optional[str]:
        """
        Send a chat request and return the model's text response.

        Returns ``None`` when Ollama is unreachable or the model is missing,
        so callers can degrade gracefully.  Specific errors are logged at
        WARNING level with the error class name so they are distinguishable.
        """
        max_tokens = num_predict if num_predict is not None else settings.llm_max_tokens
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": settings.llm_temperature,
                    "num_predict": max_tokens,
                    "num_ctx": settings.llm_context_window,
                },
                keep_alive="1h",
            )
            # ChatResponse is a Pydantic model: response.message is a Message
            # object with a .content str attribute (ollama 0.6.x).
            content: str | None = response.message.content
            return content.strip() if content else None

        except httpx.ConnectError as exc:
            logger.warning(
                "OllamaConnectionError: cannot reach Ollama at %s — %s",
                settings.ollama_host,
                exc,
            )
            return None

        except httpx.TimeoutException as exc:
            logger.warning(
                "OllamaTimeoutError: model '%s' exceeded %ds timeout — %s",
                self.model,
                settings.llm_timeout_seconds,
                exc,
            )
            return None

        except ResponseError as exc:
            # ResponseError.status_code 404 means the model is not pulled.
            status = getattr(exc, "status_code", None)
            if status == 404 or "not found" in str(exc).lower():
                logger.warning(
                    "OllamaModelNotFoundError: model '%s' is not pulled — run "
                    "`ollama pull %s` to download it. Detail: %s",
                    self.model,
                    self.model,
                    exc,
                )
            else:
                logger.warning(
                    "OllamaClientError (ResponseError status=%s): %s",
                    status,
                    exc,
                )
            return None

        except RequestError as exc:
            logger.warning("OllamaClientError (RequestError): %s", exc)
            return None

        except Exception as exc:
            logger.warning(
                "OllamaClientError (unexpected %s): %s",
                type(exc).__name__,
                exc,
            )
            return None

    def is_available(self) -> bool:
        """Return whether the local Ollama service can be reached."""
        try:
            self.client.list()
            return True
        except (httpx.ConnectError, httpx.TimeoutException):
            logger.debug("Ollama not available at %s", settings.ollama_host)
            return False
        except Exception as exc:
            logger.debug("Ollama availability check failed: %s", exc)
            return False
