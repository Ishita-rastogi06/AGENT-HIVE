"""Small, application-specific wrapper around the local Ollama client."""

from typing import Optional

from ollama import Client

from src.config import settings


class OllamaClient:
    """Send concise requests to the local Ollama model."""

    MAX_RESPONSE_TOKENS = 220
    CONTEXT_WINDOW = 2048

    def __init__(self) -> None:
        self.client = Client(host=settings.ollama_host)
        self.model = settings.ollama_model

    def ask(
        self,
        prompt: str,
        system: str = "You are a helpful AI assistant.",
    ) -> Optional[str]:
        """Return a concise model response, or None if Ollama is unavailable."""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": 0.2,
                    "num_predict": self.MAX_RESPONSE_TOKENS,
                    "num_ctx": self.CONTEXT_WINDOW,
                },
            )
            return response["message"]["content"].strip()

        except Exception:
            return None

    def is_available(self) -> bool:
        """Return whether the local Ollama service can be reached."""
        try:
            self.client.list()
            return True
        except Exception:
            return False