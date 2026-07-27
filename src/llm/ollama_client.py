"""Application-specific wrapper around the local Ollama client."""

from typing import Optional

from ollama import Client

from src.config import settings


class OllamaClient:
    """Send requests to the local Ollama model."""

    # Enough room for complete code, explanations, and test cases.
    MAX_RESPONSE_TOKENS = 1200
    CONTEXT_WINDOW = 4096

    def __init__(self) -> None:
        self.client = Client(host=settings.ollama_host)
        self.model = settings.ollama_model

    def ask(
        self,
        prompt: str,
        system: str = "You are a helpful AI assistant.",
    ) -> Optional[str]:
        """Return the model response, or None if Ollama is unavailable."""
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

            content = response.get("message", {}).get("content", "")
            return content.strip() if content else None

        except Exception:
            return None

    def is_available(self) -> bool:
        """Return whether the local Ollama service can be reached."""
        try:
            self.client.list()
            return True
        except Exception:
            return False