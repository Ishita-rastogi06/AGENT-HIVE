"""Small, application-specific wrapper around the local Ollama client."""
from typing import Optional
from ollama import Client
from src.config import settings

class OllamaClient:
    def __init__(self) -> None:
        self.client = Client(host=settings.ollama_host)
        self.model = settings.ollama_model

    def ask(self, prompt: str, system: str = "You are a helpful AI assistant.") -> Optional[str]:
        """Return a model response, or None when the local service is unavailable."""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.2},
            )
            return response["message"]["content"].strip()
        except Exception:
            return None

    def is_available(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception:
            return False
