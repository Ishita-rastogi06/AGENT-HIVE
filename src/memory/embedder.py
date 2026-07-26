"""Create vector embeddings using the local Ollama embedding model."""

from ollama import Client

from src.config import settings


class OllamaEmbedder:
    """Convert text into embeddings using Ollama."""

    def __init__(self) -> None:
        self.client = Client(host=settings.ollama_host)
        self.model = settings.embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Create one vector embedding for every supplied text.

        Raises:
            RuntimeError: When Ollama or the embedding model is unavailable.
        """
        clean_texts = [text.strip() for text in texts if text.strip()]

        if not clean_texts:
            return []

        try:
            response = self.client.embed(
                model=self.model,
                input=clean_texts,
            )
            return response["embeddings"]
        except Exception as error:
            raise RuntimeError(
                "Could not create memory embeddings. Make sure Ollama is running "
                f"and the `{self.model}` model has been downloaded."
            ) from error

    def embed_one(self, text: str) -> list[float]:
        """Create an embedding for a single piece of text."""
        embeddings = self.embed([text])
        return embeddings[0] if embeddings else []