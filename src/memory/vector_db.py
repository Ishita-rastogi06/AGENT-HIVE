"""Persistent Chroma vector database for AgentHive memories."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb

from src.config import settings
from src.memory.embedder import OllamaEmbedder


class MemoryVectorStore:
    """Save and retrieve semantic memories using Chroma and Ollama embeddings."""

    COLLECTION_NAME = "agenthive_memories"

    def __init__(self) -> None:
        settings.ensure_runtime_directories()

        storage_path: Path = settings.data_dir / "chroma_memory"
        self.client = chromadb.PersistentClient(path=str(storage_path))

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=None,
            metadata={"description": "Long-term AgentHive task memories"},
        )
        self.embedder = OllamaEmbedder()

    def save_memory(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Embed and permanently save one memory.

        Returns:
            The unique ID assigned to the saved memory.
        """
        clean_text = text.strip()

        if not clean_text:
            raise ValueError("Memory text cannot be empty.")

        memory_id = str(uuid4())
        memory_metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }

        self.collection.upsert(
            ids=[memory_id],
            documents=[clean_text],
            embeddings=[self.embedder.embed_one(clean_text)],
            metadatas=[memory_metadata],
        )

        return memory_id

    def search_memories(
        self,
        query: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Find the most semantically similar saved memories.

        Lower distance means a closer match.
        """
        clean_query = query.strip()

        if not clean_query or self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[self.embedder.embed_one(clean_query)],
            n_results=min(limit, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        memories: list[dict[str, Any]] = []

        for memory_id, document, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            memories.append(
                {
                    "id": memory_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": round(float(distance), 4),
                }
            )

        return memories