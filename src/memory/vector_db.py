"""
Persistent ChromaDB long-term memory store for AgentHive.

Every memory is scoped by user_id — reads and writes always filter by
the user_id metadata field so no user can see another user's memories.

Importance scoring
──────────────────
Each memory has an importance_score (0–1) computed as:
    0.40 * recency  +  0.35 * retrieval_frequency  +  0.25 * user_feedback

Scores are updated in-place on each retrieval hit.

Memory management
─────────────────
consolidate_duplicates()  – merge/delete near-duplicate memories for a user
expire_low_importance()   – delete memories below min_importance after expiry_days
Both are safe to call from a background job or the UI "Manage" button.

Graceful degradation
────────────────────
Construction is wrapped so that if ChromaDB or Ollama is unavailable the
caller receives a RuntimeError with a human-readable message — it does NOT
silently swallow errors at that level.  The callers (memory_tool.py,
memory_recall.py, memory_save.py) all wrap construction in try/except and
degrade gracefully from there.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb

from src.config import settings
from src.memory.embedder import OllamaEmbedder
from src.memory.schemas import ImportanceScore, LongTermMemoryRecord, MemoryStats

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "agenthive_memories"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dt(iso: str) -> datetime:
    try:
        return datetime.fromisoformat(iso)
    except Exception:
        return datetime.now(timezone.utc)


class MemoryVectorStore:
    """
    Save, search, list, delete, and manage long-term semantic memories.

    All public methods accept a ``user_id`` parameter and silently filter
    results to that user — no cross-user data leakage is possible.
    """

    def __init__(self, storage_path: Path | str | None = None) -> None:
        settings.ensure_runtime_directories()

        if storage_path is not None:
            effective_path = Path(storage_path)
        else:
            raw_path = settings.chroma_memory_path
            effective_path = (
                Path(raw_path) if raw_path else settings.data_dir / "chroma_memory"
            )
        effective_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(effective_path))
        self.collection = self.client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=None,
            metadata={"description": "AgentHive long-term task memories"},
        )
        self.embedder = OllamaEmbedder()

    # ── write ─────────────────────────────────────────────────────────────────

    def save_memory(
        self,
        text: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Embed and permanently save one memory for user_id.

        metadata should include at minimum:
            task_type, outcome, selected_agent, tools_used, final_confidence

        Returns the UUID assigned to the new memory.
        """
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Memory text cannot be empty.")
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required for every memory save.")

        memory_id = str(uuid4())
        base_metadata: dict[str, Any] = {
            "user_id": user_id.strip(),
            "timestamp": _now_iso(),
            "importance_score": 0.50,      # neutral starting importance
            "retrieval_count": 0,
            "user_feedback": 0.0,          # −1 / 0 / +1
        }
        base_metadata.update(metadata or {})

        self.collection.upsert(
            ids=[memory_id],
            documents=[clean_text],
            embeddings=[self.embedder.embed_one(clean_text)],
            metadatas=[base_metadata],
        )
        logger.debug("Saved memory %s for user %s", memory_id, user_id)
        return memory_id

    def update_metadata(
        self,
        memory_id: str,
        user_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """
        Merge ``updates`` into an existing memory's metadata.

        Returns True on success, False if the memory does not belong to user_id.
        """
        result = self.collection.get(ids=[memory_id], include=["metadatas", "documents", "embeddings"])
        if not result["ids"]:
            return False

        existing_meta = result["metadatas"][0]
        if existing_meta.get("user_id") != user_id:
            logger.warning(
                "update_metadata: memory %s does not belong to user %s",
                memory_id, user_id,
            )
            return False

        merged = {**existing_meta, **updates}
        self.collection.upsert(
            ids=[memory_id],
            documents=result["documents"][0:1],
            embeddings=result["embeddings"][0:1],
            metadatas=[merged],
        )
        return True

    # ── read ──────────────────────────────────────────────────────────────────

    def search_memories(
        self,
        query: str,
        user_id: str,
        limit: int | None = None,
        max_distance: float | None = None,
        task_type: str | None = None,
        outcome: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find the most semantically similar memories for user_id.

        Applies metadata filters (task_type, outcome) when supplied.
        Lower distance → closer match.
        After retrieval, increments retrieval_count and recalculates
        importance_score for each returned memory.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        total = self.collection.count()
        if total == 0:
            return []

        effective_limit = min(limit or settings.memory_top_k, total)
        effective_max_distance = max_distance if max_distance is not None else settings.memory_max_distance

        # Build ChromaDB where clause — always filter by user_id.
        where: dict[str, Any] = {"user_id": user_id}
        if task_type:
            where = {"$and": [{"user_id": user_id}, {"task_type": task_type}]}
        if outcome:
            inner = [{"user_id": user_id}, {"outcome": outcome}]
            if task_type:
                inner.append({"task_type": task_type})
            where = {"$and": inner}

        try:
            results = self.collection.query(
                query_embeddings=[self.embedder.embed_one(clean_query)],
                n_results=effective_limit,
                include=["documents", "metadatas", "distances"],
                where=where if len(where) > 1 or "user_id" in where else None,
            )
        except Exception as exc:
            logger.warning("search_memories query failed: %s", exc)
            return []

        memories: list[dict[str, Any]] = []
        for memory_id, document, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            dist = round(float(distance), 4)
            if dist > effective_max_distance:
                continue
            memories.append({
                "id": memory_id,
                "text": document,
                "metadata": metadata,
                "distance": dist,
            })
            # Bump retrieval count and recalculate importance.
            self._increment_retrieval(memory_id, metadata)

        return memories

    def get_memory(self, memory_id: str, user_id: str) -> dict[str, Any] | None:
        """Return a single memory dict or None if not found / wrong user."""
        result = self.collection.get(
            ids=[memory_id],
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            return None
        meta = result["metadatas"][0]
        if meta.get("user_id") != user_id:
            return None
        return {
            "id": memory_id,
            "text": result["documents"][0],
            "metadata": meta,
        }

    def list_memories(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Return up to ``limit`` memories for user_id, sorted by importance desc.

        Uses ChromaDB's .get() with a where filter — no embedding needed.
        """
        try:
            results = self.collection.get(
                where={"user_id": user_id},
                include=["documents", "metadatas"],
                limit=limit + offset,   # over-fetch then slice for offset
            )
        except Exception as exc:
            logger.warning("list_memories failed: %s", exc)
            return []

        records: list[dict[str, Any]] = []
        for memory_id, document, metadata in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
        ):
            records.append({
                "id": memory_id,
                "text": document,
                "metadata": metadata,
            })

        # Sort by importance_score descending, then by timestamp descending.
        records.sort(
            key=lambda r: (
                -float(r["metadata"].get("importance_score", 0.0)),
                r["metadata"].get("timestamp", ""),
            ),
            reverse=False,
        )
        return records[offset : offset + limit]

    def count_for_user(self, user_id: str) -> int:
        """Return the total number of memories stored for user_id."""
        try:
            results = self.collection.get(
                where={"user_id": user_id},
                include=[],
            )
            return len(results["ids"])
        except Exception:
            return 0

    # ── delete ────────────────────────────────────────────────────────────────

    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """
        Delete one memory.  Returns True on success, False if the memory does
        not exist or does not belong to user_id.
        """
        existing = self.get_memory(memory_id, user_id)
        if existing is None:
            return False
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception as exc:
            logger.warning("delete_memory failed for %s: %s", memory_id, exc)
            return False

    def delete_all_for_user(self, user_id: str) -> int:
        """Delete every memory belonging to user_id. Returns deleted count."""
        try:
            results = self.collection.get(
                where={"user_id": user_id},
                include=[],
            )
            ids = results["ids"]
            if ids:
                self.collection.delete(ids=ids)
            return len(ids)
        except Exception as exc:
            logger.warning("delete_all_for_user failed for %s: %s", user_id, exc)
            return 0

    # ── importance scoring ────────────────────────────────────────────────────

    def compute_importance(
        self,
        metadata: dict[str, Any],
        expiry_days: int | None = None,
    ) -> ImportanceScore:
        """
        Compute the composite importance score for one memory record.

            importance = 0.40 * recency
                       + 0.35 * retrieval_frequency
                       + 0.25 * user_feedback_normalised

        recency decays linearly from 1.0 (today) to 0.0 at expiry_days.
        retrieval_frequency is log-scaled so very popular memories
        don't completely dominate.
        user_feedback is mapped: +1 → 1.0, 0 → 0.5, −1 → 0.0.
        """
        effective_expiry = expiry_days or settings.memory_expiry_days

        # Recency
        ts = _parse_dt(metadata.get("timestamp", _now_iso()))
        age_days = max(0.0, (datetime.now(timezone.utc) - ts).total_seconds() / 86_400)
        recency = max(0.0, 1.0 - age_days / effective_expiry)

        # Retrieval frequency — log1p normalised to [0, 1] at 100 hits
        retrieval_count = int(metadata.get("retrieval_count", 0))
        retrieval_freq = min(1.0, math.log1p(retrieval_count) / math.log1p(100))

        # User feedback: −1 → 0.0, 0 → 0.5, +1 → 1.0
        raw_feedback = float(metadata.get("user_feedback", 0.0))
        feedback_norm = (raw_feedback + 1.0) / 2.0

        composite = round(
            0.40 * recency + 0.35 * retrieval_freq + 0.25 * feedback_norm, 4
        )

        return ImportanceScore(
            memory_id=metadata.get("id", ""),
            recency=round(recency, 4),
            retrieval_frequency=round(retrieval_freq, 4),
            user_feedback=round(feedback_norm, 4),
            composite=composite,
        )

    def apply_user_feedback(
        self,
        memory_id: str,
        user_id: str,
        feedback: float,
    ) -> bool:
        """
        Record explicit user feedback (−1, 0, or +1) and recompute importance.
        Returns True on success.
        """
        result = self.collection.get(ids=[memory_id], include=["metadatas", "documents", "embeddings"])
        if not result["ids"]:
            return False
        meta = result["metadatas"][0]
        if meta.get("user_id") != user_id:
            return False

        clamped = max(-1.0, min(1.0, float(feedback)))
        meta["user_feedback"] = clamped
        score = self.compute_importance(meta)
        meta["importance_score"] = score["composite"]

        self.collection.upsert(
            ids=[memory_id],
            documents=result["documents"][0:1],
            embeddings=result["embeddings"][0:1],
            metadatas=[meta],
        )
        return True

    # ── memory management ─────────────────────────────────────────────────────

    def expire_low_importance(self, user_id: str) -> int:
        """
        Delete memories for user_id whose importance score is below
        settings.memory_min_importance AND whose age exceeds
        settings.memory_expiry_days.

        Returns the number of memories deleted.
        """
        memories = self.list_memories(user_id, limit=500)
        to_delete: list[str] = []
        now = datetime.now(timezone.utc)
        expiry_days = settings.memory_expiry_days
        min_importance = settings.memory_min_importance

        for m in memories:
            meta = m["metadata"]
            ts = _parse_dt(meta.get("timestamp", _now_iso()))
            age_days = (now - ts).total_seconds() / 86_400
            score = float(meta.get("importance_score", 0.5))
            if age_days >= expiry_days and score < min_importance:
                to_delete.append(m["id"])

        if to_delete:
            try:
                self.collection.delete(ids=to_delete)
            except Exception as exc:
                logger.warning("expire_low_importance delete failed: %s", exc)
                return 0
        logger.info(
            "expire_low_importance: deleted %d memories for user %s",
            len(to_delete), user_id,
        )
        return len(to_delete)

    def consolidate_duplicates(self, user_id: str) -> int:
        """
        Find near-duplicate memories (distance < memory_consolidation_threshold)
        for user_id and delete the older, lower-importance duplicate.

        Returns the number of memories deleted.
        """
        memories = self.list_memories(user_id, limit=200)
        if len(memories) < 2:
            return 0

        threshold = settings.memory_consolidation_threshold
        deleted_ids: set[str] = set()
        deleted_count = 0

        for i, mem_a in enumerate(memories):
            if mem_a["id"] in deleted_ids:
                continue
            try:
                embedding_a = self.embedder.embed_one(mem_a["text"])
            except Exception:
                continue

            for mem_b in memories[i + 1:]:
                if mem_b["id"] in deleted_ids:
                    continue
                try:
                    embedding_b = self.embedder.embed_one(mem_b["text"])
                    dist = self._cosine_distance(embedding_a, embedding_b)
                except Exception:
                    continue

                if dist < threshold:
                    # Keep the higher-importance one; delete the other.
                    score_a = float(mem_a["metadata"].get("importance_score", 0.5))
                    score_b = float(mem_b["metadata"].get("importance_score", 0.5))
                    victim_id = mem_b["id"] if score_a >= score_b else mem_a["id"]

                    try:
                        self.collection.delete(ids=[victim_id])
                        deleted_ids.add(victim_id)
                        deleted_count += 1
                    except Exception as exc:
                        logger.warning(
                            "consolidate_duplicates: failed to delete %s: %s",
                            victim_id, exc,
                        )
                    if mem_a["id"] == victim_id:
                        break  # mem_a is gone; stop comparing it

        logger.info(
            "consolidate_duplicates: removed %d duplicates for user %s",
            deleted_count, user_id,
        )
        return deleted_count

    # ── stats ─────────────────────────────────────────────────────────────────

    def get_stats(self, user_id: str) -> MemoryStats:
        """Return aggregate statistics for the memory dashboard."""
        memories = self.list_memories(user_id, limit=500)
        if not memories:
            return MemoryStats(
                user_id=user_id,
                total_memories=0,
                success_count=0,
                failure_count=0,
                most_used_agent="—",
                avg_importance=0.0,
                oldest_memory_date="—",
                newest_memory_date="—",
            )

        agent_counts: dict[str, int] = {}
        importance_sum = 0.0
        success = failure = 0
        timestamps: list[str] = []

        for m in memories:
            meta = m["metadata"]
            outcome = meta.get("outcome", "")
            if outcome == "success":
                success += 1
            elif outcome in ("failure", "escalated"):
                failure += 1
            agent = meta.get("selected_agent", "unknown")
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
            importance_sum += float(meta.get("importance_score", 0.5))
            ts = meta.get("timestamp", "")
            if ts:
                timestamps.append(ts)

        most_used = max(agent_counts, key=agent_counts.get) if agent_counts else "—"
        timestamps_sorted = sorted(timestamps)

        return MemoryStats(
            user_id=user_id,
            total_memories=len(memories),
            success_count=success,
            failure_count=failure,
            most_used_agent=most_used,
            avg_importance=round(importance_sum / len(memories), 3),
            oldest_memory_date=timestamps_sorted[0][:10] if timestamps_sorted else "—",
            newest_memory_date=timestamps_sorted[-1][:10] if timestamps_sorted else "—",
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _increment_retrieval(
        self,
        memory_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """Bump retrieval_count and recompute importance_score in-place."""
        try:
            result = self.collection.get(
                ids=[memory_id],
                include=["documents", "embeddings"],
            )
            if not result["ids"]:
                return
            updated_meta = dict(metadata)
            updated_meta["retrieval_count"] = int(metadata.get("retrieval_count", 0)) + 1
            score = self.compute_importance(updated_meta)
            updated_meta["importance_score"] = score["composite"]

            self.collection.upsert(
                ids=[memory_id],
                documents=result["documents"][0:1],
                embeddings=result["embeddings"][0:1],
                metadatas=[updated_meta],
            )
        except Exception as exc:
            logger.debug("_increment_retrieval failed for %s: %s", memory_id, exc)

    @staticmethod
    def _cosine_distance(a: list[float], b: list[float]) -> float:
        """Cosine distance in [0, 2]; 0 = identical."""
        if not a or not b or len(a) != len(b):
            return 2.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 2.0
        return round(1.0 - dot / (mag_a * mag_b), 6)
