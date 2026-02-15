"""
The Librarian — Topic Router (Phase 8)

Detects and assigns topics to entries. Topics emerge organically from
content — no manual curation needed. Uses three inference strategies:

1. Tag aggregation: entry tags → match existing topic labels
2. Embedding clustering: cosine similarity to topic embeddings
3. Fallback creation: if no match, create a new topic from tags

Topics are stored in the DB with their own embeddings for efficient
query-time routing.
"""
import json
import uuid
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import numpy as np

from ..core.types import RolodexEntry, LibrarianQuery
from ..storage.schema import serialize_embedding, deserialize_embedding


class TopicRouter:
    """
    Routes entries to emergent topics.
    Topics are created on-the-fly as new clusters of knowledge emerge.
    """

    def __init__(
        self,
        conn,
        embedding_manager=None,
        confidence_threshold: float = 0.65,
        merge_threshold: float = 0.85,
        min_tags_for_topic: int = 2,
    ):
        """
        Args:
            conn: SQLite connection (shared with Rolodex)
            embedding_manager: EmbeddingManager instance for computing similarities
            confidence_threshold: Min confidence to assign a topic (0-1)
            merge_threshold: Similarity above which two topics auto-merge
            min_tags_for_topic: Minimum meaningful tags before creating a topic
        """
        self.conn = conn
        self.embeddings = embedding_manager
        self.confidence_threshold = confidence_threshold
        self.merge_threshold = merge_threshold
        self.min_tags_for_topic = min_tags_for_topic

        # In-memory cache of topic embeddings for fast clustering
        self._topic_cache: Dict[str, Dict] = {}
        self._cache_loaded = False

    # ─── Topic Inference ─────────────────────────────────────────────────

    async def infer_topic(
        self,
        entry: RolodexEntry,
    ) -> Optional[str]:
        """
        Infer the primary topic for an entry.
        Returns topic_id if a match is found or created, None otherwise.

        Strategy order:
        1. Tag-based: match entry tags against existing topic labels
        2. Embedding-based: cluster by similarity to topic embeddings
        3. Fallback: create new topic from entry tags if enough signal
        """
        self._ensure_cache_loaded()

        # Strategy 1: Tag-based inference
        topic_id = self._infer_from_tags(entry)
        if topic_id:
            self._record_assignment(entry.id, topic_id, 0.8, "tag_inference")
            return topic_id

        # Strategy 2: Embedding-based clustering
        if entry.embedding:
            topic_id, confidence = self._cluster_by_embedding(entry)
            if topic_id and confidence >= self.confidence_threshold:
                self._record_assignment(entry.id, topic_id, confidence, "embedding_clustering")
                return topic_id

        # Strategy 3: Create new topic from tags
        meaningful_tags = [
            t for t in entry.tags
            if t not in ("pending-enrichment", "prose", "code", "conversational")
            and len(t) > 2
        ]
        if len(meaningful_tags) >= self.min_tags_for_topic:
            topic_id = await self._create_topic_from_tags(
                meaningful_tags, entry
            )
            if topic_id:
                self._record_assignment(entry.id, topic_id, 0.7, "tag_creation")
                return topic_id

        return None

    async def infer_topic_for_query(
        self,
        query: LibrarianQuery,
    ) -> Optional[str]:
        """
        Infer which topic a query is most relevant to.
        Used at search time to scope results to the right namespace.
        If the matched topic is a parent, returns the parent ID
        (use get_topic_group() to expand to children).
        """
        self._ensure_cache_loaded()

        if not self._topic_cache:
            return None

        # Try keyword match first — prefer parent topics for broader recall
        query_lower = query.query_text.lower()
        best_keyword_match = None
        best_keyword_is_parent = False
        for topic_id, topic_data in self._topic_cache.items():
            label_lower = topic_data["label"].lower()
            if label_lower in query_lower or query_lower in label_lower:
                is_parent = topic_data.get("is_parent", False)
                # Prefer parent matches over child matches
                if is_parent or not best_keyword_match:
                    best_keyword_match = topic_id
                    best_keyword_is_parent = is_parent
                    if is_parent:
                        break  # Parent match is ideal, stop looking

        if best_keyword_match:
            return best_keyword_match

        # Try embedding similarity
        if hasattr(query, 'query_embedding') and getattr(query, 'query_embedding', None):
            query_emb = query.query_embedding
        elif self.embeddings:
            query_emb = await self.embeddings.embed_text(query.query_text)
        else:
            return None

        best_topic = None
        best_sim = 0.0
        for topic_id, topic_data in self._topic_cache.items():
            if topic_data.get("embedding"):
                sim = self._cosine_sim(query_emb, topic_data["embedding"])
                if sim > best_sim:
                    best_sim = sim
                    best_topic = topic_id

        if best_sim >= self.confidence_threshold:
            return best_topic

        return None

    def get_topic_group(self, topic_id: str) -> List[str]:
        """
        Given a topic ID, return all topic IDs in its group.
        If it's a parent: returns [parent_id] + all child IDs.
        If it's a child: returns [parent_id] + all sibling IDs.
        If it's standalone: returns [topic_id].
        """
        self._ensure_cache_loaded()

        topic_data = self._topic_cache.get(topic_id)
        if not topic_data:
            return [topic_id]

        # Check if this is a parent (has children)
        children = self._get_children(topic_id)
        if children:
            return [topic_id] + children

        # Check if this is a child (has a parent)
        parent_id = topic_data.get("parent_topic_id")
        if parent_id:
            siblings = self._get_children(parent_id)
            return [parent_id] + siblings

        return [topic_id]

    def _get_children(self, parent_id: str) -> List[str]:
        """Get all child topic IDs for a parent."""
        rows = self.conn.execute(
            "SELECT id FROM topics WHERE parent_topic_id = ?",
            (parent_id,)
        ).fetchall()
        return [row["id"] for row in rows]

    # ─── Topic CRUD ──────────────────────────────────────────────────────

    async def create_topic(
        self,
        label: str,
        description: str = "",
        seed_entries: Optional[List[RolodexEntry]] = None,
    ) -> str:
        """
        Create a new topic. Computes topic embedding from seed entries
        if provided. Returns topic_id.
        """
        topic_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        # Compute topic embedding from seed entries
        embedding_blob = None
        embedding_list = None
        if seed_entries and self.embeddings:
            entry_embeddings = [
                e.embedding for e in seed_entries
                if e.embedding
            ]
            if entry_embeddings:
                # Average the embeddings
                avg = np.mean(entry_embeddings, axis=0)
                norm = np.linalg.norm(avg)
                if norm > 0:
                    avg = avg / norm
                embedding_list = avg.tolist()
                embedding_blob = serialize_embedding(embedding_list)

        self.conn.execute(
            """INSERT INTO topics
               (id, label, description, created_at, last_updated,
                entry_count, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (topic_id, label, description, now, now,
             len(seed_entries) if seed_entries else 0,
             embedding_blob)
        )
        self.conn.execute(
            "INSERT INTO topics_fts (topic_id, label, description) VALUES (?, ?, ?)",
            (topic_id, label, description)
        )
        self.conn.commit()

        # Update cache
        self._topic_cache[topic_id] = {
            "label": label,
            "description": description,
            "embedding": embedding_list,
            "entry_count": len(seed_entries) if seed_entries else 0,
        }

        return topic_id

    def list_topics(self, limit: int = 50, min_entries: int = 0) -> List[Dict]:
        """List all topics with metadata."""
        sql = "SELECT * FROM topics"
        params = []
        if min_entries > 0:
            sql += " WHERE entry_count >= ?"
            params.append(min_entries)
        sql += " ORDER BY entry_count DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        return [
            {
                "id": row["id"],
                "label": row["label"],
                "description": row["description"],
                "parent_topic_id": row["parent_topic_id"],
                "entry_count": row["entry_count"],
                "created_at": row["created_at"],
                "last_updated": row["last_updated"],
            }
            for row in rows
        ]

    def get_topic(self, topic_id: str) -> Optional[Dict]:
        """Fetch a single topic by ID."""
        row = self.conn.execute(
            "SELECT * FROM topics WHERE id = ?", (topic_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "label": row["label"],
            "description": row["description"],
            "parent_topic_id": row["parent_topic_id"],
            "entry_count": row["entry_count"],
            "created_at": row["created_at"],
            "last_updated": row["last_updated"],
        }

    def get_entries_for_topic(self, topic_id: str, limit: int = 50) -> List[str]:
        """Get entry IDs assigned to a topic."""
        rows = self.conn.execute(
            """SELECT entry_id FROM topic_assignments
               WHERE topic_id = ?
               ORDER BY assigned_at DESC LIMIT ?""",
            (topic_id, limit)
        ).fetchall()
        return [row["entry_id"] for row in rows]

    def count_topics(self) -> int:
        """Count total topics."""
        return self.conn.execute("SELECT COUNT(*) as cnt FROM topics").fetchone()["cnt"]

    def count_unassigned_entries(self) -> int:
        """Count entries without a topic assignment."""
        return self.conn.execute(
            """SELECT COUNT(*) as cnt FROM rolodex_entries
               WHERE topic_id IS NULL"""
        ).fetchone()["cnt"]

    def merge_topics(self, source_id: str, target_id: str) -> int:
        """
        Merge source topic into target. Reassigns all entries,
        sets parent_topic_id on source, updates counts.
        Returns number of entries reassigned.
        """
        # Reassign entries
        self.conn.execute(
            "UPDATE rolodex_entries SET topic_id = ? WHERE topic_id = ?",
            (target_id, source_id)
        )
        self.conn.execute(
            "UPDATE topic_assignments SET topic_id = ? WHERE topic_id = ?",
            (target_id, source_id)
        )
        # Update source topic parent
        self.conn.execute(
            "UPDATE topics SET parent_topic_id = ? WHERE id = ?",
            (target_id, source_id)
        )
        # Recalculate target entry count
        count = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM topic_assignments WHERE topic_id = ?",
            (target_id,)
        ).fetchone()["cnt"]
        self.conn.execute(
            "UPDATE topics SET entry_count = ?, last_updated = ? WHERE id = ?",
            (count, datetime.utcnow().isoformat(), target_id)
        )
        self.conn.commit()

        # Refresh cache
        self._cache_loaded = False
        return count

    # ─── Internal Strategies ─────────────────────────────────────────────

    def _infer_from_tags(self, entry: RolodexEntry) -> Optional[str]:
        """Match entry tags against existing topic labels."""
        if not entry.tags or not self._topic_cache:
            return None

        entry_tags_lower = {t.lower() for t in entry.tags}

        best_match = None
        best_overlap = 0

        for topic_id, topic_data in self._topic_cache.items():
            label_words = set(topic_data["label"].lower().split())
            overlap = len(entry_tags_lower & label_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = topic_id

        # Require at least 1 tag overlap
        if best_overlap >= 1:
            return best_match

        return None

    def _cluster_by_embedding(
        self, entry: RolodexEntry
    ) -> Tuple[Optional[str], float]:
        """Find closest topic by embedding similarity."""
        if not entry.embedding or not self._topic_cache:
            return None, 0.0

        best_topic = None
        best_sim = 0.0

        for topic_id, topic_data in self._topic_cache.items():
            if topic_data.get("embedding"):
                sim = self._cosine_sim(entry.embedding, topic_data["embedding"])
                if sim > best_sim:
                    best_sim = sim
                    best_topic = topic_id

        return best_topic, best_sim

    async def _create_topic_from_tags(
        self, tags: List[str], entry: RolodexEntry
    ) -> Optional[str]:
        """Create a new topic from an entry's tags."""
        # Build label from most descriptive tags
        label = " ".join(sorted(tags[:4]))  # Cap at 4 tags for label

        # Check if similar topic already exists
        existing = self.conn.execute(
            "SELECT id FROM topics WHERE label = ?", (label,)
        ).fetchone()
        if existing:
            return existing["id"]

        # Create new topic with this entry as seed
        return await self.create_topic(
            label=label,
            description=f"Auto-created from entry tags: {', '.join(tags)}",
            seed_entries=[entry] if entry.embedding else None,
        )

    # ─── Assignment Recording ────────────────────────────────────────────

    def _record_assignment(
        self,
        entry_id: str,
        topic_id: str,
        confidence: float,
        source: str,
    ) -> None:
        """Record a topic assignment and update entry's topic_id."""
        now = datetime.utcnow().isoformat()

        # Record in assignment log
        self.conn.execute(
            """INSERT INTO topic_assignments
               (id, entry_id, topic_id, confidence, source, assigned_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), entry_id, topic_id, confidence, source, now)
        )

        # Update entry's topic_id
        self.conn.execute(
            "UPDATE rolodex_entries SET topic_id = ? WHERE id = ?",
            (topic_id, entry_id)
        )

        # Increment topic entry count
        self.conn.execute(
            """UPDATE topics SET entry_count = entry_count + 1,
               last_updated = ? WHERE id = ?""",
            (now, topic_id)
        )
        self.conn.commit()

        # Update cache
        if topic_id in self._topic_cache:
            self._topic_cache[topic_id]["entry_count"] = \
                self._topic_cache[topic_id].get("entry_count", 0) + 1

    # ─── Cache Management ────────────────────────────────────────────────

    def _ensure_cache_loaded(self) -> None:
        """Lazy-load topic cache from DB."""
        if self._cache_loaded:
            return

        self._topic_cache.clear()
        rows = self.conn.execute(
            "SELECT * FROM topics"
        ).fetchall()

        # Collect parent IDs for is_parent detection
        parent_ids = set()
        for row in rows:
            if row["parent_topic_id"]:
                parent_ids.add(row["parent_topic_id"])

        for row in rows:
            embedding = None
            if row["embedding"]:
                embedding = deserialize_embedding(row["embedding"])
            self._topic_cache[row["id"]] = {
                "label": row["label"],
                "description": row["description"],
                "embedding": embedding,
                "entry_count": row["entry_count"],
                "parent_topic_id": row["parent_topic_id"],
                "is_parent": row["id"] in parent_ids,
            }

        self._cache_loaded = True

    def invalidate_cache(self) -> None:
        """Force cache reload on next access."""
        self._cache_loaded = False

    # ─── Utilities ───────────────────────────────────────────────────────

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
