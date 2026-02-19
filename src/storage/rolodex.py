"""
The Librarian — Rolodex Storage Layer
SQLite-backed storage with full-text search and vector similarity.
All content stored verbatim. No compression, no summarization.
"""
import sqlite3
import json
import uuid
import time
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from collections import OrderedDict
from ..core.types import (
    RolodexEntry, ContentModality, EntryCategory, Tier,
    TierEvent, estimate_tokens, compute_importance_score,
    ReasoningChain
)
from .schema import (
    init_database, serialize_entry, deserialize_entry,
    serialize_embedding, deserialize_embedding
)
class Rolodex:
    """
    The Librarian's indexed catalog of all conversation content.
    Implements CRUD operations, full-text keyword search, and
    vector similarity search.
    """
    def __init__(self, db_path: str = "rolodex.db"):
        self.db_path = db_path
        self.conn = init_database(db_path)
        self._hot_cache: OrderedDict[str, RolodexEntry] = OrderedDict()
        self._hot_cache_max = 50
    # ─── Write Operations ────────────────────────────────────────────────
    def create_entry(self, entry: RolodexEntry) -> str:
        """Store a new entry. Returns the entry ID."""
        values = serialize_entry(entry)
        self.conn.execute(
            """INSERT INTO rolodex_entries
               (id, conversation_id, content, content_type, category,
                tags, source_range, access_count, last_accessed,
                created_at, tier, embedding, linked_ids, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            values
        )
        # Update FTS index
        self.conn.execute(
            "INSERT INTO rolodex_fts (entry_id, content, tags, category) VALUES (?, ?, ?, ?)",
            (entry.id, entry.content, json.dumps(entry.tags), entry.category.value)
        )
        # Write verbatim_source flag (migration-added column)
        self.conn.execute(
            "UPDATE rolodex_entries SET verbatim_source = ? WHERE id = ?",
            (1 if entry.verbatim_source else 0, entry.id)
        )
        self.conn.commit()
        # Add to hot cache (most recent entries are likely to be needed)
        self._cache_put(entry)
        return entry.id
    def batch_create_entries(self, entries: List[RolodexEntry]) -> List[str]:
        """Bulk insert entries. Returns list of IDs."""
        ids = []
        for entry in entries:
            values = serialize_entry(entry)
            self.conn.execute(
                """INSERT INTO rolodex_entries
                   (id, conversation_id, content, content_type, category,
                    tags, source_range, access_count, last_accessed,
                    created_at, tier, embedding, linked_ids, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                values
            )
            self.conn.execute(
                "INSERT INTO rolodex_fts (entry_id, content, tags, category) VALUES (?, ?, ?, ?)",
                (entry.id, entry.content, json.dumps(entry.tags), entry.category.value)
            )
            self._cache_put(entry)
            ids.append(entry.id)
        self.conn.commit()
        return ids
    def update_access(self, entry_id: str) -> None:
        """Increment access count and update last_accessed timestamp."""
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """UPDATE rolodex_entries
               SET access_count = access_count + 1, last_accessed = ?
               WHERE id = ?""",
            (now, entry_id)
        )
        self.conn.commit()
        # Update hot cache if present — refresh LRU position
        if entry_id in self._hot_cache:
            entry = self._hot_cache[entry_id]
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
            self._hot_cache.move_to_end(entry_id)  # Phase 2: refresh LRU
    # ─── Enrichment Updates (Phase 8) ────────────────────────────────────

    def update_entry_enrichment(
        self,
        entry_id: str,
        content_type: Optional[ContentModality] = None,
        category: Optional[EntryCategory] = None,
        tags: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update a stub entry with enriched data from background processing.
        Only updates non-None fields. Used by IngestionQueue workers.
        """
        updates = []
        params = []

        if content_type is not None:
            updates.append("content_type = ?")
            params.append(content_type.value)
        if category is not None:
            updates.append("category = ?")
            params.append(category.value)
        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))
            # Also update FTS index
            self.conn.execute(
                "UPDATE rolodex_fts SET tags = ? WHERE entry_id = ?",
                (json.dumps(tags), entry_id)
            )
        if embedding is not None:
            updates.append("embedding = ?")
            params.append(serialize_embedding(embedding))
        if metadata is not None:
            # Merge with existing metadata
            row = self.conn.execute(
                "SELECT metadata FROM rolodex_entries WHERE id = ?",
                (entry_id,)
            ).fetchone()
            if row:
                existing = json.loads(row["metadata"] or "{}")
                existing.update(metadata)
                updates.append("metadata = ?")
                params.append(json.dumps(existing))

        if not updates:
            return

        params.append(entry_id)
        sql = f"UPDATE rolodex_entries SET {', '.join(updates)} WHERE id = ?"
        self.conn.execute(sql, params)

        # Update FTS category if changed
        if category is not None:
            self.conn.execute(
                "UPDATE rolodex_fts SET category = ? WHERE entry_id = ?",
                (category.value, entry_id)
            )

        self.conn.commit()

        # Update hot cache if present
        if entry_id in self._hot_cache:
            entry = self._hot_cache[entry_id]
            if content_type is not None:
                entry.content_type = content_type
            if category is not None:
                entry.category = category
            if tags is not None:
                entry.tags = tags
            if embedding is not None:
                entry.embedding = embedding
            if metadata is not None:
                entry.metadata.update(metadata)

    def update_entry_metadata(
        self,
        entry_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Update only the metadata field of an entry (merge with existing)."""
        self.update_entry_enrichment(entry_id=entry_id, metadata=metadata)

    # ─── Read Operations ─────────────────────────────────────────────────
    def get_entry(self, entry_id: str) -> Optional[RolodexEntry]:
        """Fetch a single entry by ID. Checks hot cache first."""
        # Hot cache check
        if entry_id in self._hot_cache:
            return self._hot_cache[entry_id]
        # Cold storage
        row = self.conn.execute(
            "SELECT * FROM rolodex_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        if row:
            return deserialize_entry(row)
        return None
    def keyword_search(
        self, query: str, limit: int = 5, conversation_id: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> List[Tuple[RolodexEntry, float]]:
        """
        Full-text search via FTS5.
        Returns list of (entry, rank_score) tuples, best first.

        When the default AND query (all terms required) returns nothing,
        falls back to OR (any term matches) for better recall.

        Phase 12: Optional source_type filter ('conversation', 'document', 'user_knowledge').
        """
        # Try full AND query first (FTS5 default: all terms must match)
        results = self._fts_match(query, limit, conversation_id, source_type=source_type)

        # If no results and query has multiple words, try OR fallback
        if not results and ' ' in query.strip():
            fts_operators = {'AND', 'OR', 'NOT', 'NEAR'}
            terms = [
                t for t in query.strip().split()
                if t.upper() not in fts_operators and len(t) > 1
            ]
            if len(terms) > 1:
                or_query = ' OR '.join(terms)
                results = self._fts_match(or_query, limit, conversation_id, source_type=source_type)

        return results

    def _fts_match(
        self, fts_query: str, limit: int, conversation_id: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> List[Tuple[RolodexEntry, float]]:
        """Execute a single FTS5 MATCH query."""
        sql = """
            SELECT re.*, fts.rank
            FROM rolodex_fts fts
            JOIN rolodex_entries re ON re.id = fts.entry_id
            WHERE rolodex_fts MATCH ?
        """
        params: list = [fts_query]
        if conversation_id:
            sql += " AND re.conversation_id = ?"
            params.append(conversation_id)
        if source_type:
            sql += " AND re.source_type = ?"
            params.append(source_type)
        sql += " ORDER BY fts.rank LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        results = []
        for row in rows:
            entry = deserialize_entry(row)
            # FTS5 rank is negative (more negative = better match)
            # Normalize to 0-1 where 1 is best
            score = min(1.0, abs(row["rank"]) / 10.0)
            results.append((entry, score))
        return results
    def semantic_search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        min_similarity: float = 0.3,
        conversation_id: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> List[Tuple[RolodexEntry, float]]:
        """
        Vector similarity search using cosine similarity.
        Returns list of (entry, similarity_score) tuples, best first.

        Phase 12: Optional source_type filter ('conversation', 'document', 'user_knowledge').
        """
        # Fetch all entries with embeddings
        sql = "SELECT * FROM rolodex_entries WHERE embedding IS NOT NULL"
        params: list = []
        if conversation_id:
            sql += " AND conversation_id = ?"
            params.append(conversation_id)
        if source_type:
            sql += " AND source_type = ?"
            params.append(source_type)
        rows = self.conn.execute(sql, params).fetchall()
        # Compute similarities
        scored = []
        for row in rows:
            entry = deserialize_entry(row)
            if entry.embedding:
                sim = _cosine_similarity(query_embedding, entry.embedding)
                if sim >= min_similarity:
                    scored.append((entry, sim))
        # Sort by similarity descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]
    def hybrid_search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        limit: int = 5,
        keyword_weight: float = 0.4,
        semantic_weight: float = 0.6,
        min_similarity: float = 0.3,
        conversation_id: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> List[Tuple[RolodexEntry, float]]:
        """
        Combined keyword + semantic search with configurable weights.
        Returns merged, deduplicated results scored by weighted combination.

        Phase 12: Optional source_type filter ('conversation', 'document', 'user_knowledge').
        """
        keyword_results = []
        semantic_results = []
        # Keyword search
        try:
            keyword_results = self.keyword_search(
                query, limit=limit * 2, conversation_id=conversation_id,
                source_type=source_type,
            )
        except Exception:
            pass  # FTS might fail on some query formats
        # Semantic search (if embedding provided)
        if query_embedding:
            semantic_results = self.semantic_search(
                query_embedding, limit=limit * 2,
                min_similarity=min_similarity,
                conversation_id=conversation_id,
                source_type=source_type,
            )
        # Merge results
        return _merge_search_results(
            keyword_results, semantic_results,
            keyword_weight, semantic_weight, limit
        )
    # ─── Cross-Session Search (Phase 4) ─────────────────────────────────

    def boosted_hybrid_search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        current_session_id: Optional[str] = None,
        boost_factor: float = 1.5,
        limit: int = 5,
        keyword_weight: float = 0.4,
        semantic_weight: float = 0.6,
        min_similarity: float = 0.3,
        source_type: Optional[str] = None,
    ) -> List[Tuple[RolodexEntry, float]]:
        """
        Cross-session hybrid search with current-session boosting.

        Searches ALL entries (no conversation_id filter), then boosts
        scores for entries from the current session. This prioritizes
        recent context while still surfacing knowledge from past sessions.

        Phase 12: Optional source_type filter ('conversation', 'document', 'user_knowledge').
        """
        # Search globally (conversation_id=None)
        results = self.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            limit=limit * 2,  # Over-fetch to account for re-ranking
            keyword_weight=keyword_weight,
            semantic_weight=semantic_weight,
            min_similarity=min_similarity,
            conversation_id=None,  # Search ALL sessions
            source_type=source_type,
        )

        if not current_session_id or boost_factor <= 1.0:
            return results[:limit]

        # Boost current-session entries
        boosted = []
        for entry, score in results:
            if entry.conversation_id == current_session_id:
                boosted.append((entry, score * boost_factor))
            else:
                boosted.append((entry, score))

        # Re-sort by boosted score
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted[:limit]

    # ─── Topic-Scoped Search (Phase 8) ──────────────────────────────────

    def keyword_search_by_topic(
        self, query: str, topic_id: str, limit: int = 5
    ) -> List[Tuple[RolodexEntry, float]]:
        """Keyword search scoped to a specific topic."""
        sql = """
            SELECT re.*, fts.rank
            FROM rolodex_fts fts
            JOIN rolodex_entries re ON re.id = fts.entry_id
            WHERE rolodex_fts MATCH ?
            AND re.topic_id = ?
            ORDER BY fts.rank LIMIT ?
        """
        rows = self.conn.execute(sql, (query, topic_id, limit)).fetchall()
        results = []
        for row in rows:
            entry = deserialize_entry(row)
            score = min(1.0, abs(row["rank"]) / 10.0)
            results.append((entry, score))

        # OR fallback
        if not results and ' ' in query.strip():
            fts_operators = {'AND', 'OR', 'NOT', 'NEAR'}
            terms = [
                t for t in query.strip().split()
                if t.upper() not in fts_operators and len(t) > 1
            ]
            if len(terms) > 1:
                or_query = ' OR '.join(terms)
                rows = self.conn.execute(sql, (or_query, topic_id, limit)).fetchall()
                for row in rows:
                    entry = deserialize_entry(row)
                    score = min(1.0, abs(row["rank"]) / 10.0)
                    results.append((entry, score))

        return results

    def semantic_search_by_topic(
        self,
        query_embedding: List[float],
        topic_id: str,
        limit: int = 5,
        min_similarity: float = 0.3,
    ) -> List[Tuple[RolodexEntry, float]]:
        """Semantic search scoped to a specific topic."""
        rows = self.conn.execute(
            """SELECT * FROM rolodex_entries
               WHERE embedding IS NOT NULL AND topic_id = ?""",
            (topic_id,)
        ).fetchall()
        scored = []
        for row in rows:
            entry = deserialize_entry(row)
            if entry.embedding:
                sim = _cosine_similarity(query_embedding, entry.embedding)
                if sim >= min_similarity:
                    scored.append((entry, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def hybrid_search_by_topic(
        self,
        query: str,
        topic_id: str,
        query_embedding: Optional[List[float]] = None,
        limit: int = 5,
        keyword_weight: float = 0.4,
        semantic_weight: float = 0.6,
        min_similarity: float = 0.3,
    ) -> List[Tuple[RolodexEntry, float]]:
        """Combined keyword + semantic search scoped to a topic."""
        keyword_results = []
        semantic_results = []
        try:
            keyword_results = self.keyword_search_by_topic(
                query, topic_id, limit=limit * 2
            )
        except Exception:
            pass
        if query_embedding:
            semantic_results = self.semantic_search_by_topic(
                query_embedding, topic_id, limit=limit * 2,
                min_similarity=min_similarity,
            )
        return _merge_search_results(
            keyword_results, semantic_results,
            keyword_weight, semantic_weight, limit
        )

    def get_entries_by_category(
        self, category: str, conversation_id: Optional[str] = None, limit: int = 20
    ) -> List[RolodexEntry]:
        """Fetch entries filtered by category."""
        sql = "SELECT * FROM rolodex_entries WHERE category = ?"
        params: list = [category]
        if conversation_id:
            sql += " AND conversation_id = ?"
            params.append(conversation_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [deserialize_entry(row) for row in rows]
    def get_recent_entries(
        self, conversation_id: str, limit: int = 20
    ) -> List[RolodexEntry]:
        """Fetch most recently created entries for a conversation."""
        rows = self.conn.execute(
            """SELECT * FROM rolodex_entries
               WHERE conversation_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (conversation_id, limit)
        ).fetchall()
        return [deserialize_entry(row) for row in rows]
    # ─── Hot Cache ───────────────────────────────────────────────────────
    def get_hot_cache_entries(self) -> List[RolodexEntry]:
        """Return all entries currently in the hot cache."""
        return list(self._hot_cache.values())
    def search_hot_cache(self, query: str) -> List[RolodexEntry]:
        """Simple keyword match against hot cache entries."""
        query_lower = query.lower()
        results = []
        for entry in self._hot_cache.values():
            content_lower = entry.content.lower()
            tags_str = " ".join(entry.tags).lower()
            if query_lower in content_lower or query_lower in tags_str:
                results.append(entry)
        return results
    def _cache_put(self, entry: RolodexEntry):
        """Add entry to hot cache with LRU eviction."""
        self._hot_cache[entry.id] = entry
        self._hot_cache.move_to_end(entry.id)
        while len(self._hot_cache) > self._hot_cache_max:
            self._hot_cache.popitem(last=False)
    # ─── Tier Management (Phase 2) ───────────────────────────────────────

    def evaluate_tier(
        self,
        entry_id: str,
        promotion_threshold: float = 1.0,
        demotion_threshold: float = 0.3,
        recency_half_life: float = 24.0,
        age_boost_half_life: float = 48.0,
    ) -> Tuple[Tier, float]:
        """
        Compute importance score for an entry and return recommended tier.
        Does NOT mutate — caller decides whether to act on the recommendation.
        """
        entry = self.get_entry(entry_id)
        if entry is None:
            return (Tier.COLD, 0.0)
        score = compute_importance_score(
            entry,
            recency_half_life_hours=recency_half_life,
            age_boost_half_life_hours=age_boost_half_life,
        )
        if score >= promotion_threshold:
            return (Tier.HOT, score)
        elif score < demotion_threshold:
            return (Tier.COLD, score)
        else:
            # In the middle band — keep current tier (hysteresis)
            return (entry.tier, score)

    def promote_entry(self, entry_id: str) -> Optional[TierEvent]:
        """Promote an entry to HOT tier. Updates DB and adds to cache."""
        entry = self.get_entry(entry_id)
        if entry is None or entry.tier == Tier.HOT:
            return None
        old_tier = entry.tier
        # Update DB
        self.conn.execute(
            "UPDATE rolodex_entries SET tier = ? WHERE id = ?",
            ("hot", entry_id)
        )
        self.conn.commit()
        # Update in-memory
        entry.tier = Tier.HOT
        self._cache_put(entry)
        score = compute_importance_score(entry)
        return TierEvent(
            entry_id=entry_id,
            old_tier=old_tier,
            new_tier=Tier.HOT,
            score=score,
        )

    def demote_entry(self, entry_id: str) -> Optional[TierEvent]:
        """Demote an entry to COLD tier. Updates DB and removes from cache."""
        entry = self.get_entry(entry_id)
        if entry is None or entry.tier == Tier.COLD:
            return None
        # Never demote privileged-tier entries
        if entry.category in (EntryCategory.USER_KNOWLEDGE, EntryCategory.PROJECT_KNOWLEDGE):
            return None
        old_tier = entry.tier
        # Update DB
        self.conn.execute(
            "UPDATE rolodex_entries SET tier = ? WHERE id = ?",
            ("cold", entry_id)
        )
        self.conn.commit()
        # Remove from cache
        if entry_id in self._hot_cache:
            del self._hot_cache[entry_id]
        score = compute_importance_score(entry)
        return TierEvent(
            entry_id=entry_id,
            old_tier=old_tier,
            new_tier=Tier.COLD,
            score=score,
        )

    def run_tier_sweep(
        self,
        promotion_threshold: float = 1.0,
        demotion_threshold: float = 0.3,
        recency_half_life: float = 24.0,
        age_boost_half_life: float = 48.0,
    ) -> Dict[str, Any]:
        """
        Scan all entries, evaluate importance scores, promote/demote as needed.
        Returns a summary of actions taken.
        """
        events: List[TierEvent] = []
        promoted = 0
        demoted = 0
        rows = self.conn.execute("SELECT * FROM rolodex_entries").fetchall()
        for row in rows:
            entry = deserialize_entry(row)
            recommended_tier, score = self.evaluate_tier(
                entry.id,
                promotion_threshold=promotion_threshold,
                demotion_threshold=demotion_threshold,
                recency_half_life=recency_half_life,
                age_boost_half_life=age_boost_half_life,
            )
            if recommended_tier == Tier.HOT and entry.tier == Tier.COLD:
                event = self.promote_entry(entry.id)
                if event:
                    events.append(event)
                    promoted += 1
            elif recommended_tier == Tier.COLD and entry.tier == Tier.HOT:
                event = self.demote_entry(entry.id)
                if event:
                    events.append(event)
                    demoted += 1
        return {
            "entries_scanned": len(rows),
            "promoted": promoted,
            "demoted": demoted,
            "events": events,
        }

    def preload_hot_entries(self) -> int:
        """
        On startup, load all HOT-tier entries from DB into the cache.
        Returns the number of entries loaded.
        """
        rows = self.conn.execute(
            "SELECT * FROM rolodex_entries WHERE tier = 'hot' ORDER BY access_count DESC"
        ).fetchall()
        loaded = 0
        for row in rows:
            entry = deserialize_entry(row)
            self._cache_put(entry)
            loaded += 1
        return loaded

    # ─── Stats & Management ──────────────────────────────────────────────
    def get_stats(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Return database statistics."""
        where = ""
        params: list = []
        if conversation_id:
            where = "WHERE conversation_id = ?"
            params = [conversation_id]
        total = self.conn.execute(
            f"SELECT COUNT(*) as cnt FROM rolodex_entries {where}", params
        ).fetchone()["cnt"]
        categories = {}
        rows = self.conn.execute(
            f"""SELECT category, COUNT(*) as cnt
                FROM rolodex_entries {where}
                GROUP BY category ORDER BY cnt DESC""",
            params
        ).fetchall()
        for row in rows:
            categories[row["category"]] = row["cnt"]
        hot_count = len(self._hot_cache)
        # Tier distribution from DB (persistent tier, not just cache)
        tier_rows = self.conn.execute(
            f"""SELECT tier, COUNT(*) as cnt
                FROM rolodex_entries {where}
                GROUP BY tier""",
            params
        ).fetchall()
        tier_distribution = {}
        for row in tier_rows:
            tier_distribution[row["tier"]] = row["cnt"]
        # Average importance score for HOT entries
        avg_hot_score = 0.0
        hot_entries = self.conn.execute(
            "SELECT * FROM rolodex_entries WHERE tier = 'hot'"
        ).fetchall()
        if hot_entries:
            scores = [
                compute_importance_score(deserialize_entry(r))
                for r in hot_entries
            ]
            avg_hot_score = sum(scores) / len(scores)
        return {
            "total_entries": total,
            "hot_cache_entries": hot_count,
            "cold_storage_entries": total - hot_count,
            "categories": categories,
            "tier_distribution": tier_distribution,
            "avg_hot_score": round(avg_hot_score, 3),
            "db_path": self.db_path,
        }
    def create_conversation(self, conversation_id: str) -> None:
        """Register a new conversation."""
        self.conn.execute(
            "INSERT OR IGNORE INTO conversations (id, created_at) VALUES (?, ?)",
            (conversation_id, datetime.utcnow().isoformat())
        )
        self.conn.commit()
    def log_query(
        self,
        conversation_id: str,
        query_text: str,
        found: bool,
        entry_ids: List[str],
        search_time_ms: float,
        search_type: str
    ) -> None:
        """Log a query for Phase 2 analytics."""
        self.conn.execute(
            """INSERT INTO query_log
               (id, conversation_id, query_text, found, entry_ids,
                search_time_ms, search_type, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()), conversation_id, query_text,
                found, json.dumps(entry_ids), search_time_ms,
                search_type, datetime.utcnow().isoformat()
            )
        )
        self.conn.commit()
    # ─── Chain Operations (Phase 7) ──────────────────────────────────────

    def create_chain(self, chain: ReasoningChain) -> str:
        """Store a new reasoning chain. Returns the chain ID."""
        values = _serialize_chain(chain)
        self.conn.execute(
            """INSERT INTO chains
               (id, session_id, chain_index, turn_range_start, turn_range_end,
                summary, topics, related_entries, embedding, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            values
        )
        self.conn.execute(
            "INSERT INTO chains_fts (chain_id, summary, topics) VALUES (?, ?, ?)",
            (chain.id, chain.summary, json.dumps(chain.topics))
        )
        self.conn.commit()
        return chain.id

    def get_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        """Fetch a chain by ID."""
        row = self.conn.execute(
            "SELECT * FROM chains WHERE id = ?", (chain_id,)
        ).fetchone()
        if row:
            return _deserialize_chain(row)
        return None

    def get_chains_for_session(self, session_id: str) -> List[ReasoningChain]:
        """Get all chains for a session, ordered by chain_index."""
        rows = self.conn.execute(
            "SELECT * FROM chains WHERE session_id = ? ORDER BY chain_index ASC",
            (session_id,)
        ).fetchall()
        return [_deserialize_chain(row) for row in rows]

    def get_chain_by_index(
        self, session_id: str, chain_index: int
    ) -> Optional[ReasoningChain]:
        """Get a specific chain by session + index (for linked-list traversal)."""
        row = self.conn.execute(
            "SELECT * FROM chains WHERE session_id = ? AND chain_index = ?",
            (session_id, chain_index)
        ).fetchone()
        if row:
            return _deserialize_chain(row)
        return None

    def keyword_search_chains(
        self, query: str, limit: int = 5, session_id: Optional[str] = None
    ) -> List[Tuple[ReasoningChain, float]]:
        """Full-text search on chain summaries and topics.
        Falls back to OR query when AND returns nothing."""
        results = self._fts_match_chains(query, limit, session_id)

        # OR fallback for multi-word queries
        if not results and ' ' in query.strip():
            fts_operators = {'AND', 'OR', 'NOT', 'NEAR'}
            terms = [
                t for t in query.strip().split()
                if t.upper() not in fts_operators and len(t) > 1
            ]
            if len(terms) > 1:
                or_query = ' OR '.join(terms)
                results = self._fts_match_chains(or_query, limit, session_id)

        return results

    def _fts_match_chains(
        self, fts_query: str, limit: int, session_id: Optional[str] = None
    ) -> List[Tuple[ReasoningChain, float]]:
        """Execute a single FTS5 MATCH query on chains."""
        sql = """
            SELECT c.*, cfts.rank
            FROM chains_fts cfts
            JOIN chains c ON c.id = cfts.chain_id
            WHERE chains_fts MATCH ?
        """
        params: list = [fts_query]
        if session_id:
            sql += " AND c.session_id = ?"
            params.append(session_id)
        sql += " ORDER BY cfts.rank LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        results = []
        for row in rows:
            chain = _deserialize_chain(row)
            score = min(1.0, abs(row["rank"]) / 10.0)
            results.append((chain, score))
        return results

    def semantic_search_chains(
        self,
        query_embedding: List[float],
        limit: int = 5,
        min_similarity: float = 0.3,
        session_id: Optional[str] = None
    ) -> List[Tuple[ReasoningChain, float]]:
        """Vector similarity search on chain embeddings."""
        sql = "SELECT * FROM chains WHERE embedding IS NOT NULL"
        params: list = []
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)

        rows = self.conn.execute(sql, params).fetchall()
        scored = []
        for row in rows:
            chain = _deserialize_chain(row)
            if chain.embedding:
                sim = _cosine_similarity(query_embedding, chain.embedding)
                if sim >= min_similarity:
                    scored.append((chain, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def hybrid_search_chains(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        limit: int = 5,
        keyword_weight: float = 0.4,
        semantic_weight: float = 0.6,
        session_id: Optional[str] = None
    ) -> List[Tuple[ReasoningChain, float]]:
        """Combined keyword + semantic search on chains."""
        keyword_results = []
        semantic_results = []

        try:
            keyword_results = self.keyword_search_chains(
                query, limit=limit * 2, session_id=session_id
            )
        except Exception:
            pass

        if query_embedding:
            semantic_results = self.semantic_search_chains(
                query_embedding, limit=limit * 2, session_id=session_id
            )

        return _merge_chain_search_results(
            keyword_results, semantic_results,
            keyword_weight, semantic_weight, limit
        )

    def get_entries_by_ids(self, entry_ids: List[str]) -> List[RolodexEntry]:
        """Fetch multiple entries by ID list. Returns found entries."""
        if not entry_ids:
            return []
        placeholders = ",".join("?" for _ in entry_ids)
        rows = self.conn.execute(
            f"SELECT * FROM rolodex_entries WHERE id IN ({placeholders})",
            entry_ids
        ).fetchall()
        return [deserialize_entry(row) for row in rows]

    # ─── Topic CRUD (Phase 8) ────────────────────────────────────────────

    def get_entries_by_topic(
        self, topic_id: str, limit: int = 50
    ) -> List[RolodexEntry]:
        """Fetch entries assigned to a specific topic."""
        rows = self.conn.execute(
            """SELECT * FROM rolodex_entries
               WHERE topic_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (topic_id, limit)
        ).fetchall()
        return [deserialize_entry(row) for row in rows]

    def list_topics(self, limit: int = 50) -> List[dict]:
        """List all topics with metadata."""
        rows = self.conn.execute(
            "SELECT * FROM topics ORDER BY entry_count DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [
            {
                "id": row["id"],
                "label": row["label"],
                "description": row["description"],
                "entry_count": row["entry_count"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_topic(self, topic_id: str) -> Optional[dict]:
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
            "entry_count": row["entry_count"],
            "created_at": row["created_at"],
        }

    # ─── User Knowledge ─────────────────────────────────────────────

    def get_user_knowledge_entries(self) -> List[RolodexEntry]:
        """Fetch all user_knowledge entries, always active, never filtered.

        These represent persistent facts about the user that should
        always be loaded at boot and boosted in search results.
        """
        rows = self.conn.execute(
            """SELECT * FROM rolodex_entries
               WHERE category = 'user_knowledge'
               AND superseded_by IS NULL
               ORDER BY created_at ASC"""
        ).fetchall()
        entries = [deserialize_entry(row) for row in rows]
        # Always keep in hot cache
        for entry in entries:
            entry.tier = Tier.HOT
            self._cache_put(entry)
        return entries

    def get_project_knowledge_entries(self, project_filter: Optional[str] = None) -> List[RolodexEntry]:
        """Fetch all project_knowledge entries, optionally filtered by project tag.

        project_knowledge entries are:
        - Loaded conditionally at boot (when session involves the relevant project)
        - Boosted 2x in search results (between user_knowledge 3x and regular 1x)
        - Never demoted from hot tier
        - Ideal for: project-specific voice rules, content system rules, Tier 2 constraints

        If project_filter is provided, only entries whose tags contain a matching
        project identifier are returned. If None, all project_knowledge is returned.
        """
        rows = self.conn.execute(
            """SELECT * FROM rolodex_entries
               WHERE category = 'project_knowledge'
               AND superseded_by IS NULL
               ORDER BY created_at ASC"""
        ).fetchall()
        entries = [deserialize_entry(row) for row in rows]

        # Optional project-scope filter: match on tags
        if project_filter:
            pf_lower = project_filter.lower()
            entries = [
                e for e in entries
                if any(pf_lower in t.lower() for t in (e.tags or []))
            ]

        # Always keep in hot cache
        for entry in entries:
            entry.tier = Tier.HOT
            self._cache_put(entry)
        return entries

    def get_behavioral_entries(self) -> List[RolodexEntry]:
        """Fetch all behavioral entries (compressed instruction documents).

        These are YAML-like compressed versions of CLAUDE.md / INSTRUCTIONS.md.
        Always loaded at boot when prompt_compression is enabled, always HOT tier.
        """
        rows = self.conn.execute(
            """SELECT * FROM rolodex_entries
               WHERE category = 'behavioral'
               AND superseded_by IS NULL
               ORDER BY created_at ASC"""
        ).fetchall()
        entries = [deserialize_entry(row) for row in rows]
        # Always keep in hot cache
        for entry in entries:
            entry.tier = Tier.HOT
            self._cache_put(entry)
        return entries

    # ─── Entry Superseding (Corrections) ────────────────────────────

    def supersede_entry(self, old_entry_id: str, new_entry_id: str) -> bool:
        """Mark an old entry as superseded by a new one.

        The old entry stays in the DB but is excluded from all searches.
        Use this for factual error corrections — not for reasoning chains
        where the evolution of thought should be preserved.

        Returns True if the old entry existed and was updated.
        """
        row = self.conn.execute(
            "SELECT id FROM rolodex_entries WHERE id = ?", (old_entry_id,)
        ).fetchone()
        if not row:
            return False

        self.conn.execute(
            "UPDATE rolodex_entries SET superseded_by = ? WHERE id = ?",
            (new_entry_id, old_entry_id)
        )
        # Remove from FTS so it never surfaces in keyword search
        self.conn.execute(
            "DELETE FROM rolodex_fts WHERE entry_id = ?", (old_entry_id,)
        )
        self.conn.commit()

        # Evict from hot cache
        if old_entry_id in self._hot_cache:
            del self._hot_cache[old_entry_id]

        return True

    # ─── User Profile ──────────────────────────────────────────────

    def profile_set(self, key: str, value: str, session_id: Optional[str] = None) -> None:
        """Set or update a user profile preference (upsert)."""
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """INSERT INTO user_profile (key, value, source_session, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                   value = excluded.value,
                   source_session = excluded.source_session,
                   updated_at = excluded.updated_at""",
            (key, value, session_id, now)
        )
        self.conn.commit()

    def profile_get_all(self) -> Dict[str, Any]:
        """Return all profile entries as {key: {value, source_session, updated_at}}."""
        rows = self.conn.execute(
            "SELECT key, value, source_session, updated_at FROM user_profile ORDER BY key"
        ).fetchall()
        return {
            row["key"]: {
                "value": row["value"],
                "source_session": row["source_session"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        }

    def profile_delete(self, key: str) -> bool:
        """Delete a profile key. Returns True if it existed."""
        cursor = self.conn.execute(
            "DELETE FROM user_profile WHERE key = ?", (key,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    # ─── Browse (Phase 12) ─────────────────────────────────────────────

    def browse_recent(self, limit: int = 20) -> List[RolodexEntry]:
        """Fetch most recent entries across all sessions, excluding superseded."""
        rows = self.conn.execute(
            """SELECT * FROM rolodex_entries
               WHERE superseded_by IS NULL
               ORDER BY created_at DESC LIMIT ?""",
            (limit,)
        ).fetchall()
        return [deserialize_entry(row) for row in rows]

    def browse_by_source_type(self, source_type: str, limit: int = 20) -> List[RolodexEntry]:
        """Fetch entries filtered by source_type, excluding superseded."""
        rows = self.conn.execute(
            """SELECT * FROM rolodex_entries
               WHERE source_type = ? AND superseded_by IS NULL
               ORDER BY created_at DESC LIMIT ?""",
            (source_type, limit)
        ).fetchall()
        return [deserialize_entry(row) for row in rows]

    def get_session_summaries(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get session summaries with entry counts and date ranges."""
        rows = self.conn.execute(
            """SELECT
                   c.id,
                   c.created_at,
                   c.ended_at,
                   c.summary,
                   c.status,
                   COUNT(re.id) as entry_count,
                   MIN(re.created_at) as first_entry,
                   MAX(re.created_at) as last_entry
               FROM conversations c
               LEFT JOIN rolodex_entries re ON re.conversation_id = c.id
               GROUP BY c.id
               ORDER BY c.created_at DESC
               LIMIT ?""",
            (limit,)
        ).fetchall()
        return [
            {
                "session_id": row["id"],
                "created_at": row["created_at"],
                "ended_at": row["ended_at"],
                "summary": row["summary"] or "",
                "status": row["status"] or "unknown",
                "entry_count": row["entry_count"],
                "first_entry": row["first_entry"],
                "last_entry": row["last_entry"],
            }
            for row in rows
        ]

    def browse_entry_by_prefix(self, prefix: str) -> Optional[RolodexEntry]:
        """Find an entry by ID prefix (for short-id lookups)."""
        rows = self.conn.execute(
            "SELECT * FROM rolodex_entries WHERE id LIKE ?",
            (prefix + "%",)
        ).fetchall()
        if len(rows) == 1:
            return deserialize_entry(rows[0])
        elif len(rows) > 1:
            # Multiple matches — return the most recent
            entries = [deserialize_entry(r) for r in rows]
            entries.sort(key=lambda e: e.created_at, reverse=True)
            return entries[0]
        return None

    # ─── Document Registry (Phase 12) ─────────────────────────────────

    def register_document(
        self,
        doc_id: str,
        file_name: str,
        file_path: str,
        file_type: str,
        file_hash: Optional[str] = None,
        title: Optional[str] = None,
        page_count: Optional[int] = None,
        summary: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register a document in the registry. Returns doc_id."""
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """INSERT INTO documents
               (id, file_name, file_path, file_type, file_hash, title,
                page_count, summary, registered_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, file_name, file_path, file_type, file_hash,
             title, page_count, summary, now,
             json.dumps(metadata or {}))
        )
        self.conn.commit()
        return doc_id

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a registered document by ID."""
        row = self.conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "file_name": row["file_name"],
            "file_path": row["file_path"],
            "file_type": row["file_type"],
            "file_hash": row["file_hash"],
            "title": row["title"],
            "page_count": row["page_count"],
            "summary": row["summary"],
            "registered_at": row["registered_at"],
            "last_read_at": row["last_read_at"],
            "metadata": json.loads(row["metadata"] or "{}"),
        }

    def list_documents(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all registered documents."""
        rows = self.conn.execute(
            "SELECT * FROM documents ORDER BY registered_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [
            {
                "id": row["id"],
                "file_name": row["file_name"],
                "file_path": row["file_path"],
                "file_type": row["file_type"],
                "title": row["title"],
                "page_count": row["page_count"],
                "registered_at": row["registered_at"],
                "last_read_at": row["last_read_at"],
            }
            for row in rows
        ]

    def update_document_read_time(self, doc_id: str) -> None:
        """Update last_read_at timestamp for a document."""
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            "UPDATE documents SET last_read_at = ? WHERE id = ?",
            (now, doc_id)
        )
        self.conn.commit()

    def update_document_hash(self, doc_id: str, file_hash: str) -> None:
        """Update the file hash for change detection."""
        self.conn.execute(
            "UPDATE documents SET file_hash = ? WHERE id = ?",
            (file_hash, doc_id)
        )
        self.conn.commit()

    def remove_document(self, doc_id: str) -> bool:
        """Unregister a document. Entries referencing it remain but link is cleared.
        Returns True if the document existed."""
        row = self.conn.execute(
            "SELECT id FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if not row:
            return False
        # Clear document_id on linked entries (entries stay, just unlinked)
        self.conn.execute(
            "UPDATE rolodex_entries SET document_id = NULL WHERE document_id = ?",
            (doc_id,)
        )
        self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()
        return True

    def get_entries_for_document(self, doc_id: str, limit: int = 50) -> List[RolodexEntry]:
        """Fetch all entries linked to a specific document."""
        rows = self.conn.execute(
            """SELECT * FROM rolodex_entries
               WHERE document_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (doc_id, limit)
        ).fetchall()
        return [deserialize_entry(row) for row in rows]

    def update_entry_document_source(
        self,
        entry_id: str,
        document_id: str,
        source_location: str = "",
    ) -> None:
        """Set the document source fields on an entry (Phase 12)."""
        self.conn.execute(
            """UPDATE rolodex_entries
               SET source_type = 'document', document_id = ?, source_location = ?
               WHERE id = ?""",
            (document_id, source_location, entry_id)
        )
        self.conn.commit()

    def close(self):
        """Close the database connection.

        Forces a WAL checkpoint before closing to ensure all committed data
        is written to the main database file. This is critical on Windows
        where WAL files may not be properly shared between separate processes
        (each CLI invocation opens a new connection).
        """
        if self.conn:
            try:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass  # Non-fatal — DB may already be closed or read-only
            self.conn.close()
# ─── Helper Functions ─────────────────────────────────────────────────────────
def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors. Pure Python, no numpy needed."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
def _merge_search_results(
    keyword_results: List[Tuple[RolodexEntry, float]],
    semantic_results: List[Tuple[RolodexEntry, float]],
    keyword_weight: float,
    semantic_weight: float,
    limit: int
) -> List[Tuple[RolodexEntry, float]]:
    """Merge and deduplicate keyword + semantic results by weighted score."""
    scores: Dict[str, float] = {}
    entries: Dict[str, RolodexEntry] = {}
    # Normalize keyword scores to 0-1
    if keyword_results:
        max_kw = max(s for _, s in keyword_results) or 1.0
        for entry, score in keyword_results:
            norm_score = (score / max_kw) * keyword_weight
            scores[entry.id] = scores.get(entry.id, 0) + norm_score
            entries[entry.id] = entry
    # Semantic scores are already 0-1 (cosine similarity)
    for entry, score in semantic_results:
        norm_score = score * semantic_weight
        scores[entry.id] = scores.get(entry.id, 0) + norm_score
        entries[entry.id] = entry
    # Boost privileged-tier entries so they surface above bulk content
    for eid, entry in entries.items():
        if entry.category.value == "user_knowledge":
            scores[eid] *= USER_KNOWLEDGE_BOOST
        elif entry.category.value == "project_knowledge":
            scores[eid] *= PROJECT_KNOWLEDGE_BOOST

    # Sort by combined score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(entries[eid], score) for eid, score in ranked[:limit]]

USER_KNOWLEDGE_BOOST = 3.0  # user_knowledge entries score 3x higher
PROJECT_KNOWLEDGE_BOOST = 2.0  # project_knowledge entries score 2x higher

# ─── Chain Serialization (Phase 7) ───────────────────────────────────────────

def _serialize_chain(chain: ReasoningChain) -> tuple:
    """Convert ReasoningChain to tuple for SQL INSERT."""
    return (
        chain.id,
        chain.session_id,
        chain.chain_index,
        chain.turn_range_start,
        chain.turn_range_end,
        chain.summary,
        json.dumps(chain.topics),
        json.dumps(chain.related_entries),
        serialize_embedding(chain.embedding) if chain.embedding else None,
        chain.created_at.isoformat(),
    )


def _deserialize_chain(row: sqlite3.Row) -> ReasoningChain:
    """Convert database row back to ReasoningChain."""
    return ReasoningChain(
        id=row["id"],
        session_id=row["session_id"],
        chain_index=row["chain_index"],
        turn_range_start=row["turn_range_start"],
        turn_range_end=row["turn_range_end"],
        summary=row["summary"],
        topics=json.loads(row["topics"]),
        related_entries=json.loads(row["related_entries"]),
        embedding=(
            deserialize_embedding(row["embedding"])
            if row["embedding"] else None
        ),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _merge_chain_search_results(
    keyword_results: List[Tuple[ReasoningChain, float]],
    semantic_results: List[Tuple[ReasoningChain, float]],
    keyword_weight: float,
    semantic_weight: float,
    limit: int
) -> List[Tuple[ReasoningChain, float]]:
    """Merge and deduplicate keyword + semantic chain results."""
    scores: Dict[str, float] = {}
    chains: Dict[str, ReasoningChain] = {}

    if keyword_results:
        max_kw = max(s for _, s in keyword_results) or 1.0
        for chain, score in keyword_results:
            norm_score = (score / max_kw) * keyword_weight
            scores[chain.id] = scores.get(chain.id, 0) + norm_score
            chains[chain.id] = chain

    for chain, score in semantic_results:
        norm_score = score * semantic_weight
        scores[chain.id] = scores.get(chain.id, 0) + norm_score
        chains[chain.id] = chain

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(chains[cid], score) for cid, score in ranked[:limit]]
