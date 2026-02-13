


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


    def __init__(self, db_path: str = "rolodex.db"):
        self.db_path = db_path
        self.conn = init_database(db_path)
        self._hot_cache: OrderedDict[str, RolodexEntry] = OrderedDict()
        self._hot_cache_max = 50

    def create_entry(self, entry: RolodexEntry) -> str:

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
        self.conn.commit()

        self._cache_put(entry)
        return entry.id
    def batch_create_entries(self, entries: List[RolodexEntry]) -> List[str]:

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

        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """UPDATE rolodex_entries
               SET access_count = access_count + 1, last_accessed = ?
               WHERE id = ?""",
            (now, entry_id)
        )
        self.conn.commit()

        if entry_id in self._hot_cache:
            entry = self._hot_cache[entry_id]
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
            self._hot_cache.move_to_end(entry_id)


    def update_entry_enrichment(
        self,
        entry_id: str,
        content_type: Optional[ContentModality] = None,
        category: Optional[EntryCategory] = None,
        tags: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:


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

            self.conn.execute(
                "UPDATE rolodex_fts SET tags = ? WHERE entry_id = ?",
                (json.dumps(tags), entry_id)
            )
        if embedding is not None:
            updates.append("embedding = ?")
            params.append(serialize_embedding(embedding))
        if metadata is not None:

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


        if category is not None:
            self.conn.execute(
                "UPDATE rolodex_fts SET category = ? WHERE entry_id = ?",
                (category.value, entry_id)
            )

        self.conn.commit()


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

        self.update_entry_enrichment(entry_id=entry_id, metadata=metadata)


    def get_entry(self, entry_id: str) -> Optional[RolodexEntry]:


        if entry_id in self._hot_cache:
            return self._hot_cache[entry_id]

        row = self.conn.execute(
            "SELECT * FROM rolodex_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        if row:
            return deserialize_entry(row)
        return None
    def keyword_search(
        self, query: str, limit: int = 5, conversation_id: Optional[str] = None
    ) -> List[Tuple[RolodexEntry, float]]:


        results = self._fts_match(query, limit, conversation_id)


        if not results and ' ' in query.strip():
            fts_operators = {'AND', 'OR', 'NOT', 'NEAR'}
            terms = [
                t for t in query.strip().split()
                if t.upper() not in fts_operators and len(t) > 1
            ]
            if len(terms) > 1:
                or_query = ' OR '.join(terms)
                results = self._fts_match(or_query, limit, conversation_id)

        return results

    def _fts_match(
        self, fts_query: str, limit: int, conversation_id: Optional[str] = None
    ) -> List[Tuple[RolodexEntry, float]]:

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
        sql += " ORDER BY fts.rank LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        results = []
        for row in rows:
            entry = deserialize_entry(row)


            score = min(1.0, abs(row["rank"]) / 10.0)
            results.append((entry, score))
        return results
    def semantic_search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        min_similarity: float = 0.3,
        conversation_id: Optional[str] = None
    ) -> List[Tuple[RolodexEntry, float]]:


        sql = "SELECT * FROM rolodex_entries WHERE embedding IS NOT NULL"
        params: list = []
        if conversation_id:
            sql += " AND conversation_id = ?"
            params.append(conversation_id)
        rows = self.conn.execute(sql, params).fetchall()

        scored = []
        for row in rows:
            entry = deserialize_entry(row)
            if entry.embedding:
                sim = _cosine_similarity(query_embedding, entry.embedding)
                if sim >= min_similarity:
                    scored.append((entry, sim))

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
        conversation_id: Optional[str] = None
    ) -> List[Tuple[RolodexEntry, float]]:


        keyword_results = []
        semantic_results = []

        try:
            keyword_results = self.keyword_search(
                query, limit=limit * 2, conversation_id=conversation_id
            )
        except Exception:
            pass

        if query_embedding:
            semantic_results = self.semantic_search(
                query_embedding, limit=limit * 2,
                min_similarity=min_similarity,
                conversation_id=conversation_id
            )

        return _merge_search_results(
            keyword_results, semantic_results,
            keyword_weight, semantic_weight, limit
        )


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
    ) -> List[Tuple[RolodexEntry, float]]:


        results = self.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            limit=limit * 2,
            keyword_weight=keyword_weight,
            semantic_weight=semantic_weight,
            min_similarity=min_similarity,
            conversation_id=None,
        )

        if not current_session_id or boost_factor <= 1.0:
            return results[:limit]


        boosted = []
        for entry, score in results:
            if entry.conversation_id == current_session_id:
                boosted.append((entry, score * boost_factor))
            else:
                boosted.append((entry, score))


        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted[:limit]


    def keyword_search_by_topic(
        self, query: str, topic_id: str, limit: int = 5
    ) -> List[Tuple[RolodexEntry, float]]:

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

        rows = self.conn.execute(
            """SELECT * FROM rolodex_entries
               WHERE conversation_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (conversation_id, limit)
        ).fetchall()
        return [deserialize_entry(row) for row in rows]

    def get_hot_cache_entries(self) -> List[RolodexEntry]:

        return list(self._hot_cache.values())
    def search_hot_cache(self, query: str) -> List[RolodexEntry]:

        query_lower = query.lower()
        results = []
        for entry in self._hot_cache.values():
            content_lower = entry.content.lower()
            tags_str = " ".join(entry.tags).lower()
            if query_lower in content_lower or query_lower in tags_str:
                results.append(entry)
        return results
    def _cache_put(self, entry: RolodexEntry):

        self._hot_cache[entry.id] = entry
        self._hot_cache.move_to_end(entry.id)
        while len(self._hot_cache) > self._hot_cache_max:
            self._hot_cache.popitem(last=False)


    def evaluate_tier(
        self,
        entry_id: str,
        promotion_threshold: float = 1.0,
        demotion_threshold: float = 0.3,
        recency_half_life: float = 24.0,
        age_boost_half_life: float = 48.0,
    ) -> Tuple[Tier, float]:


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

            return (entry.tier, score)

    def promote_entry(self, entry_id: str) -> Optional[TierEvent]:

        entry = self.get_entry(entry_id)
        if entry is None or entry.tier == Tier.HOT:
            return None
        old_tier = entry.tier

        self.conn.execute(
            "UPDATE rolodex_entries SET tier = ? WHERE id = ?",
            ("hot", entry_id)
        )
        self.conn.commit()

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

        entry = self.get_entry(entry_id)
        if entry is None or entry.tier == Tier.COLD:
            return None
        old_tier = entry.tier

        self.conn.execute(
            "UPDATE rolodex_entries SET tier = ? WHERE id = ?",
            ("cold", entry_id)
        )
        self.conn.commit()

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


        rows = self.conn.execute(
            "SELECT * FROM rolodex_entries WHERE tier = 'hot' ORDER BY access_count DESC"
        ).fetchall()
        loaded = 0
        for row in rows:
            entry = deserialize_entry(row)
            self._cache_put(entry)
            loaded += 1
        return loaded


    def get_stats(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:

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

        tier_rows = self.conn.execute(
            f"""SELECT tier, COUNT(*) as cnt
                FROM rolodex_entries {where}
                GROUP BY tier""",
            params
        ).fetchall()
        tier_distribution = {}
        for row in tier_rows:
            tier_distribution[row["tier"]] = row["cnt"]

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


    def create_chain(self, chain: ReasoningChain) -> str:

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

        row = self.conn.execute(
            "SELECT * FROM chains WHERE id = ?", (chain_id,)
        ).fetchone()
        if row:
            return _deserialize_chain(row)
        return None

    def get_chains_for_session(self, session_id: str) -> List[ReasoningChain]:

        rows = self.conn.execute(
            "SELECT * FROM chains WHERE session_id = ? ORDER BY chain_index ASC",
            (session_id,)
        ).fetchall()
        return [_deserialize_chain(row) for row in rows]

    def get_chain_by_index(
        self, session_id: str, chain_index: int
    ) -> Optional[ReasoningChain]:

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


        results = self._fts_match_chains(query, limit, session_id)


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

        if not entry_ids:
            return []
        placeholders = ",".join("?" for _ in entry_ids)
        rows = self.conn.execute(
            f"SELECT * FROM rolodex_entries WHERE id IN ({placeholders})",
            entry_ids
        ).fetchall()
        return [deserialize_entry(row) for row in rows]


    def get_entries_by_topic(
        self, topic_id: str, limit: int = 50
    ) -> List[RolodexEntry]:

        rows = self.conn.execute(
            """SELECT * FROM rolodex_entries
               WHERE topic_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (topic_id, limit)
        ).fetchall()
        return [deserialize_entry(row) for row in rows]

    def list_topics(self, limit: int = 50) -> List[dict]:

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

    def close(self):

        if self.conn:
            self.conn.close()

def _cosine_similarity(a: List[float], b: List[float]) -> float:

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

    scores: Dict[str, float] = {}
    entries: Dict[str, RolodexEntry] = {}

    if keyword_results:
        max_kw = max(s for _, s in keyword_results) or 1.0
        for entry, score in keyword_results:
            norm_score = (score / max_kw) * keyword_weight
            scores[entry.id] = scores.get(entry.id, 0) + norm_score
            entries[entry.id] = entry

    for entry, score in semantic_results:
        norm_score = score * semantic_weight
        scores[entry.id] = scores.get(entry.id, 0) + norm_score
        entries[entry.id] = entry

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(entries[eid], score) for eid, score in ranked[:limit]]


def _serialize_chain(chain: ReasoningChain) -> tuple:

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
