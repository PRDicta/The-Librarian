"""
The Librarian — Hybrid Searcher
Combines keyword search (FTS5) with vector similarity search,
merging results with configurable weights.

Phase 8: Topic-aware routing — infers query topic and searches
within that namespace first, falling back to cross-topic.
"""
import time
from typing import List, Optional, Tuple
from ..core.types import RolodexEntry, LibrarianQuery, LibrarianResponse
from ..storage.rolodex import Rolodex
from ..indexing.embeddings import EmbeddingManager


class HybridSearcher:
    """
    Execute hybrid search queries against the rolodex.
    Searches hot cache first (fast path), then cold storage.
    Phase 8: Optional topic-aware routing via TopicRouter.
    """
    def __init__(self, rolodex: Rolodex, embedding_manager: EmbeddingManager):
        self.rolodex = rolodex
        self.embeddings = embedding_manager
        self._topic_router = None  # Set via set_topic_router()

    def set_topic_router(self, topic_router) -> None:
        """Attach a TopicRouter for topic-aware search routing."""
        self._topic_router = topic_router
    async def search(
        self,
        query: LibrarianQuery,
        cross_session: bool = False,
        session_boost_factor: float = 1.5,
    ) -> LibrarianResponse:
        """
        Execute a search query and return results.

        When cross_session=True, searches ALL sessions and boosts
        current-session results by session_boost_factor.

        Flow:
        1. Check hot cache for quick matches
        2. Generate query embedding
        3. Run search on cold storage (scoped or cross-session)
        4. Merge hot cache hits with cold storage results
        5. Deduplicate and rank
        6. Log the query for analytics
        """
        start = time.time()
        cross_session_count = 0

        # Step 1: Check hot cache
        cache_results = self._search_hot_cache(query.query_text)
        cache_hit = len(cache_results) > 0

        # Step 2: Generate query embedding for semantic search
        query_embedding = None
        if query.search_type in ("semantic", "hybrid"):
            query_embedding = await self.embeddings.embed_text(query.query_text)

        # Step 3: Search cold storage
        # Phase 8: Topic-scoped search — try topic namespace first
        # Hierarchy-aware: if a parent topic matches, search all children too
        topic_id = None
        topic_group = []
        topic_scoped_results = []
        if self._topic_router:
            try:
                topic_id = await self._topic_router.infer_topic_for_query(query)
                if topic_id:
                    topic_group = self._topic_router.get_topic_group(topic_id)
                    # Search across all topics in the group
                    for tid in topic_group:
                        results = self.rolodex.hybrid_search_by_topic(
                            query=query.query_text,
                            topic_id=tid,
                            query_embedding=query_embedding,
                            limit=query.limit,
                        )
                        topic_scoped_results.extend(results)
                    # Deduplicate and sort by score
                    seen = set()
                    deduped = []
                    for entry, score in sorted(
                        topic_scoped_results, key=lambda x: x[1], reverse=True
                    ):
                        if entry.id not in seen:
                            seen.add(entry.id)
                            deduped.append((entry, score))
                    topic_scoped_results = deduped[:query.limit]
            except Exception:
                pass  # Topic routing failure is never a blocker

        # Determine conversation_id scope
        search_conv_id = None if cross_session else query.conversation_id

        # If topic-scoped search found enough results, use them;
        # otherwise fall back to full cross-topic search
        if topic_scoped_results and len(topic_scoped_results) >= query.limit:
            cold_results = topic_scoped_results
        elif cross_session and query.search_type in ("hybrid", "keyword"):
            # Phase 4: Use boosted search for cross-session hybrid/keyword
            cold_results = self.rolodex.boosted_hybrid_search(
                query=query.query_text,
                query_embedding=query_embedding,
                current_session_id=query.conversation_id,
                boost_factor=session_boost_factor,
                limit=query.limit * 2,
            )
            # Merge topic-scoped results (they get priority)
            if topic_scoped_results:
                cold_results = self._merge_topic_results(
                    topic_scoped_results, cold_results, query.limit * 2
                )
        elif query.search_type == "keyword":
            cold_results = self.rolodex.keyword_search(
                query.query_text,
                limit=query.limit * 2,
                conversation_id=search_conv_id,
            )
        elif query.search_type == "semantic":
            cold_results = []
            if query_embedding:
                cold_results = self.rolodex.semantic_search(
                    query_embedding,
                    limit=query.limit * 2,
                    min_similarity=query.min_similarity,
                    conversation_id=search_conv_id,
                )
        else:  # hybrid (non-cross-session)
            cold_results = self.rolodex.hybrid_search(
                query=query.query_text,
                query_embedding=query_embedding,
                limit=query.limit * 2,
                conversation_id=search_conv_id,
            )

        # Step 4: Merge results
        all_entries = self._merge_results(cache_results, cold_results, query.limit)

        # Count cross-session results
        if cross_session and query.conversation_id:
            cross_session_count = sum(
                1 for e in all_entries
                if e.conversation_id != query.conversation_id
            )

        # Step 5: Update access counts for returned entries
        for entry in all_entries:
            self.rolodex.update_access(entry.id)

        elapsed_ms = (time.time() - start) * 1000

        # Step 6: Log query
        if query.conversation_id:
            self.rolodex.log_query(
                conversation_id=query.conversation_id,
                query_text=query.query_text,
                found=len(all_entries) > 0,
                entry_ids=[e.id for e in all_entries],
                search_time_ms=elapsed_ms,
                search_type=query.search_type,
            )

        # Phase 7: Chain-first search — search reasoning chains before returning
        chain_results = []
        chain_guided_entries = []
        try:
            if hasattr(self.rolodex, 'hybrid_search_chains'):
                chain_results = self.rolodex.hybrid_search_chains(
                    query=query.query_text,
                    query_embedding=query_embedding,
                    limit=3,
                    session_id=search_conv_id,
                )
                # Pull related entries from matching chains
                chain_entry_ids = set()
                for chain, _ in chain_results:
                    chain_entry_ids.update(chain.related_entries)
                # Fetch entries referenced by chains (that aren't already in results)
                existing_ids = {e.id for e in all_entries}
                new_ids = [eid for eid in chain_entry_ids if eid not in existing_ids]
                if new_ids:
                    chain_guided_entries = self.rolodex.get_entries_by_ids(new_ids)
                    all_entries.extend(chain_guided_entries)
                    all_entries = all_entries[:query.limit]  # Respect limit
        except Exception:
            pass  # Chain search failure is never a blocker

        response = LibrarianResponse(
            found=len(all_entries) > 0,
            entries=all_entries,
            chains=[chain for chain, _ in chain_results],
            search_time_ms=elapsed_ms,
            cache_hit=cache_hit,
            query=query,
        )
        # Attach cross-session count for debug/logging
        response.metadata["cross_session_results"] = cross_session_count
        response.metadata["chain_results"] = len(chain_results)
        response.metadata["chain_guided_entries"] = len(chain_guided_entries)
        # Phase 8: Topic routing metadata
        if topic_id:
            response.metadata["topic_id"] = topic_id
            response.metadata["topic_group_size"] = len(topic_group)
            response.metadata["topic_scoped_count"] = len(topic_scoped_results)
        return response
    def _search_hot_cache(self, query_text: str) -> List[RolodexEntry]:
        """Quick keyword match against hot cache entries."""
        return self.rolodex.search_hot_cache(query_text)
    def _merge_results(
        self,
        cache_results: List[RolodexEntry],
        cold_results: List[Tuple[RolodexEntry, float]],
        limit: int
    ) -> List[RolodexEntry]:
        """
        Merge hot cache hits with cold storage results.
        Hot cache hits get priority (they're already proven relevant).
        Deduplicate by entry ID.
        """
        seen_ids = set()
        merged = []
        # Cache hits first
        for entry in cache_results:
            if entry.id not in seen_ids:
                seen_ids.add(entry.id)
                merged.append(entry)
        # Then cold storage results (already ranked by score)
        for entry, score in cold_results:
            if entry.id not in seen_ids:
                seen_ids.add(entry.id)
                merged.append(entry)
        return merged[:limit]

    @staticmethod
    def _merge_topic_results(
        topic_results: List[Tuple[RolodexEntry, float]],
        general_results: List[Tuple[RolodexEntry, float]],
        limit: int,
    ) -> List[Tuple[RolodexEntry, float]]:
        """
        Merge topic-scoped results with general results.
        Topic results get priority (placed first), then general results
        fill remaining slots. Deduplicates by entry ID.
        """
        seen_ids = set()
        merged = []
        for entry, score in topic_results:
            if entry.id not in seen_ids:
                seen_ids.add(entry.id)
                merged.append((entry, score))
        for entry, score in general_results:
            if entry.id not in seen_ids:
                seen_ids.add(entry.id)
                merged.append((entry, score))
        return merged[:limit]
