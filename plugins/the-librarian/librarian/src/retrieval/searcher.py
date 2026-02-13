


import time
from typing import List, Optional, Tuple
from ..core.types import RolodexEntry, LibrarianQuery, LibrarianResponse
from ..storage.rolodex import Rolodex
from ..indexing.embeddings import EmbeddingManager


class HybridSearcher:


    def __init__(self, rolodex: Rolodex, embedding_manager: EmbeddingManager):
        self.rolodex = rolodex
        self.embeddings = embedding_manager
        self._topic_router = None

    def set_topic_router(self, topic_router) -> None:

        self._topic_router = topic_router
    async def search(
        self,
        query: LibrarianQuery,
        cross_session: bool = False,
        session_boost_factor: float = 1.5,
    ) -> LibrarianResponse:


        start = time.time()
        cross_session_count = 0


        cache_results = self._search_hot_cache(query.query_text)
        cache_hit = len(cache_results) > 0


        query_embedding = None
        if query.search_type in ("semantic", "hybrid"):
            query_embedding = await self.embeddings.embed_text(query.query_text)


        topic_id = None
        topic_group = []
        topic_scoped_results = []
        if self._topic_router:
            try:
                topic_id = await self._topic_router.infer_topic_for_query(query)
                if topic_id:
                    topic_group = self._topic_router.get_topic_group(topic_id)

                    for tid in topic_group:
                        results = self.rolodex.hybrid_search_by_topic(
                            query=query.query_text,
                            topic_id=tid,
                            query_embedding=query_embedding,
                            limit=query.limit,
                        )
                        topic_scoped_results.extend(results)

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
                pass


        search_conv_id = None if cross_session else query.conversation_id


        if topic_scoped_results and len(topic_scoped_results) >= query.limit:
            cold_results = topic_scoped_results
        elif cross_session and query.search_type in ("hybrid", "keyword"):

            cold_results = self.rolodex.boosted_hybrid_search(
                query=query.query_text,
                query_embedding=query_embedding,
                current_session_id=query.conversation_id,
                boost_factor=session_boost_factor,
                limit=query.limit * 2,
            )

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
        else:
            cold_results = self.rolodex.hybrid_search(
                query=query.query_text,
                query_embedding=query_embedding,
                limit=query.limit * 2,
                conversation_id=search_conv_id,
            )


        all_entries = self._merge_results(cache_results, cold_results, query.limit)


        if cross_session and query.conversation_id:
            cross_session_count = sum(
                1 for e in all_entries
                if e.conversation_id != query.conversation_id
            )


        for entry in all_entries:
            self.rolodex.update_access(entry.id)

        elapsed_ms = (time.time() - start) * 1000


        if query.conversation_id:
            self.rolodex.log_query(
                conversation_id=query.conversation_id,
                query_text=query.query_text,
                found=len(all_entries) > 0,
                entry_ids=[e.id for e in all_entries],
                search_time_ms=elapsed_ms,
                search_type=query.search_type,
            )


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

                chain_entry_ids = set()
                for chain, _ in chain_results:
                    chain_entry_ids.update(chain.related_entries)

                existing_ids = {e.id for e in all_entries}
                new_ids = [eid for eid in chain_entry_ids if eid not in existing_ids]
                if new_ids:
                    chain_guided_entries = self.rolodex.get_entries_by_ids(new_ids)
                    all_entries.extend(chain_guided_entries)
                    all_entries = all_entries[:query.limit]
        except Exception:
            pass

        response = LibrarianResponse(
            found=len(all_entries) > 0,
            entries=all_entries,
            chains=[chain for chain, _ in chain_results],
            search_time_ms=elapsed_ms,
            cache_hit=cache_hit,
            query=query,
        )

        response.metadata["cross_session_results"] = cross_session_count
        response.metadata["chain_results"] = len(chain_results)
        response.metadata["chain_guided_entries"] = len(chain_guided_entries)

        if topic_id:
            response.metadata["topic_id"] = topic_id
            response.metadata["topic_group_size"] = len(topic_group)
            response.metadata["topic_scoped_count"] = len(topic_scoped_results)
        return response
    def _search_hot_cache(self, query_text: str) -> List[RolodexEntry]:

        return self.rolodex.search_hot_cache(query_text)
    def _merge_results(
        self,
        cache_results: List[RolodexEntry],
        cold_results: List[Tuple[RolodexEntry, float]],
        limit: int
    ) -> List[RolodexEntry]:


        seen_ids = set()
        merged = []

        for entry in cache_results:
            if entry.id not in seen_ids:
                seen_ids.add(entry.id)
                merged.append(entry)

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
