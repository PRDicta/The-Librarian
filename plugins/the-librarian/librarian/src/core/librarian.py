


from typing import Dict, List, Optional, Any

from .types import (
    Message, MessageRole, RolodexEntry, ConversationState,
    LibrarianQuery, LibrarianResponse, SessionInfo, estimate_tokens,
)
from .llm_adapter import LLMAdapter
from .librarian_agent import LibrarianAgent
from ..storage.rolodex import Rolodex
from ..storage.session_manager import SessionManager
from ..indexing.extractor import EntryExtractor
from ..indexing.embeddings import EmbeddingManager
from ..indexing.chunker import ContentChunker
from ..indexing.ingestion_queue import IngestionQueue, IngestionTask
from ..indexing.topic_router import TopicRouter
from .context_window import ContextWindowManager
from ..retrieval.searcher import HybridSearcher
from ..retrieval.context_builder import ContextBuilder
from ..utils.config import LibrarianConfig


class TheLibrarian:


    def __init__(
        self,
        db_path: str = "rolodex.db",
        llm_adapter: Optional[LLMAdapter] = None,
        config: Optional[LibrarianConfig] = None,
    ):
        self.config = config or LibrarianConfig()
        self._llm_adapter = llm_adapter


        self.state = ConversationState()


        self.rolodex = Rolodex(db_path)
        self.rolodex._hot_cache_max = self.config.hot_cache_size
        self.rolodex.create_conversation(self.state.conversation_id)


        self.session_manager = SessionManager(self.rolodex.conn)


        emb_strategy = self.config.embedding_strategy
        voyage_key = getattr(self.config, "voyage_api_key", "")

        self.embeddings = EmbeddingManager(
            strategy=emb_strategy,
            api_key=self.config.anthropic_api_key,
            voyage_api_key=voyage_key or None,
        )


        self.chunker = ContentChunker()
        self.extractor = EntryExtractor(
            embedding_manager=self.embeddings,
            chunker=self.chunker,
            llm_adapter=llm_adapter,
        )


        self.topic_router = TopicRouter(
            conn=self.rolodex.conn,
            embedding_manager=self.embeddings,
        )


        self.searcher = HybridSearcher(self.rolodex, self.embeddings)
        self.searcher.set_topic_router(self.topic_router)
        self.context_builder = ContextBuilder()


        self.librarian_agent = LibrarianAgent(
            rolodex=self.rolodex,
            extractor=self.extractor,
            embedding_manager=self.embeddings,
            searcher=self.searcher,
            context_builder=self.context_builder,
        )


        self.librarian_agent.promotion_threshold = self.config.promotion_threshold
        self.librarian_agent.demotion_threshold = self.config.demotion_threshold
        self.librarian_agent.recency_half_life = self.config.score_recency_half_life_hours
        self.librarian_agent.age_boost_half_life = self.config.score_age_boost_half_life_hours


        self._preloaded_count = self.rolodex.preload_hot_entries()


        if self.config.preload_enabled:
            self.librarian_agent.init_preloader(
                llm_adapter=llm_adapter,
                context_max=self.config.context_window_size,
            )


        self._queue_enabled = getattr(self.config, 'ingestion_queue_enabled', False)
        self.ingestion_queue: Optional[IngestionQueue] = None
        if self._queue_enabled:
            self.ingestion_queue = IngestionQueue(
                enrichment_fn=self.librarian_agent.process_enrichment_task,
                num_workers=getattr(self.config, 'ingestion_num_workers', 2),
                pause_on_query=getattr(self.config, 'ingestion_pause_on_query', True),
            )


        self.context_window = ContextWindowManager(
            token_budget=getattr(self.config, 'context_window_budget', 20_000),
            retrieval_budget=self.config.max_context_for_retrieval,
            min_active_turns=getattr(self.config, 'context_min_active_turns', 4),
            bridge_summary_max_tokens=getattr(
                self.config, 'context_bridge_max_tokens', 1_000
            ),
        )


        self.session_manager.start_session(self.state.conversation_id)


    async def ingest(
        self,
        role: str,
        content: str,
        turn_number: Optional[int] = None,
    ) -> List[RolodexEntry]:


        msg_role = MessageRole.USER if role == "user" else MessageRole.ASSISTANT
        msg = self.state.add_message(msg_role, content)


        self.session_manager.save_message(self.state.conversation_id, msg)

        if self._queue_enabled and self.ingestion_queue:


            stub = self.ingestion_queue.create_stub_entry(
                msg, self.state.conversation_id
            )

            self.rolodex.create_entry(stub)


            task = IngestionTask(
                message=msg,
                stub_entry_ids=[stub.id],
                conversation_id=self.state.conversation_id,
                turn_number=msg.turn_number,
            )
            await self.ingestion_queue.enqueue(task)


            if not self.ingestion_queue._running:
                await self.ingestion_queue.start()

            entries = [stub]
        else:

            entries = await self.librarian_agent.index_new_messages(
                self.state.messages,
                self.state.conversation_id,
            )


        self.session_manager.update_session_activity(self.state.conversation_id)


        if entries:
            self.context_window.record_checkpoint(
                turn_number=msg.turn_number,
                entry_count=len(entries),
                token_count=msg.token_count,
            )

        return entries

    async def retrieve(
        self,
        query_text: str,
        limit: Optional[int] = None,
    ) -> LibrarianResponse:


        if self.ingestion_queue:
            await self.ingestion_queue.pause(reason="query")

        try:
            query = LibrarianQuery(
                query_text=query_text,
                search_type="hybrid",
                limit=limit or self.config.search_result_limit,
                conversation_id=self.state.conversation_id,
            )
            return await self.librarian_agent.answer_query(
                query,
                cross_session=self.config.cross_session_search,
                session_boost_factor=self.config.session_boost_factor,
            )
        finally:

            if self.ingestion_queue:
                await self.ingestion_queue.resume()

    async def retrieve_with_scores(
        self,
        query_text: str,
        limit: Optional[int] = None,
    ) -> tuple:


        response = await self.retrieve(query_text, limit)

        scores = {}
        if response.entries:

            for i, entry in enumerate(response.entries):

                scores[entry.id] = max(0.1, 1.0 - (i * 0.15))
        return response.entries, scores

    async def search(
        self,
        query_text: str,
        limit: Optional[int] = None,
    ) -> LibrarianResponse:


        return await self.retrieve(query_text, limit)

    def get_context_block(
        self,
        response: LibrarianResponse,
    ) -> str:


        return self.librarian_agent.get_context_for_query(
            response,
            current_session_id=self.state.conversation_id,
        )


    def get_context_payload(
        self,
        recall_block: str = "",
    ) -> dict:


        return self.context_window.build_context_payload(
            state=self.state,
            recall_block=recall_block,
        )

    def get_active_messages(self) -> List[Message]:


        return self.context_window.get_active_messages(self.state)


    async def preload(
        self,
        recent_messages: Optional[List[Message]] = None,
    ):


        if not self.config.preload_enabled:
            return None

        messages = recent_messages or self._get_recent_messages()
        return await self.librarian_agent.preload(
            recent_messages=messages,
            turn_number=self.state.turn_count,
            conversation_id=self.state.conversation_id,
            max_entries=self.config.preload_max_entries,
            injection_confidence=self.config.preload_injection_confidence,
            low_threshold=self.config.preload_low_threshold,
            high_threshold=self.config.preload_high_threshold,
        )


    @property
    def session_id(self) -> str:

        return self.state.conversation_id

    def start_session(self, conversation_id: Optional[str] = None) -> str:


        if conversation_id:
            self.state = ConversationState(conversation_id=conversation_id)
        else:
            self.state = ConversationState()


        self.librarian_agent._last_indexed_turn = 0

        self.rolodex.create_conversation(self.state.conversation_id)
        self.session_manager.start_session(self.state.conversation_id)
        return self.state.conversation_id

    def end_session(self, summary: str = "") -> None:

        self.session_manager.end_session(
            self.state.conversation_id, summary=summary
        )

    def resume_session(self, session_id: str) -> Optional[SessionInfo]:


        session_info = self.session_manager.get_session(session_id)
        if not session_info:
            return None

        messages = self.session_manager.load_messages(session_id)


        self.state = ConversationState(conversation_id=session_id)
        for msg in (messages or []):
            self.state.messages.append(msg)
            self.state.total_tokens += msg.token_count
            self.state.turn_count = max(self.state.turn_count, msg.turn_number)


        self.librarian_agent._last_indexed_turn = self.state.turn_count


        self.session_manager.start_session(session_id)

        return session_info

    def list_sessions(self, limit: int = 20) -> List[SessionInfo]:

        return self.session_manager.list_sessions(limit=limit)

    def find_session(self, prefix: str) -> Optional[str]:

        return self.session_manager.find_session_by_prefix(prefix)


    def run_maintenance(self) -> Dict:

        return self.librarian_agent.run_maintenance()


    def get_stats(self) -> Dict[str, Any]:

        librarian_stats = self.librarian_agent.get_stats()
        stats = {
            "conversation_id": self.state.conversation_id,
            "total_messages": len(self.state.messages),
            "total_tokens": self.state.total_tokens,
            "turn_count": self.state.turn_count,
            "llm_adapter": self._llm_adapter is not None,
            "cross_session_search": self.config.cross_session_search,
            **librarian_stats,
        }
        if self.config.preload_enabled:
            stats["pressure"] = self.librarian_agent.pressure_monitor.get_summary()

        if self.ingestion_queue:
            stats["ingestion_queue"] = self.ingestion_queue.get_stats()

        try:
            total_topics = self.topic_router.count_topics()
            unassigned = self.topic_router.count_unassigned_entries()
            total_entries = librarian_stats.get("total_entries", 0)
            coverage = (
                round((total_entries - unassigned) / total_entries * 100, 1)
                if total_entries > 0 else 0.0
            )

            top_level = self.rolodex.conn.execute(
                "SELECT COUNT(*) as cnt FROM topics WHERE parent_topic_id IS NULL"
            ).fetchone()["cnt"]
            stats["topics"] = {
                "top_level": top_level,
                "total_with_children": total_topics,
                "unassigned_entries": unassigned,
                "coverage_pct": coverage,
            }
        except Exception:
            pass

        stats["context_window"] = self.context_window.get_stats()
        all_sessions = self.session_manager.list_sessions(limit=1000)
        stats["total_sessions"] = len(all_sessions)
        return stats


    async def shutdown(self) -> None:


        if self.ingestion_queue:
            await self.ingestion_queue.shutdown()
        self.session_manager.end_session(self.state.conversation_id)
        self.rolodex.close()


    def _get_recent_messages(self, max_tokens: int = 0) -> List[Message]:

        available = max_tokens or self.config.max_context_for_history
        selected = []
        total = 0
        for msg in reversed(self.state.messages):
            cost = msg.token_count + 50
            if total + cost > available:
                break
            selected.insert(0, msg)
            total += cost
        return selected
