


import asyncio
from typing import List, Dict, Optional
from .types import (
    Message, MessageRole, RolodexEntry, LibrarianQuery, LibrarianResponse,
    TierEvent, PreloadResult, estimate_tokens
)
from .llm_adapter import LLMAdapter
from ..storage.rolodex import Rolodex
from ..indexing.extractor import EntryExtractor
from ..indexing.embeddings import EmbeddingManager
from ..indexing.ingestion_queue import IngestionQueue, IngestionTask, TaskStatus
from ..retrieval.searcher import HybridSearcher
from ..retrieval.context_builder import ContextBuilder
from ..preloading.pressure import PressureMonitor
from ..preloading.preloader import Preloader
class LibrarianAgent:


    def __init__(
        self,
        rolodex: Rolodex,
        extractor: EntryExtractor,
        embedding_manager: EmbeddingManager,
        searcher: HybridSearcher,
        context_builder: ContextBuilder,
    ):
        self.rolodex = rolodex
        self.extractor = extractor
        self.embeddings = embedding_manager
        self.searcher = searcher
        self.context_builder = context_builder

        self._last_indexed_turn: int = 0
        self._indexing_in_progress: bool = False
        self._total_entries_created: int = 0

        self.promotion_threshold: float = 1.0
        self.demotion_threshold: float = 0.3
        self.recency_half_life: float = 24.0
        self.age_boost_half_life: float = 48.0

        self._recent_tier_events: List[TierEvent] = []

        self.pressure_monitor = PressureMonitor()
        self._preloader: Optional[Preloader] = None
        self._last_preload_result: Optional[PreloadResult] = None

    async def index_new_messages(
        self,
        messages: List[Message],
        conversation_id: str,
    ) -> List[RolodexEntry]:


        if self._indexing_in_progress:
            return []
        self._indexing_in_progress = True
        try:

            new_messages = [
                m for m in messages
                if m.turn_number > self._last_indexed_turn
            ]
            if not new_messages:
                return []

            msg_dicts = [
                {"role": m.role.value, "content": m.content}
                for m in new_messages
            ]

            entries = await self.extractor.extract_from_messages(
                messages=msg_dicts,
                conversation_id=conversation_id,
                turn_number=new_messages[-1].turn_number,
            )

            if entries:
                self.rolodex.batch_create_entries(entries)
                self._total_entries_created += len(entries)

            self._last_indexed_turn = new_messages[-1].turn_number
            return entries
        finally:
            self._indexing_in_progress = False


    async def process_enrichment_task(self, task: IngestionTask) -> None:


        if not task.message:
            return

        msg = task.message
        content = msg.content
        if not content or len(content.strip()) < 20:

            for stub_id in task.stub_entry_ids:
                self.rolodex.update_entry_metadata(stub_id, {
                    "enrichment_status": "skipped",
                })
            return


        msg_dicts = [{"role": msg.role.value, "content": content}]
        entries = await self.extractor.extract_from_messages(
            messages=msg_dicts,
            conversation_id=task.conversation_id,
            turn_number=task.turn_number,
        )

        if entries and task.stub_entry_ids:

            primary_entry = entries[0]
            stub_id = task.stub_entry_ids[0]
            self.rolodex.update_entry_enrichment(
                entry_id=stub_id,
                content_type=primary_entry.content_type,
                category=primary_entry.category,
                tags=primary_entry.tags,
                embedding=primary_entry.embedding,
                metadata={"enrichment_status": "completed"},
            )


            if len(entries) > 1:
                extra_entries = entries[1:]
                self.rolodex.batch_create_entries(extra_entries)
                self._total_entries_created += len(extra_entries)

        elif task.stub_entry_ids:

            for stub_id in task.stub_entry_ids:
                self.rolodex.update_entry_metadata(stub_id, {
                    "enrichment_status": "completed_empty",
                })


    async def answer_query(
        self,
        query: LibrarianQuery,
        cross_session: bool = False,
        session_boost_factor: float = 1.5,
    ) -> LibrarianResponse:


        response = await self.searcher.search(
            query,
            cross_session=cross_session,
            session_boost_factor=session_boost_factor,
        )

        if response.found:
            self._evaluate_accessed_entries(response.entries)
        return response

    def _evaluate_accessed_entries(self, entries: List[RolodexEntry]) -> None:

        from ..core.types import Tier
        for entry in entries:
            recommended_tier, score = self.rolodex.evaluate_tier(
                entry.id,
                promotion_threshold=self.promotion_threshold,
                demotion_threshold=self.demotion_threshold,
                recency_half_life=self.recency_half_life,
                age_boost_half_life=self.age_boost_half_life,
            )
            if recommended_tier == Tier.HOT and entry.tier == Tier.COLD:
                event = self.rolodex.promote_entry(entry.id)
                if event:
                    self._recent_tier_events.append(event)
            elif recommended_tier == Tier.COLD and entry.tier == Tier.HOT:
                event = self.rolodex.demote_entry(entry.id)
                if event:
                    self._recent_tier_events.append(event)
    def get_context_for_query(
        self,
        response: LibrarianResponse,
        current_session_id: Optional[str] = None,
    ) -> str:


        if response.found:
            return self.context_builder.build_context_block(
                response.entries,
                current_session_id=current_session_id,
                chains=getattr(response, 'chains', None),
            )
        else:
            query_text = response.query.query_text if response.query else "unknown"
            return self.context_builder.build_not_found_message(query_text)


    def init_preloader(
        self,
        llm_adapter: Optional[LLMAdapter] = None,
        context_max: int = 180_000,
    ) -> None:

        self.pressure_monitor = PressureMonitor(context_max=context_max)
        self._preloader = Preloader(
            rolodex=self.rolodex,
            embedding_manager=self.embeddings,
            pressure_monitor=self.pressure_monitor,
            llm_adapter=llm_adapter,
        )

    async def preload(
        self,
        recent_messages: List[Message],
        turn_number: int,
        conversation_id: str = "",
        max_entries: int = 5,
        injection_confidence: float = 0.8,
        low_threshold: float = 0.3,
        high_threshold: float = 0.7,
    ) -> Optional[PreloadResult]:


        if self._preloader is None:
            return None
        result = await self._preloader.preload(
            recent_messages=recent_messages,
            turn_number=turn_number,
            conversation_id=conversation_id,
            max_entries=max_entries,
            injection_confidence=injection_confidence,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        self._last_preload_result = result
        return result

    def get_proactive_context(self, preload_result: PreloadResult) -> str:

        if not preload_result.injected_entries:
            return ""
        return self.context_builder.build_proactive_context_block(
            preload_result.injected_entries,
            strategy=preload_result.strategy_used,
        )


    def run_maintenance(self) -> Dict:


        sweep_result = self.rolodex.run_tier_sweep(
            promotion_threshold=self.promotion_threshold,
            demotion_threshold=self.demotion_threshold,
            recency_half_life=self.recency_half_life,
            age_boost_half_life=self.age_boost_half_life,
        )

        self._recent_tier_events.extend(sweep_result.get("events", []))
        return sweep_result

    def drain_tier_events(self) -> List[TierEvent]:

        events = list(self._recent_tier_events)
        self._recent_tier_events.clear()
        return events


    def get_stats(self) -> Dict:

        rolodex_stats = self.rolodex.get_stats()
        return {
            **rolodex_stats,
            "last_indexed_turn": self._last_indexed_turn,
            "indexing_in_progress": self._indexing_in_progress,
            "total_entries_created": self._total_entries_created,
        }
