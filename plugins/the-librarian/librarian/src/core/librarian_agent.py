"""
The Librarian — Librarian Agent
Manages the rolodex:
- Indexes conversation content (extraction pipeline)
- Answers queries from the working agent (or middleware)
- Manages hot cache with LRU eviction
- Phase 2: frequency-based tier promotion/demotion via weighted importance scoring
- Phase 3: proactive preloading with session-pressure-driven strategy selection
- Phase 8: background enrichment via IngestionQueue (100% ingestion)
"""
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
    """
    The Librarian — discrete agent managing the rolodex.
    Runs alongside the working agent, handling all memory operations.
    """
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
        # State tracking
        self._last_indexed_turn: int = 0
        self._indexing_in_progress: bool = False
        self._total_entries_created: int = 0
        # Tier management config (set by orchestrator from LibrarianConfig)
        self.promotion_threshold: float = 1.0
        self.demotion_threshold: float = 0.3
        self.recency_half_life: float = 24.0
        self.age_boost_half_life: float = 48.0
        # Tier event log for debug output
        self._recent_tier_events: List[TierEvent] = []
        # Phase 3: preloading
        self.pressure_monitor = PressureMonitor()
        self._preloader: Optional[Preloader] = None
        self._last_preload_result: Optional[PreloadResult] = None
    # ─── Indexing ─────────────────────────────────────────────────────────
    async def index_new_messages(
        self,
        messages: List[Message],
        conversation_id: str,
    ) -> List[RolodexEntry]:
        """
        Index new messages that haven't been processed yet.
        Called after each user/assistant exchange.
        Returns list of newly created rolodex entries.
        """
        if self._indexing_in_progress:
            return []
        self._indexing_in_progress = True
        try:
            # Find messages we haven't indexed yet
            new_messages = [
                m for m in messages
                if m.turn_number > self._last_indexed_turn
            ]
            if not new_messages:
                return []
            # Convert to the format the extractor expects
            msg_dicts = [
                {"role": m.role.value, "content": m.content}
                for m in new_messages
            ]
            # Extract entries
            entries = await self.extractor.extract_from_messages(
                messages=msg_dicts,
                conversation_id=conversation_id,
                turn_number=new_messages[-1].turn_number,
            )
            # Store in rolodex
            if entries:
                self.rolodex.batch_create_entries(entries)
                self._total_entries_created += len(entries)
            # Update state
            self._last_indexed_turn = new_messages[-1].turn_number
            return entries
        finally:
            self._indexing_in_progress = False
    # ─── Background Enrichment (Phase 8) ─────────────────────────────────

    async def process_enrichment_task(self, task: IngestionTask) -> None:
        """
        Process a single enrichment task from the IngestionQueue.
        Called by background workers — runs the full extraction + embedding
        pipeline on a raw stub entry, then updates it in the DB.

        This is the same work that index_new_messages() does synchronously,
        but decoupled for background execution.
        """
        if not task.message:
            return

        msg = task.message
        content = msg.content
        if not content or len(content.strip()) < 20:
            # Mark stubs as enriched even if content is too short
            for stub_id in task.stub_entry_ids:
                self.rolodex.update_entry_metadata(stub_id, {
                    "enrichment_status": "skipped",
                })
            return

        # Run extraction pipeline (chunking → extraction → embedding)
        msg_dicts = [{"role": msg.role.value, "content": content}]
        entries = await self.extractor.extract_from_messages(
            messages=msg_dicts,
            conversation_id=task.conversation_id,
            turn_number=task.turn_number,
        )

        if entries and task.stub_entry_ids:
            # Update the first stub with enriched data from first extracted entry
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

            # If extraction produced additional entries beyond the stub,
            # store them as new entries
            if len(entries) > 1:
                extra_entries = entries[1:]
                self.rolodex.batch_create_entries(extra_entries)
                self._total_entries_created += len(extra_entries)

        elif task.stub_entry_ids:
            # Extraction produced nothing — mark stub as enriched anyway
            for stub_id in task.stub_entry_ids:
                self.rolodex.update_entry_metadata(stub_id, {
                    "enrichment_status": "completed_empty",
                })

    # ─── Query Handling ──────────────────────────────────────────────────
    async def answer_query(
        self,
        query: LibrarianQuery,
        cross_session: bool = False,
        session_boost_factor: float = 1.5,
    ) -> LibrarianResponse:
        """
        Process a query from the working agent.
        Searches the rolodex and returns relevant entries.
        Phase 4: supports cross-session search with session boosting.
        After retrieval, evaluates tiers for accessed entries.
        """
        response = await self.searcher.search(
            query,
            cross_session=cross_session,
            session_boost_factor=session_boost_factor,
        )
        # Phase 2: evaluate tiers for all returned entries
        if response.found:
            self._evaluate_accessed_entries(response.entries)
        return response

    def _evaluate_accessed_entries(self, entries: List[RolodexEntry]) -> None:
        """After entries are accessed via search, check if any should change tier."""
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
        """
        Format a LibrarianResponse into a context block for injection
        into the working agent's prompt.
        Phase 4: passes current_session_id so cross-session entries are tagged.
        Phase 7: passes reasoning chains so narrative context appears first.
        """
        if response.found:
            return self.context_builder.build_context_block(
                response.entries,
                current_session_id=current_session_id,
                chains=getattr(response, 'chains', None),
            )
        else:
            query_text = response.query.query_text if response.query else "unknown"
            return self.context_builder.build_not_found_message(query_text)
    # ─── Proactive Preloading (Phase 3) ──────────────────────────────────

    def init_preloader(
        self,
        llm_adapter: Optional[LLMAdapter] = None,
        context_max: int = 180_000,
    ) -> None:
        """Initialize the preloader (called by orchestrator/middleware after construction)."""
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
        """
        Run proactive preloading. Returns PreloadResult or None if
        preloader not initialized.
        """
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
        """Format high-confidence preloaded entries for context injection."""
        if not preload_result.injected_entries:
            return ""
        return self.context_builder.build_proactive_context_block(
            preload_result.injected_entries,
            strategy=preload_result.strategy_used,
        )

    # ─── Tier Maintenance ─────────────────────────────────────────────────

    def run_maintenance(self) -> Dict:
        """
        Periodic tier sweep — scan all entries, promote/demote based on
        importance scores. Called by orchestrator every N turns.
        """
        sweep_result = self.rolodex.run_tier_sweep(
            promotion_threshold=self.promotion_threshold,
            demotion_threshold=self.demotion_threshold,
            recency_half_life=self.recency_half_life,
            age_boost_half_life=self.age_boost_half_life,
        )
        # Collect events for debug
        self._recent_tier_events.extend(sweep_result.get("events", []))
        return sweep_result

    def drain_tier_events(self) -> List[TierEvent]:
        """Pop and return all recent tier events (for debug logging)."""
        events = list(self._recent_tier_events)
        self._recent_tier_events.clear()
        return events

    # ─── Stats ───────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return Librarian operational stats."""
        rolodex_stats = self.rolodex.get_stats()
        return {
            **rolodex_stats,
            "last_indexed_turn": self._last_indexed_turn,
            "indexing_in_progress": self._indexing_in_progress,
            "total_entries_created": self._total_entries_created,
        }
