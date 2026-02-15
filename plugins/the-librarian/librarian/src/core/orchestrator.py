"""
The Librarian — Orchestrator (Demo Harness)

The central hub managing message flow between:
- User (via CLI)
- Working Agent (Claude Sonnet/Opus) — optional, needs API key
- TheLibrarian middleware (memory layer) — always available

Two modes:
1. Full demo (API key provided): Working Agent + TheLibrarian + gap detection
2. Middleware mode (no API key): TheLibrarian only, REPL for /search + /ingest

Implements the three data flows:
1. Standard: User → Working Agent → Response
2. Retrieval: Working Agent gap → TheLibrarian → Context injection → Response
3. Fallback: TheLibrarian not found → Working Agent asks user
"""
import asyncio
from typing import Optional, List, Dict, Any

from .types import (
    Message, MessageRole, ConversationState,
    LibrarianQuery, LibrarianResponse, SessionInfo, estimate_tokens
)
from .librarian import TheLibrarian
from .gap_detector import extract_gap_topic
from ..utils.config import LibrarianConfig


class Orchestrator:
    """
    Main orchestration layer for the demo CLI.
    Routes messages, manages context, and coordinates the
    working agent with TheLibrarian middleware.
    """

    def __init__(self, config: LibrarianConfig):
        self.config = config

        # Cost tracking (Phase 6a)
        self.cost_tracker = None
        if config.cost_tracking_enabled:
            from ..utils.cost_tracker import CostTracker
            self.cost_tracker = CostTracker()

        # Build LLM adapter if API key available
        llm_adapter = None
        if config.has_api_key:
            try:
                from ..indexing.anthropic_adapter import AnthropicAdapter
                llm_adapter = AnthropicAdapter(
                    api_key=config.anthropic_api_key,
                    extraction_model=config.librarian_model,
                    prediction_model=config.librarian_model,
                    cost_tracker=self.cost_tracker,
                )
            except ImportError:
                pass  # anthropic not installed, stay in verbatim mode

        # Create TheLibrarian middleware (always available, no API needed)
        self.middleware = TheLibrarian(
            db_path=config.db_path,
            llm_adapter=llm_adapter,
            config=config,
        )

        # Expose state from middleware for compatibility
        self.state = self.middleware.state

        # Create Working Agent only if API key available
        self.working_agent = None
        if config.has_api_key:
            try:
                from .working_agent import WorkingAgent
                self.working_agent = WorkingAgent(
                    api_key=config.anthropic_api_key,
                    model=config.working_agent_model,
                )
            except ImportError:
                pass  # anthropic not installed

        # Context Negotiator (Phase 6c) — Haiku-mediated context injection
        self.negotiator = None
        if config.has_api_key and config.negotiation_enabled:
            try:
                from .negotiator import ContextNegotiator
                self.negotiator = ContextNegotiator(
                    api_key=config.anthropic_api_key,
                    model=config.negotiation_model,
                    max_rounds=config.negotiation_max_rounds,
                    cost_tracker=self.cost_tracker,
                )
            except ImportError:
                pass

        # Chain Builder (Phase 7) — breadcrumb generation
        self.chain_builder = None
        try:
            from .chain_builder import ChainBuilder
            self.chain_builder = ChainBuilder(
                rolodex=self.middleware.rolodex,
                embedding_manager=self.middleware.embeddings,
                llm_adapter=llm_adapter,
                chain_interval=config.chain_interval,
                cost_tracker=self.cost_tracker,
            )
        except ImportError:
            pass
        self._last_chain_turn = 0
        self._recent_entry_ids: List[str] = []  # Track entries for chain linking

        # Preload count from middleware
        self._preloaded_count = self.middleware._preloaded_count

        # Debug callback (set by CLI)
        self._debug_callback = None

    @property
    def has_working_agent(self) -> bool:
        """Whether a working agent (LLM chat) is available."""
        return self.working_agent is not None

    # ─── Main Message Flow ────────────────────────────────────────────────

    async def process_message(self, user_input: str) -> Dict[str, Any]:
        """
        Main entry point. Process a user message and return the response.

        In full mode (with API key): queries Working Agent, detects gaps,
        retrieves context, re-queries if needed.

        In middleware mode (no API key): ingests the message, returns
        a placeholder indicating no working agent is available.
        """
        debug = {
            "librarian_active": False,
            "entries_indexed": 0,
            "gap_detected": False,
            "gap_topic": None,
            "retrieval_performed": False,
            "retrieval_found": False,
            "retrieval_entries": 0,
            "cache_hit": False,
            "search_time_ms": 0,
            "tier_events": [],
            "tier_sweep_performed": False,
            "preload_performed": False,
            "preload_strategy": "none",
            "preload_pressure": 0.0,
            "preload_injected": 0,
            "preload_cached": 0,
        }

        # 1. Ingest user message into middleware
        entries = await self.middleware.ingest("user", user_input)
        debug["entries_indexed"] = len(entries) if entries else 0

        # Track entry IDs for chain linking (Phase 7)
        if entries:
            self._recent_entry_ids.extend(e.id for e in entries)

        # Sync state reference
        self.state = self.middleware.state

        # 2. Check if Librarian should activate
        librarian_active = self.state.should_activate_librarian(
            self.config.librarian_activation_tokens
        )
        debug["librarian_active"] = librarian_active

        # 2a. Record token pressure
        if librarian_active:
            self.middleware.librarian_agent.pressure_monitor.record_tokens(
                self.state.turn_count, self.state.total_tokens
            )

        # 3. If no working agent, we're in middleware-only mode
        if not self.working_agent:
            return {
                "response": None,  # Signals to CLI: no LLM response
                "debug": debug,
            }

        # ─── Full demo mode (with Working Agent) ──────────────────────

        # 3a. Start preloading concurrently
        preload_task = None
        if librarian_active and self.config.preload_enabled:
            recent_messages = self._get_recent_messages()
            preload_task = asyncio.create_task(
                self.middleware.preload(recent_messages)
            )

        # 3b. Await preload for proactive context
        proactive_context = None
        if preload_task:
            try:
                preload_result = await preload_task
                if preload_result:
                    debug["preload_performed"] = True
                    debug["preload_strategy"] = preload_result.strategy_used
                    debug["preload_pressure"] = round(preload_result.pressure, 3)
                    debug["preload_injected"] = len(preload_result.injected_entries)
                    debug["preload_cached"] = len(preload_result.cache_warmed_entries)
                    proactive_context = self.middleware.librarian_agent.get_proactive_context(
                        preload_result
                    ) or None
            except Exception:
                pass

        # 4. Query working agent
        recent_messages = self._get_recent_messages()
        response_text = await self.working_agent.query(
            messages=recent_messages,
            user_input=user_input,
            proactive_context=proactive_context,
        )

        # 5. Check for gap signal (regex, no LLM)
        if librarian_active:
            gap_topic = extract_gap_topic(response_text)
            debug["gap_detected"] = gap_topic is not None
            debug["gap_topic"] = gap_topic

            # Record gap for pressure tracking
            if gap_topic:
                self.middleware.librarian_agent.pressure_monitor.record_gap(
                    self.state.turn_count
                )

                # 6. Retrieve candidates with scores
                candidates, scores = await self.middleware.retrieve_with_scores(
                    gap_topic
                )

                debug["retrieval_performed"] = True
                debug["retrieval_found"] = len(candidates) > 0
                debug["retrieval_entries"] = len(candidates)

                # Record query for pressure tracking
                self.middleware.librarian_agent.pressure_monitor.record_query(
                    self.state.turn_count, False
                )

                if candidates:
                    # 6a. Negotiate which entries to inject (Phase 6c)
                    if self.negotiator:
                        # Build search function for potential round 2
                        async def _refined_search(query):
                            return await self.middleware.retrieve_with_scores(query)

                        negotiation = await self.negotiator.negotiate(
                            gap_topic=gap_topic,
                            candidate_entries=candidates,
                            relevance_scores=scores,
                            budget_tokens=self.config.max_context_for_retrieval,
                            search_fn=_refined_search,
                        )

                        debug["negotiation_rounds"] = negotiation.total_rounds
                        debug["negotiation_accepted"] = len(negotiation.accepted_entries)
                        debug["negotiation_rejected"] = len(negotiation.rejected_ids)
                        debug["negotiation_budget_used"] = negotiation.budget_used
                        debug["negotiation_resolved"] = negotiation.resolved

                        # Record negotiation outcome for pressure tracking
                        if hasattr(self.middleware.librarian_agent, 'pressure_monitor'):
                            pm = self.middleware.librarian_agent.pressure_monitor
                            if hasattr(pm, 'record_negotiation'):
                                pm.record_negotiation(
                                    negotiation.resolved,
                                    negotiation.budget_used,
                                    negotiation.total_rounds,
                                )

                        injected_entries = negotiation.accepted_entries
                    else:
                        # No negotiator — fall back to injecting all (pre-6c behavior)
                        injected_entries = candidates

                    if injected_entries:
                        # 7. Build context block from accepted entries only
                        from .types import LibrarianResponse
                        negotiated_response = LibrarianResponse(
                            found=True,
                            entries=injected_entries,
                            query_text=gap_topic,
                        )
                        retrieved_context = self.middleware.get_context_block(
                            negotiated_response
                        )
                        response_text = await self.working_agent.query(
                            messages=recent_messages,
                            user_input=user_input,
                            retrieved_context=retrieved_context,
                            proactive_context=proactive_context,
                        )

        # 8. Ingest assistant response
        resp_entries = await self.middleware.ingest("assistant", response_text)
        if resp_entries:
            self._recent_entry_ids.extend(e.id for e in resp_entries)
        self.state = self.middleware.state

        # 8a. Phase 7: Generate breadcrumb if interval reached
        if self.chain_builder and librarian_active:
            try:
                if self.chain_builder.should_generate_breadcrumb(
                    self.state.turn_count, self._last_chain_turn
                ):
                    breadcrumb = await self.chain_builder.build_breadcrumb(
                        session_id=self.state.conversation_id,
                        messages=self.state.messages,
                        turn_range_start=self._last_chain_turn + 1,
                        turn_range_end=self.state.turn_count,
                        related_entry_ids=list(self._recent_entry_ids),
                    )
                    if breadcrumb:
                        self.middleware.rolodex.create_chain(breadcrumb)
                        debug["chain_generated"] = True
                        debug["chain_summary"] = breadcrumb.summary[:100]
                        self._last_chain_turn = self.state.turn_count
                        self._recent_entry_ids = []  # Reset for next interval
            except Exception:
                pass  # Chain generation is never a blocker

            # 8b. Phase 7d: Emergency snapshot if pressure is high
            try:
                pm = self.middleware.librarian_agent.pressure_monitor
                if hasattr(pm, 'should_trigger_deep_index'):
                    if pm.should_trigger_deep_index(
                        self.config.chain_deep_index_threshold
                    ):
                        emergency = await self.chain_builder.build_emergency_snapshot(
                            session_id=self.state.conversation_id,
                            messages=self.state.messages,
                            related_entry_ids=list(self._recent_entry_ids),
                        )
                        if emergency:
                            self.middleware.rolodex.create_chain(emergency)
                            debug["emergency_chain"] = True
                            debug["emergency_summary"] = emergency.summary[:100]
                            self._last_chain_turn = self.state.turn_count
                            self._recent_entry_ids = []
            except Exception:
                pass

        # 9. Periodic tier sweep
        if librarian_active and self.state.turn_count % self.config.tier_sweep_interval == 0:
            sweep = self.middleware.run_maintenance()
            debug["tier_sweep_performed"] = True
            debug["tier_sweep_summary"] = {
                "scanned": sweep.get("entries_scanned", 0),
                "promoted": sweep.get("promoted", 0),
                "demoted": sweep.get("demoted", 0),
            }

        # 10. Collect tier events
        tier_events = self.middleware.librarian_agent.drain_tier_events()
        debug["tier_events"] = tier_events

        # 11. Debug callback
        if self._debug_callback:
            self._debug_callback(debug)

        return {
            "response": response_text,
            "debug": debug,
        }

    # ─── Direct Search ─────────────────────────────────────────────────────

    async def search_rolodex(self, query_text: str) -> LibrarianResponse:
        """Directly search the rolodex (for CLI /search command)."""
        return await self.middleware.search(query_text)

    # ─── Context Window ────────────────────────────────────────────────────

    def _get_recent_messages(self) -> List[Message]:
        """Get the most recent messages that fit in the context window."""
        return self.middleware._get_recent_messages(
            self.config.max_context_for_history
        )

    # ─── Session Management (delegates to middleware) ──────────────────────

    def resume_session(self, conversation_id: str) -> Optional[SessionInfo]:
        """Resume a previous session."""
        result = self.middleware.resume_session(conversation_id)
        self.state = self.middleware.state
        return result

    def list_sessions(self, limit: int = 20) -> List[SessionInfo]:
        """List recent sessions."""
        return self.middleware.list_sessions(limit=limit)

    def find_session(self, prefix: str) -> Optional[str]:
        """Find a session ID by prefix."""
        return self.middleware.find_session(prefix)

    # ─── Stats & Management ───────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return full system stats."""
        stats = self.middleware.get_stats()
        stats["has_working_agent"] = self.has_working_agent
        stats["mode"] = "full" if self.has_working_agent else "middleware"
        stats["negotiation_enabled"] = self.negotiator is not None
        if self.cost_tracker:
            stats["api_costs"] = self.cost_tracker.get_summary()
        return stats

    def set_debug_callback(self, callback):
        """Set a callback for debug events."""
        self._debug_callback = callback

    async def shutdown(self):
        """Clean shutdown."""
        await self.middleware.shutdown()
