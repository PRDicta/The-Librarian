"""
The Librarian — Preload Orchestrator (Phase 3)

Coordinates prediction strategies with cache warming.
Like ABR streaming: adapts bandwidth to match conversation demand.

High-confidence predictions → inject proactively into context
Lower-confidence predictions → warm into hot cache for fast retrieval
"""
import json
import uuid
from typing import List, Optional
from datetime import datetime
from ..core.types import (
    Message, RolodexEntry, PreloadPrediction, PreloadResult
)
from ..core.llm_adapter import LLMAdapter
from ..storage.rolodex import Rolodex
from ..indexing.embeddings import EmbeddingManager
from .pressure import PressureMonitor
from .predictor import EmbeddingPredictor, LLMPredictor


class Preloader:
    """
    Orchestrates the preloading pipeline:
    1. Check pressure → choose strategy
    2. Run prediction (embedding or LLM)
    3. Split results by confidence threshold
    4. High confidence → mark for proactive injection
    5. All predictions → warm into hot cache
    6. Log for accuracy tracking
    """

    def __init__(
        self,
        rolodex: Rolodex,
        embedding_manager: EmbeddingManager,
        pressure_monitor: PressureMonitor,
        llm_adapter: Optional[LLMAdapter] = None,
    ):
        self.rolodex = rolodex
        self.pressure = pressure_monitor
        self.embedding_predictor = EmbeddingPredictor(rolodex, embedding_manager)
        self.llm_predictor = LLMPredictor(
            rolodex, embedding_manager,
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
    ) -> PreloadResult:
        """
        Run the full preload pipeline.

        Returns a PreloadResult with:
        - injected_entries: high-confidence, ready for proactive context
        - cache_warmed_entries: loaded into hot cache for fast retrieval
        - metadata: strategy used, pressure level, predictions
        """
        pressure_val = self.pressure.get_pressure()
        strategy = self.pressure.get_strategy(low_threshold, high_threshold)
        limit = self.pressure.get_max_entries(
            low_threshold, high_threshold, max_entries
        )

        # No preloading if insufficient data
        if strategy == "none":
            return PreloadResult(
                strategy_used="none",
                pressure=pressure_val,
                turn_number=turn_number,
            )

        # Run prediction(s)
        predictions: List[PreloadPrediction] = []

        # Always run embedding predictor (cheap)
        embed_preds = await self.embedding_predictor.predict(
            recent_messages, limit=limit
        )
        predictions.extend(embed_preds)

        # Escalate to LLM if high pressure
        if strategy == "llm":
            llm_preds = await self.llm_predictor.predict(
                recent_messages, limit=limit
            )
            # Merge, dedup by entry_id (keep highest confidence)
            seen = {p.entry_id: p for p in predictions}
            for p in llm_preds:
                if p.entry_id not in seen or p.confidence > seen[p.entry_id].confidence:
                    seen[p.entry_id] = p
            predictions = sorted(
                seen.values(), key=lambda p: p.confidence, reverse=True
            )[:limit]

        # Split by confidence threshold
        injected_entries: List[RolodexEntry] = []
        cache_warmed_entries: List[RolodexEntry] = []

        for pred in predictions:
            entry = self.rolodex.get_entry(pred.entry_id)
            if entry is None:
                continue

            if pred.confidence >= injection_confidence:
                injected_entries.append(entry)
            else:
                cache_warmed_entries.append(entry)

            # Warm into hot cache (WITHOUT updating access count —
            # preloading is speculative, don't inflate real usage stats)
            self.rolodex._cache_put(entry)

        # Log the preload event
        self._log_preload(
            conversation_id=conversation_id,
            turn_number=turn_number,
            strategy=strategy,
            pressure=pressure_val,
            predictions=predictions,
            injected_ids=[e.id for e in injected_entries],
            cache_warmed_ids=[e.id for e in cache_warmed_entries],
        )

        return PreloadResult(
            injected_entries=injected_entries,
            cache_warmed_entries=cache_warmed_entries,
            predictions=predictions,
            strategy_used=strategy,
            pressure=pressure_val,
            turn_number=turn_number,
        )

    def _log_preload(
        self,
        conversation_id: str,
        turn_number: int,
        strategy: str,
        pressure: float,
        predictions: List[PreloadPrediction],
        injected_ids: List[str],
        cache_warmed_ids: List[str],
    ) -> None:
        """Log preload event to the preload_log table."""
        try:
            self.rolodex.conn.execute(
                """INSERT INTO preload_log
                   (id, conversation_id, turn_number, strategy, pressure,
                    predicted_entry_ids, injected_entry_ids,
                    cache_warmed_entry_ids, hit_entry_ids, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    conversation_id,
                    turn_number,
                    strategy,
                    pressure,
                    json.dumps([p.entry_id for p in predictions]),
                    json.dumps(injected_ids),
                    json.dumps(cache_warmed_ids),
                    json.dumps([]),  # hit_entry_ids filled in later
                    datetime.utcnow().isoformat(),
                )
            )
            self.rolodex.conn.commit()
        except Exception:
            pass  # Don't let logging failures break preloading
