


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


        pressure_val = self.pressure.get_pressure()
        strategy = self.pressure.get_strategy(low_threshold, high_threshold)
        limit = self.pressure.get_max_entries(
            low_threshold, high_threshold, max_entries
        )


        if strategy == "none":
            return PreloadResult(
                strategy_used="none",
                pressure=pressure_val,
                turn_number=turn_number,
            )


        predictions: List[PreloadPrediction] = []


        embed_preds = await self.embedding_predictor.predict(
            recent_messages, limit=limit
        )
        predictions.extend(embed_preds)


        if strategy == "llm":
            llm_preds = await self.llm_predictor.predict(
                recent_messages, limit=limit
            )

            seen = {p.entry_id: p for p in predictions}
            for p in llm_preds:
                if p.entry_id not in seen or p.confidence > seen[p.entry_id].confidence:
                    seen[p.entry_id] = p
            predictions = sorted(
                seen.values(), key=lambda p: p.confidence, reverse=True
            )[:limit]


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


            self.rolodex._cache_put(entry)


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
                    json.dumps([]),
                    datetime.utcnow().isoformat(),
                )
            )
            self.rolodex.conn.commit()
        except Exception:
            pass
