"""
The Librarian — Trajectory Predictors (Phase 3)

Two prediction strategies behind a common interface:

1. EmbeddingPredictor: Embed the latest user message, find nearby
   rolodex entries by cosine similarity. Cheap, no LLM calls.

2. LLMPredictor: Uses an LLMAdapter to predict what topics the
   conversation will need next. More accurate, costs an API call.
   Used when session pressure is high.
"""
from typing import List, Optional

from ..core.types import (
    Message, RolodexEntry, PreloadPrediction
)
from ..core.llm_adapter import LLMAdapter
from ..storage.rolodex import Rolodex
from ..indexing.embeddings import EmbeddingManager


class EmbeddingPredictor:
    """
    Predict relevant entries by embedding proximity.

    Takes the last user message, computes its embedding, and finds
    the closest entries in the rolodex by cosine similarity.
    The similarity score becomes the prediction confidence.
    """

    def __init__(self, rolodex: Rolodex, embedding_manager: EmbeddingManager):
        self.rolodex = rolodex
        self.embeddings = embedding_manager

    async def predict(
        self,
        recent_messages: List[Message],
        limit: int = 5,
        min_similarity: float = 0.4,
    ) -> List[PreloadPrediction]:
        """
        Find rolodex entries semantically close to recent conversation.
        Returns predictions sorted by confidence (highest first).
        """
        if not recent_messages:
            return []

        # Use the last user message as the query
        last_user = None
        for msg in reversed(recent_messages):
            if msg.role.value == "user":
                last_user = msg
                break
        if not last_user:
            return []

        # Compute embedding
        try:
            query_embedding = await self.embeddings.embed_text(last_user.content)
        except Exception:
            return []

        if not query_embedding:
            return []

        # Search rolodex by semantic similarity
        results = self.rolodex.semantic_search(
            query_embedding=query_embedding,
            limit=limit,
            min_similarity=min_similarity,
        )

        # Convert to predictions (similarity = confidence)
        predictions = []
        for entry, similarity in results:
            predictions.append(PreloadPrediction(
                entry_id=entry.id,
                confidence=similarity,
                strategy="embedding",
                query_text=last_user.content[:100],
            ))
        return predictions


class LLMPredictor:
    """
    Predict relevant entries by asking an LLM via adapter.

    Sends recent conversation context to the adapter and asks it to
    predict what topics the user will ask about next. Then searches
    the rolodex for those predicted topics.

    Only used when session pressure is high — this costs an API call.
    Requires an LLMAdapter; returns empty if none provided.
    """

    def __init__(
        self,
        rolodex: Rolodex,
        embedding_manager: EmbeddingManager,
        llm_adapter: Optional[LLMAdapter] = None,
    ):
        self.rolodex = rolodex
        self.embeddings = embedding_manager
        self._adapter = llm_adapter

    async def predict(
        self,
        recent_messages: List[Message],
        limit: int = 5,
    ) -> List[PreloadPrediction]:
        """
        Ask the LLM adapter to predict next topics, then search
        rolodex for them. Returns predictions sorted by confidence.
        """
        if not recent_messages or not self._adapter:
            return []

        # Get topic predictions from LLM adapter
        try:
            predicted_topics = await self._adapter.predict_topics(
                recent_messages
            )
        except Exception:
            return []

        if not predicted_topics:
            return []

        # Search rolodex for each predicted topic
        predictions = []
        seen_ids = set()

        for topic_item in predicted_topics[:3]:
            topic = topic_item.get("topic", "")
            confidence = float(topic_item.get("confidence", 0.5))
            if not topic:
                continue

            # Try embedding search for the predicted topic
            try:
                query_emb = await self.embeddings.embed_text(topic)
                if query_emb:
                    results = self.rolodex.semantic_search(
                        query_embedding=query_emb,
                        limit=limit,
                        min_similarity=0.3,
                    )
                    for entry, sim in results:
                        if entry.id not in seen_ids:
                            seen_ids.add(entry.id)
                            combined = confidence * sim
                            predictions.append(PreloadPrediction(
                                entry_id=entry.id,
                                confidence=combined,
                                strategy="llm",
                                query_text=topic,
                            ))
            except Exception:
                continue

            # Also try keyword search as fallback
            try:
                kw_results = self.rolodex.keyword_search(
                    query=topic, limit=3
                )
                for entry, score in kw_results:
                    if entry.id not in seen_ids:
                        seen_ids.add(entry.id)
                        combined = confidence * min(1.0, score)
                        predictions.append(PreloadPrediction(
                            entry_id=entry.id,
                            confidence=combined,
                            strategy="llm",
                            query_text=topic,
                        ))
            except Exception:
                continue

        # Sort by confidence, limit
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions[:limit]
