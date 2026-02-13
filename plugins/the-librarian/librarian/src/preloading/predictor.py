


from typing import List, Optional

from ..core.types import (
    Message, RolodexEntry, PreloadPrediction
)
from ..core.llm_adapter import LLMAdapter
from ..storage.rolodex import Rolodex
from ..indexing.embeddings import EmbeddingManager


class EmbeddingPredictor:


    def __init__(self, rolodex: Rolodex, embedding_manager: EmbeddingManager):
        self.rolodex = rolodex
        self.embeddings = embedding_manager

    async def predict(
        self,
        recent_messages: List[Message],
        limit: int = 5,
        min_similarity: float = 0.4,
    ) -> List[PreloadPrediction]:


        if not recent_messages:
            return []


        last_user = None
        for msg in reversed(recent_messages):
            if msg.role.value == "user":
                last_user = msg
                break
        if not last_user:
            return []


        try:
            query_embedding = await self.embeddings.embed_text(last_user.content)
        except Exception:
            return []

        if not query_embedding:
            return []


        results = self.rolodex.semantic_search(
            query_embedding=query_embedding,
            limit=limit,
            min_similarity=min_similarity,
        )


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


        if not recent_messages or not self._adapter:
            return []


        try:
            predicted_topics = await self._adapter.predict_topics(
                recent_messages
            )
        except Exception:
            return []

        if not predicted_topics:
            return []


        predictions = []
        seen_ids = set()

        for topic_item in predicted_topics[:3]:
            topic = topic_item.get("topic", "")
            confidence = float(topic_item.get("confidence", 0.5))
            if not topic:
                continue


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


        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions[:limit]
