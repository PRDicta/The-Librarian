


import uuid
from typing import List, Dict, Optional

from ..core.types import (
    RolodexEntry, ContentModality, EntryCategory, Tier
)
from ..core.llm_adapter import LLMAdapter
from .chunker import ContentChunker
from .embeddings import EmbeddingManager
from .verbatim_extractor import VerbatimExtractor


class EntryExtractor:


    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        chunker: Optional[ContentChunker] = None,
        llm_adapter: Optional[LLMAdapter] = None,
    ):
        self.embeddings = embedding_manager or EmbeddingManager()
        self.chunker = chunker or ContentChunker()

        self._extractor = llm_adapter if llm_adapter else VerbatimExtractor()

    async def extract_from_messages(
        self,
        messages: List[Dict[str, str]],
        conversation_id: str,
        turn_number: int = 0,
    ) -> List[RolodexEntry]:


        all_entries = []

        for msg in messages:
            content = msg.get("content", "")
            if not content or len(content.strip()) < 20:
                continue


            chunks = self.chunker.chunk_conversation_turn(content)

            for chunk_info in chunks:
                chunk_text = chunk_info["text"]
                modality = chunk_info["modality"]


                raw_entries = await self._extractor.extract(
                    chunk_text, modality
                )


                for raw in raw_entries:
                    entry = await self._build_entry(
                        raw, modality, conversation_id, turn_number
                    )
                    if entry:
                        all_entries.append(entry)

        return all_entries

    async def _build_entry(
        self,
        raw: Dict,
        modality: ContentModality,
        conversation_id: str,
        turn_number: int,
    ) -> Optional[RolodexEntry]:

        content = raw.get("content", "").strip()
        if not content or len(content) < 10:
            return None


        cat_str = raw.get("category", "note").lower().strip()
        try:
            category = EntryCategory(cat_str)
        except ValueError:
            category = EntryCategory.NOTE


        tags = raw.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]
        tags = [t for t in tags if t]


        embedding = await self.embeddings.embed_text(content)

        return RolodexEntry(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            content=content,
            content_type=modality,
            category=category,
            tags=tags,
            source_range={
                "turn_number": turn_number,
            },
            embedding=embedding,
            linked_ids=[],
            tier=Tier.COLD,
        )
