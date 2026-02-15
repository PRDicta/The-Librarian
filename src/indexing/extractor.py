"""
The Librarian — Entry Extractor

Extracts discrete, reusable information items from conversation chunks.
Each extracted item becomes a rolodex entry.

Adapter-based: uses an LLMAdapter for smart extraction when available,
falls back to VerbatimExtractor (heuristic, no API calls) otherwise.
"""
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
    """
    Extracts discrete information items from conversation content.
    Uses LLMAdapter for intelligent extraction when provided,
    VerbatimExtractor as the free, offline fallback.
    """

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        chunker: Optional[ContentChunker] = None,
        llm_adapter: Optional[LLMAdapter] = None,
    ):
        self.embeddings = embedding_manager or EmbeddingManager()
        self.chunker = chunker or ContentChunker()
        # Use LLM adapter if provided, otherwise verbatim extraction
        self._extractor = llm_adapter if llm_adapter else VerbatimExtractor()

    async def extract_from_messages(
        self,
        messages: List[Dict[str, str]],
        conversation_id: str,
        turn_number: int = 0,
    ) -> List[RolodexEntry]:
        """
        Extract rolodex entries from a batch of conversation messages.

        Args:
            messages: List of {"role": str, "content": str} dicts
            conversation_id: ID of the current conversation
            turn_number: Current turn number for source_range

        Returns:
            List of newly created RolodexEntry objects (with embeddings)
        """
        all_entries = []

        for msg in messages:
            content = msg.get("content", "")
            if not content or len(content.strip()) < 20:
                continue

            # Chunk the message content
            chunks = self.chunker.chunk_conversation_turn(content)

            for chunk_info in chunks:
                chunk_text = chunk_info["text"]
                modality = chunk_info["modality"]

                # Extract entries via adapter or verbatim fallback
                raw_entries = await self._extractor.extract(
                    chunk_text, modality
                )

                # Convert to RolodexEntry objects with embeddings
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
        """Convert a raw extraction dict into a full RolodexEntry."""
        content = raw.get("content", "").strip()
        if not content or len(content) < 10:
            return None

        # Parse category
        cat_str = raw.get("category", "note").lower().strip()
        try:
            category = EntryCategory(cat_str)
        except ValueError:
            category = EntryCategory.NOTE

        # Parse tags
        tags = raw.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]
        tags = [t for t in tags if t]  # Remove empties

        # Generate embedding
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
