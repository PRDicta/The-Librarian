"""
The Librarian — LLM Adapter Protocol

Defines what The Librarian needs from an LLM, if one is available.
Uses structural typing (Protocol) — any object with these methods works.
No inheritance, no registration, no boilerplate.

Without an adapter: The Librarian runs in verbatim mode (heuristic
extraction, embedding-only preloading). Fully functional, zero API cost.

With an adapter: Smart extraction (structured fact decomposition) and
LLM-powered trajectory prediction kick in as enhancements.
"""
from typing import Dict, List, Optional, runtime_checkable
from typing import Protocol

from .types import ContentModality, Message


@runtime_checkable
class LLMAdapter(Protocol):
    """
    Protocol defining LLM capabilities The Librarian can use.

    Implementors provide two methods:
    - extract(): Break a text chunk into structured knowledge items
    - predict_topics(): Predict what the conversation will need next

    Both are async to support non-blocking API calls.
    """

    async def extract(
        self,
        chunk: str,
        modality: ContentModality,
    ) -> List[Dict]:
        """
        Extract discrete knowledge items from a text chunk.

        Args:
            chunk: Text content to extract from
            modality: Content type (CODE, PROSE, MATH, etc.)

        Returns:
            List of dicts, each with:
            - "content": str — the extracted fact/item
            - "category": str — one of: definition, example, implementation,
              instruction, decision, preference, reference, fact, warning, note
            - "tags": List[str] — search keywords
            - "linked_to": List[str] — related concept names
        """
        ...

    async def predict_topics(
        self,
        messages: List[Message],
    ) -> List[Dict]:
        """
        Predict what topics the conversation will reference next.

        Args:
            messages: Recent conversation messages (typically last 6)

        Returns:
            List of dicts, each with:
            - "topic": str — predicted topic
            - "confidence": float — 0.0 to 1.0
        """
        ...
