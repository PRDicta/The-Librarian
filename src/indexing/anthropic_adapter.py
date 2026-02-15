"""
The Librarian — Anthropic LLM Adapter

Concrete implementation of LLMAdapter using the Anthropic API.
Wraps Claude Haiku for:
- Smart extraction (structured fact decomposition from conversation chunks)
- Trajectory prediction (anticipating next conversation topics)

This is the "enhanced" path. Without this adapter, The Librarian
falls back to verbatim extraction and embedding-only prediction.
"""
import json
from typing import Dict, List

import anthropic

from ..core.types import ContentModality, Message
from ..core.llm_adapter import LLMAdapter


class AnthropicAdapter:
    """
    LLMAdapter implementation backed by Anthropic's Claude API.
    Uses Haiku for both extraction and prediction (cheap + fast).
    """

    def __init__(
        self,
        api_key: str,
        extraction_model: str = "claude-haiku-4-5-20251001",
        prediction_model: str = "claude-haiku-4-5-20251001",
        cost_tracker=None,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.extraction_model = extraction_model
        self.prediction_model = prediction_model
        self.cost_tracker = cost_tracker

    # ─── Extraction ────────────────────────────────────────────────────────

    async def extract(
        self,
        chunk: str,
        modality: ContentModality,
    ) -> List[Dict]:
        """
        Call Haiku to extract discrete information items from a chunk.
        Returns list of dicts with content, category, tags, linked_to.
        Falls back to a single verbatim entry on failure.
        """
        prompt = self._build_extraction_prompt(chunk, modality)
        try:
            response = self.client.messages.create(
                model=self.extraction_model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
                system=(
                    "You are an extraction specialist for a memory system. "
                    "Extract discrete, reusable information items from content. "
                    "Return valid JSON only. No markdown, no explanation."
                ),
            )
            # Track extraction cost
            if self.cost_tracker and hasattr(response, "usage"):
                self.cost_tracker.record(
                    call_type="extraction",
                    model=self.extraction_model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
            text = response.content[0].text.strip()
            # Handle ```json wrapping
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            items = json.loads(text)
            if isinstance(items, dict):
                items = [items]
            return items
        except (json.JSONDecodeError, Exception):
            # Graceful degradation: return the whole chunk as one entry
            return [{
                "content": chunk,
                "category": "note",
                "tags": [modality.value],
                "linked_to": [],
            }]

    def _build_extraction_prompt(
        self, chunk: str, modality: ContentModality
    ) -> str:
        """Build a modality-specific extraction prompt."""
        base = f"""Analyze the following {modality.value} content and extract discrete information items.

RULES:
1. One concept per item — each should be understandable on its own
2. Preserve exact details (names, numbers, specifics)
3. Categorize each item
4. Add relevant tags for searchability

CATEGORIES (pick one per item):
- definition: What something is
- example: How to use something
- implementation: How something works (code, algorithms)
- instruction: What the user wants done
- decision: A choice that was made
- preference: User preferences or requirements
- reference: External references, links, citations
- fact: Factual information
- warning: Caveats, pitfalls, limitations
- note: General observations
- correction: Something was wrong and got fixed ("actually...", "the bug was...", "turns out...")
- friction: A struggle, confusion, or difficulty ("couldn't figure out", "kept failing", "wrong command")
- breakthrough: A moment of clarity or success ("figured it out", "the key was", "finally works")
- pivot: A change of direction or approach ("switched to", "abandoned", "different approach")

CONTENT:
{chunk}

Return a JSON array of items:
[
  {{
    "content": "Brief, self-contained description preserving key details",
    "category": "one of the categories above",
    "tags": ["relevant", "search", "tags"],
    "linked_to": ["names of related concepts if any"]
  }}
]

Extract only meaningful items. Skip trivial filler. Return valid JSON only."""
        # Modality-specific guidance
        if modality == ContentModality.CODE:
            base += "\n\nFor code: extract function purposes, input/output specs, algorithms used, and implementation notes separately."
        elif modality == ContentModality.MATH:
            base += "\n\nFor math: extract theorem statements, proof strategies, key equations, and relationships between concepts separately."
        elif modality == ContentModality.PROSE:
            base += "\n\nFor prose: extract key claims, supporting evidence, conclusions, and named entities separately."
        return base

    # ─── Prediction ────────────────────────────────────────────────────────

    PREDICTION_PROMPT = """You are The Librarian, an AI memory system.
Given the recent conversation below, predict the 3 most likely topics
the user will ask about or reference next. Return ONLY a JSON array
of objects with "topic" and "confidence" (0-1) fields.

Example response:
[{{"topic": "database indexing strategies", "confidence": 0.85}},
 {{"topic": "Python async patterns", "confidence": 0.7}},
 {{"topic": "error handling approach", "confidence": 0.6}}]

Recent conversation:
{conversation}

Predict the next 3 topics (JSON only, no explanation):"""

    async def predict_topics(
        self,
        messages: List[Message],
    ) -> List[Dict]:
        """
        Ask Haiku to predict what topics the conversation will need next.
        Returns list of dicts with topic and confidence.
        """
        if not messages:
            return []

        # Build conversation summary
        conv_lines = []
        for msg in messages[-6:]:
            role = msg.role.value.upper()
            content = msg.content[:200]
            conv_lines.append(f"{role}: {content}")
        conversation = "\n".join(conv_lines)

        try:
            response = self.client.messages.create(
                model=self.prediction_model,
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": self.PREDICTION_PROMPT.format(
                        conversation=conversation
                    ),
                }],
            )
            # Track prediction cost
            if self.cost_tracker and hasattr(response, "usage"):
                self.cost_tracker.record(
                    call_type="prediction",
                    model=self.prediction_model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
            response_text = response.content[0].text.strip()
            predicted = json.loads(response_text)
            if not isinstance(predicted, list):
                return []
            return predicted[:3]
        except (json.JSONDecodeError, Exception):
            return []
