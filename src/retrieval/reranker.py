"""
The Librarian — Re-ranker

Takes a wide pool of candidate entries and re-ranks them using multiple
signals beyond embedding similarity. This is the "narrow" phase of the
wide-net-then-narrow search pattern.

Scoring signals:
1. Semantic similarity (from the original search score)
2. Entity match (exact string match of extracted entities in entry content)
3. Category match (from intent detection — biases toward relevant categories)
4. Recency (newer entries get a mild boost)
5. Access frequency (well-worn book principle — frequently accessed entries rank higher)

No LLM calls — purely heuristic, runs in microseconds.
"""
import time
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field
from ..core.types import RolodexEntry
from .entity_extractor import EntityExtractor, ExtractedEntities


@dataclass
class RerankerConfig:
    """Weights for each scoring signal. All should sum to ~1.0 for interpretability."""
    semantic_weight: float = 0.35
    entity_weight: float = 0.30
    category_weight: float = 0.15
    recency_weight: float = 0.10
    frequency_weight: float = 0.10

    # Verbatim boost: multiplier applied to entries with verbatim_source=True
    # Ensures original user/assistant text always outranks assistant summaries
    verbatim_boost: float = 1.5

    # Recency decay: entries older than this many days get no recency boost
    recency_horizon_days: float = 30.0

    # Frequency cap: beyond this many accesses, no additional boost
    frequency_cap: int = 50


@dataclass
class ScoredCandidate:
    """An entry with its composite re-ranking score and signal breakdown."""
    entry: RolodexEntry
    composite_score: float = 0.0
    semantic_score: float = 0.0
    entity_score: float = 0.0
    category_score: float = 0.0
    recency_score: float = 0.0
    frequency_score: float = 0.0


class Reranker:
    """
    Multi-signal re-ranker for search candidates.
    Takes a wide pool and narrows to the most relevant entries.
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()
        self.entity_extractor = EntityExtractor()

    def rerank(
        self,
        candidates: List[Tuple[RolodexEntry, float]],
        query: str,
        query_entities: Optional[ExtractedEntities] = None,
        category_bias: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[ScoredCandidate]:
        """
        Re-rank a pool of candidates using multiple signals.

        Args:
            candidates: List of (entry, search_score) from initial wide search
            query: The original query string
            query_entities: Pre-extracted entities from the query (or will extract)
            category_bias: Categories to boost (from intent detection)
            limit: Number of results to return

        Returns:
            Top N entries re-ranked by composite score
        """
        if not candidates:
            return []

        # Extract entities from query if not provided
        if query_entities is None:
            query_entities = self.entity_extractor.extract_from_query(query)

        # Normalize search scores to 0-1 range
        max_search_score = max(score for _, score in candidates) if candidates else 1.0
        if max_search_score == 0:
            max_search_score = 1.0

        # Find the most recent timestamp for recency normalization
        now = time.time()

        # Score each candidate
        scored: List[ScoredCandidate] = []
        for entry, search_score in candidates:
            sc = ScoredCandidate(entry=entry)

            # Signal 1: Semantic similarity (normalized from original search)
            sc.semantic_score = search_score / max_search_score

            # Signal 2: Entity match
            sc.entity_score = self._score_entity_match(entry, query_entities)

            # Signal 3: Category match
            sc.category_score = self._score_category_match(entry, category_bias)

            # Signal 4: Recency
            sc.recency_score = self._score_recency(entry, now)

            # Signal 5: Access frequency
            sc.frequency_score = self._score_frequency(entry)

            # Composite score
            sc.composite_score = (
                self.config.semantic_weight * sc.semantic_score
                + self.config.entity_weight * sc.entity_score
                + self.config.category_weight * sc.category_score
                + self.config.recency_weight * sc.recency_score
                + self.config.frequency_weight * sc.frequency_score
            )

            # Verbatim boost: original text always outranks summaries
            if getattr(entry, "verbatim_source", True):
                sc.composite_score *= self.config.verbatim_boost

            scored.append(sc)

        # Sort by composite score descending
        scored.sort(key=lambda s: s.composite_score, reverse=True)

        return scored[:limit]

    def _score_entity_match(
        self, entry: RolodexEntry, query_entities: ExtractedEntities
    ) -> float:
        """
        Score based on exact entity matches in entry content.
        Returns 0-1 where 1 means all query entities were found.
        """
        if not query_entities.all_entities:
            return 0.0

        content_lower = entry.content.lower()
        tags_lower = " ".join(entry.tags).lower() if entry.tags else ""
        search_text = content_lower + " " + tags_lower

        matches = 0
        for entity in query_entities.all_entities:
            if entity.lower() in search_text:
                matches += 1

        # Also check attribution match
        if query_entities.attribution:
            # If query asks about "what I said" (user attribution)
            # boost entries that have user attribution markers
            if query_entities.attribution == "user":
                user_markers = ["user", "philip", "i said", "i asked", "my "]
                if any(m in content_lower for m in user_markers):
                    matches += 0.5
            elif query_entities.attribution == "assistant":
                assistant_markers = ["assistant", "claude", "you said", "suggested"]
                if any(m in content_lower for m in assistant_markers):
                    matches += 0.5

        total_entities = len(query_entities.all_entities) + (
            0.5 if query_entities.attribution else 0
        )

        return min(matches / total_entities, 1.0) if total_entities > 0 else 0.0

    def _score_category_match(
        self, entry: RolodexEntry, category_bias: Optional[List[str]]
    ) -> float:
        """Score based on whether entry category matches the intent bias."""
        if not category_bias:
            return 0.5  # Neutral — no bias means no penalty or boost

        cat = entry.category
        cat_str = cat.value if hasattr(cat, "value") else str(cat)

        return 1.0 if cat_str in category_bias else 0.0

    def _score_recency(self, entry: RolodexEntry, now: float) -> float:
        """
        Score based on how recently the entry was created.
        Linear decay over the recency horizon.
        """
        if not hasattr(entry, "created_at") or entry.created_at is None:
            return 0.5  # Unknown age — neutral

        try:
            # created_at might be a datetime or a timestamp
            if hasattr(entry.created_at, "timestamp"):
                entry_time = entry.created_at.timestamp()
            else:
                entry_time = float(entry.created_at)

            age_seconds = now - entry_time
            horizon_seconds = self.config.recency_horizon_days * 86400

            if age_seconds <= 0:
                return 1.0
            elif age_seconds >= horizon_seconds:
                return 0.0
            else:
                return 1.0 - (age_seconds / horizon_seconds)
        except (TypeError, ValueError):
            return 0.5

    def _score_frequency(self, entry: RolodexEntry) -> float:
        """
        Score based on access frequency (the well-worn book principle).
        More accesses = higher score, capped at frequency_cap.
        """
        access_count = getattr(entry, "access_count", 0) or 0

        if access_count <= 0:
            return 0.0

        return min(access_count / self.config.frequency_cap, 1.0)
