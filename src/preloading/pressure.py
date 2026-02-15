"""
The Librarian — Session Pressure Monitor (Phase 3)

Tracks conversation memory strain using a rolling window of events.
Like ABR streaming: adapts preloading aggressiveness to match demand.

Low pressure  → conservative (embedding proximity only)
High pressure → aggressive (LLM trajectory prediction)
"""
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime


@dataclass
class _GapEvent:
    turn: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class _QueryEvent:
    turn: int
    cache_hit: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class _TokenEvent:
    turn: int
    count: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PressureMonitor:
    """
    Monitors session memory pressure using a sliding window.

    Pressure is a 0-1 value combining:
    - gap_rate:       How often the working agent hits context gaps
    - cache_miss_rate: How often searches miss the hot cache
    - token_velocity:  How fast the context window is filling up

    These are weighted and combined into a single pressure score
    that drives preloading strategy selection.
    """

    # Component weights (must sum to 1.0)
    GAP_WEIGHT = 0.5
    CACHE_MISS_WEIGHT = 0.3
    TOKEN_VELOCITY_WEIGHT = 0.2

    def __init__(self, window_size: int = 10, context_max: int = 180_000):
        self.window_size = window_size
        self.context_max = context_max
        self._gap_events: List[_GapEvent] = []
        self._query_events: List[_QueryEvent] = []
        self._token_events: List[_TokenEvent] = []
        self._current_turn: int = 0

    def record_gap(self, turn: int) -> None:
        """Record that a context gap was detected at this turn."""
        self._gap_events.append(_GapEvent(turn=turn))
        self._current_turn = max(self._current_turn, turn)

    def record_query(self, turn: int, cache_hit: bool) -> None:
        """Record a search event with cache hit/miss."""
        self._query_events.append(_QueryEvent(turn=turn, cache_hit=cache_hit))
        self._current_turn = max(self._current_turn, turn)

    def record_tokens(self, turn: int, count: int) -> None:
        """Record token accumulation at a turn."""
        self._token_events.append(_TokenEvent(turn=turn, count=count))
        self._current_turn = max(self._current_turn, turn)

    def get_pressure(self) -> float:
        """
        Compute current session pressure (0-1).

        Returns 0.0 if insufficient data (< 3 turns tracked).
        """
        if self._current_turn < 3:
            return 0.0

        window_start = max(0, self._current_turn - self.window_size)

        # Gap rate: gaps in window / window size
        gaps_in_window = sum(
            1 for e in self._gap_events if e.turn >= window_start
        )
        gap_rate = min(1.0, gaps_in_window / max(1, self.window_size))

        # Cache miss rate: misses / total queries in window
        queries_in_window = [
            e for e in self._query_events if e.turn >= window_start
        ]
        if queries_in_window:
            misses = sum(1 for q in queries_in_window if not q.cache_hit)
            cache_miss_rate = misses / len(queries_in_window)
        else:
            cache_miss_rate = 0.0

        # Token velocity: latest token count / context max
        recent_tokens = [
            e for e in self._token_events if e.turn >= window_start
        ]
        if recent_tokens:
            latest = max(recent_tokens, key=lambda e: e.turn)
            token_velocity = min(1.0, latest.count / max(1, self.context_max))
        else:
            token_velocity = 0.0

        # Weighted combination
        pressure = (
            gap_rate * self.GAP_WEIGHT
            + cache_miss_rate * self.CACHE_MISS_WEIGHT
            + token_velocity * self.TOKEN_VELOCITY_WEIGHT
        )
        return min(1.0, max(0.0, pressure))

    def get_strategy(
        self,
        low_threshold: float = 0.3,
        high_threshold: float = 0.7,
    ) -> str:
        """
        Determine preloading strategy based on pressure.

        Returns: "none", "embedding", or "llm"
        """
        pressure = self.get_pressure()
        if pressure >= high_threshold:
            return "llm"
        elif pressure >= low_threshold:
            return "embedding"
        else:
            # Low pressure — still do embedding if we have data
            if self._current_turn >= 3:
                return "embedding"
            return "none"

    def get_max_entries(
        self,
        low_threshold: float = 0.3,
        high_threshold: float = 0.7,
        default_max: int = 5,
    ) -> int:
        """How many entries to preload based on pressure level."""
        pressure = self.get_pressure()
        if pressure >= high_threshold:
            return max(default_max, 8)
        elif pressure >= low_threshold:
            return default_max
        else:
            return min(default_max, 3)

    # ─── Negotiation Feedback (Phase 6c) ────────────────────────────────

    def record_negotiation(
        self, resolved: bool, budget_used: int, rounds: int
    ):
        """
        Record the outcome of a context negotiation.
        Feeds back into pressure calculations for future decisions.
        """
        # A resolved negotiation reduces pressure (context worked)
        # An unresolved one increases pressure (need better retrieval)
        if not resolved:
            # Record as a pseudo-gap (couldn't find useful context)
            self.record_gap(self._current_turn)

    # ─── Deep Index Trigger (Phase 7d) ──────────────────────────────────

    def should_trigger_deep_index(self, threshold: float = 0.8) -> bool:
        """
        Check if token pressure exceeds deep-indexing threshold.
        When true, trigger emergency chain snapshot before compaction.
        """
        return self.get_token_fill_ratio() >= threshold

    def get_token_fill_ratio(self) -> float:
        """Get current token usage as fraction of context_max."""
        recent_tokens = [
            e for e in self._token_events
            if e.turn >= max(0, self._current_turn - 5)
        ]
        if recent_tokens:
            latest = max(recent_tokens, key=lambda e: e.turn)
            return latest.count / max(1, self.context_max)
        return 0.0

    def get_summary(self) -> Dict:
        """Return a summary dict for debug/stats."""
        return {
            "pressure": round(self.get_pressure(), 3),
            "strategy": self.get_strategy(),
            "current_turn": self._current_turn,
            "total_gaps": len(self._gap_events),
            "total_queries": len(self._query_events),
            "cache_hit_rate": (
                sum(1 for q in self._query_events if q.cache_hit)
                / max(1, len(self._query_events))
            ),
        }
