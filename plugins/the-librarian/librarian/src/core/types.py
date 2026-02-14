"""
The Librarian — Core Data Types
All shared dataclasses and enums used across the system.
This is the foundational contract that all components build on.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import math
import uuid
# ─── Enums ───────────────────────────────────────────────────────────────────
class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
class ContentModality(Enum):
    PROSE = "prose"
    CODE = "code"
    MATH = "math"
    CONVERSATIONAL = "conversational"
    STRUCTURED = "structured"
    MIXED = "mixed"
class EntryCategory(Enum):
    DEFINITION = "definition"
    EXAMPLE = "example"
    IMPLEMENTATION = "implementation"
    INSTRUCTION = "instruction"
    DECISION = "decision"
    PREFERENCE = "preference"
    REFERENCE = "reference"
    FACT = "fact"
    WARNING = "warning"
    NOTE = "note"
    # Experience markers — captures user journey moments
    CORRECTION = "correction"      # Something was wrong and got fixed
    FRICTION = "friction"          # A struggle, confusion, or difficulty
    BREAKTHROUGH = "breakthrough"  # A moment of clarity or success
    PIVOT = "pivot"                # A change of direction or approach
    # Privileged tier — persistent user facts
    USER_KNOWLEDGE = "user_knowledge"
class Tier(Enum):
    HOT = "hot"
    COLD = "cold"
# ─── Core Data Structures ────────────────────────────────────────────────────
@dataclass
class Message:
    """A single message in the conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    token_count: int = 0
    turn_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = estimate_tokens(self.content)
@dataclass
class RolodexEntry:
    """
    A single discrete item in the rolodex.
    Stored verbatim — never compressed or summarized.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    content: str = ""
    content_type: ContentModality = ContentModality.CONVERSATIONAL
    category: EntryCategory = EntryCategory.NOTE
    tags: List[str] = field(default_factory=list)
    source_range: Dict[str, int] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    tier: Tier = Tier.COLD
    embedding: Optional[List[float]] = None
    linked_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    verbatim_source: bool = True  # True = original user/assistant text; False = summary/paraphrase
@dataclass
class ConversationState:
    """Tracks the full state of an active conversation."""
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
    total_tokens: int = 0
    turn_count: int = 0
    librarian_active: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    def add_message(self, role: MessageRole, content: str) -> Message:
        """Add a message and update state."""
        self.turn_count += 1
        msg = Message(
            role=role,
            content=content,
            turn_number=self.turn_count,
        )
        self.messages.append(msg)
        self.total_tokens += msg.token_count
        return msg
    def should_activate_librarian(self, activation_threshold: int = 5000) -> bool:
        """Check if we've hit the token threshold for Librarian activation."""
        if self.librarian_active:
            return True
        if self.total_tokens >= activation_threshold:
            self.librarian_active = True
            return True
        return False
@dataclass
class LibrarianQuery:
    """A query from the working agent to the Librarian."""
    query_text: str
    search_type: str = "hybrid"      # "keyword", "semantic", "hybrid"
    limit: int = 5
    min_similarity: float = 0.5
    conversation_id: Optional[str] = None
@dataclass
class LibrarianResponse:
    """Response from the Librarian to the working agent."""
    found: bool
    entries: List[RolodexEntry] = field(default_factory=list)
    chains: List[Any] = field(default_factory=list)  # Phase 7: ReasoningChain objects
    search_time_ms: float = 0.0
    cache_hit: bool = False
    query: Optional[LibrarianQuery] = None
    metadata: Dict[str, Any] = field(default_factory=dict)  # Phase 4: cross-session info etc.
# ─── Tier Events ─────────────────────────────────────────────────────────────

@dataclass
class TierEvent:
    """Records a promotion or demotion event."""
    entry_id: str
    old_tier: Tier
    new_tier: Tier
    score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ─── Preloading (Phase 3) ───────────────────────────────────────────────────

@dataclass
class PreloadPrediction:
    """A single prediction: this entry will likely be needed soon."""
    entry_id: str
    confidence: float                # 0-1, derived from similarity or LLM
    strategy: str = "embedding"      # "embedding" or "llm"
    query_text: str = ""             # The predicted query that found this entry

@dataclass
class SessionInfo:
    """Metadata about a past or current session, for listing/display."""
    session_id: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_active: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    message_count: int = 0
    entry_count: int = 0
    summary: str = ""
    status: str = "active"  # "active" or "ended"


@dataclass
class PreloadResult:
    """
    Outcome of a preload cycle.
    Separates high-confidence entries (inject proactively) from
    lower-confidence entries (warm cache only).
    """
    injected_entries: List[RolodexEntry] = field(default_factory=list)
    cache_warmed_entries: List[RolodexEntry] = field(default_factory=list)
    predictions: List[PreloadPrediction] = field(default_factory=list)
    strategy_used: str = "none"          # "embedding", "llm", or "none"
    pressure: float = 0.0
    turn_number: int = 0


# ─── Reasoning Chains (Phase 7) ──────────────────────────────────────────────

@dataclass
class ReasoningChain:
    """
    A narrative snapshot capturing the reasoning thread of a conversation segment.
    Chains preserve the "why" of workflows — not just discrete facts but causality,
    rejected alternatives, and trade-offs.

    Linked-list traversal: prev = chain_index - 1, next = chain_index + 1.
    related_entries bridges chains → discrete rolodex facts.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    chain_index: int = 0                     # Sequential within session
    turn_range_start: int = 0                # First turn covered
    turn_range_end: int = 0                  # Last turn covered
    summary: str = ""                        # 1-3 sentences: the narrative "why"
    topics: List[str] = field(default_factory=list)
    related_entries: List[str] = field(default_factory=list)  # Rolodex entry IDs
    embedding: Optional[List[float]] = None  # Vector for semantic chain search
    created_at: datetime = field(default_factory=datetime.utcnow)

    def prev_chain_index(self) -> int:
        """Index of the previous chain in the session."""
        return max(0, self.chain_index - 1)

    def next_chain_index(self) -> int:
        """Index of the next chain in the session."""
        return self.chain_index + 1


# ─── Utility Functions ────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    Rough token estimation: ~4 characters per token for English.
    Good enough for Phase 1. Refine with proper tokenizer later.
    """
    return max(1, len(text) // 4)


def compute_importance_score(
    entry: RolodexEntry,
    recency_half_life_hours: float = 24.0,
    age_boost_half_life_hours: float = 48.0,
    now: Optional[datetime] = None,
) -> float:
    """
    Compute a dynamic importance score based on natural usage patterns.

    Like ABR streaming — the score adapts to the bandwidth of conversation
    needs without manual tuning. Frequency determines importance; time
    determines relevance.

    Components:
        access_weight = log(1 + access_count)
            Logarithmic to prevent runaway from power-law access patterns.

        recency_factor = exp(-hours_since_last_access / half_life)
            Exponential decay from last access. Entries that haven't been
            touched in a while naturally lose relevance.

        age_boost = exp(-hours_since_creation / half_life)
            Slight boost for newer entries — they haven't had time to
            accumulate accesses yet, so we give them a fair chance.

    Final: score = access_weight * recency_factor * (1 + age_boost)
    """
    if now is None:
        now = datetime.utcnow()

    # Base: how often accessed (log scale)
    access_weight = math.log1p(entry.access_count)

    # Recency: exponential decay from last access
    if entry.last_accessed:
        seconds_since_access = max(0, (now - entry.last_accessed).total_seconds())
        hours_since_access = seconds_since_access / 3600.0
        recency_factor = math.exp(-hours_since_access / recency_half_life_hours)
    else:
        # Never accessed — minimal recency
        recency_factor = 0.1

    # Age boost: newer entries get a small leg up
    seconds_since_creation = max(0, (now - entry.created_at).total_seconds())
    hours_since_creation = seconds_since_creation / 3600.0
    age_boost = math.exp(-hours_since_creation / age_boost_half_life_hours)

    return access_weight * recency_factor * (1 + age_boost)


# ─── Boot Manifest (Phase 10) ────────────────────────────────────────────────

@dataclass
class ManifestEntry:
    """A single ranked entry selected for the boot manifest."""
    entry_id: str
    composite_score: float
    token_cost: int
    topic_label: Optional[str] = None
    selection_reason: str = "census_rank"  # census_rank | topic_rep | chain_fill | delta_promotion | behavioral
    was_accessed: bool = False
    slot_rank: int = 0


@dataclass
class ManifestState:
    """The full boot manifest: a pre-computed, ranked context plan."""
    manifest_id: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    source_session_id: Optional[str] = None
    manifest_type: str = "super"  # super | incremental | refined
    total_token_cost: int = 0
    entries: List[ManifestEntry] = field(default_factory=list)
    topic_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
