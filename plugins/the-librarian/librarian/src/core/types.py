


from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import math
import uuid

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
class Tier(Enum):
    HOT = "hot"
    COLD = "cold"

@dataclass
class Message:

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
@dataclass
class ConversationState:

    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
    total_tokens: int = 0
    turn_count: int = 0
    librarian_active: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    def add_message(self, role: MessageRole, content: str) -> Message:

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

        if self.librarian_active:
            return True
        if self.total_tokens >= activation_threshold:
            self.librarian_active = True
            return True
        return False
@dataclass
class LibrarianQuery:

    query_text: str
    search_type: str = "hybrid"
    limit: int = 5
    min_similarity: float = 0.5
    conversation_id: Optional[str] = None
@dataclass
class LibrarianResponse:

    found: bool
    entries: List[RolodexEntry] = field(default_factory=list)
    chains: List[Any] = field(default_factory=list)
    search_time_ms: float = 0.0
    cache_hit: bool = False
    query: Optional[LibrarianQuery] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TierEvent:

    entry_id: str
    old_tier: Tier
    new_tier: Tier
    score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PreloadPrediction:

    entry_id: str
    confidence: float
    strategy: str = "embedding"
    query_text: str = ""

@dataclass
class SessionInfo:

    session_id: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_active: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    message_count: int = 0
    entry_count: int = 0
    summary: str = ""
    status: str = "active"


@dataclass
class PreloadResult:


    injected_entries: List[RolodexEntry] = field(default_factory=list)
    cache_warmed_entries: List[RolodexEntry] = field(default_factory=list)
    predictions: List[PreloadPrediction] = field(default_factory=list)
    strategy_used: str = "none"
    pressure: float = 0.0
    turn_number: int = 0


@dataclass
class ReasoningChain:


    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    chain_index: int = 0
    turn_range_start: int = 0
    turn_range_end: int = 0
    summary: str = ""
    topics: List[str] = field(default_factory=list)
    related_entries: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def prev_chain_index(self) -> int:

        return max(0, self.chain_index - 1)

    def next_chain_index(self) -> int:

        return self.chain_index + 1


def estimate_tokens(text: str) -> int:


    return max(1, len(text) // 4)


def compute_importance_score(
    entry: RolodexEntry,
    recency_half_life_hours: float = 24.0,
    age_boost_half_life_hours: float = 48.0,
    now: Optional[datetime] = None,
) -> float:


    if now is None:
        now = datetime.utcnow()


    access_weight = math.log1p(entry.access_count)


    if entry.last_accessed:
        seconds_since_access = max(0, (now - entry.last_accessed).total_seconds())
        hours_since_access = seconds_since_access / 3600.0
        recency_factor = math.exp(-hours_since_access / recency_half_life_hours)
    else:

        recency_factor = 0.1


    seconds_since_creation = max(0, (now - entry.created_at).total_seconds())
    hours_since_creation = seconds_since_creation / 3600.0
    age_boost = math.exp(-hours_since_creation / age_boost_half_life_hours)

    return access_weight * recency_factor * (1 + age_boost)
