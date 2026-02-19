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
    INSIGHT = "insight"
    USER_KNOWLEDGE = "user_knowledge"
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
    FACTUAL_KNOWLEDGE = "factual_knowledge"  # General factual information
    PIVOT = "pivot"                # A change of direction or approach
    # Privileged tier — persistent user facts
    USER_KNOWLEDGE = "user_knowledge"
    # Privileged tier — project-scoped knowledge (conditional loading)
    PROJECT_KNOWLEDGE = "project_knowledge"
    # Privileged tier — compressed behavioral instructions
    BEHAVIORAL = "behavioral"
class CompressionStage(Enum):
    COLD = 0      # Full prose — never compressed
    WARM = 1      # Emoji-anchored / abbreviated — pattern recognized
    HOT = 2       # Single-token — high confidence, shared context
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
    # Phase 12: Document source tracking
    source_type: str = "conversation"         # conversation | document | user_knowledge
    document_id: Optional[str] = None         # FK to documents.id (nullable)
    source_location: str = ""                 # Free-text: "§3.2, p12, heading: Authentication"
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

# ─── Claude Tokenizer (real BPE) ─────────────────────────────────────────────
# Lazy-loaded from claude.json (bundled from @anthropic-ai/tokenizer npm package).
# Falls back to heuristic estimation if tiktoken or claude.json is unavailable.

_claude_encoder = None
_claude_encoder_loaded = False


def _get_claude_encoder():
    """Lazy-load the real Claude BPE tokenizer. Returns encoder or None."""
    global _claude_encoder, _claude_encoder_loaded
    if _claude_encoder_loaded:
        return _claude_encoder
    _claude_encoder_loaded = True
    try:
        import tiktoken
        import json
        import base64
        import os

        vocab_path = os.path.join(os.path.dirname(__file__), "claude.json")
        if not os.path.isfile(vocab_path):
            return None

        with open(vocab_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        parts = config["bpe_ranks"].split(" ")
        offset = int(parts[1])
        tokens = parts[2:]
        rank_map = {base64.b64decode(t): offset + i for i, t in enumerate(tokens)}

        _claude_encoder = tiktoken.Encoding(
            name="claude",
            pat_str=config["pat_str"],
            mergeable_ranks=rank_map,
            special_tokens=config["special_tokens"],
        )
        return _claude_encoder
    except Exception:
        return None


def estimate_tokens(text: str) -> int:
    """Count tokens using Claude's real BPE tokenizer when available.

    Priority:
    1. Real Claude tokenizer (from bundled claude.json + tiktoken) — exact counts
    2. Heuristic fallback — BPE-aware estimation, ~6-10% error vs real

    The real tokenizer is lazy-loaded on first call. If tiktoken is not
    installed or claude.json is missing, the heuristic is used silently.
    """
    if not text:
        return 0

    # Try real tokenizer first
    enc = _get_claude_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass  # Fall through to heuristic

    # ── Heuristic fallback ──────────────────────────────────────────────
    import re

    tokens = 0

    chunks = re.findall(
        r'[\U00010000-\U0010ffff][\ufe00-\ufe0f\u200d]*'  # Supplementary emoji + modifiers
        r'|[\u2600-\u27bf\u2b50-\u2bff][\ufe00-\ufe0f\u200d]*'  # BMP emoji + modifiers
        r'|[\u00a7\u00a9\u00ae\u203c\u2049\u2122\u2139\u2194-\u21aa]'  # Special symbols
        r'|[\ufe00-\ufe0f\u200d]+'    # Orphaned variation selectors/ZWJ (0 tokens)
        r'|[a-zA-Z_][a-zA-Z0-9_]*'    # Words/identifiers
        r'|\d+'                         # Number sequences
        r'|[^\s\w]'                     # Punctuation/symbols
        r'|\s+',                        # Whitespace runs
        text
    )

    for chunk in chunks:
        if not chunk:
            continue

        first_char = chunk[0]
        code_point = ord(first_char)

        if code_point in range(0xFE00, 0xFE10) or code_point == 0x200D:
            continue

        if code_point > 0x1F00 or code_point in range(0x2600, 0x27C0) or code_point in range(0x2B50, 0x2C00):
            tokens += 3  # Corrected: most emoji = 2-4 tokens, avg ~3

        elif code_point in (0x00A7, 0x00A9, 0x00AE, 0x2122, 0x2139):
            tokens += 1

        elif first_char.isalpha() or first_char == '_':
            word_len = len(chunk)
            if word_len <= 7:
                tokens += 1
            elif word_len <= 12:
                tokens += 2
            else:
                tokens += max(2, word_len // 5)

        elif first_char.isdigit():
            tokens += max(1, len(chunk) // 3)

        elif first_char.isspace():
            newlines = chunk.count('\n')
            tokens += newlines
            if newlines == 0 and len(chunk) > 1:
                tokens += 1

        # Punctuation/symbols
        else:
            tokens += 1

    return max(1, tokens)


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
