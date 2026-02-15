"""
The Librarian — Configuration
Loads settings from environment variables / .env file.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
@dataclass
class LibrarianConfig:
    """All configuration for The Librarian system."""
    # API
    anthropic_api_key: str = ""
    working_agent_model: str = "claude-sonnet-4-5-20250929"
    librarian_model: str = "claude-haiku-4-5-20251001"
    # Storage
    db_path: str = "rolodex.db"
    # Embeddings
    embedding_strategy: str = "local"              # "local", "anthropic", or "hash"
    embedding_model: str = "all-MiniLM-L6-v2"     # For local sentence-transformers
    embedding_dimensions: int = 384                 # MiniLM-L6-v2 output dim
    voyage_api_key: str = ""                        # Voyage AI key (for "anthropic" strategy)
    # Cost tracking
    cost_tracking_enabled: bool = True
    # Negotiation (Phase 6c)
    negotiation_enabled: bool = True
    negotiation_max_rounds: int = 2
    negotiation_model: str = "claude-haiku-4-5-20251001"
    # Context window
    context_window_size: int = 180_000             # Max tokens for working agent
    max_context_for_history: int = 80_000          # How much of the window to use for history
    max_context_for_retrieval: int = 20_000        # Reserved for injected context
    # Librarian behavior
    librarian_activation_tokens: int = 0            # Activate immediately (was 5000)
    hot_cache_size: int = 50                       # Max entries in hot cache
    extraction_batch_size: int = 3                 # Messages to batch for extraction
    # Search
    keyword_weight: float = 0.4
    semantic_weight: float = 0.6
    search_result_limit: int = 5
    # Tier management (Phase 2)
    promotion_threshold: float = 1.0          # Importance score above this → promote to HOT
    demotion_threshold: float = 0.3           # Importance score below this → demote to COLD
    score_recency_half_life_hours: float = 24.0   # Recency decay half-life
    score_age_boost_half_life_hours: float = 48.0  # Age boost decay half-life
    tier_sweep_interval: int = 10             # Run tier sweep every N turns
    # Proactive preloading (Phase 3)
    preload_enabled: bool = True
    preload_pressure_window: int = 10         # Turns to look back for pressure calc
    preload_low_threshold: float = 0.3        # Pressure below this → embedding only
    preload_high_threshold: float = 0.7       # Pressure above this → escalate to LLM
    preload_max_entries: int = 5              # Max entries to preload per turn
    preload_injection_confidence: float = 0.8 # Above this → inject proactively
    # Cross-session persistence (Phase 4)
    cross_session_search: bool = True          # Search across all past sessions
    session_boost_factor: float = 1.5          # Score multiplier for current-session results
    # Reasoning Chains (Phase 7)
    chain_interval: int = 5                  # Generate breadcrumb every N turns
    chain_deep_index_threshold: float = 0.8  # Token pressure ratio for emergency snapshot

    # Ingestion Queue (Phase 8)
    ingestion_queue_enabled: bool = False     # Enable async background enrichment
    ingestion_num_workers: int = 2            # Background worker count
    ingestion_pause_on_query: bool = True     # Pause enrichment during retrieval

    # Context Window Manager (Phase 9)
    context_window_budget: int = 20_000       # Max tokens of history in active window
    context_min_active_turns: int = 4         # Always keep at least N recent turns
    context_bridge_max_tokens: int = 1_000    # Max tokens for the bridge summary
    # Debug
    debug_mode: bool = False
    @classmethod
    def from_env(cls, env_path: str = ".env") -> "LibrarianConfig":
        """Load configuration from environment variables."""
        # Try loading .env file if it exists
        env_file = Path(env_path)
        if env_file.exists():
            _load_dotenv(env_file)
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            working_agent_model=os.getenv("WORKING_AGENT_MODEL", cls.working_agent_model),
            librarian_model=os.getenv("LIBRARIAN_MODEL", cls.librarian_model),
            db_path=os.getenv("ROLODEX_DB_PATH", cls.db_path),
            embedding_strategy=os.getenv("EMBEDDING_STRATEGY", cls.embedding_strategy),
            voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
            negotiation_enabled=os.getenv("NEGOTIATION_ENABLED", "true").lower() == "true",
            negotiation_max_rounds=int(os.getenv("NEGOTIATION_MAX_ROUNDS", str(cls.negotiation_max_rounds))),
            promotion_threshold=float(os.getenv("PROMOTION_THRESHOLD", str(cls.promotion_threshold))),
            demotion_threshold=float(os.getenv("DEMOTION_THRESHOLD", str(cls.demotion_threshold))),
            score_recency_half_life_hours=float(os.getenv("SCORE_RECENCY_HALF_LIFE_HOURS", str(cls.score_recency_half_life_hours))),
            score_age_boost_half_life_hours=float(os.getenv("SCORE_AGE_BOOST_HALF_LIFE_HOURS", str(cls.score_age_boost_half_life_hours))),
            tier_sweep_interval=int(os.getenv("TIER_SWEEP_INTERVAL", str(cls.tier_sweep_interval))),
            preload_enabled=os.getenv("PRELOAD_ENABLED", "true").lower() == "true",
            preload_pressure_window=int(os.getenv("PRELOAD_PRESSURE_WINDOW", str(cls.preload_pressure_window))),
            preload_low_threshold=float(os.getenv("PRELOAD_LOW_THRESHOLD", str(cls.preload_low_threshold))),
            preload_high_threshold=float(os.getenv("PRELOAD_HIGH_THRESHOLD", str(cls.preload_high_threshold))),
            preload_max_entries=int(os.getenv("PRELOAD_MAX_ENTRIES", str(cls.preload_max_entries))),
            preload_injection_confidence=float(os.getenv("PRELOAD_INJECTION_CONFIDENCE", str(cls.preload_injection_confidence))),
            cross_session_search=os.getenv("CROSS_SESSION_SEARCH", "true").lower() == "true",
            session_boost_factor=float(os.getenv("SESSION_BOOST_FACTOR", str(cls.session_boost_factor))),
            debug_mode=os.getenv("DEBUG_MODE", "false").lower() == "true",
            # Phase 8: Ingestion queue
            ingestion_queue_enabled=os.getenv("INGESTION_QUEUE_ENABLED", "false").lower() == "true",
            ingestion_num_workers=int(os.getenv("INGESTION_NUM_WORKERS", str(cls.ingestion_num_workers))),
            ingestion_pause_on_query=os.getenv("INGESTION_PAUSE_ON_QUERY", "true").lower() == "true",
            # Phase 9: Context window
            context_window_budget=int(os.getenv("CONTEXT_WINDOW_BUDGET", str(cls.context_window_budget))),
            context_min_active_turns=int(os.getenv("CONTEXT_MIN_ACTIVE_TURNS", str(cls.context_min_active_turns))),
            context_bridge_max_tokens=int(os.getenv("CONTEXT_BRIDGE_MAX_TOKENS", str(cls.context_bridge_max_tokens))),
        )
    def validate(self) -> List[str]:
        """Return list of warnings (non-fatal). Empty if fully configured."""
        warnings = []
        if not self.anthropic_api_key:
            warnings.append(
                "ANTHROPIC_API_KEY not set — running in verbatim mode "
                "(heuristic extraction, embedding-only preloading)"
            )
        return warnings

    @property
    def has_api_key(self) -> bool:
        """Whether an Anthropic API key is configured."""
        return bool(self.anthropic_api_key)
def _load_dotenv(path: Path):
    """Minimal .env loader — no external dependency needed."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and not os.environ.get(key):
                    os.environ[key] = value
    except Exception:
        pass
