


import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
@dataclass
class LibrarianConfig:


    anthropic_api_key: str = ""
    working_agent_model: str = "claude-sonnet-4-5-20250929"
    librarian_model: str = "claude-haiku-4-5-20251001"

    db_path: str = "rolodex.db"

    embedding_strategy: str = "local"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    voyage_api_key: str = ""

    cost_tracking_enabled: bool = True

    negotiation_enabled: bool = True
    negotiation_max_rounds: int = 2
    negotiation_model: str = "claude-haiku-4-5-20251001"

    context_window_size: int = 180_000
    max_context_for_history: int = 80_000
    max_context_for_retrieval: int = 20_000

    librarian_activation_tokens: int = 0
    hot_cache_size: int = 50
    extraction_batch_size: int = 3

    keyword_weight: float = 0.4
    semantic_weight: float = 0.6
    search_result_limit: int = 5

    promotion_threshold: float = 1.0
    demotion_threshold: float = 0.3
    score_recency_half_life_hours: float = 24.0
    score_age_boost_half_life_hours: float = 48.0
    tier_sweep_interval: int = 10

    preload_enabled: bool = True
    preload_pressure_window: int = 10
    preload_low_threshold: float = 0.3
    preload_high_threshold: float = 0.7
    preload_max_entries: int = 5
    preload_injection_confidence: float = 0.8

    cross_session_search: bool = True
    session_boost_factor: float = 1.5

    chain_interval: int = 5
    chain_deep_index_threshold: float = 0.8


    ingestion_queue_enabled: bool = False
    ingestion_num_workers: int = 2
    ingestion_pause_on_query: bool = True


    context_window_budget: int = 20_000
    context_min_active_turns: int = 4
    context_bridge_max_tokens: int = 1_000

    debug_mode: bool = False
    @classmethod
    def from_env(cls, env_path: str = ".env") -> "LibrarianConfig":


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

            ingestion_queue_enabled=os.getenv("INGESTION_QUEUE_ENABLED", "false").lower() == "true",
            ingestion_num_workers=int(os.getenv("INGESTION_NUM_WORKERS", str(cls.ingestion_num_workers))),
            ingestion_pause_on_query=os.getenv("INGESTION_PAUSE_ON_QUERY", "true").lower() == "true",

            context_window_budget=int(os.getenv("CONTEXT_WINDOW_BUDGET", str(cls.context_window_budget))),
            context_min_active_turns=int(os.getenv("CONTEXT_MIN_ACTIVE_TURNS", str(cls.context_min_active_turns))),
            context_bridge_max_tokens=int(os.getenv("CONTEXT_BRIDGE_MAX_TOKENS", str(cls.context_bridge_max_tokens))),
        )
    def validate(self) -> List[str]:

        warnings = []
        if not self.anthropic_api_key:
            warnings.append(
                "ANTHROPIC_API_KEY not set — running in verbatim mode "
                "(heuristic extraction, embedding-only preloading)"
            )
        return warnings

    @property
    def has_api_key(self) -> bool:

        return bool(self.anthropic_api_key)
def _load_dotenv(path: Path):

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
