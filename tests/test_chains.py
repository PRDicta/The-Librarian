"""
The Librarian — Reasoning Chains Tests (Phase 7)

Tests chain CRUD, search, breadcrumb generation, context formatting,
pressure-triggered deep indexing, and chain-first search ordering.
~12 tests validating the full reasoning chain pipeline.
"""
import asyncio
import os
import sys
import tempfile
import uuid
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.types import (
    RolodexEntry, ReasoningChain, ContentModality, EntryCategory, Tier,
    Message, MessageRole, LibrarianQuery, LibrarianResponse,
)
from src.storage.rolodex import Rolodex
from src.storage.schema import serialize_embedding, deserialize_embedding
from src.retrieval.context_builder import ContextBuilder
from src.core.chain_builder import ChainBuilder
from src.preloading.pressure import PressureMonitor


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_chain(
    session_id: str = "test-session",
    chain_index: int = 0,
    turn_start: int = 1,
    turn_end: int = 5,
    summary: str = "Discussed approach to module X",
    topics: list = None,
    related_entries: list = None,
    embedding: list = None,
) -> ReasoningChain:
    """Helper to create test chains."""
    return ReasoningChain(
        id=str(uuid.uuid4()),
        session_id=session_id,
        chain_index=chain_index,
        turn_range_start=turn_start,
        turn_range_end=turn_end,
        summary=summary,
        topics=topics or ["module-x", "architecture"],
        related_entries=related_entries or [],
        embedding=embedding,
        created_at=datetime.utcnow(),
    )


def make_entry(
    content: str,
    tags: list = None,
    category: EntryCategory = EntryCategory.NOTE,
    embedding: list = None,
    conversation_id: str = "test-conv",
) -> RolodexEntry:
    """Helper to create test entries."""
    return RolodexEntry(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        content=content,
        content_type=ContentModality.PROSE,
        category=category,
        tags=tags or [],
        source_range={"turn_start": 1, "turn_end": 1},
        embedding=embedding,
    )


def make_message(role: str, content: str, turn_number: int) -> Message:
    """Helper to create test messages."""
    return Message(
        role=MessageRole.USER if role == "user" else MessageRole.ASSISTANT,
        content=content,
        turn_number=turn_number,
    )


def _tmp_rolodex():
    """Create a Rolodex with a temp DB file. Returns (rolodex, db_path)."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = f.name
    f.close()
    return Rolodex(db_path), db_path


def _safe_cleanup(rolodex, db_path):
    """Close DB and remove file (Windows-safe)."""
    try:
        rolodex.close()
    except Exception:
        pass
    try:
        os.unlink(db_path)
    except OSError:
        pass


# ─── Chain CRUD Tests ─────────────────────────────────────────────────────────

def test_chain_crud():
    """Create, retrieve, and list chains for a session."""
    rolodex, db_path = _tmp_rolodex()
    try:
        chain = make_chain(summary="Decided to merge modules X and Y for simpler routing")
        chain_id = rolodex.create_chain(chain)

        # Retrieve by ID
        retrieved = rolodex.get_chain(chain_id)
        assert retrieved is not None
        assert retrieved.id == chain.id
        assert retrieved.summary == chain.summary
        assert retrieved.topics == chain.topics
        assert retrieved.session_id == chain.session_id
        assert retrieved.chain_index == chain.chain_index

        # List for session
        chains = rolodex.get_chains_for_session("test-session")
        assert len(chains) == 1
        assert chains[0].id == chain.id

        # Non-existent
        missing = rolodex.get_chain("non-existent-id")
        assert missing is None

        print("  PASS: chain_crud")
    finally:
        _safe_cleanup(rolodex, db_path)


def test_chain_serialization():
    """Serialize/deserialize round-trip preserves all fields including embedding."""
    rolodex, db_path = _tmp_rolodex()
    try:
        embedding = [0.1, -0.5, 0.8, 0.0, -1.0, 0.33]
        chain = make_chain(
            summary="Consolidated API endpoints. Trade-off: simpler routing, lost independent X.",
            topics=["api", "routing", "consolidation"],
            related_entries=["entry-1", "entry-2", "entry-3"],
            embedding=embedding,
        )
        rolodex.create_chain(chain)

        retrieved = rolodex.get_chain(chain.id)
        assert retrieved is not None
        assert retrieved.summary == chain.summary
        assert retrieved.topics == chain.topics
        assert retrieved.related_entries == chain.related_entries
        assert retrieved.turn_range_start == chain.turn_range_start
        assert retrieved.turn_range_end == chain.turn_range_end
        assert retrieved.chain_index == chain.chain_index

        # Embedding round-trip
        assert retrieved.embedding is not None
        for a, b in zip(embedding, retrieved.embedding):
            assert abs(a - b) < 1e-6, f"Embedding mismatch: {a} vs {b}"

        print("  PASS: chain_serialization")
    finally:
        _safe_cleanup(rolodex, db_path)


def test_chain_linked_list():
    """Traverse chains via chain_index (linked-list pattern)."""
    rolodex, db_path = _tmp_rolodex()
    try:
        session_id = "linked-list-session"
        chains = []
        for i in range(4):
            chain = make_chain(
                session_id=session_id,
                chain_index=i,
                turn_start=i * 5 + 1,
                turn_end=(i + 1) * 5,
                summary=f"Segment {i}: reasoning about phase {i}",
            )
            rolodex.create_chain(chain)
            chains.append(chain)

        # Traverse forward from first chain
        current = rolodex.get_chain_by_index(session_id, 0)
        assert current is not None
        assert current.chain_index == 0

        next_chain = rolodex.get_chain_by_index(session_id, current.next_chain_index())
        assert next_chain is not None
        assert next_chain.chain_index == 1

        # Traverse to end
        last = rolodex.get_chain_by_index(session_id, 3)
        assert last is not None
        assert last.chain_index == 3

        # prev of first = 0 (clamped)
        first = rolodex.get_chain_by_index(session_id, 0)
        assert first.prev_chain_index() == 0

        # Next of last goes past end
        beyond = rolodex.get_chain_by_index(session_id, last.next_chain_index())
        assert beyond is None  # No chain_index=4 exists

        # List all in order
        all_chains = rolodex.get_chains_for_session(session_id)
        assert len(all_chains) == 4
        for i, c in enumerate(all_chains):
            assert c.chain_index == i

        print("  PASS: chain_linked_list")
    finally:
        _safe_cleanup(rolodex, db_path)


# ─── Chain Search Tests ───────────────────────────────────────────────────────

def test_chain_keyword_search():
    """FTS5 finds chains by summary content."""
    rolodex, db_path = _tmp_rolodex()
    try:
        c1 = make_chain(
            summary="Decided to use PostgreSQL for the database layer instead of MongoDB",
            topics=["postgresql", "database"],
        )
        c2 = make_chain(
            chain_index=1,
            summary="Implemented authentication flow using JWT tokens with refresh rotation",
            topics=["authentication", "jwt"],
        )
        c3 = make_chain(
            chain_index=2,
            summary="Refactored the routing module to consolidate duplicate endpoints",
            topics=["routing", "refactoring"],
        )
        rolodex.create_chain(c1)
        rolodex.create_chain(c2)
        rolodex.create_chain(c3)

        # Search for database-related chains
        results = rolodex.keyword_search_chains("PostgreSQL database")
        assert len(results) >= 1
        summaries = [chain.summary for chain, _ in results]
        assert any("PostgreSQL" in s for s in summaries)

        # Search for auth
        results = rolodex.keyword_search_chains("authentication JWT")
        assert len(results) >= 1
        summaries = [chain.summary for chain, _ in results]
        assert any("JWT" in s for s in summaries)

        print("  PASS: chain_keyword_search")
    finally:
        _safe_cleanup(rolodex, db_path)


def test_chain_semantic_search():
    """Vector similarity finds closest chain."""
    rolodex, db_path = _tmp_rolodex()
    try:
        # Chain about databases (embedding points "north")
        c1 = make_chain(
            summary="Chose SQL over NoSQL for transactional guarantees",
            embedding=[0.9, 0.1, 0.0],
        )
        # Chain about UI (embedding points "east")
        c2 = make_chain(
            chain_index=1,
            summary="Redesigned the dashboard with a minimalist approach",
            embedding=[0.1, 0.9, 0.0],
        )
        # Chain about databases variant (also "north-ish")
        c3 = make_chain(
            chain_index=2,
            summary="Added database migration scripts for schema evolution",
            embedding=[0.85, 0.15, 0.05],
        )
        rolodex.create_chain(c1)
        rolodex.create_chain(c2)
        rolodex.create_chain(c3)

        # Query close to "database" chains
        results = rolodex.semantic_search_chains(
            query_embedding=[0.88, 0.12, 0.02], limit=2
        )
        assert len(results) >= 1
        # Database chains should rank higher than UI chain
        top_summary = results[0][0].summary
        assert "dashboard" not in top_summary.lower()

        print("  PASS: chain_semantic_search")
    finally:
        _safe_cleanup(rolodex, db_path)


def test_chain_hybrid_search():
    """Combined keyword + semantic search on chains."""
    rolodex, db_path = _tmp_rolodex()
    try:
        c1 = make_chain(
            summary="Implemented caching layer using Redis for session storage",
            topics=["caching", "redis"],
            embedding=[0.8, 0.2, 0.1],
        )
        c2 = make_chain(
            chain_index=1,
            summary="Optimized API response times by adding index to user table",
            topics=["optimization", "api"],
            embedding=[0.3, 0.7, 0.2],
        )
        rolodex.create_chain(c1)
        rolodex.create_chain(c2)

        # Hybrid search — keyword "caching" + embedding close to c1
        results = rolodex.hybrid_search_chains(
            query="caching Redis",
            query_embedding=[0.75, 0.25, 0.1],
            limit=2,
        )
        assert len(results) >= 1
        top_summary = results[0][0].summary
        assert "caching" in top_summary.lower() or "Redis" in top_summary

        print("  PASS: chain_hybrid_search")
    finally:
        _safe_cleanup(rolodex, db_path)


# ─── Breadcrumb Generation Tests ─────────────────────────────────────────────

def test_chain_builder_verbatim():
    """ChainBuilder heuristic summarization without API."""
    rolodex, db_path = _tmp_rolodex()
    try:
        builder = ChainBuilder(rolodex=rolodex, chain_interval=5)

        messages = [
            make_message("user", "How should we structure the database schema?", 1),
            make_message(
                "assistant",
                "I recommend using a normalized schema with separate tables for users, "
                "orders, and products. This gives us referential integrity and makes "
                "queries more predictable. The trade-off is slightly more complex joins.",
                2,
            ),
            make_message("user", "What about indexes?", 3),
            make_message(
                "assistant",
                "We should add composite indexes on the most common query patterns. "
                "For the orders table, an index on (user_id, created_at) would cover "
                "the main dashboard query efficiently.",
                4,
            ),
        ]

        chain = asyncio.run(
            builder.build_breadcrumb(
                session_id="test-session",
                messages=messages,
                turn_range_start=1,
                turn_range_end=4,
                related_entry_ids=["entry-a", "entry-b"],
            )
        )

        assert chain is not None
        assert chain.session_id == "test-session"
        assert chain.turn_range_start == 1
        assert chain.turn_range_end == 4
        assert len(chain.summary) > 0
        assert chain.related_entries == ["entry-a", "entry-b"]

        print("  PASS: chain_builder_verbatim")
    finally:
        _safe_cleanup(rolodex, db_path)


def test_chain_builder_topics():
    """Topic extraction from messages produces meaningful keywords."""
    rolodex, db_path = _tmp_rolodex()
    try:
        builder = ChainBuilder(rolodex=rolodex, chain_interval=5)

        messages = [
            make_message(
                "user",
                "Let's implement authentication using JWT tokens with OAuth2 integration",
                1,
            ),
            make_message(
                "assistant",
                "Good choice. We'll use JWT for stateless auth with short-lived tokens "
                "and refresh rotation. OAuth2 for third-party login via Google and GitHub.",
                2,
            ),
        ]

        topics = builder._extract_topics(messages)
        assert len(topics) > 0
        assert len(topics) <= 5
        # Should find technical terms, not stopwords
        for topic in topics:
            assert len(topic) > 3
            assert topic.lower() not in {"the", "and", "for", "with", "using"}

        print("  PASS: chain_builder_topics")
    finally:
        _safe_cleanup(rolodex, db_path)


def test_breadcrumb_interval_check():
    """Breadcrumb triggers at correct turn intervals."""
    rolodex, db_path = _tmp_rolodex()
    try:
        builder = ChainBuilder(rolodex=rolodex, chain_interval=5)

        # Turn 3, last chain at turn 0 → not yet (only 3 turns, need 5)
        assert builder.should_generate_breadcrumb(3, 0) is False

        # Turn 5, last chain at turn 0 → yes (exactly 5 turns)
        assert builder.should_generate_breadcrumb(5, 0) is True

        # Turn 10, last chain at turn 5 → yes (5 turns since last)
        assert builder.should_generate_breadcrumb(10, 5) is True

        # Turn 7, last chain at turn 5 → no (only 2 turns since last)
        assert builder.should_generate_breadcrumb(7, 5) is False

        # Turn 15, last chain at turn 5 → yes (10 turns, well past interval)
        assert builder.should_generate_breadcrumb(15, 5) is True

        print("  PASS: breadcrumb_interval_check")
    finally:
        _safe_cleanup(rolodex, db_path)


# ─── Integration Tests ────────────────────────────────────────────────────────

def test_pressure_deep_index_trigger():
    """Pressure monitor detects deep-index threshold."""
    pm = PressureMonitor(context_max=100_000)

    # Low pressure — no deep index
    pm.record_tokens(5, 20_000)
    assert pm.should_trigger_deep_index(0.8) is False
    assert pm.get_token_fill_ratio() < 0.3

    # Medium pressure — still no
    pm.record_tokens(6, 50_000)
    assert pm.should_trigger_deep_index(0.8) is False

    # High pressure — triggers deep index
    pm.record_tokens(7, 85_000)
    assert pm.should_trigger_deep_index(0.8) is True
    ratio = pm.get_token_fill_ratio()
    assert ratio >= 0.8, f"Expected >= 0.8, got {ratio}"

    # Custom threshold — lower bar
    pm2 = PressureMonitor(context_max=100_000)
    pm2.record_tokens(5, 60_000)
    assert pm2.should_trigger_deep_index(0.5) is True
    assert pm2.should_trigger_deep_index(0.8) is False

    print("  PASS: pressure_deep_index_trigger")


def test_chain_related_entries():
    """Chain's related_entries link to actual rolodex entries."""
    rolodex, db_path = _tmp_rolodex()
    try:
        # Create some entries
        e1 = make_entry("Removed module X endpoint", category=EntryCategory.DECISION)
        e2 = make_entry("Consolidated routing logic", category=EntryCategory.IMPLEMENTATION)
        e3 = make_entry("Unrelated Python tip", category=EntryCategory.NOTE)
        id1 = rolodex.create_entry(e1)
        id2 = rolodex.create_entry(e2)
        id3 = rolodex.create_entry(e3)

        # Create a chain referencing first two entries
        chain = make_chain(
            summary="Merged modules X and Y. Removed X endpoint, consolidated routing.",
            related_entries=[id1, id2],
        )
        rolodex.create_chain(chain)

        # Retrieve chain and fetch its related entries
        retrieved_chain = rolodex.get_chain(chain.id)
        assert retrieved_chain is not None
        assert len(retrieved_chain.related_entries) == 2

        linked_entries = rolodex.get_entries_by_ids(retrieved_chain.related_entries)
        assert len(linked_entries) == 2
        contents = [e.content for e in linked_entries]
        assert "Removed module X endpoint" in contents
        assert "Consolidated routing logic" in contents
        # Unrelated entry should NOT be in linked entries
        assert "Unrelated Python tip" not in contents

        print("  PASS: chain_related_entries")
    finally:
        _safe_cleanup(rolodex, db_path)


def test_context_builder_chain_formatting():
    """Context builder formats chains before discrete entries."""
    builder = ContextBuilder()

    chains = [
        ReasoningChain(
            id="c1",
            session_id="s1",
            chain_index=0,
            turn_range_start=1,
            turn_range_end=5,
            summary="Decided to use PostgreSQL over MongoDB for ACID compliance.",
            topics=["database", "postgresql"],
        ),
        ReasoningChain(
            id="c2",
            session_id="s1",
            chain_index=1,
            turn_range_start=6,
            turn_range_end=10,
            summary="Implemented JWT auth. Rejected session cookies for statelessness.",
            topics=["auth", "jwt"],
        ),
    ]

    entries = [
        RolodexEntry(
            id="e1",
            conversation_id="s1",
            content="PostgreSQL chosen for ACID compliance",
            category=EntryCategory.DECISION,
        ),
    ]

    block = builder.build_context_block(entries, chains=chains)

    # Entries appear FIRST (the "what" — primary factual frame),
    # chains appear SECOND (the "why" — supplementary narrative).
    chain_pos = block.find("REASONING CONTEXT")
    entry_pos = block.find("RETRIEVED FROM MEMORY")
    assert chain_pos >= 0, "Chain header not found in context block"
    assert entry_pos >= 0, "Entry header not found in context block"
    assert entry_pos < chain_pos, "Entries should appear before chains (facts first, narrative second)"

    # Verify chain content is present
    assert "PostgreSQL over MongoDB" in block
    assert "JWT auth" in block
    assert "turns 1-5" in block
    assert "turns 6-10" in block

    # Verify entry content is present
    assert "PostgreSQL chosen for ACID" in block

    print("  PASS: context_builder_chain_formatting")


def test_context_builder_no_chains():
    """Context builder works normally when no chains are provided."""
    builder = ContextBuilder()

    entries = [
        RolodexEntry(
            id="e1",
            conversation_id="s1",
            content="Python uses indentation for blocks",
            category=EntryCategory.FACT,
        ),
    ]

    # No chains — should produce standard output
    block_no_chains = builder.build_context_block(entries, chains=None)
    assert "REASONING CONTEXT" not in block_no_chains
    assert "RETRIEVED FROM MEMORY" in block_no_chains
    assert "Python uses indentation" in block_no_chains

    # Empty chains list — same result
    block_empty = builder.build_context_block(entries, chains=[])
    assert "REASONING CONTEXT" not in block_empty

    print("  PASS: context_builder_no_chains")


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Reasoning Chain tests (Phase 7)...\n")

    print("--- Chain CRUD ---\n")
    test_chain_crud()
    test_chain_serialization()
    test_chain_linked_list()

    print("\n--- Chain Search ---\n")
    test_chain_keyword_search()
    test_chain_semantic_search()
    test_chain_hybrid_search()

    print("\n--- Breadcrumb Generation ---\n")
    test_chain_builder_verbatim()
    test_chain_builder_topics()
    test_breadcrumb_interval_check()

    print("\n--- Integration ---\n")
    test_pressure_deep_index_trigger()
    test_chain_related_entries()
    test_context_builder_chain_formatting()
    test_context_builder_no_chains()

    print("\nAll Phase 7 tests passed!")
