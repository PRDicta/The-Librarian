"""
Tests for Phase 3: Proactive Preloading.
Validates pressure monitoring, embedding prediction,
cache warming without access inflation, and confidence-based
injection splitting.
"""
import os
import sys
import asyncio
import tempfile
import uuid
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.types import (
    RolodexEntry, ContentModality, EntryCategory, Tier,
    Message, MessageRole, PreloadPrediction, PreloadResult
)
from src.storage.rolodex import Rolodex
from src.preloading.pressure import PressureMonitor
from src.preloading.predictor import EmbeddingPredictor
from src.preloading.preloader import Preloader


def make_entry(content: str, tags: list = None, category: EntryCategory = EntryCategory.NOTE,
               embedding: list = None, conversation_id: str = "test-conv") -> RolodexEntry:
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


# ─── Pressure Monitor Tests ─────────────────────────────────────────────────

def test_pressure_calculation():
    """Verify pressure formula produces correct value from known inputs."""
    pm = PressureMonitor(window_size=10, context_max=100_000)

    # Record 5 gaps at turns 3-7
    for t in range(3, 8):
        pm.record_gap(t)

    # Record 5 queries, all cache misses
    for t in range(3, 8):
        pm.record_query(t, cache_hit=False)

    # Record token count at turn 7
    pm.record_tokens(7, 50_000)

    pressure = pm.get_pressure()

    # Expected:
    #   gap_rate       = 5 / 10 = 0.5
    #   cache_miss_rate = 5 / 5  = 1.0
    #   token_velocity  = 50000 / 100000 = 0.5
    #   pressure = 0.5*0.5 + 1.0*0.3 + 0.5*0.2 = 0.25 + 0.30 + 0.10 = 0.65
    assert abs(pressure - 0.65) < 0.01, f"Expected ~0.65, got {pressure}"

    # Also verify it returns 0.0 when insufficient data
    pm_early = PressureMonitor(window_size=10, context_max=100_000)
    pm_early.record_tokens(1, 10_000)
    assert pm_early.get_pressure() == 0.0, "Should return 0 when turn < 3"
    print("  PASS: pressure_calculation")


def test_pressure_strategy_gating():
    """Low/medium/high pressure selects correct strategy."""
    # "none" — insufficient data (< 3 turns)
    pm_none = PressureMonitor(window_size=10, context_max=100_000)
    pm_none.record_tokens(1, 1000)
    assert pm_none.get_strategy(0.3, 0.7) == "none", "Should be 'none' before turn 3"

    # "embedding" — low pressure but sufficient data (>= 3 turns)
    pm_embed = PressureMonitor(window_size=10, context_max=100_000)
    pm_embed.record_tokens(3, 1000)  # Minimal tokens, no gaps, no misses
    strategy = pm_embed.get_strategy(0.3, 0.7)
    assert strategy == "embedding", f"Expected 'embedding' at low pressure, got '{strategy}'"

    # "llm" — very high pressure
    pm_llm = PressureMonitor(window_size=10, context_max=100_000)
    for t in range(3, 13):
        pm_llm.record_gap(t)
        pm_llm.record_query(t, cache_hit=False)
        pm_llm.record_tokens(t, 90_000)
    # gap_rate=1.0, miss_rate=1.0, token_vel=0.9
    # pressure = 0.5 + 0.3 + 0.18 = 0.98
    strategy = pm_llm.get_strategy(0.3, 0.7)
    assert strategy == "llm", f"Expected 'llm' at high pressure, got '{strategy}'"
    print("  PASS: pressure_strategy_gating")


# ─── Embedding Predictor Test ────────────────────────────────────────────────

def test_embedding_predictor():
    """Embedding predictor finds semantically relevant entries by proximity."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    rolodex = Rolodex(db_path)
    try:
        # Create entries with known embeddings
        e_python = make_entry(
            "Python async patterns and coroutines",
            embedding=[0.9, 0.1, 0.0],
        )
        e_js = make_entry(
            "JavaScript callback hell and promises",
            embedding=[0.1, 0.9, 0.0],
        )
        e_asyncio = make_entry(
            "Python asyncio event loop internals",
            embedding=[0.85, 0.15, 0.05],
        )
        rolodex.batch_create_entries([e_python, e_js, e_asyncio])

        # Mock embedding manager — returns a vector near Python entries
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text = AsyncMock(return_value=[0.88, 0.12, 0.02])

        predictor = EmbeddingPredictor(rolodex, mock_embeddings)
        user_msg = Message(role=MessageRole.USER, content="Tell me about async in Python")

        predictions = asyncio.run(predictor.predict([user_msg], limit=3, min_similarity=0.3))

        assert len(predictions) >= 2, f"Expected >=2 predictions, got {len(predictions)}"

        # Top prediction should be one of the Python entries, not JavaScript
        top = predictions[0]
        assert top.confidence > 0.8, f"Expected high confidence, got {top.confidence}"
        assert top.strategy == "embedding"
        assert top.entry_id != e_js.id, "JavaScript entry should not be the top prediction"
        print("  PASS: embedding_predictor")
    finally:
        _safe_cleanup(rolodex, db_path)


# ─── Preloader Integration Tests ─────────────────────────────────────────────

def _make_preloader(rolodex, pressure_monitor):
    """Helper: build a Preloader with mocked embedding manager."""
    mock_embeddings = MagicMock()
    preloader = Preloader(
        rolodex=rolodex,
        embedding_manager=mock_embeddings,
        pressure_monitor=pressure_monitor,
    )
    return preloader


def _make_active_pressure_monitor():
    """Helper: create a PressureMonitor with enough data to enable preloading."""
    pm = PressureMonitor(window_size=10, context_max=100_000)
    for t in range(3, 8):
        pm.record_tokens(t, 10_000)
    return pm


def test_preload_warms_cache():
    """Preloaded entries land in the hot cache."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    rolodex = Rolodex(db_path)
    try:
        entry = make_entry("Important preloadable fact", embedding=[0.5, 0.5, 0.0])
        eid = rolodex.create_entry(entry)

        # Clear cache so we can verify preloading puts it back
        rolodex._hot_cache.clear()
        assert eid not in rolodex._hot_cache

        pm = _make_active_pressure_monitor()
        preloader = _make_preloader(rolodex, pm)

        # Mock predictor to return our entry with moderate confidence
        preloader.embedding_predictor.predict = AsyncMock(
            return_value=[PreloadPrediction(
                entry_id=eid, confidence=0.6, strategy="embedding"
            )]
        )

        user_msg = Message(role=MessageRole.USER, content="Test message")
        result = asyncio.run(preloader.preload(
            recent_messages=[user_msg],
            turn_number=5,
            conversation_id="test",
            injection_confidence=0.8,
        ))

        # Entry should now be in hot cache
        assert eid in rolodex._hot_cache, "Preloaded entry should be in hot cache"
        assert result.strategy_used == "embedding"
        print("  PASS: preload_warms_cache")
    finally:
        _safe_cleanup(rolodex, db_path)


def test_preload_no_access_count():
    """Preloading does NOT inflate access counts — it's speculative."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    rolodex = Rolodex(db_path)
    try:
        entry = make_entry("Fact that should keep zero accesses", embedding=[0.5, 0.5, 0.0])
        eid = rolodex.create_entry(entry)

        # Verify initial access count
        row = rolodex.conn.execute(
            "SELECT access_count FROM rolodex_entries WHERE id = ?", (eid,)
        ).fetchone()
        assert row["access_count"] == 0

        # Clear cache, run preload
        rolodex._hot_cache.clear()
        pm = _make_active_pressure_monitor()
        preloader = _make_preloader(rolodex, pm)
        preloader.embedding_predictor.predict = AsyncMock(
            return_value=[PreloadPrediction(
                entry_id=eid, confidence=0.6, strategy="embedding"
            )]
        )

        user_msg = Message(role=MessageRole.USER, content="Test")
        asyncio.run(preloader.preload(
            recent_messages=[user_msg],
            turn_number=5,
            conversation_id="test",
        ))

        # Access count must still be 0 — preloading uses _cache_put, not update_access
        row = rolodex.conn.execute(
            "SELECT access_count FROM rolodex_entries WHERE id = ?", (eid,)
        ).fetchone()
        assert row["access_count"] == 0, (
            f"Preloading should not inflate access_count; got {row['access_count']}"
        )
        print("  PASS: preload_no_access_count")
    finally:
        _safe_cleanup(rolodex, db_path)


def test_high_confidence_injection():
    """Entries above injection threshold are marked for proactive injection."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    rolodex = Rolodex(db_path)
    try:
        entry = make_entry("Extremely relevant fact", embedding=[0.5, 0.5, 0.0])
        eid = rolodex.create_entry(entry)
        rolodex._hot_cache.clear()

        pm = _make_active_pressure_monitor()
        preloader = _make_preloader(rolodex, pm)

        # Return prediction with confidence ABOVE 0.8 threshold
        preloader.embedding_predictor.predict = AsyncMock(
            return_value=[PreloadPrediction(
                entry_id=eid, confidence=0.95, strategy="embedding"
            )]
        )

        user_msg = Message(role=MessageRole.USER, content="Test")
        result = asyncio.run(preloader.preload(
            recent_messages=[user_msg],
            turn_number=5,
            conversation_id="test",
            injection_confidence=0.8,
        ))

        # High confidence → injected, not just cache-warmed
        assert len(result.injected_entries) == 1, (
            f"Expected 1 injected entry, got {len(result.injected_entries)}"
        )
        assert result.injected_entries[0].id == eid
        assert len(result.cache_warmed_entries) == 0, (
            "High-confidence entry should not be in cache_warmed list"
        )
        print("  PASS: high_confidence_injection")
    finally:
        _safe_cleanup(rolodex, db_path)


def test_low_confidence_cache_only():
    """Entries below injection threshold only warm the cache, no injection."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    rolodex = Rolodex(db_path)
    try:
        entry = make_entry("Somewhat relevant fact", embedding=[0.5, 0.5, 0.0])
        eid = rolodex.create_entry(entry)
        rolodex._hot_cache.clear()

        pm = _make_active_pressure_monitor()
        preloader = _make_preloader(rolodex, pm)

        # Return prediction with confidence BELOW 0.8 threshold
        preloader.embedding_predictor.predict = AsyncMock(
            return_value=[PreloadPrediction(
                entry_id=eid, confidence=0.5, strategy="embedding"
            )]
        )

        user_msg = Message(role=MessageRole.USER, content="Test")
        result = asyncio.run(preloader.preload(
            recent_messages=[user_msg],
            turn_number=5,
            conversation_id="test",
            injection_confidence=0.8,
        ))

        # Low confidence → cache-warmed only, NOT injected
        assert len(result.injected_entries) == 0, (
            f"Expected 0 injected entries, got {len(result.injected_entries)}"
        )
        assert len(result.cache_warmed_entries) == 1, (
            f"Expected 1 cache-warmed entry, got {len(result.cache_warmed_entries)}"
        )
        assert result.cache_warmed_entries[0].id == eid

        # But the entry should still be in the hot cache
        assert eid in rolodex._hot_cache, "Entry should be warmed into hot cache"
        print("  PASS: low_confidence_cache_only")
    finally:
        _safe_cleanup(rolodex, db_path)


if __name__ == "__main__":
    print("Running Preloading tests (Phase 3)...\n")
    test_pressure_calculation()
    test_pressure_strategy_gating()
    test_embedding_predictor()
    test_preload_warms_cache()
    test_preload_no_access_count()
    test_high_confidence_injection()
    test_low_confidence_cache_only()
    print("\nAll Phase 3 tests passed!")
