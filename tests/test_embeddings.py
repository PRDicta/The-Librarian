"""
Tests for EmbeddingManager and CostTracker (Phase 6a)

Tests the embedding strategies (hash, local fallback), cost tracking,
and strategy selection. Sentence-transformers and Voyage are mocked
since they may not be available in all environments.
"""
import asyncio
import os
import sys
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.indexing.embeddings import EmbeddingManager
from src.utils.cost_tracker import CostTracker, PRICING


# ─── Embedding Tests ─────────────────────────────────────────────────────────

def test_hash_embed_deterministic():
    """Hash strategy: same text always produces same embedding."""
    mgr = EmbeddingManager(strategy="hash", dimensions=384)
    e1 = asyncio.run(mgr.embed_text("hello world"))
    e2 = asyncio.run(mgr.embed_text("hello world"))
    assert e1 == e2, "Hash embedding should be deterministic"
    assert len(e1) == 384, f"Expected 384 dims, got {len(e1)}"
    # Different text → different embedding
    e3 = asyncio.run(mgr.embed_text("goodbye world"))
    assert e1 != e3, "Different text should produce different embedding"
    print("  PASS: hash_embed_deterministic")


def test_hash_embed_unit_vector():
    """Hash embeddings should be unit vectors (norm ≈ 1.0)."""
    mgr = EmbeddingManager(strategy="hash", dimensions=384)
    emb = asyncio.run(mgr.embed_text("test vector normalization"))
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 0.01, f"Expected unit vector, got norm {norm}"
    print("  PASS: hash_embed_unit_vector")


def test_local_fallback_to_hash():
    """When sentence-transformers unavailable, local strategy falls back to hash."""
    # Force sentence_transformers import to fail
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        mgr = EmbeddingManager(strategy="local", dimensions=384)
        # _get_local_model should return None due to import failure
        mgr._local_model = None  # Reset any cached model
        result = asyncio.run(mgr.embed_text("fallback test"))
        assert len(result) == 384, f"Expected 384 dims, got {len(result)}"
        # Verify it's deterministic (hash behavior)
        result2 = asyncio.run(mgr.embed_text("fallback test"))
        assert result == result2, "Fallback should be deterministic (hash)"
    print("  PASS: local_fallback_to_hash")


def test_local_embed_with_mock():
    """Local strategy uses sentence-transformers when available."""
    mock_model = MagicMock()
    fake_embedding = np.random.randn(384).astype(np.float32)
    fake_embedding = fake_embedding / np.linalg.norm(fake_embedding)
    mock_model.encode.return_value = fake_embedding

    mgr = EmbeddingManager(strategy="local", dimensions=384)
    mgr._local_model = mock_model  # Inject mock model

    result = asyncio.run(mgr.embed_text("test with local model"))
    assert len(result) == 384, f"Expected 384 dims, got {len(result)}"
    mock_model.encode.assert_called_once()
    print("  PASS: local_embed_with_mock")


def test_voyage_fallback_to_hash():
    """When Voyage API not configured, embedding still works via fallback."""
    mgr = EmbeddingManager(strategy="anthropic", dimensions=384)
    # No voyage client initialized (no key), so _voyage_client is None
    assert mgr._voyage_client is None, "Should have no Voyage client without key"
    # embed_text should still work — falls through to hash
    result = asyncio.run(mgr.embed_text("voyage fallback test"))
    assert len(result) == 384, f"Expected 384 dims, got {len(result)}"
    # Should be deterministic (hash behavior since no client)
    result2 = asyncio.run(mgr.embed_text("voyage fallback test"))
    assert result == result2, "Should fall back to deterministic hash"
    print("  PASS: voyage_fallback_to_hash")


def test_similarity():
    """Cosine similarity: identical vectors → 1.0, orthogonal → 0.0."""
    mgr = EmbeddingManager(strategy="hash")
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert abs(mgr.similarity(a, b) - 1.0) < 0.001, "Identical vectors should have sim ~1.0"
    c = [0.0, 1.0, 0.0]
    assert abs(mgr.similarity(a, c)) < 0.001, "Orthogonal vectors should have sim ~0.0"
    print("  PASS: similarity")


# ─── Cost Tracker Tests ──────────────────────────────────────────────────────

def test_cost_tracker_records_calls():
    """CostTracker accumulates API call costs correctly."""
    tracker = CostTracker()

    # Record a Haiku extraction call (500 input, 200 output tokens)
    call = tracker.record("extraction", "claude-haiku-4-5-20251001", 500, 200)
    assert call.cost_usd > 0, "Cost should be positive"
    assert call.input_tokens == 500
    assert call.output_tokens == 200

    # Expected: (500/1M * 0.80) + (200/1M * 4.00) = 0.0004 + 0.0008 = 0.0012
    expected = (500 / 1_000_000 * 0.80) + (200 / 1_000_000 * 4.00)
    assert abs(call.cost_usd - expected) < 0.000001, f"Expected {expected}, got {call.cost_usd}"

    # Session total should match
    assert abs(tracker.get_session_cost() - expected) < 0.000001
    assert tracker.get_call_count() == 1
    print("  PASS: cost_tracker_records_calls")


def test_cost_tracker_breakdown():
    """CostTracker provides per-category breakdown."""
    tracker = CostTracker()

    # Mix of extraction and embedding calls
    tracker.record("extraction", "claude-haiku-4-5-20251001", 1000, 500)
    tracker.record("extraction", "claude-haiku-4-5-20251001", 800, 300)
    tracker.record("embedding", "voyage-3", 2000, 0)
    tracker.record("negotiation", "claude-haiku-4-5-20251001", 600, 100)

    breakdown = tracker.get_breakdown()
    assert "extraction" in breakdown, "Should have extraction category"
    assert "embedding" in breakdown, "Should have embedding category"
    assert "negotiation" in breakdown, "Should have negotiation category"

    assert breakdown["extraction"]["call_count"] == 2
    assert breakdown["embedding"]["call_count"] == 1
    assert breakdown["negotiation"]["call_count"] == 1

    # Embedding cost should be very low (Voyage is cheap)
    assert breakdown["embedding"]["cost_usd"] < breakdown["extraction"]["cost_usd"]

    # Total should sum up
    total = sum(cat["cost_usd"] for cat in breakdown.values())
    assert abs(total - tracker.get_session_cost()) < 0.000001

    # Summary should include everything
    summary = tracker.get_summary()
    assert summary["total_calls"] == 4
    assert summary["total_cost_usd"] > 0
    print("  PASS: cost_tracker_breakdown")


# ─── Runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("hash_embed_deterministic", test_hash_embed_deterministic),
        ("hash_embed_unit_vector", test_hash_embed_unit_vector),
        ("local_fallback_to_hash", test_local_fallback_to_hash),
        ("local_embed_with_mock", test_local_embed_with_mock),
        ("voyage_fallback_to_hash", test_voyage_fallback_to_hash),
        ("similarity", test_similarity),
        ("cost_tracker_records_calls", test_cost_tracker_records_calls),
        ("cost_tracker_breakdown", test_cost_tracker_breakdown),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name} — {e}")
            failed += 1

    print(f"\nEmbedding + Cost tests: {passed} passed, {failed} failed, {passed + failed} total")
    if failed > 0:
        sys.exit(1)
