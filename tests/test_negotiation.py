"""
Tests for ContextNegotiator (Phase 6c)

Tests the negotiation protocol using the heuristic fallback
(no actual API calls). Validates accept/reject logic, budget
enforcement, round limits, and integration with cost tracker
and pressure monitor.
"""
import asyncio
import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.negotiator import ContextNegotiator, NegotiationResult
from src.core.types import RolodexEntry, ContentModality, EntryCategory
from src.utils.cost_tracker import CostTracker
from src.preloading.pressure import PressureMonitor


def _make_entry(entry_id, content, category=EntryCategory.FACT, tags=None):
    """Helper to create a test RolodexEntry."""
    return RolodexEntry(
        id=entry_id,
        conversation_id="test-conv",
        content=content,
        content_type=ContentModality.PROSE,
        category=category,
        tags=tags or [],
        source_range={"turn_number": 1},
        embedding=[0.0] * 10,
    )


# ─── Negotiation Tests ──────────────────────────────────────────────────────

def test_negotiate_accepts_high_relevance():
    """Entries with high relevance scores should be accepted."""
    # Use heuristic fallback (no API key → _evaluate_candidates will fail → fallback)
    negotiator = ContextNegotiator(api_key="fake-key", max_rounds=1)

    entries = [
        _make_entry("e1", "Philip prefers dark mode for all interfaces"),
        _make_entry("e2", "The project uses Python 3.12 with asyncio"),
        _make_entry("e3", "Meeting notes from January 5th about Q1 planning"),
    ]
    # High scores → all should be accepted by heuristic (threshold 0.5)
    scores = {"e1": 0.92, "e2": 0.85, "e3": 0.78}

    result = asyncio.run(negotiator.negotiate(
        gap_topic="user preferences",
        candidate_entries=entries,
        relevance_scores=scores,
        budget_tokens=5000,
    ))

    assert isinstance(result, NegotiationResult)
    assert len(result.accepted_entries) == 3, f"All 3 should be accepted, got {len(result.accepted_entries)}"
    assert result.resolved is True
    assert result.total_rounds >= 1
    print("  PASS: negotiate_accepts_high_relevance")


def test_negotiate_rejects_low_relevance():
    """Entries with low relevance scores should be rejected."""
    negotiator = ContextNegotiator(api_key="fake-key", max_rounds=1)

    entries = [
        _make_entry("e1", "Philip prefers dark mode"),
        _make_entry("e2", "Unrelated meeting notes from March"),
        _make_entry("e3", "Random note about weather"),
    ]
    # Mix of high and low scores
    scores = {"e1": 0.90, "e2": 0.20, "e3": 0.10}

    result = asyncio.run(negotiator.negotiate(
        gap_topic="user preferences",
        candidate_entries=entries,
        relevance_scores=scores,
        budget_tokens=5000,
    ))

    accepted_ids = {e.id for e in result.accepted_entries}
    assert "e1" in accepted_ids, "High-relevance e1 should be accepted"
    assert "e2" not in accepted_ids, "Low-relevance e2 should be rejected"
    assert "e3" not in accepted_ids, "Low-relevance e3 should be rejected"
    assert len(result.rejected_ids) >= 2
    print("  PASS: negotiate_rejects_low_relevance")


def test_negotiate_respects_budget():
    """Negotiator should stop accepting entries when budget is exhausted."""
    negotiator = ContextNegotiator(api_key="fake-key", max_rounds=1)

    # Create entries with substantial content (~200 tokens each)
    long_content = "This is a detailed explanation of the concept. " * 20  # ~200 tokens
    entries = [
        _make_entry("e1", long_content),
        _make_entry("e2", long_content),
        _make_entry("e3", long_content),
        _make_entry("e4", long_content),
        _make_entry("e5", long_content),
    ]
    # All high relevance
    scores = {"e1": 0.9, "e2": 0.85, "e3": 0.8, "e4": 0.75, "e5": 0.7}

    # Very tight budget — should only fit 1-2 entries
    result = asyncio.run(negotiator.negotiate(
        gap_topic="detailed explanation",
        candidate_entries=entries,
        relevance_scores=scores,
        budget_tokens=300,  # ~1-2 entries worth
    ))

    assert len(result.accepted_entries) < 5, "Budget should limit accepted entries"
    assert result.budget_remaining >= 0, "Should not go over budget"
    assert result.budget_used <= 300, f"Budget used {result.budget_used} exceeds 300"
    print("  PASS: negotiate_respects_budget")


def test_negotiate_max_rounds():
    """Negotiator should stop after max_rounds even if unresolved."""
    negotiator = ContextNegotiator(api_key="fake-key", max_rounds=2)

    entries = [
        _make_entry("e1", "Some marginally relevant content"),
    ]
    scores = {"e1": 0.55}  # Borderline — might trigger refinement

    result = asyncio.run(negotiator.negotiate(
        gap_topic="something specific",
        candidate_entries=entries,
        relevance_scores=scores,
        budget_tokens=5000,
    ))

    assert result.total_rounds <= 2, f"Should stop after 2 rounds, ran {result.total_rounds}"
    print("  PASS: negotiate_max_rounds")


def test_negotiate_empty_candidates():
    """Negotiating with no candidates should return empty result."""
    negotiator = ContextNegotiator(api_key="fake-key")

    result = asyncio.run(negotiator.negotiate(
        gap_topic="anything",
        candidate_entries=[],
        relevance_scores={},
        budget_tokens=5000,
    ))

    assert result.accepted_entries == []
    assert result.resolved is False
    assert result.total_rounds == 0
    assert result.budget_used == 0
    print("  PASS: negotiate_empty_candidates")


def test_negotiate_records_outcome_to_pressure():
    """Negotiation outcome should update the pressure monitor."""
    pm = PressureMonitor(window_size=10, context_max=100000)
    initial_gaps = len(pm._gap_events)

    # Unresolved negotiation should increase pressure (records gap)
    pm.record_negotiation(resolved=False, budget_used=0, rounds=1)
    assert len(pm._gap_events) > initial_gaps, "Unresolved should record a gap"

    # Resolved negotiation should not add more gaps
    gaps_after = len(pm._gap_events)
    pm.record_negotiation(resolved=True, budget_used=500, rounds=1)
    assert len(pm._gap_events) == gaps_after, "Resolved should not add gaps"
    print("  PASS: negotiate_records_outcome_to_pressure")


def test_negotiate_cost_tracked():
    """Negotiation API calls should be recorded in cost tracker."""
    tracker = CostTracker()
    # Simulate what would happen if API call succeeded
    tracker.record("negotiation", "claude-haiku-4-5-20251001", 500, 100)

    breakdown = tracker.get_breakdown()
    assert "negotiation" in breakdown
    assert breakdown["negotiation"]["call_count"] == 1
    assert breakdown["negotiation"]["cost_usd"] > 0
    print("  PASS: negotiate_cost_tracked")


def test_heuristic_fallback():
    """When API fails, heuristic should accept score >= 0.5, reject < 0.5."""
    negotiator = ContextNegotiator(api_key="fake-key", max_rounds=1)

    entries = [
        _make_entry("high", "Very relevant entry"),
        _make_entry("mid", "Somewhat relevant entry"),
        _make_entry("low", "Not relevant entry"),
    ]
    scores = {"high": 0.8, "mid": 0.5, "low": 0.3}

    # Test the heuristic directly
    evaluation = negotiator._heuristic_evaluation(entries, scores, budget=5000)
    decisions = {d["entry_id"]: d["action"] for d in evaluation["decisions"]}

    assert decisions["high"] == "ACCEPT", "Score 0.8 should be accepted"
    assert decisions["mid"] == "ACCEPT", "Score 0.5 should be accepted (threshold)"
    assert decisions["low"] == "REJECT", "Score 0.3 should be rejected"
    assert evaluation["confidence"] == 0.6, "Heuristic confidence should be 0.6"
    print("  PASS: heuristic_fallback")


# ─── Runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("negotiate_accepts_high_relevance", test_negotiate_accepts_high_relevance),
        ("negotiate_rejects_low_relevance", test_negotiate_rejects_low_relevance),
        ("negotiate_respects_budget", test_negotiate_respects_budget),
        ("negotiate_max_rounds", test_negotiate_max_rounds),
        ("negotiate_empty_candidates", test_negotiate_empty_candidates),
        ("negotiate_records_outcome_to_pressure", test_negotiate_records_outcome_to_pressure),
        ("negotiate_cost_tracked", test_negotiate_cost_tracked),
        ("heuristic_fallback", test_heuristic_fallback),
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

    print(f"\nNegotiation tests: {passed} passed, {failed} failed, {passed + failed} total")
    if failed > 0:
        sys.exit(1)
