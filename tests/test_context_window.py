"""
Tests for the Context Window Manager (Phase 9).

Verifies:
- Token budget enforcement
- Ingestion checkpoint safety net
- Minimum active turns guarantee
- Bridge summary generation
- Context payload assembly
- Integration with TheLibrarian
"""
import pytest
from datetime import datetime

from src.core.context_window import ContextWindowManager, IngestionCheckpoint
from src.core.types import Message, MessageRole, ConversationState, estimate_tokens


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_message(role="user", content="Hello world", turn=1, tokens=None):
    """Create a test message."""
    msg = Message(
        role=MessageRole.USER if role == "user" else MessageRole.ASSISTANT,
        content=content,
        turn_number=turn,
    )
    if tokens is not None:
        msg.token_count = tokens
    return msg


def make_conversation(num_turns=10, tokens_per_msg=500):
    """Create a conversation with N turns, each user+assistant pair."""
    messages = []
    for i in range(1, num_turns + 1):
        content = "x" * (tokens_per_msg * 4)  # ~tokens_per_msg tokens
        messages.append(make_message("user", content, turn=i * 2 - 1))
        messages.append(make_message("assistant", content, turn=i * 2))
    return messages


# ─── Unit Tests: Token Budget ────────────────────────────────────────────────

def test_all_messages_within_budget():
    """When total tokens < budget, all messages stay active."""
    mgr = ContextWindowManager(token_budget=50_000)
    messages = [make_message(tokens=100) for _ in range(5)]
    active, pruned = mgr.compute_active_window(messages)
    assert len(active) == 5
    assert len(pruned) == 0


def test_exceeds_budget_prunes_oldest():
    """When total tokens > budget, oldest messages get pruned."""
    mgr = ContextWindowManager(token_budget=3_000, min_active_turns=2)
    messages = [make_message(tokens=1_000, turn=i) for i in range(1, 6)]
    # 5 messages × 1000 tokens = 5000 > 3000 budget
    active, pruned = mgr.compute_active_window(messages)
    assert len(active) == 3  # 3 × 1000 = 3000 = budget
    assert len(pruned) == 2


def test_min_active_turns_overrides_budget():
    """Even if budget exceeded, keep at least min_active_turns."""
    mgr = ContextWindowManager(token_budget=100, min_active_turns=4)
    messages = [make_message(tokens=1_000, turn=i) for i in range(1, 6)]
    # Budget says keep ~0, but min_active_turns=4 overrides
    active, pruned = mgr.compute_active_window(messages)
    assert len(active) >= 4


def test_empty_messages():
    """Empty conversation returns empty lists."""
    mgr = ContextWindowManager()
    active, pruned = mgr.compute_active_window([])
    assert active == []
    assert pruned == []


# ─── Unit Tests: Ingestion Checkpoints ────────────────────────────────────────

def test_checkpoint_recording():
    """Checkpoints are recorded and retrievable."""
    mgr = ContextWindowManager()
    mgr.record_checkpoint(turn_number=5, entry_count=2, token_count=500)
    mgr.record_checkpoint(turn_number=10, entry_count=1, token_count=300)
    assert mgr.total_checkpoints == 2
    assert mgr.last_checkpoint_turn == 10


def test_checkpoint_prevents_pruning_past_safe_point():
    """Never prune past the last ingestion checkpoint."""
    mgr = ContextWindowManager(token_budget=2_000, min_active_turns=2)
    messages = [make_message(tokens=1_000, turn=i) for i in range(1, 8)]
    # 7 messages × 1000 = 7000 tokens, budget 2000
    # Without checkpoint: would keep last 2
    # With checkpoint at turn 4: must keep turns 5-7 (3 messages)

    # Record checkpoint up to turn 4 (turns 1-4 are safe)
    mgr.record_checkpoint(turn_number=4, entry_count=1, token_count=1000)

    active, pruned = mgr.compute_active_window(messages)

    # All active messages should have turn_number > 4
    for msg in active:
        assert msg.turn_number >= 5, (
            f"Message with turn {msg.turn_number} should not be in active "
            f"window — it's before checkpoint turn 4"
        )


def test_no_checkpoint_allows_full_pruning():
    """Without any checkpoints, only budget + min_active_turns matter."""
    mgr = ContextWindowManager(token_budget=2_000, min_active_turns=2)
    messages = [make_message(tokens=1_000, turn=i) for i in range(1, 8)]
    active, pruned = mgr.compute_active_window(messages)
    assert len(active) == 2  # Budget allows 2, min is 2


# ─── Unit Tests: Bridge Summary ──────────────────────────────────────────────

def test_bridge_summary_generated_on_prune():
    """Pruning messages generates a bridge summary."""
    mgr = ContextWindowManager(token_budget=1_000, min_active_turns=2)
    messages = [
        make_message("user", "Tell me about Python decorators", turn=1, tokens=500),
        make_message("assistant", "Decorators wrap functions...", turn=2, tokens=500),
        make_message("user", "Now implement one", turn=3, tokens=500),
        make_message("assistant", "Here is the implementation...", turn=4, tokens=500),
    ]
    active, pruned = mgr.compute_active_window(messages)
    assert len(pruned) > 0
    assert mgr.bridge_summary != ""
    assert "Context Bridge" in mgr.bridge_summary


def test_no_bridge_when_nothing_pruned():
    """No bridge summary when all messages fit in the window."""
    mgr = ContextWindowManager(token_budget=50_000)
    messages = [make_message(tokens=100, turn=i) for i in range(1, 4)]
    active, pruned = mgr.compute_active_window(messages)
    assert len(pruned) == 0
    assert mgr.bridge_summary == ""


# ─── Unit Tests: Context Payload ──────────────────────────────────────────────

def test_context_payload_structure():
    """Payload has all required fields."""
    mgr = ContextWindowManager(token_budget=50_000)
    state = ConversationState()
    state.add_message(MessageRole.USER, "Hello")
    state.add_message(MessageRole.ASSISTANT, "Hi there")

    payload = mgr.build_context_payload(state, recall_block="some recalled context")

    assert "bridge_summary" in payload
    assert "recall_block" in payload
    assert "active_messages" in payload
    assert "metadata" in payload
    assert payload["recall_block"] == "some recalled context"
    assert len(payload["active_messages"]) == 2


def test_context_payload_with_pruning():
    """Payload includes bridge summary when messages are pruned."""
    mgr = ContextWindowManager(token_budget=500, min_active_turns=1)
    state = ConversationState()
    for i in range(10):
        state.add_message(
            MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
            "x" * 400,  # ~100 tokens each
        )

    payload = mgr.build_context_payload(state)
    meta = payload["metadata"]

    assert meta["pruned_messages"] > 0
    assert meta["active_messages"] > 0
    assert meta["active_messages"] + meta["pruned_messages"] == meta["total_messages"]
    assert payload["bridge_summary"] != ""


def test_context_payload_metadata_budget():
    """Metadata includes correct budget remaining."""
    mgr = ContextWindowManager(token_budget=10_000)
    state = ConversationState()
    state.add_message(MessageRole.USER, "x" * 4_000)  # ~1000 tokens

    payload = mgr.build_context_payload(state)
    meta = payload["metadata"]

    assert meta["budget_remaining"] > 0
    assert meta["budget_remaining"] == 10_000 - meta["active_tokens"]


# ─── Unit Tests: State Reporting ──────────────────────────────────────────────

def test_get_state():
    """get_state returns correct ContextWindowState."""
    mgr = ContextWindowManager(token_budget=2_000, min_active_turns=2)
    messages = [make_message(tokens=1_000, turn=i) for i in range(1, 6)]
    mgr.record_checkpoint(turn_number=3, entry_count=1)

    state = mgr.get_state(messages)

    assert state.total_messages == 5
    assert state.active_messages + state.pruned_messages == 5
    assert state.last_checkpoint_turn == 3


def test_get_stats():
    """get_stats returns dict for system stats."""
    mgr = ContextWindowManager(token_budget=20_000, min_active_turns=4)
    mgr.record_checkpoint(turn_number=5)

    stats = mgr.get_stats()

    assert stats["token_budget"] == 20_000
    assert stats["min_active_turns"] == 4
    assert stats["checkpoints"] == 1
    assert stats["last_checkpoint_turn"] == 5


# ─── Integration: With ConversationState ──────────────────────────────────────

def test_integration_with_conversation_state():
    """Test the full flow: add messages, record checkpoints, compute window."""
    mgr = ContextWindowManager(token_budget=3_000, min_active_turns=2)
    state = ConversationState()

    # Simulate a 10-turn conversation
    for i in range(10):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        msg = state.add_message(role, "x" * 2_000)  # ~500 tokens each

        # Ingest every other turn
        if i % 2 == 1:
            mgr.record_checkpoint(
                turn_number=msg.turn_number,
                entry_count=1,
                token_count=msg.token_count,
            )

    # 10 messages × ~500 tokens = ~5000 tokens, budget 3000
    active = mgr.get_active_messages(state)

    # Should have pruned some messages
    assert len(active) < 10
    # Should still have at least min_active_turns
    assert len(active) >= 2
    # Bridge should exist
    assert mgr.bridge_summary != ""


# ─── CLI Command Test ────────────────────────────────────────────────────────

def test_cli_window_command():
    """Test that the window CLI command works."""
    import subprocess
    result = subprocess.run(
        ["python", "librarian_cli.py", "window"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "active_messages" in data
    assert "budget_remaining" in data


import os
import json
