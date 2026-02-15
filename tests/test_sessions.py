"""
Tests for Phase 4: Cross-Session Persistence & Shared DB Support.
Validates session lifecycle, message persistence, cross-session search
with current-session boosting, session resume, and shared DB reads.
"""
import os
import sys
import uuid
import tempfile
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.types import (
    RolodexEntry, ContentModality, EntryCategory, Tier,
    Message, MessageRole, SessionInfo, ConversationState
)
from src.storage.rolodex import Rolodex
from src.storage.session_manager import SessionManager


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


# ─── Session Lifecycle ────────────────────────────────────────────────────

def test_session_lifecycle():
    """Start, update activity, and end a session — verify timestamps and status."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        rolodex = Rolodex(db_path)
        sm = SessionManager(rolodex.conn)
        conv_id = str(uuid.uuid4())

        # Start
        info = sm.start_session(conv_id)
        assert info.session_id == conv_id
        assert info.status == "active"

        # Check DB
        retrieved = sm.get_session(conv_id)
        assert retrieved is not None
        assert retrieved.status == "active"

        # Update activity
        sm.update_session_activity(conv_id)
        retrieved = sm.get_session(conv_id)
        assert retrieved.last_active is not None

        # End
        sm.end_session(conv_id, summary="Test session completed")
        retrieved = sm.get_session(conv_id)
        assert retrieved.status == "ended"
        assert retrieved.ended_at is not None
        assert retrieved.summary == "Test session completed"

        rolodex.close()
        print("  PASS: session_lifecycle")
    finally:
        os.unlink(db_path)


def test_message_persistence():
    """Save and reload messages — content, role, and turn preserved."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        rolodex = Rolodex(db_path)
        sm = SessionManager(rolodex.conn)
        conv_id = str(uuid.uuid4())
        sm.start_session(conv_id)

        # Save messages
        msg1 = Message(
            role=MessageRole.USER,
            content="What is the capital of France?",
            turn_number=1,
        )
        msg2 = Message(
            role=MessageRole.ASSISTANT,
            content="The capital of France is Paris.",
            turn_number=2,
        )
        sm.save_message(conv_id, msg1)
        sm.save_message(conv_id, msg2)

        # Reload
        loaded = sm.load_messages(conv_id)
        assert len(loaded) == 2, f"Expected 2 messages, got {len(loaded)}"
        assert loaded[0].role == MessageRole.USER
        assert loaded[0].content == "What is the capital of France?"
        assert loaded[0].turn_number == 1
        assert loaded[1].role == MessageRole.ASSISTANT
        assert loaded[1].content == "The capital of France is Paris."
        assert loaded[1].turn_number == 2

        rolodex.close()
        print("  PASS: message_persistence")
    finally:
        os.unlink(db_path)


def test_list_sessions():
    """List sessions returns them ordered by recency, respects limit."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        rolodex = Rolodex(db_path)
        sm = SessionManager(rolodex.conn)

        # Create 3 sessions
        ids = []
        for i in range(3):
            cid = str(uuid.uuid4())
            sm.start_session(cid)
            # Save a message to make each session real
            msg = Message(
                role=MessageRole.USER,
                content=f"Message in session {i}",
                turn_number=1,
            )
            sm.save_message(cid, msg)
            sm.update_session_activity(cid)
            ids.append(cid)

        # List all
        sessions = sm.list_sessions(limit=10)
        assert len(sessions) == 3, f"Expected 3 sessions, got {len(sessions)}"
        # Most recent should be first
        assert sessions[0].session_id == ids[2]

        # List with limit
        limited = sm.list_sessions(limit=2)
        assert len(limited) == 2

        rolodex.close()
        print("  PASS: list_sessions")
    finally:
        os.unlink(db_path)


# ─── Cross-Session Search ─────────────────────────────────────────────────

def test_cross_session_search():
    """Finds entries from OTHER sessions when conversation_id=None."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        rolodex = Rolodex(db_path)
        session_a = "session-aaa"
        session_b = "session-bbb"

        # Create entries in two different sessions
        e1 = make_entry(
            "Python async patterns are powerful",
            tags=["python"], conversation_id=session_a,
        )
        e2 = make_entry(
            "Rust borrow checker prevents memory errors",
            tags=["rust"], conversation_id=session_b,
        )
        rolodex.batch_create_entries([e1, e2])

        # Search scoped to session_a — should only find Python entry
        scoped = rolodex.keyword_search("patterns", conversation_id=session_a)
        scoped_ids = {e.id for e, _ in scoped}
        assert e1.id in scoped_ids, "Should find Python entry in session_a"
        assert e2.id not in scoped_ids, "Should NOT find Rust entry in session_a scope"

        # Search globally (conversation_id=None) — should find both
        global_py = rolodex.keyword_search("patterns", conversation_id=None)
        assert len(global_py) >= 1, "Global search should find entries"

        global_rust = rolodex.keyword_search("borrow", conversation_id=None)
        global_rust_ids = {e.id for e, _ in global_rust}
        assert e2.id in global_rust_ids, "Global search should find Rust entry"

        rolodex.close()
        print("  PASS: cross_session_search")
    finally:
        os.unlink(db_path)


def test_session_boost():
    """Current-session entries score higher than cross-session entries."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        rolodex = Rolodex(db_path)
        current = "current-session"
        old = "old-session"

        # Create similar entries in different sessions
        e_current = make_entry(
            "Fibonacci sequence in Python implementation",
            tags=["fibonacci", "python"],
            conversation_id=current,
            embedding=[0.9, 0.1, 0.0],
        )
        e_old = make_entry(
            "Fibonacci numbers and golden ratio theory",
            tags=["fibonacci", "math"],
            conversation_id=old,
            embedding=[0.85, 0.15, 0.05],
        )
        rolodex.batch_create_entries([e_current, e_old])

        # Boosted search with current session prioritized
        results = rolodex.boosted_hybrid_search(
            query="Fibonacci",
            query_embedding=[0.87, 0.13, 0.02],
            current_session_id=current,
            boost_factor=1.5,
            limit=5,
        )

        assert len(results) >= 2, f"Expected >=2 results, got {len(results)}"

        # Current session entry should be ranked first (boosted)
        top_entry, top_score = results[0]
        assert top_entry.conversation_id == current, (
            f"Expected current-session entry first, got {top_entry.conversation_id}"
        )

        # Verify boost was applied — current session score should be
        # higher than unboosted score of old session entry
        old_entry, old_score = results[1]
        assert top_score > old_score, (
            f"Boosted score {top_score} should > old score {old_score}"
        )

        rolodex.close()
        print("  PASS: session_boost")
    finally:
        os.unlink(db_path)


def test_session_resume():
    """ConversationState rebuilt correctly from stored messages."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        rolodex = Rolodex(db_path)
        sm = SessionManager(rolodex.conn)
        conv_id = str(uuid.uuid4())
        sm.start_session(conv_id)

        # Simulate a 3-turn conversation
        messages = [
            Message(role=MessageRole.USER, content="Hello", turn_number=1),
            Message(role=MessageRole.ASSISTANT, content="Hi there!", turn_number=2),
            Message(role=MessageRole.USER, content="What is 2+2?", turn_number=3),
        ]
        for msg in messages:
            sm.save_message(conv_id, msg)

        # End the session
        sm.end_session(conv_id)

        # Rebuild ConversationState (simulating what orchestrator.resume does)
        loaded = sm.load_messages(conv_id)
        state = ConversationState(conversation_id=conv_id)
        for msg in loaded:
            state.messages.append(msg)
            state.total_tokens += msg.token_count
            state.turn_count = max(state.turn_count, msg.turn_number)

        assert len(state.messages) == 3
        assert state.turn_count == 3
        assert state.total_tokens > 0
        assert state.messages[0].content == "Hello"
        assert state.messages[2].content == "What is 2+2?"

        rolodex.close()
        print("  PASS: session_resume")
    finally:
        os.unlink(db_path)


def test_shared_db_concurrent_read():
    """Two Rolodex instances can read the same DB file simultaneously."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        # Instance 1: create entries
        rolodex1 = Rolodex(db_path)
        entry = make_entry("Shared knowledge across instances",
                           tags=["shared"], conversation_id="session-1")
        eid = rolodex1.create_entry(entry)

        # Instance 2: open same DB, read the entry
        rolodex2 = Rolodex(db_path)
        retrieved = rolodex2.get_entry(eid)
        assert retrieved is not None, "Instance 2 should read entry from shared DB"
        assert retrieved.content == "Shared knowledge across instances"

        # Instance 2: add its own entry
        e2 = make_entry("Knowledge from instance 2",
                        tags=["instance2"], conversation_id="session-2")
        e2_id = rolodex2.create_entry(e2)

        # Instance 1: should see instance 2's entry
        retrieved2 = rolodex1.get_entry(e2_id)
        assert retrieved2 is not None, "Instance 1 should see instance 2's entry"
        assert retrieved2.content == "Knowledge from instance 2"

        rolodex1.close()
        rolodex2.close()
        print("  PASS: shared_db_concurrent_read")
    finally:
        os.unlink(db_path)


def test_session_entry_count():
    """Entry count tracks correctly per session."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        rolodex = Rolodex(db_path)
        sm = SessionManager(rolodex.conn)

        session_a = "session-count-a"
        session_b = "session-count-b"
        sm.start_session(session_a)
        sm.start_session(session_b)

        # Create entries in session A
        for i in range(3):
            e = make_entry(f"Entry {i} for session A", conversation_id=session_a)
            rolodex.create_entry(e)

        # Create entries in session B
        for i in range(5):
            e = make_entry(f"Entry {i} for session B", conversation_id=session_b)
            rolodex.create_entry(e)

        # Check counts
        count_a = sm._count_entries(session_a)
        count_b = sm._count_entries(session_b)
        assert count_a == 3, f"Expected 3 entries in A, got {count_a}"
        assert count_b == 5, f"Expected 5 entries in B, got {count_b}"

        # End session and check final metadata
        sm.end_session(session_a)
        info_a = sm.get_session(session_a)
        assert info_a.entry_count == 3, f"Expected 3, got {info_a.entry_count}"

        rolodex.close()
        print("  PASS: session_entry_count")
    finally:
        os.unlink(db_path)


if __name__ == "__main__":
    print("Running Session tests (Phase 4)...\n")
    test_session_lifecycle()
    test_message_persistence()
    test_list_sessions()
    test_cross_session_search()
    test_session_boost()
    test_session_resume()
    test_shared_db_concurrent_read()
    test_session_entry_count()
    print("\nAll Phase 4 tests passed!")
