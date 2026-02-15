"""
Tests for TheLibrarian middleware (Phase 4.5)

Validates the public API works completely without any LLM/API dependency.
All tests run in verbatim mode (no adapter).
"""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.librarian import TheLibrarian
from src.utils.config import LibrarianConfig


def _make_librarian(db_path=None):
    """Create a TheLibrarian in verbatim mode (no LLM adapter)."""
    if db_path is None:
        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = f.name
        f.close()
    config = LibrarianConfig()
    config.embedding_strategy = "hash"
    config.librarian_activation_tokens = 0  # Activate immediately
    return TheLibrarian(db_path=db_path, config=config), db_path


# ─── Test: Ingest without LLM ─────────────────────────────────────────────

def test_ingest_without_llm():
    """Ingest messages in verbatim mode — no API key, no adapter."""
    lib, db_path = _make_librarian()
    try:
        entries = asyncio.run(lib.ingest(
            "user",
            "Python decorators are a powerful metaprogramming feature that wraps functions",
        ))
        assert len(entries) >= 1, f"Expected at least 1 entry, got {len(entries)}"
        assert any("decorator" in e.content.lower() for e in entries), \
            "Expected entry content to contain 'decorator'"
        print("  PASS: ingest_without_llm")
    finally:
        asyncio.run(lib.shutdown())
        os.unlink(db_path)


# ─── Test: Retrieve after ingest ──────────────────────────────────────────

def test_retrieve_after_ingest():
    """Ingest a message, then retrieve it by keyword search."""
    lib, db_path = _make_librarian()
    try:
        asyncio.run(lib.ingest(
            "user",
            "The Fibonacci sequence starts with 0 and 1, each subsequent number is the sum",
        ))
        response = asyncio.run(lib.retrieve("Fibonacci"))
        assert response.found, "Expected to find the ingested entry"
        assert len(response.entries) >= 1
        assert any("fibonacci" in e.content.lower() for e in response.entries)
        print("  PASS: retrieve_after_ingest")
    finally:
        asyncio.run(lib.shutdown())
        os.unlink(db_path)


# ─── Test: Context block formatting ──────────────────────────────────────

def test_context_block_formatting():
    """get_context_block returns a formatted string for prompt injection."""
    lib, db_path = _make_librarian()
    try:
        asyncio.run(lib.ingest(
            "user",
            "The capital of France is Paris and it has the Eiffel Tower landmark",
        ))
        response = asyncio.run(lib.retrieve("France capital"))
        if response.found:
            block = lib.get_context_block(response)
            assert isinstance(block, str)
            assert len(block) > 10, "Expected non-trivial context block"
        print("  PASS: context_block_formatting")
    finally:
        asyncio.run(lib.shutdown())
        os.unlink(db_path)


# ─── Test: Session lifecycle ──────────────────────────────────────────────

def test_session_lifecycle_middleware():
    """Start, use, and end a session via TheLibrarian."""
    lib, db_path = _make_librarian()
    try:
        session_id = lib.session_id
        assert session_id is not None and len(session_id) > 0

        asyncio.run(lib.ingest("user", "Hello, this is a test message for session tracking"))
        asyncio.run(lib.ingest("assistant", "Got it, I understand this is a test"))

        sessions = lib.list_sessions()
        assert len(sessions) >= 1, "Expected at least one session"

        # Current session should be in the list
        session_ids = [s.session_id for s in sessions]
        assert session_id in session_ids

        lib.end_session()
        print("  PASS: session_lifecycle_middleware")
    finally:
        asyncio.run(lib.shutdown())
        os.unlink(db_path)


# ─── Test: Stats ──────────────────────────────────────────────────────────

def test_stats():
    """get_stats works without LLM adapter."""
    lib, db_path = _make_librarian()
    try:
        asyncio.run(lib.ingest("user", "Testing stats with some meaningful content about databases"))
        stats = lib.get_stats()
        assert "conversation_id" in stats
        assert "total_messages" in stats
        assert stats["total_messages"] >= 1
        assert stats["llm_adapter"] is False  # No adapter
        print("  PASS: stats")
    finally:
        asyncio.run(lib.shutdown())
        os.unlink(db_path)


# ─── Test: Multiple sessions ─────────────────────────────────────────────

def test_cross_session_middleware():
    """Entries from one session are findable from another."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = f.name
    f.close()
    try:
        # Session 1
        lib1, _ = _make_librarian(db_path)
        asyncio.run(lib1.ingest(
            "user",
            "Quantum entanglement is a phenomenon where particles become correlated",
        ))
        lib1.end_session()
        asyncio.run(lib1.shutdown())

        # Session 2 (same DB, new session)
        lib2, _ = _make_librarian(db_path)
        response = asyncio.run(lib2.retrieve("quantum entanglement"))
        assert response.found, "Expected to find entry from previous session"
        asyncio.run(lib2.shutdown())

        print("  PASS: cross_session_middleware")
    finally:
        os.unlink(db_path)


# ─── Test: Config validates without API key ───────────────────────────────

def test_config_no_api_key():
    """Config.validate() returns warnings (not errors) when API key missing."""
    config = LibrarianConfig()
    warnings = config.validate()
    # Should get a warning, not crash
    assert len(warnings) >= 1
    assert "verbatim" in warnings[0].lower()
    assert config.has_api_key is False
    print("  PASS: config_no_api_key")


# ─── Test: Maintenance runs without LLM ──────────────────────────────────

def test_maintenance():
    """Tier sweep works in verbatim mode."""
    lib, db_path = _make_librarian()
    try:
        asyncio.run(lib.ingest("user", "Important data that should be stored and managed properly"))
        result = lib.run_maintenance()
        assert isinstance(result, dict)
        assert "entries_scanned" in result
        print("  PASS: maintenance")
    finally:
        asyncio.run(lib.shutdown())
        os.unlink(db_path)


# ─── Run all ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Middleware tests (Phase 4.5)...\n")
    test_ingest_without_llm()
    test_retrieve_after_ingest()
    test_context_block_formatting()
    test_session_lifecycle_middleware()
    test_stats()
    test_cross_session_middleware()
    test_config_no_api_key()
    test_maintenance()
    print("\nAll Middleware tests passed!")
