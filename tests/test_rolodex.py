"""
Tests for the Rolodex storage layer.
Validates CRUD, keyword search, semantic search, hot cache,
and Phase 2 tier management.
"""
import os
import sys
import tempfile
import uuid
from datetime import datetime, timedelta
# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.core.types import (
    RolodexEntry, ContentModality, EntryCategory, Tier,
    compute_importance_score
)
from src.storage.rolodex import Rolodex
from src.storage.schema import serialize_embedding, deserialize_embedding


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


def test_create_and_retrieve():
    """Can we store and retrieve an entry?"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    rolodex = Rolodex(db_path)
    try:
        entry = make_entry("The Fibonacci sequence starts with 0, 1, 1, 2, 3, 5...")
        entry_id = rolodex.create_entry(entry)
        retrieved = rolodex.get_entry(entry_id)
        assert retrieved is not None
        assert retrieved.content == entry.content
        assert retrieved.id == entry.id
        print("  PASS: create_and_retrieve")
    finally:
        _safe_cleanup(rolodex, db_path)


def test_batch_create():
    """Can we bulk insert entries?"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    rolodex = Rolodex(db_path)
    try:
        entries = [
            make_entry("Python is a programming language", tags=["python"]),
            make_entry("Rust is a systems language", tags=["rust"]),
            make_entry("JavaScript runs in browsers", tags=["javascript"]),
        ]
        ids = rolodex.batch_create_entries(entries)
        assert len(ids) == 3
        stats = rolodex.get_stats()
        assert stats["total_entries"] == 3
        print("  PASS: batch_create")
    finally:
        _safe_cleanup(rolodex, db_path)


def test_keyword_search():
    """Does FTS keyword search work?"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    rolodex = Rolodex(db_path)
    try:
        entries = [
            make_entry("The Fibonacci function computes numbers recursively"),
            make_entry("Quick sort is an efficient sorting algorithm"),
            make_entry("Fibonacci numbers appear in nature and mathematics"),
        ]
        rolodex.batch_create_entries(entries)
        results = rolodex.keyword_search("Fibonacci")
        assert len(results) >= 2
        # Both Fibonacci entries should be found
        contents = [e.content for e, _ in results]
        assert any("Fibonacci function" in c for c in contents)
        assert any("Fibonacci numbers" in c for c in contents)
        print("  PASS: keyword_search")
    finally:
        _safe_cleanup(rolodex, db_path)


def test_semantic_search():
    """Does vector similarity search work?"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    rolodex = Rolodex(db_path)
    try:
        # Create entries with simple fake embeddings
        # Entry about recursion (embedding points "north")
        e1 = make_entry("Recursion is when a function calls itself",
                        embedding=[0.9, 0.1, 0.0])
        # Entry about sorting (embedding points "east")
        e2 = make_entry("Bubble sort compares adjacent elements",
                        embedding=[0.1, 0.9, 0.0])
        # Entry about recursion variant (also points "north-ish")
        e3 = make_entry("Tail recursion optimization avoids stack overflow",
                        embedding=[0.8, 0.2, 0.1])
        rolodex.batch_create_entries([e1, e2, e3])
        # Query embedding similar to "recursion" entries
        query_emb = [0.85, 0.15, 0.05]
        results = rolodex.semantic_search(query_emb, limit=2)
        assert len(results) >= 1
        # The recursion entries should rank higher than sorting
        top_content = results[0][0].content
        assert "recursion" in top_content.lower() or "Recursion" in top_content
        print("  PASS: semantic_search")
    finally:
        _safe_cleanup(rolodex, db_path)


def test_hybrid_search():
    """Does combined keyword + semantic search work?"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    rolodex = Rolodex(db_path)
    try:
        entries = [
            make_entry("Python list comprehensions are concise",
                       tags=["python", "lists"], embedding=[0.8, 0.2, 0.0]),
            make_entry("JavaScript array map function transforms elements",
                       tags=["javascript", "arrays"], embedding=[0.3, 0.7, 0.1]),
            make_entry("Python dictionary comprehensions create dicts",
                       tags=["python", "dicts"], embedding=[0.75, 0.25, 0.05]),
        ]
        rolodex.batch_create_entries(entries)
        # Hybrid search for "Python" with embedding similar to Python entries
        results = rolodex.hybrid_search(
            query="Python",
            query_embedding=[0.78, 0.22, 0.02],
            limit=3
        )
        assert len(results) >= 2
        # Python entries should rank higher
        top_content = results[0][0].content
        assert "Python" in top_content
        print("  PASS: hybrid_search")
    finally:
        _safe_cleanup(rolodex, db_path)


def test_hot_cache():
    """Does the hot cache work with LRU eviction?"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    rolodex = Rolodex(db_path)
    try:
        rolodex._hot_cache_max = 3  # Small cache for testing
        entries = [
            make_entry(f"Entry number {i}") for i in range(5)
        ]
        ids = rolodex.batch_create_entries(entries)
        # Only last 3 should be in hot cache
        assert len(rolodex._hot_cache) == 3
        assert ids[0] not in rolodex._hot_cache
        assert ids[1] not in rolodex._hot_cache
        assert ids[4] in rolodex._hot_cache
        # But all 5 should be retrievable from cold storage
        for eid in ids:
            entry = rolodex.get_entry(eid)
            assert entry is not None
        print("  PASS: hot_cache")
    finally:
        _safe_cleanup(rolodex, db_path)


def test_access_tracking():
    """Does access count increment?"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    rolodex = Rolodex(db_path)
    try:
        entry = make_entry("Important fact about quantum computing")
        entry_id = rolodex.create_entry(entry)
        # Access it 3 times
        for _ in range(3):
            rolodex.update_access(entry_id)
        retrieved = rolodex.get_entry(entry_id)
        # Check cold storage (bypass cache)
        row = rolodex.conn.execute(
            "SELECT access_count FROM rolodex_entries WHERE id = ?",
            (entry_id,)
        ).fetchone()
        assert row["access_count"] == 3
        print("  PASS: access_tracking")
    finally:
        _safe_cleanup(rolodex, db_path)


def test_embedding_serialization():
    """Do embeddings survive serialization round-trip?"""
    original = [0.123, -0.456, 0.789, 0.0, -1.0]
    blob = serialize_embedding(original)
    restored = deserialize_embedding(blob)
    for a, b in zip(original, restored):
        assert abs(a - b) < 1e-6
    print("  PASS: embedding_serialization")


def test_stats():
    """Do stats report correctly?"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    rolodex = Rolodex(db_path)
    try:
        entries = [
            make_entry("Def one", category=EntryCategory.DEFINITION),
            make_entry("Def two", category=EntryCategory.DEFINITION),
            make_entry("Example one", category=EntryCategory.EXAMPLE),
        ]
        rolodex.batch_create_entries(entries)
        stats = rolodex.get_stats()
        assert stats["total_entries"] == 3
        assert stats["categories"]["definition"] == 2
        assert stats["categories"]["example"] == 1
        print("  PASS: stats")
    finally:
        _safe_cleanup(rolodex, db_path)


# ─── Phase 2: Tier Management Tests ──────────────────────────────────────────

def test_importance_score_calculation():
    """Does the importance score formula produce correct relative values?"""
    now = datetime.utcnow()
    # Entry with high access, recently accessed
    hot_entry = make_entry("Frequently used fact")
    hot_entry.access_count = 10
    hot_entry.last_accessed = now - timedelta(minutes=5)
    hot_entry.created_at = now - timedelta(hours=2)
    # Entry with no access
    cold_entry = make_entry("Never accessed fact")
    cold_entry.access_count = 0
    cold_entry.last_accessed = None
    cold_entry.created_at = now - timedelta(hours=2)
    # Entry with old access (stale)
    stale_entry = make_entry("Stale fact")
    stale_entry.access_count = 5
    stale_entry.last_accessed = now - timedelta(hours=72)
    stale_entry.created_at = now - timedelta(hours=100)
    hot_score = compute_importance_score(hot_entry, now=now)
    cold_score = compute_importance_score(cold_entry, now=now)
    stale_score = compute_importance_score(stale_entry, now=now)
    # Hot should be highest
    assert hot_score > stale_score, f"hot {hot_score} should > stale {stale_score}"
    assert hot_score > cold_score, f"hot {hot_score} should > cold {cold_score}"
    # Cold (never accessed) should have score near 0 (log(1+0) * 0.1 * ...)
    assert cold_score < 0.5, f"cold score {cold_score} should be low"
    # Stale should be between (has accesses but old)
    assert stale_score > cold_score, f"stale {stale_score} should > cold {cold_score}"
    print("  PASS: importance_score_calculation")


def test_promotion_on_high_access():
    """Does an entry get promoted to HOT after enough accesses?"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    rolodex = Rolodex(db_path)
    try:
        entry = make_entry("Important reusable concept")
        entry_id = rolodex.create_entry(entry)
        # Access it enough times to build up the score
        for _ in range(8):
            rolodex.update_access(entry_id)
        # Evaluate — should recommend HOT
        recommended_tier, score = rolodex.evaluate_tier(
            entry_id, promotion_threshold=1.0
        )
        assert recommended_tier == Tier.HOT, f"Expected HOT, got {recommended_tier} (score={score})"
        # Actually promote
        event = rolodex.promote_entry(entry_id)
        assert event is not None
        assert event.new_tier == Tier.HOT
        # Verify DB and cache
        row = rolodex.conn.execute(
            "SELECT tier FROM rolodex_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        assert row["tier"] == "hot"
        assert entry_id in rolodex._hot_cache
        print("  PASS: promotion_on_high_access")
    finally:
        _safe_cleanup(rolodex, db_path)


def test_demotion_on_decay():
    """Does a HOT entry get demoted when its score decays?"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    rolodex = Rolodex(db_path)
    try:
        entry = make_entry("Once-popular fact")
        entry.access_count = 3
        entry.last_accessed = datetime.utcnow() - timedelta(hours=200)
        entry.created_at = datetime.utcnow() - timedelta(hours=300)
        entry_id = rolodex.create_entry(entry)
        # Manually set tier to HOT in DB
        rolodex.conn.execute(
            "UPDATE rolodex_entries SET tier = 'hot' WHERE id = ?", (entry_id,)
        )
        rolodex.conn.commit()
        # Evaluate — old last_accessed should decay the score below demotion
        recommended_tier, score = rolodex.evaluate_tier(
            entry_id, demotion_threshold=0.3
        )
        assert recommended_tier == Tier.COLD, f"Expected COLD, got {recommended_tier} (score={score})"
        print("  PASS: demotion_on_decay")
    finally:
        _safe_cleanup(rolodex, db_path)


def test_preload_hot_entries():
    """Do HOT entries get loaded into cache on startup?"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        # First instance: create entries and promote some
        rolodex1 = Rolodex(db_path)
        e1 = make_entry("Hot entry one")
        e2 = make_entry("Hot entry two")
        e3 = make_entry("Cold entry")
        id1 = rolodex1.create_entry(e1)
        id2 = rolodex1.create_entry(e2)
        id3 = rolodex1.create_entry(e3)
        rolodex1.conn.execute("UPDATE rolodex_entries SET tier = 'hot' WHERE id = ?", (id1,))
        rolodex1.conn.execute("UPDATE rolodex_entries SET tier = 'hot' WHERE id = ?", (id2,))
        rolodex1.conn.commit()
        rolodex1.close()
        # Second instance: simulate restart
        rolodex2 = Rolodex(db_path)
        assert len(rolodex2._hot_cache) == 0  # Cache starts empty
        loaded = rolodex2.preload_hot_entries()
        assert loaded == 2, f"Expected 2 preloaded, got {loaded}"
        assert id1 in rolodex2._hot_cache
        assert id2 in rolodex2._hot_cache
        assert id3 not in rolodex2._hot_cache
        rolodex2.close()
        print("  PASS: preload_hot_entries")
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


def test_tier_sweep():
    """Does a tier sweep correctly promote and demote entries?"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    rolodex = Rolodex(db_path)
    try:
        now = datetime.utcnow()
        # Entry that should be promoted (high access, recent)
        promote_me = make_entry("Heavily used fact")
        promote_me.access_count = 12
        promote_me.last_accessed = now - timedelta(minutes=1)
        promote_me.created_at = now - timedelta(hours=1)
        pid = rolodex.create_entry(promote_me)
        # Simulate the access count in DB too
        rolodex.conn.execute(
            "UPDATE rolodex_entries SET access_count = 12, last_accessed = ? WHERE id = ?",
            ((now - timedelta(minutes=1)).isoformat(), pid)
        )
        # Entry that should be demoted (was HOT but stale)
        demote_me = make_entry("Forgotten fact")
        demote_me.access_count = 2
        demote_me.last_accessed = now - timedelta(hours=200)
        demote_me.created_at = now - timedelta(hours=300)
        did = rolodex.create_entry(demote_me)
        # Promote it first (via API), then make it stale via DB
        rolodex.promote_entry(did)
        rolodex.conn.execute(
            "UPDATE rolodex_entries SET access_count = 2, last_accessed = ? WHERE id = ?",
            ((now - timedelta(hours=200)).isoformat(), did)
        )
        rolodex.conn.commit()
        # Clear from cache so sweep reads fresh DB state
        if did in rolodex._hot_cache:
            del rolodex._hot_cache[did]
        # Run sweep
        result = rolodex.run_tier_sweep(
            promotion_threshold=1.0,
            demotion_threshold=0.3,
        )
        assert result["promoted"] >= 1, f"Expected at least 1 promotion, got {result['promoted']}"
        assert result["demoted"] >= 1, f"Expected at least 1 demotion, got {result['demoted']}"
        # Verify in DB
        hot_row = rolodex.conn.execute(
            "SELECT tier FROM rolodex_entries WHERE id = ?", (pid,)
        ).fetchone()
        cold_row = rolodex.conn.execute(
            "SELECT tier FROM rolodex_entries WHERE id = ?", (did,)
        ).fetchone()
        assert hot_row["tier"] == "hot", f"Promoted entry should be HOT"
        assert cold_row["tier"] == "cold", f"Demoted entry should be COLD"
        print("  PASS: tier_sweep")
    finally:
        _safe_cleanup(rolodex, db_path)


def test_update_access_refreshes_lru():
    """Does update_access refresh the entry's LRU position?"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    rolodex = Rolodex(db_path)
    try:
        rolodex._hot_cache_max = 3
        # Create 3 entries (fills cache)
        entries = [make_entry(f"Entry {i}") for i in range(3)]
        ids = rolodex.batch_create_entries(entries)
        # ids[0] is now the LRU (oldest). Access it to refresh.
        rolodex.update_access(ids[0])
        # Now add a 4th entry — should evict ids[1] (new LRU), not ids[0]
        e4 = make_entry("Entry 3")
        id4 = rolodex.create_entry(e4)
        assert ids[0] in rolodex._hot_cache, "Refreshed entry should survive eviction"
        assert ids[1] not in rolodex._hot_cache, "Un-refreshed entry should be evicted"
        assert id4 in rolodex._hot_cache, "New entry should be in cache"
        print("  PASS: update_access_refreshes_lru")
    finally:
        _safe_cleanup(rolodex, db_path)


if __name__ == "__main__":
    print("Running Rolodex tests...\n")
    test_create_and_retrieve()
    test_batch_create()
    test_keyword_search()
    test_semantic_search()
    test_hybrid_search()
    test_hot_cache()
    test_access_tracking()
    test_embedding_serialization()
    test_stats()
    print("\n--- Phase 2: Tier Management ---\n")
    test_importance_score_calculation()
    test_promotion_on_high_access()
    test_demotion_on_decay()
    test_preload_hot_entries()
    test_tier_sweep()
    test_update_access_refreshes_lru()
    print("\nAll tests passed!")
