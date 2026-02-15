"""
Tests for Phase 10: Manifest-Based Boot System.

Validates the full manifest lifecycle: super boot (census + rank + pack),
incremental update (delta scoring), session-close refinement (behavioral
signal), invalidation (force-fresh), and recall tracking.
"""
import os
import sys
import uuid
import json
import tempfile
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.types import (
    RolodexEntry, ContentModality, EntryCategory, Tier,
    ManifestEntry, ManifestState, estimate_tokens,
)
from src.storage.rolodex import Rolodex
from src.storage.manifest_manager import ManifestManager


# ─── Helpers ──────────────────────────────────────────────────────────────

def make_entry(
    content: str,
    tags: list = None,
    category: EntryCategory = EntryCategory.NOTE,
    conversation_id: str = "test-conv",
    access_count: int = 0,
    last_accessed: datetime = None,
    created_at: datetime = None,
    topic_id: str = None,
) -> RolodexEntry:
    """Helper to create test entries."""
    entry = RolodexEntry(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        content=content,
        content_type=ContentModality.PROSE,
        category=category,
        tags=tags or [],
        source_range={"turn_start": 1, "turn_end": 1},
        access_count=access_count,
        last_accessed=last_accessed,
        created_at=created_at or datetime.utcnow(),
    )
    return entry


def make_rolodex():
    """Create a temporary Rolodex for testing."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = f.name
    f.close()
    rolodex = Rolodex(db_path)
    return rolodex, db_path


def create_topic(conn, label: str, entry_count: int = 0) -> str:
    """Create a topic in the DB and return its ID."""
    topic_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO topics (id, label, description, created_at, entry_count)
           VALUES (?, ?, '', ?, ?)""",
        (topic_id, label, datetime.utcnow().isoformat(), entry_count)
    )
    conn.commit()
    return topic_id


def assign_topic(conn, entry_id: str, topic_id: str):
    """Assign an entry to a topic."""
    conn.execute(
        "UPDATE rolodex_entries SET topic_id = ? WHERE id = ?",
        (topic_id, entry_id)
    )
    conn.commit()


def log_query(conn, session_id: str, query: str, found: bool, entry_ids: list):
    """Log a query to query_log."""
    conn.execute(
        """INSERT INTO query_log (id, conversation_id, query_text, found, entry_ids, search_time_ms, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (str(uuid.uuid4()), session_id, query, found, json.dumps(entry_ids), 10.0, datetime.utcnow().isoformat())
    )
    conn.commit()


# ─── Tests ────────────────────────────────────────────────────────────────

def test_get_latest_manifest_empty_db():
    """No manifest exists → returns None gracefully."""
    rolodex, db_path = make_rolodex()
    try:
        mm = ManifestManager(rolodex.conn, rolodex)
        result = mm.get_latest_manifest()
        assert result is None, "Expected None when no manifest exists"
    finally:
        rolodex.close()
        os.unlink(db_path)


def test_super_manifest_empty_db():
    """Super boot on empty DB → manifest with 0 entries."""
    rolodex, db_path = make_rolodex()
    try:
        mm = ManifestManager(rolodex.conn, rolodex)
        manifest = mm.build_super_manifest(available_budget=5000)
        assert manifest is not None
        assert manifest.manifest_type == "super"
        assert len(manifest.entries) == 0
        assert manifest.total_token_cost == 0
    finally:
        rolodex.close()
        os.unlink(db_path)


def test_super_manifest_respects_token_budget():
    """Entries pack within budget — never exceed."""
    rolodex, db_path = make_rolodex()
    try:
        mm = ManifestManager(rolodex.conn, rolodex)

        # Create entries with known token costs
        now = datetime.utcnow()
        for i in range(20):
            entry = make_entry(
                content=f"Entry number {i} with some meaningful content about topic {i % 5}. " * 10,
                access_count=10 - (i % 10),
                last_accessed=now - timedelta(hours=i),
                created_at=now - timedelta(days=i),
            )
            rolodex.create_entry(entry)

        # Small budget that can't fit all entries
        budget = 500
        manifest = mm.build_super_manifest(available_budget=budget)

        # Verify budget respected
        actual_cost = sum(me.token_cost for me in manifest.entries)
        assert actual_cost <= budget, f"Token cost {actual_cost} exceeds budget {budget}"
        assert len(manifest.entries) > 0, "Should have selected some entries"
        assert manifest.manifest_type == "super"
    finally:
        rolodex.close()
        os.unlink(db_path)


def test_super_manifest_topic_breadth():
    """At least 1 entry per top topic when budget allows."""
    rolodex, db_path = make_rolodex()
    try:
        mm = ManifestManager(rolodex.conn, rolodex)
        now = datetime.utcnow()

        # Create 3 topics
        topic_ids = {}
        for label in ["authentication", "database", "frontend"]:
            topic_ids[label] = create_topic(rolodex.conn, label, entry_count=5)

        # Create 5 entries per topic, with varying access counts
        for label, tid in topic_ids.items():
            for i in range(5):
                entry = make_entry(
                    content=f"Content about {label} topic, variation {i}. " * 5,
                    access_count=5 + i,
                    last_accessed=now - timedelta(hours=i),
                    created_at=now - timedelta(days=i),
                )
                rolodex.create_entry(entry)
                assign_topic(rolodex.conn, entry.id, tid)

        # Generous budget
        manifest = mm.build_super_manifest(available_budget=10000)

        # Check we got at least one entry per topic
        topics_in_manifest = {me.topic_label for me in manifest.entries if me.topic_label}
        assert "authentication" in topics_in_manifest, "Missing authentication topic"
        assert "database" in topics_in_manifest, "Missing database topic"
        assert "frontend" in topics_in_manifest, "Missing frontend topic"

        # Check at least some are marked as topic_rep
        topic_reps = [me for me in manifest.entries if me.selection_reason == "topic_rep"]
        assert len(topic_reps) >= 3, f"Expected 3 topic_rep entries, got {len(topic_reps)}"
    finally:
        rolodex.close()
        os.unlink(db_path)


def test_super_manifest_excludes_user_knowledge():
    """user_knowledge entries are excluded (they're loaded separately)."""
    rolodex, db_path = make_rolodex()
    try:
        mm = ManifestManager(rolodex.conn, rolodex)
        now = datetime.utcnow()

        # Create a user_knowledge entry
        uk_entry = make_entry(
            content="User prefers dark mode",
            category=EntryCategory.USER_KNOWLEDGE,
            access_count=100,
            last_accessed=now,
        )
        rolodex.create_entry(uk_entry)

        # Create a regular entry
        regular = make_entry(
            content="Regular note about the project",
            access_count=5,
            last_accessed=now,
        )
        rolodex.create_entry(regular)

        manifest = mm.build_super_manifest(available_budget=5000)

        manifest_ids = {me.entry_id for me in manifest.entries}
        assert uk_entry.id not in manifest_ids, "user_knowledge should be excluded"
        assert regular.id in manifest_ids, "Regular entry should be included"
    finally:
        rolodex.close()
        os.unlink(db_path)


def test_incremental_manifest_promotes_new_high_scorers():
    """New entries with high scores evict weak existing entries."""
    rolodex, db_path = make_rolodex()
    try:
        mm = ManifestManager(rolodex.conn, rolodex)
        now = datetime.utcnow()

        # Create initial entries (old, low access)
        old_ids = []
        for i in range(5):
            entry = make_entry(
                content=f"Old entry {i} with minimal relevance" * 5,
                access_count=1,
                last_accessed=now - timedelta(days=30),
                created_at=now - timedelta(days=30),
            )
            rolodex.create_entry(entry)
            old_ids.append(entry.id)

        # Build initial super manifest
        manifest = mm.build_super_manifest(available_budget=2000)
        assert len(manifest.entries) > 0

        # Create new high-value entries (after manifest)
        import time
        time.sleep(0.1)  # Ensure created_at is after manifest.updated_at
        new_ids = []
        for i in range(3):
            entry = make_entry(
                content=f"Important new entry {i} with high access" * 5,
                access_count=50,
                last_accessed=now,
                created_at=datetime.utcnow(),
            )
            rolodex.create_entry(entry)
            new_ids.append(entry.id)

        # Build incremental manifest
        incremental = mm.build_incremental_manifest(manifest, available_budget=2000)
        assert incremental.manifest_type == "incremental"

        # At least some new entries should be in the manifest
        manifest_ids = {me.entry_id for me in incremental.entries}
        new_in_manifest = [nid for nid in new_ids if nid in manifest_ids]
        assert len(new_in_manifest) > 0, "New high-scoring entries should be promoted"
    finally:
        rolodex.close()
        os.unlink(db_path)


def test_incremental_manifest_no_changes():
    """When no new entries exist, manifest stays current."""
    rolodex, db_path = make_rolodex()
    try:
        mm = ManifestManager(rolodex.conn, rolodex)
        now = datetime.utcnow()

        entry = make_entry(
            content="Stable entry" * 10,
            access_count=5,
            last_accessed=now,
        )
        rolodex.create_entry(entry)

        manifest = mm.build_super_manifest(available_budget=5000)
        original_id = manifest.manifest_id

        # Incremental with no new entries
        result = mm.build_incremental_manifest(manifest, available_budget=5000)
        # Should just bump timestamp, not create new manifest
        assert result.updated_at >= manifest.updated_at
    finally:
        rolodex.close()
        os.unlink(db_path)


def test_refine_manifest_penalizes_dead_weight():
    """Unaccessed entries get score penalty after refinement."""
    rolodex, db_path = make_rolodex()
    try:
        mm = ManifestManager(rolodex.conn, rolodex)
        now = datetime.utcnow()
        session_id = str(uuid.uuid4())

        # Create entries
        accessed_entry = make_entry(
            content="This entry will be accessed" * 10,
            conversation_id=session_id,
            access_count=10,
            last_accessed=now,
        )
        rolodex.create_entry(accessed_entry)

        dead_entry = make_entry(
            content="This entry will NOT be accessed" * 10,
            conversation_id=session_id,
            access_count=10,
            last_accessed=now,
        )
        rolodex.create_entry(dead_entry)

        # Build manifest
        manifest = mm.build_super_manifest(available_budget=5000)

        # Record original scores
        original_scores = {me.entry_id: me.composite_score for me in manifest.entries}

        # Simulate a query that returns only the accessed entry
        log_query(rolodex.conn, session_id, "test query", True, [accessed_entry.id])

        # Refine
        refined = mm.refine_manifest(manifest, session_id, available_budget=5000)

        # Check scores
        for me in refined.entries:
            if me.entry_id == accessed_entry.id:
                # Should be boosted (1.5x)
                assert me.was_accessed or me.selection_reason == "behavioral"
            elif me.entry_id == dead_entry.id:
                orig = original_scores.get(me.entry_id, 0)
                if orig > 0:
                    # Dead weight should be penalized
                    assert me.composite_score <= orig, "Dead weight should have lower score"
    finally:
        rolodex.close()
        os.unlink(db_path)


def test_refine_manifest_surfaces_emerged_topics():
    """New topics from the session get included in refined manifest."""
    rolodex, db_path = make_rolodex()
    try:
        mm = ManifestManager(rolodex.conn, rolodex)
        now = datetime.utcnow()
        session_id = str(uuid.uuid4())

        # Create old entries in topic A
        topic_a = create_topic(rolodex.conn, "topic_a")
        for i in range(3):
            entry = make_entry(
                content=f"Old topic A content {i}" * 10,
                access_count=5,
                last_accessed=now - timedelta(hours=1),
                created_at=now - timedelta(days=1),
            )
            rolodex.create_entry(entry)
            assign_topic(rolodex.conn, entry.id, topic_a)

        # Build initial manifest (only topic A)
        manifest = mm.build_super_manifest(available_budget=10000)
        topics_before = {me.topic_label for me in manifest.entries if me.topic_label}
        assert "topic_a" in topics_before

        # Now create entries in new topic B during this session
        topic_b = create_topic(rolodex.conn, "topic_b")
        new_entry = make_entry(
            content="New topic B emerged in this session" * 10,
            conversation_id=session_id,
            access_count=3,
            last_accessed=now,
        )
        rolodex.create_entry(new_entry)
        assign_topic(rolodex.conn, new_entry.id, topic_b)

        # Refine
        refined = mm.refine_manifest(manifest, session_id, available_budget=10000)
        topics_after = {me.topic_label for me in refined.entries if me.topic_label}
        assert "topic_b" in topics_after, "New topic should emerge in refined manifest"
    finally:
        rolodex.close()
        os.unlink(db_path)


def test_invalidate_clears_all():
    """Force-fresh invalidation removes all manifests."""
    rolodex, db_path = make_rolodex()
    try:
        mm = ManifestManager(rolodex.conn, rolodex)
        now = datetime.utcnow()

        # Create entries and manifest
        entry = make_entry(content="test content" * 10, access_count=5, last_accessed=now)
        rolodex.create_entry(entry)
        mm.build_super_manifest(available_budget=5000)

        # Verify manifest exists
        assert mm.get_latest_manifest() is not None

        # Invalidate
        count = mm.invalidate()
        assert count >= 1

        # Verify gone
        assert mm.get_latest_manifest() is None
    finally:
        rolodex.close()
        os.unlink(db_path)


def test_mark_entry_accessed():
    """Recall tracking marks entries as accessed."""
    rolodex, db_path = make_rolodex()
    try:
        mm = ManifestManager(rolodex.conn, rolodex)
        now = datetime.utcnow()

        entry = make_entry(content="trackable entry" * 10, access_count=5, last_accessed=now)
        rolodex.create_entry(entry)
        manifest = mm.build_super_manifest(available_budget=5000)

        # Initially not accessed
        assert not manifest.entries[0].was_accessed

        # Mark accessed
        mm.mark_entry_accessed(manifest.manifest_id, entry.id)

        # Reload and verify
        reloaded = mm.get_latest_manifest()
        matched = [me for me in reloaded.entries if me.entry_id == entry.id]
        assert len(matched) == 1
        assert matched[0].was_accessed
    finally:
        rolodex.close()
        os.unlink(db_path)


def test_manifest_entries_survive_superseded():
    """If a manifest entry's source gets superseded, it's dropped on reload."""
    rolodex, db_path = make_rolodex()
    try:
        mm = ManifestManager(rolodex.conn, rolodex)
        now = datetime.utcnow()

        entry = make_entry(content="will be superseded" * 10, access_count=5, last_accessed=now)
        rolodex.create_entry(entry)
        manifest = mm.build_super_manifest(available_budget=5000)
        assert len(manifest.entries) == 1

        # Supersede the entry
        new_entry = make_entry(content="replacement entry" * 10, access_count=5, last_accessed=now)
        rolodex.create_entry(new_entry)
        rolodex.supersede_entry(entry.id, new_entry.id)

        # Reload manifest — superseded entry should be gone
        reloaded = mm.get_latest_manifest()
        manifest_ids = {me.entry_id for me in reloaded.entries}
        assert entry.id not in manifest_ids, "Superseded entry should be dropped"
    finally:
        rolodex.close()
        os.unlink(db_path)


def test_manifest_stats():
    """Stats command returns correct data."""
    rolodex, db_path = make_rolodex()
    try:
        mm = ManifestManager(rolodex.conn, rolodex)
        now = datetime.utcnow()

        # No manifest
        stats = mm.get_stats()
        assert stats["total_manifests"] == 0
        assert not stats["has_active_manifest"]

        # Create manifest
        entry = make_entry(content="stats test" * 10, access_count=5, last_accessed=now)
        rolodex.create_entry(entry)
        mm.build_super_manifest(available_budget=5000)

        stats = mm.get_stats()
        assert stats["total_manifests"] >= 1
        assert stats["has_active_manifest"]
        assert stats["entry_count"] >= 1
    finally:
        rolodex.close()
        os.unlink(db_path)


def test_count_entries_after():
    """count_entries_after returns correct count."""
    rolodex, db_path = make_rolodex()
    try:
        mm = ManifestManager(rolodex.conn, rolodex)
        now = datetime.utcnow()
        before = now - timedelta(hours=1)

        # Create entries
        for i in range(3):
            entry = make_entry(content=f"entry {i}" * 5, access_count=1, last_accessed=now)
            rolodex.create_entry(entry)

        count = mm.count_entries_after(before)
        assert count == 3

        count_future = mm.count_entries_after(now + timedelta(hours=1))
        assert count_future == 0
    finally:
        rolodex.close()
        os.unlink(db_path)


def test_full_lifecycle():
    """Integration: super boot → ingest → end (refine) → incremental boot."""
    rolodex, db_path = make_rolodex()
    try:
        mm = ManifestManager(rolodex.conn, rolodex)
        now = datetime.utcnow()
        session_1 = str(uuid.uuid4())

        # Phase 1: Create initial entries and super boot
        for i in range(10):
            entry = make_entry(
                content=f"Initial entry {i} with content about project alpha. " * 5,
                conversation_id=session_1,
                access_count=i + 1,
                last_accessed=now - timedelta(hours=i),
                created_at=now - timedelta(days=i),
            )
            rolodex.create_entry(entry)

        manifest_1 = mm.build_super_manifest(available_budget=5000)
        assert manifest_1.manifest_type == "super"
        assert len(manifest_1.entries) > 0
        initial_count = len(manifest_1.entries)

        # Phase 2: Simulate session activity — log some queries
        accessed_ids = [manifest_1.entries[0].entry_id] if manifest_1.entries else []
        if accessed_ids:
            log_query(rolodex.conn, session_1, "project alpha", True, accessed_ids)
            mm.mark_entry_accessed(manifest_1.manifest_id, accessed_ids[0])

        # Phase 3: End session → refine
        refined = mm.refine_manifest(manifest_1, session_1, available_budget=5000)
        assert refined.manifest_type == "refined"
        assert refined.source_session_id == session_1

        # Phase 4: New session — add new entries
        session_2 = str(uuid.uuid4())
        import time
        time.sleep(0.1)
        for i in range(5):
            entry = make_entry(
                content=f"New session 2 entry {i} about project beta. " * 5,
                conversation_id=session_2,
                access_count=20,  # High access = high score
                last_accessed=datetime.utcnow(),
                created_at=datetime.utcnow(),
            )
            rolodex.create_entry(entry)

        # Phase 5: Incremental boot
        latest = mm.get_latest_manifest()
        incremental = mm.build_incremental_manifest(latest, available_budget=5000)
        assert incremental.manifest_type == "incremental"

        # Some new entries should have been promoted
        delta_entries = [me for me in incremental.entries if me.selection_reason == "delta_promotion"]
        assert len(delta_entries) > 0, "New high-score entries should be promoted via delta"

    finally:
        rolodex.close()
        os.unlink(db_path)
