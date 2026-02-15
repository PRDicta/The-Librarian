"""
Tests for Phase 8: IngestionQueue + TopicRouter + Topic-scoped search.
"""
import asyncio
import os
import sys
import tempfile
import shutil
import pytest

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from src.core.librarian import TheLibrarian
from src.core.types import Message, MessageRole, RolodexEntry, EntryCategory, Tier
from src.indexing.ingestion_queue import IngestionQueue, IngestionTask, TaskStatus
from src.indexing.topic_router import TopicRouter
from src.utils.config import LibrarianConfig

# ─── Fixtures ────────────────────────────────────────────────────────────────

test_dir = None
db_path = None


def setup_module(module):
    global test_dir, db_path
    test_dir = tempfile.mkdtemp(prefix="librarian_phase8_test_")
    db_path = os.path.join(test_dir, "test_rolodex.db")


def teardown_module(module):
    global test_dir
    if test_dir and os.path.exists(test_dir):
        shutil.rmtree(test_dir)


def _make_lib(queue_enabled=False):
    """Create TheLibrarian with a fresh test DB."""
    config = LibrarianConfig(
        ingestion_queue_enabled=queue_enabled,
        ingestion_num_workers=1,
        ingestion_pause_on_query=True,
    )
    return TheLibrarian(db_path=db_path, config=config)


# ─── IngestionQueue Unit Tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_queue_create_stub_entry():
    """Stub entries have content but no embedding."""
    queue = IngestionQueue()
    msg = Message(role=MessageRole.USER, content="Test content here", turn_number=1)
    stub = queue.create_stub_entry(msg, "conv-123")

    assert stub.content == "Test content here"
    assert stub.conversation_id == "conv-123"
    assert stub.embedding is None
    assert stub.category == EntryCategory.NOTE
    assert "pending-enrichment" in stub.tags
    assert stub.metadata.get("enrichment_status") == "pending"


@pytest.mark.asyncio
async def test_queue_enqueue_and_process():
    """Tasks enqueue and get processed by workers."""
    processed = []

    async def mock_enrichment(task):
        processed.append(task.id)

    queue = IngestionQueue(enrichment_fn=mock_enrichment, num_workers=1)
    await queue.start()

    task = IngestionTask(conversation_id="test", turn_number=1)
    await queue.enqueue(task)

    # Wait for processing
    drained = await queue.wait_for_drain(timeout=5.0)
    assert drained
    assert task.id in processed

    await queue.shutdown()


@pytest.mark.asyncio
async def test_queue_pause_resume():
    """Workers pause and resume correctly."""
    call_count = 0

    async def slow_enrichment(task):
        nonlocal call_count
        call_count += 1

    queue = IngestionQueue(enrichment_fn=slow_enrichment, num_workers=1)
    await queue.start()

    # Pause
    await queue.pause()
    assert queue.is_paused()

    # Enqueue while paused
    task = IngestionTask(conversation_id="test", turn_number=1)
    await queue.enqueue(task)

    # Give time — should NOT process
    await asyncio.sleep(0.2)
    assert call_count == 0  # Still paused

    # Resume
    await queue.resume()
    assert not queue.is_paused()

    drained = await queue.wait_for_drain(timeout=5.0)
    assert drained
    assert call_count == 1

    await queue.shutdown()


@pytest.mark.asyncio
async def test_queue_stats():
    """Queue reports accurate stats."""
    queue = IngestionQueue(num_workers=2)
    stats = queue.get_stats()
    assert stats["enabled"] is True
    assert stats["num_workers"] == 2
    assert stats["completed"] == 0


# ─── Integration: Legacy Mode (queue disabled) ──────────────────────────────

@pytest.mark.asyncio
async def test_legacy_ingest_still_works():
    """With queue disabled, ingest works as before (synchronous)."""
    lib = _make_lib(queue_enabled=False)

    entries = await lib.ingest("user", "I prefer dark mode and use Python 3.12 with type hints.")
    assert len(entries) >= 1
    # Legacy mode: entries have embeddings
    assert entries[0].embedding is not None
    assert entries[0].category != EntryCategory.NOTE or entries[0].tags != ["pending-enrichment"]

    lib.rolodex.close()


# ─── Integration: Queue Mode ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_queue_ingest_returns_stub():
    """With queue enabled, ingest returns stub immediately."""
    config = LibrarianConfig(
        ingestion_queue_enabled=True,
        ingestion_num_workers=1,
    )
    lib = TheLibrarian(db_path=os.path.join(test_dir, "queue_test.db"), config=config)

    entries = await lib.ingest("user", "I prefer dark mode and use Python 3.12 with type hints.")
    assert len(entries) == 1

    stub = entries[0]
    assert stub.content == "I prefer dark mode and use Python 3.12 with type hints."
    assert "pending-enrichment" in stub.tags
    assert stub.metadata.get("enrichment_status") == "pending"

    # Stub should be FTS-searchable immediately
    kw_results = lib.rolodex.keyword_search("dark mode", limit=5)
    assert len(kw_results) >= 1

    # Wait for background enrichment to complete
    if lib.ingestion_queue:
        await lib.ingestion_queue.wait_for_drain(timeout=10.0)
        await lib.ingestion_queue.shutdown()

    lib.rolodex.close()


# ─── TopicRouter Tests ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_topic_router_create_topic():
    """Topics can be created and listed."""
    lib = _make_lib()
    router = TopicRouter(conn=lib.rolodex.conn, embedding_manager=lib.embeddings)

    topic_id = await router.create_topic(
        label="Python decorators",
        description="Everything about Python decorators",
    )
    assert topic_id

    topics = router.list_topics()
    assert len(topics) >= 1
    assert any(t["label"] == "Python decorators" for t in topics)

    lib.rolodex.close()


@pytest.mark.asyncio
async def test_topic_router_infer_from_tags():
    """TopicRouter assigns entries to topics based on tag overlap."""
    lib = _make_lib()
    router = TopicRouter(conn=lib.rolodex.conn, embedding_manager=lib.embeddings)

    # Create a topic
    await router.create_topic(label="python decorators")

    # Create an entry with matching tags
    entry = RolodexEntry(
        content="Use @functools.wraps for decorators",
        tags=["python", "decorators", "functools"],
        category=EntryCategory.IMPLEMENTATION,
    )

    topic_id = await router.infer_topic(entry)
    assert topic_id is not None

    lib.rolodex.close()


@pytest.mark.asyncio
async def test_topic_router_create_new_topic_from_tags():
    """TopicRouter creates a new topic when no match exists."""
    td = tempfile.mkdtemp(prefix="topic_test_")
    dp = os.path.join(td, "test.db")
    lib = TheLibrarian(db_path=dp)
    router = TopicRouter(conn=lib.rolodex.conn, embedding_manager=lib.embeddings)

    # Entry with enough tags but no matching topic
    entry = RolodexEntry(
        content="Docker containers with Kubernetes orchestration",
        tags=["docker", "kubernetes", "containers", "orchestration"],
        category=EntryCategory.FACT,
        embedding=await lib.embeddings.embed_text("Docker containers with Kubernetes"),
    )

    topic_id = await router.infer_topic(entry)
    assert topic_id is not None

    # Verify topic was created
    topics = router.list_topics()
    assert len(topics) >= 1

    lib.rolodex.close()
    shutil.rmtree(td)


@pytest.mark.asyncio
async def test_topic_count_and_unassigned():
    """count_topics and count_unassigned_entries work."""
    td = tempfile.mkdtemp(prefix="topic_count_test_")
    dp = os.path.join(td, "test.db")
    lib = TheLibrarian(db_path=dp)
    router = TopicRouter(conn=lib.rolodex.conn, embedding_manager=lib.embeddings)

    # Ingest something (creates entries without topics)
    await lib.ingest("user", "I use FastAPI with SQLAlchemy for my backend API.")

    unassigned = router.count_unassigned_entries()
    assert unassigned >= 1

    lib.rolodex.close()
    shutil.rmtree(td)


# ─── Topic-Scoped Search Tests ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_topic_scoped_keyword_search():
    """Keyword search scoped to a topic only returns entries in that topic."""
    td = tempfile.mkdtemp(prefix="topic_search_test_")
    dp = os.path.join(td, "test.db")
    lib = TheLibrarian(db_path=dp)
    router = TopicRouter(conn=lib.rolodex.conn, embedding_manager=lib.embeddings)

    # Create two topics
    auth_topic = await router.create_topic(label="authentication")
    db_topic = await router.create_topic(label="database")

    # Ingest entries and assign topics manually
    entries_auth = await lib.ingest("user", "JWT tokens are used for authentication in our API")
    entries_db = await lib.ingest("user", "PostgreSQL is the database backend for our application")

    if entries_auth:
        lib.rolodex.conn.execute(
            "UPDATE rolodex_entries SET topic_id = ? WHERE id = ?",
            (auth_topic, entries_auth[0].id)
        )
    if entries_db:
        lib.rolodex.conn.execute(
            "UPDATE rolodex_entries SET topic_id = ? WHERE id = ?",
            (db_topic, entries_db[0].id)
        )
    lib.rolodex.conn.commit()

    # Search within auth topic
    results = lib.rolodex.keyword_search_by_topic("authentication", auth_topic, limit=5)
    for entry, _ in results:
        assert entry.content  # Should find auth-related entries

    # Search within db topic for "authentication" should find nothing
    results_db = lib.rolodex.keyword_search_by_topic("authentication", db_topic, limit=5)
    assert len(results_db) == 0

    lib.rolodex.close()
    shutil.rmtree(td)


# ─── CLI Topics Command Test ────────────────────────────────────────────────

def test_cli_topics_list():
    """Topics list command works via subprocess."""
    import subprocess
    CLI = os.path.join(SCRIPT_DIR, "librarian_cli.py")
    PYTHON = sys.executable

    td = tempfile.mkdtemp(prefix="cli_topics_test_")
    dp = os.path.join(td, "test.db")
    sp = os.path.join(td, "session.json")

    env = {k: v for k, v in os.environ.items() if v is not None}
    env["LIBRARIAN_DB_PATH"] = dp
    env["LIBRARIAN_SESSION_FILE"] = sp

    # Boot first
    subprocess.run([PYTHON, CLI, "boot"], env=env, capture_output=True, timeout=15)

    # Run topics list
    result = subprocess.run(
        [PYTHON, CLI, "topics", "list"],
        env=env, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0
    # Should either show "No topics yet" or valid output
    assert "topics" in result.stdout.lower() or "No topics" in result.stdout

    shutil.rmtree(td)
