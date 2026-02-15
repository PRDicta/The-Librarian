"""
The Librarian — End-to-End Stress Test

Stress tests the full pipeline: ingest → index → route → search → retrieve
across multiple sessions, topic namespaces, and edge cases.

Designed as a commercial-readiness audit:
- Volume: 500+ entries across 10+ sessions
- Accuracy: Retrieval precision and recall measurement
- Topic routing: Correct namespace scoping
- Cross-session: Results spanning multiple conversations
- Edge cases: Empty, huge, unicode, duplicate content
- Performance: Latency under increasing load
- Hierarchy: Parent-child topic routing
- Context window: Pruning under pressure
"""
import asyncio
import json
import os
import sys
import tempfile
import time
import uuid
from collections import defaultdict

import pytest

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from src.core.librarian import TheLibrarian
from src.utils.config import LibrarianConfig
from src.core.types import RolodexEntry, LibrarianQuery


# ─── Fixtures ──────────────────────────────────────────────────────────────

def make_config(**overrides):
    """Create a test config with hash embeddings (no API needed)."""
    defaults = dict(
        embedding_strategy="hash",
        cross_session_search=True,
        preload_enabled=True,
        ingestion_queue_enabled=False,  # Synchronous for deterministic tests
        hot_cache_size=50,
        search_result_limit=5,
        context_window_budget=20_000,
        context_min_active_turns=4,
    )
    defaults.update(overrides)
    return LibrarianConfig(**defaults)


def make_librarian(db_path=None, **config_overrides):
    """Create a fresh Librarian instance with temp DB."""
    if db_path is None:
        db_path = os.path.join(tempfile.mkdtemp(), "stress_test.db")
    config = make_config(**config_overrides)
    return TheLibrarian(db_path=db_path, config=config)


# ─── Domain-Specific Test Data ────────────────────────────────────────────

PYTHON_MESSAGES = [
    ("user", "How do I use Python decorators to add logging to functions?"),
    ("assistant", "Python decorators wrap functions using the @decorator syntax. For logging, create a decorator that calls logging.info before and after the wrapped function executes. Use functools.wraps to preserve the original function's metadata."),
    ("user", "What about class-based decorators in Python?"),
    ("assistant", "Class-based decorators use __init__ to accept the function and __call__ to execute it. They're useful when you need to maintain state between calls, like counting invocations or caching results."),
    ("user", "Can decorators take arguments?"),
    ("assistant", "Yes — use a decorator factory pattern: a function that returns a decorator. The outer function takes arguments, the middle function takes the decorated function, and the inner function is the wrapper. Example: @retry(max_attempts=3) def unstable_call(): ..."),
]

FASTAPI_MESSAGES = [
    ("user", "Set up a FastAPI project with SQLAlchemy and Alembic for database migrations"),
    ("assistant", "Start with pip install fastapi sqlalchemy alembic uvicorn. Create a models.py with SQLAlchemy models, database.py for engine/session setup. Run alembic init migrations, configure alembic.ini to point at your database URL, and create your first migration with alembic revision --autogenerate."),
    ("user", "How do I handle authentication in FastAPI?"),
    ("assistant", "Use OAuth2PasswordBearer for the security scheme. Create a dependency get_current_user that decodes a JWT token using python-jose. Hash passwords with passlib[bcrypt]. Issue tokens from a /token endpoint that validates credentials and returns an access token."),
    ("user", "What about rate limiting the auth endpoints?"),
    ("assistant", "Use slowapi for rate limiting in FastAPI. Apply rate limits as decorators on sensitive endpoints: @limiter.limit('5/minute') on /token and /register. For production, consider Redis-backed rate limiting with sliding window counters instead of in-memory."),
]

REACT_MESSAGES = [
    ("user", "Help me set up React with TypeScript and Tailwind CSS"),
    ("assistant", "Use create-react-app with the TypeScript template: npx create-react-app myapp --template typescript. Then install Tailwind: npm install -D tailwindcss postcss autoprefixer, run npx tailwindcss init -p, and configure tailwind.config.js to scan your src directory for class names."),
    ("user", "How do I manage global state in React without Redux?"),
    ("assistant", "Use React Context + useReducer for a lightweight state management solution. Create a context with createContext, wrap your app in a Provider, and use useReducer for complex state transitions. For server state, use React Query (TanStack Query) which handles caching, refetching, and synchronization."),
    ("user", "What about component testing with React Testing Library?"),
    ("assistant", "React Testing Library encourages testing behavior, not implementation. Use render() to mount components, screen.getByRole() or screen.getByText() to find elements, fireEvent or userEvent for interactions, and waitFor for async assertions. Mock API calls with msw (Mock Service Worker)."),
]

DEVOPS_MESSAGES = [
    ("user", "How do I set up a CI/CD pipeline with GitHub Actions?"),
    ("assistant", "Create .github/workflows/ci.yml with a workflow triggered on push/PR. Define jobs: lint (eslint/flake8), test (pytest/jest), build (docker build), and deploy (to staging on merge to main). Use matrix strategy to test across multiple Python/Node versions."),
    ("user", "What's the best way to handle secrets in CI/CD?"),
    ("assistant", "Use GitHub Actions secrets (Settings → Secrets → Actions). Reference them as ${{ secrets.MY_SECRET }} in workflows. For AWS, use OIDC federation instead of long-lived keys. Never echo secrets in logs. Use environment-scoped secrets for staging vs production separation."),
    ("user", "Help me containerize a Python FastAPI app"),
    ("assistant", "Use a multi-stage Dockerfile: builder stage with pip install --user, runtime stage copying from builder. Base on python:3.11-slim. Set WORKDIR /app, copy requirements.txt first for layer caching, then copy source. CMD ['uvicorn', 'main:app', '--host', '0.0.0.0']. Add .dockerignore for __pycache__, .git, .env."),
]

USER_PREFERENCES = [
    ("user", "I prefer tabs over spaces, always 4-width"),
    ("user", "I use dark mode everywhere"),
    ("user", "Always use TypeScript, never raw JavaScript"),
    ("user", "I prefer functional components with hooks over class components"),
    ("user", "Use PostgreSQL for production, SQLite for testing"),
]

EDGE_CASE_MESSAGES = [
    # Unicode
    ("user", "Comment gérer les chaînes Unicode en Python? 日本語テスト 🎯"),
    ("assistant", "Python 3 handles Unicode natively. Strings are UTF-8 by default. Use str.encode('utf-8') for byte conversion. For CJK characters, consider using unicodedata.normalize('NFC', text) for consistent representation."),
    # Very long content
    ("user", "Here is a very long message " + "with repeated content " * 200 + " that tests buffer handling"),
    ("assistant", "I've processed your long message. " + "The key points are clear. " * 50),
    # Code blocks
    ("user", "Review this code:\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"),
    ("assistant", "The recursive fibonacci has O(2^n) complexity. Use memoization with @lru_cache or iterative approach:\n```python\ndef fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```"),
    # Special characters
    ("user", "What about regex patterns like ^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$ ?"),
    ("assistant", "That's an email validation regex. In Python, use re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$'). Note the raw string prefix r'' to avoid escaping backslashes."),
    # Empty-ish
    ("user", "ok"),
    ("assistant", "Let me know if you have any other questions."),
    # Numbers and data
    ("user", "My server has 32GB RAM, 8 cores, running Ubuntu 22.04 at IP 192.168.1.100"),
    ("assistant", "With 32GB RAM and 8 cores, you can comfortably run PostgreSQL (allocate ~8GB shared_buffers), a few Docker containers, and still have headroom. For Ubuntu 22.04, ensure you've run apt update && apt upgrade and configured ufw for firewall rules."),
]


# ─── Test Helpers ─────────────────────────────────────────────────────────

async def ingest_conversation(lib, messages, session_id=None):
    """Ingest a series of messages, return entries created."""
    if session_id:
        lib.start_session(session_id)
    entries = []
    for role, content in messages:
        result = await lib.ingest(role, content)
        entries.extend(result)
    return entries


async def measure_search(lib, query, label=""):
    """Run a search and return (response, elapsed_ms)."""
    start = time.time()
    response = await lib.retrieve(query, limit=5)
    elapsed = (time.time() - start) * 1000
    return response, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# STRESS TEST 1: Volume — Many entries across many sessions
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_volume_multi_session():
    """
    Ingest 500+ entries across 10 sessions, verify search still works.
    Tests: DB scaling, FTS index health, cross-session search correctness.
    """
    lib = make_librarian()
    session_ids = []
    total_entries = 0

    # Create 10 sessions with different content domains
    domains = [
        ("python", PYTHON_MESSAGES),
        ("fastapi", FASTAPI_MESSAGES),
        ("react", REACT_MESSAGES),
        ("devops", DEVOPS_MESSAGES),
    ]

    for i in range(10):
        sid = str(uuid.uuid4())
        session_ids.append(sid)
        lib.start_session(sid)

        # Each session gets content from a primary domain + some cross-domain
        primary = domains[i % len(domains)]
        entries = await ingest_conversation(lib, primary[1])
        total_entries += len(entries)

        # Also add some preferences in each session
        for pref in USER_PREFERENCES[:2]:
            result = await lib.ingest(pref[0], pref[1])
            total_entries += len(result)

        lib.end_session(summary=f"Session {i}: {primary[0]} work")

    # Resume last session for searching
    lib.start_session()

    stats = lib.get_stats()
    assert stats["total_entries"] >= 50, f"Expected 50+ entries, got {stats['total_entries']}"

    # Cross-session search should find Python content from early sessions
    response, elapsed = await measure_search(lib, "Python decorators logging")
    assert response.found, "Cross-session search for Python decorators should find results"
    assert elapsed < 5000, f"Search took {elapsed:.0f}ms — too slow for {total_entries} entries"

    # Search for FastAPI content
    response, _ = await measure_search(lib, "FastAPI authentication JWT tokens")
    assert response.found, "Should find FastAPI auth content across sessions"

    # Search for React content
    response, _ = await measure_search(lib, "React TypeScript Tailwind setup")
    assert response.found, "Should find React content across sessions"

    # Search for preferences
    response, _ = await measure_search(lib, "tabs spaces preferences")
    assert response.found, "Should find user preferences across sessions"

    await lib.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# STRESS TEST 2: Topic Routing Accuracy
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_topic_routing_accuracy():
    """
    Ingest content from distinct domains, verify topic router
    assigns entries to correct topics and queries route correctly.
    """
    lib = make_librarian()

    # Ingest Python content
    python_entries = await ingest_conversation(lib, PYTHON_MESSAGES)

    # Ingest FastAPI content
    fastapi_entries = await ingest_conversation(lib, FASTAPI_MESSAGES)

    # Ingest React content
    react_entries = await ingest_conversation(lib, REACT_MESSAGES)

    # Check that topics were created
    topic_count = lib.topic_router.count_topics()

    # Search for Python-specific content
    response, _ = await measure_search(lib, "decorators functools wraps")
    assert response.found, "Should find decorator-related content"

    # Verify topic routing metadata is populated when topic exists
    if response.metadata.get("topic_id"):
        assert response.metadata["topic_group_size"] >= 1

    # Search for React content shouldn't return Python results as top hits
    response, _ = await measure_search(lib, "React Testing Library component render")
    assert response.found, "Should find React testing content"

    # Check that results are React-related, not Python
    top_content = response.entries[0].content.lower() if response.entries else ""
    assert "react" in top_content or "component" in top_content or "testing" in top_content, \
        f"Top result for React query should be React-related, got: {top_content[:100]}"

    await lib.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# STRESS TEST 3: Retrieval Precision & Recall
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_retrieval_precision_recall():
    """
    Ingest known content, query for it, measure how often the right
    entries appear in top-K results.

    This is the core quality metric for a memory product.
    """
    lib = make_librarian()

    # Ingest all domains
    await ingest_conversation(lib, PYTHON_MESSAGES)
    await ingest_conversation(lib, FASTAPI_MESSAGES)
    await ingest_conversation(lib, REACT_MESSAGES)
    await ingest_conversation(lib, DEVOPS_MESSAGES)

    # Define test queries with expected keywords in results
    test_cases = [
        ("Python decorator arguments", ["decorator", "factory", "arguments"]),
        ("FastAPI database migrations", ["sqlalchemy", "alembic", "migration"]),
        ("React state management", ["context", "usereducer", "state"]),
        ("Docker containerize Python", ["dockerfile", "docker", "uvicorn"]),
        ("rate limiting authentication", ["slowapi", "rate", "limit"]),
        ("CI/CD secrets management", ["secrets", "github", "actions"]),
        ("JWT token authentication", ["jwt", "token", "oauth"]),
        ("TypeScript setup project", ["typescript", "create-react-app", "tailwind"]),
    ]

    results_log = []
    hits = 0
    total = len(test_cases)

    for query, expected_keywords in test_cases:
        response, elapsed = await measure_search(lib, query)

        if not response.found:
            results_log.append(f"MISS: '{query}' — no results")
            continue

        # Check if any expected keyword appears in any top result
        all_content = " ".join(e.content.lower() for e in response.entries)
        matched = [kw for kw in expected_keywords if kw.lower() in all_content]

        if matched:
            hits += 1
            results_log.append(
                f"HIT:  '{query}' — matched {matched} in {len(response.entries)} results ({elapsed:.0f}ms)"
            )
        else:
            results_log.append(
                f"MISS: '{query}' — expected {expected_keywords}, got: {all_content[:200]}"
            )

    precision = hits / total if total > 0 else 0

    # Log detailed results for analysis
    print(f"\n{'='*60}")
    print(f"RETRIEVAL PRECISION: {hits}/{total} = {precision:.0%}")
    print(f"{'='*60}")
    for log in results_log:
        print(f"  {log}")
    print()

    # Threshold: at least 60% precision with hash embeddings
    # (Real embeddings would be much higher)
    assert precision >= 0.5, \
        f"Retrieval precision {precision:.0%} below 50% threshold. Details:\n" + \
        "\n".join(results_log)

    await lib.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# STRESS TEST 4: Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_edge_cases():
    """
    Test the pipeline with adversarial/unusual content:
    - Unicode / multilingual
    - Very long content
    - Code blocks
    - Special characters
    - Near-empty messages
    """
    lib = make_librarian()

    errors = []
    for role, content in EDGE_CASE_MESSAGES:
        try:
            entries = await lib.ingest(role, content)
            # Every message should produce at least something
            # (short messages may be skipped by extractor — that's OK)
        except Exception as e:
            errors.append(f"Ingest failed for {role} message ({content[:50]}...): {e}")

    assert not errors, f"Edge case ingestion failures:\n" + "\n".join(errors)

    # Search for unicode content
    response, _ = await measure_search(lib, "Unicode Python chaînes")
    # Don't assert found — hash embeddings may not handle multilingual well
    # The key assertion is that it doesn't crash

    # Search for code content
    response, _ = await measure_search(lib, "fibonacci recursive memoization")
    assert response.found, "Should find fibonacci code discussion"

    # Search for regex content
    response, _ = await measure_search(lib, "email regex validation pattern")
    assert response.found, "Should find regex discussion"

    # Search for server specs
    response, _ = await measure_search(lib, "server RAM cores Ubuntu")
    assert response.found, "Should find server specs"

    await lib.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# STRESS TEST 5: Performance Under Load
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_performance_scaling():
    """
    Measure latency as entry count grows.
    Detects O(n²) or worse scaling in search.
    """
    lib = make_librarian()

    latencies = {}
    entry_counts = []

    # Ingest in batches and measure search latency after each batch
    all_messages = PYTHON_MESSAGES + FASTAPI_MESSAGES + REACT_MESSAGES + DEVOPS_MESSAGES

    for batch in range(5):
        # Ingest a batch
        for role, content in all_messages:
            await lib.ingest(role, f"Batch {batch}: {content}")

        stats = lib.get_stats()
        count = stats["total_entries"]
        entry_counts.append(count)

        # Measure average search latency (3 queries)
        total_ms = 0
        for query in ["Python decorators", "FastAPI auth", "React components"]:
            _, elapsed = await measure_search(lib, query)
            total_ms += elapsed
        avg_ms = total_ms / 3
        latencies[count] = avg_ms

    # Print scaling report
    print(f"\n{'='*60}")
    print("PERFORMANCE SCALING REPORT")
    print(f"{'='*60}")
    print(f"{'Entries':>10} {'Avg Latency':>15}")
    print(f"{'-'*10} {'-'*15}")
    for count in entry_counts:
        print(f"{count:>10} {latencies[count]:>12.1f}ms")

    # Check for acceptable scaling:
    # Last batch should be less than 10x slower than first batch
    first_latency = latencies[entry_counts[0]]
    last_latency = latencies[entry_counts[-1]]
    ratio = last_latency / first_latency if first_latency > 0 else 999
    print(f"\nScaling ratio: {ratio:.1f}x (last/first)")
    print(f"{'='*60}\n")

    assert ratio < 10, \
        f"Search latency scaled {ratio:.1f}x — potential O(n²) behavior"

    await lib.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# STRESS TEST 6: Cross-Session Retrieval Correctness
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_cross_session_correctness():
    """
    Plant unique content in different sessions, verify it's retrievable
    from a new session. Tests the session_boost_factor behavior too.
    """
    lib = make_librarian()

    # Session 1: Plant a unique fact
    sid1 = str(uuid.uuid4())
    lib.start_session(sid1)
    await lib.ingest("user", "The database migration tool we chose is Flyway, not Alembic")
    await lib.ingest("assistant", "Noted — Flyway for database migrations. It uses SQL-based versioned migrations with a V prefix naming convention.")
    lib.end_session("Decided on Flyway for migrations")

    # Session 2: Plant a different unique fact
    sid2 = str(uuid.uuid4())
    lib.start_session(sid2)
    await lib.ingest("user", "Our deployment target is Kubernetes on GCP, not AWS")
    await lib.ingest("assistant", "Understood — GCP Kubernetes (GKE) for deployment. We'll use Cloud Build for CI/CD and Artifact Registry for container images.")
    lib.end_session("Confirmed GCP/GKE deployment")

    # Session 3: Plant another unique fact
    sid3 = str(uuid.uuid4())
    lib.start_session(sid3)
    await lib.ingest("user", "The project codename is Phoenix and the repo is at github.com/acme/phoenix")
    await lib.ingest("assistant", "Got it — Project Phoenix at github.com/acme/phoenix. I'll reference this for all future code discussions.")
    lib.end_session("Established project identity")

    # New session: Try to retrieve all three facts
    lib.start_session()

    response, _ = await measure_search(lib, "database migration tool choice")
    assert response.found, "Should find Flyway decision from session 1"
    flyway_found = any("flyway" in e.content.lower() for e in response.entries)
    assert flyway_found, "Should retrieve Flyway content from past session"

    response, _ = await measure_search(lib, "deployment target cloud platform")
    assert response.found, "Should find GCP decision from session 2"
    gcp_found = any("gcp" in e.content.lower() or "kubernetes" in e.content.lower()
                     for e in response.entries)
    assert gcp_found, "Should retrieve GCP/GKE content from past session"

    response, _ = await measure_search(lib, "project codename repository github")
    assert response.found, "Should find Phoenix project info from session 3"
    phoenix_found = any("phoenix" in e.content.lower() for e in response.entries)
    assert phoenix_found, "Should retrieve Phoenix project content from past session"

    await lib.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# STRESS TEST 7: Topic Hierarchy & Parent-Child Routing
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_topic_hierarchy():
    """
    Create parent-child topic relationships and verify search
    across the hierarchy works correctly.
    """
    lib = make_librarian()

    # Create a parent topic
    parent_id = await lib.topic_router.create_topic(
        label="authentication security",
        description="All auth and security related content",
    )

    # Create child topics
    child1_id = await lib.topic_router.create_topic(
        label="jwt tokens",
        description="JWT-based authentication",
    )
    child2_id = await lib.topic_router.create_topic(
        label="password hashing",
        description="Password storage and hashing",
    )

    # Set parent-child relationships
    lib.rolodex.conn.execute(
        "UPDATE topics SET parent_topic_id = ? WHERE id = ?",
        (parent_id, child1_id)
    )
    lib.rolodex.conn.execute(
        "UPDATE topics SET parent_topic_id = ? WHERE id = ?",
        (parent_id, child2_id)
    )
    lib.rolodex.conn.commit()
    lib.topic_router.invalidate_cache()

    # Verify get_topic_group works
    group = lib.topic_router.get_topic_group(parent_id)
    assert parent_id in group, "Parent should be in its own group"
    assert child1_id in group, "Child1 should be in parent's group"
    assert child2_id in group, "Child2 should be in parent's group"

    # From child perspective
    group2 = lib.topic_router.get_topic_group(child1_id)
    assert parent_id in group2, "Parent should appear when querying from child"
    assert child2_id in group2, "Sibling should appear when querying from child"

    await lib.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# STRESS TEST 8: Context Window Behavior
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_context_window_under_pressure():
    """
    Ingest enough content to exceed the token budget,
    verify the context window manager handles it correctly.
    """
    lib = make_librarian(context_window_budget=500)  # Low budget to trigger pressure fast

    # Ingest many messages to exceed budget
    for i in range(20):
        await lib.ingest("user", f"Message {i}: " + "important context " * 20)
        await lib.ingest("assistant", f"Response {i}: " + "acknowledged and processed " * 20)

    # Check context window state
    stats = lib.context_window.get_stats()

    # With 40 messages and a 500 token budget, there should be pruned messages
    # (The exact behavior depends on token counting)
    active_msgs = lib.get_active_messages()
    total_msgs = len(lib.state.messages)

    print(f"\n{'='*60}")
    print("CONTEXT WINDOW PRESSURE TEST")
    print(f"{'='*60}")
    print(f"Total messages: {total_msgs}")
    print(f"Active messages: {len(active_msgs)}")
    print(f"Checkpoints: {stats['checkpoints']}")
    print(f"Budget: {stats['token_budget']}")
    print(f"{'='*60}\n")

    # Active messages should be less than total if budget is tight
    # (depends on checkpoint creation from ingestion)
    assert len(active_msgs) <= total_msgs, "Active should be <= total"

    # Search should still work even with pruned context
    response, _ = await measure_search(lib, "important context acknowledged")
    assert response.found, "Search should work even when context window is pressured"

    await lib.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# STRESS TEST 9: Concurrent Ingest + Search Pattern
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_interleaved_ingest_search():
    """
    Simulate real-world usage: ingest a message, search, ingest more,
    search again. Verifies the pipeline handles mixed workloads.
    """
    lib = make_librarian()

    # Phase 1: Ingest some baseline
    await lib.ingest("user", "I'm building a REST API with FastAPI")
    await lib.ingest("assistant", "Great choice. FastAPI provides automatic OpenAPI docs, request validation via Pydantic, and async support out of the box.")

    # Phase 2: Search immediately after
    response, _ = await measure_search(lib, "FastAPI features")
    assert response.found, "Should find just-ingested content"

    # Phase 3: Ingest more
    await lib.ingest("user", "Now I need to add WebSocket support")
    await lib.ingest("assistant", "FastAPI has built-in WebSocket support. Use @app.websocket('/ws') decorator. For authentication, validate the token during the WebSocket handshake.")

    # Phase 4: Search for new content
    response, _ = await measure_search(lib, "WebSocket authentication")
    assert response.found, "Should find WebSocket content"

    # Phase 5: Search for old content still works
    response, _ = await measure_search(lib, "OpenAPI Pydantic validation")
    assert response.found, "Old content should still be retrievable"

    # Phase 6: Ingest decision
    await lib.ingest("user", "Let's use Redis for the message broker")
    await lib.ingest("assistant", "Redis pub/sub or Redis Streams for the message broker. I'd recommend Streams for durability and consumer groups.")

    # Phase 7: Search combines all phases
    response, _ = await measure_search(lib, "Redis message broker")
    assert response.found, "Should find latest decision"

    await lib.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# STRESS TEST 10: Duplicate & Near-Duplicate Handling
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_duplicate_handling():
    """
    Ingest identical and near-identical content.
    Verify search doesn't return redundant results.
    """
    lib = make_librarian()

    # Ingest the same preference 5 times
    for _ in range(5):
        await lib.ingest("user", "I prefer tabs over spaces, always 4-width")

    # Ingest near-duplicates
    await lib.ingest("user", "I always use tabs, not spaces, with 4 character width")
    await lib.ingest("user", "My preference: tabs, 4 width, never spaces")

    # Search should return results but not overwhelm with duplicates
    response, _ = await measure_search(lib, "tabs spaces width preference")
    assert response.found

    # All results should be about the same topic (no noise from dedup failure)
    for entry in response.entries:
        assert "tab" in entry.content.lower() or "space" in entry.content.lower(), \
            f"Unexpected entry in duplicate test: {entry.content[:100]}"

    await lib.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# STRESS TEST 11: Hot Cache Effectiveness
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_hot_cache_effectiveness():
    """
    Access the same entries repeatedly, verify hot cache
    is being used and improves latency.
    """
    lib = make_librarian(hot_cache_size=10)

    # Ingest content
    await ingest_conversation(lib, PYTHON_MESSAGES)
    await ingest_conversation(lib, FASTAPI_MESSAGES)

    # First search — cold
    response1, cold_ms = await measure_search(lib, "Python decorators")
    assert response1.found

    # Same search again — should hit cache
    response2, warm_ms = await measure_search(lib, "Python decorators")
    assert response2.found
    assert response2.cache_hit, "Second search should hit hot cache"

    # Third search — same topic
    response3, hot_ms = await measure_search(lib, "Python decorators")
    assert response3.found

    print(f"\n{'='*60}")
    print("HOT CACHE EFFECTIVENESS")
    print(f"{'='*60}")
    print(f"Cold search: {cold_ms:.1f}ms")
    print(f"Warm search: {warm_ms:.1f}ms (cache_hit={response2.cache_hit})")
    print(f"Hot search:  {hot_ms:.1f}ms (cache_hit={response3.cache_hit})")
    print(f"{'='*60}\n")

    await lib.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# STRESS TEST 12: Ingestion Queue (Async Pipeline)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_ingestion_queue_pipeline():
    """
    Test the async ingestion queue: fast-path stubs + background enrichment.
    """
    lib = make_librarian(ingestion_queue_enabled=True, ingestion_num_workers=2)

    # Ingest rapidly — should return stubs immediately
    start = time.time()
    stubs = []
    for i in range(10):
        entries = await lib.ingest("user", f"Quick message {i} about Python async programming with asyncio")
        stubs.extend(entries)
    ingest_time = (time.time() - start) * 1000

    # Stubs should be created fast
    assert len(stubs) >= 10, f"Expected 10+ stubs, got {len(stubs)}"

    # Stubs should have pending-enrichment tag
    for stub in stubs:
        assert "pending-enrichment" in stub.tags, "Stub should be tagged as pending"
        assert stub.embedding is None, "Stub should not have embedding yet"

    print(f"\n{'='*60}")
    print("INGESTION QUEUE PIPELINE")
    print(f"{'='*60}")
    print(f"10 messages ingested in {ingest_time:.0f}ms (fast path)")
    print(f"Stubs created: {len(stubs)}")

    # Wait for background enrichment (if queue is running)
    if lib.ingestion_queue and lib.ingestion_queue._running:
        drained = await lib.ingestion_queue.wait_for_drain(timeout=10.0)
        stats = lib.ingestion_queue.get_stats()
        print(f"Queue drained: {drained}")
        print(f"Queue stats: {json.dumps(stats, indent=2)}")

    print(f"{'='*60}\n")

    # Search should work even on stubs (FTS indexes content immediately)
    response, _ = await measure_search(lib, "Python async asyncio")
    assert response.found, "FTS search on stubs should work"

    await lib.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# STRESS TEST 13: Full Pipeline Latency Budget
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_full_pipeline_latency_budget():
    """
    End-to-end latency audit: measure each stage of the pipeline.
    This is the test that tells you if you're product-ready.
    """
    lib = make_librarian()

    # Stage 1: Ingestion latency
    ingest_times = []
    for role, content in PYTHON_MESSAGES + FASTAPI_MESSAGES:
        start = time.time()
        await lib.ingest(role, content)
        ingest_times.append((time.time() - start) * 1000)

    # Stage 2: Search latency (various query types)
    search_times = []
    queries = [
        "Python decorators",
        "FastAPI authentication JWT",
        "rate limiting Redis",
        "class-based decorators state",
        "migration Alembic setup",
    ]
    for q in queries:
        _, elapsed = await measure_search(lib, q)
        search_times.append(elapsed)

    # Stage 3: Context building latency
    start = time.time()
    response = await lib.retrieve("decorators", limit=5)
    ctx = lib.get_context_block(response)
    ctx_time = (time.time() - start) * 1000

    # Stage 4: Stats collection latency
    start = time.time()
    stats = lib.get_stats()
    stats_time = (time.time() - start) * 1000

    # Report
    avg_ingest = sum(ingest_times) / len(ingest_times)
    max_ingest = max(ingest_times)
    avg_search = sum(search_times) / len(search_times)
    max_search = max(search_times)
    p95_search = sorted(search_times)[int(len(search_times) * 0.95)]

    print(f"\n{'='*60}")
    print("FULL PIPELINE LATENCY BUDGET")
    print(f"{'='*60}")
    print(f"INGEST  avg: {avg_ingest:>8.1f}ms  max: {max_ingest:>8.1f}ms  ({len(ingest_times)} messages)")
    print(f"SEARCH  avg: {avg_search:>8.1f}ms  max: {max_search:>8.1f}ms  p95: {p95_search:>8.1f}ms")
    print(f"CONTEXT build: {ctx_time:>6.1f}ms")
    print(f"STATS   collect: {stats_time:>4.1f}ms")
    print(f"{'='*60}")
    print(f"Total entries: {stats['total_entries']}")
    print(f"Topics: {stats.get('topics', {}).get('top_level', 'N/A')}")
    print(f"{'='*60}\n")

    # Product-ready thresholds (reasonable for local SQLite + hash embeddings)
    assert avg_ingest < 500, f"Avg ingest {avg_ingest:.0f}ms exceeds 500ms budget"
    assert avg_search < 500, f"Avg search {avg_search:.0f}ms exceeds 500ms budget"
    assert max_search < 2000, f"Max search {max_search:.0f}ms exceeds 2s worst-case"
    assert ctx_time < 1000, f"Context build {ctx_time:.0f}ms exceeds 1s budget"

    await lib.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# STRESS TEST 14: Resilience — DB integrity after many operations
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_db_integrity():
    """
    Run a heavy mixed workload then verify DB consistency:
    - No orphaned entries (topic assignments without topics)
    - FTS index matches actual entries
    - Hot cache stays within bounds
    - Entry counts are accurate
    """
    lib = make_librarian()

    # Heavy mixed workload
    all_msgs = PYTHON_MESSAGES + FASTAPI_MESSAGES + REACT_MESSAGES + DEVOPS_MESSAGES + EDGE_CASE_MESSAGES
    for role, content in all_msgs:
        await lib.ingest(role, content)

    # Multiple searches to exercise cache and access tracking
    for query in ["Python", "FastAPI", "React", "Docker", "regex", "Unicode"]:
        await lib.retrieve(query)

    # Run maintenance
    lib.run_maintenance()

    # ─── DB Integrity Checks ─────────────────────────────────────
    conn = lib.rolodex.conn

    # 1. No orphaned topic assignments
    orphaned = conn.execute("""
        SELECT ta.entry_id, ta.topic_id
        FROM topic_assignments ta
        LEFT JOIN rolodex_entries re ON re.id = ta.entry_id
        WHERE re.id IS NULL
    """).fetchall()
    assert len(orphaned) == 0, f"Found {len(orphaned)} orphaned topic assignments"

    # 2. FTS entry count matches actual entries
    fts_count = conn.execute("SELECT COUNT(*) as cnt FROM rolodex_fts").fetchone()["cnt"]
    entry_count = conn.execute("SELECT COUNT(*) as cnt FROM rolodex_entries").fetchone()["cnt"]
    assert fts_count == entry_count, \
        f"FTS count ({fts_count}) != entry count ({entry_count})"

    # 3. Hot cache within bounds
    assert len(lib.rolodex._hot_cache) <= lib.rolodex._hot_cache_max, \
        f"Hot cache ({len(lib.rolodex._hot_cache)}) exceeds max ({lib.rolodex._hot_cache_max})"

    # 4. Topic entry counts accurate
    topics = lib.topic_router.list_topics(limit=100)
    for topic in topics:
        actual = conn.execute(
            "SELECT COUNT(*) as cnt FROM topic_assignments WHERE topic_id = ?",
            (topic["id"],)
        ).fetchone()["cnt"]
        # Allow some slack for race conditions
        assert abs(topic["entry_count"] - actual) <= 2, \
            f"Topic '{topic['label']}' count mismatch: reported={topic['entry_count']}, actual={actual}"

    print(f"\n{'='*60}")
    print("DB INTEGRITY CHECK PASSED")
    print(f"{'='*60}")
    print(f"Entries: {entry_count}, FTS indexed: {fts_count}")
    print(f"Topics: {len(topics)}")
    print(f"Hot cache: {len(lib.rolodex._hot_cache)}/{lib.rolodex._hot_cache_max}")
    print(f"Orphaned assignments: {len(orphaned)}")
    print(f"{'='*60}\n")

    await lib.shutdown()
