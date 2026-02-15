#!/usr/bin/env python3
"""
The Librarian — Topic Migration Script (Phase 8)

Retroactively assigns topics to all existing entries using TopicRouter.
Non-destructive: logs all assignments, can be re-run safely.

Usage:
    python scripts/migrate_topics.py [--db-path PATH] [--batch-size N]
"""
import asyncio
import os
import sys
import json
import time

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from src.core.librarian import TheLibrarian
from src.indexing.topic_router import TopicRouter
from src.storage.schema import deserialize_entry


async def migrate_topics(db_path: str, batch_size: int = 50):
    """
    Scan all entries and assign topics using TopicRouter.

    Strategy:
    1. Load all entries from DB
    2. Process in batches
    3. For each entry with tags/embedding, infer topic
    4. Log assignments to topic_assignments table
    5. Report summary
    """
    print(f"Loading database from: {db_path}")
    lib = TheLibrarian(db_path=db_path)

    router = TopicRouter(
        conn=lib.rolodex.conn,
        embedding_manager=lib.embeddings,
    )

    # Get all entries
    rows = lib.rolodex.conn.execute(
        "SELECT * FROM rolodex_entries ORDER BY created_at ASC"
    ).fetchall()
    total = len(rows)
    print(f"Found {total} entries to process")

    assigned = 0
    skipped = 0
    new_topics = 0
    start_time = time.time()

    # Count existing topics before
    topics_before = router.count_topics()

    for i in range(0, total, batch_size):
        batch = rows[i:i + batch_size]
        for row in batch:
            entry = deserialize_entry(row)

            # Skip if already assigned
            if entry.metadata.get("topic_assigned"):
                skipped += 1
                continue

            topic_id = await router.infer_topic(entry)
            if topic_id:
                assigned += 1
            else:
                skipped += 1

        # Progress update
        processed = min(i + batch_size, total)
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        print(f"  Processed {processed}/{total} ({rate:.0f} entries/sec) — "
              f"{assigned} assigned, {skipped} skipped")

    # Count new topics created
    topics_after = router.count_topics()
    new_topics = topics_after - topics_before

    elapsed_total = time.time() - start_time

    summary = {
        "total_entries": total,
        "assigned": assigned,
        "skipped": skipped,
        "new_topics_created": new_topics,
        "total_topics": topics_after,
        "elapsed_seconds": round(elapsed_total, 2),
    }

    print(f"\nMigration complete:")
    print(json.dumps(summary, indent=2))

    # Show topic breakdown
    topics = router.list_topics(limit=20)
    if topics:
        print(f"\nTop topics:")
        for t in topics:
            print(f"  [{t['entry_count']:3d}] {t['label']}")

    lib.rolodex.close()
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Migrate existing entries to topics")
    parser.add_argument("--db-path", default=None,
                        help="Path to rolodex.db (default: auto-detect)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Batch size for processing (default: 50)")
    args = parser.parse_args()

    db_path = args.db_path
    if not db_path:
        # Try to find the DB
        candidates = [
            os.path.join(PARENT_DIR, "rolodex.db"),
            os.environ.get("LIBRARIAN_DB_PATH", ""),
        ]
        for c in candidates:
            if c and os.path.exists(c):
                db_path = c
                break
        if not db_path:
            print("Error: Could not find rolodex.db. Use --db-path to specify.")
            sys.exit(1)

    asyncio.run(migrate_topics(db_path, batch_size=args.batch_size))
