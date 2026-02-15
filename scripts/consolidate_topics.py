#!/usr/bin/env python3
"""
The Librarian — Topic Consolidation Script (Phase 8)

Clusters similar topics into parent groups using embedding similarity.
Creates a two-level hierarchy: parent topics (broad) → child topics (specific).

Child topics keep their entries. Parent topics aggregate for query routing.
Non-destructive: child topics remain intact, parent_topic_id links them up.

Usage:
    python scripts/consolidate_topics.py [--db-path PATH] [--threshold 0.70]
"""
import asyncio
import os
import sys
import json
import time
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

import numpy as np
from src.indexing.topic_router import TopicRouter
from src.storage.schema import deserialize_embedding, serialize_embedding
from src.indexing.embeddings import EmbeddingManager
from src.utils.config import LibrarianConfig

import sqlite3


def load_topics_with_embeddings(conn):
    """Load all topics that have embeddings."""
    rows = conn.execute(
        "SELECT * FROM topics WHERE embedding IS NOT NULL AND parent_topic_id IS NULL"
    ).fetchall()
    topics = []
    for row in rows:
        emb = deserialize_embedding(row["embedding"])
        if emb:
            topics.append({
                "id": row["id"],
                "label": row["label"],
                "description": row["description"],
                "entry_count": row["entry_count"],
                "embedding": np.array(emb, dtype=np.float32),
            })
    return topics


def cosine_similarity_matrix(topics):
    """Build NxN cosine similarity matrix from topic embeddings."""
    n = len(topics)
    embeddings = np.stack([t["embedding"] for t in topics])
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embeddings / norms
    # Pairwise cosine similarity
    sim_matrix = normed @ normed.T
    return sim_matrix


def agglomerative_cluster(topics, sim_matrix, threshold=0.70):
    """
    Simple agglomerative clustering.
    Groups topics whose pairwise similarity exceeds threshold.
    Returns list of clusters, each cluster is a list of topic indices.
    """
    n = len(topics)
    assigned = [False] * n
    clusters = []

    # Sort all pairs by similarity (descending)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((sim_matrix[i, j], i, j))
    pairs.sort(reverse=True)

    # Map from topic index → cluster index
    topic_to_cluster = {}

    for sim, i, j in pairs:
        if sim < threshold:
            break

        ci = topic_to_cluster.get(i)
        cj = topic_to_cluster.get(j)

        if ci is None and cj is None:
            # New cluster
            cluster_idx = len(clusters)
            clusters.append({i, j})
            topic_to_cluster[i] = cluster_idx
            topic_to_cluster[j] = cluster_idx
        elif ci is not None and cj is None:
            # Add j to i's cluster
            clusters[ci].add(j)
            topic_to_cluster[j] = ci
        elif ci is None and cj is not None:
            # Add i to j's cluster
            clusters[cj].add(i)
            topic_to_cluster[i] = cj
        elif ci != cj:
            # Merge two clusters
            clusters[ci].update(clusters[cj])
            for idx in clusters[cj]:
                topic_to_cluster[idx] = ci
            clusters[cj] = set()  # Empty the merged cluster

    # Add singletons (topics that didn't cluster with anything)
    for i in range(n):
        if i not in topic_to_cluster:
            clusters.append({i})

    # Filter out empty clusters
    clusters = [c for c in clusters if len(c) > 0]
    return clusters


def pick_parent_label(topics, cluster_indices):
    """
    Choose a label for the parent topic.
    Strategy: use the label of the highest-entry-count child,
    then simplify it to broader terms.
    """
    children = [topics[i] for i in cluster_indices]
    children.sort(key=lambda t: t["entry_count"], reverse=True)

    if len(children) == 1:
        return children[0]["label"], children[0]["description"]

    # Collect all unique words from child labels
    all_words = []
    for child in children:
        words = child["label"].lower().split()
        all_words.extend(words)

    # Find most common words (appear in multiple children)
    word_freq = defaultdict(int)
    for child in children:
        seen = set()
        for w in child["label"].lower().split():
            if w not in seen:
                word_freq[w] += 1
                seen.add(w)

    # Pick words that appear in at least 2 children, or top child's words
    common_words = [w for w, freq in word_freq.items() if freq >= 2]
    if not common_words:
        common_words = children[0]["label"].lower().split()[:3]

    # Cap at 3 words for readability
    label = " ".join(sorted(common_words[:3]))

    desc = f"Parent topic grouping {len(children)} subtopics: " + \
           ", ".join(c["label"] for c in children[:5])
    if len(children) > 5:
        desc += f" (+{len(children) - 5} more)"

    return label, desc


async def consolidate(db_path: str, threshold: float = 0.70):
    """
    Main consolidation pipeline:
    1. Load all leaf topics with embeddings
    2. Build similarity matrix
    3. Cluster using agglomerative approach
    4. Create parent topics for multi-member clusters
    5. Set parent_topic_id on children
    """
    print(f"Loading database: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    topics = load_topics_with_embeddings(conn)
    print(f"Loaded {len(topics)} topics with embeddings")

    if len(topics) < 2:
        print("Not enough topics to consolidate.")
        conn.close()
        return

    # Build similarity matrix
    print("Computing pairwise similarities...")
    sim_matrix = cosine_similarity_matrix(topics)

    # Cluster
    print(f"Clustering with threshold={threshold}...")
    clusters = agglomerative_cluster(topics, sim_matrix, threshold)

    multi_clusters = [c for c in clusters if len(c) > 1]
    singletons = [c for c in clusters if len(c) == 1]
    print(f"Found {len(multi_clusters)} groups, {len(singletons)} singletons")

    # Create parent topics
    router = TopicRouter(conn=conn)
    parents_created = 0
    children_linked = 0

    from datetime import datetime
    import uuid

    for cluster_indices in multi_clusters:
        label, desc = pick_parent_label(topics, cluster_indices)

        # Check if parent already exists
        existing = conn.execute(
            "SELECT id FROM topics WHERE label = ?", (label,)
        ).fetchone()
        if existing:
            parent_id = existing["id"]
        else:
            # Create parent topic with averaged embedding
            child_embeddings = [topics[i]["embedding"] for i in cluster_indices]
            avg_emb = np.mean(child_embeddings, axis=0)
            norm = np.linalg.norm(avg_emb)
            if norm > 0:
                avg_emb = avg_emb / norm
            emb_blob = serialize_embedding(avg_emb.tolist())

            parent_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            total_entries = sum(topics[i]["entry_count"] for i in cluster_indices)

            conn.execute(
                """INSERT INTO topics
                   (id, label, description, created_at, last_updated,
                    entry_count, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (parent_id, label, desc, now, now, total_entries, emb_blob)
            )
            conn.execute(
                "INSERT INTO topics_fts (topic_id, label, description) VALUES (?, ?, ?)",
                (parent_id, label, desc)
            )
            parents_created += 1

        # Link children to parent
        for idx in cluster_indices:
            child_id = topics[idx]["id"]
            conn.execute(
                "UPDATE topics SET parent_topic_id = ? WHERE id = ?",
                (parent_id, child_id)
            )
            children_linked += 1

    conn.commit()

    # Summary
    total_after = conn.execute(
        "SELECT COUNT(*) as cnt FROM topics WHERE parent_topic_id IS NULL"
    ).fetchone()["cnt"]

    summary = {
        "topics_before": len(topics),
        "clusters_found": len(multi_clusters),
        "singletons": len(singletons),
        "parent_topics_created": parents_created,
        "children_linked": children_linked,
        "top_level_topics_after": total_after,
        "threshold": threshold,
    }
    print(f"\nConsolidation complete:")
    print(json.dumps(summary, indent=2))

    # Show the hierarchy
    parents = conn.execute(
        """SELECT * FROM topics
           WHERE parent_topic_id IS NULL
           ORDER BY entry_count DESC LIMIT 30"""
    ).fetchall()
    print(f"\nTop-level topics ({total_after} total):")
    for p in parents:
        child_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM topics WHERE parent_topic_id = ?",
            (p["id"],)
        ).fetchone()["cnt"]
        suffix = f" ({child_count} subtopics)" if child_count > 0 else ""
        print(f"  [{p['entry_count']:3d}] {p['label']}{suffix}")

    conn.close()
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Consolidate topics into hierarchy")
    parser.add_argument("--db-path", default=None,
                        help="Path to rolodex.db (default: auto-detect)")
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Similarity threshold for grouping (default: 0.70)")
    args = parser.parse_args()

    db_path = args.db_path
    if not db_path:
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

    asyncio.run(consolidate(db_path, threshold=args.threshold))
