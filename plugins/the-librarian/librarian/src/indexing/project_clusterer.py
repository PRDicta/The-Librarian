"""
The Librarian — Project Clusterer (Phase 13)

Infers project-level groupings from topic co-occurrence patterns.
Topics that consistently appear together in the same sessions are
grouped into emergent "projects." Users can optionally name them.

Implements Option C (hybrid): auto-infer clusters, optional user naming.
Project clusters drive the session-focus feature — at boot, the user
picks which project/workstream they're continuing, and the manifest
biases context loading toward that cluster.

Design principles:
    - Zero-config by default (clusters emerge from usage)
    - User naming is optional and stored as user_knowledge
    - Clusters update incrementally on session close
    - Stale clusters decay naturally as topics go cold
"""
import json
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from ..core.types import estimate_tokens


class ProjectClusterer:
    """
    Infers and manages project-level groupings of topics.

    A project cluster is a set of topics that co-occur in sessions.
    The clustering algorithm:
      1. Build a topic co-occurrence matrix from session data
      2. Group topics that appear together above a threshold
      3. Merge overlapping groups into project clusters
      4. Track cluster activity (last_active, entry_count)
    """

    def __init__(
        self,
        conn,
        min_cooccurrence: int = 2,
        min_topics_per_cluster: int = 2,
    ):
        """
        Args:
            conn: SQLite connection (shared with Rolodex)
            min_cooccurrence: Minimum sessions two topics must share
                              to be considered related
            min_topics_per_cluster: Minimum topics to form a cluster
        """
        self.conn = conn
        self.min_cooccurrence = min_cooccurrence
        self.min_topics_per_cluster = min_topics_per_cluster

    # ─── Cluster Inference ────────────────────────────────────────────────

    def rebuild_clusters(self) -> List[Dict]:
        """
        Full rebuild: infer project clusters from topic co-occurrence
        across all sessions. Preserves user-named labels.

        Returns list of cluster dicts.
        """
        now = datetime.utcnow()

        # 1. Build topic-per-session matrix
        session_topics = self._get_session_topic_matrix()
        if not session_topics:
            return []

        # 2. Compute co-occurrence counts
        cooccurrence = self._compute_cooccurrence(session_topics)

        # 3. Build adjacency graph (topics connected if cooccurrence >= threshold)
        adjacency = defaultdict(set)
        for (t1, t2), count in cooccurrence.items():
            if count >= self.min_cooccurrence:
                adjacency[t1].add(t2)
                adjacency[t2].add(t1)

        # 4. Find connected components (clusters)
        visited = set()
        raw_clusters = []
        for topic_id in adjacency:
            if topic_id in visited:
                continue
            # BFS to find connected component
            component = set()
            queue = [topic_id]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            if len(component) >= self.min_topics_per_cluster:
                raw_clusters.append(component)

        # 5. Include singleton high-activity topics as their own clusters
        all_topic_ids = set()
        for topics in session_topics.values():
            all_topic_ids.update(topics)
        for topic_id in all_topic_ids:
            if topic_id not in visited:
                # Check if this topic has enough entries to stand alone
                count = self._get_topic_entry_count(topic_id)
                if count >= 5:
                    raw_clusters.append({topic_id})

        # 6. Preserve existing user-named clusters
        existing = self._get_existing_clusters()
        user_named = {c["id"]: c for c in existing if c.get("is_user_named")}

        # 7. Match new clusters to existing ones (by topic overlap)
        final_clusters = []
        used_existing = set()

        for component in raw_clusters:
            topic_ids = sorted(component)
            best_match = None
            best_overlap = 0

            for eid, ec in user_named.items():
                existing_topics = set(json.loads(ec["topic_ids"]))
                overlap = len(component & existing_topics)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = eid

            if best_match and best_overlap >= len(component) * 0.5:
                # Update existing named cluster with current topics
                ec = user_named[best_match]
                cluster_id = best_match
                label = ec["label"]
                is_user_named = True
                used_existing.add(best_match)
            else:
                # New auto-generated cluster
                cluster_id = str(uuid.uuid4())
                label = self._generate_cluster_label(topic_ids)
                is_user_named = False

            # Compute cluster stats
            last_active = self._get_cluster_last_active(topic_ids)
            entry_count = sum(self._get_topic_entry_count(tid) for tid in topic_ids)
            session_count = self._count_sessions_with_topics(topic_ids, session_topics)

            final_clusters.append({
                "id": cluster_id,
                "label": label,
                "topic_ids": topic_ids,
                "is_user_named": is_user_named,
                "last_active": last_active,
                "entry_count": entry_count,
                "session_count": session_count,
            })

        # 8. Persist
        self._write_clusters(final_clusters, now)

        return final_clusters

    def update_clusters_for_session(self, session_id: str) -> None:
        """
        Incremental update: check if the current session's topics
        affect any existing clusters. Lighter than full rebuild.
        """
        # Get topics from this session
        session_topic_ids = self._get_topics_for_session(session_id)
        if not session_topic_ids:
            return

        now = datetime.utcnow()
        clusters = self._get_existing_clusters()

        if not clusters:
            # No clusters yet — trigger full rebuild
            self.rebuild_clusters()
            return

        # Update last_active and entry_count for matching clusters
        updated = False
        for cluster in clusters:
            cluster_topics = set(json.loads(cluster["topic_ids"]))
            if cluster_topics & set(session_topic_ids):
                self.conn.execute(
                    """UPDATE project_clusters
                       SET last_active = ?,
                           session_count = session_count + 1,
                           entry_count = ?
                       WHERE id = ?""",
                    (
                        now.isoformat(),
                        sum(self._get_topic_entry_count(tid) for tid in cluster_topics),
                        cluster["id"],
                    )
                )
                updated = True

        if updated:
            self.conn.commit()

        # Check if any session topics are orphaned (not in any cluster)
        all_clustered = set()
        for c in clusters:
            all_clustered.update(json.loads(c["topic_ids"]))

        orphaned = set(session_topic_ids) - all_clustered
        if orphaned:
            # New topics appeared — trigger rebuild to re-cluster
            self.rebuild_clusters()

    # ─── Session Focus ────────────────────────────────────────────────────

    def suggest_focus(self, limit: int = 3) -> List[Dict]:
        """
        Return the top N work streams for the session-focus prompt.

        Each suggestion includes:
            - project_label: the project name (auto or user-named)
            - topic_label: the most recent active topic within that project
            - topic_id: for manifest biasing
            - cluster_id: for grouping
            - last_active: when this stream was last worked on
            - entry_count: scope indicator

        Returns them ordered: hottest first.
        """
        clusters = self._get_existing_clusters()

        if not clusters:
            # Fall back to raw topic activity if no clusters exist yet
            return self._suggest_from_raw_topics(limit)

        suggestions = []
        for c in clusters:
            topic_ids = json.loads(c["topic_ids"])
            if not topic_ids:
                continue

            # Find the most recently active topic in this cluster
            best_topic = self._get_most_active_topic(topic_ids)
            if not best_topic:
                continue

            last_active = c["last_active"] or c["created_at"]

            suggestions.append({
                "cluster_id": c["id"],
                "project_label": c["label"] or "Unnamed Project",
                "topic_label": best_topic["label"],
                "topic_id": best_topic["id"],
                "topic_ids": topic_ids,
                "last_active": last_active,
                "entry_count": c["entry_count"] or 0,
                "is_user_named": bool(c["is_user_named"]),
            })

        # Sort by last_active descending
        suggestions.sort(key=lambda x: x["last_active"] or "", reverse=True)
        return suggestions[:limit]

    def _suggest_from_raw_topics(self, limit: int = 3) -> List[Dict]:
        """
        Fallback: suggest from raw topic activity when no clusters exist.
        Picks the most recently active topics.
        """
        rows = self.conn.execute(
            """SELECT t.id, t.label, t.entry_count, t.last_updated,
                      MAX(re.created_at) AS last_entry_at
               FROM topics t
               LEFT JOIN rolodex_entries re ON re.topic_id = t.id
               WHERE t.entry_count > 0
               GROUP BY t.id
               ORDER BY last_entry_at DESC
               LIMIT ?""",
            (limit,)
        ).fetchall()

        suggestions = []
        for r in rows:
            suggestions.append({
                "cluster_id": None,
                "project_label": None,
                "topic_label": r["label"],
                "topic_id": r["id"],
                "topic_ids": [r["id"]],
                "last_active": r["last_entry_at"] or r["last_updated"],
                "entry_count": r["entry_count"],
                "is_user_named": False,
            })
        return suggestions

    # ─── User Naming ──────────────────────────────────────────────────────

    def name_cluster(self, cluster_id: str, label: str) -> bool:
        """
        Set a user-defined label for a project cluster.
        Returns True if the cluster was found and renamed.
        """
        result = self.conn.execute(
            """UPDATE project_clusters
               SET label = ?, is_user_named = 1
               WHERE id = ?""",
            (label, cluster_id)
        )
        self.conn.commit()
        return result.rowcount > 0

    def name_cluster_by_topic(self, topic_id: str, label: str) -> Optional[str]:
        """
        Name the cluster containing a given topic.
        Returns the cluster_id if found, None otherwise.
        """
        clusters = self._get_existing_clusters()
        for c in clusters:
            topic_ids = json.loads(c["topic_ids"])
            if topic_id in topic_ids:
                self.name_cluster(c["id"], label)
                return c["id"]
        return None

    # ─── Internal: Data Queries ───────────────────────────────────────────

    def _get_session_topic_matrix(self) -> Dict[str, List[str]]:
        """
        Build a mapping of session_id → list of topic_ids that appeared.
        Uses rolodex_entries joined with topic_assignments.
        """
        rows = self.conn.execute(
            """SELECT DISTINCT re.conversation_id, re.topic_id
               FROM rolodex_entries re
               WHERE re.topic_id IS NOT NULL
               AND re.topic_id != ''
               AND (re.superseded_by IS NULL OR re.superseded_by = '')"""
        ).fetchall()

        session_topics = defaultdict(set)
        for r in rows:
            session_topics[r["conversation_id"]].add(r["topic_id"])

        return {sid: list(topics) for sid, topics in session_topics.items()}

    def _compute_cooccurrence(
        self, session_topics: Dict[str, List[str]]
    ) -> Dict[Tuple[str, str], int]:
        """
        Count how many sessions each pair of topics co-occurs in.
        """
        cooccurrence = defaultdict(int)
        for _sid, topics in session_topics.items():
            topics = sorted(topics)
            for i in range(len(topics)):
                for j in range(i + 1, len(topics)):
                    pair = (topics[i], topics[j])
                    cooccurrence[pair] += 1
        return dict(cooccurrence)

    def _get_topics_for_session(self, session_id: str) -> List[str]:
        """Get unique topic IDs from entries in a session."""
        rows = self.conn.execute(
            """SELECT DISTINCT topic_id FROM rolodex_entries
               WHERE conversation_id = ?
               AND topic_id IS NOT NULL AND topic_id != ''""",
            (session_id,)
        ).fetchall()
        return [r["topic_id"] for r in rows]

    def _get_topic_entry_count(self, topic_id: str) -> int:
        """Count entries assigned to a topic."""
        row = self.conn.execute(
            "SELECT entry_count FROM topics WHERE id = ?", (topic_id,)
        ).fetchone()
        return row["entry_count"] if row else 0

    def _get_cluster_last_active(self, topic_ids: List[str]) -> Optional[str]:
        """Find the most recent entry creation time across a set of topics."""
        if not topic_ids:
            return None
        placeholders = ",".join("?" * len(topic_ids))
        row = self.conn.execute(
            f"""SELECT MAX(created_at) AS last
                FROM rolodex_entries
                WHERE topic_id IN ({placeholders})""",
            topic_ids
        ).fetchone()
        return row["last"] if row and row["last"] else None

    def _count_sessions_with_topics(
        self, topic_ids: List[str], session_topics: Dict[str, List[str]]
    ) -> int:
        """Count sessions that contain at least one of the given topics."""
        topic_set = set(topic_ids)
        count = 0
        for _sid, topics in session_topics.items():
            if topic_set & set(topics):
                count += 1
        return count

    def _get_most_active_topic(self, topic_ids: List[str]) -> Optional[Dict]:
        """
        Get the most recently active topic from a set of topic IDs.
        Returns dict with id, label, or None.
        """
        if not topic_ids:
            return None
        placeholders = ",".join("?" * len(topic_ids))
        row = self.conn.execute(
            f"""SELECT t.id, t.label,
                       MAX(re.created_at) AS last_entry
                FROM topics t
                LEFT JOIN rolodex_entries re ON re.topic_id = t.id
                WHERE t.id IN ({placeholders})
                GROUP BY t.id
                ORDER BY last_entry DESC
                LIMIT 1""",
            topic_ids
        ).fetchone()
        if row:
            return {"id": row["id"], "label": row["label"]}
        return None

    def _get_existing_clusters(self) -> List[Dict]:
        """Load all project clusters from the DB."""
        rows = self.conn.execute(
            "SELECT * FROM project_clusters ORDER BY last_active DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def _generate_cluster_label(self, topic_ids: List[str]) -> Optional[str]:
        """
        Auto-generate a cluster label from its topic labels.
        Uses the highest-entry-count topic as the primary name.
        """
        if not topic_ids:
            return None
        placeholders = ",".join("?" * len(topic_ids))
        rows = self.conn.execute(
            f"""SELECT label, entry_count FROM topics
                WHERE id IN ({placeholders})
                ORDER BY entry_count DESC""",
            topic_ids
        ).fetchall()
        if not rows:
            return None
        # Use top topic label, possibly with count suffix
        primary = rows[0]["label"]
        if len(rows) > 1:
            return f"{primary} (+{len(rows) - 1} related)"
        return primary

    def _write_clusters(self, clusters: List[Dict], now: datetime) -> None:
        """
        Persist clusters. Replaces non-user-named clusters;
        updates user-named ones in place.
        """
        # Remove non-user-named clusters (they'll be rewritten)
        self.conn.execute(
            "DELETE FROM project_clusters WHERE is_user_named = 0"
        )

        for c in clusters:
            if c["is_user_named"]:
                # Update existing user-named cluster
                self.conn.execute(
                    """UPDATE project_clusters
                       SET topic_ids = ?, last_active = ?,
                           session_count = ?, entry_count = ?
                       WHERE id = ?""",
                    (
                        json.dumps(c["topic_ids"]),
                        c["last_active"],
                        c["session_count"],
                        c["entry_count"],
                        c["id"],
                    )
                )
            else:
                # Insert new auto-generated cluster
                self.conn.execute(
                    """INSERT INTO project_clusters
                       (id, label, topic_ids, is_user_named, created_at,
                        last_active, session_count, entry_count)
                       VALUES (?, ?, ?, 0, ?, ?, ?, ?)""",
                    (
                        c["id"],
                        c["label"],
                        json.dumps(c["topic_ids"]),
                        now.isoformat(),
                        c["last_active"],
                        c["session_count"],
                        c["entry_count"],
                    )
                )

        self.conn.commit()
