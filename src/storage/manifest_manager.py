"""
The Librarian — Manifest Manager (Phase 10)

Pre-computed boot context plan. Instead of running naive keyword queries
at boot, we build a ranked manifest of entries that should be loaded,
then refine it each session based on behavioral signal.

Lifecycle:
    1. Super boot (first time / force-fresh): full census, rank, pack
    2. Session close: refine manifest with behavioral signal
    3. Incremental boot: delta update if new entries exist
    4. Instant boot: manifest is current, load directly
"""
import json
import sqlite3
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..core.types import (
    ManifestEntry, ManifestState, RolodexEntry,
    compute_importance_score, estimate_tokens,
)


class ManifestManager:
    """
    Manages the boot manifest — a pre-computed, behaviorally-refined
    context plan stored in SQLite.
    """

    def __init__(self, conn: sqlite3.Connection, rolodex):
        self.conn = conn
        self.rolodex = rolodex

    # ─── Read ─────────────────────────────────────────────────────────────

    def get_latest_manifest(self) -> Optional[ManifestState]:
        """
        Load the most recent manifest and its entries.
        Returns None if no manifest exists or all entries are stale.
        """
        row = self.conn.execute(
            "SELECT * FROM boot_manifest ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None

        manifest_id = row["id"]

        # Load manifest entries, joined with rolodex to verify existence
        entry_rows = self.conn.execute(
            """SELECT me.*, re.id AS re_id
               FROM manifest_entries me
               LEFT JOIN rolodex_entries re ON me.entry_id = re.id
               WHERE me.manifest_id = ?
               AND re.id IS NOT NULL
               AND (re.superseded_by IS NULL OR re.superseded_by = '')
               ORDER BY me.slot_rank ASC""",
            (manifest_id,)
        ).fetchall()

        entries = [
            ManifestEntry(
                entry_id=r["entry_id"],
                composite_score=r["composite_score"],
                token_cost=r["token_cost"],
                topic_label=r["topic_label"],
                selection_reason=r["selection_reason"],
                was_accessed=bool(r["was_accessed"]),
                slot_rank=r["slot_rank"],
            )
            for r in entry_rows
        ]

        return ManifestState(
            manifest_id=manifest_id,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            source_session_id=row["source_session_id"],
            manifest_type=row["manifest_type"],
            total_token_cost=row["total_token_cost"],
            entries=entries,
            topic_summary=json.loads(row["topic_summary"]) if row["topic_summary"] else {},
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    # ─── Super Boot ───────────────────────────────────────────────────────

    def build_super_manifest(self, available_budget: int) -> ManifestState:
        """
        Full census build. Scores every non-superseded entry, selects
        by topic-weighted breadth + greedy fill, packs within budget.

        Called on first boot or after invalidation.
        """
        start = time.time()
        now = datetime.utcnow()

        # 1. Census: fetch all active entries with scoring columns
        rows = self.conn.execute(
            """SELECT id, content, access_count, last_accessed, created_at,
                      tier, category, topic_id
               FROM rolodex_entries
               WHERE (superseded_by IS NULL OR superseded_by = '')
               AND category NOT IN ('user_knowledge', 'project_knowledge')"""
        ).fetchall()

        if not rows:
            return self._write_manifest([], "super", 0, {}, now, start)

        # 2. Score everything
        scored = []
        for r in rows:
            entry = _lightweight_entry(r)
            score = compute_importance_score(entry, now=now)
            token_cost = estimate_tokens(r["content"])
            topic_label = self._resolve_topic_label(r["topic_id"])
            scored.append({
                "id": r["id"],
                "score": score,
                "token_cost": token_cost,
                "topic_label": topic_label,
                "topic_id": r["topic_id"],
            })

        # 3. Topic aggregation — sum scores per topic
        topic_agg: Dict[str, float] = defaultdict(float)
        topic_entries: Dict[str, list] = defaultdict(list)
        unassigned = []

        for item in scored:
            tl = item["topic_label"]
            if tl:
                topic_agg[tl] += item["score"]
                topic_entries[tl].append(item)
            else:
                unassigned.append(item)

        # Rank topics by aggregate importance
        ranked_topics = sorted(topic_agg.items(), key=lambda x: x[1], reverse=True)

        # 4. Topic-weighted selection: best entry per top topic
        selected_ids = set()
        manifest_entries: List[ManifestEntry] = []
        budget_used = 0

        for topic_label, _agg_score in ranked_topics:
            if budget_used >= available_budget:
                break
            # Best entry in this topic
            candidates = sorted(topic_entries[topic_label], key=lambda x: x["score"], reverse=True)
            for cand in candidates:
                if cand["id"] in selected_ids:
                    continue
                if budget_used + cand["token_cost"] > available_budget:
                    continue
                manifest_entries.append(ManifestEntry(
                    entry_id=cand["id"],
                    composite_score=cand["score"],
                    token_cost=cand["token_cost"],
                    topic_label=cand["topic_label"],
                    selection_reason="topic_rep",
                    slot_rank=len(manifest_entries) + 1,
                ))
                selected_ids.add(cand["id"])
                budget_used += cand["token_cost"]
                break  # One per topic in this pass

        # 5. Greedy fill: remaining budget by raw score
        all_by_score = sorted(scored, key=lambda x: x["score"], reverse=True)
        for item in all_by_score:
            if budget_used >= available_budget:
                break
            if item["id"] in selected_ids:
                continue
            if budget_used + item["token_cost"] > available_budget:
                continue
            manifest_entries.append(ManifestEntry(
                entry_id=item["id"],
                composite_score=item["score"],
                token_cost=item["token_cost"],
                topic_label=item["topic_label"],
                selection_reason="census_rank",
                slot_rank=len(manifest_entries) + 1,
            ))
            selected_ids.add(item["id"])
            budget_used += item["token_cost"]

        # 6. Build topic summary for metadata
        topic_summary = {}
        for tl, agg in ranked_topics[:20]:
            count = len(topic_entries[tl])
            topic_summary[tl] = {"count": count, "aggregate_score": round(agg, 3)}

        return self._write_manifest(
            manifest_entries, "super", budget_used, topic_summary, now, start
        )

    # ─── Focused Boot (Phase 13) ─────────────────────────────────────────

    def build_focused_manifest(
        self, available_budget: int, focus_topic_ids: List[str] = None,
        focus_multiplier: float = 3.0,
    ) -> ManifestState:
        """
        Build a manifest biased toward a user-selected topic cluster.

        Like build_super_manifest, but applies a multiplier to scores
        of entries in the focused topic set. This concentrates the
        context window on the workstream the user chose.

        Args:
            available_budget: Token budget for the manifest.
            focus_topic_ids: Topic IDs to bias toward (from suggest-focus).
            focus_multiplier: Score multiplier for focused entries (default 3x).
        """
        start = time.time()
        now = datetime.utcnow()
        focus_set = set(focus_topic_ids or [])

        # 1. Census: same as super boot
        rows = self.conn.execute(
            """SELECT id, content, access_count, last_accessed, created_at,
                      tier, category, topic_id
               FROM rolodex_entries
               WHERE (superseded_by IS NULL OR superseded_by = '')
               AND category NOT IN ('user_knowledge', 'project_knowledge')"""
        ).fetchall()

        if not rows:
            return self._write_manifest([], "focused", 0, {}, now, start)

        # 2. Score everything, with focus bias
        scored = []
        for r in rows:
            entry = _lightweight_entry(r)
            score = compute_importance_score(entry, now=now)

            # Apply focus multiplier if entry's topic is in the focus set
            if focus_set and r["topic_id"] in focus_set:
                score *= focus_multiplier

            token_cost = estimate_tokens(r["content"])
            topic_label = self._resolve_topic_label(r["topic_id"])
            scored.append({
                "id": r["id"],
                "score": score,
                "token_cost": token_cost,
                "topic_label": topic_label,
                "topic_id": r["topic_id"],
                "is_focused": r["topic_id"] in focus_set if focus_set else False,
            })

        # 3. Topic aggregation
        topic_agg: Dict[str, float] = defaultdict(float)
        topic_entries: Dict[str, list] = defaultdict(list)

        for item in scored:
            tl = item["topic_label"]
            if tl:
                topic_agg[tl] += item["score"]
                topic_entries[tl].append(item)

        ranked_topics = sorted(topic_agg.items(), key=lambda x: x[1], reverse=True)

        # 4. Topic-weighted selection: best entry per top topic
        selected_ids = set()
        manifest_entries: List[ManifestEntry] = []
        budget_used = 0

        for topic_label, _agg_score in ranked_topics:
            if budget_used >= available_budget:
                break
            candidates = sorted(topic_entries[topic_label], key=lambda x: x["score"], reverse=True)
            for cand in candidates:
                if cand["id"] in selected_ids:
                    continue
                if budget_used + cand["token_cost"] > available_budget:
                    continue
                reason = "focus_topic_rep" if cand.get("is_focused") else "topic_rep"
                manifest_entries.append(ManifestEntry(
                    entry_id=cand["id"],
                    composite_score=cand["score"],
                    token_cost=cand["token_cost"],
                    topic_label=cand["topic_label"],
                    selection_reason=reason,
                    slot_rank=len(manifest_entries) + 1,
                ))
                selected_ids.add(cand["id"])
                budget_used += cand["token_cost"]
                break

        # 5. Greedy fill: remaining budget by raw score
        all_by_score = sorted(scored, key=lambda x: x["score"], reverse=True)
        for item in all_by_score:
            if budget_used >= available_budget:
                break
            if item["id"] in selected_ids:
                continue
            if budget_used + item["token_cost"] > available_budget:
                continue
            reason = "focus_fill" if item.get("is_focused") else "census_rank"
            manifest_entries.append(ManifestEntry(
                entry_id=item["id"],
                composite_score=item["score"],
                token_cost=item["token_cost"],
                topic_label=item["topic_label"],
                selection_reason=reason,
                slot_rank=len(manifest_entries) + 1,
            ))
            selected_ids.add(item["id"])
            budget_used += item["token_cost"]

        # 6. Topic summary
        topic_summary = {}
        for tl, agg in ranked_topics[:20]:
            count = len(topic_entries[tl])
            topic_summary[tl] = {"count": count, "aggregate_score": round(agg, 3)}

        return self._write_manifest(
            manifest_entries, "focused", budget_used, topic_summary, now, start
        )

    # ─── Incremental Boot ─────────────────────────────────────────────────

    def build_incremental_manifest(
        self, previous: ManifestState, available_budget: int
    ) -> ManifestState:
        """
        Delta update: score entries created since last manifest,
        compete against lowest-ranked existing entries.
        """
        start = time.time()
        now = datetime.utcnow()

        # Find new entries since manifest was last updated
        rows = self.conn.execute(
            """SELECT id, content, access_count, last_accessed, created_at,
                      tier, category, topic_id
               FROM rolodex_entries
               WHERE created_at > ?
               AND (superseded_by IS NULL OR superseded_by = '')
               AND category NOT IN ('user_knowledge', 'project_knowledge')""",
            (previous.updated_at.isoformat(),)
        ).fetchall()

        if not rows:
            # Nothing new — manifest is current, just bump timestamp
            self.conn.execute(
                "UPDATE boot_manifest SET updated_at = ? WHERE id = ?",
                (now.isoformat(), previous.manifest_id)
            )
            self.conn.commit()
            previous.updated_at = now
            return previous

        # Score new entries
        new_scored = []
        for r in rows:
            entry = _lightweight_entry(r)
            score = compute_importance_score(entry, now=now)
            token_cost = estimate_tokens(r["content"])
            topic_label = self._resolve_topic_label(r["topic_id"])
            new_scored.append({
                "id": r["id"],
                "score": score,
                "token_cost": token_cost,
                "topic_label": topic_label,
            })

        # Sort new by score desc
        new_scored.sort(key=lambda x: x["score"], reverse=True)

        # Copy previous entries, re-score them too
        existing = list(previous.entries)
        # Re-score existing entries against current time
        for me in existing:
            re_row = self.conn.execute(
                "SELECT access_count, last_accessed, created_at FROM rolodex_entries WHERE id = ?",
                (me.entry_id,)
            ).fetchone()
            if re_row:
                e = _lightweight_entry_from_cols(
                    re_row["access_count"],
                    re_row["last_accessed"],
                    re_row["created_at"]
                )
                me.composite_score = compute_importance_score(e, now=now)

        # Compete: for each new entry, try to evict the weakest existing
        existing.sort(key=lambda x: x.composite_score)
        budget_used = sum(me.token_cost for me in existing)

        for new_item in new_scored:
            # Can we fit it without eviction?
            if budget_used + new_item["token_cost"] <= available_budget:
                existing.append(ManifestEntry(
                    entry_id=new_item["id"],
                    composite_score=new_item["score"],
                    token_cost=new_item["token_cost"],
                    topic_label=new_item["topic_label"],
                    selection_reason="delta_promotion",
                ))
                budget_used += new_item["token_cost"]
                continue

            # Otherwise evict weakest if new is stronger
            if existing and new_item["score"] > existing[0].composite_score:
                evicted = existing.pop(0)
                budget_used -= evicted.token_cost
                if budget_used + new_item["token_cost"] <= available_budget:
                    existing.append(ManifestEntry(
                        entry_id=new_item["id"],
                        composite_score=new_item["score"],
                        token_cost=new_item["token_cost"],
                        topic_label=new_item["topic_label"],
                        selection_reason="delta_promotion",
                    ))
                    budget_used += new_item["token_cost"]
                else:
                    # Put the evicted one back — new one too big
                    existing.insert(0, evicted)
                    budget_used += evicted.token_cost

        # Re-rank by score desc
        existing.sort(key=lambda x: x.composite_score, reverse=True)
        for i, me in enumerate(existing):
            me.slot_rank = i + 1

        # Build topic summary from final set
        topic_summary = self._build_topic_summary(existing)

        return self._write_manifest(
            existing, "incremental", budget_used, topic_summary, now, start
        )

    # ─── Session Close Refinement ─────────────────────────────────────────

    def refine_manifest(
        self,
        manifest: ManifestState,
        session_id: str,
        available_budget: int,
    ) -> ManifestState:
        """
        Refine manifest based on actual session behavior.
        Called at session close.
        """
        start = time.time()
        now = datetime.utcnow()

        # 1. Access audit: which manifest entries were actually recalled?
        accessed_entry_ids = self._get_accessed_entries(session_id)

        entries = list(manifest.entries)
        for me in entries:
            if me.entry_id in accessed_entry_ids:
                me.was_accessed = True
                # Boost accessed entries
                me.composite_score *= 1.5
                me.selection_reason = "behavioral"
            else:
                # Penalize dead weight
                me.composite_score *= 0.5

        # 2. Emerged topics: entries created this session in new topics
        session_entries = self.conn.execute(
            """SELECT id, content, access_count, last_accessed, created_at,
                      tier, category, topic_id
               FROM rolodex_entries
               WHERE conversation_id = ?
               AND (superseded_by IS NULL OR superseded_by = '')
               AND category NOT IN ('user_knowledge', 'project_knowledge')""",
            (session_id,)
        ).fetchall()

        manifest_topics = {me.topic_label for me in entries if me.topic_label}
        new_candidates = []
        for r in session_entries:
            topic_label = self._resolve_topic_label(r["topic_id"])
            if topic_label and topic_label not in manifest_topics:
                entry = _lightweight_entry(r)
                score = compute_importance_score(entry, now=now)
                new_candidates.append({
                    "id": r["id"],
                    "score": score * 1.2,  # Slight boost for emerging topics
                    "token_cost": estimate_tokens(r["content"]),
                    "topic_label": topic_label,
                })

        # 3. Recall misses: queries that didn't find what they needed
        miss_entry_ids = self._get_recall_miss_entries(session_id)
        for eid in miss_entry_ids:
            if eid not in {me.entry_id for me in entries}:
                row = self.conn.execute(
                    "SELECT id, content, access_count, last_accessed, created_at, topic_id "
                    "FROM rolodex_entries WHERE id = ?",
                    (eid,)
                ).fetchone()
                if row:
                    entry = _lightweight_entry(row)
                    score = compute_importance_score(entry, now=now)
                    topic_label = self._resolve_topic_label(row["topic_id"])
                    new_candidates.append({
                        "id": row["id"],
                        "score": score,
                        "token_cost": estimate_tokens(row["content"]),
                        "topic_label": topic_label,
                    })

        # 4. Compete new candidates into the manifest
        budget_used = sum(me.token_cost for me in entries)
        new_candidates.sort(key=lambda x: x["score"], reverse=True)
        entries.sort(key=lambda x: x.composite_score)

        existing_ids = {me.entry_id for me in entries}
        for cand in new_candidates:
            if cand["id"] in existing_ids:
                continue
            # Try to fit or evict
            if budget_used + cand["token_cost"] <= available_budget:
                entries.append(ManifestEntry(
                    entry_id=cand["id"],
                    composite_score=cand["score"],
                    token_cost=cand["token_cost"],
                    topic_label=cand["topic_label"],
                    selection_reason="behavioral",
                ))
                budget_used += cand["token_cost"]
                existing_ids.add(cand["id"])
            elif entries and cand["score"] > entries[0].composite_score:
                evicted = entries.pop(0)
                budget_used -= evicted.token_cost
                existing_ids.discard(evicted.entry_id)
                if budget_used + cand["token_cost"] <= available_budget:
                    entries.append(ManifestEntry(
                        entry_id=cand["id"],
                        composite_score=cand["score"],
                        token_cost=cand["token_cost"],
                        topic_label=cand["topic_label"],
                        selection_reason="behavioral",
                    ))
                    budget_used += cand["token_cost"]
                    existing_ids.add(cand["id"])
                else:
                    entries.insert(0, evicted)
                    budget_used += evicted.token_cost
                    existing_ids.add(evicted.entry_id)

        # 5. Re-rank
        entries.sort(key=lambda x: x.composite_score, reverse=True)
        for i, me in enumerate(entries):
            me.slot_rank = i + 1

        topic_summary = self._build_topic_summary(entries)

        return self._write_manifest(
            entries, "refined", budget_used, topic_summary, now, start,
            source_session_id=session_id,
        )

    # ─── Invalidation ─────────────────────────────────────────────────────

    def invalidate(self) -> int:
        """
        Delete all manifests. Next boot will run super boot.
        Returns count of deleted manifests.
        """
        count = self.conn.execute("SELECT COUNT(*) FROM boot_manifest").fetchone()[0]
        self.conn.execute("DELETE FROM manifest_entries")
        self.conn.execute("DELETE FROM boot_manifest")
        self.conn.commit()
        return count

    # ─── Recall Tracking ──────────────────────────────────────────────────

    def mark_entry_accessed(self, manifest_id: int, entry_id: str) -> None:
        """Mark a manifest entry as accessed during this session."""
        self.conn.execute(
            "UPDATE manifest_entries SET was_accessed = 1 WHERE manifest_id = ? AND entry_id = ?",
            (manifest_id, entry_id)
        )
        self.conn.commit()

    # ─── Stats ────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return manifest system statistics."""
        manifest_count = self.conn.execute(
            "SELECT COUNT(*) FROM boot_manifest"
        ).fetchone()[0]

        latest = self.get_latest_manifest()

        stats: Dict[str, Any] = {
            "total_manifests": manifest_count,
            "has_active_manifest": latest is not None,
        }

        if latest:
            accessed = sum(1 for me in latest.entries if me.was_accessed)
            stats.update({
                "manifest_id": latest.manifest_id,
                "manifest_type": latest.manifest_type,
                "entry_count": len(latest.entries),
                "total_token_cost": latest.total_token_cost,
                "topics_represented": len(latest.topic_summary),
                "entries_accessed": accessed,
                "access_rate": round(accessed / max(1, len(latest.entries)), 3),
                "created_at": latest.created_at.isoformat(),
                "updated_at": latest.updated_at.isoformat(),
                "source_session": latest.source_session_id,
            })

        return stats

    # ─── Count New Entries ────────────────────────────────────────────────

    def count_entries_after(self, after: datetime) -> int:
        """Count entries created after the given timestamp."""
        row = self.conn.execute(
            """SELECT COUNT(*) FROM rolodex_entries
               WHERE created_at > ?
               AND (superseded_by IS NULL OR superseded_by = '')
               AND category NOT IN ('user_knowledge', 'project_knowledge')""",
            (after.isoformat(),)
        ).fetchone()
        return row[0] if row else 0

    # ─── Internal ─────────────────────────────────────────────────────────

    def _write_manifest(
        self,
        entries: List[ManifestEntry],
        manifest_type: str,
        budget_used: int,
        topic_summary: Dict,
        now: datetime,
        start_time: float,
        source_session_id: Optional[str] = None,
    ) -> ManifestState:
        """Persist a new manifest to the DB."""
        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        metadata = {
            "build_time_ms": elapsed_ms,
            "entries_considered": len(entries),
        }

        cursor = self.conn.execute(
            """INSERT INTO boot_manifest
               (created_at, updated_at, source_session_id, manifest_type,
                total_token_cost, entry_count, topic_summary, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now.isoformat(), now.isoformat(), source_session_id,
                manifest_type, budget_used, len(entries),
                json.dumps(topic_summary), json.dumps(metadata),
            )
        )
        manifest_id = cursor.lastrowid

        # Write entries
        for me in entries:
            self.conn.execute(
                """INSERT INTO manifest_entries
                   (manifest_id, entry_id, composite_score, token_cost,
                    topic_label, selection_reason, was_accessed, slot_rank)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    manifest_id, me.entry_id, me.composite_score,
                    me.token_cost, me.topic_label, me.selection_reason,
                    int(me.was_accessed), me.slot_rank,
                )
            )

        self.conn.commit()

        return ManifestState(
            manifest_id=manifest_id,
            created_at=now,
            updated_at=now,
            source_session_id=source_session_id,
            manifest_type=manifest_type,
            total_token_cost=budget_used,
            entries=entries,
            topic_summary=topic_summary,
            metadata=metadata,
        )

    def _resolve_topic_label(self, topic_id: Optional[str]) -> Optional[str]:
        """Resolve a topic_id to its label."""
        if not topic_id:
            return None
        row = self.conn.execute(
            "SELECT label FROM topics WHERE id = ?", (topic_id,)
        ).fetchone()
        return row["label"] if row else None

    def _get_accessed_entries(self, session_id: str) -> set:
        """Get entry IDs that were returned in queries during this session."""
        rows = self.conn.execute(
            """SELECT entry_ids FROM query_log
               WHERE conversation_id = ? AND found = 1""",
            (session_id,)
        ).fetchall()
        accessed = set()
        for row in rows:
            ids = json.loads(row["entry_ids"]) if row["entry_ids"] else []
            accessed.update(ids)
        return accessed

    def _get_recall_miss_entries(self, session_id: str) -> set:
        """
        Get entry IDs from queries that found results NOT in the manifest.
        These represent gaps — things the user needed that weren't preloaded.
        """
        rows = self.conn.execute(
            """SELECT entry_ids FROM query_log
               WHERE conversation_id = ? AND found = 1""",
            (session_id,)
        ).fetchall()
        all_found = set()
        for row in rows:
            ids = json.loads(row["entry_ids"]) if row["entry_ids"] else []
            all_found.update(ids)
        return all_found

    def _build_topic_summary(self, entries: List[ManifestEntry]) -> Dict:
        """Build topic summary from a set of manifest entries."""
        summary: Dict[str, Any] = defaultdict(lambda: {"count": 0, "aggregate_score": 0.0})
        for me in entries:
            if me.topic_label:
                summary[me.topic_label]["count"] += 1
                summary[me.topic_label]["aggregate_score"] += me.composite_score
        # Round scores
        for v in summary.values():
            v["aggregate_score"] = round(v["aggregate_score"], 3)
        return dict(summary)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _lightweight_entry(row) -> RolodexEntry:
    """
    Build a minimal RolodexEntry from a DB row — just enough
    for compute_importance_score. Avoids full deserialization overhead.
    """
    return RolodexEntry(
        id=row["id"],
        access_count=row["access_count"],
        last_accessed=(
            datetime.fromisoformat(row["last_accessed"])
            if row["last_accessed"] else None
        ),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _lightweight_entry_from_cols(
    access_count: int,
    last_accessed: Optional[str],
    created_at: str,
) -> RolodexEntry:
    """Build minimal RolodexEntry from individual columns."""
    return RolodexEntry(
        access_count=access_count,
        last_accessed=(
            datetime.fromisoformat(last_accessed)
            if last_accessed else None
        ),
        created_at=datetime.fromisoformat(created_at),
    )
