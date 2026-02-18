"""
The Librarian — Maintenance Engine

Background hygiene passes that improve the knowledge graph during idle time.
Called by the `maintain` CLI command when the user is inactive.

Passes:
    1. Contradiction detection — find entries on same topic with conflicting claims
    2. Orphaned correction linking — find CORRECTION entries not linked via superseded_by
    3. Near-duplicate merging — find entries saying the same thing in different words
    4. Entry promotion — promote high-value cold entries to user_knowledge
    5. Stale temporal flagging — flag entries with time-sensitive claims that may be outdated
    6. Compression learning — analyze codebook patterns, score confidence, promote stages

Token budget: each pass has a budget. The engine stops when the total budget is exhausted.
Cooldown: won't run if the last maintenance completed within `cooldown_hours`.
"""

import json
import os
import sqlite3
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .types import EntryCategory, CompressionStage, RolodexEntry, Tier, estimate_tokens
from ..storage.schema import deserialize_entry, ensure_codebook_schema


# ─── Maintenance Log Schema ──────────────────────────────────────────────────

MAINTENANCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS maintenance_log (
    id TEXT PRIMARY KEY,
    started_at DATETIME NOT NULL,
    completed_at DATETIME,
    session_id TEXT,
    passes_run TEXT NOT NULL DEFAULT '[]',
    actions_taken INTEGER DEFAULT 0,
    entries_scanned INTEGER DEFAULT 0,
    contradictions_found INTEGER DEFAULT 0,
    orphans_linked INTEGER DEFAULT 0,
    duplicates_merged INTEGER DEFAULT 0,
    entries_promoted INTEGER DEFAULT 0,
    stale_flagged INTEGER DEFAULT 0,
    compressions_learned INTEGER DEFAULT 0,
    token_budget INTEGER DEFAULT 0,
    tokens_used INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_maintenance_completed
    ON maintenance_log(completed_at DESC);
"""


def ensure_maintenance_schema(conn: sqlite3.Connection):
    """Create the maintenance_log table if it doesn't exist."""
    conn.executescript(MAINTENANCE_SCHEMA)
    # Safe column addition for existing databases
    try:
        conn.execute("ALTER TABLE maintenance_log ADD COLUMN compressions_learned INTEGER DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.commit()


# ─── Pulse (Heartbeat) ──────────────────────────────────────────────────────

def pulse_check(conn: sqlite3.Connection, session_file_path: str) -> Dict[str, Any]:
    """
    Lightweight heartbeat check. Returns status of The Librarian.

    Checks:
        1. DB is readable
        2. Session file exists and has a valid session_id
        3. The session_id corresponds to an active conversation

    Returns a dict with:
        alive: bool
        session_id: str or None
        needs_boot: bool
        entry_count: int
        last_entry_at: str or None
    """
    import os

    result = {
        "alive": False,
        "session_id": None,
        "needs_boot": True,
        "entry_count": 0,
        "last_entry_at": None,
    }

    # Check DB readable
    try:
        row = conn.execute("SELECT COUNT(*) as cnt FROM rolodex_entries WHERE superseded_by IS NULL").fetchone()
        result["entry_count"] = row["cnt"] if row else 0
    except Exception:
        return result

    # Check last entry timestamp
    try:
        row = conn.execute("SELECT MAX(created_at) as latest FROM rolodex_entries").fetchone()
        result["last_entry_at"] = row["latest"] if row else None
    except Exception:
        pass

    # Check session file
    if os.path.exists(session_file_path):
        try:
            with open(session_file_path, "r") as f:
                data = json.load(f)
            sid = data.get("session_id")
            if sid:
                result["session_id"] = sid
                # Verify session exists in DB
                row = conn.execute(
                    "SELECT id FROM conversations WHERE id = ?", (sid,)
                ).fetchone()
                if row:
                    result["alive"] = True
                    result["needs_boot"] = False
        except (json.JSONDecodeError, IOError):
            pass

    return result


# ─── Cooldown Check ──────────────────────────────────────────────────────────

def check_cooldown(
    conn: sqlite3.Connection, cooldown_hours: float = 4.0
) -> Tuple[bool, Optional[str]]:
    """
    Check if enough time has passed since the last maintenance run.

    Returns (can_run: bool, last_completed_at: str or None).
    """
    ensure_maintenance_schema(conn)

    row = conn.execute(
        "SELECT completed_at FROM maintenance_log WHERE completed_at IS NOT NULL ORDER BY completed_at DESC LIMIT 1"
    ).fetchone()

    if not row or not row["completed_at"]:
        return True, None

    last_completed = datetime.fromisoformat(row["completed_at"])
    cutoff = datetime.utcnow() - timedelta(hours=cooldown_hours)

    return last_completed < cutoff, row["completed_at"]


# ─── Core Maintenance Engine ─────────────────────────────────────────────────

class MaintenanceEngine:
    """
    Runs structured hygiene passes over the knowledge graph.

    Usage:
        engine = MaintenanceEngine(conn, session_id, token_budget=15000)
        report = engine.run_all()
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        session_id: Optional[str] = None,
        token_budget: int = 15000,
        max_entries_per_pass: int = 200,
        stale_threshold_hours: float = 48.0,
        similarity_threshold: float = 0.85,
    ):
        self.conn = conn
        self.session_id = session_id
        self.token_budget = token_budget
        self.tokens_used = 0
        self.max_entries_per_pass = max_entries_per_pass
        self.stale_threshold_hours = stale_threshold_hours
        self.similarity_threshold = similarity_threshold

        # Counters
        self.contradictions_found = 0
        self.orphans_linked = 0
        self.duplicates_merged = 0
        self.entries_promoted = 0
        self.stale_flagged = 0
        self.compressions_learned = 0
        self.entries_scanned = 0
        self.actions: List[Dict[str, Any]] = []
        self.passes_run: List[str] = []

    def _budget_remaining(self) -> int:
        return max(0, self.token_budget - self.tokens_used)

    def _charge_tokens(self, text: str) -> int:
        cost = estimate_tokens(text)
        self.tokens_used += cost
        return cost

    def _get_active_entries(
        self,
        limit: int = 500,
        recent_first: bool = True,
        category_filter: Optional[str] = None,
    ) -> List[RolodexEntry]:
        """Fetch non-superseded entries, optionally filtered."""
        sql = "SELECT * FROM rolodex_entries WHERE superseded_by IS NULL"
        params: list = []
        if category_filter:
            sql += " AND category = ?"
            params.append(category_filter)
        if recent_first:
            sql += " ORDER BY created_at DESC"
        else:
            sql += " ORDER BY created_at ASC"
        sql += " LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        return [deserialize_entry(row) for row in rows]

    def _get_entries_by_topic(self, topic_id: str, limit: int = 50) -> List[RolodexEntry]:
        """Fetch active entries for a specific topic."""
        rows = self.conn.execute(
            """SELECT * FROM rolodex_entries
               WHERE topic_id = ? AND superseded_by IS NULL
               ORDER BY created_at DESC LIMIT ?""",
            (topic_id, limit)
        ).fetchall()
        return [deserialize_entry(row) for row in rows]

    def _supersede(self, old_id: str, new_id: str):
        """Mark an entry as superseded and remove from FTS."""
        self.conn.execute(
            "UPDATE rolodex_entries SET superseded_by = ? WHERE id = ?",
            (new_id, old_id)
        )
        self.conn.execute(
            "DELETE FROM rolodex_fts WHERE entry_id = ?", (old_id,)
        )

    def _flag_entry(self, entry_id: str, flag_type: str, detail: str):
        """Add a maintenance flag to an entry's metadata."""
        row = self.conn.execute(
            "SELECT metadata FROM rolodex_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        if row:
            meta = json.loads(row["metadata"] or "{}")
            if "maintenance_flags" not in meta:
                meta["maintenance_flags"] = []
            meta["maintenance_flags"].append({
                "type": flag_type,
                "detail": detail,
                "flagged_at": datetime.utcnow().isoformat(),
            })
            self.conn.execute(
                "UPDATE rolodex_entries SET metadata = ? WHERE id = ?",
                (json.dumps(meta), entry_id)
            )

    # ─── Pass 1: Contradiction Detection ─────────────────────────────────

    def pass_contradiction_detection(self) -> int:
        """
        Find entries within the same topic that contain conflicting claims.

        Strategy: group entries by topic, look for numeric claims or
        status assertions that contradict each other. When found, supersede
        the older entry with the newer one.

        Returns number of contradictions resolved.
        """
        self.passes_run.append("contradiction_detection")
        resolved = 0

        # Get all topics with more than 1 entry
        topics = self.conn.execute(
            """SELECT id, label FROM topics WHERE entry_count > 1
               ORDER BY entry_count DESC LIMIT ?""",
            (self.max_entries_per_pass,)
        ).fetchall()

        for topic_row in topics:
            if self._budget_remaining() <= 0:
                break

            topic_id = topic_row["id"]
            topic_label = topic_row["label"]
            entries = self._get_entries_by_topic(topic_id, limit=30)
            self.entries_scanned += len(entries)

            # Look for numeric percentage claims that conflict
            resolved += self._detect_numeric_contradictions(entries, topic_label)

            # Look for status/state claims that conflict
            resolved += self._detect_status_contradictions(entries, topic_label)

        # Also scan entries without topics — group by tag overlap
        resolved += self._detect_unassigned_contradictions()

        self.contradictions_found = resolved
        return resolved

    def _detect_numeric_contradictions(
        self, entries: List[RolodexEntry], topic_label: str
    ) -> int:
        """Find entries with conflicting numeric claims (percentages, counts, etc.)."""
        import re

        resolved = 0
        # Extract entries with percentage claims
        pct_entries: List[Tuple[RolodexEntry, List[Tuple[str, float]]]] = []

        for entry in entries:
            self._charge_tokens(entry.content)
            # Find percentage patterns: "40%", "~40%", "at 40%", etc.
            pct_matches = re.findall(
                r'(?:~|about |approximately |around |at |to |revised to )?(\d+(?:\.\d+)?)\s*[%％]',
                entry.content
            )
            if pct_matches:
                # Also extract context words around each percentage
                context_pcts = []
                for m in re.finditer(
                    r'(\w+(?:\s+\w+){0,3})\s+(?:~|about |approximately |around |at |to |revised to )?(\d+(?:\.\d+)?)\s*[%％]',
                    entry.content
                ):
                    context = m.group(1).lower()
                    value = float(m.group(2))
                    context_pcts.append((context, value))
                if context_pcts:
                    pct_entries.append((entry, context_pcts))

        # Compare pairs for contradictions
        for i, (entry_a, pcts_a) in enumerate(pct_entries):
            for j, (entry_b, pcts_b) in enumerate(pct_entries):
                if j <= i:
                    continue
                if self._budget_remaining() <= 0:
                    return resolved

                # Check if they share context words but differ in value
                for ctx_a, val_a in pcts_a:
                    for ctx_b, val_b in pcts_b:
                        # Shared context words (at least 1 word overlap)
                        words_a = set(ctx_a.split())
                        words_b = set(ctx_b.split())
                        overlap = words_a & words_b

                        if overlap and abs(val_a - val_b) > 5:
                            # Contradiction found — supersede the older one
                            older = entry_a if entry_a.created_at < entry_b.created_at else entry_b
                            newer = entry_b if older is entry_a else entry_a

                            self._supersede(older.id, newer.id)
                            self.actions.append({
                                "type": "contradiction_resolved",
                                "old_id": older.id,
                                "new_id": newer.id,
                                "topic": topic_label,
                                "detail": f"Conflicting values: {val_a}% vs {val_b}% in context '{overlap}'",
                            })
                            resolved += 1
                            break
                    else:
                        continue
                    break

        return resolved

    def _detect_status_contradictions(
        self, entries: List[RolodexEntry], topic_label: str
    ) -> int:
        """Find entries with conflicting status/state claims."""
        import re

        resolved = 0
        # Look for "X is Y" / "X was Y" patterns that contradict
        status_patterns = [
            r'(?:is|was|are|were)\s+(?:now\s+)?(?:at\s+)?(\d+(?:\.\d+)?)\s*[%％]',
            r'(?:rated|revised|updated|changed)\s+(?:to|at)\s+(\d+(?:\.\d+)?)\s*[%％]',
            r'(?:complete|done|finished|ready)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*[%％]',
        ]

        rated_entries: List[Tuple[RolodexEntry, float]] = []
        for entry in entries:
            for pattern in status_patterns:
                m = re.search(pattern, entry.content, re.IGNORECASE)
                if m:
                    rated_entries.append((entry, float(m.group(1))))
                    break

        # If multiple entries rate the same topic differently, keep newest
        if len(rated_entries) > 1:
            # Sort by created_at ascending (oldest first)
            rated_entries.sort(key=lambda x: x[0].created_at)

            # Supersede all but the most recent
            newest = rated_entries[-1]
            for entry, val in rated_entries[:-1]:
                if abs(val - newest[1]) > 5:  # Only if actually different
                    self._supersede(entry.id, newest[0].id)
                    self.actions.append({
                        "type": "contradiction_resolved",
                        "old_id": entry.id,
                        "new_id": newest[0].id,
                        "topic": topic_label,
                        "detail": f"Stale rating {val}% superseded by {newest[1]}%",
                    })
                    resolved += 1

        return resolved

    def _detect_unassigned_contradictions(self) -> int:
        """Check for contradictions among entries without topic assignments."""
        resolved = 0

        # Get CORRECTION entries that might indicate unresolved conflicts
        corrections = self._get_active_entries(
            limit=50, category_filter="correction"
        )

        for correction in corrections:
            if self._budget_remaining() <= 0:
                break
            self._charge_tokens(correction.content)
            self.entries_scanned += 1

            # Search for entries this correction might be correcting
            # by looking for keyword overlap in non-correction entries
            keywords = self._extract_keywords(correction.content)
            if not keywords:
                continue

            # FTS search for matching entries
            try:
                fts_query = " OR ".join(keywords[:5])
                rows = self.conn.execute(
                    """SELECT re.* FROM rolodex_fts fts
                       JOIN rolodex_entries re ON re.id = fts.entry_id
                       WHERE rolodex_fts MATCH ?
                       AND re.category != 'correction'
                       AND re.superseded_by IS NULL
                       ORDER BY fts.rank LIMIT 5""",
                    (fts_query,)
                ).fetchall()

                for row in rows:
                    candidate = deserialize_entry(row)
                    # If the correction is newer and the candidate is older,
                    # check for factual conflict
                    if (correction.created_at > candidate.created_at
                        and self._content_conflicts(correction.content, candidate.content)):
                        self._supersede(candidate.id, correction.id)
                        self.actions.append({
                            "type": "contradiction_resolved",
                            "old_id": candidate.id,
                            "new_id": correction.id,
                            "detail": "Correction entry found for uncorrected original",
                        })
                        resolved += 1
                        break
            except Exception:
                continue

        return resolved

    def _content_conflicts(self, correction_text: str, original_text: str) -> bool:
        """Heuristic check if a correction entry contradicts an original.

        Looks for patterns like "not X — Y" or "revised to Y" in the correction
        that directly contradict claims in the original.
        """
        import re

        # Look for "not X" patterns in correction
        not_patterns = re.findall(
            r'(?:not|no longer|wrong|incorrect|revised|corrected|updated)\s+(.+?)(?:\.|—|,|\s+but)',
            correction_text, re.IGNORECASE
        )

        for negated in not_patterns:
            negated_lower = negated.lower().strip()
            if len(negated_lower) > 3 and negated_lower in original_text.lower():
                return True

        return False

    def _extract_keywords(self, text: str, max_keywords: int = 8) -> List[str]:
        """Extract significant keywords from text for FTS matching."""
        import re

        stopwords = {
            'the', 'a', 'an', 'is', 'was', 'are', 'were', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'shall', 'to', 'of',
            'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'only', 'own', 'same', 'than', 'too', 'very', 'just',
            'that', 'this', 'these', 'those', 'it', 'its', 'they', 'them',
            'their', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she',
            'her', 'who', 'which', 'what', 'when', 'where', 'how', 'why',
        }

        words = re.findall(r'[a-zA-Z_]\w{2,}', text.lower())
        meaningful = [w for w in words if w not in stopwords and len(w) > 2]

        # Prioritize less common words (appear fewer times = more distinctive)
        freq = defaultdict(int)
        for w in meaningful:
            freq[w] += 1

        # Sort by frequency ascending (rarest first), then take top N
        unique = sorted(set(meaningful), key=lambda w: freq[w])
        return unique[:max_keywords]

    # ─── Pass 2: Orphaned Correction Linking ─────────────────────────────

    def pass_orphaned_corrections(self) -> int:
        """
        Find CORRECTION category entries that don't have a superseded_by link
        to the entry they're correcting. Attempt to auto-link them.

        Returns number of orphans successfully linked.
        """
        self.passes_run.append("orphaned_corrections")
        linked = 0

        # Get all CORRECTION entries
        corrections = self._get_active_entries(
            limit=self.max_entries_per_pass,
            category_filter="correction",
        )

        for correction in corrections:
            if self._budget_remaining() <= 0:
                break

            self._charge_tokens(correction.content)
            self.entries_scanned += 1

            # Check if this correction already supersedes something
            # (it's the new_id in some superseded_by chain)
            row = self.conn.execute(
                "SELECT id FROM rolodex_entries WHERE superseded_by = ?",
                (correction.id,)
            ).fetchone()
            if row:
                continue  # Already linked — skip

            # Try to find what this correction is correcting
            keywords = self._extract_keywords(correction.content)
            if not keywords:
                continue

            try:
                fts_query = " OR ".join(keywords[:5])
                candidates = self.conn.execute(
                    """SELECT re.*, fts.rank FROM rolodex_fts fts
                       JOIN rolodex_entries re ON re.id = fts.entry_id
                       WHERE rolodex_fts MATCH ?
                       AND re.id != ?
                       AND re.category != 'correction'
                       AND re.superseded_by IS NULL
                       AND re.created_at < ?
                       ORDER BY fts.rank LIMIT 5""",
                    (fts_query, correction.id, correction.created_at.isoformat())
                ).fetchall()

                for candidate_row in candidates:
                    candidate = deserialize_entry(candidate_row)
                    if self._content_conflicts(correction.content, candidate.content):
                        self._supersede(candidate.id, correction.id)
                        self.actions.append({
                            "type": "orphan_linked",
                            "correction_id": correction.id,
                            "original_id": candidate.id,
                            "detail": f"Linked orphaned correction to original entry",
                        })
                        linked += 1
                        break
            except Exception:
                continue

        self.orphans_linked = linked
        return linked

    # ─── Pass 3: Near-Duplicate Merging ──────────────────────────────────

    def pass_near_duplicate_merging(self) -> int:
        """
        Find entries that say nearly the same thing and merge them by
        superseding the older/shorter one with the newer/longer one.

        Uses token-level Jaccard similarity on recent entries.

        Returns number of duplicates merged.
        """
        self.passes_run.append("near_duplicate_merging")
        merged = 0

        # Focus on recent entries (most likely to have duplicates from
        # re-ingestion across sessions)
        entries = self._get_active_entries(
            limit=self.max_entries_per_pass,
            recent_first=True,
        )

        # Build token sets for comparison
        entry_tokens: List[Tuple[RolodexEntry, set]] = []
        for entry in entries:
            if self._budget_remaining() <= 0:
                break
            self._charge_tokens(entry.content)
            self.entries_scanned += 1
            tokens = set(entry.content.lower().split())
            if len(tokens) >= 5:  # Skip very short entries
                entry_tokens.append((entry, tokens))

        # Compare pairs (O(n^2) but limited by max_entries_per_pass)
        superseded_ids = set()
        for i, (entry_a, tokens_a) in enumerate(entry_tokens):
            if entry_a.id in superseded_ids:
                continue
            for j, (entry_b, tokens_b) in enumerate(entry_tokens):
                if j <= i or entry_b.id in superseded_ids:
                    continue
                if self._budget_remaining() <= 0:
                    break

                # Jaccard similarity
                intersection = len(tokens_a & tokens_b)
                union = len(tokens_a | tokens_b)
                if union == 0:
                    continue
                similarity = intersection / union

                if similarity >= self.similarity_threshold:
                    # Keep the longer/newer one
                    if len(entry_b.content) >= len(entry_a.content):
                        keep, drop = entry_b, entry_a
                    else:
                        keep, drop = entry_a, entry_b

                    # Don't merge if they're different categories (intentional)
                    if keep.category != drop.category:
                        continue

                    # Don't merge user_knowledge — those are curated
                    if keep.category == EntryCategory.USER_KNOWLEDGE:
                        continue

                    self._supersede(drop.id, keep.id)
                    superseded_ids.add(drop.id)
                    self.actions.append({
                        "type": "duplicate_merged",
                        "kept_id": keep.id,
                        "dropped_id": drop.id,
                        "similarity": round(similarity, 3),
                        "detail": f"Jaccard {similarity:.0%} overlap",
                    })
                    merged += 1

        self.duplicates_merged = merged
        return merged

    # ─── Pass 4: Entry Promotion ─────────────────────────────────────────

    def pass_entry_promotion(self) -> int:
        """
        Identify cold entries that are frequently accessed or contain
        high-value content and promote them.

        Promotion criteria:
            - access_count >= 5 (frequently recalled)
            - Category is fact, definition, or decision (high-value types)
            - Not already user_knowledge

        Returns number of entries promoted.
        """
        self.passes_run.append("entry_promotion")
        promoted = 0

        # Find frequently accessed cold entries
        rows = self.conn.execute(
            """SELECT * FROM rolodex_entries
               WHERE tier = 'cold'
               AND superseded_by IS NULL
               AND category NOT IN ('user_knowledge', 'correction')
               AND access_count >= 5
               ORDER BY access_count DESC
               LIMIT ?""",
            (self.max_entries_per_pass,)
        ).fetchall()

        for row in rows:
            if self._budget_remaining() <= 0:
                break

            entry = deserialize_entry(row)
            self._charge_tokens(entry.content)
            self.entries_scanned += 1

            # Promote to HOT tier
            self.conn.execute(
                "UPDATE rolodex_entries SET tier = 'hot' WHERE id = ?",
                (entry.id,)
            )
            self.actions.append({
                "type": "promoted",
                "entry_id": entry.id,
                "access_count": entry.access_count,
                "category": entry.category.value,
                "detail": f"Promoted to HOT (access_count={entry.access_count})",
            })
            promoted += 1

        # Also look for high-value categories that should be hot
        high_value_categories = ['fact', 'decision', 'definition']
        for cat in high_value_categories:
            rows = self.conn.execute(
                """SELECT * FROM rolodex_entries
                   WHERE tier = 'cold'
                   AND superseded_by IS NULL
                   AND category = ?
                   AND access_count >= 3
                   ORDER BY access_count DESC
                   LIMIT 10""",
                (cat,)
            ).fetchall()

            for row in rows:
                if self._budget_remaining() <= 0:
                    break

                entry = deserialize_entry(row)
                self._charge_tokens(entry.content)
                self.entries_scanned += 1

                self.conn.execute(
                    "UPDATE rolodex_entries SET tier = 'hot' WHERE id = ?",
                    (entry.id,)
                )
                self.actions.append({
                    "type": "promoted",
                    "entry_id": entry.id,
                    "access_count": entry.access_count,
                    "category": entry.category.value,
                    "detail": f"High-value {cat} promoted to HOT",
                })
                promoted += 1

        self.entries_promoted = promoted
        return promoted

    # ─── Pass 5: Stale Temporal Flagging ─────────────────────────────────

    def pass_stale_temporal_flagging(self) -> int:
        """
        Flag entries that contain time-sensitive claims and are older
        than the stale threshold.

        Looks for patterns like:
            - "currently", "right now", "as of today"
            - "X is at Y%" (status assertions)
            - "we just", "we recently"

        Flags entries with metadata rather than superseding them —
        the assistant should verify these on next recall.

        Returns number of entries flagged.
        """
        import re

        self.passes_run.append("stale_temporal_flagging")
        flagged = 0

        cutoff = datetime.utcnow() - timedelta(hours=self.stale_threshold_hours)

        # Temporal indicator patterns
        temporal_patterns = [
            r'\bcurrently\b',
            r'\bright now\b',
            r'\bas of today\b',
            r'\bas of now\b',
            r'\bat the moment\b',
            r'\bwe just\b',
            r'\bwe recently\b',
            r'\bjust (?:finished|completed|shipped|launched|built|deployed)\b',
            r'\b(?:is|are) (?:now|currently) (?:at|running|using|on)\b',
            r'\btoday(?:\'s| we)\b',
            r'\bthis (?:morning|afternoon|evening|session)\b',
        ]

        # Get older entries that haven't been flagged yet
        rows = self.conn.execute(
            """SELECT * FROM rolodex_entries
               WHERE superseded_by IS NULL
               AND created_at < ?
               AND category NOT IN ('user_knowledge')
               ORDER BY created_at DESC
               LIMIT ?""",
            (cutoff.isoformat(), self.max_entries_per_pass)
        ).fetchall()

        for row in rows:
            if self._budget_remaining() <= 0:
                break

            entry = deserialize_entry(row)

            # Skip if already flagged
            meta = entry.metadata or {}
            existing_flags = meta.get("maintenance_flags", [])
            if any(f.get("type") == "stale_temporal" for f in existing_flags):
                continue

            self._charge_tokens(entry.content)
            self.entries_scanned += 1

            # Check for temporal language
            for pattern in temporal_patterns:
                if re.search(pattern, entry.content, re.IGNORECASE):
                    self._flag_entry(entry.id, "stale_temporal",
                                   f"Contains temporal language, created {entry.created_at.isoformat()}")
                    self.actions.append({
                        "type": "stale_flagged",
                        "entry_id": entry.id,
                        "age_hours": round((datetime.utcnow() - entry.created_at).total_seconds() / 3600, 1),
                        "pattern_matched": pattern,
                        "detail": f"Temporal claim may be outdated",
                    })
                    flagged += 1
                    break

        self.stale_flagged = flagged
        return flagged

    # ─── Pass 6: Compression Learning ───────────────────────────────────

    def pass_compression_learning(
        self,
        warm_threshold: int = 3,
        hot_threshold: int = 8,
        warm_confidence: float = 0.8,
        hot_confidence: float = 0.9,
        vocab_pack_dir: Optional[str] = None,
    ) -> int:
        """Analyze codebook patterns, update confidence scores, promote stages.

        Three sub-steps:
        1. Confidence scoring — based on times_seen and stability
        2. Stage promotion — COLD→WARM at threshold, WARM→HOT at higher threshold
        3. Vocab pack generation — write learned.json for boot-time loading

        Args:
            warm_threshold: times_seen required for COLD→WARM promotion
            hot_threshold: times_seen required for WARM→HOT promotion
            warm_confidence: minimum confidence for COLD→WARM
            hot_confidence: minimum confidence for WARM→HOT
            vocab_pack_dir: directory to write learned.json (auto-detected if None)

        Returns:
            Number of patterns promoted or updated
        """
        self.passes_run.append("compression_learning")
        ensure_codebook_schema(self.conn)

        if self._budget_remaining() <= 0:
            return 0

        now = datetime.utcnow().isoformat()
        learned = 0

        # ── Step 1: Confidence scoring ────────────────────────────────────
        rows = self.conn.execute(
            """SELECT id, pattern_text, warm_form, hot_form, stage,
                      token_cost_original, token_cost_warm, token_cost_hot,
                      times_seen, confidence, first_seen_at, last_seen_at,
                      promoted_at, metadata
               FROM compression_codebook
               ORDER BY times_seen DESC"""
        ).fetchall()

        self._charge_tokens(f"codebook_scan_{len(rows)}")

        for row in rows:
            if self._budget_remaining() <= 0:
                break

            entry_id = row["id"]
            stage = row["stage"]
            times_seen = row["times_seen"]
            old_confidence = row["confidence"]
            meta = json.loads(row["metadata"]) if row["metadata"] else {}

            # Determine promotion target and threshold
            if stage == CompressionStage.COLD.value:
                target_threshold = warm_threshold
            elif stage == CompressionStage.WARM.value:
                target_threshold = hot_threshold
            else:
                continue  # Already HOT, nothing to do

            # Stability factor: check if pattern_text has been consistent
            # (metadata can track variation count; default to stable)
            variation_count = meta.get("variation_count", 0)
            stability = 1.0 if variation_count <= 1 else max(0.3, 1.0 - (variation_count * 0.15))

            # Compute confidence
            raw_confidence = min(1.0, times_seen / target_threshold)
            confidence = round(raw_confidence * stability, 3)

            # Update confidence if changed
            if abs(confidence - old_confidence) > 0.001:
                self.conn.execute(
                    "UPDATE compression_codebook SET confidence = ? WHERE id = ?",
                    (confidence, entry_id)
                )

            # ── Step 2: Stage promotion ───────────────────────────────────
            promoted = False

            if stage == CompressionStage.COLD.value and confidence >= warm_confidence and times_seen >= warm_threshold:
                # COLD → WARM: the warm_form already exists from compile recording
                # Validate token savings
                tok_orig = row["token_cost_original"] or 0
                tok_warm = row["token_cost_warm"] or 0
                if tok_warm < tok_orig:
                    self.conn.execute(
                        """UPDATE compression_codebook SET
                            stage = ?, promoted_at = ?, confidence = ?
                           WHERE id = ?""",
                        (CompressionStage.WARM.value, now, confidence, entry_id)
                    )
                    self.actions.append({
                        "type": "compression_promoted",
                        "entry_id": entry_id,
                        "from_stage": "COLD",
                        "to_stage": "WARM",
                        "pattern": row["pattern_text"][:80],
                        "warm_form": row["warm_form"][:40],
                        "token_savings": tok_orig - tok_warm,
                        "times_seen": times_seen,
                    })
                    promoted = True
                    learned += 1

            elif stage == CompressionStage.WARM.value and confidence >= hot_confidence and times_seen >= hot_threshold:
                # WARM → HOT: collapse to single emoji or minimal token
                # The hot_form is the most dominant emoji from the warm_form
                import re
                emoji_pat = re.compile(
                    r'[\U0001F300-\U0001F9FF\U00002702-\U000027B0\U0000FE00-\U0000FEFF'
                    r'\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]'
                )
                emojis = emoji_pat.findall(row["warm_form"])
                if emojis:
                    hot_form = emojis[0]  # Primary emoji becomes the hot token
                else:
                    # No emoji — use the abbreviation as-is (already minimal)
                    hot_form = row["warm_form"]

                tok_hot = estimate_tokens(hot_form)
                tok_orig = row["token_cost_original"] or 0

                if tok_hot < (row["token_cost_warm"] or tok_orig):
                    self.conn.execute(
                        """UPDATE compression_codebook SET
                            stage = ?, hot_form = ?, token_cost_hot = ?,
                            promoted_at = ?, confidence = ?
                           WHERE id = ?""",
                        (CompressionStage.HOT.value, hot_form, tok_hot,
                         now, confidence, entry_id)
                    )
                    self.actions.append({
                        "type": "compression_promoted",
                        "entry_id": entry_id,
                        "from_stage": "WARM",
                        "to_stage": "HOT",
                        "pattern": row["pattern_text"][:80],
                        "warm_form": row["warm_form"][:40],
                        "hot_form": hot_form,
                        "token_savings": tok_orig - tok_hot,
                        "times_seen": times_seen,
                    })
                    promoted = True
                    learned += 1

            self.entries_scanned += 1

        # ── Step 3: Vocab pack generation ─────────────────────────────────
        # Write learned.json with all WARM+ patterns for boot-time loading
        if learned > 0 or not rows:
            self._generate_learned_vocab_pack(vocab_pack_dir)

        self.conn.commit()
        self.compressions_learned = learned
        return learned

    def _generate_learned_vocab_pack(self, vocab_pack_dir: Optional[str] = None):
        """Write a learned.json vocab pack from WARM+ codebook entries.

        The vocab pack format matches _load_vocab_pack() expectations:
        [{"pattern": "regex", "replacement": "abbreviation", "flags": "vi"}, ...]
        """
        import re

        rows = self.conn.execute(
            """SELECT pattern_text, warm_form, hot_form, stage, confidence
               FROM compression_codebook
               WHERE stage >= ? AND confidence >= 0.8
               ORDER BY confidence DESC, times_seen DESC""",
            (CompressionStage.WARM.value,)
        ).fetchall()

        if not rows:
            return

        pack_entries = []
        for row in rows:
            # Use hot_form if available, otherwise warm_form
            replacement = row["hot_form"] if row["hot_form"] and row["stage"] == CompressionStage.HOT.value else row["warm_form"]
            # Build regex from pattern_text: escape and add word boundaries
            pattern_text = row["pattern_text"]
            pattern_regex = r'\b' + re.escape(pattern_text) + r'\b'

            pack_entries.append({
                "pattern": pattern_regex,
                "replacement": replacement,
                "flags": "vi",
                "confidence": row["confidence"],
                "stage": row["stage"],
            })

        # Determine output directory
        if not vocab_pack_dir:
            # Try standard locations
            this_dir = os.path.dirname(os.path.abspath(__file__))
            cli_dir = os.path.dirname(this_dir)  # Go up from src/core/ to librarian/
            # Check if we're in src/core — go up two levels to librarian/
            if os.path.basename(this_dir) == "core":
                cli_dir = os.path.dirname(os.path.dirname(this_dir))
            vocab_pack_dir = os.path.join(cli_dir, "vocab_packs")

        os.makedirs(vocab_pack_dir, exist_ok=True)
        pack_path = os.path.join(vocab_pack_dir, "learned.json")

        with open(pack_path, "w", encoding="utf-8") as f:
            json.dump(pack_entries, f, indent=2, ensure_ascii=False)

        self.actions.append({
            "type": "vocab_pack_generated",
            "path": pack_path,
            "entries": len(pack_entries),
        })

    # ─── Run All Passes ──────────────────────────────────────────────────

    def run_all(self) -> Dict[str, Any]:
        """
        Run all maintenance passes in order, respecting the token budget.

        Returns a comprehensive report.
        """
        ensure_maintenance_schema(self.conn)

        log_id = str(uuid.uuid4())
        started_at = datetime.utcnow().isoformat()

        # Record start
        self.conn.execute(
            """INSERT INTO maintenance_log
               (id, started_at, session_id, token_budget)
               VALUES (?, ?, ?, ?)""",
            (log_id, started_at, self.session_id, self.token_budget)
        )
        self.conn.commit()

        # Run passes in priority order
        try:
            # 1. Contradictions (highest priority — wrong data is worse than redundant data)
            if self._budget_remaining() > 0:
                self.pass_contradiction_detection()

            # 2. Orphaned corrections (close the loop on known issues)
            if self._budget_remaining() > 0:
                self.pass_orphaned_corrections()

            # 3. Near-duplicate merging (reduce noise)
            if self._budget_remaining() > 0:
                self.pass_near_duplicate_merging()

            # 4. Entry promotion (surface valuable content)
            if self._budget_remaining() > 0:
                self.pass_entry_promotion()

            # 5. Stale flagging (informational only)
            if self._budget_remaining() > 0:
                self.pass_stale_temporal_flagging()

            # 6. Compression learning (analyze codebook, promote patterns)
            if self._budget_remaining() > 0:
                self.pass_compression_learning()

            self.conn.commit()
        except Exception as e:
            # Log the error but don't crash
            self.actions.append({
                "type": "error",
                "detail": str(e),
            })

        completed_at = datetime.utcnow().isoformat()
        total_actions = (
            self.contradictions_found + self.orphans_linked +
            self.duplicates_merged + self.entries_promoted +
            self.stale_flagged + self.compressions_learned
        )

        # Update log
        self.conn.execute(
            """UPDATE maintenance_log SET
                completed_at = ?,
                passes_run = ?,
                actions_taken = ?,
                entries_scanned = ?,
                contradictions_found = ?,
                orphans_linked = ?,
                duplicates_merged = ?,
                entries_promoted = ?,
                stale_flagged = ?,
                compressions_learned = ?,
                tokens_used = ?,
                metadata = ?
               WHERE id = ?""",
            (
                completed_at,
                json.dumps(self.passes_run),
                total_actions,
                self.entries_scanned,
                self.contradictions_found,
                self.orphans_linked,
                self.duplicates_merged,
                self.entries_promoted,
                self.stale_flagged,
                self.compressions_learned,
                self.tokens_used,
                json.dumps({"actions": self.actions}),
                log_id,
            )
        )
        self.conn.commit()

        return {
            "maintenance_id": log_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "passes_run": self.passes_run,
            "entries_scanned": self.entries_scanned,
            "token_budget": self.token_budget,
            "tokens_used": self.tokens_used,
            "summary": {
                "contradictions_found": self.contradictions_found,
                "orphans_linked": self.orphans_linked,
                "duplicates_merged": self.duplicates_merged,
                "entries_promoted": self.entries_promoted,
                "stale_flagged": self.stale_flagged,
                "compressions_learned": self.compressions_learned,
                "total_actions": total_actions,
            },
            "actions": self.actions,
        }
