"""
The Librarian — Session Manager (Phase 4)

Handles session lifecycle, message persistence, and cross-session
awareness. Each session is a conversation with a unique ID; the
rolodex DB persists all sessions, and messages are stored verbatim
for resume capability.

Multiple processes can share one DB file (SQLite WAL handles it).
"""
import uuid
import sqlite3
from typing import List, Optional
from datetime import datetime
from ..core.types import (
    Message, MessageRole, SessionInfo, estimate_tokens
)


class SessionManager:
    """
    Manages session lifecycle and message persistence.

    Responsibilities:
    - Register/end sessions in the conversations table
    - Persist every message for session resume
    - List and load past sessions
    - Track session metadata (message count, last active, summary)
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ─── Session Lifecycle ────────────────────────────────────────────────

    def start_session(self, conversation_id: str) -> SessionInfo:
        """
        Register a new session. Safe to call on existing sessions
        (uses INSERT OR REPLACE to update timestamps).
        """
        now = datetime.utcnow()
        self.conn.execute(
            """INSERT OR REPLACE INTO conversations
               (id, created_at, last_active, status, message_count,
                entry_count, total_tokens, summary)
               VALUES (?, ?, ?, 'active', 0, 0, 0, '')""",
            (conversation_id, now.isoformat(), now.isoformat())
        )
        self.conn.commit()
        return SessionInfo(
            session_id=conversation_id,
            started_at=now,
            last_active=now,
            status="active",
        )

    def end_session(
        self, conversation_id: str, summary: str = ""
    ) -> None:
        """
        Mark a session as ended. Updates final counts and summary.
        """
        now = datetime.utcnow()
        # Compute final counts from DB
        msg_count = self._count_messages(conversation_id)
        entry_count = self._count_entries(conversation_id)
        token_count = self._count_tokens(conversation_id)

        self.conn.execute(
            """UPDATE conversations
               SET ended_at = ?, last_active = ?, status = 'ended',
                   message_count = ?, entry_count = ?,
                   total_tokens = ?, summary = ?
               WHERE id = ?""",
            (
                now.isoformat(), now.isoformat(),
                msg_count, entry_count, token_count,
                summary, conversation_id
            )
        )
        self.conn.commit()

    def update_session_activity(self, conversation_id: str) -> None:
        """Bump last_active and message_count after each turn."""
        now = datetime.utcnow()
        msg_count = self._count_messages(conversation_id)
        self.conn.execute(
            """UPDATE conversations
               SET last_active = ?, message_count = ?
               WHERE id = ?""",
            (now.isoformat(), msg_count, conversation_id)
        )
        self.conn.commit()

    # ─── Session Listing ──────────────────────────────────────────────────

    def list_sessions(self, limit: int = 20) -> List[SessionInfo]:
        """
        List recent sessions, most recently active first.
        Returns SessionInfo objects with metadata.
        """
        rows = self.conn.execute(
            """SELECT id, created_at, last_active, ended_at,
                      message_count, entry_count, summary, status
               FROM conversations
               ORDER BY COALESCE(last_active, created_at) DESC
               LIMIT ?""",
            (limit,)
        ).fetchall()

        sessions = []
        for row in rows:
            sessions.append(SessionInfo(
                session_id=row["id"],
                started_at=datetime.fromisoformat(row["created_at"]),
                last_active=(
                    datetime.fromisoformat(row["last_active"])
                    if row["last_active"] else None
                ),
                ended_at=(
                    datetime.fromisoformat(row["ended_at"])
                    if row["ended_at"] else None
                ),
                message_count=row["message_count"] or 0,
                entry_count=row["entry_count"] or 0,
                summary=row["summary"] or "",
                status=row["status"] or "active",
            ))
        return sessions

    def get_session(self, conversation_id: str) -> Optional[SessionInfo]:
        """Get info about a specific session."""
        row = self.conn.execute(
            """SELECT id, created_at, last_active, ended_at,
                      message_count, entry_count, summary, status
               FROM conversations WHERE id = ?""",
            (conversation_id,)
        ).fetchone()
        if not row:
            return None
        return SessionInfo(
            session_id=row["id"],
            started_at=datetime.fromisoformat(row["created_at"]),
            last_active=(
                datetime.fromisoformat(row["last_active"])
                if row["last_active"] else None
            ),
            ended_at=(
                datetime.fromisoformat(row["ended_at"])
                if row["ended_at"] else None
            ),
            message_count=row["message_count"] or 0,
            entry_count=row["entry_count"] or 0,
            summary=row["summary"] or "",
            status=row["status"] or "active",
        )

    # ─── Message Persistence ──────────────────────────────────────────────

    def save_message(self, conversation_id: str, message: Message) -> str:
        """
        Persist a single message to the messages table.
        Returns the message ID.
        """
        msg_id = str(uuid.uuid4())
        self.conn.execute(
            """INSERT INTO messages
               (id, conversation_id, role, content, turn_number,
                token_count, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                msg_id,
                conversation_id,
                message.role.value,
                message.content,
                message.turn_number,
                message.token_count,
                message.timestamp.isoformat(),
            )
        )
        self.conn.commit()
        return msg_id

    def load_messages(self, conversation_id: str) -> List[Message]:
        """
        Load all messages for a session, ordered by turn number.
        Used for session resume.
        """
        rows = self.conn.execute(
            """SELECT role, content, turn_number, token_count, timestamp
               FROM messages
               WHERE conversation_id = ?
               ORDER BY turn_number ASC, timestamp ASC""",
            (conversation_id,)
        ).fetchall()

        messages = []
        for row in rows:
            msg = Message(
                role=MessageRole(row["role"]),
                content=row["content"],
                turn_number=row["turn_number"],
                token_count=row["token_count"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )
            messages.append(msg)
        return messages

    # ─── Internal Helpers ─────────────────────────────────────────────────

    def _count_messages(self, conversation_id: str) -> int:
        """Count persisted messages for a session."""
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ?",
            (conversation_id,)
        ).fetchone()
        return row["cnt"] if row else 0

    def _count_entries(self, conversation_id: str) -> int:
        """Count rolodex entries for a session."""
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM rolodex_entries WHERE conversation_id = ?",
            (conversation_id,)
        ).fetchone()
        return row["cnt"] if row else 0

    def _count_tokens(self, conversation_id: str) -> int:
        """Sum token counts for all messages in a session."""
        row = self.conn.execute(
            "SELECT COALESCE(SUM(token_count), 0) as total FROM messages WHERE conversation_id = ?",
            (conversation_id,)
        ).fetchone()
        return row["total"] if row else 0

    def find_session_by_prefix(self, prefix: str) -> Optional[str]:
        """
        Find a session ID by its prefix (for user-friendly /resume commands).
        Returns the full session ID if exactly one match, None otherwise.
        """
        rows = self.conn.execute(
            "SELECT id FROM conversations WHERE id LIKE ?",
            (prefix + "%",)
        ).fetchall()
        if len(rows) == 1:
            return rows[0]["id"]
        return None
