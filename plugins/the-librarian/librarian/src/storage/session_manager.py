


import uuid
import sqlite3
from typing import List, Optional
from datetime import datetime
from ..core.types import (
    Message, MessageRole, SessionInfo, estimate_tokens
)


class SessionManager:


    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn


    def start_session(self, conversation_id: str) -> SessionInfo:


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


        now = datetime.utcnow()

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

        now = datetime.utcnow()
        msg_count = self._count_messages(conversation_id)
        self.conn.execute(
            """UPDATE conversations
               SET last_active = ?, message_count = ?
               WHERE id = ?""",
            (now.isoformat(), msg_count, conversation_id)
        )
        self.conn.commit()


    def list_sessions(self, limit: int = 20) -> List[SessionInfo]:


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


    def save_message(self, conversation_id: str, message: Message) -> str:


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


    def _count_messages(self, conversation_id: str) -> int:

        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ?",
            (conversation_id,)
        ).fetchone()
        return row["cnt"] if row else 0

    def _count_entries(self, conversation_id: str) -> int:

        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM rolodex_entries WHERE conversation_id = ?",
            (conversation_id,)
        ).fetchone()
        return row["cnt"] if row else 0

    def _count_tokens(self, conversation_id: str) -> int:

        row = self.conn.execute(
            "SELECT COALESCE(SUM(token_count), 0) as total FROM messages WHERE conversation_id = ?",
            (conversation_id,)
        ).fetchone()
        return row["total"] if row else 0

    def find_session_by_prefix(self, prefix: str) -> Optional[str]:


        rows = self.conn.execute(
            "SELECT id FROM conversations WHERE id LIKE ?",
            (prefix + "%",)
        ).fetchall()
        if len(rows) == 1:
            return rows[0]["id"]
        return None
