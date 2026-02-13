


import sqlite3
import json
import struct
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from ..core.types import (
    RolodexEntry, ContentModality, EntryCategory, Tier
)

SCHEMA_SQL = """
-- Core rolodex entries: every piece of indexed content
CREATE TABLE IF NOT EXISTS rolodex_entries (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    content TEXT NOT NULL,
    content_type TEXT NOT NULL,
    category TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    source_range TEXT NOT NULL DEFAULT '{}',
    access_count INTEGER DEFAULT 0,
    last_accessed DATETIME,
    created_at DATETIME NOT NULL,
    tier TEXT DEFAULT 'cold',
    embedding BLOB,
    linked_ids TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}'
);
-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_entries_conversation
    ON rolodex_entries(conversation_id);
CREATE INDEX IF NOT EXISTS idx_entries_category
    ON rolodex_entries(category);
CREATE INDEX IF NOT EXISTS idx_entries_tier
    ON rolodex_entries(tier);
CREATE INDEX IF NOT EXISTS idx_entries_access_count
    ON rolodex_entries(access_count DESC);
CREATE INDEX IF NOT EXISTS idx_entries_created_at
    ON rolodex_entries(created_at DESC);
-- Full-text search index for keyword queries
CREATE VIRTUAL TABLE IF NOT EXISTS rolodex_fts USING fts5(
    entry_id,
    content,
    tags,
    category,
    tokenize='porter unicode61'
);
-- Conversation tracking
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at DATETIME NOT NULL,
    ended_at DATETIME,
    total_tokens INTEGER DEFAULT 0,
    entry_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active'
);
-- Query log for analytics
CREATE TABLE IF NOT EXISTS query_log (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    found BOOLEAN NOT NULL,
    entry_ids TEXT DEFAULT '[]',
    search_time_ms REAL,
    search_type TEXT,
    timestamp DATETIME NOT NULL
);
-- Phase 4: Persistent message storage for session resume/cross-session
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    token_count INTEGER DEFAULT 0,
    timestamp DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_turn
    ON messages(conversation_id, turn_number);

-- Phase 3: Preload log for tracking prediction accuracy
CREATE TABLE IF NOT EXISTS preload_log (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    pressure REAL NOT NULL,
    predicted_entry_ids TEXT DEFAULT '[]',
    injected_entry_ids TEXT DEFAULT '[]',
    cache_warmed_entry_ids TEXT DEFAULT '[]',
    hit_entry_ids TEXT DEFAULT '[]',
    timestamp DATETIME NOT NULL
);

-- Phase 7: Reasoning Chains — narrative indexes capturing the "why"
CREATE TABLE IF NOT EXISTS chains (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    chain_index INTEGER NOT NULL,
    turn_range_start INTEGER NOT NULL,
    turn_range_end INTEGER NOT NULL,
    summary TEXT NOT NULL,
    topics TEXT NOT NULL DEFAULT '[]',
    related_entries TEXT NOT NULL DEFAULT '[]',
    embedding BLOB,
    created_at DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chains_session_index
    ON chains(session_id, chain_index);
CREATE INDEX IF NOT EXISTS idx_chains_created_at
    ON chains(created_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS chains_fts USING fts5(
    chain_id,
    summary,
    topics,
    tokenize='porter unicode61'
);

-- Phase 8: Topics — emergent categories of knowledge
CREATE TABLE IF NOT EXISTS topics (
    id TEXT PRIMARY KEY,
    label TEXT UNIQUE NOT NULL,
    description TEXT DEFAULT '',
    parent_topic_id TEXT,
    created_at DATETIME NOT NULL,
    last_updated DATETIME,
    entry_count INTEGER DEFAULT 0,
    embedding BLOB
);

CREATE INDEX IF NOT EXISTS idx_topics_label
    ON topics(label);
CREATE INDEX IF NOT EXISTS idx_topics_parent
    ON topics(parent_topic_id);

CREATE VIRTUAL TABLE IF NOT EXISTS topics_fts USING fts5(
    topic_id,
    label,
    description,
    tokenize='porter unicode61'
);

-- Phase 8: Topic assignment log (traces topic evolution)
CREATE TABLE IF NOT EXISTS topic_assignments (
    id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    source TEXT NOT NULL,
    assigned_at DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_topic_assignments_entry
    ON topic_assignments(entry_id);
CREATE INDEX IF NOT EXISTS idx_topic_assignments_topic
    ON topic_assignments(topic_id);
"""

def init_database(db_path: str) -> sqlite3.Connection:


    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA_SQL)
    conn.commit()

    _safe_add_columns(conn)
    return conn


def _safe_add_columns(conn: sqlite3.Connection):

    additions = [

        ("conversations", "summary", "TEXT DEFAULT ''"),
        ("conversations", "last_active", "DATETIME"),
        ("conversations", "message_count", "INTEGER DEFAULT 0"),

        ("rolodex_entries", "topic_id", "TEXT"),
    ]
    for table, column, col_type in additions:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            conn.commit()
        except sqlite3.OperationalError:
            pass

def serialize_entry(entry: RolodexEntry) -> tuple:

    return (
        entry.id,
        entry.conversation_id,
        entry.content,
        entry.content_type.value,
        entry.category.value,
        json.dumps(entry.tags),
        json.dumps(entry.source_range),
        entry.access_count,
        entry.last_accessed.isoformat() if entry.last_accessed else None,
        entry.created_at.isoformat(),
        entry.tier.value,
        serialize_embedding(entry.embedding) if entry.embedding else None,
        json.dumps(entry.linked_ids),
        json.dumps(entry.metadata),
    )
def deserialize_entry(row: sqlite3.Row) -> RolodexEntry:

    return RolodexEntry(
        id=row["id"],
        conversation_id=row["conversation_id"],
        content=row["content"],
        content_type=ContentModality(row["content_type"]),
        category=EntryCategory(row["category"]),
        tags=json.loads(row["tags"]),
        source_range=json.loads(row["source_range"]),
        access_count=row["access_count"],
        last_accessed=(
            datetime.fromisoformat(row["last_accessed"])
            if row["last_accessed"] else None
        ),
        created_at=datetime.fromisoformat(row["created_at"]),
        tier=Tier(row["tier"]),
        embedding=(
            deserialize_embedding(row["embedding"])
            if row["embedding"] else None
        ),
        linked_ids=json.loads(row["linked_ids"]) if row["linked_ids"] else [],
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
    )
def serialize_embedding(embedding: List[float]) -> bytes:

    return struct.pack(f"{len(embedding)}f", *embedding)
def deserialize_embedding(blob: bytes) -> List[float]:

    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))
