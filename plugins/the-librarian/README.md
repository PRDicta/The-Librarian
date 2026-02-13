# The Librarian

**Persistent memory for Claude.** The Librarian indexes your files and conversations into a local SQLite database, giving Claude perfect recall across sessions.

## What it does

- **Folder scanning**: Point The Librarian at any folder and it reads, chunks, and indexes every text file — code, docs, configs, markdown, everything.
- **Conversation memory**: Every substantive exchange is automatically stored. Preferences, decisions, project facts, code patterns — nothing gets lost.
- **Smart retrieval**: Hybrid search (keyword + semantic) finds relevant context when you need it, even across sessions.
- **Context window management**: Keeps Claude's active context lean by safely offloading older content to the rolodex, retrievable on demand.

## Prerequisites

The Librarian runs inside the Claude environment (Cowork or Claude Code), which includes Python out of the box. All Python dependencies are installed automatically when you run `/librarian-start` — no manual setup required.

## Getting started

After installing the plugin, run:

```
/librarian-start
```

This installs dependencies, boots the memory system, and walks you through scanning your first folder.

## Commands

| Command | Description |
|---------|-------------|
| `/librarian-start` | First-time setup and folder scanning |
| `/recall <query>` | Search your memory for past context |
| `/librarian-stats` | See what The Librarian knows about you |

## How it works

The Librarian uses a three-tier architecture:

1. **Active Context** — Claude's current context window (slim, focused)
2. **Hot Cache** — Frequently accessed entries kept in memory for fast retrieval
3. **Cold Storage** — Everything else in a local SQLite database with FTS5 full-text search

Content is chunked intelligently based on its type (prose, code, structured data, math) and stored verbatim — no compression, no summarization. Frequency of access determines importance, and entries are promoted or demoted between tiers automatically.

## Privacy

Everything stays on your machine. The rolodex is a local SQLite file. No data is sent to external services beyond the normal Claude conversation. The only API calls are to Anthropic's models (which you're already using).

## License

Proprietary — see [LICENSE](LICENSE) for details.
