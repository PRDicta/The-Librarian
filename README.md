# The Librarian

**Stop re-explaining your project every session.**

The Librarian gives Claude persistent memory. It indexes your files and conversations into a local SQLite rolodex, so what you told Claude last week — preferences, decisions, code patterns, project context — is still there next time you sit down.

> 3+ hours. 1,073 turns. Zero compactions. One continuous conversation.

## Why this matters

Claude is stateless. Every new session starts from scratch. The Librarian fixes that — not with retrieval tricks or cloud vector databases, but with a structured, local memory architecture that keeps Claude sharp across sessions without bloating the context window.

## What it does

- **Scans your folders** — point it at any directory and it reads, chunks, and indexes every text file: code, docs, configs, markdown, everything
- **Remembers your conversations** — preferences, decisions, project facts, and code patterns are automatically stored across sessions
- **Retrieves on demand** — hybrid search (keyword + semantic) finds relevant context when you need it, even from weeks ago
- **Manages context intelligently** — keeps Claude’s active window lean by offloading older content to the rolodex, retrievable instantly

## How it works

Three-tier architecture:

1. **Active Context** — Claude’s current context window, kept slim and focused
2. **Hot Cache** — frequently accessed entries held in memory for instant retrieval
3. **Cold Storage** — everything else in a local SQLite database with FTS5 full-text search

Content moves between tiers automatically based on access frequency. Nothing is compressed or summarized — entries are stored verbatim and promoted when needed.

## Two modes, zero friction

- **Verbatim mode** (default) — works immediately at install. No API key, no configuration. Uses heuristic extraction and local embeddings (`all-MiniLM-L6-v2`).
- **Enhanced mode** — add an Anthropic API key for LLM-powered extraction, topic routing, and negotiation. Optional upgrade, not a requirement.

## Install

```
/plugin marketplace add PRDicta/The-Librarian
/plugin install the-librarian@The-Librarian
```

Then run `/librarian-start` to set up and scan your first folder.

## Commands

| Command | What it does |
|---------|-------------|
| `/librarian-start` | First-time setup and folder scanning |
| `/recall <query>` | Search your memory |
| `/librarian-stats` | See what The Librarian knows |

## Privacy

Everything stays on your machine. The rolodex is a local SQLite file. No data leaves your computer beyond the normal Claude conversation. No cloud databases, no external services, no telemetry.

## License

Proprietary — see [LICENSE](plugins/the-librarian/LICENSE) for details.
