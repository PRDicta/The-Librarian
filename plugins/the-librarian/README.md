# The Librarian

**Stop re-explaining your project every session.**

The Librarian gives Claude persistent memory — file indexing, conversation recall, and context management that survives across sessions. Everything is stored locally in a SQLite rolodex on your machine.

## Quick start

```
/librarian-start
```

That’s it. Dependencies install automatically, the memory system boots, and you’re walked through scanning your first folder. No API keys required. No manual setup.

## What happens next

Once The Librarian is running, it works in the background:

- **Your files are indexed.** Code, docs, configs, markdown — chunked intelligently by content type and stored verbatim in the rolodex.
- **Your conversations are remembered.** Preferences, decisions, project facts, code patterns — captured automatically, available next session.
- **Context stays lean.** The Librarian offloads older content from Claude’s active window and retrieves it on demand, so sessions run longer without compacting.

Need something from a previous session? Just ask, or use `/recall`:

```
/recall authentication flow we discussed last week
```

## Commands

| Command | What it does |
|---------|-------------|
| `/librarian-start` | First-time setup and folder scanning |
| `/recall <query>` | Search your memory |
| `/librarian-stats` | See what The Librarian knows |

## Architecture

Three-tier memory with automatic promotion:

1. **Active Context** — Claude’s current window, kept slim
2. **Hot Cache** — frequently accessed entries, instant retrieval
3. **Cold Storage** — local SQLite with FTS5 full-text search

Hybrid search combines keyword matching with semantic similarity (local embeddings via `all-MiniLM-L6-v2`). No cloud vector databases. No external services.

## Two modes

| | Verbatim (default) | Enhanced |
|---|---|---|
| Setup | None — works at install | Add an Anthropic API key |
| Extraction | Heuristic, rule-based | LLM-powered with topic routing |
| Embeddings | Local (`all-MiniLM-L6-v2`) | Local (`all-MiniLM-L6-v2`) |
| Storage | Local SQLite | Local SQLite |
| Cost | Free | API usage charges apply |

Enhanced mode is an optional upgrade. The Librarian is fully functional without it.

## Privacy

Everything stays on your machine. The rolodex is a local SQLite file. No data leaves your computer beyond the normal Claude conversation. No telemetry, no cloud sync, no external dependencies.

## Prerequisites

The Librarian runs inside the Claude environment (Cowork or Claude Code), which includes Python out of the box. All Python dependencies are installed automatically when you run `/librarian-start`.

## License

Proprietary — see [LICENSE](LICENSE) for details.
