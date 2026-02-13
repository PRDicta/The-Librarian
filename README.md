# The Librarian

**Persistent memory for Claude.** The Librarian gives Claude perfect recall across sessions by indexing your files and conversations into a local SQLite database.

## What it does

- **Scans your folders** — point it at any directory and it reads, chunks, and indexes every text file: code, docs, configs, markdown, everything
- **Remembers your conversations** — preferences, decisions, project facts, code patterns are automatically stored across sessions
- **Retrieves on demand** — hybrid search (keyword + semantic) finds relevant context when you need it, even from weeks ago
- **Manages context intelligently** — keeps Claude's active window lean by safely offloading older content, retrievable instantly

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

Everything stays on your machine. The rolodex is a local SQLite file. No data leaves your computer beyond the normal Claude conversation.

## License

Proprietary. Copyright (c) 2026 Dicta Technologies Inc. All rights reserved. See [LICENSE](plugins/the-librarian/LICENSE) for terms.

---

Built by [Dicta](https://usedicta.com)
