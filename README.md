# The Librarian

**Persistent memory for AI assistants.** The Librarian gives language models perfect recall across conversations — preferences, decisions, project context, and past discussions survive between sessions.

It works at the application layer, not the model layer. Ground truth is stored locally and injected at retrieval time, making it compatible with any LLM regardless of architecture.

## What it does

The Librarian sits between you and your AI assistant, maintaining a local knowledge base that grows over time:

- **Conversation memory** — Every substantive exchange is indexed. Preferences, decisions, code patterns, and project facts persist across sessions automatically.
- **Hybrid search** — Combines FTS5 keyword matching with ONNX-accelerated semantic embeddings (all-MiniLM-L6-v2) for accurate retrieval. Query expansion and multi-signal reranking surface the right context.
- **Three-tier storage** — Active context stays lean. Frequently accessed entries are promoted to a hot cache. Everything else lives in cold storage, retrieved on demand.
- **User knowledge** — Facts about the user (preferences, corrections, biographical context) get a permanent 3x search boost and are loaded at every boot.
- **Context window management** — Automatically tracks token budgets and offloads content to the rolodex before context overflows.
- **Temporal grounding** — Timestamps everything and flags stale entries, so the assistant never presents outdated information as current truth.
- **Dual mode** — Works out of the box in verbatim mode (no API key needed), or with an Anthropic API key for enhanced extraction and enrichment.

## Installation

Download the latest release for your platform from [Releases](https://github.com/PRDicta/The-Librarian/releases):

| Platform | Artifact |
|----------|----------|
| Windows  | `TheLibrarian-windows.tar.gz` |
| macOS    | `TheLibrarian-macos.tar.gz` |
| Linux    | `TheLibrarian-linux.tar.gz` |

Extract the archive and run the `librarian` binary. On first run, use the `init` command to set up your workspace:

```bash
librarian init /path/to/your/project
```

This copies the CLI, source files, and ONNX model into the target directory.

## Commands

All interaction happens through the `librarian` CLI:

```
librarian boot [--compact|--full-context]   # Start or resume a session
librarian ingest <role> "<text>"            # Store a message
librarian recall "<query>"                  # Search memory
librarian remember "<fact>"                 # Store a user-knowledge fact
librarian stats                             # Session and memory health
librarian end "<summary>"                   # Close a session
librarian profile set <key> <value>         # Set a user preference
librarian profile show                      # View preferences
librarian pulse                             # Heartbeat check
librarian maintain                          # Background knowledge graph hygiene
```

### Search options

```
librarian recall "<query>" --source conversation|document|user_knowledge
librarian recall "<query>" --fresh [hours]   # Prioritize recent entries
```

## How it works

### Storage

All data lives in a single SQLite database (`rolodex.db`) in your project directory. No external servers, no cloud dependencies. The database uses WAL mode for concurrent read/write safety.

Entries are categorized automatically (note, preference, decision, code, etc.) and tagged with temporal metadata. Each entry gets a vector embedding for semantic search alongside FTS5 indexing for keyword search.

### Search pipeline

1. **Query expansion** — The query is expanded into multiple search variants with entity extraction and intent classification.
2. **Wide-net retrieval** — Each variant pulls up to 15 candidates via hybrid search (keyword + semantic).
3. **Reranking** — A multi-signal reranker scores candidates on semantic relevance, entity overlap, category match, recency, and access frequency.
4. **Context assembly** — Top results are formatted into a context block with metadata, reasoning chains, and source attribution.

### Embedding

The Librarian bundles an ONNX-optimized all-MiniLM-L6-v2 model (~25MB) for local semantic embeddings. No API calls needed for search. The embedding strategy follows a fallback chain: Anthropic API → local sentence-transformers → ONNX Runtime → deterministic hash (always available).

### Reasoning chains

When the assistant's thinking process matters (design decisions, debugging sessions, multi-step analyses), The Librarian captures reasoning chains — ordered sequences of steps that preserve the "why" alongside the "what."

## Building from source

Requirements: Python 3.12+, pip

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-onnx.txt

# Export the ONNX model
pip install torch sentence-transformers onnx onnxscript
python scripts/export_onnx_model.py

# Build the standalone binary
python build.py
```

The build produces a PyInstaller bundle in `dist/librarian/` (Windows/Linux) or `dist/The Librarian.app/` (macOS).

### Running tests

```bash
pip install -r requirements-dev.txt
pytest tests/
```

### CI

The GitHub Actions workflow builds and smoke-tests on all three platforms (Windows, macOS, Linux) on every push to `main`. The smoke test runs 10 end-to-end checks: binary launch, boot, ingest, recall, embedding verification, cross-topic search, and session lifecycle.

## Data and backup

Your memory is a single file: `rolodex.db`. Standard SQLite — portable, self-contained, no server required.

While running, SQLite creates companion files (`rolodex.db-wal`, `rolodex.db-shm`) for crash safety. These are transient and fold back into the main database when the connection closes.

**To back up:** Copy all three files if The Librarian is running. If it's stopped, only `rolodex.db` is needed.

## License

Proprietary. Copyright (c) 2026 Dicta Technologies Inc. All rights reserved. See [LICENSE](plugins/the-librarian/LICENSE) for terms.
