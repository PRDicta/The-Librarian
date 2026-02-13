# The Librarian CLI Reference

All commands output JSON. All paths are relative to `${CLAUDE_PLUGIN_ROOT}/librarian/`.

## Commands

### boot
Initialize or resume a session. Returns context from past sessions.
```bash
python librarian_cli.py boot
```
**Output:** `{ status, mode, session_id, resumed, total_entries, past_sessions, context_block, context_window }`

### ingest
Store a message in the rolodex.
```bash
python librarian_cli.py ingest <user|assistant> "<text>"
```
**Output:** `{ ingested, session_id, checkpoint, total_checkpoints }`

### batch-ingest
Ingest multiple messages from a JSON file.
```bash
python librarian_cli.py batch-ingest <file.json|->
```
**Input format:** `[{"role": "user", "content": "..."}, ...]`
**Output:** `{ ingested, messages_processed, session_id }`

### recall
Search memory and return formatted context.
```bash
python librarian_cli.py recall "<query>"
```
**Output:** Formatted context block (not JSON) with numbered entries, or "No relevant memories found."

### scan
Recursively scan a directory and ingest all text files.
```bash
python librarian_cli.py scan <directory>
```
**Output:** Newline-delimited JSON events:
- `scan_start`: `{ files_found, skipped_files }`
- `scan_progress`: `{ files_processed, files_total, percent, entries_created }`
- `scan_complete`: `{ files_scanned, files_ingested, entries_created, elapsed_seconds }`

**Behavior:**
- Reads all UTF-8-decodable files
- Skips binaries, files > 512KB, and ignored directories (.git, node_modules, etc.)
- Prefixes each file's content with `[Source file: relative/path]` metadata
- Creates entries using the standard ingestion pipeline (chunking, extraction, embedding)

### stats
Return session and rolodex statistics.
```bash
python librarian_cli.py stats
```
**Output:** `{ total_entries, total_sessions, topics, context_window, ... }`

### end
End the current session with an optional summary.
```bash
python librarian_cli.py end "<summary>"
```

### window
Show context window state.
```bash
python librarian_cli.py window
```
**Output:** `{ active_messages, pruned_messages, active_tokens, budget_remaining, checkpoints, bridge_summary }`

### topics
Topic management subcommands.
```bash
python librarian_cli.py topics list          # List all topics
python librarian_cli.py topics show <id>     # Show topic details
python librarian_cli.py topics search "<q>"  # Search topics
python librarian_cli.py topics stats         # Topic coverage stats
```
