---
description: See what The Librarian knows about you
allowed-tools: Bash
---

# /librarian-stats — Memory Overview

Show the user a summary of what The Librarian has stored in their rolodex.

## Execute

```bash
python "${CLAUDE_PLUGIN_ROOT}/librarian/librarian_cli.py" stats
```

## Present Results

Parse the JSON output and present a friendly, readable summary. Include:

- **Total memories**: How many rolodex entries exist
- **Sessions**: How many past conversation sessions are recorded
- **Topics**: How many topics have been identified, and what the top ones are
- **Coverage**: What percentage of entries have been categorized into topics
- **Context window**: Current budget utilization if in an active session

Keep the tone conversational. This command is about transparency — the user wants to understand what The Librarian knows. If the rolodex is empty (total_entries = 0), suggest running `/librarian-start` to scan a folder and build the initial knowledge base.
