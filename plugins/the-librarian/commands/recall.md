---
description: Search your memory for past context
allowed-tools: Bash
argument-hint: <query>
---

# /recall — Search Memory

Search The Librarian's memory for relevant past context. Takes a natural language query and returns matching entries from the rolodex.

## Usage

The user provides a query as `$ARGUMENTS`. If no query is provided, ask the user what they'd like to recall.

## Execute

```bash
python "${CLAUDE_PLUGIN_ROOT}/librarian/librarian_cli.py" recall "$ARGUMENTS"
```

## Present Results

If results are found, the output is a formatted context block with numbered entries. Present the results naturally — don't dump raw output. Instead, synthesize the relevant memories into a coherent response that answers what the user was looking for.

If the output says "No relevant memories found.", let the user know and suggest they try different search terms. Broader queries often work better than very specific ones.
