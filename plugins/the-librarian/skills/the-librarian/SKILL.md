---
name: the-librarian
description: >
  Persistent memory across Cowork sessions via The Librarian. ALWAYS activate
  at the start of every conversation to load past context about the user.
  Provides recall of user preferences, project details, decisions, code patterns,
  and past discussions. Also activate when the user references previous sessions,
  says "remember", "last time", "before", or when continuity with past work
  would improve the response.
version: 0.1.0
---

# The Librarian — Cowork Memory

You have persistent memory. The Librarian indexes conversations and files into a local SQLite rolodex and retrieves relevant context on demand. It survives across sessions — what the user told you last week is still there.

The Librarian also manages your **context window** — keeping it lean so you never hit compaction. Only the recent working set stays in active context; everything older is safely in the rolodex, retrievable on demand via recall.

## Boot Protocol

**At the start of every session, before doing any other work, run:**

```bash
python "${CLAUDE_PLUGIN_ROOT}/librarian/librarian_cli.py" boot
```

Parse the JSON response:
- `user_profile`: Structured key-value dict of user preferences (name, timezone, response style, etc.). Always present if profile has been set. **Use this to personalize your interactions** — don't announce it, just apply it.
- `context_block`: Past context about this user, with the user profile block at the top (if set). **Don't announce it or parrot it back** — just know it. Let it inform how you greet them and approach their requests.
- `context_window.bridge_summary`: If non-null, this summarizes conversation history that was pruned from your active context. Treat it as orientation — you had this conversation, the details are in the rolodex, and you can recall specifics on demand.
- `context_window.active_messages` / `budget_remaining`: Your current window utilization. No action needed — just awareness.
- `total_entries: 0` means this is a fresh start. No past context yet. Suggest the user run `/librarian-start` to scan a folder.

## When to Ingest

**Ingest every substantive exchange.** The Librarian captures everything worth keeping — the background enrichment system handles categorization and topic assignment automatically. Each successful ingest creates a **checkpoint** — a guarantee that this content is safely in the rolodex and can be pruned from your active context window if needed.

**INGEST these (target ~100% of substantive content):**
- User preferences ("I prefer tabs", "always use TypeScript", "dark mode")
- Project facts ("my app uses FastAPI + SQLAlchemy", "the repo is at github.com/...")
- Decisions made ("we decided to use Redis for caching")
- Corrections ("actually, the function is called processData, not process_data")
- Code written or reviewed (significant snippets, not one-liners)
- Names, terms, or abbreviations the user defines
- Explicit requests to remember ("remember this", "note that", "keep in mind")
- Key outcomes ("the bug was caused by a race condition in the worker pool")
- Architectural discussions and trade-offs
- Implementation work — what was built, what files were changed, what tests pass
- Debugging sessions — root cause found, fix applied
- Any exchange you'd want to recall in a future session

**SKIP only these:**
- Greetings, thanks, "ok", "got it", "sure"
- Meta-conversation ("should we proceed?", "what do you think?")
- Content already ingested in this session

**How to ingest:**

```bash
python "${CLAUDE_PLUGIN_ROOT}/librarian/librarian_cli.py" ingest user "The user's substantive message"
python "${CLAUDE_PLUGIN_ROOT}/librarian/librarian_cli.py" ingest assistant "Your key response or decision"
```

Ingest user messages and your own responses separately when both contain durable info. You can summarize lightly when ingesting — strip the fluff, keep the facts.

The response includes `checkpoint` and `total_checkpoints` — these tell you how much of the conversation is safely backed up in the rolodex. The context window manager uses these checkpoints to know what's safe to prune from active context.

**Ingest frequently.** Each ingest creates a checkpoint that extends the safety net. The more checkpoints you have, the leaner the active context window can be, and the further you are from compaction.

## When to Recall

Before responding to a query, consider: **would past context help here?**

**Recall when:**
- The user says "last time", "before", "remember when", "we discussed"
- They mention a project, person, or term that likely came up before
- You feel like you *should* know something but don't
- The topic probably has prior context (their preferences, their codebase, their workflow)
- They ask you to continue work from a previous session
- The boot's `bridge_summary` mentions relevant earlier work
- You need details from earlier in this session that may have been pruned from active context

**How to recall:**

```bash
python "${CLAUDE_PLUGIN_ROOT}/librarian/librarian_cli.py" recall "the query topic"
```

This returns a formatted context block with relevant past entries. Fold the results into your response naturally — don't quote the block verbatim, just use the information.

If nothing relevant is found, the output will say "No relevant memories found." — just proceed normally.

**Recall is your long-term memory.** With the context window manager keeping your active context lean, recall becomes more important — it's how you access anything older than the recent working set. Use it liberally.

## User Profile

The Librarian has a dedicated **user profile** — a structured key-value store for persistent user preferences. Unlike regular rolodex entries, profile values are deduplicated (setting a key replaces the old value) and always loaded on boot.

**Set a preference:**
```bash
python "${CLAUDE_PLUGIN_ROOT}/librarian/librarian_cli.py" profile set <key> <value>
```

**View all preferences:**
```bash
python "${CLAUDE_PLUGIN_ROOT}/librarian/librarian_cli.py" profile show
```

**Remove a preference:**
```bash
python "${CLAUDE_PLUGIN_ROOT}/librarian/librarian_cli.py" profile delete <key>
```

Common profile keys: `name`, `timezone`, `response_style`, `preferred_language`, `editor`, `os`. Keys are normalized to lowercase with underscores.

When the user states a preference ("I prefer concise responses", "my timezone is Eastern"), use `profile set` to capture it durably — don't just ingest it as a regular entry.

## Context Window Awareness

The Librarian tracks your context budget:
- **20,000 token budget** for conversation history in the active window
- **Ingestion checkpoints** guarantee that pruned content is safely in the rolodex
- **Bridge summary** gives you continuity when older messages are pruned

You can check the window state at any time:

```bash
python "${CLAUDE_PLUGIN_ROOT}/librarian/librarian_cli.py" window
```

You don't need to manage this actively — the system handles pruning automatically. But be aware: **the more you ingest, the safer you are from compaction**, because each checkpoint extends what can be safely pruned from active context.

## End of Session

When the user says goodbye, or the session is clearly ending, close it out:

```bash
python "${CLAUDE_PLUGIN_ROOT}/librarian/librarian_cli.py" end "Brief summary of what was accomplished"
```

Keep the summary to one sentence. This helps future sessions understand what happened.

## Error Handling

If any CLI command fails, **continue normally**. Memory is a bonus, never a blocker. Log the error mentally and move on. The user's current task always takes priority over memory operations.

Common issues:
- `ModuleNotFoundError` → dependencies not installed. Run `/librarian-start` to set up.
- `.cowork_session` stale → delete it and re-boot. The DB is fine.
- `rolodex.db` missing → boot will create a fresh one automatically.
