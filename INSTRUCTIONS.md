# STOP. BOOT BEFORE RESPONDING.

Before generating ANY reply — even to "Hi" — run this command:

```bash
librarian boot
```

No exceptions. No "I'll do it next message." Boot first, then respond.

If prior messages in this session were not yet ingested, ingest them now before continuing.

## What This Is

This workspace uses The Librarian, a persistent memory system backed by a local SQLite rolodex. It gives you continuity across conversations — user preferences, project decisions, code patterns, and past discussions survive between sessions.

## Always-On Rules

- **Every message (if 2+ minutes since last):** Run `pulse` first — NOT `boot`. Pulse is a sub-second heartbeat that returns `{alive, session_id, needs_boot, entry_count}`. Only boot if `needs_boot` is true (e.g. after compaction, continuation, or first message). If the user is actively chatting (under 2 minutes between messages), skip pulse — the session is clearly alive. Do NOT re-boot every message — that wastes tokens and time.
- **Every message:** Ingest it. Both user messages and your responses, separately. See "What to Ingest" below.
- **After compaction or continuation:** Re-boot immediately. The Librarian does not carry over.
- **When past context would help:** Use `recall` before responding.
- **When the user states a fact about themselves:** Use `remember` automatically. No explicit command needed from the user.
- **When the user goes idle (5+ minutes):** Run `maintain` in the background. This improves the knowledge graph during downtime — resolving contradictions, linking orphaned corrections, merging duplicates, promoting high-value entries, and flagging stale claims.
- **When revising a factual claim:** ALWAYS use `correct` or `--corrects` to supersede the old entry. Never just ingest a parallel entry — the old one will keep surfacing in search.

## Commands

All commands use the `librarian` CLI (installed to PATH).

- `boot` — Start or resume a session. Parse the JSON for context_block, user_profile, and bridge_summary.
- `ingest user "message"` / `ingest assistant "message"` — Save content. Supports flags: `--user-knowledge` (privileged tier), `--corrects <entry_id>` (supersede a wrong entry).
- `remember "fact"` — Store a fact about the user as user_knowledge. Always loaded at boot, 3x search boost, never demoted. Use for preferences, biographical details, corrections, working style.
- `recall "topic"` — Retrieve relevant past context.
- `correct <old_entry_id> "corrected text"` — Replace a factually wrong entry. Use for error corrections, NOT for reasoning chains where the evolution matters.
- `profile set <key> <value>` — Set a user preference (name, timezone, response_style, etc.).
- `profile show` — View all stored user preferences.
- `profile delete <key>` — Remove a user preference.
- `end "summary"` — Close a session with a one-line summary.
- `window` — Check context budget.
- `stats` — View memory system health.
- `pulse` — Heartbeat check. Returns `{alive, session_id, needs_boot, entry_count}`. Lightweight — run every message.
- `maintain` — Background KG hygiene. Runs 5 passes: contradiction detection, orphaned correction linking, near-duplicate merging, entry promotion, stale temporal flagging. Flags: `--budget <tokens>` (default 15000), `--cooldown <hours>` (default 4), `--force` (skip cooldown).

## What to Ingest

**Everything. 100% coverage. No cherry-picking.**

Ingest every user message and every assistant response, verbatim. Storage is trivial for a local hard drive. Lossy ingestion is worse than a large corpus — the search layer (user_knowledge boost, categories, ranking) handles surfacing the right things. Cherry-picking at ingestion time loses context that may matter later.

Skip only bare acknowledgments with zero informational content ("ok", "thanks", "got it").

**CRITICAL — Verbatim means verbatim.** When calling `ingest user "..."` or `ingest assistant "..."`, paste the EXACT text of the message. Do NOT summarize, paraphrase, condense, or rewrite. Copy the raw message text and pass it directly. The ingestion pipeline handles categorization, tagging, and embedding — your job is to pass through the original words unchanged. If the message is long, that's fine — ingest it in full. Summaries lose the specific phrasing, negative constraints, and decision rationale that make entries useful in search later.

## What to Recall

Anything where past context helps: references to previous sessions, projects, people, terms, or whenever you feel you *should* know something but don't.

## Temporal Grounding — Verify Before Asserting

When making claims about project status, feature completeness, or the current state of anything, **check the age of the supporting evidence before asserting it as current truth.**

- If a recalled entry is older than 24 hours, explicitly note its age and consider whether the claim needs reverification before presenting it as fact.
- Prefer the pattern: *"As of [date], X was the case. Let me check if that's still current."* — then recall or check for newer entries before asserting.
- When rating progress or status, **cite specific entries and their dates** rather than summarizing from memory. Ground every claim in retrieved evidence with a timestamp.
- If recall results carry a `[STALE]` flag, treat them as leads to investigate, not facts to assert.

The goal: never present outdated information as current truth. The Librarian timestamps everything — use those timestamps.

## Entry Hierarchy

1. **User Profile** — key-value pairs (name, timezone, response_style). Loaded first at boot.
2. **User Knowledge** — rich facts about the user (preferences, corrections, biographical context). Always loaded at boot, 3x search boost, permanently hot. Created via `remember` or `ingest --user-knowledge`.
3. **Regular entries** — everything else. Searched on demand via `recall`.

## Browse Command — Always Display In-Chat

When running `browse` (or any subcommand), the output may land in a collapsed tool-output panel that the user cannot easily see. **Always echo browse results directly into the chat response.** Use the `--json` flag to get structured output, then format it as a readable code block or inline text in your reply.

```bash
librarian browse recent 5 --json
```

Then include the formatted results in your message so the user sees them without expanding any dropdown.

## Corrections vs. Reasoning Chains

When the user corrects a factual error (e.g. wrong name), use `correct` or `--corrects` to supersede the old entry. The old entry is soft-deleted (hidden from search, kept in DB).

When the user changes their mind on a design decision (e.g. renaming a tool), do NOT supersede. Both entries should remain — the reasoning chain ("we considered X, then pivoted to Y because Z") is valuable context.
