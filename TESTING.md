# Testing The Librarian

Thanks for helping test. This document walks you through setup, what to try, and how to report back.

## Setup

You need Claude Desktop (Cowork mode) or Claude Code. Then run:

```
/plugin marketplace add PRDicta/The-Librarian
/plugin install the-librarian@The-Librarian
```

Then start a new session and run:

```
/librarian-start
```

This installs dependencies and boots the memory system. You’ll be asked to scan a folder — pick a project folder you’re actively working in. The more files and variety, the better the test.

**No API key is needed.** The Librarian works out of the box in verbatim mode. If you have an Anthropic API key and want to test enhanced mode, set it as `ANTHROPIC_API_KEY` in your environment before starting.

## What to test

### 1. First-run experience

- [ ] Did `/librarian-start` complete without errors?
- [ ] Were dependencies installed automatically?
- [ ] Did the folder scan finish? How many files were indexed?
- [ ] Anything confusing or unclear in the setup process?

### 2. Memory across sessions

This is the core value. After your first session:

- [ ] Start a **new session** and run `/librarian-start` again (it should boot instantly, no re-scan needed)
- [ ] Ask Claude something that references your previous session — does it remember?
- [ ] Try `/recall <something you discussed last time>` — does it return relevant results?

### 3. Session length

Pay attention to how long your sessions run before compacting:

- [ ] How many minutes/hours before you see "Compacting conversation..."?
- [ ] How tool-heavy was the session? (lots of file edits, browser use, code execution?)
- [ ] Did context quality degrade over time, or stay consistent?

**Baseline: 3+ hours and 1,073 turns without a single compaction in our development session.**

### 4. Recall quality

- [ ] Try `/recall` with specific queries — does it find what you’re looking for?
- [ ] Try vague queries — does it surface the right context?
- [ ] Try queries about things from 2+ sessions ago — still there?

### 5. Edge cases

- [ ] Large folders (1,000+ files) — does the scan complete?
- [ ] Binary files, images, non-text content — are they skipped gracefully?
- [ ] Very long sessions — does memory management hold up?
- [ ] Multiple projects — does context stay separated?

### 6. Stats

Run `/librarian-stats` periodically and note:

- [ ] Total entries — is it growing as expected?
- [ ] Category breakdown — does it match your usage?
- [ ] Anything surprising or incorrect?

## How to report

Open a GitHub Issue on this repo with:

1. **Environment**: Cowork or Claude Code, OS, anything unusual about your setup
2. **What you tested**: Which scenarios from above
3. **What happened**: Expected vs. actual behavior
4. **Session length**: How long before compaction (if it happened at all)
5. **Overall impression**: Does it feel like Claude remembers you?

If something breaks, include any error messages you see. Screenshots are helpful.

## What not to worry about

- The repo is private — your data stays local and the plugin never phones home
- If something doesn’t work, that’s exactly what this test is for
- You can uninstall cleanly at any time

## Uninstall

```
/plugin uninstall the-librarian
```

This removes the plugin. Your local rolodex database remains in your project folder if you want to inspect it, or you can delete it manually.
