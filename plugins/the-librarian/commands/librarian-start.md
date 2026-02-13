---
description: Set up The Librarian — scan a folder to build your memory
allowed-tools: Read, Bash, AskUserQuestion
model: sonnet
---

# /librarian-start — Onboarding & Folder Scan

This command initializes The Librarian for first-time users or rescans a folder to update the knowledge base.

## Step 1: Install Dependencies

Run the install script to ensure all Python dependencies are present:

```bash
bash "${CLAUDE_PLUGIN_ROOT}/scripts/install.sh"
```

The script outputs JSON. Check the `status` field:
- `"ok"` → dependencies are ready, continue to Step 2
- `"error"` → show the user the `message` field, which contains specific guidance (e.g., "Python is not installed" or which packages failed). Do not proceed until dependencies are resolved.

## Step 2: Boot The Librarian

```bash
python "${CLAUDE_PLUGIN_ROOT}/librarian/librarian_cli.py" boot
```

Parse the JSON response. If `total_entries` > 0, inform the user that The Librarian already has memories and ask if they want to scan an additional folder or if they're all set.

## Step 3: Identify the Folder to Scan

If the user provided a folder path as an argument (`$ARGUMENTS`), use that path directly.

Otherwise, ask the user which folder they'd like The Librarian to index. The default should be their current workspace folder. Explain that The Librarian will read all text-based files (code, docs, config, markdown, etc.) and build a searchable knowledge base from them.

Let the user know:
- Only text-based files are read (binaries, images, and media are skipped)
- Files over 512KB are skipped (minified bundles, data dumps)
- Common non-essential directories are skipped (.git, node_modules, __pycache__, etc.)
- Everything stays local — nothing is sent anywhere except through the normal Claude conversation

## Step 4: Run the Folder Scan

```bash
python "${CLAUDE_PLUGIN_ROOT}/librarian/librarian_cli.py" scan "<folder_path>"
```

This outputs newline-delimited JSON progress events. Parse them and give the user periodic updates:
- When the scan starts: report how many files were found
- During scanning: report progress percentage every ~25%
- When complete: report total files ingested, entries created, and time elapsed

The scan may take a while for large folders. Reassure the user that this is a one-time operation — future sessions boot instantly from the existing rolodex.

## Step 5: Confirm Success

After the scan completes, run stats to show the user what The Librarian now knows:

```bash
python "${CLAUDE_PLUGIN_ROOT}/librarian/librarian_cli.py" stats
```

Summarize the results conversationally: how many entries, how many topics, and what kinds of content were indexed. End by letting the user know that The Librarian will now automatically remember their conversations going forward, and they can use `/recall` to search their memory at any time.
