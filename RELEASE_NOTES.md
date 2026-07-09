# Release Notes — The Librarian

If The Librarian was useful to you, please consider [buying me a drink](https://buymeacoffee.com/chief_librarian).

## v1.3.1 — Boot Crash Fix (2026-07-08)

Fixes a `NameError` on boot that affected every default boot in v1.3.0.

### Fixed

- **Boot crash (`emoji_compression_active`).** The default (full) boot path
  referenced an uninitialized `emoji_compression_active` flag, raising a
  `NameError` before boot could return. This fired on every standard `boot`
  regardless of compression settings; only the `--compact` and
  `--full-context` sub-modes were unaffected. The flag is now initialized and
  tracked independently from the abbreviation-compression flag, and both
  flags are reported consistently across all boot modes.

Thanks to **PennyPineappleRain** for the detailed report and the fix.

---

## v1.3.0 — Scheduled Tasks & Automation (2026-03-04)

Adds first-class scheduled task support, enabling automated maintenance
and update monitoring without manual intervention.

### New

- **Daily health check** — Scheduled task that runs every morning at 9 AM PT.
  Boots The Librarian, runs pulse/stats/maintain/window/topics/history, and
  reports a concise health summary. Catches anomalies (zero entries, DB errors,
  stale sessions) before they become problems.

- **Weekly update check** — Scheduled task that runs every Friday at 10 AM PT.
  Compares the local installed version against the latest GitHub release and
  prompts the user to update if a newer version is available. Handles the
  pull and file copy automatically if approved.

### Infrastructure

- Scheduled tasks use Cowork's new `create_scheduled_task` API with cron
  expressions in local timezone (no UTC conversion needed).
- Update check follows the existing development workflow: local workspace
  stays unified with PRDicta/The-Librarian repo; personal data (rolodex.db)
  is never pushed.

---

## v1.2.0 — Operations Block & Project Knowledge (2026-02-19)

- Boot-injected operations block for session rules
- Auto-maintain on boot
- Project knowledge tier — project-scoped privileged context with 3x boost
- Sync after every ingest

---

## v1.1.0 — License Change (2026-02-16)

**License: Dual-licensed under AGPL-3.0 + Commercial**

The Librarian has moved from a proprietary license to a dual-license model,
following the approach pioneered by MySQL:

- **AGPL-3.0** — The open-source edition is now available under the GNU Affero
  General Public License v3.0. You are free to use, modify, and distribute
  The Librarian under AGPL terms. If you modify the software and make it
  available over a network, you must release your modified source code under
  the AGPL-3.0.

- **Commercial License** — For OEMs, ISVs, SaaS providers, and enterprises
  that want to embed or distribute The Librarian in proprietary products
  without AGPL obligations, a commercial license is available from
  Dicta Technologies Inc. Contact licensing@usedicta.com for terms.

This change makes The Librarian freely available to the open-source community
while protecting against proprietary forks that don't contribute back.

See [LICENSE](LICENSE) and
[COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) for full details.

---

## v1.0.0 — Initial Release (2026-02-16)

**License: Dual-licensed under AGPL-3.0 + Commercial**

Persistent memory for Cowork — first production release.

### Features

- Hybrid search: FTS5 keyword + ONNX semantic embeddings (all-MiniLM-L6-v2)
- Query expansion with entity extraction and multi-signal reranking
- Reasoning chains — captures the "why" alongside the "what"
- User knowledge tier with 3x search boost, always loaded at boot
- Temporal grounding and staleness detection
- Context window management with manifest-based pruning
- At-rest database management — background knowledge graph hygiene that runs
  during idle periods, performing contradiction detection, orphaned correction
  linking, near-duplicate merging, entry promotion, and stale temporal flagging
- Cross-platform builds: Windows, macOS, Linux
- Full CI pipeline with 10-point smoke test on all platforms

### Platforms

| Platform     | Artifact                       |
|-------------|-------------------------------|
| Windows x64 | `TheLibrarian-windows.tar.gz` |
| macOS arm64  | `TheLibrarian-macos.tar.gz`   |
| Linux x64    | `TheLibrarian-linux.tar.gz`   |
