# Release Notes — The Librarian

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

See [LICENSE](plugins/the-librarian/LICENSE) and
[COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) for full details.

---

## v1.0.0 — Initial Release (2026-02-16)

**License: Dual-licensed under AGPL-3.0 + Commercial**

Persistent memory for AI assistants — first production release.

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
