#!/usr/bin/env python3
"""
The Librarian — Cowork CLI

Thin wrapper around TheLibrarian for Bash invocation from Cowork skills.
Session continuity via .cowork_session file alongside rolodex.db.

Usage:
    python librarian_cli.py boot [--compact|--full-context]  # Init/resume session
    python librarian_cli.py ingest <role> "<text>"           # Store a message
    python librarian_cli.py batch-ingest <file.json>         # Ingest multiple messages
    python librarian_cli.py recall "<query>"                 # Search → context block
    python librarian_cli.py stats                            # Session stats (JSON)
    python librarian_cli.py end "<summary>"                  # End session
    python librarian_cli.py schema                           # Dump DB schema
    python librarian_cli.py history <first|recent|count|range>  # Session history

batch-ingest JSON format:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
"""
import asyncio
import json
import os
import sys

# Ensure UTF-8 output on all platforms (Windows defaults to cp1252 which
# can't handle the Unicode box-drawing chars in context blocks)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src.__version__ import __version__

# Housekeeping: run fuse cleanup every N ingestions
FUSE_CLEANUP_INTERVAL = 10

# Lazy imports — TheLibrarian pulls in numpy, embeddings, etc.
# Deferred to first use so the process starts fast for simple commands.
_TheLibrarian = None
_LibrarianConfig = None


def _lazy_imports():
    """Import heavy modules on first use."""
    global _TheLibrarian, _LibrarianConfig
    if _TheLibrarian is None:
        from src.core.librarian import TheLibrarian
        from src.utils.config import LibrarianConfig
        _TheLibrarian = TheLibrarian
        _LibrarianConfig = LibrarianConfig


def _load_config():
    """Load config from .env file."""
    _lazy_imports()
    env_file = os.path.join(SCRIPT_DIR, ".env")
    return _LibrarianConfig.from_env(env_path=env_file)


# Allow override via env vars (useful for testing)
DB_PATH = os.environ.get("LIBRARIAN_DB_PATH", os.path.join(SCRIPT_DIR, "rolodex.db"))
SESSION_FILE = os.environ.get("LIBRARIAN_SESSION_FILE", os.path.join(SCRIPT_DIR, ".cowork_session"))

# ─── LLM Adapter ───────────────────────────────────────────────────────

def _build_adapter():
    """Create AnthropicAdapter if API key is available, else None."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None, "verbatim"
    try:
        from src.indexing.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(api_key=api_key), "enhanced"
    except Exception:
        return None, "verbatim"


def _make_librarian():
    """Create TheLibrarian with adapter if available."""
    _lazy_imports()
    adapter, mode = _build_adapter()
    return _TheLibrarian(db_path=DB_PATH, llm_adapter=adapter), mode


def load_session_id():
    """Load active session ID from file, or None."""
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r") as f:
                data = json.load(f)
            return data.get("session_id")
        except (json.JSONDecodeError, IOError):
            return None
    return None


def save_session_id(session_id):
    """Persist session ID to file."""
    with open(SESSION_FILE, "w") as f:
        json.dump({"session_id": session_id}, f)


def clear_session_file():
    """Remove session file."""
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)


def close_db(lib):
    """Close the DB connection without ending the session."""
    lib.rolodex.close()


def _cleanup_fuse_hidden():
    """Move orphaned .fuse_hidden* files to a designated junk folder.

    These are FUSE filesystem artifacts created when files are replaced or
    deleted while something holds an open handle. They're always safe to
    remove — they're orphaned file handles with no functional purpose.

    Files are moved (not deleted) because the FUSE mount may not allow
    deletion. Moving to a single .fuse_junk/ folder corrals them for
    easy manual cleanup.
    """
    workspace = os.path.dirname(SCRIPT_DIR)  # Parent of librarian/
    junk_dir = os.path.join(workspace, ".fuse_junk")
    moved = 0
    removed = 0
    errors = 0

    for root, dirs, files in os.walk(workspace):
        # Don't recurse into the junk folder itself
        if os.path.abspath(root) == os.path.abspath(junk_dir):
            dirs.clear()
            continue
        for fname in files:
            if fname.startswith('.fuse_hidden'):
                src = os.path.join(root, fname)
                # Try delete first (works inside the VM's own directories)
                try:
                    os.remove(src)
                    removed += 1
                    continue
                except OSError:
                    pass
                # Delete failed — move to junk folder instead
                try:
                    os.makedirs(junk_dir, exist_ok=True)
                    dst = os.path.join(junk_dir, fname)
                    # Avoid collisions by appending a counter
                    if os.path.exists(dst):
                        base, ext = os.path.splitext(fname)
                        counter = 1
                        while os.path.exists(dst):
                            dst = os.path.join(junk_dir, f"{base}_{counter}{ext}")
                            counter += 1
                    os.rename(src, dst)
                    moved += 1
                except OSError:
                    errors += 1

    if removed > 0 or moved > 0 or errors > 0:
        print(json.dumps({
            "housekeeping": "fuse_cleanup",
            "removed": removed,
            "moved_to_junk": moved,
            "errors": errors,
        }), file=sys.stderr)


def _ensure_session(lib, *, caller="unknown"):
    """Auto-boot guard: ensure we have an active session.

    If a .cowork_session file exists, resume it. Otherwise, treat this as
    an unplanned boot — create a new session and log a warning event so
    the rolodex records that an automatic recovery occurred.

    Returns True if a session was resumed/created, False if something failed.
    """
    session_id = load_session_id()
    if session_id:
        info = lib.resume_session(session_id)
        if info:
            return True

    # No session file, or resume failed — auto-boot
    save_session_id(lib.session_id)

    # Log the auto-boot as a warning so it's visible in the rolodex
    import datetime
    warning_msg = (
        f"[AUTO-BOOT] The Librarian was not booted before '{caller}' was called. "
        f"A new session was created automatically at {datetime.datetime.utcnow().isoformat()}Z. "
        f"This likely means a context compaction or continuation occurred without re-invoking the skill."
    )
    # Print to stderr so it doesn't pollute JSON stdout
    print(json.dumps({
        "warning": "auto_boot",
        "reason": f"No active session when '{caller}' was called",
        "session_id": lib.session_id,
    }), file=sys.stderr)

    return True


def _load_instructions():
    """Load INSTRUCTIONS.md from the application directory.

    Search order:
    1. PyInstaller bundle (sys._MEIPASS)
    2. Next to the frozen executable (installed layout)
    3. Next to this script (development layout)

    Returns the markdown content as a string, or None if not found.
    """
    candidates = []

    # Frozen bundle (PyInstaller)
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        candidates.append(os.path.join(meipass, "INSTRUCTIONS.md"))

    # Next to executable (Inno Setup install)
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        candidates.append(os.path.join(exe_dir, "INSTRUCTIONS.md"))
        candidates.append(os.path.join(exe_dir, "lib", "INSTRUCTIONS.md"))

    # Development layout (next to this script)
    candidates.append(os.path.join(SCRIPT_DIR, "INSTRUCTIONS.md"))

    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except OSError:
                continue

    return None


def _check_for_update():
    """Best-effort version check against GitHub Releases.

    Returns dict with latest_version and download_url if an update exists,
    or None if current or check fails. Never raises — failures are silent.
    """
    import urllib.request
    VERSION_URL = "https://raw.githubusercontent.com/PRDicta/The-Librarian/main/version.json"
    try:
        req = urllib.request.Request(VERSION_URL, headers={"User-Agent": "TheLibrarian/" + __version__})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latest = data.get("version", __version__)
        if latest != __version__:
            return {
                "latest_version": latest,
                "download_url": data.get("download_url", ""),
                "message": data.get("message", f"Update available: {latest}"),
            }
    except Exception:
        pass
    return None


async def cmd_boot(compact=False, full_context=False):
    """Initialize or resume session. Returns context JSON.

    Modes:
      --compact       Fast boot: profile + user_knowledge + session metadata only.
                      Designed for immediate responsiveness — the AI can start
                      replying while a background agent loads full context.
      --full-context  Return only the manifest-based context block (the heavy
                      payload). Designed to be called by a background agent
                      after compact boot. Skips session init (already done).
      (default)       Full boot: everything in one shot (legacy behavior).

    Phase 10: Manifest-based boot. Instead of firing hardcoded keyword
    queries, we build/load a ranked manifest of entries selected by
    topic-weighted importance scoring and refined by session behavior.
    """
    lib, mode = _make_librarian()

    # Try resuming existing session
    existing_id = load_session_id()
    resumed = False
    if existing_id:
        info = lib.resume_session(existing_id)
        if info:
            resumed = True

    save_session_id(lib.session_id)

    # Get stats
    stats = lib.get_stats()
    past_sessions = lib.list_sessions(limit=10)

    # ─── Fixed-cost context: profile + user_knowledge (always loaded) ────
    from src.retrieval.context_builder import ContextBuilder
    from src.core.types import estimate_tokens
    cb = ContextBuilder()

    profile = lib.rolodex.profile_get_all()
    profile_block = cb.build_profile_block(profile) if profile else ""

    uk_entries = lib.rolodex.get_user_knowledge_entries()
    uk_block = cb.build_user_knowledge_block(uk_entries) if uk_entries else ""

    fixed_token_cost = estimate_tokens(profile_block + uk_block)

    # Serialize profile for structured access
    user_profile_json = {k: v["value"] for k, v in profile.items()} if profile else {}

    # Phase 9: Context window state
    window_state = lib.context_window.get_state(lib.state.messages)
    bridge = lib.context_window.bridge_summary

    # ─── Compact boot: fast path ─────────────────────────────────────────
    if compact:
        # Return only the lightweight essentials — no manifest, no instructions
        preamble_parts = [p for p in [profile_block, uk_block] if p]
        context_block = "\n\n".join(preamble_parts) if preamble_parts else ""

        output = {
            "status": "ok",
            "version": __version__,
            "mode": mode,
            "boot_type": "compact",
            "session_id": lib.session_id,
            "resumed": resumed,
            "total_entries": stats.get("total_entries", 0),
            "user_knowledge_entries": len(uk_entries),
            "past_sessions": len(past_sessions),
            "user_profile": user_profile_json,
            "context_block": context_block,
            "context_window": {
                "active_messages": window_state.active_messages,
                "pruned_messages": window_state.pruned_messages,
                "active_tokens": window_state.active_tokens,
                "budget_remaining": window_state.budget_remaining,
                "checkpoints": lib.context_window.total_checkpoints,
                "bridge_summary": bridge if bridge else None,
            },
        }

        # Housekeeping
        _cleanup_fuse_hidden()
        close_db(lib)
        print(json.dumps(output, indent=2))
        return

    # ─── Phase 10: Manifest-based context ────────────────────────────────
    from src.storage.manifest_manager import ManifestManager

    mm = ManifestManager(lib.rolodex.conn, lib.rolodex)
    token_budget = 20000  # retrieval budget
    available_budget = max(0, token_budget - fixed_token_cost)

    manifest = mm.get_latest_manifest()
    manifest_info = {}

    if stats.get("total_entries", 0) > 0:
        if manifest is None:
            # Super boot: first time or after invalidation
            manifest = mm.build_super_manifest(available_budget)
            manifest_info = {"boot_type": "super", "entries_selected": len(manifest.entries)}
        else:
            # Check for new entries since manifest was last updated
            new_count = mm.count_entries_after(manifest.updated_at)
            if new_count > 0:
                manifest = mm.build_incremental_manifest(manifest, available_budget)
                manifest_info = {"boot_type": "incremental", "new_entries": new_count, "entries_selected": len(manifest.entries)}
            else:
                manifest_info = {"boot_type": "cached", "entries_selected": len(manifest.entries)}

    # ─── Build context block from manifest entries ───────────────────────
    manifest_context = ""
    if manifest and manifest.entries:
        entry_ids = [me.entry_id for me in manifest.entries]
        entries = lib.rolodex.get_entries_by_ids(entry_ids)

        # Sort entries to match manifest slot_rank order
        id_to_rank = {me.entry_id: me.slot_rank for me in manifest.entries}
        entries.sort(key=lambda e: id_to_rank.get(e.id, 999))

        # Chain gap-fill: include reasoning chains only for underrepresented topics
        chains = _get_gap_fill_chains(lib, manifest)

        manifest_context = cb.build_context_block(entries, lib.session_id, chains)

    # ─── Full-context mode: return only the manifest payload ─────────────
    if full_context:
        output = {
            "status": "ok",
            "boot_type": "full_context",
            "session_id": lib.session_id,
            "manifest": manifest_info,
            "context_block": manifest_context,
        }
        close_db(lib)
        print(json.dumps(output, indent=2))
        return

    # ─── Default: full boot (legacy behavior) ────────────────────────────
    # Build final context: profile first, then user knowledge, then manifest entries
    preamble_parts = [p for p in [profile_block, uk_block] if p]
    if preamble_parts:
        context_block = "\n\n".join(preamble_parts) + "\n\n" + manifest_context if manifest_context else "\n\n".join(preamble_parts)
    else:
        context_block = manifest_context

    # ─── Load behavioral instructions (INSTRUCTIONS.md) ────────────────
    # In the installed version, this file ships with the app and provides
    # the model's behavioral contract (boot/ingest/recall protocol).
    # This replaces the workspace CLAUDE.md — instructions are versioned
    # with the application, not the workspace.
    instructions_block = _load_instructions()

    # ─── Version check (non-blocking, best-effort) ────────────────────
    update_info = _check_for_update()

    output = {
        "status": "ok",
        "version": __version__,
        "mode": mode,
        "boot_type": "full",
        "session_id": lib.session_id,
        "resumed": resumed,
        "total_entries": stats.get("total_entries", 0),
        "user_knowledge_entries": len(uk_entries),
        "past_sessions": len(past_sessions),
        "user_profile": user_profile_json,
        "context_block": context_block,
        "manifest": manifest_info,
        "context_window": {
            "active_messages": window_state.active_messages,
            "pruned_messages": window_state.pruned_messages,
            "active_tokens": window_state.active_tokens,
            "budget_remaining": window_state.budget_remaining,
            "checkpoints": lib.context_window.total_checkpoints,
            "bridge_summary": bridge if bridge else None,
        },
    }

    if instructions_block:
        output["instructions"] = instructions_block
    if update_info:
        output["update_available"] = update_info

    # Housekeeping: clean up FUSE artifacts on every boot
    _cleanup_fuse_hidden()

    close_db(lib)
    print(json.dumps(output, indent=2))


def _get_gap_fill_chains(lib, manifest):
    """
    Include reasoning chains only for topics underrepresented in the manifest.
    A topic is underrepresented if it has exactly 1 entry in the manifest.
    """
    if not manifest or not manifest.entries:
        return []

    # Count entries per topic in manifest
    topic_counts = {}
    for me in manifest.entries:
        if me.topic_label:
            topic_counts[me.topic_label] = topic_counts.get(me.topic_label, 0) + 1

    # Find underrepresented topics (1 entry only)
    thin_topics = {t for t, c in topic_counts.items() if c == 1}
    if not thin_topics:
        return []

    # Get recent chains and filter to those covering thin topics
    try:
        recent_sessions = lib.list_sessions(limit=5)
        chains = []
        for session in recent_sessions:
            session_chains = lib.rolodex.get_chains_for_session(session.session_id)
            for chain in session_chains:
                chain_topics = set(chain.topics) if chain.topics else set()
                if chain_topics & thin_topics:
                    chains.append(chain)
                    if len(chains) >= 3:  # Cap at 3 chains
                        return chains
        return chains
    except Exception:
        return []


def _get_entry_category(cat_str):
    """Resolve an EntryCategory from string, with lazy import."""
    from src.core.types import EntryCategory
    try:
        return EntryCategory(cat_str)
    except ValueError:
        return EntryCategory.NOTE


async def cmd_ingest(role, content, corrects_id=None, as_user_knowledge=False, is_summary=False):
    """Ingest a message into the rolodex."""
    lib, _ = _make_librarian()
    _ensure_session(lib, caller="ingest")

    entries = await lib.ingest(role, content)

    # Handle --summary flag: mark entries as non-verbatim (assistant paraphrase)
    if is_summary and entries:
        for entry in entries:
            entry.verbatim_source = False
            lib.rolodex.conn.execute(
                "UPDATE rolodex_entries SET verbatim_source = 0 WHERE id = ?",
                (entry.id,)
            )
        lib.rolodex.conn.commit()

    # Handle --user-knowledge flag
    if as_user_knowledge and entries:
        for entry in entries:
            lib.rolodex.update_entry_enrichment(
                entry_id=entry.id,
                category=_get_entry_category("user_knowledge"),
            )

    # Handle --corrects flag
    if corrects_id and entries:
        lib.rolodex.supersede_entry(corrects_id, entries[0].id)

    # Phase 9: Include checkpoint and window state
    window = lib.context_window.get_stats()

    # Housekeeping: run fuse cleanup every N ingestions
    checkpoint = window["last_checkpoint_turn"]
    if checkpoint > 0 and checkpoint % FUSE_CLEANUP_INTERVAL == 0:
        _cleanup_fuse_hidden()

    result = {
        "ingested": len(entries),
        "session_id": lib.session_id,
        "checkpoint": checkpoint,
        "total_checkpoints": window["checkpoints"],
    }
    if as_user_knowledge:
        result["user_knowledge"] = True

    close_db(lib)
    print(json.dumps(result))


async def cmd_batch_ingest(json_path):
    """Ingest multiple messages from a JSON file in a single process.

    Expects a JSON array of {"role": "user"|"assistant", "content": "..."} objects.
    Reads from file path, or from stdin if json_path is "-".
    """
    # Read JSON source
    if json_path == "-":
        raw = sys.stdin.read()
    else:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = f.read()

    messages = json.loads(raw)
    if not isinstance(messages, list):
        print(json.dumps({"error": "Expected JSON array of messages"}))
        sys.exit(1)

    lib, _ = _make_librarian()
    _ensure_session(lib, caller="batch-ingest")

    total_ingested = 0
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        if role not in ("user", "assistant") or not content.strip():
            continue
        entries = await lib.ingest(role, content)
        total_ingested += len(entries)

    close_db(lib)
    print(json.dumps({
        "ingested": total_ingested,
        "messages_processed": len(messages),
        "session_id": lib.session_id,
    }))


async def cmd_recall(query):
    """Search memory, return formatted context block.

    Phase 11: Wide-net-then-narrow search pattern.
    1. Query expansion generates multiple search variants + extracts entities
    2. Wide net: pull 15 results per variant (up to ~100 candidates)
    3. Re-ranker narrows using 5 signals: semantic, entity match, category,
       recency, access frequency
    4. Return top 5 re-ranked results
    """
    lib, _ = _make_librarian()
    _ensure_session(lib, caller="recall")

    # Phase 10+11: Query expansion with entity extraction
    from src.retrieval.query_expander import QueryExpander
    from src.retrieval.reranker import Reranker
    expander = QueryExpander()
    reranker = Reranker()
    expanded = expander.expand(query)

    # Phase 11: Wide net — pull 15 results per variant instead of 5
    WIDE_NET_LIMIT = 15
    all_candidates = []  # List of (entry, score) tuples
    seen_ids = set()

    for variant in expanded.variants:
        response = await lib.retrieve(variant, limit=WIDE_NET_LIMIT)
        if response.found:
            for entry in response.entries:
                if entry.id not in seen_ids:
                    seen_ids.add(entry.id)
                    # Use search position as a proxy score (first = highest)
                    score = 1.0 - (len(all_candidates) * 0.01)
                    all_candidates.append((entry, max(score, 0.1)))

    if all_candidates:
        # Phase 11: Re-rank the wide pool using multiple signals
        scored = reranker.rerank(
            candidates=all_candidates,
            query=query,
            query_entities=expanded.entities,
            category_bias=expanded.category_bias,
            limit=5,
        )

        # Extract entries from scored candidates
        all_entries = [sc.entry for sc in scored]

        # Phase 10: Track manifest access — mark recalled entries
        from src.storage.manifest_manager import ManifestManager
        mm = ManifestManager(lib.rolodex.conn, lib.rolodex)
        active_manifest = mm.get_latest_manifest()
        if active_manifest:
            for entry in all_entries:
                mm.mark_entry_accessed(active_manifest.manifest_id, entry.id)

        # Build a synthetic response for context block formatting
        from src.core.types import LibrarianResponse, LibrarianQuery
        synthetic_response = LibrarianResponse(
            found=True,
            entries=all_entries,
            query=LibrarianQuery(query_text=query),
        )

        # Phase 7: Chain results from the primary query
        primary_response = await lib.retrieve(query, limit=5)
        chains = getattr(primary_response, 'chains', [])
        if chains:
            print(f"[{len(chains)} reasoning chain(s) matched]")

        # Show search metadata
        entity_count = len(expanded.entities.all_entities) if expanded.entities else 0
        meta_parts = []
        if expanded.intent != "exploratory":
            meta_parts.append(f"intent: {expanded.intent}")
        meta_parts.append(f"{len(expanded.variants)} variants")
        meta_parts.append(f"{len(all_candidates)} candidates")
        if entity_count > 0:
            meta_parts.append(f"{entity_count} entities")
        print(f"[{' | '.join(meta_parts)}]")

        print(lib.get_context_block(synthetic_response))
    else:
        print("No relevant memories found.")

    close_db(lib)


async def cmd_stats():
    """Return session statistics as JSON."""
    lib, _ = _make_librarian()
    _ensure_session(lib, caller="stats")

    stats = lib.get_stats()
    close_db(lib)
    print(json.dumps(stats, indent=2, default=str))


async def cmd_end(summary=""):
    """End the current session. Refines the boot manifest with behavioral signal."""
    lib, _ = _make_librarian()
    _ensure_session(lib, caller="end")

    session_id = lib.session_id

    # Phase 10: Refine manifest before closing session
    from src.storage.manifest_manager import ManifestManager
    from src.core.types import estimate_tokens
    from src.retrieval.context_builder import ContextBuilder

    mm = ManifestManager(lib.rolodex.conn, lib.rolodex)
    current_manifest = mm.get_latest_manifest()
    manifest_refined = False

    if current_manifest:
        # Calculate available budget (same as boot)
        cb = ContextBuilder()
        profile = lib.rolodex.profile_get_all()
        profile_block = cb.build_profile_block(profile) if profile else ""
        uk_entries = lib.rolodex.get_user_knowledge_entries()
        uk_block = cb.build_user_knowledge_block(uk_entries) if uk_entries else ""
        fixed_cost = estimate_tokens(profile_block + uk_block)
        available_budget = max(0, 20000 - fixed_cost)

        mm.refine_manifest(current_manifest, session_id, available_budget)
        manifest_refined = True

    lib.end_session(summary=summary)
    clear_session_file()

    # Housekeeping: clean up FUSE artifacts on session end
    _cleanup_fuse_hidden()

    await lib.shutdown()
    print(json.dumps({
        "ended": session_id,
        "summary": summary,
        "manifest_refined": manifest_refined,
    }))


async def cmd_topics(subcmd, args):
    """Topic management commands."""
    lib, _ = _make_librarian()
    _ensure_session(lib, caller="topics")

    from src.indexing.topic_router import TopicRouter
    router = TopicRouter(
        conn=lib.rolodex.conn,
        embedding_manager=lib.embeddings,
    )

    if subcmd == "list":
        topics = router.list_topics(limit=50)
        if not topics:
            print("No topics yet. Topics emerge as entries are ingested and categorized.")
        else:
            print(json.dumps(topics, indent=2, default=str))

    elif subcmd == "show":
        if not args:
            print(json.dumps({"error": "Usage: topics show <topic-id>"}))
            sys.exit(1)
        topic = router.get_topic(args[0])
        if topic:
            entry_ids = router.get_entries_for_topic(args[0], limit=20)
            topic["entry_ids"] = entry_ids
            print(json.dumps(topic, indent=2, default=str))
        else:
            print(json.dumps({"error": f"Topic not found: {args[0]}"}))

    elif subcmd == "search":
        if not args:
            print(json.dumps({"error": "Usage: topics search \"<query>\""}))
            sys.exit(1)
        rows = lib.rolodex.conn.execute(
            """SELECT t.* FROM topics_fts fts
               JOIN topics t ON t.id = fts.topic_id
               WHERE topics_fts MATCH ?
               ORDER BY fts.rank LIMIT 10""",
            (args[0],)
        ).fetchall()
        results = [{"id": r["id"], "label": r["label"], "entry_count": r["entry_count"]} for r in rows]
        print(json.dumps(results, indent=2))

    elif subcmd == "stats":
        total = router.count_topics()
        unassigned = router.count_unassigned_entries()
        total_entries = lib.rolodex.conn.execute(
            "SELECT COUNT(*) as cnt FROM rolodex_entries"
        ).fetchone()["cnt"]
        coverage = ((total_entries - unassigned) / total_entries * 100) if total_entries > 0 else 0
        print(json.dumps({
            "total_topics": total,
            "total_entries": total_entries,
            "assigned_entries": total_entries - unassigned,
            "unassigned_entries": unassigned,
            "coverage_percent": round(coverage, 1),
        }, indent=2))

    else:
        print(json.dumps({"error": f"Unknown topics subcommand: {subcmd}. Use list|show|search|stats"}))
        sys.exit(1)

    close_db(lib)


async def cmd_scan(directory):
    """Scan a directory and ingest all readable text files into the rolodex.

    Walks the directory recursively, reads text-decodable files, chunks them
    using the existing content chunker, and ingests each file's content with
    source file metadata. Skips binaries, large files, and common ignore patterns.

    Outputs progress as newline-delimited JSON objects so the caller can
    stream updates, with a final summary object.
    """
    import time
    import pathlib

    # ─── Ignore patterns ────────────────────────────────────────────
    IGNORE_DIRS = {
        '.git', '.svn', '.hg', 'node_modules', '__pycache__', '.venv',
        'venv', 'env', '.env', '.tox', '.mypy_cache', '.pytest_cache',
        'dist', 'build', '.next', '.nuxt', '.output', 'target',
        '.idea', '.vscode', '.DS_Store', 'coverage', '.nyc_output',
        'egg-info', '.eggs', '.cache', '.parcel-cache', 'bower_components',
        '.terraform', '.sass-cache', 'vendor',
    }

    IGNORE_EXTENSIONS = {
        # Binaries / compiled
        '.pyc', '.pyo', '.so', '.dylib', '.dll', '.exe', '.o', '.a',
        '.class', '.jar', '.war',
        # Archives
        '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar',
        # Media
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg',
        '.mp3', '.mp4', '.wav', '.avi', '.mov', '.webm', '.webp',
        # Fonts
        '.woff', '.woff2', '.ttf', '.eot', '.otf',
        # Data blobs
        '.sqlite', '.db', '.db-wal', '.db-shm', '.pickle', '.pkl',
        # Minified / generated
        '.min.js', '.min.css', '.map',
        # Office binaries (handled separately if needed)
        '.docx', '.xlsx', '.pptx', '.pdf',
        # Lock files
        '.lock',
    }

    IGNORE_FILENAMES = {
        'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
        'Pipfile.lock', 'poetry.lock', 'composer.lock',
        '.DS_Store', 'Thumbs.db',
    }

    MAX_FILE_SIZE = 512 * 1024  # 512KB — skip huge files

    # ─── Walk and collect files ─────────────────────────────────────
    target = pathlib.Path(directory).resolve()
    if not target.is_dir():
        print(json.dumps({"error": f"Not a directory: {directory}"}))
        sys.exit(1)

    files_to_scan = []
    skipped_dirs = 0
    skipped_files = 0

    for root, dirs, files in os.walk(target):
        # Prune ignored directories in-place
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith('.')]

        for fname in files:
            fpath = os.path.join(root, fname)

            # Skip by filename
            if fname in IGNORE_FILENAMES:
                skipped_files += 1
                continue

            # Skip by extension
            suffix = pathlib.Path(fname).suffix.lower()
            if suffix in IGNORE_EXTENSIONS:
                skipped_files += 1
                continue

            # Skip by combined extension (.min.js, .min.css)
            if fname.endswith('.min.js') or fname.endswith('.min.css'):
                skipped_files += 1
                continue

            # Skip by size
            try:
                size = os.path.getsize(fpath)
                if size > MAX_FILE_SIZE or size == 0:
                    skipped_files += 1
                    continue
            except OSError:
                skipped_files += 1
                continue

            files_to_scan.append(fpath)

    total_files = len(files_to_scan)
    print(json.dumps({
        "event": "scan_start",
        "directory": str(target),
        "files_found": total_files,
        "skipped_files": skipped_files,
    }), flush=True)

    if total_files == 0:
        print(json.dumps({
            "event": "scan_complete",
            "files_scanned": 0,
            "files_ingested": 0,
            "entries_created": 0,
            "skipped_files": skipped_files,
            "elapsed_seconds": 0,
        }))
        return

    # ─── Init Librarian ─────────────────────────────────────────────
    lib, mode = _make_librarian()
    _ensure_session(lib, caller="scan")

    # ─── Scan and ingest ────────────────────────────────────────────
    start_time = time.time()
    files_ingested = 0
    total_entries = 0
    errors = 0

    for i, fpath in enumerate(files_to_scan):
        rel_path = os.path.relpath(fpath, target)

        # Try reading as UTF-8 text
        try:
            with open(fpath, 'r', encoding='utf-8', errors='strict') as f:
                content = f.read()
        except (UnicodeDecodeError, PermissionError, OSError):
            # Not text-decodable or not readable — skip
            skipped_files += 1
            continue

        if not content.strip():
            continue

        # Prefix content with source metadata for the Librarian
        source_header = f"[Source file: {rel_path}]\n\n"
        annotated_content = source_header + content

        # Ingest as "user" role (it's user's knowledge base)
        try:
            entries = await lib.ingest("user", annotated_content)
            files_ingested += 1
            total_entries += len(entries)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(json.dumps({
                    "event": "scan_error",
                    "file": rel_path,
                    "error": str(e),
                }), flush=True)
            continue

        # Progress update every 10 files
        if (i + 1) % 10 == 0 or (i + 1) == total_files:
            elapsed = time.time() - start_time
            print(json.dumps({
                "event": "scan_progress",
                "files_processed": i + 1,
                "files_total": total_files,
                "files_ingested": files_ingested,
                "entries_created": total_entries,
                "elapsed_seconds": round(elapsed, 1),
                "percent": round((i + 1) / total_files * 100, 1),
            }), flush=True)

    elapsed = time.time() - start_time
    close_db(lib)

    print(json.dumps({
        "event": "scan_complete",
        "files_scanned": total_files,
        "files_ingested": files_ingested,
        "entries_created": total_entries,
        "errors": errors,
        "skipped_files": skipped_files,
        "elapsed_seconds": round(elapsed, 1),
    }))


async def cmd_retag():
    """Re-index all existing entries with the current extraction pipeline.

    Walks every entry in the rolodex, re-runs the verbatim extractor's
    categorization and tag extraction (including Phase 11 entity extraction
    and attribution tagging), and updates the entry in place.

    Content and embeddings are untouched — only tags and categories are refreshed.
    This is a metadata-only migration, safe to run at any time.

    Designed to be re-run after any extraction pipeline improvements.
    """
    import time

    lib, _ = _make_librarian()
    _ensure_session(lib, caller="retag")

    from src.indexing.verbatim_extractor import VerbatimExtractor
    from src.core.types import ContentModality, EntryCategory

    extractor = VerbatimExtractor()

    # Fetch all entries directly from the DB
    rows = lib.rolodex.conn.execute(
        "SELECT id, content, category, tags, content_type FROM rolodex_entries"
    ).fetchall()

    total = len(rows)
    updated = 0
    errors = 0
    start_time = time.time()

    print(json.dumps({
        "event": "retag_start",
        "total_entries": total,
    }), flush=True)

    for i, row in enumerate(rows):
        entry_id = row["id"]
        content = row["content"]
        old_tags = json.loads(row["tags"]) if row["tags"] else []
        old_category = row["category"]

        # Determine modality from content_type field
        content_type_str = row["content_type"] or "prose"
        try:
            modality = ContentModality(content_type_str)
        except ValueError:
            modality = ContentModality.PROSE

        try:
            # Re-run extraction pipeline
            results = await extractor.extract(content, modality)
            if not results:
                continue

            new_category = results[0]["category"]
            new_tags = results[0]["tags"]

            # Check if anything changed
            category_changed = new_category != old_category
            tags_changed = set(new_tags) != set(old_tags)

            if category_changed or tags_changed:
                # Update in the rolodex
                update_kwargs = {}
                if tags_changed:
                    update_kwargs["tags"] = new_tags
                if category_changed:
                    try:
                        update_kwargs["category"] = EntryCategory(new_category)
                    except ValueError:
                        pass  # Keep old category if new one is invalid

                if update_kwargs:
                    lib.rolodex.update_entry_enrichment(
                        entry_id=entry_id,
                        **update_kwargs,
                    )
                    updated += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(json.dumps({
                    "event": "retag_error",
                    "entry_id": entry_id,
                    "error": str(e),
                }), flush=True)

        # Progress update every 50 entries
        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            print(json.dumps({
                "event": "retag_progress",
                "processed": i + 1,
                "total": total,
                "updated": updated,
                "elapsed_seconds": round(elapsed, 1),
                "percent": round((i + 1) / total * 100, 1),
            }), flush=True)

    elapsed = time.time() - start_time
    close_db(lib)

    print(json.dumps({
        "event": "retag_complete",
        "total_entries": total,
        "updated": updated,
        "unchanged": total - updated - errors,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
    }))


async def cmd_remember(content):
    """Ingest content as user_knowledge — privileged, always-on context.

    user_knowledge entries are:
    - Always loaded at boot (between profile and retrieved context)
    - Boosted 3x in search results
    - Never demoted from hot tier
    - Ideal for: preferences, biographical details, corrections, working style
    """
    lib, _ = _make_librarian()
    _ensure_session(lib, caller="remember")

    entries = await lib.ingest("user", content)

    # Recategorize as user_knowledge and promote to hot
    for entry in entries:
        lib.rolodex.update_entry_enrichment(
            entry_id=entry.id,
            category=_get_entry_category("user_knowledge"),
        )

    close_db(lib)
    print(json.dumps({
        "remembered": len(entries),
        "entry_ids": [e.id for e in entries],
        "session_id": lib.session_id,
        "content_preview": content[:120],
    }))


async def cmd_correct(old_entry_id, corrected_text):
    """Supersede a factually wrong entry with corrected content.

    The old entry is soft-deleted (hidden from search, kept in DB).
    Use for error corrections — NOT for reasoning chains where the
    evolution of thought should be preserved.
    """
    lib, _ = _make_librarian()
    _ensure_session(lib, caller="correct")

    # Ingest the corrected content as user_knowledge
    entries = await lib.ingest("user", corrected_text)
    if not entries:
        close_db(lib)
        print(json.dumps({"error": "Failed to create corrected entry"}))
        return

    new_entry = entries[0]
    lib.rolodex.update_entry_enrichment(
        entry_id=new_entry.id,
        category=_get_entry_category("user_knowledge"),
    )

    # Supersede the old entry
    existed = lib.rolodex.supersede_entry(old_entry_id, new_entry.id)

    close_db(lib)
    print(json.dumps({
        "corrected": existed,
        "old_entry_id": old_entry_id,
        "new_entry_id": new_entry.id,
        "session_id": lib.session_id,
    }))


async def cmd_profile(subcmd, args):
    """Manage user profile key-value pairs."""
    lib, _ = _make_librarian()
    _ensure_session(lib, caller="profile")
    session_id = load_session_id()

    if subcmd == "set":
        if len(args) < 2:
            print(json.dumps({"error": "Usage: profile set <key> <value>"}))
            close_db(lib)
            return
        key = args[0]
        value = " ".join(args[1:])
        lib.rolodex.profile_set(key, value, session_id=session_id)
        print(json.dumps({"profile_set": key, "value": value}))

    elif subcmd == "show":
        profile = lib.rolodex.profile_get_all()
        if not profile:
            print(json.dumps({"profile": {}, "message": "No profile entries yet. Use 'profile set <key> <value>' to add."}))
        else:
            print(json.dumps({"profile": {k: v["value"] for k, v in profile.items()}}, indent=2))

    elif subcmd == "delete":
        if not args:
            print(json.dumps({"error": "Usage: profile delete <key>"}))
            close_db(lib)
            return
        key = args[0]
        existed = lib.rolodex.profile_delete(key)
        print(json.dumps({"profile_deleted": key, "existed": existed}))

    else:
        print(json.dumps({"error": f"Unknown profile subcommand: {subcmd}. Use set|show|delete"}))

    close_db(lib)


async def cmd_window():
    """Show context window state — what's active vs pruned."""
    lib, _ = _make_librarian()
    _ensure_session(lib, caller="window")

    state = lib.context_window.get_state(lib.state.messages)
    result = {
        "active_messages": state.active_messages,
        "pruned_messages": state.pruned_messages,
        "active_tokens": state.active_tokens,
        "pruned_tokens": state.pruned_tokens,
        "budget_remaining": state.budget_remaining,
        "last_checkpoint_turn": state.last_checkpoint_turn,
        "checkpoints": lib.context_window.total_checkpoints,
        "bridge_summary_tokens": state.bridge_summary_tokens,
        "bridge_summary": lib.context_window.bridge_summary or "(none — nothing pruned yet)",
    }
    print(json.dumps(result, indent=2))
    close_db(lib)


async def cmd_manifest(subcmd, args):
    """Manifest management commands."""
    lib, _ = _make_librarian()
    _ensure_session(lib, caller="manifest")

    from src.storage.manifest_manager import ManifestManager
    mm = ManifestManager(lib.rolodex.conn, lib.rolodex)

    if subcmd == "show":
        manifest = mm.get_latest_manifest()
        if not manifest:
            print(json.dumps({"message": "No manifest exists. Run boot to create one."}))
        else:
            entries_detail = []
            for me in manifest.entries:
                entries_detail.append({
                    "rank": me.slot_rank,
                    "entry_id": me.entry_id[:8],
                    "score": round(me.composite_score, 4),
                    "tokens": me.token_cost,
                    "topic": me.topic_label or "(unassigned)",
                    "reason": me.selection_reason,
                    "accessed": me.was_accessed,
                })
            print(json.dumps({
                "manifest_id": manifest.manifest_id,
                "type": manifest.manifest_type,
                "entry_count": len(manifest.entries),
                "total_token_cost": manifest.total_token_cost,
                "topics_represented": len(manifest.topic_summary),
                "created_at": manifest.created_at.isoformat(),
                "updated_at": manifest.updated_at.isoformat(),
                "source_session": manifest.source_session_id,
                "entries": entries_detail,
                "topic_summary": manifest.topic_summary,
            }, indent=2))

    elif subcmd == "fresh":
        count = mm.invalidate()
        print(json.dumps({
            "invalidated": count,
            "message": "Manifest cleared. Next boot will run full super boot.",
        }))

    elif subcmd == "stats":
        stats = mm.get_stats()
        print(json.dumps(stats, indent=2))

    else:
        print(json.dumps({
            "error": f"Unknown manifest subcommand: {subcmd}. Use show|fresh|stats"
        }))
        sys.exit(1)

    close_db(lib)


async def cmd_schema():
    """Dump the database schema — table names, columns, types, and indexes.

    Provides a quick reference for the DB structure without requiring
    direct sqlite3 access or schema guesswork.
    """
    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(DB_PATH)
    conn.row_factory = _sqlite3.Row

    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()

    schema = {}
    for t in tables:
        tname = t["name"]
        cols = conn.execute(f"PRAGMA table_info([{tname}])").fetchall()
        schema[tname] = [
            {"name": c["name"], "type": c["type"], "pk": bool(c["pk"]), "notnull": bool(c["notnull"])}
            for c in cols
        ]

    # Also grab indexes
    indexes = conn.execute(
        "SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL ORDER BY tbl_name"
    ).fetchall()
    index_list = [{"name": ix["name"], "table": ix["tbl_name"]} for ix in indexes]

    conn.close()
    print(json.dumps({
        "tables": schema,
        "indexes": index_list,
        "db_path": DB_PATH,
    }, indent=2))


async def cmd_history(subcmd, args):
    """Query session history without direct DB access.

    Subcommands:
        first       — Show the earliest session (date, ID)
        recent [N]  — Show the N most recent sessions (default 10)
        count       — Total session count
        range       — First and last session dates + total count
    """
    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(DB_PATH)
    conn.row_factory = _sqlite3.Row

    if subcmd == "first":
        row = conn.execute(
            "SELECT id, created_at FROM conversations ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if row:
            print(json.dumps({"first_session": {"id": row["id"], "created_at": row["created_at"]}}))
        else:
            print(json.dumps({"first_session": None}))

    elif subcmd == "recent":
        limit = int(args[0]) if args else 10
        rows = conn.execute(
            "SELECT id, created_at FROM conversations ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        print(json.dumps({
            "recent_sessions": [{"id": r["id"], "created_at": r["created_at"]} for r in rows]
        }, indent=2))

    elif subcmd == "count":
        row = conn.execute("SELECT COUNT(*) as cnt FROM conversations").fetchone()
        print(json.dumps({"total_sessions": row["cnt"]}))

    elif subcmd == "range":
        first = conn.execute("SELECT MIN(created_at) as dt FROM conversations").fetchone()
        last = conn.execute("SELECT MAX(created_at) as dt FROM conversations").fetchone()
        count = conn.execute("SELECT COUNT(*) as cnt FROM conversations").fetchone()
        print(json.dumps({
            "first_session": first["dt"],
            "last_session": last["dt"],
            "total_sessions": count["cnt"],
        }))

    else:
        print(json.dumps({"error": f"Unknown history subcommand: {subcmd}. Use first|recent|count|range"}))
        sys.exit(1)

    conn.close()


async def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: librarian_cli.py <boot|ingest|recall|stats|end|topics|window|schema|history> [args]"}))
        sys.exit(1)

    cmd = sys.argv[1].lower()

    try:
        if cmd == "boot":
            compact = "--compact" in sys.argv[2:]
            full_context = "--full-context" in sys.argv[2:]
            await cmd_boot(compact=compact, full_context=full_context)

        elif cmd == "ingest":
            if len(sys.argv) < 4:
                print(json.dumps({"error": "Usage: librarian_cli.py ingest <user|assistant> \"<text>\" [--user-knowledge] [--corrects <id>] [--summary]"}))
                sys.exit(1)
            role = sys.argv[2].lower()
            content = sys.argv[3]
            if role not in ("user", "assistant"):
                print(json.dumps({"error": "Role must be 'user' or 'assistant'"}))
                sys.exit(1)
            # Parse optional flags
            as_user_knowledge = False
            corrects_id = None
            is_summary = False
            remaining = sys.argv[4:]
            i = 0
            while i < len(remaining):
                if remaining[i] == "--user-knowledge":
                    as_user_knowledge = True
                elif remaining[i] == "--summary":
                    is_summary = True
                elif remaining[i] == "--corrects" and i + 1 < len(remaining):
                    i += 1
                    corrects_id = remaining[i]
                i += 1
            await cmd_ingest(role, content, corrects_id=corrects_id, as_user_knowledge=as_user_knowledge, is_summary=is_summary)

        elif cmd == "batch-ingest":
            if len(sys.argv) < 3:
                print(json.dumps({"error": "Usage: librarian_cli.py batch-ingest <file.json|->"}))
                sys.exit(1)
            await cmd_batch_ingest(sys.argv[2])

        elif cmd == "recall":
            if len(sys.argv) < 3:
                print(json.dumps({"error": "Usage: librarian_cli.py recall \"<query>\""}))
                sys.exit(1)
            await cmd_recall(sys.argv[2])

        elif cmd == "stats":
            await cmd_stats()

        elif cmd == "end":
            summary = sys.argv[2] if len(sys.argv) > 2 else ""
            await cmd_end(summary)

        elif cmd == "topics":
            subcmd = sys.argv[2].lower() if len(sys.argv) > 2 else "list"
            args = sys.argv[3:] if len(sys.argv) > 3 else []
            await cmd_topics(subcmd, args)

        elif cmd == "window":
            await cmd_window()

        elif cmd == "scan":
            if len(sys.argv) < 3:
                print(json.dumps({"error": "Usage: librarian_cli.py scan <directory>"}))
                sys.exit(1)
            await cmd_scan(sys.argv[2])

        elif cmd == "retag":
            await cmd_retag()

        elif cmd == "remember":
            if len(sys.argv) < 3:
                print(json.dumps({"error": "Usage: librarian_cli.py remember \"<fact about the user>\""}))
                sys.exit(1)
            await cmd_remember(sys.argv[2])

        elif cmd == "correct":
            if len(sys.argv) < 4:
                print(json.dumps({"error": "Usage: librarian_cli.py correct <old_entry_id> \"<corrected text>\""}))
                sys.exit(1)
            await cmd_correct(sys.argv[2], sys.argv[3])

        elif cmd == "profile":
            subcmd = sys.argv[2].lower() if len(sys.argv) > 2 else "show"
            args = sys.argv[3:] if len(sys.argv) > 3 else []
            await cmd_profile(subcmd, args)

        elif cmd == "manifest":
            subcmd = sys.argv[2].lower() if len(sys.argv) > 2 else "stats"
            args = sys.argv[3:] if len(sys.argv) > 3 else []
            await cmd_manifest(subcmd, args)

        elif cmd == "schema":
            await cmd_schema()

        elif cmd == "history":
            subcmd = sys.argv[2].lower() if len(sys.argv) > 2 else "range"
            args = sys.argv[3:] if len(sys.argv) > 3 else []
            await cmd_history(subcmd, args)

        else:
            print(json.dumps({"error": f"Unknown command: {cmd}. Use boot|ingest|recall|remember|correct|profile|scan|retag|stats|end|topics|window|manifest|schema|history"}))
            sys.exit(1)

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
