#!/usr/bin/env python3


import asyncio
import json
import os
import sys


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


_TheLibrarian = None
_LibrarianConfig = None


def _lazy_imports():

    global _TheLibrarian, _LibrarianConfig
    if _TheLibrarian is None:
        from src.core.librarian import TheLibrarian
        from src.utils.config import LibrarianConfig
        _TheLibrarian = TheLibrarian
        _LibrarianConfig = LibrarianConfig


def _load_config():

    _lazy_imports()
    env_file = os.path.join(SCRIPT_DIR, ".env")
    return _LibrarianConfig.from_env(env_path=env_file)


DB_PATH = os.environ.get("LIBRARIAN_DB_PATH", os.path.join(SCRIPT_DIR, "rolodex.db"))
SESSION_FILE = os.environ.get("LIBRARIAN_SESSION_FILE", os.path.join(SCRIPT_DIR, ".cowork_session"))


def _build_adapter():

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None, "verbatim"
    try:
        from src.indexing.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(api_key=api_key), "enhanced"
    except Exception:
        return None, "verbatim"


def _make_librarian():

    _lazy_imports()
    adapter, mode = _build_adapter()
    return _TheLibrarian(db_path=DB_PATH, llm_adapter=adapter), mode


def load_session_id():

    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r") as f:
                data = json.load(f)
            return data.get("session_id")
        except (json.JSONDecodeError, IOError):
            return None
    return None


def save_session_id(session_id):

    with open(SESSION_FILE, "w") as f:
        json.dump({"session_id": session_id}, f)


def clear_session_file():

    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)


def close_db(lib):

    lib.rolodex.close()


async def cmd_boot():

    lib, mode = _make_librarian()


    existing_id = load_session_id()
    resumed = False
    if existing_id:
        info = lib.resume_session(existing_id)
        if info:
            resumed = True

    save_session_id(lib.session_id)


    stats = lib.get_stats()
    past_sessions = lib.list_sessions(limit=10)


    context_block = ""
    if stats.get("total_entries", 0) > 0:

        for query in ["user preferences decisions", "project context recent work"]:
            response = await lib.retrieve(query, limit=3)
            if response.found:
                context_block = lib.get_context_block(response)
                break


    window_state = lib.context_window.get_state(lib.state.messages)
    bridge = lib.context_window.bridge_summary

    output = {
        "status": "ok",
        "mode": mode,
        "session_id": lib.session_id,
        "resumed": resumed,
        "total_entries": stats.get("total_entries", 0),
        "past_sessions": len(past_sessions),
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

    close_db(lib)
    print(json.dumps(output, indent=2))


async def cmd_ingest(role, content):

    lib, _ = _make_librarian()


    session_id = load_session_id()
    if session_id:
        lib.resume_session(session_id)
    else:
        save_session_id(lib.session_id)

    entries = await lib.ingest(role, content)


    window = lib.context_window.get_stats()
    close_db(lib)
    print(json.dumps({
        "ingested": len(entries),
        "session_id": lib.session_id,
        "checkpoint": window["last_checkpoint_turn"],
        "total_checkpoints": window["checkpoints"],
    }))


async def cmd_batch_ingest(json_path):


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


    session_id = load_session_id()
    if session_id:
        lib.resume_session(session_id)
    else:
        save_session_id(lib.session_id)

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

    lib, _ = _make_librarian()

    session_id = load_session_id()
    if session_id:
        lib.resume_session(session_id)

    response = await lib.retrieve(query, limit=5)

    if response.found:

        chains = getattr(response, 'chains', [])
        if chains:
            print(f"[{len(chains)} reasoning chain(s) matched]")
        print(lib.get_context_block(response))
    else:
        print("No relevant memories found.")

    close_db(lib)


async def cmd_stats():

    lib, _ = _make_librarian()

    session_id = load_session_id()
    if session_id:
        lib.resume_session(session_id)

    stats = lib.get_stats()
    close_db(lib)
    print(json.dumps(stats, indent=2, default=str))


async def cmd_end(summary=""):

    lib, _ = _make_librarian()

    session_id = load_session_id()
    if session_id:
        lib.resume_session(session_id)

    lib.end_session(summary=summary)
    ended_id = lib.session_id
    clear_session_file()
    await lib.shutdown()
    print(json.dumps({"ended": ended_id, "summary": summary}))


async def cmd_topics(subcmd, args):

    lib, _ = _make_librarian()

    session_id = load_session_id()
    if session_id:
        lib.resume_session(session_id)

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


    import time
    import pathlib


    IGNORE_DIRS = {
        '.git', '.svn', '.hg', 'node_modules', '__pycache__', '.venv',
        'venv', 'env', '.env', '.tox', '.mypy_cache', '.pytest_cache',
        'dist', 'build', '.next', '.nuxt', '.output', 'target',
        '.idea', '.vscode', '.DS_Store', 'coverage', '.nyc_output',
        'egg-info', '.eggs', '.cache', '.parcel-cache', 'bower_components',
        '.terraform', '.sass-cache', 'vendor',
    }

    IGNORE_EXTENSIONS = {

        '.pyc', '.pyo', '.so', '.dylib', '.dll', '.exe', '.o', '.a',
        '.class', '.jar', '.war',

        '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar',

        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg',
        '.mp3', '.mp4', '.wav', '.avi', '.mov', '.webm', '.webp',

        '.woff', '.woff2', '.ttf', '.eot', '.otf',

        '.sqlite', '.db', '.db-wal', '.db-shm', '.pickle', '.pkl',

        '.min.js', '.min.css', '.map',

        '.docx', '.xlsx', '.pptx', '.pdf',

        '.lock',
    }

    IGNORE_FILENAMES = {
        'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
        'Pipfile.lock', 'poetry.lock', 'composer.lock',
        '.DS_Store', 'Thumbs.db',
    }

    MAX_FILE_SIZE = 512 * 1024


    target = pathlib.Path(directory).resolve()
    if not target.is_dir():
        print(json.dumps({"error": f"Not a directory: {directory}"}))
        sys.exit(1)

    files_to_scan = []
    skipped_dirs = 0
    skipped_files = 0

    for root, dirs, files in os.walk(target):

        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith('.')]

        for fname in files:
            fpath = os.path.join(root, fname)


            if fname in IGNORE_FILENAMES:
                skipped_files += 1
                continue


            suffix = pathlib.Path(fname).suffix.lower()
            if suffix in IGNORE_EXTENSIONS:
                skipped_files += 1
                continue


            if fname.endswith('.min.js') or fname.endswith('.min.css'):
                skipped_files += 1
                continue


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


    lib, mode = _make_librarian()
    session_id = load_session_id()
    if session_id:
        lib.resume_session(session_id)
    else:
        save_session_id(lib.session_id)


    start_time = time.time()
    files_ingested = 0
    total_entries = 0
    errors = 0

    for i, fpath in enumerate(files_to_scan):
        rel_path = os.path.relpath(fpath, target)


        try:
            with open(fpath, 'r', encoding='utf-8', errors='strict') as f:
                content = f.read()
        except (UnicodeDecodeError, PermissionError, OSError):

            skipped_files += 1
            continue

        if not content.strip():
            continue


        source_header = f"[Source file: {rel_path}]\n\n"
        annotated_content = source_header + content


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


async def cmd_window():

    lib, _ = _make_librarian()
    session_id = load_session_id()
    if session_id:
        lib.resume_session(session_id)

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


async def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: librarian_cli.py <boot|ingest|recall|stats|end|topics|window> [args]"}))
        sys.exit(1)

    cmd = sys.argv[1].lower()

    try:
        if cmd == "boot":
            await cmd_boot()

        elif cmd == "ingest":
            if len(sys.argv) < 4:
                print(json.dumps({"error": "Usage: librarian_cli.py ingest <user|assistant> \"<text>\""}))
                sys.exit(1)
            role = sys.argv[2].lower()
            content = sys.argv[3]
            if role not in ("user", "assistant"):
                print(json.dumps({"error": "Role must be 'user' or 'assistant'"}))
                sys.exit(1)
            await cmd_ingest(role, content)

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

        else:
            print(json.dumps({"error": f"Unknown command: {cmd}. Use boot|ingest|recall|scan|stats|end|topics|window"}))
            sys.exit(1)

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
