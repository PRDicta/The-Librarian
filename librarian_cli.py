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
_deps_checked = False


def _ensure_dependencies():
    """Auto-install pip dependencies on first run inside Cowork VM.

    The frozen Windows .exe bundles everything, but when Cowork's Linux VM
    runs the Python source directly, packages like sentence-transformers and
    torch may not be installed.

    Strategy: install core deps first (small, always succeed), then attempt
    ML deps separately (large, may fail on disk-constrained VMs). If ML
    deps fail, the embedding system falls back to hash-based mode.

    Only runs once per process. Skips entirely if we're in a frozen build.
    """
    global _deps_checked
    if _deps_checked or getattr(sys, 'frozen', False):
        return
    _deps_checked = True

    # Check for cached dependency status (avoids re-probing every boot)
    deps_flag = os.path.join(SCRIPT_DIR, ".deps_ok")
    if os.path.isfile(deps_flag):
        try:
            # Verify the flag is fresh (less than 7 days old)
            import time as _time
            age = _time.time() - os.path.getmtime(deps_flag)
            if age < 7 * 86400:  # 7 days
                return
        except OSError:
            pass

    import subprocess as _sp
    pip_base = [sys.executable, "-m", "pip", "install",
                "--break-system-packages", "-q", "--disable-pip-version-check"]

    # Step 1: Check if core deps are present
    core_missing = False
    for probe in ["anthropic", "rich"]:
        try:
            __import__(probe)
        except ImportError:
            core_missing = True
            break

    # Step 2: Install core deps if needed (lightweight, ~5 MB)
    if core_missing:
        req_core = os.path.join(SCRIPT_DIR, "requirements.txt")
        if os.path.isfile(req_core):
            try:
                _sp.run(pip_base + ["-r", req_core, "--no-cache-dir"],
                        timeout=120, capture_output=True)
            except Exception:
                pass

    # Step 3: Check if we have a viable embedding backend
    # Priority: sentence-transformers > onnxruntime+tokenizers > hash fallback
    has_embeddings = False
    try:
        import sentence_transformers  # noqa: F401
        has_embeddings = True
    except ImportError:
        pass

    if not has_embeddings:
        # Check if ONNX Runtime AND tokenizers are both available
        onnx_ready = False
        try:
            import onnxruntime  # noqa: F401
            import tokenizers  # noqa: F401
            onnx_ready = True
            has_embeddings = True
        except ImportError:
            pass

        # If ONNX Runtime exists but tokenizers is missing, install just tokenizers
        if not onnx_ready:
            try:
                import onnxruntime  # noqa: F401
                # onnxruntime present but tokenizers missing — lightweight install
                req_onnx = os.path.join(SCRIPT_DIR, "requirements-onnx.txt")
                if os.path.isfile(req_onnx):
                    try:
                        _sp.run(pip_base + ["-r", req_onnx, "--no-cache-dir"],
                                timeout=120, capture_output=True)
                        has_embeddings = True
                    except Exception:
                        pass
            except ImportError:
                pass  # No onnxruntime at all

    # Step 4: Only attempt heavy ML deps if no embedding backend exists
    if not has_embeddings:
        # Try ONNX tier first (lightweight, ~55 MB)
        req_onnx = os.path.join(SCRIPT_DIR, "requirements-onnx.txt")
        if os.path.isfile(req_onnx):
            try:
                _sp.run(pip_base + ["-r", req_onnx, "--no-cache-dir"],
                        timeout=120, capture_output=True)
                # Check if it worked
                try:
                    import onnxruntime  # noqa: F401
                    import tokenizers  # noqa: F401
                    has_embeddings = True
                except ImportError:
                    pass
            except Exception:
                pass

    if not has_embeddings:
        # Last resort: full ML stack (~2 GB)
        req_ml = os.path.join(SCRIPT_DIR, "requirements-ml.txt")
        if os.path.isfile(req_ml):
            try:
                _sp.run(pip_base + ["-r", req_ml, "--no-cache-dir"],
                        timeout=300, capture_output=True)
            except Exception:
                pass  # ML deps failed (likely disk space) — hash fallback will activate

    # Write cache flag so subsequent boots skip all of this
    try:
        with open(deps_flag, "w") as f:
            f.write("ok")
    except OSError:
        pass  # FUSE mount may not support this — non-fatal


def _lazy_imports():
    """Import heavy modules on first use."""
    global _TheLibrarian, _LibrarianConfig
    if _TheLibrarian is None:
        _ensure_dependencies()
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


def _clean_db_satellites(db_path):
    """Best-effort removal of journal/wal/shm files alongside a DB."""
    for suffix in ("-wal", "-journal", "-shm"):
        try:
            p = db_path + suffix
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


def _check_db_health(db_path):
    """Detect and recover from corrupt/0-byte DB files.

    Returns the db_path to use (may differ from input if recovery required
    overwrite instead of delete). Returns None if the file doesn't exist yet
    (caller should let TheLibrarian create a fresh one).
    """
    if not os.path.exists(db_path):
        return db_path  # Doesn't exist yet — will be created fresh

    is_corrupt = False

    if os.path.getsize(db_path) == 0:
        is_corrupt = True
        detail = "0-byte rolodex.db"
    else:
        # Quick SQLite header check
        try:
            with open(db_path, "rb") as f:
                header = f.read(16)
            if not header.startswith(b"SQLite format 3"):
                is_corrupt = True
                detail = "non-SQLite rolodex.db"
        except OSError:
            is_corrupt = True
            detail = "unreadable rolodex.db"

    if not is_corrupt:
        return db_path

    # Strategy 1: try to delete the corrupt file
    try:
        os.remove(db_path)
        _clean_db_satellites(db_path)
        print(json.dumps({"housekeeping": "corrupt_db_recovery",
                          "detail": f"Removed {detail}"}),
              file=sys.stderr)
        return db_path
    except OSError:
        pass

    # Strategy 2: overwrite with a minimal valid SQLite DB
    # (for filesystems that allow write but not delete)
    try:
        import sqlite3
        import tempfile
        # Create a valid DB in a temp location, then copy over
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name
        conn = sqlite3.connect(tmp_path)
        conn.execute("CREATE TABLE _recovery (id INTEGER PRIMARY KEY)")
        conn.execute("DROP TABLE _recovery")
        conn.close()
        import shutil
        shutil.copy2(tmp_path, db_path)
        os.remove(tmp_path)
        _clean_db_satellites(db_path)
        print(json.dumps({"housekeeping": "corrupt_db_recovery",
                          "detail": f"Overwrote {detail} with fresh DB"}),
              file=sys.stderr)
        return db_path
    except OSError:
        pass

    # Strategy 3: neither delete nor overwrite worked — this path is unusable
    return None


def _test_sqlite_writable(db_path):
    """Quick check: can SQLite actually open and write to this path?

    Returns True if a table can be created and dropped, False otherwise.
    """
    import sqlite3
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.execute("CREATE TABLE IF NOT EXISTS _probe (id INTEGER PRIMARY KEY)")
        conn.execute("DROP TABLE IF EXISTS _probe")
        conn.close()
        return True
    except Exception:
        return False


def _resolve_db_path():
    """Determine the best writable DB path, with fallback chain.

    Order:
    1. DB_PATH (default: alongside librarian_cli.py, or LIBRARIAN_DB_PATH env)
    2. If corrupt → try to recover in-place
    3. If mounted FS blocks SQLite → fall back to session-local writable path
    4. Last resort → /tmp/rolodex.db (ephemeral, won't persist)

    Returns (db_path, fallback_used) tuple.
    """
    # Step 1: health check (handles corrupt/0-byte files)
    resolved = _check_db_health(DB_PATH)

    if resolved is not None:
        # Step 2: verify SQLite can actually operate here
        if _test_sqlite_writable(resolved):
            return resolved, False

        # SQLite can't write — mounted FS issue
        print(json.dumps({"housekeeping": "sqlite_fs_incompatible",
                          "detail": f"SQLite cannot operate at {resolved}, trying fallback"}),
              file=sys.stderr)

    # Step 3: try session-local writable path (persists within session)
    # /sessions/<slug>/ root is writable even when mnt/ has FS constraints
    session_root = None
    cwd = os.getcwd()
    # Detect session root from CWD or DB_PATH
    for path in [cwd, DB_PATH]:
        parts = path.split("/sessions/")
        if len(parts) > 1:
            slug = parts[1].split("/")[0]
            candidate = f"/sessions/{slug}/rolodex.db"
            if _test_sqlite_writable(candidate):
                session_root = candidate
                break

    if session_root:
        # Restore DB from SQL dump if available (binary copies through FUSE corrupt).
        # The sync-back process writes a SQL text dump + manifest to the mount.
        # On boot, we rebuild from that dump instead of copying the binary DB.
        import sqlite3 as _sqlite3
        mount_dir = os.path.dirname(DB_PATH)
        manifest_path = os.path.join(mount_dir, "rolodex_sync_manifest.json")
        dump_path = os.path.join(mount_dir, "rolodex_dump.sql")

        # Check if local DB has real data (not just an empty probe file).
        # _test_sqlite_writable creates an 8KB file as a side effect, so
        # we can't rely on file existence or size — check actual entry count.
        local_entry_count = 0
        if os.path.exists(session_root) and os.path.getsize(session_root) > 0:
            try:
                _probe = _sqlite3.connect(session_root)
                local_entry_count = _probe.execute(
                    "SELECT count(*) FROM rolodex_entries"
                ).fetchone()[0]
                _probe.close()
            except Exception:
                local_entry_count = 0  # table doesn't exist or DB is corrupt

        should_rebuild = False
        rebuild_reason = ""

        if os.path.exists(manifest_path) and os.path.exists(dump_path):
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                dump_size = os.path.getsize(dump_path)
                manifest_entry_count = manifest.get("entry_count", 0)
                # Verify dump integrity
                if dump_size == manifest.get("dump_size", -1):
                    import hashlib
                    dump_hash = hashlib.md5(open(dump_path, 'rb').read()).hexdigest()
                    if dump_hash == manifest.get("dump_md5", ""):
                        if local_entry_count == 0:
                            # No real data locally — rebuild from dump
                            should_rebuild = True
                            rebuild_reason = "no_local_data"
                        elif manifest_entry_count > local_entry_count:
                            # Dump has more entries — another session synced updates
                            should_rebuild = True
                            rebuild_reason = f"dump_has_more_entries (dump={manifest_entry_count}, local={local_entry_count})"
                    else:
                        print(json.dumps({"housekeeping": "dump_hash_mismatch",
                                          "detail": f"Expected {manifest.get('dump_md5')}, got {dump_hash}"}),
                              file=sys.stderr)
            except (json.JSONDecodeError, IOError, OSError) as e:
                print(json.dumps({"housekeeping": "manifest_read_failed",
                                  "detail": str(e)}),
                      file=sys.stderr)

        if should_rebuild:
            try:
                # Remove stale local DB if it exists
                if os.path.exists(session_root):
                    os.remove(session_root)
                # Rebuild from SQL dump
                with open(dump_path, 'r') as f:
                    sql = f.read()
                db = _sqlite3.connect(session_root)
                db.executescript(sql)
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                db.close()
                rebuilt_size = os.path.getsize(session_root)
                # Quick verification
                db = _sqlite3.connect(session_root)
                entry_count = db.execute("SELECT count(*) FROM rolodex_entries").fetchone()[0]
                db.close()
                print(json.dumps({"housekeeping": "db_rebuilt_from_dump",
                                  "detail": f"Rebuilt {session_root} from {dump_path}",
                                  "reason": rebuild_reason,
                                  "size_bytes": rebuilt_size,
                                  "entry_count": entry_count}),
                      file=sys.stderr)
            except Exception as e:
                print(json.dumps({"housekeeping": "db_rebuild_failed",
                                  "detail": str(e)}),
                      file=sys.stderr)
                # Fall through — try binary copy as last resort
                if not (os.path.exists(session_root) and os.path.getsize(session_root) > 0):
                    should_rebuild = False  # Let binary fallback try

        # Fallback: binary copy if no dump available and no local DB
        if not should_rebuild and local_entry_count == 0:
            mount_available = resolved and os.path.exists(resolved) and os.path.getsize(resolved) > 0
            if mount_available:
                import shutil
                try:
                    shutil.copy2(resolved, session_root)
                    print(json.dumps({"housekeeping": "db_auto_copy",
                                      "detail": f"Copied {resolved} → {session_root} (binary fallback)",
                                      "warning": "Binary copy through FUSE may corrupt — prefer SQL dump",
                                      "size_bytes": os.path.getsize(session_root)}),
                          file=sys.stderr)
                except Exception as e:
                    print(json.dumps({"housekeeping": "db_auto_copy_failed",
                                      "detail": str(e)}),
                          file=sys.stderr)

        print(json.dumps({"housekeeping": "db_fallback",
                          "detail": f"Using session-local DB at {session_root}",
                          "warning": "Changes will auto-sync back via SQL dump to mounted folder"}),
              file=sys.stderr)
        return session_root, True

    # Step 4: last resort — /tmp
    tmp_path = "/tmp/rolodex.db"
    if _test_sqlite_writable(tmp_path):
        print(json.dumps({"housekeeping": "db_fallback",
                          "detail": f"Using ephemeral DB at {tmp_path}",
                          "warning": "DB will not persist after session ends"}),
              file=sys.stderr)
        return tmp_path, True

    # Nothing works — let it fail naturally so the error is visible
    return DB_PATH, False


def _make_librarian():
    """Create TheLibrarian with adapter if available."""
    _lazy_imports()
    db_path, fallback_used = _resolve_db_path()
    adapter, mode = _build_adapter()
    lib = _TheLibrarian(db_path=db_path, llm_adapter=adapter)
    if fallback_used:
        lib._db_fallback = True
        lib._db_original_path = DB_PATH
    return lib, mode


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


def _sync_db_back(lib=None, force=False):
    """Sync session-local DB back to the mounted folder via SQL dump.

    Binary SQLite file copies through FUSE are unreliable — the filesystem
    silently truncates or corrupts B-tree pages even when writes report
    success. Instead, we dump the DB as SQL text (which FUSE handles
    correctly for plain text files) and write a manifest so the next
    session can rebuild from the dump.

    Called at session end and periodically during ingestion.

    Args:
        lib: TheLibrarian instance (uses its _db_fallback/_db_original_path attrs)
        force: If True, sync even if no fallback is active
    """
    # Accept either an instance or try to infer from global DB_PATH
    original_path = getattr(lib, '_db_original_path', None) if lib else None
    is_fallback = getattr(lib, '_db_fallback', False) if lib else False

    if not is_fallback and not force:
        return False

    if not original_path:
        original_path = DB_PATH

    # Current working DB is the session-local one
    session_db = lib.rolodex.db_path if lib else None
    if not session_db or not os.path.exists(session_db):
        return False

    # Don't sync if source and dest are the same
    if os.path.abspath(session_db) == os.path.abspath(original_path):
        return False

    try:
        import sqlite3 as _sqlite3
        import hashlib

        # Flush any pending writes via SQLite checkpoint
        if lib and hasattr(lib, 'rolodex') and hasattr(lib.rolodex, 'conn'):
            try:
                lib.rolodex.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass

        mount_dir = os.path.dirname(original_path)
        dump_path = os.path.join(mount_dir, "rolodex_dump.sql")

        # Dump DB as SQL text, stripping FTS shadow tables.
        # SQLite's iterdump() emits CREATE TABLE + INSERT for FTS shadow
        # tables (_config, _content, _data, _docsize, _idx), but creating
        # the FTS virtual table auto-creates them — causing "already exists"
        # errors on reimport. We detect FTS virtual tables, build the shadow
        # name set, and skip any statements that reference them.
        db = _sqlite3.connect(session_db)

        # Find FTS virtual table names
        fts_tables = set()
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE '%fts5%'"
        ).fetchall():
            fts_tables.add(row[0])
        # Build shadow table name set
        fts_shadow_names = set()
        for base in fts_tables:
            for suffix in ('_config', '_content', '_data', '_docsize', '_idx'):
                fts_shadow_names.add(base + suffix)

        with open(dump_path, 'w') as f:
            for line in db.iterdump():
                # Skip statements referencing FTS shadow tables
                skip = False
                for shadow in fts_shadow_names:
                    # iterdump quotes table names: "tablename" or 'tablename'
                    if f'"{shadow}"' in line or f"'{shadow}'" in line:
                        skip = True
                        break
                if not skip:
                    f.write(line + '\n')
        db.close()

        # Verify the dump landed
        dump_size = os.path.getsize(dump_path)
        dump_hash = hashlib.md5(open(dump_path, 'rb').read()).hexdigest()

        # Write manifest so the next session knows a dump is available
        manifest = {
            "dump_file": "rolodex_dump.sql",
            "dump_size": dump_size,
            "dump_md5": dump_hash,
            "source_session": getattr(lib, '_db_original_path', session_db),
            "entry_count": lib.rolodex.conn.execute(
                "SELECT count(*) FROM rolodex_entries"
            ).fetchone()[0] if lib and hasattr(lib, 'rolodex') else -1,
            "synced_at": __import__('datetime').datetime.utcnow().isoformat(),
        }
        manifest_path = os.path.join(mount_dir, "rolodex_sync_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        print(json.dumps({"housekeeping": "db_sync_back",
                          "method": "sql_dump",
                          "detail": f"Dumped {session_db} → {dump_path}",
                          "dump_size": dump_size,
                          "dump_md5": dump_hash}),
              file=sys.stderr)
        return True
    except Exception as e:
        print(json.dumps({"housekeeping": "db_sync_back_failed",
                          "detail": str(e)}),
              file=sys.stderr)
        return False


# Track ingestion count for periodic sync-back
_ingest_since_last_sync = 0
SYNC_BACK_INTERVAL = 10  # Sync back to mount every N ingestions


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

    # Serialize profile for structured access (needed early for compression check)
    user_profile_json = {k: v["value"] for k, v in profile.items()} if profile else {}

    # Prompt compression: load behavioral entries if enabled
    compression_enabled = user_profile_json.get('prompt_compression', '').lower() == 'on'
    behavioral_entries = []
    behavioral_block = ""
    if compression_enabled:
        behavioral_entries = lib.rolodex.get_behavioral_entries()
        behavioral_block = cb.build_behavioral_block(behavioral_entries) if behavioral_entries else ""

    fixed_token_cost = estimate_tokens(profile_block + behavioral_block + uk_block)

    # Phase 9: Context window state
    window_state = lib.context_window.get_state(lib.state.messages)
    bridge = lib.context_window.bridge_summary

    # ─── Phase 13: Suggest focus areas for session start ─────────────────
    suggested_focus = None
    try:
        from src.indexing.project_clusterer import ProjectClusterer
        pc = ProjectClusterer(lib.rolodex.conn)
        focus_suggestions = pc.suggest_focus(limit=3)
        if focus_suggestions:
            from datetime import datetime as _dt
            _now = _dt.utcnow()
            for s in focus_suggestions:
                if s.get("last_active"):
                    try:
                        la = _dt.fromisoformat(s["last_active"])
                        delta = _now - la
                        if delta.days == 0:
                            hours = delta.seconds // 3600
                            s["last_active_relative"] = "just now" if hours == 0 else f"{hours}h ago"
                        elif delta.days == 1:
                            s["last_active_relative"] = "yesterday"
                        else:
                            s["last_active_relative"] = f"{delta.days}d ago"
                    except (ValueError, TypeError):
                        s["last_active_relative"] = "unknown"
            suggested_focus = focus_suggestions
    except Exception:
        pass  # Graceful degradation if clusters aren't built yet

    # ─── Detect embedding strategy and build warnings ──────────────────
    embedding_strategy = getattr(lib.embeddings, 'strategy', 'unknown')
    warnings = []
    if embedding_strategy == "hash":
        warnings.append(
            "Semantic embeddings unavailable — using hash-based fallback. "
            "Search will work but relies on keyword overlap rather than meaning. "
            "This is normal on first boot if ML dependencies are still installing."
        )
    if mode == "verbatim":
        warnings.append(
            "No API key detected — running in verbatim mode. "
            "Ingestion uses heuristic extraction instead of LLM-enhanced summarization."
        )

    # ─── Compact boot: fast path ─────────────────────────────────────────
    if compact:
        # Return only the lightweight essentials — no manifest, no instructions
        preamble_parts = [p for p in [profile_block, behavioral_block, uk_block] if p]
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
            "behavioral_entries": len(behavioral_entries),
            "prompt_compression_enabled": compression_enabled,
            "past_sessions": len(past_sessions),
            "user_profile": user_profile_json,
            "embedding_strategy": embedding_strategy,
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

        if warnings:
            output["warnings"] = warnings

        if suggested_focus:
            output["suggested_focus"] = suggested_focus

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
    # Build final context: profile first, then behavioral (if enabled), then user knowledge, then manifest
    preamble_parts = [p for p in [profile_block, behavioral_block, uk_block] if p]
    if preamble_parts:
        context_block = "\n\n".join(preamble_parts) + "\n\n" + manifest_context if manifest_context else "\n\n".join(preamble_parts)
    else:
        context_block = manifest_context

    # ─── Load behavioral instructions (INSTRUCTIONS.md) ────────────────
    # Skip raw file if prompt compression is enabled AND behavioral entries exist.
    # Fall back to raw file if compression is on but nothing compiled yet.
    instructions_block = None
    if not (compression_enabled and behavioral_entries):
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
        "behavioral_entries": len(behavioral_entries),
        "prompt_compression_enabled": compression_enabled,
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


async def cmd_ingest(role, content, corrects_id=None, as_user_knowledge=False, is_summary=False,
                     doc_id=None, source_location=None):
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

    # Phase 12: Handle --doc and --loc flags (document source citation)
    if doc_id and entries:
        for entry in entries:
            lib.rolodex.update_entry_document_source(
                entry_id=entry.id,
                document_id=doc_id,
                source_location=source_location or "",
            )

    # Phase 9: Include checkpoint and window state
    window = lib.context_window.get_stats()

    # Housekeeping: run fuse cleanup every N ingestions
    checkpoint = window["last_checkpoint_turn"]
    if checkpoint > 0 and checkpoint % FUSE_CLEANUP_INTERVAL == 0:
        _cleanup_fuse_hidden()

    # Periodic sync-back: keep mounted DB in sync during long sessions
    global _ingest_since_last_sync
    _ingest_since_last_sync += 1
    if _ingest_since_last_sync >= SYNC_BACK_INTERVAL:
        _sync_db_back(lib)
        _ingest_since_last_sync = 0

    result = {
        "ingested": len(entries),
        "session_id": lib.session_id,
        "checkpoint": checkpoint,
        "total_checkpoints": window["checkpoints"],
    }
    if as_user_knowledge:
        result["user_knowledge"] = True
    if doc_id:
        result["document_id"] = doc_id
        if source_location:
            result["source_location"] = source_location

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


async def cmd_recall(query, source_type=None, fresh=False, fresh_hours=48.0):
    """Search memory, return formatted context block.

    Phase 11: Wide-net-then-narrow search pattern.
    1. Query expansion generates multiple search variants + extracts entities
    2. Wide net: pull 15 results per variant (up to ~100 candidates)
    3. Re-ranker narrows using 5 signals: semantic, entity match, category,
       recency, access frequency
    4. Return top 5 re-ranked results

    Phase 12: Optional --source filter ('conversation', 'document', 'user_knowledge').
    Phase 13: Optional --fresh flag — prioritize recent entries, filtering out
    anything older than fresh_hours (default 48h). Useful when verifying
    whether a previously-known status is still current.
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
                # Phase 12: Filter by source_type if specified
                if source_type and getattr(entry, 'source_type', 'conversation') != source_type:
                    continue
                if entry.id not in seen_ids:
                    seen_ids.add(entry.id)
                    # Use search position as a proxy score (first = highest)
                    score = 1.0 - (len(all_candidates) * 0.01)
                    all_candidates.append((entry, max(score, 0.1)))

    if all_candidates:
        # Phase 13: --fresh mode — filter to recent entries and boost recency
        if fresh:
            from datetime import datetime as _dt
            now = _dt.utcnow()
            cutoff_seconds = fresh_hours * 3600
            fresh_candidates = [
                (entry, score) for entry, score in all_candidates
                if hasattr(entry, 'created_at') and entry.created_at is not None
                and (now - entry.created_at).total_seconds() < cutoff_seconds
            ]
            filtered_count = len(all_candidates) - len(fresh_candidates)
            if fresh_candidates:
                all_candidates = fresh_candidates
            # If all candidates are stale, keep them but warn
            elif filtered_count > 0:
                print(f"[fresh: all {len(all_candidates)} candidates older than {fresh_hours}h — showing anyway with staleness flags]", flush=True)

        # Phase 11: Re-rank the wide pool using multiple signals
        rerank_limit = 10 if fresh else 5  # Wider pool in fresh mode for recency re-sort
        scored = reranker.rerank(
            candidates=all_candidates,
            query=query,
            query_entities=expanded.entities,
            category_bias=expanded.category_bias,
            limit=rerank_limit,
        )

        # Phase 13: In fresh mode, re-sort by recency (newest first) after reranking
        if fresh:
            scored.sort(
                key=lambda sc: sc.entry.created_at.timestamp()
                if hasattr(sc.entry.created_at, 'timestamp') else 0,
                reverse=True,
            )
            scored = scored[:5]

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
        if fresh:
            meta_parts.append(f"fresh: <{fresh_hours}h")
        if expanded.intent != "exploratory":
            meta_parts.append(f"intent: {expanded.intent}")
        meta_parts.append(f"{len(expanded.variants)} variants")
        meta_parts.append(f"{len(all_candidates)} candidates")
        if entity_count > 0:
            meta_parts.append(f"{entity_count} entities")
        print(f"[{' | '.join(meta_parts)}]", flush=True)

        print(lib.get_context_block(synthetic_response), flush=True)
    else:
        print("No relevant memories found.", flush=True)

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

    # Phase 13: Update project clusters with this session's topic data
    try:
        from src.indexing.project_clusterer import ProjectClusterer
        pc = ProjectClusterer(lib.rolodex.conn)
        pc.update_clusters_for_session(session_id)
    except Exception:
        pass  # Non-fatal — clusters will rebuild on next boot if needed

    lib.end_session(summary=summary)
    clear_session_file()

    # Sync DB back to mounted folder before shutdown
    synced = _sync_db_back(lib)

    # Housekeeping: clean up FUSE artifacts on session end
    _cleanup_fuse_hidden()

    await lib.shutdown()
    print(json.dumps({
        "ended": session_id,
        "summary": summary,
        "manifest_refined": manifest_refined,
        "db_synced_back": synced,
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

    # High-value write — sync back immediately
    _sync_db_back(lib)

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

    # High-value write — sync back immediately
    _sync_db_back(lib)

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


async def cmd_compile(content=None, file_path=None):
    """Store compressed behavioral instructions in the rolodex.

    Usage:
      compile --content "yaml..."   # Store pre-compressed content directly
      compile <file_path>           # Read compressed content from file

    The LLM does the compression (no API key needed). This command stores
    the result as behavioral entries, superseding any previous compilation.
    """
    from datetime import datetime as _dt
    lib, _ = _make_librarian()
    _ensure_session(lib, caller="compile")
    session_id = lib.session_id

    # Input handling
    if content:
        compressed_text = content
        source_file = "direct_input"
    elif file_path:
        if not os.path.isfile(file_path):
            close_db(lib)
            print(json.dumps({"error": f"File not found: {file_path}"}))
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                compressed_text = f.read()
            source_file = os.path.basename(file_path)
        except OSError as e:
            close_db(lib)
            print(json.dumps({"error": f"Failed to read file: {e}"}))
            return
    else:
        close_db(lib)
        print(json.dumps({"error": "Usage: compile --content 'yaml' or compile <file_path>"}))
        return

    from src.core.types import estimate_tokens
    compressed_tokens = estimate_tokens(compressed_text)

    # Ingest the compressed content
    entries = await lib.ingest("assistant", compressed_text)

    # Recategorize as behavioral
    for entry in entries:
        lib.rolodex.update_entry_enrichment(
            entry_id=entry.id,
            category=_get_entry_category("behavioral"),
        )
        # Mark as non-verbatim (it's compressed, not original prose)
        lib.rolodex.conn.execute(
            "UPDATE rolodex_entries SET verbatim_source = 0 WHERE id = ?",
            (entry.id,)
        )
        # Store compilation metadata
        metadata = json.dumps({
            "source_file": source_file,
            "compiled_at": _dt.utcnow().isoformat(),
            "compressed_token_count": compressed_tokens,
            "content_type": "behavioral_instructions",
        })
        lib.rolodex.conn.execute(
            "UPDATE rolodex_entries SET metadata = ? WHERE id = ?",
            (metadata, entry.id)
        )
    lib.rolodex.conn.commit()

    # Supersede old behavioral entries
    new_ids = {e.id for e in entries}
    old_rows = lib.rolodex.conn.execute(
        """SELECT id FROM rolodex_entries
           WHERE category = 'behavioral'
           AND superseded_by IS NULL"""
    ).fetchall()
    superseded = []
    for (old_id,) in old_rows:
        if old_id not in new_ids:
            lib.rolodex.supersede_entry(old_id, entries[0].id if entries else old_id)
            superseded.append(old_id)

    # High-value write — sync back immediately
    _sync_db_back(lib)

    close_db(lib)
    print(json.dumps({
        "compiled": len(entries),
        "entry_ids": [e.id for e in entries],
        "source_file": source_file,
        "compressed_tokens": compressed_tokens,
        "superseded_entries": len(superseded),
        "session_id": session_id,
        "status": "ok",
    }))


async def cmd_settings(subcmd=None, args=None):
    """Manage system settings and toggleable features.

    Usage:
      settings show                     # List all settings with descriptions
      settings set <key> <value>        # Enable/disable a feature
    """
    lib, _ = _make_librarian()
    _ensure_session(lib, caller="settings")
    session_id = lib.session_id

    AVAILABLE_SETTINGS = {
        "prompt_compression": {
            "description": "Compress instruction files to YAML-like format, saving 50-70% boot tokens",
            "options": ["on", "off"],
            "default": "off",
        },
    }

    if not subcmd or subcmd == "show":
        profile = lib.rolodex.profile_get_all()
        settings_out = {}
        for key, meta in AVAILABLE_SETTINGS.items():
            current = None
            if profile and key in profile:
                current = profile[key].get("value")
            settings_out[key] = {
                "description": meta["description"],
                "options": meta["options"],
                "current": current or meta["default"],
                "default": meta["default"],
            }
        close_db(lib)
        print(json.dumps({"settings": settings_out}, indent=2))

    elif subcmd == "set":
        if not args or len(args) < 2:
            close_db(lib)
            print(json.dumps({"error": "Usage: settings set <key> <value>"}))
            return

        key, value = args[0], args[1]
        if key not in AVAILABLE_SETTINGS:
            close_db(lib)
            print(json.dumps({"error": f"Unknown setting: {key}. Available: {list(AVAILABLE_SETTINGS.keys())}"}))
            return

        meta = AVAILABLE_SETTINGS[key]
        if value not in meta["options"]:
            close_db(lib)
            print(json.dumps({"error": f"Invalid value '{value}' for {key}. Options: {meta['options']}"}))
            return

        lib.rolodex.profile_set(key, value, session_id=session_id)
        close_db(lib)
        print(json.dumps({
            "setting_changed": key,
            "new_value": value,
            "takes_effect": "next boot",
        }))

    else:
        close_db(lib)
        print(json.dumps({"error": f"Unknown subcommand: {subcmd}. Use show|set"}))


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


# ─── Browse Formatting Helpers ────────────────────────────────────────

def _format_browse_entry(entry, compact=True):
    """Format a single RolodexEntry for browse output."""
    eid = entry.id[:8]
    cat = entry.category.value.upper()
    src = getattr(entry, 'source_type', 'conversation') or 'conversation'
    tier = entry.tier.value
    access = entry.access_count
    created = entry.created_at.strftime("%Y-%m-%d %H:%M") if entry.created_at else "?"
    tags = ", ".join(entry.tags) if entry.tags else ""

    header = f"[{eid}] [{cat}] [{src}] created: {created}  accessed: {access}x  tier: {tier}"
    lines = [header]

    if tags:
        lines.append(f"  Tags: {tags}")

    doc_id = getattr(entry, 'document_id', None)
    loc = getattr(entry, 'source_location', '')
    if doc_id:
        source_line = f"  Source: doc:{doc_id[:8]}"
        if loc:
            source_line += f" {loc}"
        lines.append(source_line)

    lines.append("  ───")
    if compact:
        preview = entry.content[:200].replace("\n", " ")
        if len(entry.content) > 200:
            preview += "..."
        lines.append(f"  {preview}")
    else:
        for line in entry.content.split("\n"):
            lines.append(f"  {line}")
    lines.append("  ═══")
    return "\n".join(lines)


def _format_browse_list(entries, title="", compact=True):
    """Format a list of entries for browse output."""
    parts = []
    if title:
        parts.append(f"{'─' * 60}")
        parts.append(f"  {title} ({len(entries)} entries)")
        parts.append(f"{'─' * 60}")
    for entry in entries:
        parts.append(_format_browse_entry(entry, compact=compact))
    return "\n".join(parts)


# ─── Browse Command ───────────────────────────────────────────────────

def _entry_to_dict(entry, compact=True):
    """Serialize a RolodexEntry to a JSON-friendly dict."""
    d = {
        "id": entry.id[:8],
        "full_id": entry.id,
        "category": entry.category.value.upper(),
        "source_type": getattr(entry, 'source_type', 'conversation') or 'conversation',
        "tier": entry.tier.value,
        "access_count": entry.access_count,
        "created_at": entry.created_at.strftime("%Y-%m-%d %H:%M") if entry.created_at else None,
        "tags": entry.tags or [],
    }
    doc_id = getattr(entry, 'document_id', None)
    if doc_id:
        d["document_id"] = doc_id[:8]
        d["source_location"] = getattr(entry, 'source_location', '') or ''

    if compact:
        preview = entry.content[:200].replace("\n", " ")
        if len(entry.content) > 200:
            preview += "..."
        d["content"] = preview
    else:
        d["content"] = entry.content

    return d


async def cmd_browse(subcmd, args, as_json=False):
    """Browse the rolodex — view entries, filter by category/source/topic.

    --json flag outputs structured JSON for programmatic / in-chat display.
    """
    lib, _ = _make_librarian()
    _ensure_session(lib, caller="browse")

    if subcmd == "recent":
        limit = int(args[0]) if args else 20
        entries = lib.rolodex.browse_recent(limit)
        if as_json:
            print(json.dumps({"title": "Most Recent Entries", "count": len(entries),
                              "entries": [_entry_to_dict(e) for e in entries]}, indent=2))
        else:
            print(_format_browse_list(entries, title="Most Recent Entries", compact=True))

    elif subcmd == "category":
        if not args:
            print(json.dumps({"error": "Usage: browse category <category> [limit]"}))
            close_db(lib)
            sys.exit(1)
        cat = args[0]
        limit = int(args[1]) if len(args) > 1 else 20
        entries = lib.rolodex.get_entries_by_category(cat, limit=limit)
        if as_json:
            print(json.dumps({"title": f"Category: {cat}", "count": len(entries),
                              "entries": [_entry_to_dict(e) for e in entries]}, indent=2))
        else:
            print(_format_browse_list(entries, title=f"Category: {cat}", compact=True))

    elif subcmd == "source":
        if not args:
            print(json.dumps({"error": "Usage: browse source <conversation|document|user_knowledge> [limit]"}))
            close_db(lib)
            sys.exit(1)
        source = args[0]
        limit = int(args[1]) if len(args) > 1 else 20
        entries = lib.rolodex.browse_by_source_type(source, limit=limit)
        if as_json:
            print(json.dumps({"title": f"Source: {source}", "count": len(entries),
                              "entries": [_entry_to_dict(e) for e in entries]}, indent=2))
        else:
            print(_format_browse_list(entries, title=f"Source: {source}", compact=True))

    elif subcmd == "topic":
        if not args:
            print(json.dumps({"error": "Usage: browse topic <topic_id> [limit]"}))
            close_db(lib)
            sys.exit(1)
        topic_id = args[0]
        limit = int(args[1]) if len(args) > 1 else 20
        entries = lib.rolodex.get_entries_by_topic(topic_id, limit=limit)
        topic = lib.rolodex.get_topic(topic_id)
        label = topic["label"] if topic else topic_id
        if as_json:
            print(json.dumps({"title": f"Topic: {label}", "count": len(entries),
                              "entries": [_entry_to_dict(e) for e in entries]}, indent=2))
        else:
            print(_format_browse_list(entries, title=f"Topic: {label}", compact=True))

    elif subcmd == "entry":
        if not args:
            print(json.dumps({"error": "Usage: browse entry <entry_id or prefix>"}))
            close_db(lib)
            sys.exit(1)
        entry_id = args[0]
        entry = lib.rolodex.get_entry(entry_id)
        if not entry:
            entry = lib.rolodex.browse_entry_by_prefix(entry_id)
        if not entry:
            print(json.dumps({"error": f"Entry not found: {entry_id}"}))
            close_db(lib)
            sys.exit(1)
        if as_json:
            d = _entry_to_dict(entry, compact=False)
            d["full_id"] = entry.id
            d["conversation_id"] = entry.conversation_id
            d["content_type"] = entry.content_type.value
            d["verbatim_source"] = entry.verbatim_source
            d["document_id_full"] = getattr(entry, 'document_id', None)
            d["source_location"] = getattr(entry, 'source_location', '') or ''
            d["linked_ids"] = entry.linked_ids or None
            d["metadata"] = entry.metadata or None
            print(json.dumps({"title": f"Entry: {entry.id[:8]}", "entry": d}, indent=2))
        else:
            output = _format_browse_entry(entry, compact=False)
            meta_lines = [
                output,
                f"  ── Full Metadata ──",
                f"  ID:              {entry.id}",
                f"  Conversation:    {entry.conversation_id}",
                f"  Content Type:    {entry.content_type.value}",
                f"  Verbatim:        {entry.verbatim_source}",
                f"  Source Type:     {getattr(entry, 'source_type', 'conversation')}",
                f"  Document ID:     {getattr(entry, 'document_id', None) or '—'}",
                f"  Source Location: {getattr(entry, 'source_location', '') or '—'}",
                f"  Linked IDs:      {entry.linked_ids or '—'}",
                f"  Metadata:        {json.dumps(entry.metadata) if entry.metadata else '—'}",
            ]
            print("\n".join(meta_lines))

    elif subcmd == "knowledge":
        entries = lib.rolodex.get_user_knowledge_entries()
        if as_json:
            print(json.dumps({"title": "User Knowledge (always loaded)", "count": len(entries),
                              "entries": [_entry_to_dict(e, compact=False) for e in entries]}, indent=2))
        else:
            print(_format_browse_list(entries, title="User Knowledge (always loaded)", compact=False))

    elif subcmd == "sessions":
        limit = int(args[0]) if args else 20
        sessions = lib.rolodex.get_session_summaries(limit)
        if as_json:
            session_list = []
            for s in sessions:
                session_list.append({
                    "id": s["session_id"][:8],
                    "full_id": s["session_id"],
                    "created_at": s["created_at"][:16] if s["created_at"] else None,
                    "status": s["status"],
                    "entry_count": s["entry_count"],
                    "summary": s["summary"][:120] if s["summary"] else None,
                })
            print(json.dumps({"title": "Sessions", "count": len(sessions),
                              "sessions": session_list}, indent=2))
        else:
            parts = [
                f"{'─' * 60}",
                f"  Sessions ({len(sessions)} shown)",
                f"{'─' * 60}",
            ]
            for s in sessions:
                sid = s["session_id"][:8]
                created = s["created_at"][:16] if s["created_at"] else "?"
                status = s["status"]
                count = s["entry_count"]
                summary = s["summary"][:80] if s["summary"] else "—"
                parts.append(f"  [{sid}] {created}  entries: {count}  status: {status}")
                parts.append(f"    {summary}")
            print("\n".join(parts))

    elif subcmd == "search":
        if not args:
            print(json.dumps({"error": "Usage: browse search <query> [limit]"}))
            close_db(lib)
            sys.exit(1)
        query = args[0]
        limit = int(args[1]) if len(args) > 1 else 10
        results = lib.rolodex.keyword_search(query, limit=limit)
        entries = [entry for entry, score in results]
        if as_json:
            print(json.dumps({"title": f'Search: "{query}"', "count": len(entries),
                              "entries": [_entry_to_dict(e) for e in entries]}, indent=2))
        else:
            print(_format_browse_list(entries, title=f'Search: "{query}"', compact=True))

    else:
        print(json.dumps({
            "error": f"Unknown browse subcommand: {subcmd}",
            "usage": "browse recent|category|source|topic|entry|knowledge|sessions|search [--json]"
        }))
        sys.exit(1)

    close_db(lib)


async def cmd_register_doc(file_path, title=None):
    """Register a document in the document registry (Phase 12).

    Extracts metadata (title, page count, hash) without reading full content.
    The document is read on-demand via read-doc when needed.
    """
    from src.indexing.doc_readers import get_document_metadata, detect_file_type

    file_type = detect_file_type(file_path)
    if file_type == "unknown":
        print(json.dumps({"error": f"Unsupported file type: {os.path.splitext(file_path)[1]}"}))
        sys.exit(1)

    if not os.path.isfile(file_path):
        print(json.dumps({"error": f"File not found: {file_path}"}))
        sys.exit(1)

    meta = get_document_metadata(file_path)
    doc_id = str(__import__("uuid").uuid4())

    lib, _ = _make_librarian()
    _ensure_session(lib, caller="register-doc")

    lib.rolodex.register_document(
        doc_id=doc_id,
        file_name=meta.file_name,
        file_path=os.path.abspath(file_path),
        file_type=meta.file_type,
        file_hash=meta.file_hash,
        title=title or meta.title or meta.file_name,
        page_count=meta.page_count,
        summary="",
        metadata=meta.metadata,
    )

    close_db(lib)
    print(json.dumps({
        "registered": True,
        "doc_id": doc_id,
        "file_name": meta.file_name,
        "file_type": meta.file_type,
        "title": title or meta.title or meta.file_name,
        "page_count": meta.page_count,
        "file_hash": meta.file_hash[:16] + "..." if meta.file_hash else None,
        "file_size": meta.file_size,
    }, indent=2))


async def cmd_read_doc(doc_id, pages=None):
    """Read a registered document on-demand (Phase 12).

    Looks up the document in the registry, reads via the appropriate
    format handler, and returns extracted text.
    """
    from src.indexing.doc_readers import read_document

    lib, _ = _make_librarian()
    _ensure_session(lib, caller="read-doc")

    doc = lib.rolodex.get_document(doc_id)
    if not doc:
        close_db(lib)
        print(json.dumps({"error": f"Document not found: {doc_id}"}))
        sys.exit(1)

    result = read_document(doc["file_path"], pages=pages)

    if result.success:
        # Update last read timestamp
        lib.rolodex.update_document_read_time(doc_id)
        close_db(lib)
        print(json.dumps({
            "doc_id": doc_id,
            "file_name": doc["file_name"],
            "text": result.text,
            "metadata": result.metadata,
            "headings": result.headings,
        }))
    else:
        close_db(lib)
        print(json.dumps({
            "error": result.error,
            "doc_id": doc_id,
            "file_path": doc["file_path"],
        }))
        sys.exit(1)


async def cmd_docs(subcmd, args):
    """Document registry management (Phase 12)."""
    lib, _ = _make_librarian()
    _ensure_session(lib, caller="docs")

    if subcmd == "list":
        docs = lib.rolodex.list_documents()
        if not docs:
            print(json.dumps({"documents": [], "message": "No documents registered. Use 'register-doc <path>' to add one."}))
        else:
            print(json.dumps({"documents": docs}, indent=2, default=str))

    elif subcmd == "show":
        if not args:
            print(json.dumps({"error": "Usage: docs show <doc_id>"}))
            close_db(lib)
            sys.exit(1)
        doc = lib.rolodex.get_document(args[0])
        if not doc:
            print(json.dumps({"error": f"Document not found: {args[0]}"}))
        else:
            # Also fetch linked entries
            entries = lib.rolodex.get_entries_for_document(args[0])
            doc["linked_entries"] = len(entries)
            doc["entry_previews"] = [
                {
                    "id": e.id[:8],
                    "source_location": e.source_location,
                    "content_preview": e.content[:120],
                    "created_at": e.created_at.isoformat(),
                }
                for e in entries[:10]
            ]
            print(json.dumps(doc, indent=2, default=str))

    elif subcmd == "refresh":
        if not args:
            print(json.dumps({"error": "Usage: docs refresh <doc_id>"}))
            close_db(lib)
            sys.exit(1)
        doc = lib.rolodex.get_document(args[0])
        if not doc:
            print(json.dumps({"error": f"Document not found: {args[0]}"}))
        else:
            from src.indexing.doc_readers import compute_file_hash
            if not os.path.isfile(doc["file_path"]):
                print(json.dumps({"error": f"File not found on disk: {doc['file_path']}"}))
            else:
                new_hash = compute_file_hash(doc["file_path"])
                changed = new_hash != doc.get("file_hash")
                if changed:
                    lib.rolodex.update_document_hash(args[0], new_hash)
                print(json.dumps({
                    "doc_id": args[0],
                    "changed": changed,
                    "old_hash": (doc.get("file_hash") or "")[:16] + "...",
                    "new_hash": new_hash[:16] + "...",
                }))

    elif subcmd == "remove":
        if not args:
            print(json.dumps({"error": "Usage: docs remove <doc_id>"}))
            close_db(lib)
            sys.exit(1)
        existed = lib.rolodex.remove_document(args[0])
        print(json.dumps({
            "removed": existed,
            "doc_id": args[0],
            "note": "Linked entries remain but document_id cleared." if existed else "Document not found.",
        }))

    else:
        print(json.dumps({"error": f"Unknown docs subcommand: {subcmd}. Use list|show|refresh|remove"}))
        sys.exit(1)

    close_db(lib)


async def cmd_suggest_focus(limit=3):
    """Suggest work streams for session-focus selection (Phase 13).

    Queries project clusters (or raw topics as fallback) and returns
    the top N most recent work streams. Designed to feed into an
    AskUserQuestion-style prompt at session start.

    Output JSON includes:
        - suggestions: list of {project_label, topic_label, topic_id, topic_ids, last_active, entry_count}
        - count: number of suggestions
    """
    lib, _ = _make_librarian()
    _ensure_session(lib, caller="suggest-focus")

    from src.indexing.project_clusterer import ProjectClusterer
    pc = ProjectClusterer(lib.rolodex.conn)

    suggestions = pc.suggest_focus(limit=limit)

    # Format last_active as relative time for display
    from datetime import datetime as dt
    now = dt.utcnow()
    for s in suggestions:
        if s.get("last_active"):
            try:
                la = dt.fromisoformat(s["last_active"])
                delta = now - la
                if delta.days == 0:
                    hours = delta.seconds // 3600
                    if hours == 0:
                        s["last_active_relative"] = "just now"
                    elif hours == 1:
                        s["last_active_relative"] = "1 hour ago"
                    else:
                        s["last_active_relative"] = f"{hours} hours ago"
                elif delta.days == 1:
                    s["last_active_relative"] = "yesterday"
                else:
                    s["last_active_relative"] = f"{delta.days} days ago"
            except (ValueError, TypeError):
                s["last_active_relative"] = "unknown"
        else:
            s["last_active_relative"] = "unknown"

    close_db(lib)
    print(json.dumps({
        "suggestions": suggestions,
        "count": len(suggestions),
    }, indent=2))


async def cmd_focus_boot(topic_ids_json):
    """Rebuild manifest with focus bias toward selected topic cluster (Phase 13).

    Called after the user selects a focus from suggest-focus results.
    Rebuilds the manifest with a 3x weight multiplier for entries
    in the selected topic cluster.

    Args:
        topic_ids_json: JSON array of topic IDs to bias toward.
    """
    topic_ids = json.loads(topic_ids_json)
    if not topic_ids:
        print(json.dumps({"error": "No topic IDs provided"}))
        sys.exit(1)

    lib, mode = _make_librarian()
    _ensure_session(lib, caller="focus-boot")

    from src.retrieval.context_builder import ContextBuilder
    from src.core.types import estimate_tokens
    from src.storage.manifest_manager import ManifestManager

    cb = ContextBuilder()

    # Calculate available budget (same as boot)
    profile = lib.rolodex.profile_get_all()
    profile_block = cb.build_profile_block(profile) if profile else ""
    uk_entries = lib.rolodex.get_user_knowledge_entries()
    uk_block = cb.build_user_knowledge_block(uk_entries) if uk_entries else ""
    fixed_cost = estimate_tokens(profile_block + uk_block)
    available_budget = max(0, 20000 - fixed_cost)

    mm = ManifestManager(lib.rolodex.conn, lib.rolodex)
    manifest = mm.build_focused_manifest(available_budget, focus_topic_ids=topic_ids)

    # Build context block from focused manifest
    manifest_context = ""
    if manifest and manifest.entries:
        entry_ids = [me.entry_id for me in manifest.entries]
        entries = lib.rolodex.get_entries_by_ids(entry_ids)
        id_to_rank = {me.entry_id: me.slot_rank for me in manifest.entries}
        entries.sort(key=lambda e: id_to_rank.get(e.id, 999))
        chains = _get_gap_fill_chains(lib, manifest)
        manifest_context = cb.build_context_block(entries, lib.session_id, chains)

    close_db(lib)
    print(json.dumps({
        "status": "ok",
        "boot_type": "focused",
        "focus_topic_ids": topic_ids,
        "manifest_entries": len(manifest.entries) if manifest else 0,
        "total_token_cost": manifest.total_token_cost if manifest else 0,
        "context_block": manifest_context,
    }, indent=2))


async def cmd_projects(subcmd, args):
    """Project cluster management (Phase 13)."""
    lib, _ = _make_librarian()
    _ensure_session(lib, caller="projects")

    from src.indexing.project_clusterer import ProjectClusterer
    pc = ProjectClusterer(lib.rolodex.conn)

    if subcmd == "list":
        clusters = pc._get_existing_clusters()
        if not clusters:
            print(json.dumps({"projects": [], "message": "No project clusters yet. They emerge from topic co-occurrence across sessions."}))
        else:
            formatted = []
            for c in clusters:
                topic_ids = json.loads(c["topic_ids"]) if isinstance(c["topic_ids"], str) else c["topic_ids"]
                # Resolve topic labels
                topic_labels = []
                for tid in topic_ids:
                    row = lib.rolodex.conn.execute("SELECT label FROM topics WHERE id = ?", (tid,)).fetchone()
                    if row:
                        topic_labels.append(row["label"])
                formatted.append({
                    "id": c["id"][:8],
                    "full_id": c["id"],
                    "label": c["label"],
                    "is_user_named": bool(c["is_user_named"]),
                    "topics": topic_labels,
                    "topic_count": len(topic_ids),
                    "entry_count": c["entry_count"],
                    "session_count": c["session_count"],
                    "last_active": c["last_active"],
                })
            print(json.dumps({"projects": formatted}, indent=2))

    elif subcmd == "rebuild":
        clusters = pc.rebuild_clusters()
        print(json.dumps({
            "rebuilt": True,
            "cluster_count": len(clusters),
            "clusters": [
                {"label": c["label"], "topic_count": len(c["topic_ids"]),
                 "entry_count": c["entry_count"]}
                for c in clusters
            ],
        }, indent=2))

    elif subcmd == "name":
        if len(args) < 2:
            print(json.dumps({"error": "Usage: projects name <cluster_id_or_prefix> <label>"}))
            close_db(lib)
            sys.exit(1)
        cluster_id = args[0]
        label = " ".join(args[1:])
        # Try prefix match
        row = lib.rolodex.conn.execute(
            "SELECT id FROM project_clusters WHERE id LIKE ?",
            (cluster_id + "%",)
        ).fetchone()
        if row:
            pc.name_cluster(row["id"], label)
            print(json.dumps({"named": True, "cluster_id": row["id"][:8], "label": label}))
        else:
            print(json.dumps({"error": f"Cluster not found: {cluster_id}"}))

    else:
        print(json.dumps({"error": f"Unknown projects subcommand: {subcmd}. Use list|rebuild|name"}))
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


def cmd_init(target_dir):
    """Initialize a workspace folder for Cowork use.

    Copies the Librarian source into <target>/librarian/ and generates a
    CLAUDE.md with Cowork-compatible paths so the model can boot on first message.
    """
    import shutil
    target = os.path.abspath(target_dir)
    lib_dest = os.path.join(target, "librarian")

    # 0. Clean up stale DB artifacts from any previous failed attempts
    stale_files = ["rolodex.db", "rolodex.db-wal", "rolodex.db-journal",
                   "rolodex.db-shm", ".cowork_session"]
    cleaned = []
    for sf in stale_files:
        sf_path = os.path.join(target, sf)
        if os.path.exists(sf_path):
            try:
                os.remove(sf_path)
                cleaned.append(sf)
            except OSError:
                pass  # Best effort
    # Also check inside librarian/ subfolder
    for sf in stale_files:
        sf_path = os.path.join(target, "librarian", sf)
        if os.path.exists(sf_path):
            try:
                os.remove(sf_path)
                cleaned.append(f"librarian/{sf}")
            except OSError:
                pass

    # 1. Copy source (with retry for Windows file locks)
    if os.path.exists(lib_dest):
        import time as _time
        for attempt in range(3):
            try:
                shutil.rmtree(lib_dest)
                break
            except PermissionError:
                if attempt < 2:
                    _time.sleep(2)
                else:
                    print(json.dumps({
                        "error": f"Cannot remove {lib_dest} — files are locked. "
                                 "Close any Cowork sessions using this folder, then retry."
                    }))
                    sys.exit(1)

    os.makedirs(lib_dest, exist_ok=True)

    # Determine source location: frozen (PyInstaller) vs development layout
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        # Frozen build: source files are bundled in _cowork_source/
        cowork_src = os.path.join(meipass, "_cowork_source")
    elif getattr(sys, 'frozen', False):
        # Inno Setup installed layout: check next to the exe
        exe_dir = os.path.dirname(sys.executable)
        cowork_src = os.path.join(exe_dir, "_cowork_source")
        if not os.path.isdir(cowork_src):
            cowork_src = os.path.join(exe_dir, "lib", "_cowork_source")
    else:
        # Development layout: source is next to this script
        cowork_src = None

    if cowork_src and os.path.isdir(cowork_src):
        # Extract from frozen bundle
        for fname in ("librarian_cli.py", "main.py", "requirements.txt", "requirements-onnx.txt", "requirements-ml.txt"):
            src = os.path.join(cowork_src, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(lib_dest, fname))
        # Copy src/ tree from bundle
        bundle_src = os.path.join(cowork_src, "src")
        dst_src = os.path.join(lib_dest, "src")
        if os.path.isdir(bundle_src):
            shutil.copytree(bundle_src, dst_src, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".pytest_cache"
            ))
        # Copy ONNX model if bundled (non-fatal — hash fallback exists)
        try:
            bundle_model = os.path.join(cowork_src, "models", "all-MiniLM-L6-v2")
            if os.path.isdir(bundle_model):
                dst_model = os.path.join(lib_dest, "models", "all-MiniLM-L6-v2")
                os.makedirs(dst_model, exist_ok=True)
                for mf in ("model.onnx", "tokenizer.json"):
                    src_mf = os.path.join(bundle_model, mf)
                    if os.path.isfile(src_mf):
                        shutil.copy2(src_mf, os.path.join(dst_model, mf))
        except OSError:
            pass  # ONNX model is optional — hash embeddings work as fallback
    else:
        # Development layout: copy from SCRIPT_DIR
        for fname in ("librarian_cli.py", "main.py", "requirements.txt", "requirements-onnx.txt", "requirements-ml.txt"):
            src = os.path.join(SCRIPT_DIR, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(lib_dest, fname))
        src_dir = os.path.join(SCRIPT_DIR, "src")
        dst_src = os.path.join(lib_dest, "src")
        if os.path.isdir(src_dir):
            shutil.copytree(src_dir, dst_src, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".pytest_cache"
            ))
        # Copy ONNX model from dev layout (non-fatal — hash fallback exists)
        try:
            dev_model = os.path.join(SCRIPT_DIR, "lib", "models", "all-MiniLM-L6-v2")
            if os.path.isdir(dev_model):
                dst_model = os.path.join(lib_dest, "models", "all-MiniLM-L6-v2")
                os.makedirs(dst_model, exist_ok=True)
                for mf in ("model.onnx", "tokenizer.json"):
                    src_mf = os.path.join(dev_model, mf)
                    if os.path.isfile(src_mf):
                        shutil.copy2(src_mf, os.path.join(dst_model, mf))
        except OSError:
            pass  # ONNX model is optional — hash embeddings work as fallback

    # 2. Create a pre-seeded rolodex.db with schema + welcome context
    #    This avoids the 0-byte DB problem — the file ships as a valid SQLite DB.
    import sqlite3
    from datetime import datetime
    import uuid

    db_path = os.path.join(lib_dest, "rolodex.db")
    try:
        # Import schema from the just-copied source
        init_src = os.path.join(lib_dest, "src")
        _orig_path = sys.path[:]
        sys.path.insert(0, lib_dest)
        from src.storage.schema import SCHEMA_SQL, _safe_add_columns
        sys.path[:] = _orig_path

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        _safe_add_columns(conn)

        # Seed a welcome user_knowledge entry so first boot has context
        welcome_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        seed_session_id = str(uuid.uuid4())

        # Create the seed session
        conn.execute(
            "INSERT INTO conversations (id, created_at, status, summary, last_active, message_count) "
            "VALUES (?, ?, 'ended', 'Workspace initialized by The Librarian installer.', ?, 0)",
            (seed_session_id, now, now)
        )

        # Seed a user_knowledge entry describing what The Librarian is
        conn.execute(
            "INSERT INTO rolodex_entries "
            "(id, conversation_id, content, content_type, category, tags, source_range, "
            " access_count, created_at, tier, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                welcome_id, seed_session_id,
                "The Librarian is a persistent memory system for AI conversations. "
                "It stores every exchange in a local SQLite rolodex — user preferences, "
                "project decisions, code patterns, and past discussions survive between sessions. "
                "Use 'remember' to store facts about yourself, 'recall' to search past context, "
                "and 'profile set' to configure preferences like name and timezone.",
                "prose", "user_knowledge",
                json.dumps(["the librarian", "welcome", "getting started", "memory system"]),
                json.dumps({}),
                0, now, "hot",
                json.dumps({"seeded_by": "init", "version": __version__})
            )
        )

        # Add FTS entry
        conn.execute(
            "INSERT INTO rolodex_fts (entry_id, content, tags, category) VALUES (?, ?, ?, ?)",
            (
                welcome_id,
                "The Librarian is a persistent memory system for AI conversations. "
                "It stores every exchange in a local SQLite rolodex.",
                "the librarian welcome getting started memory system",
                "user_knowledge"
            )
        )

        conn.commit()
        conn.close()
        db_seeded = True
    except Exception as e:
        # Non-fatal — the DB will be created at first boot instead
        db_seeded = False
        db_seed_error = str(e)

    # 3. Derive the folder name for the wildcard path
    folder_name = os.path.basename(target)

    # 4. Generate CLAUDE.md
    claude_md = f"""# STOP. BOOT BEFORE RESPONDING.

Before generating ANY reply — even to "Hi" — run this command:

```bash
python /sessions/*/mnt/{folder_name}/librarian/librarian_cli.py boot --compact
```

The wildcard `*` resolves to whatever Cowork session slug is active. No hardcoded paths.

After boot, immediately launch a background agent to load the full context:

```bash
python /sessions/*/mnt/{folder_name}/librarian/librarian_cli.py boot --full-context
```

No exceptions. No "I'll do it next message." Boot first, then respond.

If prior messages in this session were not yet ingested, ingest them now before continuing.

## First Boot Advisory

If the boot JSON shows `total_entries` <= 1 (i.e. only the seed entry or empty), this is likely the user's first session with The Librarian. **Proactively** tell the user:

"Welcome! This is your first session with The Librarian. The first boot is a bit slower while I set up the tools I need to remember our conversations — installing dependencies and initializing the memory system. Future sessions will be much faster. Just talk to me naturally and I'll start learning about you and your work."

This must appear BEFORE any other output. Don't wait for the user to notice latency.

## Post-Boot: Always Offer Task Options

After boot, if the user's opening message does NOT specify a clear task (e.g. "Hi", "let's get to work", "hey", or any casual opener), use the AskUserQuestion tool to present task options before doing anything else. Derive the options from the `suggested_focus` clusters in the boot JSON — these represent the user's most recent active projects. Always include a catch-all "Something new" option.

Do NOT just ask "what are we working on?" in plain text. Use the AskUserQuestion tool so the user gets clickable choices.

## What This Is

This workspace uses The Librarian, a persistent memory system by Dicta Technologies Inc. It's backed by a local SQLite rolodex that gives you continuity across conversations — user preferences, project decisions, code patterns, and past discussions survive between sessions.

## How to Interact

**Just have a conversation.** The Librarian works best when you talk naturally. There's nothing special you need to do — your preferences, decisions, and context are remembered automatically across sessions.

Behind the scenes, the system ingests every exchange, recalls relevant past context when needed, and learns your working style over time. You don't need to manage any of this manually.

## Always-On Rules

- **Every message:** Verify The Librarian is booted. If unsure, boot again.
- **Every message:** Ingest it. Both user messages and your responses, separately. See "What to Ingest" below.
- **After compaction or continuation:** Re-boot immediately. The Librarian does not carry over.
- **When past context would help:** Use `recall` before responding.
- **When the user states a fact about themselves:** Use `remember` automatically. No explicit command needed from the user.

## Commands (For Advanced Use)

These commands are primarily used by the AI assistant behind the scenes. Users rarely need to invoke them directly — conversation is the primary interface. They're documented here for transparency and for power users who want fine-grained control.

All commands use the wildcard CLI path: `python /sessions/*/mnt/{folder_name}/librarian/librarian_cli.py`

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

## What to Ingest

**Everything. 100% coverage. No cherry-picking.**

Ingest every user message and every assistant response, verbatim. Storage is trivial for a local hard drive. Lossy ingestion is worse than a large corpus — the search layer (user_knowledge boost, categories, ranking) handles surfacing the right things. Cherry-picking at ingestion time loses context that may matter later.

Skip only bare acknowledgments with zero informational content ("ok", "thanks", "got it").

## What to Recall

Anything where past context helps: references to previous sessions, projects, people, terms, or whenever you feel you *should* know something but don't.

## Entry Hierarchy

1. **User Profile** — key-value pairs (name, timezone, response_style). Loaded first at boot.
2. **User Knowledge** — rich facts about the user (preferences, corrections, biographical context). Always loaded at boot, 3x search boost, permanently hot. Created via `remember` or `ingest --user-knowledge`.
3. **Regular entries** — everything else. Searched on demand via `recall`.

## Browse Command — Always Display In-Chat

When running `browse` (or any subcommand), the output may land in a collapsed tool-output panel that the user cannot easily see. **Always echo browse results directly into the chat response.** Use the `--json` flag to get structured output, then format it as a readable code block or inline text in your reply.

```bash
python /sessions/*/mnt/{folder_name}/librarian/librarian_cli.py browse recent 5 --json
```

Then include the formatted results in your message so the user sees them without expanding any dropdown.

## Corrections vs. Reasoning Chains

When the user corrects a factual error (e.g. wrong name), use `correct` or `--corrects` to supersede the old entry. The old entry is soft-deleted (hidden from search, kept in DB).

When the user changes their mind on a design decision (e.g. renaming a tool), do NOT supersede. Both entries should remain — the reasoning chain ("we considered X, then pivoted to Y because Z") is valuable context.
"""

    claude_md_path = os.path.join(target, "CLAUDE.md")
    with open(claude_md_path, "w", encoding="utf-8") as f:
        f.write(claude_md.strip() + "\n")

    files_created = ["CLAUDE.md", "librarian/librarian_cli.py", "librarian/main.py",
                      "librarian/requirements.txt", "librarian/src/"]
    if db_seeded:
        files_created.append("librarian/rolodex.db")

    result = {
        "status": "ok",
        "initialized": target,
        "folder_name": folder_name,
        "files_created": files_created,
        "db_seeded": db_seeded,
        "stale_files_cleaned": cleaned if cleaned else [],
        "next_steps": [
            f"Open Cowork and select '{target}' as your folder",
            "Send any message — The Librarian will boot automatically",
        ]
    }
    if not db_seeded:
        result["db_seed_error"] = db_seed_error
    print(json.dumps(result, indent=2))


# ─── GUI Installer (tkinter) ──────────────────────────────────────────

def cmd_install_gui():
    """Launch a zero-friction GUI installer using tkinter.

    This runs when the user double-clicks the app/exe with no arguments.
    Works on Windows, macOS, and Linux.
    Shows a simple window: title → folder picker → Install → success message.
    No external tools (Inno Setup, NSIS, etc.) required.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from src.platform_utils import get_gui_font

    font_family, font_base = get_gui_font()

    default_workspace = os.path.join(os.path.expanduser("~"), "Documents", "My Librarian")

    root = tk.Tk()
    root.title("The Librarian — Setup")
    root.geometry("520x380")
    root.resizable(False, False)

    # Center on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() - 520) // 2
    y = (root.winfo_screenheight() - 380) // 2
    root.geometry(f"+{x}+{y}")

    # Try to set a neutral background
    bg = "#f5f5f5"
    root.configure(bg=bg)

    # ── Title ──
    tk.Label(root, text="The Librarian", font=(font_family, font_base + 10, "bold"),
             bg=bg).pack(pady=(25, 2))
    tk.Label(root, text="Persistent Memory for AI Conversations",
             font=(font_family, font_base), fg="#666", bg=bg).pack(pady=(0, 20))

    # ── Workspace selection ──
    tk.Label(root, text="Choose your workspace folder:", font=(font_family, font_base),
             bg=bg, anchor="w").pack(anchor="w", padx=35)

    picker_frame = tk.Frame(root, bg=bg)
    picker_frame.pack(fill="x", padx=35, pady=(5, 0))

    path_var = tk.StringVar(value=default_workspace)
    entry = tk.Entry(picker_frame, textvariable=path_var, font=(font_family, font_base - 1))
    entry.pack(side="left", fill="x", expand=True)

    def browse():
        initial = path_var.get().strip()
        if not os.path.isdir(initial):
            initial = os.path.expanduser("~")
        chosen = filedialog.askdirectory(initialdir=initial, title="Select Workspace Folder")
        if chosen:
            path_var.set(chosen)

    tk.Button(picker_frame, text="Browse...", command=browse,
              font=(font_family, font_base - 1)).pack(side="right", padx=(8, 0))

    tk.Label(root, text="This is the folder you'll select in Cowork or Claude Code.\n"
                        "The default works great for most people.",
             font=(font_family, font_base - 2), fg="#888", bg=bg).pack(pady=(4, 15))

    # ── Status label ──
    status_var = tk.StringVar(value="")
    status_label = tk.Label(root, textvariable=status_var, font=(font_family, font_base - 1),
                            fg="#0066cc", bg=bg)
    status_label.pack(pady=(0, 5))

    # ── Install action ──
    def do_install():
        target = path_var.get().strip()
        if not target:
            messagebox.showerror("Error", "Please choose a workspace folder.")
            return

        install_btn.config(state="disabled", text="Setting up...")
        status_var.set("Initializing your workspace...")
        root.update()

        try:
            # Ensure the target directory exists
            os.makedirs(target, exist_ok=True)
            cmd_init(target)

            # Optional: self-install to AppData + PATH (best-effort, non-blocking)
            _self_install_to_path()

            status_var.set("")
            messagebox.showinfo(
                "You're All Set!",
                f"Your workspace is ready at:\n{target}\n\n"
                "To get started:\n"
                "1. Open Cowork (in the Claude desktop app)\n"
                "2. Click 'Select folder'\n"
                "3. Navigate to your workspace folder\n"
                "4. Send any message \u2014 The Librarian will introduce itself\n\n"
                "That's it. The Librarian remembers everything from here on out."
            )
            root.destroy()
        except Exception as e:
            status_var.set("")
            install_btn.config(state="normal", text="Install")
            messagebox.showerror("Setup Failed", f"Something went wrong:\n{e}")

    install_btn = tk.Button(root, text="Install", command=do_install,
                            font=(font_family, font_base + 2, "bold"), width=18,
                            relief="flat", bg="#0066cc", fg="white",
                            activebackground="#004999", activeforeground="white",
                            cursor="hand2")
    install_btn.pack(pady=(5, 15))

    # ── Footer ──
    tk.Label(root, text=f"v{__version__}  \u2022  Dicta Technologies Inc.",
             font=(font_family, font_base - 2), fg="#aaa", bg=bg).pack(side="bottom", pady=8)

    root.mainloop()


def _self_install_to_path():
    """Best-effort: copy the frozen app to system location and add to PATH.

    Works on Windows (registry), macOS (shell rc files), and Linux (shell rc files).
    This is a bonus for CLI/Claude Code users. If it fails, the workspace
    is still fully functional for Cowork — so failures are silently ignored.
    """
    if not getattr(sys, 'frozen', False):
        return  # Only relevant for frozen builds

    try:
        import shutil
        from src.platform_utils import (
            get_system, get_install_base_dir, get_cli_executable_name, add_to_path
        )

        system = get_system()
        src_exe = sys.executable
        src_dir = os.path.dirname(src_exe)

        # Get platform-specific install directories
        install_base = get_install_base_dir()
        bin_dir = os.path.join(install_base, "bin")
        lib_dir = os.path.join(install_base, "lib")

        os.makedirs(bin_dir, exist_ok=True)

        # Copy the entire frozen bundle to lib/
        if os.path.abspath(src_dir) != os.path.abspath(lib_dir):
            if os.path.exists(lib_dir):
                shutil.rmtree(lib_dir, ignore_errors=True)
            shutil.copytree(src_dir, lib_dir)

        # Copy CLI executable to bin/ for clean PATH entry
        exe_name = get_cli_executable_name()
        src_exe_path = os.path.join(lib_dir, exe_name)
        dst_exe_path = os.path.join(bin_dir, exe_name)
        if os.path.isfile(src_exe_path):
            shutil.copy2(src_exe_path, dst_exe_path)
            # Make executable on Unix-like systems
            if system in ("darwin", "linux"):
                os.chmod(dst_exe_path, 0o755)

        # Add bin/ to user PATH using platform-specific method
        add_to_path(bin_dir)
    except Exception:
        pass  # Best-effort — Cowork doesn't need this


async def cmd_pulse():
    """Heartbeat check — is The Librarian alive and running?

    Lightweight probe that returns status without booting.
    If not alive, returns needs_boot=True so the caller knows to boot.
    """
    from src.core.maintenance import pulse_check

    db_path, _ = _resolve_db_path()
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    result = pulse_check(conn, SESSION_FILE)
    conn.close()
    print(json.dumps(result))


async def cmd_maintain(
    token_budget: int = 0,
    cooldown_hours: float = 4.0,
    force: bool = False,
):
    """Run background maintenance passes on the knowledge graph.

    Checks cooldown first (skip if ran recently), then runs all passes:
    contradiction detection, orphaned correction linking, near-duplicate
    merging, entry promotion, and stale temporal flagging.

    Token budget is dynamic by default: measures the actual DB content size
    and sets the budget to cover the full corpus with 20% headroom. For
    large DBs (enterprise scale), caps at MAINTAIN_BUDGET_CAP to keep
    the duplicate pass (O(n^2)) bounded. At the cap, maintenance rotates
    through the DB across multiple idle windows.
    """
    from src.core.maintenance import MaintenanceEngine, check_cooldown

    # Budget cap for large DBs — keeps the O(n^2) duplicate pass bounded.
    # At 500k tokens (~30k entries), a single pass takes ~2-3s. Beyond that,
    # topic rotation across idle windows is the better strategy.
    MAINTAIN_BUDGET_CAP = 500_000

    lib, mode = _make_librarian()
    session_id = load_session_id()

    # Check cooldown (unless --force)
    if not force:
        can_run, last_completed = check_cooldown(lib.rolodex.conn, cooldown_hours)
        if not can_run:
            print(json.dumps({
                "status": "skipped",
                "reason": "cooldown",
                "last_completed_at": last_completed,
                "cooldown_hours": cooldown_hours,
                "message": f"Last maintenance ran at {last_completed}. Use --force to override.",
            }))
            close_db(lib)
            return

    # Dynamic budget: measure DB, cover it all (with cap for large DBs)
    if token_budget == 0:
        row = lib.rolodex.conn.execute(
            "SELECT SUM(LENGTH(content)) / 4 as est_tokens FROM rolodex_entries WHERE superseded_by IS NULL"
        ).fetchone()
        db_tokens = row["est_tokens"] or 15000
        # 1.2x headroom so passes don't run out mid-scan
        token_budget = min(int(db_tokens * 1.2), MAINTAIN_BUDGET_CAP)
        budget_mode = "dynamic"
    else:
        budget_mode = "manual"

    # Run maintenance
    engine = MaintenanceEngine(
        conn=lib.rolodex.conn,
        session_id=session_id,
        token_budget=token_budget,
    )
    report = engine.run_all()
    report["budget_mode"] = budget_mode

    print(json.dumps(report, indent=2))
    close_db(lib)


async def main():
    if len(sys.argv) < 2:
        # No arguments: launch GUI installer if frozen, else show usage
        if getattr(sys, 'frozen', False):
            cmd_install_gui()
            return
        print(json.dumps({"error": "Usage: librarian_cli.py <boot|ingest|recall|stats|end|topics|window|schema|history|init> [args]"}))
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
            doc_id = None
            source_location = None
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
                elif remaining[i] == "--doc" and i + 1 < len(remaining):
                    i += 1
                    doc_id = remaining[i]
                elif remaining[i] == "--loc" and i + 1 < len(remaining):
                    i += 1
                    source_location = remaining[i]
                i += 1
            await cmd_ingest(role, content, corrects_id=corrects_id, as_user_knowledge=as_user_knowledge,
                             is_summary=is_summary, doc_id=doc_id, source_location=source_location)

        elif cmd == "batch-ingest":
            if len(sys.argv) < 3:
                print(json.dumps({"error": "Usage: librarian_cli.py batch-ingest <file.json|->"}))
                sys.exit(1)
            await cmd_batch_ingest(sys.argv[2])

        elif cmd == "recall":
            if len(sys.argv) < 3:
                print(json.dumps({"error": "Usage: librarian_cli.py recall \"<query>\" [--source conversation|document] [--fresh [hours]]"}))
                sys.exit(1)
            recall_query = sys.argv[2]
            recall_source = None
            recall_fresh = False
            recall_fresh_hours = 48.0
            remaining = sys.argv[3:]
            ri = 0
            while ri < len(remaining):
                if remaining[ri] == "--source" and ri + 1 < len(remaining):
                    ri += 1
                    recall_source = remaining[ri]
                elif remaining[ri] == "--fresh":
                    recall_fresh = True
                    # Optional hours argument
                    if ri + 1 < len(remaining) and not remaining[ri + 1].startswith("--"):
                        try:
                            recall_fresh_hours = float(remaining[ri + 1])
                            ri += 1
                        except ValueError:
                            pass  # Not a number, leave default
                ri += 1
            await cmd_recall(recall_query, source_type=recall_source, fresh=recall_fresh, fresh_hours=recall_fresh_hours)

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

        elif cmd == "browse":
            subcmd = sys.argv[2].lower() if len(sys.argv) > 2 else "recent"
            raw_args = sys.argv[3:] if len(sys.argv) > 3 else []
            browse_json = "--json" in raw_args
            browse_args = [a for a in raw_args if a != "--json"]
            await cmd_browse(subcmd, browse_args, as_json=browse_json)

        elif cmd == "register-doc":
            if len(sys.argv) < 3:
                print(json.dumps({"error": "Usage: librarian_cli.py register-doc \"<file_path>\" [--title \"...\"]"}))
                sys.exit(1)
            reg_path = sys.argv[2]
            reg_title = None
            remaining = sys.argv[3:]
            ri = 0
            while ri < len(remaining):
                if remaining[ri] == "--title" and ri + 1 < len(remaining):
                    ri += 1
                    reg_title = remaining[ri]
                ri += 1
            await cmd_register_doc(reg_path, title=reg_title)

        elif cmd == "read-doc":
            if len(sys.argv) < 3:
                print(json.dumps({"error": "Usage: librarian_cli.py read-doc <doc_id> [--pages \"1-5\"]"}))
                sys.exit(1)
            rd_id = sys.argv[2]
            rd_pages = None
            remaining = sys.argv[3:]
            ri = 0
            while ri < len(remaining):
                if remaining[ri] == "--pages" and ri + 1 < len(remaining):
                    ri += 1
                    rd_pages = remaining[ri]
                ri += 1
            await cmd_read_doc(rd_id, pages=rd_pages)

        elif cmd == "docs":
            subcmd = sys.argv[2].lower() if len(sys.argv) > 2 else "list"
            args = sys.argv[3:] if len(sys.argv) > 3 else []
            await cmd_docs(subcmd, args)

        elif cmd == "manifest":
            subcmd = sys.argv[2].lower() if len(sys.argv) > 2 else "stats"
            args = sys.argv[3:] if len(sys.argv) > 3 else []
            await cmd_manifest(subcmd, args)

        elif cmd == "suggest-focus":
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 3
            await cmd_suggest_focus(limit=limit)

        elif cmd == "focus-boot":
            if len(sys.argv) < 3:
                print(json.dumps({"error": "Usage: librarian_cli.py focus-boot '<json_array_of_topic_ids>'"}))
                sys.exit(1)
            await cmd_focus_boot(sys.argv[2])

        elif cmd == "projects":
            subcmd = sys.argv[2].lower() if len(sys.argv) > 2 else "list"
            args = sys.argv[3:] if len(sys.argv) > 3 else []
            await cmd_projects(subcmd, args)

        elif cmd == "init":
            if len(sys.argv) < 3:
                print(json.dumps({"error": "Usage: librarian_cli.py init <target_folder>"}))
                sys.exit(1)
            cmd_init(sys.argv[2])
            return

        elif cmd == "install":
            # Explicit install command — launches the GUI
            cmd_install_gui()
            return

        elif cmd == "schema":
            await cmd_schema()

        elif cmd == "history":
            subcmd = sys.argv[2].lower() if len(sys.argv) > 2 else "range"
            args = sys.argv[3:] if len(sys.argv) > 3 else []
            await cmd_history(subcmd, args)

        elif cmd == "pulse":
            await cmd_pulse()

        elif cmd == "maintain":
            # Parse optional flags
            maintain_budget = 0  # 0 = dynamic (auto-size to DB)
            maintain_cooldown = 4.0
            maintain_force = False
            remaining = sys.argv[2:]
            mi = 0
            while mi < len(remaining):
                if remaining[mi] == "--budget" and mi + 1 < len(remaining):
                    mi += 1
                    maintain_budget = int(remaining[mi])
                elif remaining[mi] == "--cooldown" and mi + 1 < len(remaining):
                    mi += 1
                    maintain_cooldown = float(remaining[mi])
                elif remaining[mi] == "--force":
                    maintain_force = True
                mi += 1
            await cmd_maintain(
                token_budget=maintain_budget,
                cooldown_hours=maintain_cooldown,
                force=maintain_force,
            )

        elif cmd == "compile":
            # Parse: compile --content "yaml..." OR compile <file_path>
            compile_content = None
            compile_file = None
            remaining = sys.argv[2:]
            ci = 0
            while ci < len(remaining):
                if remaining[ci] == "--content" and ci + 1 < len(remaining):
                    ci += 1
                    compile_content = remaining[ci]
                elif not remaining[ci].startswith("--"):
                    compile_file = remaining[ci]
                ci += 1
            await cmd_compile(content=compile_content, file_path=compile_file)

        elif cmd == "settings":
            subcmd = sys.argv[2] if len(sys.argv) > 2 else None
            remaining = sys.argv[3:] if len(sys.argv) > 3 else []
            await cmd_settings(subcmd, remaining)

        else:
            print(json.dumps({"error": f"Unknown command: {cmd}. Use boot|ingest|recall|remember|correct|profile|compile|settings|scan|retag|stats|end|topics|window|manifest|schema|history|browse|register-doc|read-doc|docs|suggest-focus|focus-boot|projects|pulse|maintain"}))
            sys.exit(1)

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
