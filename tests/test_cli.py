"""
The Librarian — CLI Integration Tests (Phase 5)

Tests the Cowork CLI wrapper (librarian_cli.py) end-to-end.
Each test spawns actual subprocesses to validate the full pipeline.
Uses a temp directory for DB/session files to avoid touching the real rolodex.
"""
import json
import os
import subprocess
import sys
import tempfile
import shutil

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(SCRIPT_DIR, "librarian_cli.py")
PYTHON = sys.executable

passed = 0
failed = 0
test_dir = None
db_path = None
session_path = None


def setup():
    """Create temp directory for test DB and session file."""
    global test_dir, db_path, session_path
    test_dir = tempfile.mkdtemp(prefix="librarian_cli_test_")
    db_path = os.path.join(test_dir, "test_rolodex.db")
    session_path = os.path.join(test_dir, "test_session.json")


def teardown():
    """Clean up temp directory."""
    global test_dir
    if test_dir and os.path.exists(test_dir):
        shutil.rmtree(test_dir)


# Pytest hooks — setup/teardown don't run automatically unless wired up
def setup_module(module):
    setup()

def teardown_module(module):
    teardown()


def run_cli(*args):
    """Run CLI with test env vars. Returns (returncode, parsed_output, raw_stdout)."""
    cmd = [PYTHON, CLI] + list(args)
    # Filter out None values — subprocess.run requires all env values to be strings
    env = {k: v for k, v in os.environ.items() if v is not None}
    env["LIBRARIAN_DB_PATH"] = db_path
    env["LIBRARIAN_SESSION_FILE"] = session_path
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        cwd=SCRIPT_DIR,
        env=env,
        timeout=30,
    )
    raw = result.stdout.strip()
    # Include stderr in raw output for debugging failures
    if result.returncode != 0 and result.stderr:
        raw = raw + "\nSTDERR: " + result.stderr.strip()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        parsed = raw
    return result.returncode, parsed, raw


def run_test(name, fn):
    """Run a test function and track results."""
    global passed, failed
    try:
        fn()
        print(f"  PASS: {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {name} — {e}")
        failed += 1


# ─── Tests ──────────────────────────────────────────────────────────────

def test_boot_new_session():
    """First boot creates a new session and returns valid JSON."""
    code, data, _ = run_cli("boot")
    assert code == 0, f"Exit code {code}, output: {data}"
    assert data["status"] == "ok"
    assert "session_id" in data
    assert data["resumed"] is False
    assert isinstance(data["total_entries"], int)
    assert os.path.exists(session_path), "Session file not created"


def test_boot_resume():
    """Second boot resumes the existing session."""
    # First boot
    _, data1, _ = run_cli("boot")
    session_id = data1["session_id"]

    # Second boot — should resume
    code, data2, _ = run_cli("boot")
    assert code == 0
    assert data2["resumed"] is True
    assert data2["session_id"] == session_id


def test_ingest_user_message():
    """Ingesting a user message returns entry count."""
    run_cli("boot")
    code, data, _ = run_cli("ingest", "user", "I prefer dark mode and always use Python 3.12 with type hints.")
    assert code == 0, f"Exit code {code}, output: {data}"
    assert data["ingested"] >= 1


def test_ingest_assistant_message():
    """Ingesting an assistant message works too."""
    code, data, _ = run_cli("ingest", "assistant", "Noted! I will use dark mode styling and Python 3.12 features throughout.")
    assert code == 0, f"Exit code {code}, output: {data}"
    assert data["ingested"] >= 1


def test_recall_finds_ingested():
    """Recall returns ingested content."""
    code, output, raw = run_cli("recall", "dark mode preferences")
    assert code == 0, f"Exit code {code}, output: {raw}"
    assert "dark mode" in raw.lower() or "RETRIEVED FROM MEMORY" in raw, f"Expected context, got: {raw}"


def test_batch_ingest():
    """Batch-ingest multiple messages in one process spawn."""
    import tempfile
    batch = json.dumps([
        {"role": "user", "content": "My project uses FastAPI with SQLAlchemy ORM for the backend."},
        {"role": "assistant", "content": "Noted: FastAPI + SQLAlchemy backend stack for this project."},
    ])
    batch_file = os.path.join(test_dir, "batch.json")
    with open(batch_file, "w", encoding="utf-8") as f:
        f.write(batch)

    code, data, raw = run_cli("batch-ingest", batch_file)
    assert code == 0, f"Exit code {code}, output: {raw}"
    assert data["messages_processed"] == 2
    assert data["ingested"] >= 2, f"Expected >= 2 ingested, got {data['ingested']}"


def test_stats_returns_json():
    """Stats returns valid JSON with expected keys."""
    code, data, _ = run_cli("stats")
    assert code == 0
    assert isinstance(data, dict)
    assert "total_entries" in data
    assert "conversation_id" in data
    assert data["total_entries"] >= 2, f"Expected >= 2 entries, got {data['total_entries']}"


def test_end_session():
    """End closes the session and clears the session file."""
    code, data, _ = run_cli("end", "Tested CLI integration")
    assert code == 0
    assert "ended" in data
    assert not os.path.exists(session_path), "Session file should be removed"


def test_error_bad_command():
    """Unknown command returns error JSON and exit 1."""
    code, data, _ = run_cli("foobar")
    assert code == 1
    assert "error" in data


# ─── Run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running CLI Integration tests (Phase 5)...\n")
    setup()
    try:
        run_test("boot_new_session", test_boot_new_session)
        run_test("boot_resume", test_boot_resume)
        run_test("ingest_user_message", test_ingest_user_message)
        run_test("ingest_assistant_message", test_ingest_assistant_message)
        run_test("recall_finds_ingested", test_recall_finds_ingested)
        run_test("batch_ingest", test_batch_ingest)
        run_test("stats_returns_json", test_stats_returns_json)
        run_test("end_session", test_end_session)
        run_test("error_bad_command", test_error_bad_command)
    finally:
        teardown()

    print()
    if failed:
        print(f"{failed} test(s) FAILED, {passed} passed.")
        sys.exit(1)
    else:
        print("All CLI Integration tests passed!")
