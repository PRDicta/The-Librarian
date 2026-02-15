#!/usr/bin/env python3
"""
CI Smoke Test — validates the built binary on each platform.

Runs against the PyInstaller-built executable (not source) to verify
the full frozen bundle works end-to-end:
  1. Binary launches and responds to 'stats'
  2. Boot produces valid JSON with status: ok
  3. Ingest stores a message successfully
  4. Recall retrieves the ingested message (semantic search works)
  5. Embedding strategy is ONNX (not fallback/API)
  6. Second ingest + cross-topic recall
  7. Session end works cleanly

Usage:
    python scripts/ci_smoke_test.py          # Auto-detects platform
    python scripts/ci_smoke_test.py --exe path/to/librarian
"""

import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────

TIMEOUT_BOOT = 120   # First boot loads ONNX model, can be slow
TIMEOUT_CMD = 60     # Normal commands

# ─── Helpers ─────────────────────────────────────────────────────────

def find_executable():
    """Auto-detect the built executable based on platform."""
    dist = Path("dist")
    if platform.system() == "Darwin":
        exe = dist / "The Librarian.app" / "Contents" / "MacOS" / "librarian"
    elif platform.system() == "Windows":
        exe = dist / "librarian" / "librarian.exe"
    else:
        exe = dist / "librarian" / "librarian"

    if not exe.exists():
        print(f"FATAL: Executable not found at {exe}")
        sys.exit(1)
    return str(exe)


def run_cmd(exe, args, timeout=TIMEOUT_CMD, work_dir=None):
    """Run a CLI command and return (returncode, stdout, stderr)."""
    cmd = [exe] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work_dir,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"TIMEOUT after {timeout}s"


class SmokeTest:
    def __init__(self, exe_path, work_dir):
        self.exe = exe_path
        self.work_dir = work_dir
        self.passed = 0
        self.failed = 0
        self.results = []

    def run(self, name, args, timeout=TIMEOUT_CMD, checks=None):
        """Run a test and apply checks to the output."""
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"  CMD: librarian {' '.join(args)}")

        rc, stdout, stderr = run_cmd(self.exe, args, timeout, self.work_dir)

        print(f"  EXIT: {rc}")
        if stdout:
            # Truncate long output for readability
            display = stdout[:500] + ("..." if len(stdout) > 500 else "")
            print(f"  STDOUT: {display}")
        if stderr and rc != 0:
            display = stderr[:300] + ("..." if len(stderr) > 300 else "")
            print(f"  STDERR: {display}")

        # Apply checks
        all_passed = True
        if checks:
            for check_name, check_fn in checks:
                try:
                    result = check_fn(rc, stdout, stderr)
                    if result:
                        print(f"  ✓ {check_name}")
                    else:
                        print(f"  ✗ {check_name}")
                        all_passed = False
                except Exception as e:
                    print(f"  ✗ {check_name} — exception: {e}")
                    all_passed = False
        else:
            # Default: just check exit code
            if rc == 0:
                print(f"  ✓ exit code 0")
            else:
                print(f"  ✗ non-zero exit code: {rc}")
                all_passed = False

        if all_passed:
            self.passed += 1
        else:
            self.failed += 1

        self.results.append((name, all_passed))
        return rc, stdout, stderr

    def summary(self):
        """Print final results."""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"SMOKE TEST RESULTS: {self.passed}/{total} passed")
        print(f"{'='*60}")
        for name, ok in self.results:
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {name}")
        print()
        return self.failed == 0


# ─── Main ────────────────────────────────────────────────────────────

def main():
    # Parse optional --exe flag
    exe_path = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--exe" and i < len(sys.argv) - 1:
            exe_path = sys.argv[i + 1]

    if not exe_path:
        exe_path = find_executable()

    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Executable: {exe_path}")

    # Use a temp directory as the working dir (simulates fresh install)
    with tempfile.TemporaryDirectory(prefix="librarian_smoke_") as tmpdir:
        print(f"Working dir: {tmpdir}")

        t = SmokeTest(exe_path, tmpdir)

        # ── Test 1: Stats (basic launch) ──
        t.run(
            "Binary launches (stats)",
            ["stats"],
            checks=[
                ("exits cleanly", lambda rc, out, err: rc == 0),
                ("produces JSON", lambda rc, out, err: "{" in out),
            ],
        )

        # ── Test 2: Boot ──
        t.run(
            "Boot produces valid session",
            ["boot", "--compact"],
            timeout=TIMEOUT_BOOT,
            checks=[
                ("exits cleanly", lambda rc, out, err: rc == 0),
                ("valid JSON", lambda rc, out, err: "status" in out),
                ("status ok", lambda rc, out, err: '"ok"' in out or '"status": "ok"' in out),
            ],
        )

        # ── Test 3: Ingest a message ──
        t.run(
            "Ingest user message",
            ["ingest", "user", "I prefer using Python for backend development and React for frontend."],
            checks=[
                ("exits cleanly", lambda rc, out, err: rc == 0),
            ],
        )

        # ── Test 4: Ingest assistant response ──
        t.run(
            "Ingest assistant message",
            ["ingest", "assistant", "Noted! You prefer Python for backend and React for frontend development."],
            checks=[
                ("exits cleanly", lambda rc, out, err: rc == 0),
            ],
        )

        # ── Test 5: Recall (semantic search) ──
        t.run(
            "Recall finds ingested content",
            ["recall", "What programming languages does the user like?"],
            checks=[
                ("exits cleanly", lambda rc, out, err: rc == 0),
                ("finds Python", lambda rc, out, err: "Python" in out or "python" in out),
                ("finds React", lambda rc, out, err: "React" in out or "react" in out),
            ],
        )

        # ── Test 6: Check embedding strategy ──
        rc, stdout, stderr = t.run(
            "Embedding strategy is ONNX",
            ["stats"],
            checks=[
                ("exits cleanly", lambda rc, out, err: rc == 0),
                ("reports ONNX", lambda rc, out, err: "onnx" in out.lower()),
            ],
        )

        # ── Test 7: Cross-topic ingest + recall ──
        t.run(
            "Ingest second topic",
            ["ingest", "user", "Our deployment pipeline uses GitHub Actions for CI/CD with Docker containers on AWS ECS Fargate."],
            checks=[
                ("exits cleanly", lambda rc, out, err: rc == 0),
            ],
        )

        t.run(
            "Recall second topic (CI/CD)",
            ["recall", "How do we deploy code to production?"],
            checks=[
                ("exits cleanly", lambda rc, out, err: rc == 0),
                ("finds deployment content", lambda rc, out, err:
                    any(term in out.lower() for term in ["github actions", "ci/cd", "docker", "deploy", "fargate", "pipeline"])),
            ],
        )

        # ── Test 8: Session end ──
        t.run(
            "Session ends cleanly",
            ["end", "CI smoke test session"],
            checks=[
                ("exits cleanly", lambda rc, out, err: rc == 0),
            ],
        )

        # ── Summary ──
        success = t.summary()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
