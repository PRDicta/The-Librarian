#!/usr/bin/env python3
"""
The Librarian — Build Pipeline

Automates the full build from source to distributable .exe:
  1. Create clean virtual environment with production deps
  2. Install CPU-only PyTorch
  3. Pre-download the all-MiniLM-L6-v2 embedding model
  4. Run PyInstaller with librarian.spec
  5. Verify the frozen build

The resulting dist/librarian/librarian.exe IS the installer.
Double-clicking it (or running without arguments) launches a tkinter GUI
that lets the user pick a workspace folder and initializes everything.
No Inno Setup or other external installer tools needed.

Usage:
    python build.py              # Lean build — small installer .exe (~30 MB)
    python build.py --full       # Full build — self-contained CLI (~300+ MB)
    python build.py --skip-venv  # Skip venv creation (use current env)
    python build.py --clean      # Remove build artifacts first

Requirements:
    - Python 3.10+ on Windows
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ─── Constants ────────────────────────────────────────────────────────

BUILD_DIR = Path(__file__).parent.resolve()
VENV_DIR = BUILD_DIR / ".build_venv"
DIST_DIR = BUILD_DIR / "dist"
MODEL_DIR = BUILD_DIR / "lib" / "models" / "all-MiniLM-L6-v2"
SPEC_FILE = BUILD_DIR / "librarian.spec"
# ISS_FILE removed — the .exe IS the installer now (tkinter GUI built in)

MODEL_NAME = "all-MiniLM-L6-v2"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# Files the model needs (everything sentence-transformers expects)
MODEL_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "modules.json",
    "sentence_bert_config.json",
]


def log(msg: str):
    """Timestamped build log."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run(cmd: list[str], **kwargs):
    """Run a subprocess, raising on failure."""
    log(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, **kwargs)
    return result


def get_venv_python() -> str:
    """Return path to the venv Python executable."""
    if platform.system() == "Windows":
        return str(VENV_DIR / "Scripts" / "python.exe")
    return str(VENV_DIR / "bin" / "python")


def get_venv_pip() -> str:
    """Return path to the venv pip executable."""
    if platform.system() == "Windows":
        return str(VENV_DIR / "Scripts" / "pip.exe")
    return str(VENV_DIR / "bin" / "pip")


def pip_install(args: list[str]):
    """Run pip via 'python -m pip' to avoid Windows self-upgrade lock."""
    python = get_venv_python()
    run([python, "-m", "pip"] + args)


def robust_rmtree(path: Path, retries: int = 3, delay: float = 2.0):
    """shutil.rmtree with retry logic for Windows file locks."""
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt < retries - 1:
                log(f"  Locked files in {path}, retrying in {delay}s... ({attempt + 1}/{retries})")
                time.sleep(delay)
            else:
                raise


# ─── Step 1: Virtual Environment ─────────────────────────────────────

def create_venv():
    """Create a clean virtual environment with production dependencies."""
    log("Step 1: Creating clean virtual environment...")

    if VENV_DIR.exists():
        log(f"  Removing existing venv at {VENV_DIR}")
        robust_rmtree(VENV_DIR)

    run([sys.executable, "-m", "venv", str(VENV_DIR)])

    # Upgrade pip (use python -m pip to avoid Windows exe-locking issue)
    pip_install(["install", "--upgrade", "pip", "setuptools", "wheel"])

    # Install CPU-only PyTorch first (before sentence-transformers pulls in GPU version)
    log("  Installing CPU-only PyTorch...")
    pip_install(["install", "torch", "--index-url", TORCH_CPU_INDEX])

    # Install production dependencies (exclude test deps)
    log("  Installing production dependencies...")
    prod_deps = [
        "sentence-transformers>=2.2.0",
        "numpy>=1.24.0",
        "anthropic>=0.40.0",
        "rich>=13.0.0",
        "safetensors",
        "tokenizers",
        "transformers",
        "huggingface-hub",
        "httpx",
    ]
    pip_install(["install"] + prod_deps)

    # Install PyInstaller
    pip_install(["install", "pyinstaller>=6.0"])

    log("  Virtual environment ready.")


# ─── Step 2: Download Model ──────────────────────────────────────────

def download_model():
    """Pre-download the embedding model for bundling."""
    log("Step 2: Downloading embedding model...")

    if MODEL_DIR.exists() and (MODEL_DIR / "config.json").exists():
        log(f"  Model already exists at {MODEL_DIR}, skipping download.")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    python = get_venv_python() if VENV_DIR.exists() else sys.executable

    # Use sentence-transformers to download, then copy to our bundle location
    download_script = f"""
import os, shutil
from sentence_transformers import SentenceTransformer

# This downloads to HuggingFace cache
model = SentenceTransformer("{MODEL_NAME}")
cache_path = model[0].auto_model.config._name_or_path

# If cache_path is just the model name, find the actual cached directory
if not os.path.isdir(cache_path):
    from huggingface_hub import snapshot_download
    cache_path = snapshot_download(repo_id="sentence-transformers/{MODEL_NAME}")

print(f"MODEL_CACHE_PATH={{cache_path}}")

# Copy model files to bundle location
target = r"{MODEL_DIR}"
for item in os.listdir(cache_path):
    src = os.path.join(cache_path, item)
    dst = os.path.join(target, item)
    if os.path.isfile(src):
        shutil.copy2(src, dst)
    elif os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

print("Model files copied to bundle directory.")
"""

    run([python, "-c", download_script])

    # Verify essential files exist
    missing = [f for f in MODEL_FILES if not (MODEL_DIR / f).exists()]
    if missing:
        log(f"  WARNING: Missing model files: {missing}")
        log("  The model may still work if safetensors/pytorch_model.bin is present.")

    # Check for model weights
    has_weights = (
        (MODEL_DIR / "model.safetensors").exists()
        or (MODEL_DIR / "pytorch_model.bin").exists()
    )
    if not has_weights:
        raise FileNotFoundError(
            f"No model weights found in {MODEL_DIR}. "
            "Expected model.safetensors or pytorch_model.bin."
        )

    model_size = sum(f.stat().st_size for f in MODEL_DIR.rglob("*") if f.is_file())
    log(f"  Model downloaded: {model_size / 1024 / 1024:.1f} MB")


# ─── Step 3: PyInstaller Build ───────────────────────────────────────

def run_pyinstaller():
    """Run PyInstaller with the custom spec file."""
    log("Step 3: Running PyInstaller...")

    if not SPEC_FILE.exists():
        raise FileNotFoundError(f"Spec file not found: {SPEC_FILE}")

    # Clean previous build (use robust_rmtree to handle Windows file locks)
    clean_targets = [DIST_DIR / "librarian", BUILD_DIR / "build"]
    if platform.system() == "Darwin":
        clean_targets.append(DIST_DIR / "The Librarian.app")
    for d in clean_targets:
        if d.exists():
            log(f"  Cleaning {d}")
            robust_rmtree(d)

    python = get_venv_python() if VENV_DIR.exists() else sys.executable

    run([
        python, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        str(SPEC_FILE),
    ], cwd=str(BUILD_DIR))

    # Verify output exists (different paths per platform)
    if platform.system() == "Darwin":
        exe_path = DIST_DIR / "The Librarian.app" / "Contents" / "MacOS" / "librarian"
        dist_root = DIST_DIR / "The Librarian.app"
        dist_label = "dist/The Librarian.app/"
    elif platform.system() == "Windows":
        exe_path = DIST_DIR / "librarian" / "librarian.exe"
        dist_root = DIST_DIR / "librarian"
        dist_label = "dist/librarian/"
    else:
        exe_path = DIST_DIR / "librarian" / "librarian"
        dist_root = DIST_DIR / "librarian"
        dist_label = "dist/librarian/"

    if not exe_path.exists():
        raise FileNotFoundError(f"PyInstaller output not found: {exe_path}")

    dist_size = sum(f.stat().st_size for f in dist_root.rglob("*") if f.is_file())
    log(f"  PyInstaller complete: {dist_size / 1024 / 1024:.0f} MB in {dist_label}")


# ─── Step 4: Verify Frozen Build ─────────────────────────────────────

def verify_build():
    """Run the frozen executable to verify it works."""
    log("Step 4: Verifying frozen build...")

    if platform.system() == "Darwin":
        exe_path = DIST_DIR / "The Librarian.app" / "Contents" / "MacOS" / "librarian"
    elif platform.system() == "Windows":
        exe_path = DIST_DIR / "librarian" / "librarian.exe"
    else:
        exe_path = DIST_DIR / "librarian" / "librarian"

    if not exe_path.exists():
        raise FileNotFoundError(f"Executable not found: {exe_path}")

    # Test 1: --help / basic invocation
    try:
        result = subprocess.run(
            [str(exe_path), "stats"],
            capture_output=True, text=True, timeout=60,
            cwd=str(BUILD_DIR),  # Needs a directory for rolodex.db
        )
        log(f"  stats exit code: {result.returncode}")
        if result.stdout:
            log(f"  stdout: {result.stdout[:200]}")
        if result.returncode != 0 and result.stderr:
            log(f"  stderr: {result.stderr[:300]}")
    except subprocess.TimeoutExpired:
        log("  WARNING: stats command timed out (60s). May be slow on first run.")

    # Test 2: Version check (import test)
    try:
        result = subprocess.run(
            [str(exe_path), "boot"],
            capture_output=True, text=True, timeout=120,
            cwd=str(BUILD_DIR),
        )
        if '"status": "ok"' in result.stdout:
            log("  boot command: OK (valid JSON output)")
        else:
            log(f"  boot command: unexpected output: {result.stdout[:200]}")
    except subprocess.TimeoutExpired:
        log("  WARNING: boot command timed out (120s). First-run model load can be slow.")

    log("  Build verification complete.")


# ─── Step 5: Generate version.json ────────────────────────────────────

def generate_version_json():
    """Create version.json for the update-check mechanism."""
    log("Step 5: Generating version.json...")

    import json
    from src.__version__ import __version__

    version_data = {
        "version": __version__,
        "download_url": f"https://github.com/PRDicta/The-Librarian/releases/download/v{__version__}/TheLibrarianSetup.exe",
        "release_notes": f"https://github.com/PRDicta/The-Librarian/releases/tag/v{__version__}",
        "message": f"The Librarian v{__version__} is available.",
    }

    version_file = BUILD_DIR / "version.json"
    with open(version_file, "w") as f:
        json.dump(version_data, f, indent=2)

    log(f"  version.json written ({__version__})")


# ─── Clean ────────────────────────────────────────────────────────────

def clean():
    """Remove all build artifacts."""
    log("Cleaning build artifacts...")
    for d in [
        DIST_DIR,
        BUILD_DIR / "build",
        BUILD_DIR / "Output",
        VENV_DIR,
    ]:
        if d.exists():
            log(f"  Removing {d}")
            shutil.rmtree(d)

    # Don't remove lib/models — that's a deliberate download
    log("  Clean complete. (Model cache at lib/models/ preserved.)")


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build The Librarian installer")
    parser.add_argument("--full", action="store_true",
                        help="Full build with ML stack (~300+ MB). Default is lean installer (~30 MB)")
    parser.add_argument("--skip-venv", action="store_true",
                        help="Skip virtual environment creation (use current env)")
    parser.add_argument("--clean", action="store_true",
                        help="Remove build artifacts before building")
    parser.add_argument("--clean-only", action="store_true",
                        help="Only clean, don't build")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify an existing build")
    args = parser.parse_args()

    start = time.time()
    log(f"The Librarian Build Pipeline")
    log(f"Platform: {platform.system()} {platform.machine()}")
    log(f"Python: {sys.version}")
    log(f"Build dir: {BUILD_DIR}")

    if args.clean or args.clean_only:
        clean()
        if args.clean_only:
            return

    if args.verify_only:
        verify_build()
        return

    # Set build mode for the spec file to read
    build_mode = "full" if args.full else "lean"
    os.environ["LIBRARIAN_BUILD_MODE"] = build_mode
    log(f"Build mode: {build_mode.upper()}")
    if not args.full:
        log("  (Lean installer — no PyTorch/model bundled. Use --full for self-contained CLI.)")

    try:
        if not args.skip_venv:
            create_venv()
        else:
            log("Step 1: Skipping venv (--skip-venv)")

        if args.full:
            download_model()
        else:
            log("Step 2: Skipping model download (lean build)")

        run_pyinstaller()
        verify_build()
        generate_version_json()

        elapsed = time.time() - start
        log(f"Build complete in {elapsed / 60:.1f} minutes.")

        # Summary
        print("\n" + "=" * 60)
        print("BUILD SUMMARY")
        print("=" * 60)
        print(f"  Mode:        {build_mode.upper()}")
        print(f"  Platform:    {platform.system()} {platform.machine()}")
        if platform.system() == "Darwin":
            print(f"  App bundle:  dist/The Librarian.app/")
            print(f"  Executable:  dist/The Librarian.app/Contents/MacOS/librarian")
        elif platform.system() == "Windows":
            print(f"  Frozen app:  dist/librarian/")
            print(f"  Installer:   dist/librarian/librarian.exe  (double-click to install)")
        else:
            print(f"  Frozen app:  dist/librarian/")
            print(f"  Executable:  dist/librarian/librarian")
        print(f"  Version:     version.json")
        print(f"  Duration:    {elapsed / 60:.1f} min")
        print("=" * 60)

    except Exception as e:
        log(f"BUILD FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
