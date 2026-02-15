# -*- mode: python ; coding: utf-8 -*-
"""
The Librarian — PyInstaller Spec File

Freezes the CLI application with all dependencies into a self-contained
directory. Bundles CPU-only PyTorch + sentence-transformers + the
all-MiniLM-L6-v2 embedding model for offline operation.

Build command:
    pyinstaller librarian.spec

Output (Windows):
    dist/librarian/              — frozen application folder
    dist/librarian/librarian.exe — entry point

Output (macOS):
    dist/The Librarian.app/      — macOS application bundle
"""

import os
import sys
import platform as plat
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# ─── Paths ────────────────────────────────────────────────────────────
SPEC_DIR = os.path.abspath(SPECPATH)
MODEL_DIR = os.path.join(SPEC_DIR, "lib", "models", "all-MiniLM-L6-v2")

# ─── Platform Detection ──────────────────────────────────────────────
BUILD_PLATFORM = plat.system()  # "Windows", "Darwin", "Linux"
IS_MACOS = BUILD_PLATFORM == "Darwin"
IS_WINDOWS = BUILD_PLATFORM == "Windows"

# ─── Build Mode ──────────────────────────────────────────────────────
# LEAN mode: small installer-only .exe (~30 MB). No PyTorch, no model.
# FULL mode: self-contained CLI with all ML deps (~300+ MB).
# Set via environment variable: LIBRARIAN_BUILD_MODE=lean|full (default: lean)
BUILD_MODE = os.environ.get("LIBRARIAN_BUILD_MODE", "lean").lower()
IS_LEAN = BUILD_MODE == "lean"

if not IS_LEAN:
    # Full build needs the pre-downloaded model
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"Bundled model not found at {MODEL_DIR}. "
            "Run build.py first to download the model."
        )

# ─── Hidden Imports ───────────────────────────────────────────────────
# PyInstaller can't trace these from static analysis of librarian_cli.py

# Base imports needed by both lean and full builds
hidden_imports = [
    # Standard lib that PyInstaller sometimes misses
    "sqlite3",
    "hashlib",
    "asyncio",
    "dataclasses",
    "tkinter",
    "tkinter.filedialog",
    "tkinter.messagebox",
]

if IS_LEAN:
    # Lean build: installer GUI + ONNX embeddings. No PyTorch/ML stack.
    hidden_imports += [
        # ONNX embedding runtime
        "numpy",
        "numpy.core",
        "numpy.core._methods",
        "numpy.lib",
        "numpy.lib.format",
        "onnxruntime",
        # Tokenizer for ONNX model
        "tokenizers",
        "tokenizers.implementations",
        "tokenizers.models",
        "tokenizers.normalizers",
        "tokenizers.pre_tokenizers",
        "tokenizers.processors",
        "tokenizers.decoders",
        # Networking for Anthropic SDK
        "anthropic",
        "httpx",
        "httpcore",
        "anyio",
        "sniffio",
        "certifi",
        "h11",
        "rich",
        "rich.console",
        "rich.text",
    ]
else:
    # Full build: everything including ML stack
    hidden_imports += [
        # Core ML stack
        "sentence_transformers",
        "sentence_transformers.models",
        "sentence_transformers.util",
        "torch",
        "torch.nn",
        "torch.nn.functional",
        "numpy",
        "numpy.core",
        "numpy.core._methods",
        "numpy.lib",
        "numpy.lib.format",
        # Tokenizer
        "tokenizers",
        "transformers",
        "transformers.models.bert",
        "transformers.models.bert.tokenization_bert",
        "transformers.models.bert.modeling_bert",
        "transformers.models.bert.configuration_bert",
        # HuggingFace Hub
        "huggingface_hub",
        "huggingface_hub.utils",
        # Networking
        "anthropic",
        "httpx",
        "httpcore",
        "anyio",
        "sniffio",
        "certifi",
        "h11",
        # CLI
        "rich",
        "rich.console",
        "rich.text",
        # Safetensors
        "safetensors",
        "safetensors.torch",
        # Tokenizers backend
        "tokenizers.implementations",
        "tokenizers.models",
        "tokenizers.normalizers",
        "tokenizers.pre_tokenizers",
        "tokenizers.processors",
        "tokenizers.decoders",
        "tokenizers.trainers",
    ]
    # Also pull in all sentence_transformers submodules dynamically
    hidden_imports += collect_submodules("sentence_transformers")

# ─── Data Files ───────────────────────────────────────────────────────
datas = []

if not IS_LEAN:
    # Full build: collect package metadata for ML stack
    datas += collect_data_files("sentence_transformers")
    datas += collect_data_files("transformers")
    datas += collect_data_files("tokenizers")

    # Bundle the pre-downloaded embedding model
    datas += [(MODEL_DIR, os.path.join("models", "all-MiniLM-L6-v2"))]

# Both modes: bundle INSTRUCTIONS.md
instructions_path = os.path.join(SPEC_DIR, "INSTRUCTIONS.md")
if os.path.isfile(instructions_path):
    datas += [(instructions_path, ".")]

# Both modes: bundle version info
version_path = os.path.join(SPEC_DIR, "src", "__version__.py")
if os.path.isfile(version_path):
    datas += [(version_path, os.path.join("src"))]

# Bundle the Cowork-ready source tree for `librarian init` extraction.
# These get frozen into _cowork_source/ inside the bundle so init can
# copy them into a user's workspace folder.
cowork_src_dir = os.path.join(SPEC_DIR, "src")
if os.path.isdir(cowork_src_dir):
    for root, dirs, files in os.walk(cowork_src_dir):
        # Skip __pycache__ and test dirs
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache")]
        for f in files:
            if f.endswith((".pyc", ".pyo")):
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(root, SPEC_DIR)
            datas += [(full, os.path.join("_cowork_source", rel))]

# Bundle the CLI and main.py for init extraction
for init_file in ["librarian_cli.py", "main.py", "requirements.txt", "requirements-onnx.txt", "requirements-ml.txt"]:
    init_path = os.path.join(SPEC_DIR, init_file)
    if os.path.isfile(init_path):
        datas += [(init_path, "_cowork_source")]

# Bundle ONNX model for Cowork VM deployment (~25 MB)
# This eliminates the need for PyTorch (~800 MB) at runtime
onnx_model_dir = os.path.join(SPEC_DIR, "lib", "models", "all-MiniLM-L6-v2")
for onnx_file in ["model.onnx", "tokenizer.json"]:
    onnx_path = os.path.join(onnx_model_dir, onnx_file)
    if os.path.isfile(onnx_path):
        # _cowork_source/ path: for extraction into user workspaces
        datas += [(onnx_path, os.path.join("_cowork_source", "models", "all-MiniLM-L6-v2"))]
        # models/ path: for direct use by the frozen binary via sys._MEIPASS
        datas += [(onnx_path, os.path.join("models", "all-MiniLM-L6-v2"))]

# ─── Exclusions ───────────────────────────────────────────────────────
# Always exclude dev/test tools
excludes = [
    "pytest", "pytest_asyncio", "_pytest", "tests",
    "IPython", "jupyter", "notebook",
    "matplotlib", "pandas", "scipy", "sklearn", "PIL", "cv2",
    "tensorflow", "torchaudio", "torchvision",
]

if IS_LEAN:
    # Lean build: exclude the heavy ML stack (PyTorch, sentence-transformers).
    # Keep numpy, onnxruntime, tokenizers — the lean build uses ONNX for embeddings.
    excludes += [
        "torch", "sentence_transformers", "transformers",
        "safetensors", "huggingface_hub",
    ]
else:
    # Full build: just strip CUDA (we're CPU-only)
    excludes += [
        "torch.cuda", "torch.distributed", "torch.backends.cudnn",
        "transformers.models.gpt2", "transformers.models.t5",
        "transformers.models.llama", "transformers.models.whisper",
    ]

# ─── Analysis ─────────────────────────────────────────────────────────
a = Analysis(
    [os.path.join(SPEC_DIR, "librarian_cli.py")],
    pathex=[SPEC_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

if not IS_LEAN:
    # ─── Strip CUDA binaries from torch ──────────────────────────────
    # Even with excludes above, torch ships .dll/.so files for CUDA.
    CUDA_PATTERNS = [
        "cublas", "cudart", "cudnn", "cufft", "curand", "cusolver",
        "cusparse", "nccl", "nvrtc", "nvToolsExt", "cupti",
        "cuda_runtime", "torch_cuda",
    ]
    a.binaries = [
        (name, path, typ)
        for name, path, typ in a.binaries
        if not any(pat.lower() in name.lower() for pat in CUDA_PATTERNS)
    ]

# ─── Build ────────────────────────────────────────────────────────────
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# UPX must be DISABLED on macOS — it breaks code signatures and notarization.
upx_enabled = not IS_MACOS

# Entitlements file for macOS hardened runtime
entitlements_path = os.path.join(SPEC_DIR, "entitlements.plist")
entitlements = entitlements_path if IS_MACOS and os.path.isfile(entitlements_path) else None

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="librarian",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=upx_enabled,
    console=True,  # CLI tool, not GUI
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=entitlements,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=upx_enabled,
    upx_exclude=[],
    name="librarian",
)

# ─── macOS App Bundle ────────────────────────────────────────────────
# On macOS, wrap the COLLECT output into a proper .app bundle.
# This gives users a double-clickable application with correct
# Gatekeeper integration and Finder presentation.
if IS_MACOS:
    app = BUNDLE(
        coll,
        name="The Librarian.app",
        icon=None,  # TODO: add .icns icon
        bundle_identifier="com.dictatech.librarian",
        info_plist={
            "NSPrincipalClass": "NSApplication",
            "CFBundleName": "The Librarian",
            "CFBundleDisplayName": "The Librarian",
            "CFBundleShortVersionString": "1.0.0",
            "NSHighResolutionCapable": True,
        },
    )
