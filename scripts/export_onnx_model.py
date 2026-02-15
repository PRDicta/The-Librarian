#!/usr/bin/env python3
"""
Export all-MiniLM-L6-v2 to ONNX format for lightweight deployment.

The ONNX model (~25 MB) replaces the need for PyTorch (~800 MB) and
sentence-transformers (~150 MB) at runtime. ONNX Runtime (already
pre-installed on Cowork VMs at ~49 MB) handles inference.

Usage:
    python scripts/export_onnx_model.py

Output:
    lib/models/all-MiniLM-L6-v2/model.onnx      (~25 MB)
    lib/models/all-MiniLM-L6-v2/tokenizer.json   (already present)

Requirements (for export only — not needed at runtime):
    - torch
    - sentence-transformers
    - onnx
"""
import os
import sys
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "lib", "models", "all-MiniLM-L6-v2")
MODEL_NAME = "all-MiniLM-L6-v2"


def export():
    print(f"Exporting {MODEL_NAME} to ONNX format...")

    # Ensure model directory exists and has the base model.
    # On CI runners or fresh machines, download it automatically.
    if not os.path.isdir(MODEL_DIR):
        print(f"  Model not found locally — downloading {MODEL_NAME} from HuggingFace...")
        try:
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer(MODEL_NAME)
            os.makedirs(MODEL_DIR, exist_ok=True)
            st_model.save(MODEL_DIR)
            print(f"  Model downloaded and saved to {MODEL_DIR}")
        except ImportError:
            print(f"Model directory not found: {MODEL_DIR}")
            print("Install sentence-transformers to auto-download, or run build.py first:")
            print(f"  pip install sentence-transformers && python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{MODEL_NAME}')\"")
            sys.exit(1)

    import torch
    from transformers import AutoModel, AutoTokenizer

    # Load the model
    print("  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR)
    model.eval()

    # Create dummy input
    dummy_text = "This is a test sentence for ONNX export."
    inputs = tokenizer(dummy_text, return_tensors="pt", padding="max_length",
                       truncation=True, max_length=128)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs.get("token_type_ids", torch.zeros_like(input_ids))

    # Export to ONNX
    onnx_path = os.path.join(MODEL_DIR, "model.onnx")
    print(f"  Exporting to {onnx_path}...")

    # Export to a temp path first, then ensure it's self-contained
    onnx_tmp = onnx_path + ".tmp"
    torch.onnx.export(
        model,
        (input_ids, attention_mask, token_type_ids),
        onnx_tmp,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # CRITICAL: Force self-contained ONNX file (no external .data file).
    # Newer PyTorch versions export large models with external data by default,
    # producing a small stub + a separate model.onnx.data file. We need
    # everything in one file for clean bundling.
    import onnx
    print("  Converting to self-contained format (no external data)...")
    onnx_model = onnx.load(onnx_tmp)
    onnx.save_model(onnx_model, onnx_path, save_as_external_data=False)

    # Clean up temp files
    if os.path.exists(onnx_tmp):
        os.remove(onnx_tmp)
    # Remove any .data files from external-data export
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".data") or f == "model.onnx.tmp":
            os.remove(os.path.join(MODEL_DIR, f))

    # Verify the file is self-contained (should be ~23 MB, not ~0.7 MB)
    onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
    if onnx_size < 5:
        print(f"  WARNING: model.onnx is only {onnx_size:.1f} MB — likely still a stub!")
        print("  Expected ~23 MB for a self-contained all-MiniLM-L6-v2 model.")
        sys.exit(1)
    print(f"  ONNX model: {onnx_size:.1f} MB (self-contained)")

    # Verify tokenizer.json exists (needed by the Rust tokenizers library at runtime)
    tokenizer_json = os.path.join(MODEL_DIR, "tokenizer.json")
    if not os.path.isfile(tokenizer_json):
        print("  Saving tokenizer.json...")
        tokenizer.save_pretrained(MODEL_DIR)

    # Report sizes
    onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  ONNX model: {onnx_size:.1f} MB")

    total_size = sum(
        os.path.getsize(os.path.join(MODEL_DIR, f))
        for f in os.listdir(MODEL_DIR)
        if os.path.isfile(os.path.join(MODEL_DIR, f))
    ) / 1024 / 1024
    print(f"  Total model dir: {total_size:.1f} MB")

    # Verify the export works
    print("  Verifying ONNX model...")
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        import numpy as np
        outputs = session.run(None, {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
            "token_type_ids": token_type_ids.numpy(),
        })
        print(f"  Verification OK — output shape: {outputs[0].shape}")
    except ImportError:
        print("  Skipping verification (onnxruntime not installed)")
    except Exception as e:
        print(f"  WARNING: Verification failed: {e}")

    print(f"\nDone. ONNX model ready at: {onnx_path}")
    print("This file should be bundled with the installer for Cowork VM deployment.")


if __name__ == "__main__":
    export()
