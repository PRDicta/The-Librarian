"""
The Librarian — Embedding Manager
Generates vector embeddings for rolodex entries.

Supports four strategies:
- "anthropic": Voyage embeddings via voyageai SDK — production quality, API cost
- "local": Sentence-transformers (all-MiniLM-L6-v2) — real semantic embeddings, free, needs PyTorch
- "onnx": ONNX Runtime + all-MiniLM-L6-v2 — real semantic embeddings, free, lightweight (~25 MB model)
- "hash": Deterministic hash-based pseudo-embeddings — free, offline, keyword overlap only

Fallback chain: anthropic → local → onnx → hash
"""
import hashlib
import os
from typing import List, Optional
import numpy as np


class EmbeddingManager:
    """
    Generate and manage embeddings for text content.
    Strategy pattern allows swapping embedding backends.
    """

    def __init__(
        self,
        strategy: str = "local",
        dimensions: int = 384,
        api_key: Optional[str] = None,
        voyage_api_key: Optional[str] = None,
        cost_tracker=None,
    ):
        """
        Args:
            strategy: "local", "onnx", "anthropic", or "hash"
            dimensions: Embedding vector dimensions (384 for MiniLM, 1024 for Voyage-3)
            api_key: Anthropic API key (unused for embeddings, kept for compat)
            voyage_api_key: Voyage AI API key (required for "anthropic" strategy)
            cost_tracker: Optional CostTracker instance for recording API costs
        """
        self.strategy = strategy
        self.dimensions = dimensions
        self.api_key = api_key
        self.voyage_api_key = voyage_api_key
        self.cost_tracker = cost_tracker

        # Lazy-loaded backends
        self._local_model = None
        self._onnx_session = None
        self._onnx_tokenizer = None
        self._voyage_client = None

        # Initialize Voyage client if strategy is anthropic
        if strategy == "anthropic" and voyage_api_key:
            try:
                import voyageai
                self._voyage_client = voyageai.Client(api_key=voyage_api_key)
            except ImportError:
                print("Warning: voyageai not installed, falling back to local strategy")
                self.strategy = "local"
            except Exception as e:
                print(f"Warning: Voyage init failed ({e}), falling back to local strategy")
                self.strategy = "local"

        # Eagerly probe the actual available backend so self.strategy
        # reflects reality from the start (not after first embed call).
        if self.strategy == "local":
            try:
                from sentence_transformers import SentenceTransformer  # noqa: F401
            except ImportError:
                # sentence-transformers not available — try ONNX
                try:
                    import onnxruntime  # noqa: F401
                    if self._find_onnx_model() is not None:
                        self.strategy = "onnx"
                    else:
                        self.strategy = "hash"
                except ImportError:
                    self.strategy = "hash"

    def _get_local_model(self):
        """Lazy-load sentence-transformers model on first use.

        When running as a frozen PyInstaller bundle, the model is pre-bundled
        at lib/models/all-MiniLM-L6-v2 inside the install directory.
        sys._MEIPASS points to the PyInstaller extraction root.
        """
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                import sys
                import os

                model_name = "all-MiniLM-L6-v2"

                # Check for bundled model (PyInstaller frozen build)
                bundled_path = None
                if getattr(sys, '_MEIPASS', None):
                    candidate = os.path.join(sys._MEIPASS, "models", model_name)
                    if os.path.isdir(candidate):
                        bundled_path = candidate

                # Also check relative to the executable (Inno Setup install layout)
                if bundled_path is None and getattr(sys, 'frozen', False):
                    exe_dir = os.path.dirname(sys.executable)
                    candidate = os.path.join(exe_dir, "lib", "models", model_name)
                    if os.path.isdir(candidate):
                        bundled_path = candidate

                self._local_model = SentenceTransformer(
                    bundled_path if bundled_path else model_name
                )
            except ImportError:
                return None
            except Exception:
                return None
        return self._local_model

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text string.

        Fallback chain: anthropic → local → onnx → hash
        """
        if self.strategy == "anthropic" and self._voyage_client:
            return await self._embed_voyage(text)
        elif self.strategy == "local":
            result = self._embed_local(text)
            if result is not None:
                return result
            # Local failed — try ONNX before falling back to hash
            result = self._embed_onnx(text)
            if result is not None:
                self.strategy = "onnx"  # Record the active strategy
                return result
            self.strategy = "hash"
            return self._embed_hash_semantic(text)
        elif self.strategy == "onnx":
            result = self._embed_onnx(text)
            if result is not None:
                return result
            self.strategy = "hash"
            return self._embed_hash_semantic(text)
        else:
            return self._embed_hash_semantic(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if self.strategy == "anthropic" and self._voyage_client:
            return await self._embed_voyage_batch(texts)
        elif self.strategy == "local":
            result = self._embed_local_batch(texts)
            if result is not None:
                return result
            result = self._embed_onnx_batch(texts)
            if result is not None:
                self.strategy = "onnx"
                return result
            self.strategy = "hash"
            return [self._embed_hash_semantic(t) for t in texts]
        elif self.strategy == "onnx":
            result = self._embed_onnx_batch(texts)
            if result is not None:
                return result
            self.strategy = "hash"
            return [self._embed_hash_semantic(t) for t in texts]
        else:
            return [self._embed_hash_semantic(t) for t in texts]

    def similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    # ─── Local Sentence-Transformers Strategy ────────────────────────────────

    def _embed_local(self, text: str) -> Optional[List[float]]:
        """Single text embedding via sentence-transformers (all-MiniLM-L6-v2)."""
        model = self._get_local_model()
        if model is None:
            return None
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def _embed_local_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Batch embedding via sentence-transformers."""
        model = self._get_local_model()
        if model is None:
            return None
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    # ─── ONNX Runtime Strategy ───────────────────────────────────────────────

    def _find_onnx_model(self) -> Optional[str]:
        """Locate the ONNX model file.

        Search order:
        1. Next to this script: models/all-MiniLM-L6-v2/model.onnx
        2. In the librarian workspace: librarian/models/all-MiniLM-L6-v2/model.onnx
        3. PyInstaller bundle: _MEIPASS/models/all-MiniLM-L6-v2/model.onnx
        """
        import sys
        model_name = "all-MiniLM-L6-v2"

        candidates = []

        # Relative to this file's package (src/indexing/)
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        script_dir = os.path.dirname(os.path.dirname(pkg_dir))  # up to librarian/
        candidates.append(os.path.join(script_dir, "models", model_name, "model.onnx"))

        # PyInstaller frozen
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            candidates.append(os.path.join(meipass, "models", model_name, "model.onnx"))

        # Frozen exe layout
        if getattr(sys, 'frozen', False):
            exe_dir = os.path.dirname(sys.executable)
            candidates.append(os.path.join(exe_dir, "lib", "models", model_name, "model.onnx"))

        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    def _get_onnx_session(self):
        """Lazy-load ONNX Runtime session and tokenizer on first use."""
        if self._onnx_session is not None:
            return self._onnx_session, self._onnx_tokenizer

        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            model_path = self._find_onnx_model()
            if model_path is None:
                return None, None

            model_dir = os.path.dirname(model_path)
            tokenizer_path = os.path.join(model_dir, "tokenizer.json")
            if not os.path.isfile(tokenizer_path):
                return None, None

            self._onnx_session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"]
            )
            self._onnx_tokenizer = Tokenizer.from_file(tokenizer_path)
            # Set truncation and padding for the tokenizer
            self._onnx_tokenizer.enable_truncation(max_length=128)
            self._onnx_tokenizer.enable_padding(length=128)

            return self._onnx_session, self._onnx_tokenizer
        except ImportError:
            return None, None
        except Exception:
            return None, None

    def _embed_onnx(self, text: str) -> Optional[List[float]]:
        """Single text embedding via ONNX Runtime."""
        session, tokenizer = self._get_onnx_session()
        if session is None:
            return None
        try:
            encoded = tokenizer.encode(text)
            input_ids = np.array([encoded.ids], dtype=np.int64)
            attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
            token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

            outputs = session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                }
            )
            # Mean pooling over token embeddings (output[0] is last_hidden_state)
            token_embeddings = outputs[0]  # shape: (1, seq_len, 384)
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
            summed = np.sum(token_embeddings * mask_expanded, axis=1)
            counts = np.sum(mask_expanded, axis=1)
            counts = np.maximum(counts, 1e-9)
            mean_pooled = summed / counts
            # Normalize
            norm = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
            norm = np.maximum(norm, 1e-9)
            normalized = mean_pooled / norm
            return normalized[0].tolist()
        except Exception:
            return None

    def _embed_onnx_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Batch embedding via ONNX Runtime."""
        session, tokenizer = self._get_onnx_session()
        if session is None:
            return None
        try:
            results = []
            # Process in small batches to manage memory
            for text in texts:
                result = self._embed_onnx(text)
                if result is None:
                    return None
                results.append(result)
            return results
        except Exception:
            return None

    # ─── Voyage API Strategy ─────────────────────────────────────────────────

    async def _embed_voyage(self, text: str) -> List[float]:
        """Single text embedding via Voyage API."""
        try:
            response = self._voyage_client.embed(
                [text],
                model="voyage-3",
                input_type="document",
            )
            # Track cost
            if self.cost_tracker and hasattr(response, "total_tokens"):
                self.cost_tracker.record(
                    call_type="embedding",
                    model="voyage-3",
                    input_tokens=response.total_tokens,
                )
            return response.embeddings[0]
        except Exception as e:
            print(f"Warning: Voyage embedding failed ({e}), falling back to local")
            result = self._embed_local(text)
            return result if result is not None else self._embed_hash_semantic(text)

    async def _embed_voyage_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding via Voyage API."""
        try:
            response = self._voyage_client.embed(
                texts,
                model="voyage-3",
                input_type="document",
            )
            if self.cost_tracker and hasattr(response, "total_tokens"):
                self.cost_tracker.record(
                    call_type="embedding",
                    model="voyage-3",
                    input_tokens=response.total_tokens,
                )
            return response.embeddings
        except Exception as e:
            print(f"Warning: Voyage batch embedding failed ({e}), falling back to local")
            result = self._embed_local_batch(texts)
            return result if result is not None else [self._embed_hash_semantic(t) for t in texts]

    # ─── Hash-Based Fallback Strategy ────────────────────────────────────────

    def _embed_hash(self, text: str) -> List[float]:
        """
        Deterministic pseudo-embedding using SHA-256 hash.
        Not semantically meaningful, but:
        - Free (no API calls)
        - Works offline
        - Deterministic (same text → same embedding)
        - Good enough for testing the pipeline
        """
        text = text.strip().lower()
        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        np.random.seed(int.from_bytes(hash_bytes[:4], "big"))
        vec = np.random.randn(self.dimensions).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def _embed_hash_semantic(self, text: str) -> List[float]:
        """
        Slightly smarter hash embedding that preserves some word-level signal.
        Combines word-level hashes so texts with shared words have some similarity.
        """
        words = text.strip().lower().split()
        if not words:
            return [0.0] * self.dimensions
        word_vecs = []
        for word in words:
            hash_bytes = hashlib.sha256(word.encode("utf-8")).digest()
            np.random.seed(int.from_bytes(hash_bytes[:4], "big"))
            vec = np.random.randn(self.dimensions).astype(np.float32)
            word_vecs.append(vec)
        avg = np.mean(word_vecs, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        return avg.tolist()
