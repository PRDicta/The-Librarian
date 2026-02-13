


import hashlib
from typing import List, Optional
import numpy as np


class EmbeddingManager:


    def __init__(
        self,
        strategy: str = "local",
        dimensions: int = 384,
        api_key: Optional[str] = None,
        voyage_api_key: Optional[str] = None,
        cost_tracker=None,
    ):


        self.strategy = strategy
        self.dimensions = dimensions
        self.api_key = api_key
        self.voyage_api_key = voyage_api_key
        self.cost_tracker = cost_tracker


        self._local_model = None
        self._voyage_client = None


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

    def _get_local_model(self):

        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._local_model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                return None
            except Exception:
                return None
        return self._local_model

    async def embed_text(self, text: str) -> List[float]:

        if self.strategy == "anthropic" and self._voyage_client:
            return await self._embed_voyage(text)
        elif self.strategy == "local":
            result = self._embed_local(text)
            if result is not None:
                return result


            return self._embed_hash_semantic(text)
        else:
            return self._embed_hash_semantic(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:

        if self.strategy == "anthropic" and self._voyage_client:
            return await self._embed_voyage_batch(texts)
        elif self.strategy == "local":
            result = self._embed_local_batch(texts)
            if result is not None:
                return result
            return [self._embed_hash_semantic(t) for t in texts]
        else:
            return [self._embed_hash_semantic(t) for t in texts]

    def similarity(self, a: List[float], b: List[float]) -> float:

        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))


    def _embed_local(self, text: str) -> Optional[List[float]]:

        model = self._get_local_model()
        if model is None:
            return None
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def _embed_local_batch(self, texts: List[str]) -> Optional[List[List[float]]]:

        model = self._get_local_model()
        if model is None:
            return None
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]


    async def _embed_voyage(self, text: str) -> List[float]:

        try:
            response = self._voyage_client.embed(
                [text],
                model="voyage-3",
                input_type="document",
            )

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


    def _embed_hash(self, text: str) -> List[float]:


        text = text.strip().lower()
        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        np.random.seed(int.from_bytes(hash_bytes[:4], "big"))
        vec = np.random.randn(self.dimensions).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def _embed_hash_semantic(self, text: str) -> List[float]:


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
