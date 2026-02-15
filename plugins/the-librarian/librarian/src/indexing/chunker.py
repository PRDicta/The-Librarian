"""
The Librarian — Content Chunker
Detects content modality (prose, code, math, conversational)
and chunks accordingly. Context switches between modalities
are natural breakpoints for rolodex sections.
"""
import re
from typing import List, Tuple
from ..core.types import ContentModality
class ContentChunker:
    """
    Detects content modality and splits text into semantically
    coherent chunks appropriate for that modality.
    """
    # ─── Modality Detection ──────────────────────────────────────────────
    def detect_modality(self, text: str) -> ContentModality:
        """
        Detect the primary content modality of a text block.
        Uses heuristics — fast, no API calls needed.
        """
        scores = {
            ContentModality.CODE: self._score_code(text),
            ContentModality.MATH: self._score_math(text),
            ContentModality.STRUCTURED: self._score_structured(text),
            ContentModality.PROSE: self._score_prose(text),
            ContentModality.CONVERSATIONAL: self._score_conversational(text),
        }
        best = max(scores, key=scores.get)
        best_score = scores[best]
        # If no strong signal, default to conversational
        if best_score < 0.3:
            return ContentModality.CONVERSATIONAL
        return best
    def _score_code(self, text: str) -> float:
        """Score likelihood of code content."""
        indicators = [
            (r"```", 0.4),                            # Code fences
            (r"def\s+\w+\s*\(", 0.3),                 # Python functions
            (r"function\s+\w+", 0.3),                  # JS functions
            (r"class\s+\w+", 0.25),                    # Class definitions
            (r"(import|from)\s+\w+", 0.2),             # Imports
            (r"(if|else|for|while|return)\s", 0.15),   # Control flow
            (r"[{}\[\];]", 0.1),                       # Brackets/semicolons
            (r"(=>|->|::|\.\\.)", 0.1),                # Operators
            (r"^\s{4,}", 0.15),                        # Indentation
        ]
        return self._compute_score(text, indicators)
    def _score_math(self, text: str) -> float:
        """Score likelihood of mathematical content."""
        indicators = [
            (r"\\(frac|sum|int|prod|lim)", 0.4),       # LaTeX commands
            (r"\$\$?[^$]+\$\$?", 0.35),                # LaTeX delimiters
            (r"(theorem|proof|lemma|corollary)", 0.3),  # Math terms
            (r"[∑∫∏∂∇√∞±≤≥≠≈∈∉⊂⊃∪∩]", 0.3),         # Math symbols
            (r"(Q\.E\.D\.|QED|□)", 0.25),               # Proof endings
            (r"\b(iff|implies|therefore)\b", 0.2),      # Logic terms
            (r"[=<>]{1,2}", 0.05),                      # Comparisons (weak signal)
        ]
        return self._compute_score(text, indicators)
    def _score_structured(self, text: str) -> float:
        """Score likelihood of structured/tabular content."""
        indicators = [
            (r"\|.*\|.*\|", 0.4),                      # Markdown tables
            (r"^\s*[-*]\s+", 0.2),                     # Bullet lists
            (r"^\s*\d+\.\s+", 0.2),                    # Numbered lists
            (r":\s*\n\s+", 0.15),                      # Key-value blocks
        ]
        return self._compute_score(text, indicators)
    def _score_prose(self, text: str) -> float:
        """Score likelihood of prose/narrative content."""
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_len = (
            sum(len(s.split()) for s in sentences if s.strip()) /
            max(1, len([s for s in sentences if s.strip()]))
        )
        # Prose tends to have longer sentences
        score = min(1.0, avg_sentence_len / 20.0) * 0.4
        # Paragraph structure
        paragraphs = text.split("\n\n")
        if len(paragraphs) >= 2:
            score += 0.2
        # Low code/math signal boosts prose score
        if self._score_code(text) < 0.15 and self._score_math(text) < 0.15:
            score += 0.2
        return min(1.0, score)
    def _score_conversational(self, text: str) -> float:
        """Score likelihood of conversational content."""
        indicators = [
            (r"\?$", 0.2),                                    # Questions
            (r"^(yes|no|sure|okay|thanks|please)\b", 0.2),    # Casual words
            (r"(I think|I believe|in my opinion)", 0.15),      # Opinion markers
        ]
        score = self._compute_score(text, indicators)
        # Short text is more likely conversational
        if len(text.split()) < 50:
            score += 0.2
        return min(1.0, score)
    def _compute_score(self, text: str, indicators: List[Tuple[str, float]]) -> float:
        """Sum indicator scores found in text, capped at 1.0."""
        score = 0.0
        for pattern, weight in indicators:
            if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                score += weight
        return min(1.0, score)
    # ─── Chunking ────────────────────────────────────────────────────────
    def chunk(self, text: str, modality: ContentModality = None) -> List[dict]:
        """
        Split text into chunks appropriate for the detected modality.
        Returns list of {"text": str, "modality": ContentModality} dicts.
        """
        if modality is None:
            modality = self.detect_modality(text)
        if modality == ContentModality.CODE:
            chunks = self._chunk_code(text)
        elif modality == ContentModality.MATH:
            chunks = self._chunk_math(text)
        elif modality == ContentModality.STRUCTURED:
            chunks = self._chunk_structured(text)
        elif modality == ContentModality.PROSE:
            chunks = self._chunk_prose(text)
        else:
            chunks = self._chunk_conversational(text)
        return [{"text": c, "modality": modality} for c in chunks if c.strip()]
    def chunk_conversation_turn(self, text: str) -> List[dict]:
        """
        Chunk a single conversation turn, detecting modality shifts.
        A turn might contain mixed content (e.g., prose then code).
        Returns chunks with their detected modalities.
        """
        segments = self._split_by_modality(text)
        all_chunks = []
        for segment_text, modality in segments:
            chunks = self.chunk(segment_text, modality)
            all_chunks.extend(chunks)
        return all_chunks
    def _split_by_modality(self, text: str) -> List[Tuple[str, ContentModality]]:
        """
        Split text at modality boundaries.
        E.g., prose followed by a code block = two segments.
        """
        segments = []
        # Split on code fences first (most clear boundary)
        parts = re.split(r"(```[\s\S]*?```)", text)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.startswith("```"):
                segments.append((part, ContentModality.CODE))
            else:
                modality = self.detect_modality(part)
                segments.append((part, modality))
        return segments if segments else [(text, self.detect_modality(text))]
    def _chunk_code(self, text: str, max_lines: int = 50) -> List[str]:
        """
        Split code by function/class boundaries.
        Falls back to line-count-based splitting.
        """
        # Try to split on function/class definitions
        pattern = r"(?=^(?:def |class |function |const |let |var |pub fn |fn ))"
        blocks = re.split(pattern, text, flags=re.MULTILINE)
        blocks = [b.strip() for b in blocks if b.strip()]
        # If no natural boundaries found, split by line count
        if len(blocks) <= 1:
            lines = text.split("\n")
            blocks = []
            for i in range(0, len(lines), max_lines):
                chunk = "\n".join(lines[i:i + max_lines])
                if chunk.strip():
                    blocks.append(chunk)
        return blocks
    def _chunk_prose(self, text: str, target_tokens: int = 500) -> List[str]:
        """
        Split prose on paragraph boundaries, merging small paragraphs.
        Target ~500 tokens per chunk (roughly half a page).
        """
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current = []
        current_tokens = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_tokens = len(para) // 4  # Rough estimate
            if current_tokens + para_tokens > target_tokens and current:
                chunks.append("\n\n".join(current))
                current = [para]
                current_tokens = para_tokens
            else:
                current.append(para)
                current_tokens += para_tokens
        if current:
            chunks.append("\n\n".join(current))
        return chunks
    def _chunk_math(self, text: str) -> List[str]:
        """
        Split math by theorem/proof/equation boundaries.
        Preserves logical structure.
        """
        # Split on theorem-like markers
        pattern = r"(?=^(?:Theorem|Lemma|Proof|Corollary|Proposition|Definition|Example)\b)"
        blocks = re.split(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
        blocks = [b.strip() for b in blocks if b.strip()]
        if len(blocks) <= 1:
            # Fall back to double-newline splitting
            return self._chunk_prose(text, target_tokens=300)
        return blocks
    def _chunk_structured(self, text: str) -> List[str]:
        """
        Split structured content, preserving tables and lists as units.
        """
        # Keep tables together
        parts = re.split(r"(\n\s*\n)", text)
        chunks = []
        current = []
        for part in parts:
            if re.match(r"\n\s*\n", part):
                if current:
                    chunk = "".join(current).strip()
                    if chunk:
                        chunks.append(chunk)
                    current = []
            else:
                current.append(part)
        if current:
            chunk = "".join(current).strip()
            if chunk:
                chunks.append(chunk)
        return chunks if chunks else [text]
    def _chunk_conversational(self, text: str) -> List[str]:
        """
        Split conversational content by topic.
        For simple cases, keep each exchange as one chunk.
        """
        # If short enough, keep as one chunk
        if len(text) // 4 < 300:
            return [text]
        # Otherwise split on paragraph/topic boundaries
        return self._chunk_prose(text, target_tokens=300)
