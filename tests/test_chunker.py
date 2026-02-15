"""
Tests for ContentChunker (Phase 6b)

Validates modality detection, scoring functions, chunking behavior,
and edge cases. Pure functions, no mocking needed.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.types import ContentModality
from src.indexing.chunker import ContentChunker


# ─── Modality Detection ─────────────────────────────────────────────────────

def test_detect_code():
    """Python code with imports and functions → CODE modality."""
    chunker = ContentChunker()
    text = """```python
import os
from pathlib import Path

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def factorial(self, n):
        if n == 0:
            return 1
        return n * self.factorial(n-1)
```"""
    result = chunker.detect_modality(text)
    assert result == ContentModality.CODE, f"Expected CODE, got {result}"
    print("  PASS: detect_code")


def test_detect_math():
    """LaTeX equations and theorem markers → MATH modality."""
    chunker = ContentChunker()
    text = r"""Theorem 1.3: For any $n \geq 1$, we have
$$\sum_{k=1}^{n} k = \frac{n(n+1)}{2}$$
Proof: By induction on $n$. QED"""
    result = chunker.detect_modality(text)
    assert result == ContentModality.MATH, f"Expected MATH, got {result}"
    print("  PASS: detect_math")


def test_detect_prose():
    """Multi-sentence paragraphs with no code/math → PROSE modality."""
    chunker = ContentChunker()
    text = """The history of artificial intelligence began in the 1950s when
researchers first explored the possibility of creating thinking machines.
Alan Turing proposed his famous test in 1950, suggesting that a machine
could be considered intelligent if it could fool a human evaluator.

Over the following decades, the field experienced several waves of optimism
and disappointment, often referred to as AI winters. These cycles were
driven by the gap between ambitious promises and practical achievements."""
    result = chunker.detect_modality(text)
    assert result == ContentModality.PROSE, f"Expected PROSE, got {result}"
    print("  PASS: detect_prose")


def test_detect_structured():
    """Markdown table + bullet list → STRUCTURED modality."""
    chunker = ContentChunker()
    text = """| Language | Typing | Speed |
| --- | --- | --- |
| Python | Dynamic | Moderate |
| Rust | Static | Fast |
| JavaScript | Dynamic | Fast |

Key takeaways:
- Python is best for prototyping
- Rust is best for performance
- JavaScript is best for web

1. Choose based on requirements
2. Consider team expertise
3. Evaluate ecosystem support"""
    result = chunker.detect_modality(text)
    assert result == ContentModality.STRUCTURED, f"Expected STRUCTURED, got {result}"
    print("  PASS: detect_structured")


def test_detect_conversational():
    """Short question → CONVERSATIONAL modality."""
    chunker = ContentChunker()
    text = "What do you think about the new design?"
    result = chunker.detect_modality(text)
    assert result == ContentModality.CONVERSATIONAL, f"Expected CONVERSATIONAL, got {result}"
    print("  PASS: detect_conversational")


# ─── Scoring Functions ───────────────────────────────────────────────────────

def test_score_code_indicators():
    """Code fences, def/class, imports should score high for code."""
    chunker = ContentChunker()
    code_text = """```python
import numpy as np
def process(data):
    for item in data:
        yield item * 2
```"""
    plain_text = "The weather today is sunny and warm with a light breeze."
    code_score = chunker._score_code(code_text)
    plain_score = chunker._score_code(plain_text)
    assert code_score > 0.5, f"Code text should score high: {code_score}"
    assert plain_score < 0.2, f"Plain text should score low: {plain_score}"
    print("  PASS: score_code_indicators")


def test_score_math_indicators():
    """LaTeX, math symbols, theorem keywords should score high for math."""
    chunker = ContentChunker()
    math_text = r"""Theorem: Let $f(x) = \sum_{n=0}^{\infty} a_n x^n$.
Then $\int f(x) dx = \sum_{n=0}^{\infty} \frac{a_n}{n+1} x^{n+1} + C$."""
    plain_text = "I went to the store and bought some groceries for dinner."
    math_score = chunker._score_math(math_text)
    plain_score = chunker._score_math(plain_text)
    assert math_score > 0.5, f"Math text should score high: {math_score}"
    assert plain_score < 0.15, f"Plain text should score low: {plain_score}"
    print("  PASS: score_math_indicators")


def test_score_conversational_indicators():
    """Questions and casual words should score high for conversational."""
    chunker = ContentChunker()
    chat_text = "Yes, I think that makes sense. Do you agree?"
    formal_text = """The empirical evidence suggests a strong correlation between
the two variables. Multiple regression analysis confirms statistical
significance at the p < 0.05 level across all measured parameters."""
    chat_score = chunker._score_conversational(chat_text)
    formal_score = chunker._score_conversational(formal_text)
    assert chat_score > 0.3, f"Chat text should score high: {chat_score}"
    assert chat_score > formal_score, f"Chat should outscore formal: {chat_score} vs {formal_score}"
    print("  PASS: score_conversational_indicators")


# ─── Chunking Behavior ──────────────────────────────────────────────────────

def test_chunk_code_splits_on_functions():
    """Code with multiple functions should split at function boundaries."""
    chunker = ContentChunker()
    code = """def hello():
    print("hello")

def world():
    print("world")

def goodbye():
    print("bye")"""
    chunks = chunker._chunk_code(code)
    assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"
    # Each chunk should contain at least one function
    has_hello = any("hello" in c for c in chunks)
    has_world = any("world" in c for c in chunks)
    assert has_hello and has_world, "Chunks should contain the functions"
    print("  PASS: chunk_code_splits_on_functions")


def test_chunk_prose_respects_paragraphs():
    """Prose should split at paragraph boundaries, respecting token target."""
    chunker = ContentChunker()
    # Create text with 3 clear paragraphs, each ~200 tokens (800 chars)
    para = "This is a test paragraph with enough words to matter. " * 15
    text = f"{para}\n\n{para}\n\n{para}"
    chunks = chunker._chunk_prose(text, target_tokens=300)
    # With 3 large paragraphs and a 300-token target, should get multiple chunks
    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
    # No chunk should contain a bare double-newline split (paragraphs merged properly)
    for chunk in chunks:
        assert chunk.strip(), "No empty chunks"
    print("  PASS: chunk_prose_respects_paragraphs")


def test_chunk_conversation_turn_mixed():
    """Prose followed by code fence → two chunks with different modalities."""
    chunker = ContentChunker()
    text = """Here is an explanation of the algorithm. It works by recursively
dividing the problem into smaller subproblems until a base case is reached.
The time complexity is O(n log n) which makes it efficient for large inputs.

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
```"""
    chunks = chunker.chunk_conversation_turn(text)
    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
    modalities = [c["modality"] for c in chunks]
    assert ContentModality.CODE in modalities, f"Should have CODE chunk: {modalities}"
    # The non-code part should be prose or conversational (not code)
    non_code = [m for m in modalities if m != ContentModality.CODE]
    assert len(non_code) >= 1, "Should have at least one non-code chunk"
    print("  PASS: chunk_conversation_turn_mixed")


def test_split_by_modality_detects_fences():
    """Code fences should be extracted as CODE segments."""
    chunker = ContentChunker()
    text = """Some intro text here.

```javascript
const x = 42;
console.log(x);
```

And some more text after."""
    segments = chunker._split_by_modality(text)
    assert len(segments) >= 2, f"Expected multiple segments, got {len(segments)}"
    # Find the code segment
    code_segments = [s for s in segments if s[1] == ContentModality.CODE]
    assert len(code_segments) >= 1, "Should have at least one CODE segment"
    assert "const x = 42" in code_segments[0][0], "Code content should be preserved"
    print("  PASS: split_by_modality_detects_fences")


# ─── Edge Cases ──────────────────────────────────────────────────────────────

def test_empty_and_whitespace():
    """Empty or whitespace-only input should return empty or single chunk."""
    chunker = ContentChunker()
    # Empty string
    result = chunker.chunk("")
    # Should either return empty list or single chunk with whitespace
    non_empty = [c for c in result if c["text"].strip()]
    assert len(non_empty) == 0, f"Empty input should yield no meaningful chunks: {result}"
    # Whitespace only
    result2 = chunker.chunk("   \n\n   ")
    non_empty2 = [c for c in result2 if c["text"].strip()]
    assert len(non_empty2) == 0, f"Whitespace should yield no meaningful chunks: {result2}"
    print("  PASS: empty_and_whitespace")


def test_short_text_stays_whole():
    """Text shorter than chunk threshold should not be split."""
    chunker = ContentChunker()
    text = "Quick note: meeting moved to 3pm."
    chunks = chunker.chunk_conversation_turn(text)
    assert len(chunks) == 1, f"Short text should stay as one chunk, got {len(chunks)}"
    assert chunks[0]["text"].strip() == text.strip()
    print("  PASS: short_text_stays_whole")


# ─── Runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # Modality detection
        ("detect_code", test_detect_code),
        ("detect_math", test_detect_math),
        ("detect_prose", test_detect_prose),
        ("detect_structured", test_detect_structured),
        ("detect_conversational", test_detect_conversational),
        # Scoring
        ("score_code_indicators", test_score_code_indicators),
        ("score_math_indicators", test_score_math_indicators),
        ("score_conversational_indicators", test_score_conversational_indicators),
        # Chunking behavior
        ("chunk_code_splits_on_functions", test_chunk_code_splits_on_functions),
        ("chunk_prose_respects_paragraphs", test_chunk_prose_respects_paragraphs),
        ("chunk_conversation_turn_mixed", test_chunk_conversation_turn_mixed),
        ("split_by_modality_detects_fences", test_split_by_modality_detects_fences),
        # Edge cases
        ("empty_and_whitespace", test_empty_and_whitespace),
        ("short_text_stays_whole", test_short_text_stays_whole),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name} — {e}")
            failed += 1

    print(f"\nChunker tests: {passed} passed, {failed} failed, {passed + failed} total")
    if failed > 0:
        sys.exit(1)
