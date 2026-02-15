"""
Tests for VerbatimExtractor and GapDetector (Phase 4.5)

Validates heuristic extraction and gap detection work without any LLM.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.types import ContentModality
from src.indexing.verbatim_extractor import VerbatimExtractor
from src.core.gap_detector import detect_gap, extract_gap_topic


# ─── VerbatimExtractor Tests ──────────────────────────────────────────────

def test_extract_code():
    """CODE modality → implementation category with code identifiers as tags."""
    extractor = VerbatimExtractor()
    chunk = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)"""
    results = asyncio.run(extractor.extract(chunk, ContentModality.CODE))
    assert len(results) == 1
    assert results[0]["category"] == "implementation"
    assert "fibonacci" in results[0]["tags"]
    assert results[0]["content"] == chunk.strip()
    print("  PASS: extract_code")


def test_extract_prose():
    """PROSE modality → note category."""
    extractor = VerbatimExtractor()
    chunk = "Machine learning models require large datasets for training. The quality of the data directly impacts the accuracy of predictions made by the model."
    results = asyncio.run(extractor.extract(chunk, ContentModality.PROSE))
    assert len(results) == 1
    assert results[0]["category"] == "note"
    assert "prose" in results[0]["tags"]
    print("  PASS: extract_prose")


def test_extract_decision():
    """Decision signals override modality-based category."""
    extractor = VerbatimExtractor()
    chunk = "We decided to use PostgreSQL instead of MySQL for the database backend."
    results = asyncio.run(extractor.extract(chunk, ContentModality.CONVERSATIONAL))
    assert len(results) == 1
    assert results[0]["category"] == "decision"
    print("  PASS: extract_decision")


def test_extract_preference():
    """Preference signals override modality-based category."""
    extractor = VerbatimExtractor()
    chunk = "I prefer using Python over Java for data analysis tasks because of the ecosystem."
    results = asyncio.run(extractor.extract(chunk, ContentModality.CONVERSATIONAL))
    assert len(results) == 1
    assert results[0]["category"] == "preference"
    print("  PASS: extract_preference")


def test_extract_warning():
    """Warning signals override modality-based category."""
    extractor = VerbatimExtractor()
    chunk = "Be careful with mutable default arguments in Python functions, it is a common pitfall."
    results = asyncio.run(extractor.extract(chunk, ContentModality.PROSE))
    assert len(results) == 1
    assert results[0]["category"] == "warning"
    print("  PASS: extract_warning")


def test_extract_short_text_skipped():
    """Text shorter than 10 chars is skipped."""
    extractor = VerbatimExtractor()
    results = asyncio.run(extractor.extract("hi", ContentModality.CONVERSATIONAL))
    assert len(results) == 0
    print("  PASS: extract_short_text_skipped")


def test_tag_extraction_code_identifiers():
    """Code identifiers are extracted as tags."""
    extractor = VerbatimExtractor()
    chunk = """class DatabaseManager:
    def connect(self, host):
        pass

    def execute_query(self, sql):
        pass"""
    results = asyncio.run(extractor.extract(chunk, ContentModality.CODE))
    tags = results[0]["tags"]
    assert "DatabaseManager" in tags or "databasemanager" in [t.lower() for t in tags]
    assert any("connect" in t for t in tags)
    print("  PASS: tag_extraction_code_identifiers")


def test_tag_extraction_technical_terms():
    """CamelCase and snake_case terms are extracted."""
    extractor = VerbatimExtractor()
    chunk = "The ContentChunker uses detect_modality to classify content types in a modular way."
    results = asyncio.run(extractor.extract(chunk, ContentModality.PROSE))
    tags = results[0]["tags"]
    tag_lower = [t.lower() for t in tags]
    assert "contentchunker" in tag_lower or "detect_modality" in tag_lower
    print("  PASS: tag_extraction_technical_terms")


# ─── GapDetector Tests ────────────────────────────────────────────────────

def test_detect_gap_basic():
    """Detect common gap signal patterns."""
    assert detect_gap("I don't have access to that configuration file") is not None
    assert detect_gap("Let me look that up in my notes") is not None
    assert detect_gap("I'm not sure about the specifics of that") is not None
    assert detect_gap("We discussed this earlier in the conversation") is not None
    print("  PASS: detect_gap_basic")


def test_no_false_positives():
    """Normal responses should not trigger gap detection."""
    assert detect_gap("Here is the Python code you requested.") is None
    assert detect_gap("The answer is 42.") is None
    assert detect_gap("I can help you with that task.") is None
    print("  PASS: no_false_positives")


def test_extract_gap_topic():
    """Extract the actual topic from a gap signal."""
    topic = extract_gap_topic(
        "I don't have access to the database configuration details you shared."
    )
    assert topic is not None
    assert len(topic) > 5
    # The topic should be about the actual subject, not the gap signal
    print("  PASS: extract_gap_topic")


def test_extract_gap_topic_none():
    """No topic when no gap detected."""
    topic = extract_gap_topic("Everything is working perfectly fine.")
    assert topic is None
    print("  PASS: extract_gap_topic_none")


# ─── Run all ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Verbatim + Gap Detector tests (Phase 4.5)...\n")

    # VerbatimExtractor
    test_extract_code()
    test_extract_prose()
    test_extract_decision()
    test_extract_preference()
    test_extract_warning()
    test_extract_short_text_skipped()
    test_tag_extraction_code_identifiers()
    test_tag_extraction_technical_terms()

    # GapDetector
    test_detect_gap_basic()
    test_no_false_positives()
    test_extract_gap_topic()
    test_extract_gap_topic_none()

    print("\nAll Verbatim + Gap Detector tests passed!")
