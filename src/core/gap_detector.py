"""
The Librarian — Gap Detector

Pure-function gap detection extracted from the working agent.
Detects when an LLM response signals missing context — phrases like
"I don't have that information" or "let me look that up."

No LLM dependency. Regex-only. Host apps can use this independently
to decide when to query The Librarian for retrieval.

Usage:
    from src.core.gap_detector import detect_gap, extract_gap_topic

    response = "I don't have access to the configuration details..."
    topic = extract_gap_topic(response)
    if topic:
        context = await librarian.retrieve(topic)
"""
import re
from typing import List, Optional


# ─── Gap Signal Patterns ──────────────────────────────────────────────────

GAP_PATTERNS = [
    r"I don'?t have (?:access to|information about|details on|context for)",
    r"I'?m not (?:sure|certain) (?:about|what|which|how)",
    r"I (?:don'?t|do not) (?:recall|remember|see) (?:the|any|that)",
    r"(?:earlier|previously|before) in (?:our|the) conversation",
    r"(?:you|we) (?:mentioned|discussed|shared|talked about) (?:earlier|before|previously)",
    r"I (?:need|would need) (?:more|additional) (?:context|information|details)",
    r"(?:could you|can you) (?:remind me|share again|re-?send|provide)",
    r"I don'?t (?:have|see) (?:that|the) (?:file|document|article|code|information)",
    r"let me (?:look that up|check|search for that)",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in GAP_PATTERNS]


# ─── Public API ────────────────────────────────────────────────────────────

def detect_gap(response: str) -> Optional[str]:
    """
    Detect if a response contains indicators of missing context.

    Args:
        response: The LLM's response text to check

    Returns:
        The matched gap phrase if found, None otherwise
    """
    for pattern in _COMPILED_PATTERNS:
        match = pattern.search(response)
        if match:
            return match.group(0)
    return None


def extract_gap_topic(response: str) -> Optional[str]:
    """
    Extract the topic/subject of the knowledge gap from a response.
    Used to construct a retrieval query for The Librarian.

    Args:
        response: The LLM's response text

    Returns:
        The topic string to search for, or None if no gap detected
    """
    gap_signal = detect_gap(response)
    if not gap_signal:
        return None

    # Find the sentence containing the gap signal
    sentences = re.split(r'[.!?\n]', response)
    for sentence in sentences:
        for pattern in _COMPILED_PATTERNS:
            if pattern.search(sentence):
                # Remove the gap signal prefix to isolate the topic
                topic = sentence.strip()
                topic = pattern.sub("", topic).strip()
                topic = re.sub(r"^[,.\s]+", "", topic).strip()
                if topic and len(topic) > 5:
                    return topic

    # Fallback: return the whole gap signal
    return gap_signal
