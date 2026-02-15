"""
The Librarian — Verbatim Extractor

No-LLM extraction strategy. Stores conversation chunks verbatim
with heuristic categorization and tag extraction.

This aligns with the core design principle: no compression, ever.
The search engine (FTS5 + embeddings) handles relevance at retrieval
time. We just need to get the content into the rolodex with enough
metadata to make it findable.

Modality detection comes from ContentChunker (regex-based, free).
Tags come from simple pattern matching (code identifiers, proper
nouns, quoted strings). Categories map from modality.
"""
import re
from datetime import datetime
from typing import Dict, List

from ..core.types import ContentModality
from ..retrieval.entity_extractor import EntityExtractor


# ─── Temporal Tags ───────────────────────────────────────────────────────────

_TIME_BUCKETS = {
    range(5, 12): "morning",
    range(12, 17): "afternoon",
    range(17, 21): "evening",
}


def _temporal_tags(dt: datetime = None) -> List[str]:
    """Generate date and time tags from a datetime."""
    dt = dt or datetime.now()
    tags = [
        dt.strftime("%Y-%m-%d"),           # "2026-02-13"
        dt.strftime("%A").lower(),          # "thursday"
        dt.strftime("%B").lower(),          # "february"
        str(dt.year),                       # "2026"
    ]
    # Time-of-day bucket
    hour = dt.hour
    for r, label in _TIME_BUCKETS.items():
        if hour in r:
            tags.append(label)
            break
    else:
        tags.append("night")               # 9pm–4am
    # Clock hour for precision (cross-platform: %-I is POSIX-only, %#I is Windows-only)
    hour12 = dt.hour % 12 or 12
    am_pm = "am" if dt.hour < 12 else "pm"
    tags.append(f"{hour12}{am_pm}")  # "9pm", "10am"
    return tags


# ─── Modality → Category Mapping ──────────────────────────────────────────

MODALITY_CATEGORY_MAP = {
    ContentModality.CODE: "implementation",
    ContentModality.MATH: "fact",
    ContentModality.PROSE: "note",
    ContentModality.CONVERSATIONAL: "note",
    ContentModality.STRUCTURED: "reference",
}


class VerbatimExtractor:
    """
    Heuristic-based extraction — no LLM calls.

    Implements the same interface as LLMAdapter.extract() so it can
    be used as a drop-in fallback. Content goes in verbatim; metadata
    is inferred from patterns in the text.
    """

    def __init__(self):
        self._entity_extractor = EntityExtractor()

    async def extract(
        self,
        chunk: str,
        modality: ContentModality,
    ) -> List[Dict]:
        """
        Extract entries from a chunk using heuristics.
        Returns one entry per chunk (verbatim content + inferred metadata).
        """
        if not chunk or len(chunk.strip()) < 10:
            return []

        category = self._categorize(chunk, modality)
        tags = self._extract_tags(chunk, modality)

        # Phase 11: Attribution tagging — detect who said/owns this content
        entities = self._entity_extractor.extract_from_content(chunk)
        if entities.attribution:
            tags.append(f"attributed:{entities.attribution}")
        # Add extracted proper nouns as tags (capped)
        for noun in entities.proper_nouns[:3]:
            tag = noun.lower()
            if tag not in tags and len(tag) >= 3:
                tags.append(tag)

        return [{
            "content": chunk.strip(),
            "category": category,
            "tags": tags,
            "linked_to": [],
        }]

    # ─── Categorization ───────────────────────────────────────────────────

    def _categorize(self, text: str, modality: ContentModality) -> str:
        """
        Assign a category based on modality + content heuristics.
        Overrides the modality default when strong signals are present.
        """
        # Check for strong category signals regardless of modality
        lower = text.lower()

        # ── Experience markers (check before generic categories) ──

        # Correction signals — something was wrong and got fixed
        if any(phrase in lower for phrase in [
            "actually,", "actually it", "actually the",
            "correction:", "that was wrong", "i was wrong",
            "not what i meant", "should have been", "the fix was",
            "fixed by", "the problem was", "root cause",
            "turns out", "it turns out", "realized the issue",
            "the bug was", "the error was", "corrected",
        ]):
            return "correction"

        # Friction signals — struggle, confusion, difficulty
        if any(phrase in lower for phrase in [
            "struggling", "struggled", "stuck on", "got stuck",
            "confused by", "confusing", "frustrating", "frustrated",
            "couldn't figure", "couldn't get", "didn't work",
            "kept failing", "keeps failing", "wrong command",
            "wrong approach", "wasted time", "took forever",
            "pain point", "headache", "annoying",
            "doesn't make sense", "not working",
        ]):
            return "friction"

        # Breakthrough signals — moments of clarity or success
        if any(phrase in lower for phrase in [
            "figured it out", "finally works", "that worked",
            "eureka", "breakthrough", "the key was",
            "the trick is", "the solution was", "now it works",
            "solved it", "got it working", "cracked it",
            "the insight was", "the answer is",
        ]):
            return "breakthrough"

        # Pivot signals — change of direction
        if any(phrase in lower for phrase in [
            "instead,", "pivoted to", "switched to",
            "changed approach", "abandoned", "scrapped",
            "let's try", "different approach", "new direction",
            "going with", "moving to", "replaced with",
            "threw out", "starting over", "rethinking",
        ]):
            return "pivot"

        # ── Standard categories ──

        # Decision signals
        if any(phrase in lower for phrase in [
            "we decided", "the decision", "i chose", "let's go with",
            "agreed to", "we'll use", "the plan is",
        ]):
            return "decision"

        # Instruction signals
        if any(phrase in lower for phrase in [
            "please", "i want", "i need", "can you", "could you",
            "make sure", "don't forget", "remember to",
        ]):
            return "instruction"

        # Preference signals
        if any(phrase in lower for phrase in [
            "i prefer", "i like", "i don't like", "always use",
            "never use", "my preference", "i'd rather",
        ]):
            return "preference"

        # Warning signals
        if any(phrase in lower for phrase in [
            "warning", "careful", "watch out", "don't", "avoid",
            "pitfall", "gotcha", "caveat", "be aware",
        ]):
            return "warning"

        # Definition signals
        if any(phrase in lower for phrase in [
            " is a ", " is an ", " refers to", " means ",
            "defined as", "definition:",
        ]):
            return "definition"

        # Fall back to modality-based category
        return MODALITY_CATEGORY_MAP.get(modality, "note")

    # ─── Tag Extraction ───────────────────────────────────────────────────

    def _extract_tags(self, text: str, modality: ContentModality) -> List[str]:
        """
        Extract search-friendly tags from text using regex heuristics.
        """
        tags = set()

        # Always add modality as a tag
        tags.add(modality.value)

        # Code identifiers: function/class/variable names
        if modality == ContentModality.CODE:
            tags.update(self._extract_code_identifiers(text))
        else:
            # Look for inline code references in prose
            inline_code = re.findall(r'`([^`]+)`', text)
            for code in inline_code[:5]:
                tag = code.strip()
                if 3 <= len(tag) <= 40:
                    tags.add(tag)

        # Proper nouns (capitalized words after sentence boundaries)
        proper_nouns = re.findall(
            r'(?<=[.!?] )(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            text
        )
        # Also catch mid-sentence capitalized words (likely proper nouns)
        mid_caps = re.findall(
            r'(?<=\s)([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*)',
            text
        )
        for noun in (proper_nouns + mid_caps)[:5]:
            noun = noun.strip()
            if 3 <= len(noun) <= 30 and noun.lower() not in _COMMON_WORDS:
                tags.add(noun.lower())

        # Quoted strings (likely important terms)
        quoted = re.findall(r'"([^"]{3,30})"', text)
        for q in quoted[:3]:
            tags.add(q.lower())

        # Technical terms (CamelCase, snake_case, UPPER_CASE)
        camel = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', text)
        snake = re.findall(r'\b([a-z]+(?:_[a-z]+)+)\b', text)
        upper = re.findall(r'\b([A-Z]+(?:_[A-Z]+)+)\b', text)
        for term in (camel + snake + upper)[:5]:
            if 3 <= len(term) <= 40:
                tags.add(term.lower())

        # Temporal tags — always included, don't count toward the cap
        temporal = _temporal_tags()
        semantic_tags = sorted(tags)[:10]
        return temporal + semantic_tags

    def _extract_code_identifiers(self, text: str) -> List[str]:
        """Extract function names, class names, etc. from code."""
        identifiers = set()

        # Python: def name, class Name
        for match in re.finditer(r'(?:def|class)\s+(\w+)', text):
            identifiers.add(match.group(1))

        # JavaScript/TypeScript: function name, const name, let name
        for match in re.finditer(
            r'(?:function|const|let|var)\s+(\w+)', text
        ):
            identifiers.add(match.group(1))

        # Rust: fn name, struct Name, impl Name
        for match in re.finditer(
            r'(?:fn|struct|impl|enum|trait)\s+(\w+)', text
        ):
            identifiers.add(match.group(1))

        # Go: func name, type Name
        for match in re.finditer(r'(?:func|type)\s+(\w+)', text):
            identifiers.add(match.group(1))

        # Import targets
        for match in re.finditer(r'(?:import|from)\s+(\w+)', text):
            identifiers.add(match.group(1))

        return [i for i in identifiers if len(i) > 2][:8]


# Common words to skip when extracting proper nouns
_COMMON_WORDS = {
    "the", "this", "that", "these", "those", "here", "there",
    "when", "where", "what", "which", "who", "how", "why",
    "for", "and", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "are", "also", "been",
    "have", "has", "will", "with", "would", "could", "should",
    "may", "might", "must", "shall", "each", "make", "like",
    "just", "over", "such", "take", "than", "them", "very",
    "some", "into", "most", "other", "then", "now", "look",
    "only", "come", "its", "after", "use", "two", "way",
    "about", "many", "time", "been", "more", "from",
    "let", "note", "see", "sure", "yes", "well",
}
