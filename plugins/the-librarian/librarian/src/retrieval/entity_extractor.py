"""
The Librarian — Entity Extractor

Extracts named entities and specific terms from queries and content for
exact-match boosting during search. Catches what embeddings miss:
proper nouns, file paths, technical identifiers, and attribution signals.

No LLM calls — entirely heuristic, runs in microseconds.
"""
import re
from typing import List, Set
from dataclasses import dataclass, field


@dataclass
class ExtractedEntities:
    """Result of entity extraction."""
    proper_nouns: List[str] = field(default_factory=list)
    file_paths: List[str] = field(default_factory=list)
    technical_terms: List[str] = field(default_factory=list)
    attribution: str = ""  # "user", "assistant", or ""
    all_entities: List[str] = field(default_factory=list)  # Flattened unique list


# Attribution patterns — signals that content is attributed to a specific speaker
_USER_ATTRIBUTION = [
    r"\bI\s+said\b", r"\bI\s+told\b", r"\bI\s+asked\b",
    r"\bI\s+suggested\b", r"\bI\s+mentioned\b", r"\bI\s+thought\b",
    r"\bmy\s+analogy\b", r"\bmy\s+idea\b", r"\bmy\s+approach\b",
    r"\bmy\s+suggestion\b", r"\bmy\s+question\b",
    r"\bI\s+used\b", r"\bI\s+was\b", r"\bI\s+had\b",
    r"\bwhat\s+I\b", r"\bwhat\s+did\s+I\b",
    r"\bwhen\s+I\b", r"\bwhere\s+I\b",
]

_ASSISTANT_ATTRIBUTION = [
    r"\byou\s+said\b", r"\byou\s+told\b", r"\byou\s+suggested\b",
    r"\byou\s+mentioned\b", r"\byou\s+thought\b",
    r"\byour\s+analogy\b", r"\byour\s+idea\b", r"\byour\s+approach\b",
    r"\byour\s+suggestion\b",
    r"\bwhat\s+you\b", r"\bwhat\s+did\s+you\b",
]

# Named person attribution (e.g., "Philip's analogy", "Philip said")
_NAMED_ATTRIBUTION = re.compile(
    r"\b([A-Z][a-z]+)(?:'s\s+\w+|\s+said|\s+told|\s+suggested|\s+mentioned|\s+asked)",
    re.UNICODE
)

# File path patterns
_FILE_PATH = re.compile(
    r"(?:"
    r"[A-Za-z]:\\[\w\\./\- ]+"       # Windows paths: C:\Users\...
    r"|/[\w/.\- ]+"                    # Unix paths: /src/core/...
    r"|[\w\-]+\.(?:py|js|ts|jsx|tsx|rs|go|java|cpp|c|h|rb|sh|sql|md|txt|json|yaml|yml|toml|cfg|ini|docx|xlsx|pdf|csv)"  # filename.ext
    r")"
)

# Technical terms — identifiers with underscores, dots, or camelCase internals
_TECHNICAL_TERM = re.compile(
    r"\b(?:"
    r"[a-z]+_[a-z_]+"                 # snake_case: _last_indexed_turn
    r"|[a-z]+[A-Z][a-zA-Z]+"          # camelCase: querySelector
    r"|[A-Z][a-z]+[A-Z][a-zA-Z]+"     # PascalCase: TopicRouter
    r"|[A-Z]{2,}[a-z][a-zA-Z]*"       # Acronym+word: FTSSearch
    r")\b"
)

# Words to exclude from proper noun detection
_COMMON_SENTENCE_STARTERS = {
    "the", "this", "that", "these", "those", "it", "its",
    "what", "which", "where", "when", "how", "who", "whom", "why",
    "can", "could", "would", "should", "will", "do", "does", "did",
    "is", "are", "was", "were", "am", "be", "been", "being",
    "have", "has", "had", "if", "but", "and", "or", "not", "no",
    "so", "yet", "for", "nor", "about", "after", "before",
    "now", "then", "here", "there", "also", "just", "very",
    "my", "your", "his", "her", "our", "their",
    "let", "please", "okay", "ok", "yes", "yeah", "sure",
    "hey", "hi", "hello",
    # Common verbs that appear capitalized at sentence start
    "show", "tell", "find", "search", "recall", "remember",
    "get", "give", "make", "take", "help", "use",
}

# Known project-specific proper nouns (bootstrap — grows via ingestion metadata)
_KNOWN_PROPER_NOUNS = {
    "librarian", "rolodex", "cowork", "claude", "anthropic",
    "haiku", "sonnet", "opus", "voyage",
    "sqlite", "fts5", "numpy", "fastapi", "pydantic",
    "philip", "dicta",
}


class EntityExtractor:
    """
    Extracts named entities from text for exact-match search boosting.
    Purely heuristic — no LLM calls, runs in microseconds.
    """

    def extract_from_query(self, query: str) -> ExtractedEntities:
        """Extract entities from a search query."""
        result = ExtractedEntities()

        # 1. Attribution detection
        result.attribution = self._detect_attribution(query)

        # 2. Named person extraction
        named_match = _NAMED_ATTRIBUTION.search(query)
        if named_match:
            name = named_match.group(1)
            if name.lower() not in _COMMON_SENTENCE_STARTERS:
                result.proper_nouns.append(name)

        # 3. Proper noun extraction (capitalized words)
        result.proper_nouns.extend(self._extract_proper_nouns(query))
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for noun in result.proper_nouns:
            lower = noun.lower()
            if lower not in seen:
                seen.add(lower)
                deduped.append(noun)
        result.proper_nouns = deduped

        # 4. File paths
        result.file_paths = _FILE_PATH.findall(query)

        # 5. Technical terms
        result.technical_terms = list(set(_TECHNICAL_TERM.findall(query)))

        # 6. Build flattened entity list
        all_ents: list = []
        all_ents.extend(result.proper_nouns)
        all_ents.extend(result.file_paths)
        all_ents.extend(result.technical_terms)
        result.all_entities = list(dict.fromkeys(all_ents))  # Dedup preserving order

        return result

    def extract_from_content(self, content: str) -> ExtractedEntities:
        """Extract entities from ingested content (for attribution tagging)."""
        result = ExtractedEntities()

        # Attribution in content is inverted:
        # "I said" in content from user role = user attribution
        # "actually, the bug was..." = user correction
        result.attribution = self._detect_content_attribution(content)

        # Proper nouns
        result.proper_nouns = self._extract_proper_nouns(content)

        # File paths
        result.file_paths = _FILE_PATH.findall(content)

        # Technical terms
        result.technical_terms = list(set(_TECHNICAL_TERM.findall(content)))

        # Flatten
        all_ents: list = []
        all_ents.extend(result.proper_nouns)
        all_ents.extend(result.file_paths)
        all_ents.extend(result.technical_terms)
        result.all_entities = list(dict.fromkeys(all_ents))

        return result

    def _detect_attribution(self, query: str) -> str:
        """Detect whether a query is asking about something the user or assistant said."""
        lower = query.lower()

        for pattern in _USER_ATTRIBUTION:
            if re.search(pattern, lower):
                return "user"

        for pattern in _ASSISTANT_ATTRIBUTION:
            if re.search(pattern, lower):
                return "assistant"

        return ""

    def _detect_content_attribution(self, content: str) -> str:
        """Detect attribution in ingested content."""
        lower = content.lower()

        # Look for named attribution patterns
        if _NAMED_ATTRIBUTION.search(content):
            match = _NAMED_ATTRIBUTION.search(content)
            name = match.group(1).lower()
            if name in _KNOWN_PROPER_NOUNS or name not in _COMMON_SENTENCE_STARTERS:
                return "named"

        # Look for explicit user voice markers
        user_markers = [
            r"\buser\s+said\b", r"\buser\s+asked\b", r"\buser\s+wants\b",
            r"\buser\s+suggested\b", r"\buser\s+noted\b",
            r"\bphilip'?s?\b",
        ]
        for pattern in user_markers:
            if re.search(pattern, lower):
                return "user"

        return ""

    def _extract_proper_nouns(self, text: str) -> List[str]:
        """Extract likely proper nouns from text."""
        proper_nouns = []
        seen: Set[str] = set()

        # Split into sentences (rough)
        sentences = re.split(r'[.!?]\s+|\n', text)

        for sentence in sentences:
            words = sentence.strip().split()
            for i, word in enumerate(words):
                # Clean punctuation
                clean = re.sub(r"[^a-zA-Z']", "", word)
                if not clean or len(clean) < 2:
                    continue

                # Check if capitalized
                if clean[0].isupper():
                    lower = clean.lower()

                    # Skip if it's the first word (sentence starter) unless it's a known name
                    if i == 0 and lower not in _KNOWN_PROPER_NOUNS:
                        continue

                    # Skip common non-proper words
                    if lower in _COMMON_SENTENCE_STARTERS:
                        continue

                    # It's capitalized, not a sentence starter, not common — likely proper noun
                    if lower not in seen:
                        seen.add(lower)
                        proper_nouns.append(clean)

                # Also catch known proper nouns even if lowercase
                elif word.lower() in _KNOWN_PROPER_NOUNS:
                    if word.lower() not in seen:
                        seen.add(word.lower())
                        proper_nouns.append(word)

        return proper_nouns
