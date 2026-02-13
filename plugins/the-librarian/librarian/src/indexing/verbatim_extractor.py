


import re
from typing import Dict, List

from ..core.types import ContentModality


MODALITY_CATEGORY_MAP = {
    ContentModality.CODE: "implementation",
    ContentModality.MATH: "fact",
    ContentModality.PROSE: "note",
    ContentModality.CONVERSATIONAL: "note",
    ContentModality.STRUCTURED: "reference",
}


class VerbatimExtractor:


    async def extract(
        self,
        chunk: str,
        modality: ContentModality,
    ) -> List[Dict]:


        if not chunk or len(chunk.strip()) < 10:
            return []

        category = self._categorize(chunk, modality)
        tags = self._extract_tags(chunk, modality)

        return [{
            "content": chunk.strip(),
            "category": category,
            "tags": tags,
            "linked_to": [],
        }]


    def _categorize(self, text: str, modality: ContentModality) -> str:


        lower = text.lower()


        if any(phrase in lower for phrase in [
            "we decided", "the decision", "i chose", "let's go with",
            "agreed to", "we'll use", "the plan is",
        ]):
            return "decision"


        if any(phrase in lower for phrase in [
            "please", "i want", "i need", "can you", "could you",
            "make sure", "don't forget", "remember to",
        ]):
            return "instruction"


        if any(phrase in lower for phrase in [
            "i prefer", "i like", "i don't like", "always use",
            "never use", "my preference", "i'd rather",
        ]):
            return "preference"


        if any(phrase in lower for phrase in [
            "warning", "careful", "watch out", "don't", "avoid",
            "pitfall", "gotcha", "caveat", "be aware",
        ]):
            return "warning"


        if any(phrase in lower for phrase in [
            " is a ", " is an ", " refers to", " means ",
            "defined as", "definition:",
        ]):
            return "definition"


        return MODALITY_CATEGORY_MAP.get(modality, "note")


    def _extract_tags(self, text: str, modality: ContentModality) -> List[str]:


        tags = set()


        tags.add(modality.value)


        if modality == ContentModality.CODE:
            tags.update(self._extract_code_identifiers(text))
        else:

            inline_code = re.findall(r'`([^`]+)`', text)
            for code in inline_code[:5]:
                tag = code.strip()
                if 3 <= len(tag) <= 40:
                    tags.add(tag)


        proper_nouns = re.findall(
            r'(?<=[.!?] )(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            text
        )

        mid_caps = re.findall(
            r'(?<=\s)([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*)',
            text
        )
        for noun in (proper_nouns + mid_caps)[:5]:
            noun = noun.strip()
            if 3 <= len(noun) <= 30 and noun.lower() not in _COMMON_WORDS:
                tags.add(noun.lower())


        quoted = re.findall(r'"([^"]{3,30})"', text)
        for q in quoted[:3]:
            tags.add(q.lower())


        camel = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', text)
        snake = re.findall(r'\b([a-z]+(?:_[a-z]+)+)\b', text)
        upper = re.findall(r'\b([A-Z]+(?:_[A-Z]+)+)\b', text)
        for term in (camel + snake + upper)[:5]:
            if 3 <= len(term) <= 40:
                tags.add(term.lower())

        return sorted(tags)[:10]

    def _extract_code_identifiers(self, text: str) -> List[str]:

        identifiers = set()


        for match in re.finditer(r'(?:def|class)\s+(\w+)', text):
            identifiers.add(match.group(1))


        for match in re.finditer(
            r'(?:function|const|let|var)\s+(\w+)', text
        ):
            identifiers.add(match.group(1))


        for match in re.finditer(
            r'(?:fn|struct|impl|enum|trait)\s+(\w+)', text
        ):
            identifiers.add(match.group(1))


        for match in re.finditer(r'(?:func|type)\s+(\w+)', text):
            identifiers.add(match.group(1))


        for match in re.finditer(r'(?:import|from)\s+(\w+)', text):
            identifiers.add(match.group(1))

        return [i for i in identifiers if len(i) > 2][:8]


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
