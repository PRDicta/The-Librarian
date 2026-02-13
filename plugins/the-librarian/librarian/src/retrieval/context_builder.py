


from typing import List, Optional, Any
from ..core.types import RolodexEntry
class ContextBuilder:


    HEADER = "═══ RETRIEVED FROM MEMORY ═══"
    FOOTER = "═══ END RETRIEVED CONTEXT ═══"
    SEPARATOR = "───"

    CHAIN_HEADER = "═══ REASONING CONTEXT ═══"
    CHAIN_FOOTER = "═══ END REASONING CONTEXT ═══"

    def build_context_block(
        self,
        entries: List[RolodexEntry],
        current_session_id: Optional[str] = None,
        chains: Optional[List[Any]] = None,
    ) -> str:


        parts = []


        if chains:
            chain_block = self._build_chain_block(chains)
            if chain_block:
                parts.append(chain_block)
                parts.append("")


        if entries:
            lines = [self.HEADER, ""]
            for i, entry in enumerate(entries):
                lines.append(self._format_entry(entry, i + 1, current_session_id))
                lines.append("")
            lines.append(self.FOOTER)
            parts.append("\n".join(lines))

        return "\n".join(parts)

    def _build_chain_block(self, chains: List[Any]) -> str:

        if not chains:
            return ""
        lines = [self.CHAIN_HEADER, ""]
        for chain in chains:
            turn_range = f"turns {chain.turn_range_start}-{chain.turn_range_end}"
            topics = ", ".join(chain.topics) if chain.topics else ""
            line = f"[{turn_range}]"
            if topics:
                line += f"  Topics: {topics}"
            lines.append(line)
            lines.append(chain.summary)
            lines.append(self.SEPARATOR)
            lines.append("")
        lines.append(self.CHAIN_FOOTER)
        return "\n".join(lines)

    def _format_entry(
        self,
        entry: RolodexEntry,
        index: int,
        current_session_id: Optional[str] = None,
    ) -> str:

        parts = []

        category_label = entry.category.value.upper()
        tags_str = ", ".join(entry.tags) if entry.tags else ""
        header = f"[{index}] [{category_label}]"
        if tags_str:
            header += f"  Tags: {tags_str}"

        if (current_session_id
                and entry.conversation_id
                and entry.conversation_id != current_session_id):
            short_id = entry.conversation_id[:8]
            header += f"  [From prior session: {short_id}]"
        parts.append(header)

        parts.append(entry.content)

        parts.append(self.SEPARATOR)
        return "\n".join(parts)


    PROACTIVE_HEADER = "═══ PROACTIVE CONTEXT (anticipated) ═══"
    PROACTIVE_FOOTER = "═══ END PROACTIVE CONTEXT ═══"

    def build_proactive_context_block(
        self, entries: List[RolodexEntry], strategy: str = "embedding"
    ) -> str:


        if not entries:
            return ""
        lines = [self.PROACTIVE_HEADER]
        lines.append(f"(Predicted via {strategy} — use if relevant)\n")
        for i, entry in enumerate(entries):
            lines.append(self._format_entry(entry, i + 1))
            lines.append("")
        lines.append(self.PROACTIVE_FOOTER)
        return "\n".join(lines)

    def build_not_found_message(self, query_text: str) -> str:


        return (
            f"[Librarian: No matching entries found for '{query_text}'. "
            f"This information may not have been discussed yet, or the query "
            f"may need to be rephrased. Consider asking the user directly.]"
        )
