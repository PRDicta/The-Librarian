"""
The Librarian — Context Builder
Formats retrieved rolodex entries into a coherent context block
for injection into the working agent's prompt.
"""
from typing import List, Optional, Any, Dict
from ..core.types import RolodexEntry
class ContextBuilder:
    """
    Formats retrieved entries into a readable context block
    that gets injected into the working agent's context window.
    """
    HEADER = "═══ RETRIEVED FROM MEMORY ═══"
    FOOTER = "═══ END RETRIEVED CONTEXT ═══"
    SEPARATOR = "───"
    # Phase 7: Reasoning chain headers
    CHAIN_HEADER = "═══ REASONING CONTEXT ═══"
    CHAIN_FOOTER = "═══ END REASONING CONTEXT ═══"

    PROFILE_HEADER = "═══ USER PROFILE ═══"
    PROFILE_FOOTER = "═══ END USER PROFILE ═══"

    USER_KNOWLEDGE_HEADER = "═══ USER KNOWLEDGE ═══"
    USER_KNOWLEDGE_FOOTER = "═══ END USER KNOWLEDGE ═══"

    def build_context_block(
        self,
        entries: List[RolodexEntry],
        current_session_id: Optional[str] = None,
        chains: Optional[List[Any]] = None,
    ) -> str:
        """
        Format a list of retrieved entries into a single context string.
        Designed to be clearly delineated so the working agent knows
        this is injected reference material, not conversation.

        Verbatim entries appear FIRST — they set the factual frame.
        Reasoning chains follow as supplementary narrative ("why").

        When current_session_id is provided, entries from other sessions
        get a [From prior session] marker.
        """
        parts = []

        # Verbatim entries first (the "what" — always primary)
        if entries:
            lines = [self.HEADER, ""]
            for i, entry in enumerate(entries):
                lines.append(self._format_entry(entry, i + 1, current_session_id))
                lines.append("")
            lines.append(self.FOOTER)
            parts.append("\n".join(lines))

        # Reasoning chains second (narrative "why" — supplementary)
        if chains:
            chain_block = self._build_chain_block(chains)
            if chain_block:
                parts.append("")
                parts.append(chain_block)

        return "\n".join(parts)

    def _build_chain_block(self, chains: List[Any]) -> str:
        """Format reasoning chains into a narrative context block."""
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
        """Format a single entry with metadata header."""
        parts = []
        # Header line with category and tags
        category_label = entry.category.value.upper()
        tags_str = ", ".join(entry.tags) if entry.tags else ""
        verbatim_flag = "" if getattr(entry, "verbatim_source", True) else "  [SUMMARY]"
        header = f"[{index}] [{category_label}]{verbatim_flag}"
        if tags_str:
            header += f"  Tags: {tags_str}"
        # Phase 4: tag cross-session entries
        if (current_session_id
                and entry.conversation_id
                and entry.conversation_id != current_session_id):
            short_id = entry.conversation_id[:8]
            header += f"  [From prior session: {short_id}]"
        parts.append(header)
        # Content (verbatim — this is the whole point)
        parts.append(entry.content)
        # Separator
        parts.append(self.SEPARATOR)
        return "\n".join(parts)
    # ─── Proactive Context (Phase 3) ─────────────────────────────────

    PROACTIVE_HEADER = "═══ PROACTIVE CONTEXT (anticipated) ═══"
    PROACTIVE_FOOTER = "═══ END PROACTIVE CONTEXT ═══"

    def build_proactive_context_block(
        self, entries: List[RolodexEntry], strategy: str = "embedding"
    ) -> str:
        """
        Format high-confidence preloaded entries for proactive injection.
        Visually distinct from reactive retrieved context so the working
        agent knows this is anticipatory, not a direct retrieval.
        """
        if not entries:
            return ""
        lines = [self.PROACTIVE_HEADER]
        lines.append(f"(Predicted via {strategy} — use if relevant)\n")
        for i, entry in enumerate(entries):
            lines.append(self._format_entry(entry, i + 1))
            lines.append("")
        lines.append(self.PROACTIVE_FOOTER)
        return "\n".join(lines)

    def build_profile_block(self, profile: Dict[str, Any]) -> str:
        """Format the user profile as a readable block for context injection."""
        if not profile:
            return ""
        lines = [self.PROFILE_HEADER, ""]
        for key, entry in profile.items():
            display_key = key.replace("_", " ").title()
            lines.append(f"{display_key}: {entry['value']}")
        lines.append("")
        lines.append(self.PROFILE_FOOTER)
        return "\n".join(lines)

    def build_user_knowledge_block(self, entries: list) -> str:
        """Format user_knowledge entries as a persistent context block.

        These are always injected at boot alongside the profile.
        """
        if not entries:
            return ""
        lines = [self.USER_KNOWLEDGE_HEADER, ""]
        for entry in entries:
            tags_str = ", ".join(entry.tags) if entry.tags else ""
            if tags_str:
                lines.append(f"[{tags_str}]")
            lines.append(entry.content)
            lines.append(self.SEPARATOR)
            lines.append("")
        lines.append(self.USER_KNOWLEDGE_FOOTER)
        return "\n".join(lines)

    def build_not_found_message(self, query_text: str) -> str:
        """
        Generate a message when the Librarian can't find anything.
        The working agent uses this to fall back to asking the user.
        """
        return (
            f"[Librarian: No matching entries found for '{query_text}'. "
            f"This information may not have been discussed yet, or the query "
            f"may need to be rephrased. Consider asking the user directly.]"
        )
