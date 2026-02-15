"""
The Librarian — Context Window Manager (Phase 9)

Manages what stays in the active context window vs what gets offloaded
to the rolodex. The goal: never hit compaction by keeping the working
buffer lean while relying on The Librarian for anything older.

Two controls determine the buffer boundary:
    1. Token budget — hard cap on how many tokens of conversation history
       stay in the active window (default ~20K tokens).
    2. Ingestion checkpoint — safety net: never prune past the last
       successful ingestion, because that's the guarantee the content
       is safely in the rolodex and retrievable.

What stays in the active window:
    - Recent messages within the token budget
    - Librarian recall block (injected fresh each turn)
    - A "bridge summary" of pruned content so the LLM keeps continuity

What gets offloaded:
    - Everything older than the buffer boundary
    - Fully indexed in the rolodex, searchable via hybrid search
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from .types import Message, MessageRole, ConversationState, estimate_tokens


@dataclass
class IngestionCheckpoint:
    """Records a successful ingestion event."""
    turn_number: int
    timestamp: datetime
    entry_count: int  # How many entries were created
    token_count: int  # Tokens in the ingested message


@dataclass
class ContextWindowState:
    """Snapshot of the current context window allocation."""
    total_messages: int
    active_messages: int          # Messages in the working buffer
    pruned_messages: int          # Messages offloaded to rolodex
    active_tokens: int            # Tokens in the working buffer
    pruned_tokens: int            # Tokens offloaded
    budget_remaining: int         # Tokens available before budget hit
    last_checkpoint_turn: int     # Turn number of most recent ingestion
    bridge_summary_tokens: int    # Tokens used by the bridge summary
    retrieval_budget: int         # Tokens reserved for Librarian recall


class ContextWindowManager:
    """
    Controls the sliding window over conversation history.

    Instead of keeping the full conversation in context (which causes
    compaction), only keeps the recent working set. Everything older
    is safely in The Librarian's rolodex and can be recalled on demand.

    The manager produces two outputs:
        1. active_messages — the messages to include in context
        2. bridge_summary — a compact summary of what was pruned,
           giving the LLM continuity without the full history
    """

    def __init__(
        self,
        token_budget: int = 20_000,
        retrieval_budget: int = 20_000,
        min_active_turns: int = 4,
        bridge_summary_max_tokens: int = 1_000,
    ):
        """
        Args:
            token_budget: Max tokens of conversation history to keep
                          in the active window.
            retrieval_budget: Tokens reserved for Librarian recall block.
            min_active_turns: Always keep at least this many recent turns,
                              even if they exceed the token budget slightly.
            bridge_summary_max_tokens: Max tokens for the bridge summary
                                       of pruned content.
        """
        self.token_budget = token_budget
        self.retrieval_budget = retrieval_budget
        self.min_active_turns = min_active_turns
        self.bridge_summary_max_tokens = bridge_summary_max_tokens

        # Track ingestion checkpoints
        self._checkpoints: List[IngestionCheckpoint] = []

        # Bridge summary of pruned messages
        self._bridge_summary: str = ""
        self._bridge_tokens: int = 0

        # Pruning state
        self._prune_boundary: int = 0  # Index into messages list

    # ─── Ingestion Tracking ──────────────────────────────────────────

    def record_checkpoint(
        self,
        turn_number: int,
        entry_count: int = 1,
        token_count: int = 0,
    ) -> None:
        """
        Record that a message at this turn was successfully ingested
        into the rolodex. This turn is now safe to prune from the
        active window.
        """
        self._checkpoints.append(IngestionCheckpoint(
            turn_number=turn_number,
            timestamp=datetime.utcnow(),
            entry_count=entry_count,
            token_count=token_count,
        ))

    @property
    def last_checkpoint_turn(self) -> int:
        """Turn number of the most recent ingestion checkpoint."""
        if not self._checkpoints:
            return 0
        return self._checkpoints[-1].turn_number

    @property
    def total_checkpoints(self) -> int:
        return len(self._checkpoints)

    # ─── Window Computation ──────────────────────────────────────────

    def compute_active_window(
        self,
        messages: List[Message],
    ) -> Tuple[List[Message], List[Message]]:
        """
        Given the full conversation history, determine which messages
        stay in the active window and which get pruned.

        Returns:
            (active_messages, pruned_messages)

        Rules:
            1. Always keep at least min_active_turns recent messages
            2. Never prune past the last ingestion checkpoint
            3. Stay within token_budget (soft limit — rule 1 takes priority)
        """
        if not messages:
            return [], []

        n = len(messages)

        # Start from the end and work backward, accumulating tokens
        active_start = n  # Will count backward
        running_tokens = 0

        for i in range(n - 1, -1, -1):
            msg_tokens = messages[i].token_count
            if running_tokens + msg_tokens > self.token_budget:
                # Would exceed budget — check minimum turns
                active_count = n - i - 1
                if active_count >= self.min_active_turns:
                    active_start = i + 1
                    break
                # Below minimum — keep going despite budget
                running_tokens += msg_tokens
            else:
                running_tokens += msg_tokens
                active_start = i

        # Safety net: never prune past the last checkpoint
        if self._checkpoints:
            last_safe_turn = self.last_checkpoint_turn
            # Find the message index for this turn
            for i, msg in enumerate(messages):
                if msg.turn_number > last_safe_turn:
                    # Everything from this index onward hasn't been ingested
                    # so we can't prune it
                    if i < active_start:
                        active_start = i
                    break

        # Clamp
        active_start = max(0, min(active_start, n))

        pruned = messages[:active_start]
        active = messages[active_start:]

        # Update internal state
        self._prune_boundary = active_start

        # Update bridge summary if we pruned anything new
        if pruned:
            self._update_bridge_summary(pruned)

        return active, pruned

    def get_active_messages(
        self,
        state: ConversationState,
    ) -> List[Message]:
        """
        Convenience: compute and return just the active messages
        for the given conversation state.
        """
        active, _ = self.compute_active_window(state.messages)
        return active

    # ─── Bridge Summary ──────────────────────────────────────────────

    def _update_bridge_summary(self, pruned_messages: List[Message]) -> None:
        """
        Build a compact summary of pruned messages so the LLM
        maintains continuity. Not a full summary — just enough
        to orient: what topics were discussed, key decisions made,
        and the general arc.
        """
        if not pruned_messages:
            self._bridge_summary = ""
            self._bridge_tokens = 0
            return

        # Collect key signals from pruned messages
        total_turns = len(pruned_messages)
        user_msgs = [m for m in pruned_messages if m.role == MessageRole.USER]
        asst_msgs = [m for m in pruned_messages if m.role == MessageRole.ASSISTANT]
        total_tokens = sum(m.token_count for m in pruned_messages)

        # Extract first/last user messages for arc
        first_user = user_msgs[0].content[:200] if user_msgs else ""
        last_user = user_msgs[-1].content[:200] if user_msgs else ""

        # Build bridge
        parts = [
            f"[Context Bridge: {total_turns} earlier messages "
            f"({total_tokens:,} tokens) offloaded to memory]",
        ]

        if first_user:
            parts.append(f"Session started with: \"{first_user.strip()}\"")

        if last_user and last_user != first_user:
            parts.append(f"Before current work: \"{last_user.strip()}\"")

        # Add checkpoint info
        if self._checkpoints:
            parts.append(
                f"[{len(self._checkpoints)} ingestion checkpoints — "
                f"all earlier content is searchable via recall]"
            )

        self._bridge_summary = "\n".join(parts)
        self._bridge_tokens = estimate_tokens(self._bridge_summary)

    @property
    def bridge_summary(self) -> str:
        """The current bridge summary text, or empty if nothing pruned."""
        return self._bridge_summary

    # ─── Context Assembly ────────────────────────────────────────────

    def build_context_payload(
        self,
        state: ConversationState,
        recall_block: str = "",
    ) -> Dict[str, Any]:
        """
        Build the full context payload for the LLM.

        Returns a dict with:
            - bridge_summary: str — compact summary of pruned history
            - recall_block: str — Librarian's retrieved context
            - active_messages: List[Message] — recent conversation
            - metadata: dict — window stats for debugging
        """
        active, pruned = self.compute_active_window(state.messages)

        active_tokens = sum(m.token_count for m in active)
        pruned_tokens = sum(m.token_count for m in pruned)
        recall_tokens = estimate_tokens(recall_block) if recall_block else 0

        return {
            "bridge_summary": self._bridge_summary if pruned else "",
            "recall_block": recall_block,
            "active_messages": active,
            "metadata": {
                "total_messages": len(state.messages),
                "active_messages": len(active),
                "pruned_messages": len(pruned),
                "active_tokens": active_tokens,
                "pruned_tokens": pruned_tokens,
                "bridge_tokens": self._bridge_tokens,
                "recall_tokens": recall_tokens,
                "total_context_tokens": (
                    active_tokens + self._bridge_tokens + recall_tokens
                ),
                "budget_remaining": max(
                    0, self.token_budget - active_tokens
                ),
                "last_checkpoint_turn": self.last_checkpoint_turn,
                "checkpoints": len(self._checkpoints),
            },
        }

    # ─── Status ──────────────────────────────────────────────────────

    def get_state(self, messages: Optional[List[Message]] = None) -> ContextWindowState:
        """Return a snapshot of the current window state."""
        if messages:
            active, pruned = self.compute_active_window(messages)
        else:
            active = []
            pruned = []

        active_tokens = sum(m.token_count for m in active)
        pruned_tokens = sum(m.token_count for m in pruned)

        return ContextWindowState(
            total_messages=len(active) + len(pruned),
            active_messages=len(active),
            pruned_messages=len(pruned),
            active_tokens=active_tokens,
            pruned_tokens=pruned_tokens,
            budget_remaining=max(0, self.token_budget - active_tokens),
            last_checkpoint_turn=self.last_checkpoint_turn,
            bridge_summary_tokens=self._bridge_tokens,
            retrieval_budget=self.retrieval_budget,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return stats dict for inclusion in system stats."""
        return {
            "token_budget": self.token_budget,
            "retrieval_budget": self.retrieval_budget,
            "min_active_turns": self.min_active_turns,
            "checkpoints": len(self._checkpoints),
            "last_checkpoint_turn": self.last_checkpoint_turn,
            "prune_boundary": self._prune_boundary,
            "bridge_tokens": self._bridge_tokens,
        }
