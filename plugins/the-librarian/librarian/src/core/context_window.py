


from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from .types import Message, MessageRole, ConversationState, estimate_tokens


@dataclass
class IngestionCheckpoint:

    turn_number: int
    timestamp: datetime
    entry_count: int
    token_count: int


@dataclass
class ContextWindowState:

    total_messages: int
    active_messages: int
    pruned_messages: int
    active_tokens: int
    pruned_tokens: int
    budget_remaining: int
    last_checkpoint_turn: int
    bridge_summary_tokens: int
    retrieval_budget: int


class ContextWindowManager:


    def __init__(
        self,
        token_budget: int = 20_000,
        retrieval_budget: int = 20_000,
        min_active_turns: int = 4,
        bridge_summary_max_tokens: int = 1_000,
    ):


        self.token_budget = token_budget
        self.retrieval_budget = retrieval_budget
        self.min_active_turns = min_active_turns
        self.bridge_summary_max_tokens = bridge_summary_max_tokens


        self._checkpoints: List[IngestionCheckpoint] = []


        self._bridge_summary: str = ""
        self._bridge_tokens: int = 0


        self._prune_boundary: int = 0


    def record_checkpoint(
        self,
        turn_number: int,
        entry_count: int = 1,
        token_count: int = 0,
    ) -> None:


        self._checkpoints.append(IngestionCheckpoint(
            turn_number=turn_number,
            timestamp=datetime.utcnow(),
            entry_count=entry_count,
            token_count=token_count,
        ))

    @property
    def last_checkpoint_turn(self) -> int:

        if not self._checkpoints:
            return 0
        return self._checkpoints[-1].turn_number

    @property
    def total_checkpoints(self) -> int:
        return len(self._checkpoints)


    def compute_active_window(
        self,
        messages: List[Message],
    ) -> Tuple[List[Message], List[Message]]:


        if not messages:
            return [], []

        n = len(messages)


        active_start = n
        running_tokens = 0

        for i in range(n - 1, -1, -1):
            msg_tokens = messages[i].token_count
            if running_tokens + msg_tokens > self.token_budget:

                active_count = n - i - 1
                if active_count >= self.min_active_turns:
                    active_start = i + 1
                    break

                running_tokens += msg_tokens
            else:
                running_tokens += msg_tokens
                active_start = i


        if self._checkpoints:
            last_safe_turn = self.last_checkpoint_turn

            for i, msg in enumerate(messages):
                if msg.turn_number > last_safe_turn:


                    if i < active_start:
                        active_start = i
                    break


        active_start = max(0, min(active_start, n))

        pruned = messages[:active_start]
        active = messages[active_start:]


        self._prune_boundary = active_start


        if pruned:
            self._update_bridge_summary(pruned)

        return active, pruned

    def get_active_messages(
        self,
        state: ConversationState,
    ) -> List[Message]:


        active, _ = self.compute_active_window(state.messages)
        return active


    def _update_bridge_summary(self, pruned_messages: List[Message]) -> None:


        if not pruned_messages:
            self._bridge_summary = ""
            self._bridge_tokens = 0
            return


        total_turns = len(pruned_messages)
        user_msgs = [m for m in pruned_messages if m.role == MessageRole.USER]
        asst_msgs = [m for m in pruned_messages if m.role == MessageRole.ASSISTANT]
        total_tokens = sum(m.token_count for m in pruned_messages)


        first_user = user_msgs[0].content[:200] if user_msgs else ""
        last_user = user_msgs[-1].content[:200] if user_msgs else ""


        parts = [
            f"[Context Bridge: {total_turns} earlier messages "
            f"({total_tokens:,} tokens) offloaded to memory]",
        ]

        if first_user:
            parts.append(f"Session started with: \"{first_user.strip()}\"")

        if last_user and last_user != first_user:
            parts.append(f"Before current work: \"{last_user.strip()}\"")


        if self._checkpoints:
            parts.append(
                f"[{len(self._checkpoints)} ingestion checkpoints — "
                f"all earlier content is searchable via recall]"
            )

        self._bridge_summary = "\n".join(parts)
        self._bridge_tokens = estimate_tokens(self._bridge_summary)

    @property
    def bridge_summary(self) -> str:

        return self._bridge_summary


    def build_context_payload(
        self,
        state: ConversationState,
        recall_block: str = "",
    ) -> Dict[str, Any]:


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


    def get_state(self, messages: Optional[List[Message]] = None) -> ContextWindowState:

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

        return {
            "token_budget": self.token_budget,
            "retrieval_budget": self.retrieval_budget,
            "min_active_turns": self.min_active_turns,
            "checkpoints": len(self._checkpoints),
            "last_checkpoint_turn": self.last_checkpoint_turn,
            "prune_boundary": self._prune_boundary,
            "bridge_tokens": self._bridge_tokens,
        }
