


import uuid
from typing import List, Optional
from datetime import datetime

from .types import ReasoningChain, Message, MessageRole
from ..storage.rolodex import Rolodex
from ..indexing.embeddings import EmbeddingManager


CHAIN_SUMMARY_PROMPT = """Summarize the reasoning thread in this conversation segment.
Focus on WHAT was discussed and WHY — capture decisions, trade-offs, and rejected alternatives.
Keep it to 1-3 sentences, narrative style. Do not list facts; tell the story of the reasoning.

Conversation segment:
{messages}

Summary:"""

EMERGENCY_SUMMARY_PROMPT = """EMERGENCY SNAPSHOT: The context window is filling up.
Summarize the COMPLETE reasoning flow of this segment before it's lost.
Preserve: key reasoning chains, decisions made, alternatives considered, and why.
Keep to 3-5 sentences max.

Conversation segment:
{messages}

Summary:"""


class ChainBuilder:


    def __init__(
        self,
        rolodex: Rolodex,
        embedding_manager: Optional[EmbeddingManager] = None,
        llm_adapter=None,
        chain_interval: int = 5,
        cost_tracker=None,
    ):
        self.rolodex = rolodex
        self.embeddings = embedding_manager
        self.llm_adapter = llm_adapter
        self.chain_interval = chain_interval
        self.cost_tracker = cost_tracker

    async def build_breadcrumb(
        self,
        session_id: str,
        messages: List[Message],
        turn_range_start: int,
        turn_range_end: int,
        related_entry_ids: List[str],
    ) -> Optional[ReasoningChain]:


        segment_messages = [
            m for m in messages
            if turn_range_start <= m.turn_number <= turn_range_end
        ]

        if not segment_messages:
            return None


        if self.llm_adapter:
            summary = await self._summarize_with_llm(
                segment_messages, CHAIN_SUMMARY_PROMPT
            )
        else:
            summary = self._summarize_verbatim(segment_messages)

        if not summary:
            return None

        topics = self._extract_topics(segment_messages)


        existing_chains = self.rolodex.get_chains_for_session(session_id)
        next_index = len(existing_chains)


        embedding = None
        if self.embeddings:
            try:
                embedding = await self.embeddings.embed_text(summary)
            except Exception:
                pass

        return ReasoningChain(
            id=str(uuid.uuid4()),
            session_id=session_id,
            chain_index=next_index,
            turn_range_start=turn_range_start,
            turn_range_end=turn_range_end,
            summary=summary,
            topics=topics,
            related_entries=related_entry_ids,
            embedding=embedding,
            created_at=datetime.utcnow(),
        )

    async def build_emergency_snapshot(
        self,
        session_id: str,
        messages: List[Message],
        related_entry_ids: List[str],
    ) -> Optional[ReasoningChain]:


        if not messages:
            return None


        existing_chains = self.rolodex.get_chains_for_session(session_id)
        last_indexed_turn = 0
        if existing_chains:
            last_indexed_turn = existing_chains[-1].turn_range_end


        if last_indexed_turn >= messages[-1].turn_number:
            return None

        segment_messages = [
            m for m in messages if m.turn_number > last_indexed_turn
        ]

        if not segment_messages:
            return None


        if self.llm_adapter:
            summary = await self._summarize_with_llm(
                segment_messages, EMERGENCY_SUMMARY_PROMPT,
                max_chars_per_message=1000
            )
        else:
            summary = self._summarize_verbatim(segment_messages)

        if not summary:
            return None

        summary += " [EMERGENCY SNAPSHOT]"
        topics = self._extract_topics(segment_messages)
        next_index = len(existing_chains)

        embedding = None
        if self.embeddings:
            try:
                embedding = await self.embeddings.embed_text(summary)
            except Exception:
                pass

        return ReasoningChain(
            id=str(uuid.uuid4()),
            session_id=session_id,
            chain_index=next_index,
            turn_range_start=last_indexed_turn + 1,
            turn_range_end=messages[-1].turn_number,
            summary=summary,
            topics=topics,
            related_entries=related_entry_ids,
            embedding=embedding,
            created_at=datetime.utcnow(),
        )

    async def _summarize_with_llm(
        self,
        messages: List[Message],
        prompt_template: str,
        max_chars_per_message: int = 500,
    ) -> str:

        msg_text = "\n".join([
            f"{m.role.value.upper()}: {m.content[:max_chars_per_message]}"
            for m in messages
        ])

        prompt = prompt_template.format(messages=msg_text)

        try:
            import anthropic

            if hasattr(self.llm_adapter, '_client'):
                client = self.llm_adapter._client
            elif hasattr(self.llm_adapter, 'client'):
                client = self.llm_adapter.client
            else:

                api_key = getattr(self.llm_adapter, 'api_key', None)
                if not api_key:
                    return self._summarize_verbatim(messages)
                client = anthropic.Anthropic(api_key=api_key)

            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )


            if self.cost_tracker and hasattr(response, "usage"):
                self.cost_tracker.record(
                    call_type="chain_summary",
                    model="claude-haiku-4-5-20251001",
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )

            if response.content:
                return response.content[0].text.strip()
        except Exception:
            pass


        return self._summarize_verbatim(messages)

    def _summarize_verbatim(self, messages: List[Message]) -> str:

        assistant_messages = [
            m for m in messages if m.role == MessageRole.ASSISTANT
        ]

        if not assistant_messages:

            user_messages = [
                m for m in messages if m.role == MessageRole.USER
            ]
            if user_messages:
                return f"Discussion: {user_messages[-1].content[:200]}..."
            return ""


        for msg in reversed(assistant_messages):
            if len(msg.content) > 100:
                return f"Discussing: {msg.content[:200]}..."


        combined = " ".join(m.content[:100] for m in assistant_messages)
        return f"Discussing: {combined[:200]}..."

    def _extract_topics(self, messages: List[Message]) -> List[str]:

        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'shall', 'can',
            'and', 'but', 'or', 'not', 'no', 'nor', 'of', 'in', 'to',
            'for', 'with', 'by', 'from', 'at', 'on', 'up', 'out', 'if',
            'about', 'into', 'through', 'then', 'than', 'too', 'very',
            'just', 'that', 'this', 'these', 'those', 'what', 'which',
            'when', 'where', 'how', 'who', 'whom', 'why', 'here', 'there',
            'each', 'every', 'all', 'both', 'few', 'more', 'most', 'other',
            'some', 'such', 'only', 'own', 'same', 'also', 'like', 'well',
            'back', 'even', 'still', 'way', 'take', 'come', 'make', 'know',
            'think', 'see', 'look', 'want', 'give', 'use', 'find', 'tell',
            'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call', 'need',
            'keep', 'let', 'begin', 'show', 'hear', 'play', 'run', 'move',
            'live', 'believe', 'bring', 'happen', 'write', 'provide', 'sit',
            'stand', 'lose', 'pay', 'meet', 'include', 'continue', 'set',
            'learn', 'change', 'lead', 'understand', 'watch', 'follow',
            'stop', 'create', 'speak', 'read', 'allow', 'add', 'spend',
            'grow', 'open', 'walk', 'win', 'offer', 'remember', 'love',
            'consider', 'appear', 'buy', 'wait', 'serve', 'die', 'send',
            'expect', 'build', 'stay', 'fall', 'cut', 'reach', 'kill',
            'remain', 'using', 'going', 'doing', 'having', 'getting',
            'it', 'he', 'she', 'they', 'we', 'you', 'i', 'me',
            'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our',
            'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
        }

        topics = []
        for msg in messages:
            words = msg.content.lower().split()
            for word in words:
                word = word.strip('.,!?;:()[]{}"\'-_`~')
                if len(word) > 3 and len(word) < 25 and word not in stopwords:
                    if word not in topics:
                        topics.append(word)
                    if len(topics) >= 5:
                        break
            if len(topics) >= 5:
                break

        return topics[:5]

    def should_generate_breadcrumb(
        self, turn_count: int, last_chain_turn: int
    ) -> bool:

        return (turn_count - last_chain_turn) >= self.chain_interval
