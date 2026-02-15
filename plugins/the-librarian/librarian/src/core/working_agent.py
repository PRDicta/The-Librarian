"""
The Librarian — Working Agent
Wrapper around Claude Sonnet/Opus. Handles API calls,
system prompt injection, and gap signal detection.

This is the DEMO component — only used when an API key is available.
Gap detection has been extracted to gap_detector.py for standalone use.
"""
import re
from typing import List, Optional, Dict
import anthropic
from .types import Message, MessageRole, estimate_tokens
from .gap_detector import detect_gap, extract_gap_topic as _extract_gap_topic
SYSTEM_PROMPT = """You are a knowledgeable and helpful assistant.
You have access to a memory system called "The Librarian" that stores information from our entire conversation. When you encounter a gap — something you should know from earlier in our conversation but can't find in your current context — signal it clearly:
\- Say "Let me look that up..." when you need to retrieve earlier context
\- Say "I need more context on..." when information is missing
\- Say "I don't have that information in my current context" when referencing something not present
The system will retrieve relevant context and provide it to you. When you receive retrieved context, use it naturally — you don't need to cite the retrieval mechanism to the user.
Important: Do NOT make up information you're unsure about. If something from earlier in the conversation isn't in your current context, signal the gap rather than guessing."""
class WorkingAgent:
    """
    The primary conversational agent (Claude Sonnet/Opus).
    Handles user interactions and signals gaps to the Librarian.
    """
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
    async def query(
        self,
        messages: List[Message],
        user_input: str,
        retrieved_context: Optional[str] = None,
        proactive_context: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> str:
        """
        Query the working agent.
        Args:
            messages: Conversation history (for context window)
            user_input: The new user message
            retrieved_context: Injected context from Librarian (optional)
            proactive_context: Anticipated context from preloading (optional)
            max_tokens: Max response tokens
        Returns:
            The model's response text
        """
        # Build API messages
        api_messages = self._build_messages(
            messages, user_input, retrieved_context, proactive_context
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system_prompt,
            messages=api_messages,
        )
        return response.content[0].text
    def detect_gap_signal(self, response: str) -> Optional[str]:
        """Detect if response contains gap indicators. Delegates to gap_detector."""
        return detect_gap(response)

    def extract_gap_topic(self, response: str) -> Optional[str]:
        """Extract gap topic from response. Delegates to gap_detector."""
        return _extract_gap_topic(response)
    def _build_messages(
        self,
        history: List[Message],
        user_input: str,
        retrieved_context: Optional[str] = None,
        proactive_context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Build the API messages array.
        Includes conversation history, proactive context (Phase 3),
        retrieved context (if any), and the new user message.
        """
        api_messages = []
        # Add conversation history
        for msg in history:
            if msg.role == MessageRole.SYSTEM:
                continue  # System handled separately
            api_messages.append({
                "role": msg.role.value,
                "content": msg.content,
            })
        # Phase 3: inject proactive context (high-confidence preloads)
        if proactive_context:
            api_messages.append({
                "role": "assistant",
                "content": f"[Anticipating context you may need...]\n\n{proactive_context}\n\nI have some potentially relevant context ready. Let me respond to your message.",
            })
        # Inject retrieved context before the user's message
        if retrieved_context:
            # Add as an assistant message with the retrieved context
            api_messages.append({
                "role": "assistant",
                "content": f"[Retrieving relevant context from memory...]\n\n{retrieved_context}\n\nI now have the context I needed. Let me respond to your message.",
            })
        # Add new user message
        api_messages.append({
            "role": "user",
            "content": user_input,
        })
        # Ensure messages alternate properly (Anthropic API requirement)
        api_messages = self._fix_message_alternation(api_messages)
        return api_messages
    def _fix_message_alternation(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Ensure messages alternate between user and assistant.
        Anthropic's API requires this pattern.
        """
        if not messages:
            return messages
        fixed = [messages[0]]
        for msg in messages[1:]:
            if msg["role"] == fixed[-1]["role"]:
                # Merge consecutive same-role messages
                fixed[-1]["content"] += "\n\n" + msg["content"]
            else:
                fixed.append(msg)
        # Ensure it starts with user
        if fixed and fixed[0]["role"] != "user":
            fixed.insert(0, {"role": "user", "content": "[conversation start]"})
        return fixed
