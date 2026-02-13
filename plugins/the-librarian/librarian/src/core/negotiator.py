


import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .types import RolodexEntry, estimate_tokens


@dataclass
class NegotiationRound:

    entries_evaluated: int
    accepted: List[str]
    rejected: List[str]
    budget_after: int
    refined_query: Optional[str] = None


@dataclass
class NegotiationResult:

    accepted_entries: List[RolodexEntry]
    rejected_ids: List[str]
    rounds: List[NegotiationRound]
    budget_used: int
    budget_remaining: int
    total_rounds: int
    resolved: bool


NEGOTIATION_PROMPT = """You are The Librarian's context negotiator. Your job is to decide which memory entries
should be injected into the assistant's context to resolve an information gap.

THE GAP:
The assistant was asked about: "{gap_topic}"
The assistant couldn't answer because it lacks context from earlier in the conversation.
{chain_context}
CANDIDATE ENTRIES FROM MEMORY:
{entries_block}

TOKEN BUDGET: {budget} tokens remaining for context injection.

INSTRUCTIONS:
For each entry, decide ACCEPT or REJECT based on:
1. Does this entry directly help answer the gap? (ACCEPT if yes)
2. Is this entry relevant but not necessary? (REJECT to save budget)
3. Would including this waste budget on tangential info? (REJECT)

If NONE of the entries can resolve the gap, suggest a REFINED search query.

Respond with ONLY valid JSON:
{{
  "decisions": [
    {{"entry_id": "...", "action": "ACCEPT"  or "REJECT", "reason": "brief reason"}}
  ],
  "refined_query": null or "better search terms if entries don't help",
  "confidence": 0.0 to 1.0 that the accepted entries resolve the gap
}}"""


class ContextNegotiator:


    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        max_rounds: int = 2,
        cost_tracker=None,
    ):


        self._api_key = api_key
        self.model = model
        self.max_rounds = max_rounds
        self.cost_tracker = cost_tracker
        self._client = None

    def _get_client(self):

        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    async def negotiate(
        self,
        gap_topic: str,
        candidate_entries: List[RolodexEntry],
        relevance_scores: Dict[str, float],
        budget_tokens: int,
        search_fn=None,
        candidate_chains=None,
    ) -> NegotiationResult:


        if not candidate_entries:
            return NegotiationResult(
                accepted_entries=[],
                rejected_ids=[],
                rounds=[],
                budget_used=0,
                budget_remaining=budget_tokens,
                total_rounds=0,
                resolved=False,
            )

        rounds = []
        current_entries = candidate_entries
        current_scores = relevance_scores
        remaining_budget = budget_tokens
        all_accepted: List[RolodexEntry] = []
        all_rejected_ids: List[str] = []

        for round_num in range(self.max_rounds):

            evaluation = await self._evaluate_candidates(
                gap_topic, current_entries, current_scores, remaining_budget,
                chains=candidate_chains if round_num == 0 else None,
            )


            accepted_ids = set()
            rejected_ids = set()
            for decision in evaluation.get("decisions", []):
                eid = decision.get("entry_id", "")
                action = decision.get("action", "REJECT").upper()
                if action == "ACCEPT":
                    accepted_ids.add(eid)
                else:
                    rejected_ids.add(eid)


            round_accepted = []
            round_budget_used = 0
            for entry in current_entries:
                if entry.id in accepted_ids:
                    entry_tokens = estimate_tokens(entry.content) + 20
                    if round_budget_used + entry_tokens <= remaining_budget:
                        round_accepted.append(entry)
                        round_budget_used += entry_tokens
                    else:
                        rejected_ids.add(entry.id)

            remaining_budget -= round_budget_used
            all_accepted.extend(round_accepted)
            all_rejected_ids.extend(rejected_ids)

            refined_query = evaluation.get("refined_query")
            confidence = evaluation.get("confidence", 0.5)

            rounds.append(NegotiationRound(
                entries_evaluated=len(current_entries),
                accepted=[e.id for e in round_accepted],
                rejected=list(rejected_ids),
                budget_after=remaining_budget,
                refined_query=refined_query,
            ))


            if confidence >= 0.7 or not refined_query or not search_fn:
                break


            if round_num < self.max_rounds - 1 and refined_query and search_fn:
                new_entries, new_scores = await search_fn(refined_query)

                seen_ids = {e.id for e in candidate_entries}
                current_entries = [e for e in new_entries if e.id not in seen_ids]
                current_scores = {
                    eid: s for eid, s in new_scores.items()
                    if eid not in seen_ids
                }
                if not current_entries:
                    break

        total_budget_used = budget_tokens - remaining_budget
        return NegotiationResult(
            accepted_entries=all_accepted,
            rejected_ids=all_rejected_ids,
            rounds=rounds,
            budget_used=total_budget_used,
            budget_remaining=remaining_budget,
            total_rounds=len(rounds),
            resolved=len(all_accepted) > 0,
        )

    async def _evaluate_candidates(
        self,
        gap_topic: str,
        entries: List[RolodexEntry],
        scores: Dict[str, float],
        budget: int,
        chains=None,
    ) -> Dict:


        entries_block = self._format_entries_for_prompt(entries, scores)


        chain_context = ""
        if chains:
            chain_lines = [
                "",
                "REASONING CONTEXT (narrative trail from memory):",
            ]
            for chain in chains:
                topics = ", ".join(chain.topics) if chain.topics else ""
                turn_range = f"turns {chain.turn_range_start}-{chain.turn_range_end}"
                line = f"  [{turn_range}]"
                if topics:
                    line += f" Topics: {topics}"
                chain_lines.append(line)
                chain_lines.append(f"  {chain.summary}")
            chain_lines.append("")
            chain_context = "\n".join(chain_lines)

        prompt = NEGOTIATION_PROMPT.format(
            gap_topic=gap_topic,
            chain_context=chain_context,
            entries_block=entries_block,
            budget=budget,
        )

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )


            if self.cost_tracker and hasattr(response, "usage"):
                self.cost_tracker.record(
                    call_type="negotiation",
                    model=self.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )

            text = response.content[0].text.strip()

            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            return json.loads(text)

        except Exception:

            return self._heuristic_evaluation(entries, scores, budget)

    def _heuristic_evaluation(
        self,
        entries: List[RolodexEntry],
        scores: Dict[str, float],
        budget: int,
    ) -> Dict:


        decisions = []
        for entry in entries:
            score = scores.get(entry.id, 0.0)
            action = "ACCEPT" if score >= 0.5 else "REJECT"
            decisions.append({
                "entry_id": entry.id,
                "action": action,
                "reason": f"score={score:.2f}",
            })
        return {
            "decisions": decisions,
            "refined_query": None,
            "confidence": 0.6,
        }

    def _format_entries_for_prompt(
        self,
        entries: List[RolodexEntry],
        scores: Dict[str, float],
    ) -> str:

        lines = []
        for i, entry in enumerate(entries):
            score = scores.get(entry.id, 0.0)
            tokens = estimate_tokens(entry.content)
            category = entry.category.value.upper() if entry.category else "NOTE"
            tags = ", ".join(entry.tags[:5]) if entry.tags else ""
            preview = entry.content[:150].replace("\n", " ")
            lines.append(
                f"{i+1}. [{category}] \"{preview}...\" "
                f"(id: {entry.id[:12]}, relevance: {score:.2f}, ~{tokens} tokens)"
            )
            if tags:
                lines.append(f"   Tags: {tags}")
        return "\n".join(lines)
