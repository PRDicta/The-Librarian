


from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


PRICING = {

    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},

    "voyage-3": {"input": 0.06, "output": 0.0},
    "voyage-3-lite": {"input": 0.02, "output": 0.0},
}


@dataclass
class APICall:

    call_type: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CostTracker:


    def __init__(self):
        self._calls: List[APICall] = []
        self._total_cost: float = 0.0

    def record(
        self,
        call_type: str,
        model: str,
        input_tokens: int,
        output_tokens: int = 0,
    ) -> APICall:


        pricing = PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost = (
            (input_tokens / 1_000_000) * pricing["input"]
            + (output_tokens / 1_000_000) * pricing["output"]
        )

        api_call = APICall(
            call_type=call_type,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 8),
        )

        self._calls.append(api_call)
        self._total_cost += cost
        return api_call

    def get_session_cost(self) -> float:

        return round(self._total_cost, 6)

    def get_call_count(self) -> int:

        return len(self._calls)

    def get_breakdown(self) -> Dict[str, Dict[str, float]]:


        breakdown: Dict[str, Dict[str, float]] = {}
        for call in self._calls:
            if call.call_type not in breakdown:
                breakdown[call.call_type] = {
                    "cost_usd": 0.0,
                    "call_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            cat = breakdown[call.call_type]
            cat["cost_usd"] += call.cost_usd
            cat["call_count"] += 1
            cat["input_tokens"] += call.input_tokens
            cat["output_tokens"] += call.output_tokens


        for cat in breakdown.values():
            cat["cost_usd"] = round(cat["cost_usd"], 6)

        return breakdown

    def get_summary(self) -> Dict[str, object]:

        return {
            "total_cost_usd": self.get_session_cost(),
            "total_calls": self.get_call_count(),
            "breakdown": self.get_breakdown(),
        }

    def reset(self):

        self._calls.clear()
        self._total_cost = 0.0
