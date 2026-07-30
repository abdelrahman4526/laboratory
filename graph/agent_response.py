from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class AgentResponse:
    response:        str
    intent:          Optional[str]
    booking_saved:   Optional[bool]
    complaint_saved: Optional[bool]
    inquiry_saved:   Optional[bool]
    usage:           dict

    @staticmethod
    def from_result(result: dict) -> "AgentResponse":
        usage = {
            "intent":      result.get("intent_usage")      or {},
            "lab_info":    result.get("lab_info_usage")    or {},
            "booking":     result.get("booking_usage")     or {},
            "complaint":   result.get("complaint_usage")   or {},
            "direct":      result.get("direct_usage")      or {},
            "inquiry":     result.get("inquiry_usage")     or {},
        }

        return AgentResponse(
            response        = result.get("response") or "",
            intent          = result.get("intent"),
            booking_saved   = result.get("booking_saved"),
            complaint_saved = result.get("complaint_saved"),
            inquiry_saved   = result.get("inquiry_saved"),
            usage           = usage,
        )

    def to_dict(self) -> dict:
        return {
            "response":        self.response,
            "intent":          self.intent,
            "booking_saved":   self.booking_saved,
            "complaint_saved": self.complaint_saved,
            "inquiry_saved":   self.inquiry_saved,
            "usage":           self.usage,
        }
