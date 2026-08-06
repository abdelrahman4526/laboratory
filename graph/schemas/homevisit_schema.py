from typing import Optional

from pydantic import BaseModel, Field
from typing import Literal, Optional

class HomevisitData(BaseModel):
    name: Optional[str] = Field(
    None,
    description=(
        "Patient's full name (رباعي) - must include at least 4 parts "
        "(First + Father + Grandfather + Family name) as commonly used in Egypt. "
        "If patient provided fewer parts, extract only what's available; "
        "do NOT invent missing name parts."
    )
)
    phone_number: Optional[str] = Field(
        None,
        description="Patient phone number."
    )

    details: Optional[str] = Field(
        None,
        description="Requested laboratory test(s) exactly as understood from the conversation."
    )

    date: Optional[str] = Field(
        None,
        description="Resolved appointment date in ISO format if possible."
    )
    address: Optional[str] = Field(
        None,
        description="Patient home address for the visit, exactly as understood from the conversation."
    )
    time: Optional[str] = Field(
        None,
        description="Patient Preferred Appointment Time - extract in 24-hour format (HH:MM), and make sure to correctly determine AM/PM whether the time is written in Arabic or English, numeric or worded "
    )
class HomevisitResponse(BaseModel):

    reply: str = Field(
        description="Reply that will be sent to the user."
    )

    summary: str = Field(
        description=(
            "Updated English conversation summary. "
            "Always keep all collected homevisit information "
            "(patient name, phone, requested tests, appointment date, "
            "confirmation status) so future turns can continue without "
            "asking again."
        )
    )

    visit: HomevisitData = Field(
        description="Structured booking information extracted from this turn."
    )

    confirmed: bool = Field(
        description=(
            "True only if the user explicitly confirms the booking "
            "(e.g. تمام، ماشي، اه، أيوة، yes, confirm)."
        )
    )
    action: Literal["new", "reschedule"] = Field(
        "new",
        description=(
            "new: A client booking a meeting for the first time or a separate additional booking. "
            "reschedule: A client with an existing homevisit booking who wants to change it ( phone/name/address/date/time). "
                   ),
    )

    ready_to_save: bool = Field(
        description=(
            "True only if name, phone, requested tests,tmie, and appointment date "
            "are all available from the conversation."
        )
    )