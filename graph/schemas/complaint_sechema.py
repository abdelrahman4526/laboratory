from typing import Optional
from pydantic import BaseModel, Field


class ComplaintData(BaseModel):
    phone: Optional[str] = Field(
        None,
        description="User phone number."
    )

    complaint_text: Optional[str] = Field(
        None,
        description="Detailed complaint."
    )


class ComplaintResponse(BaseModel):
    reply: str = Field(description="Reply to the user.")

    summary: str = Field(
        description="Updated English conversation summary."
    )

    complaint: ComplaintData = Field(
        description="Structured complaint information."
    )

    confirmed: bool = Field(
        description="True if enough information exists to submit."
    )

    ready_to_save: bool = Field(
        description="True only if phone and complaint_text exist."
    )