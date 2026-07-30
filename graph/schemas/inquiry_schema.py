from pydantic import BaseModel, Field


class InquiryResponse(BaseModel):

    reply: str = Field(
        description="Reply to send to the patient."
    )

    summary: str = Field(
        description="Updated English summary of the conversation."
    )