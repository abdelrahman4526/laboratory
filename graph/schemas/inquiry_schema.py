from typing import Literal, Optional
from pydantic import BaseModel, Field

class LabTestItem(BaseModel):
    name: str = Field(
        description="Standardized lab test name, exactly as matched in Verified Lab Information."
    )
    price: float | None = Field(
        default=None,
        description="Price in EGP as a numeric value if present in Verified Lab Information, otherwise null. Never invent."
    )
    preparation: str | None = Field(
        default=None,
        description="Preparation instructions if present, otherwise null."
    )
    result_time: str | None = Field(
        default=None,
        description="Result turnaround time if present, otherwise null."
    )


# ----------------------------------------------------
# Inquiry Models
# ----------------------------------------------------
class InquiryResponse(BaseModel):
    reply: str = Field(
    default="",
    description=(
        "The FULL reply text shown to the patient, including the formatted "
        "lab test details (STRICT LAB INFORMATION FORMAT) and total price "
        "when applicable. This is the ONLY field the patient sees — "
        "`tests` and `total_price` below are structured metadata for "
        "logging/analytics ONLY and are never rendered to the user."
    )
)
    
    total_price: float | None = Field(
    default=None,
    description=(
        "Sum of prices for all lab tests mentioned in `reply` that have a "
        "valid price. Set to null if no prices are available or not all "
        "tests have a verified price."
    )
)
    summary: str = Field(
        description="Updated English summary of the conversation context and patient intent."
    )