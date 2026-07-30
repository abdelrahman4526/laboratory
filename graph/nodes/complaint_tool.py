from langchain_core.tools import tool

from graph.utils import get_platform_name
from software_service.complaint_services import ComplaintService, ComplaintResult

@tool
def save_complaint_tool(
    phone: str,
    complaint_text: str,
    comes_from: str = "unknown",
) -> ComplaintResult:
    """
    Save a user complaint to the database.

    Returns a ComplaintResult.
    """
    platform_name = get_platform_name(comes_from)

    return ComplaintService.create_complaint(
        phone_number=phone,
        complaint_text=complaint_text,
        comes_from=platform_name,
    )
