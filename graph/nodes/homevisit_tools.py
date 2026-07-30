from langchain_core.tools import tool

from graph.utils import get_platform_name
from software_service.homevisit_service import homevisitService ,homevisitResult

@tool
def save_visit_tool(
    name: str,
    phone: str,
    date: str,
    details: str,
    address: str,
    comes_from: str = "unknown",
) -> homevisitResult:
    """
    Save a confirmed appointment booking to the database.

    Returns a BookingResult.
    """
    platform_name = get_platform_name(comes_from)

    return homevisitService.create_visit(
        name=name,
        phone_number=phone,
        date=date,
        details=details,
        comes_from=platform_name,
    )
