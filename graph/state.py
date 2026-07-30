from typing import Any, TypedDict, Optional

class AgentState(TypedDict):

    # platform
    page_id: Optional[str]
    sender_id: Optional[str]
    platform_id: Optional[int]
    platform_name: Optional[str]

    # conversation
    user_message: str
    response: Optional[str]

    # routing
    intent: Optional[str]
    refined_query: Optional[str]

   
    # memory
    summary: Optional[str]
    last_bot_message: Optional[str]

    # flags
    booking_saved: Optional[bool]
    complaint_saved: Optional[bool]
    inquiry_saved: Optional[bool]

    # usage
    intent_usage: Optional[dict]
    retrieval_usage: Optional[dict]
    booking_usage: Optional[dict]
    complaint_usage: Optional[dict]
    inquiry_usage: Optional[dict]
    direct_usage: Optional[dict]

    booking_reference: Optional[str]
    booking_pdf: Optional[bytes]


    rag_context: Optional[str]
    search_results: Optional[list]