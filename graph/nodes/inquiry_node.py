from langchain_core.messages import HumanMessage, SystemMessage

from graph.schemas.inquiry_schema import InquiryResponse
from graph.state import AgentState
from graph.utils import detect_language_fallback
from llm.llm import get_gemini
from software_service.client_services import ClientService

INQUIRY_SYSTEM_PROMPT = """
You are a helpful laboratory assistant.

Your task is to answer patient questions about laboratory tests.

====================
RULES
====================

1. Answer ONLY using the information inside "Retrieved Knowledge".
2. Never invent prices, specimen types, preparation instructions, or durations.
3. If no relevant knowledge is available, politely say that the requested test could not be found.
4. If the patient asks generally about laboratory services, answer using the provided retrieved knowledge only.
5. Suggest booking only if appropriate.
6. Match the user's language.
"""


def inquiry_node(state: AgentState) -> dict:

    page_id = state.get("page_id")
    sender_id = state.get("sender_id")
    platform_id = state.get("platform_id")

    user_message = state["user_message"]

    current_summary = state.get("summary") or ""
    last_bot_message = state.get("last_bot_message") or ""

    rag_context = state.get("rag_context", "")

    llm = get_gemini()
    structured_llm = llm.with_structured_output(
        InquiryResponse,
        include_raw=True,
    )

    system_prompt = f"""
{INQUIRY_SYSTEM_PROMPT}

====================
RETRIEVED KNOWLEDGE
====================

{rag_context or "(No relevant laboratory information was retrieved.)"}

====================
MEMORY
====================

Summary:
{current_summary}

Last Bot Message:
{last_bot_message}
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    try:

        result = structured_llm.invoke(messages)

        parsed: InquiryResponse = result["parsed"]
        raw_response = result["raw"]

    except Exception as e:

        print(f"[Inquiry Node] LLM error: {e}")

        fallback = detect_language_fallback(
            user_message,
            arabic="عذرًا، حدث خطأ مؤقت أثناء معالجة الاستفسار.",
            default="Sorry, a temporary error occurred.",
        )

        return {
            "response": fallback,
            "summary": current_summary,
            "last_bot_message": fallback,
            "inquiry_saved": False,
            "inquiry_usage": None,
        }

    usage = getattr(raw_response, "usage_metadata", None)

    inquiry_usage = (
        {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        if usage
        else None
    )

    clean_reply = parsed.reply

    try:

        ClientService.update_client_summary_and_last_bot_message(
            sender_id=sender_id,
            page_id=page_id,
            platform_id=platform_id,
            summary=parsed.summary,
            last_bot_message=clean_reply,
        )

    except Exception as e:
        print(f"[Inquiry Node] Persist error: {e}")

    return {
        "response": clean_reply,
        "summary": parsed.summary,
        "last_bot_message": clean_reply,
        "inquiry_saved": True,
        "inquiry_usage": inquiry_usage,
    }