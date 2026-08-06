from langchain_core.messages import HumanMessage, SystemMessage

from graph.nodes.complaint_tool import save_complaint_tool
from graph.schemas.complaint_sechema import ComplaintResponse
from graph.state import AgentState
from graph.utils import detect_language_fallback
from llm.llm import get_gemini
from software_service.client_services import ClientService

COMPLAINT_SYSTEM_PROMPT = """
You are a professional customer support assistant for a medical laboratory.

Your task is to register customer complaints.

====================
REQUIRED FIELDS
====================

- phone
- complaint_text

====================
RULES
====================

1. Never ask for fields already provided in the conversation summary.
2. Ask for ONE missing field at a time.
3. Match the user's language.
4. confirmed=true when the user clearly wants to submit the complaint.
5. ready_to_save=true ONLY if phone and complaint_text are both available.
"""


def complaint_node(state: AgentState) -> dict:

    page_id = state.get("page_id")
    sender_id = state.get("sender_id")
    platform_id = state.get("platform_id")

    user_message = state["user_message"]
    current_summary = state.get("summary") or ""
    last_bot_message = state.get("last_bot_message") or ""

    llm = get_gemini()
    structured_llm = llm.with_structured_output(
        ComplaintResponse,
        include_raw=True,
    )

    system_prompt = f"""
{COMPLAINT_SYSTEM_PROMPT}

====================
ALREADY COLLECTED
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
        parsed: ComplaintResponse = result["parsed"]
        raw_response = result["raw"]

    except Exception as e:
        print(f"[Complaint Node] {e}")

        fallback = detect_language_fallback(
            user_message,
            arabic="عذرًا، حدث خطأ مؤقت. حاول مرة أخرى.",
            default="Sorry, a temporary error occurred. Please try again.",
        )

        return {
            "response": fallback,
            "summary": current_summary,
            "last_bot_message": fallback,
            "complaint_saved": False,
            "complaint_usage": None,
        }

    usage = getattr(raw_response, "usage_metadata", None)

    complaint_usage = (
        {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        if usage
        else None
    )

    complaint_data = parsed.complaint.model_dump(exclude_none=True)

    required_fields = [
        "phone",
        "complaint_text",
    ]

    all_fields_present = all(
        complaint_data.get(field)
        for field in required_fields
    )

    complaint_saved = False

    if (
        parsed.ready_to_save
        and parsed.confirmed
        and all_fields_present
    ):

        try:

            result = save_complaint_tool.invoke(
                input={
                    **complaint_data,
                    "comes_from": str(platform_id or "unknown"),
                }
            )

            if result.success:

                complaint_saved = True

                clean_reply = detect_language_fallback(
                    user_message,
                    arabic="تم تسجيل شكواك بنجاح ✅ وسيتواصل معك فريقنا في أقرب وقت.",
                    default="Your complaint has been successfully registered. Our team will contact you soon.",
                )

            else:
                raise ValueError(result.message)

        except Exception as e:

            print(f"[Complaint Node] Tool Error: {e}")

            complaint_saved = False
            parsed.summary = current_summary

            clean_reply = detect_language_fallback(
                user_message,
                arabic="حدث خطأ أثناء تسجيل الشكوى. حاول مرة أخرى.",
                default="An error occurred while saving your complaint.",
            )

    else:
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
        print(f"[Complaint Node] Persist Error: {e}")

    return {
        "response": clean_reply,
        "summary": parsed.summary,
        "last_bot_message": clean_reply,
        "complaint_saved": complaint_saved,
        "complaint_usage": complaint_usage,
    }