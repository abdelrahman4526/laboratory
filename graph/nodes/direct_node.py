from langchain_core.messages import HumanMessage, SystemMessage
from graph.prompt_service.lab_data import LabDataService
from graph.state import AgentState
from graph.utils import detect_language_fallback
from llm.llm import get_gemini
from software_service.client_services import ClientService
from pydantic import BaseModel, Field

class DirectResponse(BaseModel):
    reply: str = Field(description="The response message to the user.")
    summary: str = Field(description="An updated English summary of the user's overall state.")

DIRECT_SYSTEM_PROMPT = """
You are a friendly laboratory customer service representative.
Your task is to handle greetings, general chit-chat, or direct inquiries about the laboratory (such as working hours, contact numbers, address).

====================
RULES
====================
1. Be polite, concise, and helpful.
2. Rely only on the Laboratory Info provided. Do not make up information.
3. Keep the conversation contextually natural.
4. Match the user's language.
"""

def direct_node(state: AgentState) -> dict:
    page_id          = state.get("page_id")
    sender_id        = state.get("sender_id")
    platform_id      = state.get("platform_id")
    user_message     = state["user_message"]
    current_summary  = state.get("summary") or ""
    last_bot_message = state.get("last_bot_message") or ""

    lab_info, _ = LabDataService.get_all_lab_data(page_id)

    llm            = get_gemini()
    structured_llm = llm.with_structured_output(DirectResponse, include_raw=True)

    system_prompt = f"""
{DIRECT_SYSTEM_PROMPT}

====================
LABORATORY DATA
====================
{lab_info}

====================
ALREADY COLLECTED
====================
Summary:          {current_summary}
Last bot message: {last_bot_message}
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    try:
        result        = structured_llm.invoke(messages)
        parsed: DirectResponse = result["parsed"]
        raw_response  = result["raw"]
        clean_reply   = parsed.reply
        new_summary   = parsed.summary
    except Exception as e:
        print(f"[Direct Node] LLM error: {e}")
        clean_reply = detect_language_fallback(
            user_message,
            arabic="أهلاً بك في مختبرنا الطبي! كيف يمكنني مساعدتك اليوم؟ يمكنك حجز موعد، أو تقديم استفسار عن التحاليل، أو تسجيل شكوى.",
            default="Welcome to our medical laboratory! How can I help you today? You can book an appointment, make an inquiry, or submit a complaint.",
        )
        new_summary = current_summary
        raw_response = None

    usage = getattr(raw_response, "usage_metadata", None)
    direct_usage = (
        {
            "input_tokens":  usage.get("input_tokens",  0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens":  usage.get("total_tokens",  0),
        }
        if usage
        else None
    )

    try:
        ClientService.update_client_summary_and_last_bot_message(
            sender_id=sender_id,
            page_id=page_id,
            platform_id=platform_id,
            summary=new_summary,
            last_bot_message=clean_reply,
        )
    except Exception as e:
        print(f"[Direct Node] Persist error: {e}")

    return {
        "response":          clean_reply,
        "summary":           new_summary,
        "last_bot_message":  clean_reply,
        "direct_usage":      direct_usage,
    }
