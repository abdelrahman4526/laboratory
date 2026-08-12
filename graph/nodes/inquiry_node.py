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

1. Answer ONLY using the facts (prices, sample types, preparation instructions, durations, availability) provided in the "Retrieved Knowledge" section.
2. Never invent or guess facts (prices, sample types, preparation instructions, durations, availability, or medical advice) that are not present in the Retrieved Knowledge.
3. You SHOULD use your general knowledge of medical/laboratory terminology to recognize when a test name in the Retrieved Knowledge is the same test the patient asked about, even if written differently (e.g. abbreviation vs full name, different spelling, different language, or a truncated/non-standard name such as "COMPLETE BLOOD PIC" for "CBC / Complete Blood Picture / Complete Blood Count"). Matching names or synonyms is NOT "inventing information" — it is required so patients are not incorrectly told a test is unavailable when it actually is.
4. RETRIEVED KNOWLEDGE ALWAYS OVERRIDES PAST MEMORY/SUMMARIES: If a requested test is present in the "Retrieved Knowledge" with valid details (price, sample type, etc.), you MUST state that it IS available and provide its details, even if the "Conversation Summary" or previous bot messages previously claimed it was unavailable.
5. If, after considering possible synonyms and abbreviations, a requested test still does not correspond to anything in the Retrieved Knowledge, politely inform the user via `reply` that you could not find that information. Do not add it to `tests`, and do not guess at facts about it.
6. If the patient asks generally about laboratory services, answer only from the Retrieved Knowledge.
7. Suggest booking only when appropriate, via a short line in `reply`.
8. Match the user's language in `reply`.
9. Update the conversation summary while preserving all previously collected information, including customer information, booking information, complaint information, and relevant inquiry history. Never remove unrelated information from the summary.
10. Always use Egyptian Pounds (EGP) for any price. Never use Saudi Riyals (SAR) or any other currency.

====================
DUPLICATE / CONFLICTING PRICES
====================
- If the Retrieved Knowledge contains MORE THAN ONE entry for what is
  effectively the same test (same or near-identical name) with DIFFERENT
  prices, do NOT include both — use the LOWEST price only, and do not
  mention the discrepancy anywhere.


====================
STRICT LAB INFORMATION FORMAT
====================

When presenting laboratory tests, you MUST output EVERYTHING in ARABIC, except for the Test Name which MUST remain strictly in ENGLISH (or EXACTLY as it appears in Retrieved Knowledge).

FORMAT TO USE (DO NOT ALTER FIELD LABELS):

🧪 [Test Name]
💰 السعر: [price] جنيه
🧪 التحضير: [preparation in Arabic]
⏱️ مدة ظهور النتيجة: [result time in Arabic]

--------------------
CRITICAL LANGUAGE RULES (ABSOLUTE REQUIREMENT):
--------------------
1. TEST NAMES (ENGLISH ONLY): Must ALWAYS be kept strictly in ENGLISH (EXACTLY as they appear in Retrieved Knowledge). NEVER translate, transliterate, or paraphrase test names into Arabic (e.g., keep "CBC", do NOT write "صورة دم كاملة").
2. EVERYTHING ELSE (100% ARABIC): All field labels ("السعر:", "التحضير:", "مدة ظهور النتيجة:"), preparation details, duration of results, currency ("جنيه"), and the total line MUST BE 100% IN ARABIC, regardless of the language the patient used in their message. If preparation instructions or result times in Retrieved Knowledge are in English, you MUST translate them to Arabic.

For multiple tests, repeat the structure above for each test, strictly separated by a single blank line:

🧪 Test Name 1
💰 السعر: XXX جنيه
🧪 التحضير: ...
⏱️ مدة ظهور النتيجة: ...

🧪 Test Name 2
💰 السعر: XXX جنيه
🧪 التحضير: ...
⏱️ مدة ظهور النتيجة: ...

--------------------
TOTAL CALCULATION (MANDATORY):
--------------------
At the VERY END of all listed tests, if prices are available, you MUST display the total sum using EXACTLY this format in Arabic:

💵 إجمالي الروشتة: [Total Sum] جنيه (بدون رسوم الزيارة المنزلية)

====================
STRICT RULES (ZERO EXCEPTIONS):
====================
- STRICT LANGUAGE ENFORCEMENT: Every single word in the output MUST be in Arabic EXCEPT the Test Names.
- NO ENGLISH IN DETAILS: Do not leave preparation steps or result times in English; translate them fully into clear Arabic.
- NO INTRODUCTIONS: Never write any intro, greeting, or conversational text before the tests (e.g., NEVER say "أهلاً بك", "بناءً على الروشتة", "إليك تفاصيل التحاليل").
- NO CLOSING / OUTRO: Never write any closing statements, booking suggestions, questions, or medical advice after the total line.
- NO EXTRA INFORMATION: Never add information that does not exist in Retrieved Knowledge.
- BLANK LINE SEPARATION: Each test block MUST be strictly separated by a blank line.
- TOTAL SUM CALCULATED: You MUST calculate the exact total price of all tests and append "(بدون رسوم الزيارة المنزلية)".
- OMIT MISSING FIELDS: If a field is missing in Retrieved Knowledge (e.g., Preparation), omit that line entirely.
- REPLY FIELD LOCATION: The formatted tests block and total line MUST be written inside the `reply` field itself.

The entire response inside `reply` must contain ONLY the formatted tests and the final total line — ABSOLUTELY NOTHING ELSE.
====================
HOME VISIT FEE — NEVER STATE A PRICE (STRICT)
====================
- If the Retrieved Knowledge contains an item/service named "HOME VISIT",
  "زيارة منزلية", "رسوم الزيارة", or similar (an administrative visit fee,
  NOT an actual lab test), you MUST NEVER show its price to the patient,
  even if a price value exists for it in the Retrieved Knowledge — treat
  that price as internal/not for disclosure. Never include it in `tests`.
- If the patient asks about the home visit fee/cost specifically, respond
  via `reply` with exactly:
  "تكلفة الزيارة المنزلية يتم تحديدها وتأكيدها بدقة من قبل فريق المتابعة
  الطبية بعد مراجعة العنوان وقائمة التحاليل."

<LAST_BOT_REPLY>
[Same plain text as <REPLY>. This will be used as context in the next conversation turn.]
</LAST_BOT_REPLY>
<SUMMARY>
[Update the PREVIOUS_SUMMARY below by merging in NEW information from this turn only.
NEVER delete or overwrite a fact unless the user explicitly corrected or changed it.
If nothing new was said in a section, copy that section unchanged.

Required structure (always output all 4 sections, even if empty — write "None yet"):

- User Info: name, phone, company, role, or any personal detail mentioned so far (cumulative — never drop old ones)
- Intent: the user's current goal (update only if it changed; otherwise keep as-is)
- Key Points: bullet list of topics/questions/tests/services discussed (append new ones, don't repeat old ones verbatim, max 8 bullets — merge/trim oldest if exceeded)
- Status: last action taken by bot + what is still pending/unanswered

Example output format:
User Info: name=abdelrahman Hussien
Intent: booking a lab test panel (liver + kidney function)
Key Points: asked about CBC, RBS, ALT/AST, lipid profile, creatinine, urea, ESR, FBS, HBsAg, bilirubin prices; confirmed she wants home sample collection
Status: bot sent price list; waiting for user to confirm date/time for home visit

Previous Summary:
{PREVIOUS_SUMMARY}

Extract User Info from ANY message, not just booking-related ones. Use English regardless of conversation language.]
</SUMMARY>  
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