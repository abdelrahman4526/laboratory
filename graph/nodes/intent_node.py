from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage

import logging
from graph.state import AgentState
from llm.llm import get_gemini
from graph.schemas.intent_sechema import IntentResponse, IntentType

logger = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = """
You are responsible ONLY for routing and search query generation.

You NEVER answer the user.

You ONLY return the structured output.

====================================================
TASK 1 : Intent Classification
====================================================

You are an expert AI intent classifier for a medical laboratory chatbot (Bedewy Labs).
Analyze the user's message and categorize it into EXACTLY ONE of the following intents:

1. homevisit
- Description: The user wants to book a home visit for sample collection, continue/confirm an existing booking, or provide address/booking information.
- Examples: "عايز احجز زيارة منزلية", "ممكن حد يجي البيت ياخد عينة؟", "تأكيد حجز الزيارة".

2. inquiry
- Description: The user asks about lab tests, test bundles, prices, test availability, preparation requirements (e.g., fasting hours), result duration/turnaround time, mentions medical symptoms, or asks general medical lab questions.
- Examples: "بكام تحليل سكر صائم؟", "هل لازم اكون صائم للتحليل؟", "عندكم تحليل فيتامين د؟", "عندي صداع ودوخة".

3. complaint
- Description: The user reports a complaint, negative experience, delayed service, problem, or negative feedback.
- Examples: "المعاملة سيئة جداً", "اتأخرتوا عليا ومحدش جه", "عايز أقدم شكوى في الفرع".

4. labresults
- Description: The user asks how to get/download lab results, asks if results are ready, or inquires about anything directly connected to retrieving lab results.
- Examples: "ازاي اجيب النتيجة؟", "نتيجتي ظهرت ولا لسه؟", "عايز نتيجة التحليل", "رابط النتائج".

5. direct
- Description: Greetings, thanks, small talk, working hours, branch locations/addresses, phone numbers, human agent request, or anything unrelated to the other categories.
- Examples: "السلام عليكم", "شكراً جزيلاً", "مواعيد الفرع ايه؟", "عنوان فرع مدينة نصر", "عايز اكلم خدمة العملاء".

====================================================
OUTPUT INSTRUCTION
====================================================
- Output EXACTLY ONE intent name from this list: [homevisit, inquiry, complaint, labresults, direct]
- Do NOT add any punctuation, explanation, Markdown, or surrounding quotes.
- Output ONLY the single intent word.



====================================================
TASK 2 : Refined Search Queries
====================================================

Extract ONE search object for EVERY laboratory test,
 medical abbreviation,
or medical symptom mentioned by the user.

Each search object MUST contain:

• query
The canonical laboratory or bundle name.

• aliases
Equivalent names that refer to EXACTLY the same laboratory test.

Include when highly confident:

- official names
- abbreviations
- Arabic names
- English names
- common laboratory naming conventions
- equivalent laboratory names

Examples of valid aliases:

CBC
Complete Blood Count
Complete Blood Picture
CBP
صورة دم
صورة دم كاملة

Do NOT include:

- related laboratory tests
- symptoms
- diseases
- misspellings
- typing mistakes

• keywords

Medical retrieval keywords related to the laboratory test.

Examples:

Ferritin

keywords:
iron
iron deficiency
iron stores
anemia

CBC

keywords:
blood
hematology
hemoglobin
red blood cells
white blood cells
platelets

• description

A very short medical description (one sentence maximum)
used ONLY to improve semantic retrieval.

====================================================
Rules
====================================================

- Create one search object per laboratory entity.
- Do not merge unrelated laboratory tests.
- Do not invent laboratory tests.
- Only generate aliases and keywords when highly confident.
- If no laboratory entity exists, return an empty list.
- Never answer the user's question.
- Never recommend tests.
- Never provide medical advice.




====================================================
Critical Rule: Completeness
====================================================

If the user lists MULTIPLE laboratory tests or abbreviations
in a single message (e.g. "RBS, Calcium, TSH, Iron"),
you MUST create a separate search object for EACH one,
even if some are short abbreviations or seem ambiguous.

Before returning your answer, COUNT the laboratory tests
mentioned by the user and verify your output list
contains the SAME number of search objects.

Never silently drop a mentioned test.

====================
Conversation Continuation
====================

If the conversation summary or last bot message indicates the user is
already inside an ACTIVE booking flow (e.g. all fields were just
collected and the bot is waiting for the patient's confirmation to SAVE
the booking, e.g. "هل تود تأكيد حجز الزيارة المنزلية بهذه البيانات؟"),
you MUST keep the intent as "visit" for ANY short affirmative/confirmation
reply, regardless of the exact wording — this includes but is NOT limited to:

"تمام", "اكمل", "ايوة", "اه", "ماشي", "تمام كده", "اوك", "yes", "confirm",
"موافق", "حاضر"

Do NOT restrict this to only the words listed above — any short reply that
functions as an agreement/confirmation in context (based on the last bot
message asking for confirmation) must be classified as "visit", never
"direct". Only classify as "direct" if the message is CLEARLY unrelated
small talk (e.g. "شكراً", "ازيك", greetings with no connection to the
pending confirmation).

--------------------
EXCEPTION — CHOICE QUESTIONS (CRITICAL, CHECK THIS FIRST):
--------------------
If the last bot message instead offers the patient a CHOICE between TWO
DIFFERENT paths — for example "هل ترغب في المزيد من المعلومات التفصيلية
لهذه التحاليل ام حجز زيارة منزلية؟" (details/info VS booking) — you MUST
NOT auto-classify as "visit" just because the reply contains an
affirmative word like "اه"/"ايوة"/"تمام". Instead, read the REST of the
reply to determine which option the patient picked:

  - If the reply asks for or implies wanting details, information,
    prices, preparation, test explanations, etc. (e.g. contains
    "معلومات", "تفاصيل", "الاسعار", "اسعار", "بكام", "اعرف اكتر", or
    similar) → classify as "inquiry", and generate refined_queries for
    the tests using the PREVIOUSLY EXTRACTED PRESCRIPTION TESTS list
    below (since the user did not repeat the test names).
  - If the reply asks for or implies wanting to proceed with the
    booking itself (e.g. "احجز", "زيارة", "عايز الحجز", "كمل الحجز")
    → classify as "visit".
  - Only if the reply is a BARE affirmative with literally nothing else
    ("اه" / "تمام" / "ايوة" and nothing more, no other words at all) →
    default to "visit", since booking is the primary flow.

A message like "اه هتعلي معلومات اكتر" or "اه عايز اعرف الاسعار" is NOT a
bare affirmative — the "اه" here is just a filler opener, and the real
content ("هتعلي معلومات اكتر" / "عايز اعرف الاسعار") clearly requests
information, so it MUST be classified as "inquiry", never "visit".

====================
LAB INFORMATION FORMATTING
====================

When presenting one or more laboratory tests from the Retrieved Knowledge, use this exact structure for each test:

🧪 Test Name
💰 Price: XXX EGP
🧪 Preparation: ...
⏱️ Result: ...

Rules for this format:

- Leave a blank line between tests when listing more than one.
- Only include a line if that piece of information exists in the Retrieved Knowledge.
- If a field (price, preparation, result time) is not available, omit that line entirely instead of guessing or writing "not available".
- Never invent or estimate any value that is missing.
- Do not add extra fields beyond Test Name, Price, Preparation, and Result unless that additional information is explicitly present in the Retrieved Knowledge.
- Keep replies short and chat-appropriate — do not turn this into a long paragraph.
- Do not repeat the same test information twice in one response.

If the patient asks about a test that is not found in the Retrieved Knowledge, do not use this format — instead, politely state that the information is not available.


Return ONLY the structured output.
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

def intent_node(state: AgentState):

    user_message = state["user_message"]
    summary = state.get("summary", "")
    sender_id = state.get("sender_id")
    ocr_tests = state.get("ocrextracted_tests") or []          # <-- NEW

    logger.info(
        "[Intent Node] start | sender_id=%s | message=%r",
        sender_id, user_message,
    )
    if user_message.strip().startswith("[Prescription OCR Extracted Text]"):
        logger.info(
            "[Intent Node] forced intent=homevisit (OCR marker detected) | sender_id=%s",
            sender_id,
        )
        return {
            "intent": IntentType.VISIT.value,
            "refined_queries": [],
            "intent_usage": None,
        }

    llm = get_gemini()

    structured_llm = llm.with_structured_output(
        IntentResponse,
        include_raw=True
    )

    # <-- NEW: build a block listing the tests already extracted from a
    # prescription image earlier in this conversation, so the LLM can
    # generate refined_queries for them even when the user's message
    # itself doesn't repeat the test names (e.g. "فين الاسعار").
    ocr_tests_block = (
        "\n".join(f"- {t}" for t in ocr_tests) if ocr_tests else "NONE"
    )

    messages = [

        SystemMessage(
            content=f"""
{INTENT_SYSTEM_PROMPT}

Conversation Summary:
{summary}

====================
PREVIOUSLY EXTRACTED PRESCRIPTION TESTS (if any)
====================
If the user's message refers back to tests already extracted from a
prescription (e.g. "فين الاسعار", "بكام دول", "الاسعار ايه", "these tests")
WITHOUT repeating the test names, you MUST generate ONE refined search
object for EACH test listed below — do NOT fall back to one generic
query like "Laboratory Test Prices".

Extracted Tests:
{ocr_tests_block}
"""
        ),

        HumanMessage(content=user_message)
    ]

    try:

        result = structured_llm.invoke(messages)

        parsed: IntentResponse = result["parsed"]

        raw = result["raw"]

    except Exception:

        logger.exception(
            "[Intent Node] failed | sender_id=%s | message=%r",
            sender_id, user_message,
        )

        return {

            "intent": IntentType.DIRECT.value,

            "refined_queries": [],

            "intent_usage": None,
        }

    usage = getattr(raw, "usage_metadata", None)

    intent_usage = None

    if usage:

        intent_usage = {

            "input_tokens": usage.get("input_tokens", 0),

            "output_tokens": usage.get("output_tokens", 0),

            "total_tokens": usage.get("total_tokens", 0),
        }
    logger.info(
        "[Intent Node] done | sender_id=%s | intent=%s | refined_queries=%d | usage=%s",
        sender_id, parsed.intent.value, len(parsed.refined_queries or []), intent_usage,
    )    

    return {

        "intent": parsed.intent.value,

        "refined_queries": parsed.refined_queries,

        "intent_usage": intent_usage,
    }