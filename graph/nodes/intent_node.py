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

====================================================
Conversation Continuation
====================================================

If the conversation summary indicates the user is
already inside a booking flow,

keep the intent as booking,

even if the latest message is short, such as:

"تمام"

"اكمل"

"ايوة"

"yes"

"confirm"


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
[Cumulative conversation summary. STRICT RULES:
1. Build on the previous summary — copy it first, then update only what changed.
2. Always capture in this structure:
   - User Info: any personal details mentioned (name, phone, company, role, etc.)
   - Intent: what the user is trying to accomplish
   - Key Points: important topics, questions, or concerns raised
   - Status: what just happened + what is still pending
3. Extract User Info from ANY message — not just booking context.
4. Use English regardless of conversation language.]
</SUMMARY>
"""

def intent_node(state: AgentState):

    user_message = state["user_message"]
    summary = state.get("summary", "")
    sender_id = state.get("sender_id")
 
    logger.info(
        "[Intent Node] start | sender_id=%s | message=%r",
        sender_id, user_message,
    )

    llm = get_gemini()

    structured_llm = llm.with_structured_output(
        IntentResponse,
        include_raw=True
    )

    messages = [

        SystemMessage(
            content=f"""
{INTENT_SYSTEM_PROMPT}

Conversation Summary:
{summary}
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