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

Choose exactly ONE intent.

homevisit
The user wants to book ahomevisit, continue a booking, confirm a booking,
or provide booking information.

inquiry
The user asks about laboratory tests, bundles, prices,
availability, preparation, result duration,
mentions medical symptoms,
or asks medical laboratory questions.

complaint
The user reports a complaint, negative experience,
problem or feedback.

direct
Greetings, thanks, small talk,
working hours, location,
phone numbers,
or anything unrelated to laboratory retrieval.

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

====================================================

Return ONLY the structured output.
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