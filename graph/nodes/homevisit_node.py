from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from graph.nodes.homevisit_tools import save_visit_tool
from graph.schemas.homevisit_schema import HomevisitResponse
from graph.state import AgentState
from graph.utils import detect_language_fallback, generate_booking_pdf
from llm.llm import get_gemini
from software_service.client_services import ClientService
from software_service.homevisit_service import homevisitService

BOOKING_SYSTEM_PROMPT = """
You are an expert, empathetic, and professional AI Assistant for a Medical Laboratory specializing in Home Visit Sample Collection (خدمة الزيارات المنزلية لسحب العينات).

Your primary task is to assist patients in booking a Home Visit by collecting 5 required fields, processing prescription OCR data, and answering inquiries accurately.

====================
REQUIRED FIELDS
====================
1. name: Patient's Full Name - must be the full name including at least 4 parts (First, Father, Grandfather, Family name) as commonly used in Egypt (الاسم رباعي، يعني لازم يكتب الاسم كامل مكون من 4 أجزاء على الأقل زي الاسم الأول واسم الأب والجد والعائلة))
2. phone: Contact Phone Number (رقم الهاتف)
3. address: Detailed Home Address (العنوان التفصيلي للزيارة المنزلية)
4. details: List of required medical tests/analyses (التحاليل الطبية المطلوبة)
5. date: Preferred Appointment Date  (اليوم المطلوب للزيارة)
6. time: Preferred Appointment Time - extract in 24-hour format (HH:MM), and make sure to correctly determine AM/PM whether the time is written in Arabic or English, numeric or worded (e.g. "الساعة 4 العصر", "10 صباحاً", "3pm", "٥ بعد الضهر") (الوقت المطلوب للزيارة)
================================================================================
HANDLING AN EXISTING REQUEST (see EXISTING_BOOKING section below) — ACTION FIELD
================================================================================
If an EXISTING_BOOKING is present, you MUST set the `action` field in your
response to one of: "new", based on the user's latest message:


1. action = "reschedule" — The user wants to CHANGE / RESCHEDULE their existing
   meeting .
   You may reuse name/phone from EXISTING_BOOKING if the user doesn't repeat
   them, but always collect the NEW name /phone/topic for this request. Proceed
   like a normal booking flow (ask for missing fields, confirm, then set
   ready_to_save=true once confirmed). This will UPDATE the existing booking
   in place — it will NOT create a separate new booking.   

2. action = "new" — The user wants to submit a NEW / SEPARATE request.
   This is the DEFAULT whenever the user has explicitly initiated a booking
   in the CURRENT conversation (e.g. said "عايز احجز", "عايز اعرف عن خدمة معينة",
   or agreed to schedule a meeting after being offered one) — treat this as
   "new" automatically WITHOUT asking the user to disambiguate, even if an
   EXISTING_BOOKING from a previous, unrelated conversation exists. Proceed
   like a normal flow (ask for missing fields, confirm, then set
   ready_to_save=true once confirmed).

3. Only ask the user to disambiguate ("new" vs "info_only") ONCE, and ONLY if
   the user's very first message in this flow is itself vague (e.g. just
   "اه"/"تمام" with zero context about wanting a new request or an update).
   NEVER ask this disambiguation question more than once in the same
   conversation. If you already asked it and the user's reply is still a
   plain agreement word ("اه", "ايوة", "تمام", "ok"), default to action="new"
   and move on to collecting the next missing field (name, phone, or topic) —
   do NOT repeat the disambiguation question a second time.

====================
CRITICAL RULES & BEHAVIOR
====================

1. PRESCRIPTION & OCR EXTRACTED TESTS RULE:
   - When prescription data (OCR extracted text/tests) is provided in the context:
     a) Politely inform the patient: "تمت قراءة الروشتة بنجاح! التحاليل المستخرجة هي: [List extracted tests]."
     b) Ask the patient: "هل ترغب في حجز هذه التحاليل المستخرجة لزيارتك المنزلية؟"
     c) If the patient agrees (e.g., "نعم", "اه", "تمام", "أيوه", "أكدها"), automatically set the 'details' field to these extracted tests and move on to collect the next missing field (name, phone, address, or date).

2. HOME VISIT PRICING RULE (STRICT):
   - Never invent, estimate, or hallucinate prices, transport fees, or test durations.
   - If the "Matched Service" context contains explicit price information, state it directly to the patient.
   - If "Matched Service" DOES NOT contain price info, politely inform the patient:
     "تكلفة التحاليل ورسوم الزيارة المنزلية يتم تحديدها وتأكيدها بدقة من قبل فريق المتابعة الطبية بعد مراجعة العنوان وقائمة التحاليل."
   - Do NOT say "I will check and confirm" if you don't have the price. Proceed directly to collecting missing fields in the same reply.

3. CONVERSATION FLOW:
   - Match the patient's language and tone (default to polite Arabic).
   - Ask for ONE missing field at a time to keep the conversation simple and clear.
   - Never re-ask for a field that has already been collected.

4. CONFIRMATION & SAVING LOGIC:
   - Step 1: Collect all 5 required fields (name, phone, address, details, date).
   - Step 2: Once ALL 5 fields are collected, present a clear, formatted summary of the collected details to the patient and explicitly ask:
     "هل تود تأكيد حجز الزيارة المنزلية بهذه البيانات؟"
   - Set ready_to_save = true ONLY when all 5 required fields are present.
   - Set confirmed = true ONLY when the patient explicitly replies with a confirmation (e.g., "نعم", "تأكيد", "تمام", "موافق").

====================
STRICT LAB INFORMATION FORMAT
====================

When presenting laboratory tests, you MUST use ONLY this format:

🧪 [Test Name]

💰 Price: [price] EGP

🧪 Preparation: [preparation]

⏱️ Result: [result time]


Rules:

- Never write an introduction before the tests.
- Never write a conclusion after the tests.
- Never use paragraphs describing multiple tests.
- Never say "بناءً على الروشتة" or similar phrases.
- Never add recommendations, questions, or booking suggestions.
- Never add information that does not exist in Retrieved Knowledge.
- Never combine multiple tests in one sentence.
- Each test must be separated by a blank line.

For multiple tests, repeat the same structure:

🧪 Test Name
💰 Price: XXX EGP
🧪 Preparation: ...
⏱️ Result: ...

🧪 Test Name
💰 Price: XXX EGP
🧪 Preparation: ...
⏱️ Result: ...

Only include fields available in Retrieved Knowledge.
If a field is missing, remove the entire line.

The response must contain ONLY the formatted laboratory information.
====================
OUTPUT STRUCTURE (JSON)
====================
Always return your response with the following JSON structure alongside your friendly user reply:

{
  "reply": "Your polite response text to the user here...",
  "collected_fields": {
    "name": "Collected Name or null",
    "phone": "Collected Phone or null",
    "address": "Collected Address or null",
    "details": "Collected Tests or null",
    "date": "Collected Date or null",
    "time":"Collected time or null"
  },
  "ready_to_save": false,
  "confirmed": false
}
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


def visit_node(state: AgentState) -> dict:
    page_id          = state.get("page_id")
    sender_id        = state.get("sender_id")
    platform_id      = state.get("platform_id")
    user_message     = state["user_message"]
    current_summary  = state.get("summary") or ""
    last_bot_message = state.get("last_bot_message") or ""
    matched_context  = state.get("rag_context") or ""

    now = datetime.now()
    current_time_info = now.strftime("Today is %A, %B %d, %Y. Current time is %I:%M %p")

    # ── existing booking ── لازم تتجاب هنا، قبل الـ LLM call ─────────────
    existing_booking = homevisitService.get_latest_booking(sender_id, page_id)

    existing_booking_context = (
        f"""
====================
EXISTING_BOOKING (آخر حجز مسجل لهذا العميل)
====================
Reference: {existing_booking.reference_id}
Name: {existing_booking.name}
Phone: {existing_booking.phone_number}
Address: {existing_booking.address}
Details: {existing_booking.details}
Date: {existing_booking.date}
time :{existing_booking.time}
Status: {existing_booking.status}
"""
        if existing_booking else
        "\n====================\nEXISTING_BOOKING\n====================\n(لا يوجد حجز سابق لهذا العميل)\n"
    )

    llm            = get_gemini()
    structured_llm = llm.with_structured_output(HomevisitResponse, include_raw=True)

    system_prompt = f"""
{BOOKING_SYSTEM_PROMPT}

====================
CRITICAL: CURRENT TEMPORAL CONTEXT
====================
{current_time_info}

====================
VERIFIED LAB INFORMATION
====================
{matched_context or "(No matching laboratory test found. Do not invent prices or medical information.)"}
{existing_booking_context}
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

    # ── LLM call ──
    try:
        result       = structured_llm.invoke(messages)
        parsed       = result["parsed"]
        raw_response = result["raw"]
        print(
        f"[Booking Node] parsed | ready_to_save={parsed.ready_to_save} "
        f"| confirmed={parsed.confirmed} | action={parsed.action} "
        f"| visit={parsed.visit.model_dump(exclude_none=True)}"
    )   

    # existing_booking already fetched above — شيل السطر بتاعها من هنا لو كان موجود 
    except Exception as e:
        print(f"[Booking Node] LLM error: {e}")
        fallback = detect_language_fallback(
            user_message,
            arabic="عذرًا، حدث خطأ مؤقت. حاول مرة أخرى.",
            default="Sorry, a temporary error occurred. Please try again.",
        )
        return {
            "response":         fallback,
            "summary":          current_summary,
            "last_bot_message": fallback,
            "booking_saved":    False,
            "booking_reference": None,
            "booking_pdf":      None,
            "booking_usage":    None,
        }
 
    # ── usage ─────────────────────────────────────────────────────────────────
    usage = getattr(raw_response, "usage_metadata", None)
    booking_usage = (
        {
            "input_tokens":  usage.get("input_tokens",  0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens":  usage.get("total_tokens",  0),
        }
        if usage
        else None
    )
 
    visit_data = parsed.visit.model_dump(exclude_none=True)
 
  
 
    required_fields    = ["name", "phone_number", "details", "date","address","time"]
    all_fields_present = all(
    visit_data.get(f)
    for f in required_fields
)
 

    # ── existing booking ────────────────────────────────────────────────────
      # أو مثلاً: ClientService.get_last_visit(sender_id, page_id, platform_id)

    visit_saved     = False
    visit_reference = None
    booking_pdf     = None  # الـPDF بيتبعت بس لما الحالة تتغير لـ Attended من الداشبورد

    # ── save / reschedule ────────────────────────────────────────────────────
    if parsed.ready_to_save and parsed.confirmed and all_fields_present:

        # حماية من التكرار: لو نفس بيانات آخر حجز موجودة بالفعل ومفيش طلب "new" صريح
        if (
            existing_booking
            and parsed.action != "new"
            and existing_booking.name == visit_data.get("name")
            and existing_booking.phone_number == visit_data.get("phone_number")
            and existing_booking.address == visit_data.get("address")
            and existing_booking.details == visit_data.get("details")
            and existing_booking.date == visit_data.get("date")
            and existing_booking.date == visit_data.get("time") 
        ):
            clean_reply     = parsed.reply
            visit_saved     = True
            visit_reference = existing_booking.reference_id

        else:
            try:
                if parsed.action == "reschedule" and existing_booking:
                    result = homevisitService.update_visit(
                        visit_id=existing_booking.id,
                        name=visit_data.get("name"),
                        phone_number=visit_data.get("phone_number"),
                        date=visit_data.get("date"),
                        details=visit_data.get("details"),
                        address=visit_data.get("address"),
                        time=visit_data.get("time"),
                    )

                    if result.success and result.visit:
                        visit_saved     = True
                        visit_reference = result.visit.reference_id
                        clean_reply = detect_language_fallback(
                            user_message,
                            arabic=(
                                f"تم تعديل موعد الزيارة بنجاح ✅\n"
                                f"رقم الطلب: *{visit_reference}*\n"
                                f"هيتم تأكيد الموعد الجديد معاك من فريقنا قريبًا."
                            ),
                            default=(
                                f"Your visit has been rescheduled successfully ✅\n"
                                f"Reference: *{visit_reference}*\n"
                                f"Our team will confirm the new time with you shortly."
                            ),
                        )
                    else:
                        raise ValueError(result.message)
                else:
                    result = save_visit_tool.invoke(input={
                        **visit_data,
                        "comes_from": f"Facebook:{sender_id}:{page_id}",
                    })

                    if result.success and result.visit:
                        visit_saved     = True
                        visit_reference = result.visit.reference_id
                        clean_reply = detect_language_fallback(
                            user_message,
                            arabic=(
                                f"تم استلام طلب الحجز الخاص بيك ✅\n"
                                f"رقم الطلب: *{visit_reference}*\n"
                                f"هيتم تأكيد الحجز معاك من فريقنا قريبًا."
                            ),
                            default=(
                                f"Your booking request has been received ✅\n"
                                f"Reference: *{visit_reference}*\n"
                                f"Our team will confirm it with you shortly."
                            ),
                        )
                    else:
                        raise ValueError(result.message)

            except Exception as e:
                print(f"[homevisit Node] Tool error: {e}")
                visit_saved = False
                clean_reply = detect_language_fallback(
                    user_message,
                    arabic="حدث خطأ أثناء حفظ الحجز. حاول مرة أخرى.",
                    default="An error occurred while saving your booking. Please try again.",
                )
    else:
        clean_reply = parsed.reply
 
    # ── persist client state ──────────────────────────────────────────────────
    try:
        ClientService.update_client_summary_and_last_bot_message(
            sender_id=sender_id,
            page_id=page_id,
            platform_id=platform_id,
            summary=parsed.summary,
            last_bot_message=clean_reply,
        )
    except Exception as e:
        print(f"[homevisit Node] Persist error: {e}")
 
    print(
        f"[Booking Node] done | saved={visit_saved} "
        f"| ref={visit_reference} | usage={booking_usage}"
    )
 
    return {
        "response":          clean_reply,
        "summary":           parsed.summary,
        "last_bot_message":  clean_reply,
        "visit_saved":     visit_saved,
        "visit_reference":visit_reference,
        "booking_pdf":       booking_pdf,
        "booking_usage":     booking_usage,
    }
 



























