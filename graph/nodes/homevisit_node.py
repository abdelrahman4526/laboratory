from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from graph.nodes.homevisit_tools import save_visit_tool
from graph.schemas.homevisit_schema import HomevisitResponse
from graph.state import AgentState
from graph.utils import detect_language_fallback, generate_booking_pdf, extract_ocr_tests
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
PRESCRIPTION OCR STATE — LOCKED CONTEXT (NEW)
================================================================================
The application layer (not you) is responsible for running OCR on any prescription
image ONCE, and then passing you the result as a fixed context block on every
subsequent turn, in this shape:
 
<EXTRACTED_TESTS>
[list of test names extracted from the prescription image, exactly as read, or
"NONE" if no prescription has been successfully read yet]
</EXTRACTED_TESTS>
 
Rules for this block:
- If <EXTRACTED_TESTS> is present and is NOT "NONE", this means a prescription
  image WAS already successfully read earlier in this conversation. You MUST
  treat it as ground truth for the rest of the conversation:
    * NEVER re-evaluate, question, or reject the clarity/validity of the
      prescription image again in this conversation.
    * NEVER say the image is unclear/invalid if <EXTRACTED_TESTS> is populated
      — that message is ONLY for the turn where OCR genuinely failed and
      <EXTRACTED_TESTS> is "NONE".
    * When the patient confirms they want these tests, copy the list from
      <EXTRACTED_TESTS> into `details` VERBATIM. Do not paraphrase, shorten,
      reorder, add, or drop any item.
- If the patient's confirmed test list needs to be referenced again later in
  the conversation (pricing, summary, final confirmation), always pull it from
  <EXTRACTED_TESTS> / the already-collected `details` field — never
  reconstruct it from memory of the conversation or from the SUMMARY.
 
====================
ANTI-HALLUCINATION RULE FOR TEST NAMES (NEW — STRICT)
====================
- You may ONLY state, price, or describe a test that appears EITHER:
    a) verbatim (or as an unambiguous exact synonym) in <EXTRACTED_TESTS> /
       the patient's own message, AND
    b) has a matching entry in Retrieved Knowledge / "Matched Service".
- If a test mentioned by the patient or extracted by OCR has NO confident
  exact match in Retrieved Knowledge, DO NOT substitute it with the closest
  or most similar-sounding test from Retrieved Knowledge. Instead say so
  explicitly and ask the patient to confirm/clarify the test name, e.g.:
  "لم أتمكن من إيجاد تحليل مطابق باسم '[X]' في قاعدة البيانات، هل يمكنك تأكيد
  الاسم أو كتابته بشكل مختلف؟"
- Never output a test name in the STRICT LAB INFORMATION FORMAT that is not
  either explicitly requested by the patient or explicitly present in
  <EXTRACTED_TESTS>. Under no circumstances show unrelated tests "in case
  they're relevant."
 

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
   - When prescription data (OCR extracted text/tests) is provided in the
     context, the application (not you) already shows the patient the
     extracted test list in a fixed format BEFORE your reply. In `reply`,
     write ONLY a short line asking: "هل ترغب في حجز هذه التحاليل المستخرجة
     لزيارتك المنزلية؟" — do NOT list the test names yourself, do NOT write
     "تمت قراءة الروشتة بنجاح" or repeat the test list; the application
     already displays it.
   - If the patient agrees (e.g., "نعم", "اه", "تمام", "أيوه", "أكدها"),
     automatically set the 'details' field to these extracted tests and
     move on to collect the next missing field (name, phone, address, or
     date).
2. HOME VISIT PRICING RULE (STRICT):

   a) TEST PRICES (per lab test):
      - Never invent, estimate, or hallucinate prices.
      - If the "Matched Service" context contains explicit price info for a
        specific TEST, state it directly using the STRICT LAB INFORMATION
        FORMAT below.

   b) HOME VISIT FEE / رسوم الزيارة المنزلية (STRICT — NO EXCEPTIONS):
      - If the patient asks about the home visit fee, transportation fee,
        "رسوم الزيارة", "سعر الزيارة المنزلية", "كام تمن الزيارة", or anything
        referring to the cost of the VISIT ITSELF (not a specific lab test),
        you MUST NEVER state a number or amount for it — even if a visit fee
        number happens to appear anywhere in the Matched Service or context.
      - ALWAYS respond with exactly this message instead:
        "تكلفة الزيارة المنزلية يتم تحديدها وتأكيدها بدقة من قبل فريق المتابعة
        الطبية بعد مراجعة العنوان وقائمة التحاليل."
      - Do NOT say "هتأكد وأرجعلك" — go directly to collecting the next
        missing field in the same reply.
      - This rule applies REGARDLESS of what other pricing info exists in
        context. Visit fee = never a number. Test price = only from Matched
        Service, per rule (a).

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
LAB TEST OUTPUT (STRICT)
====================
Whenever you present ANY laboratory test details (price, preparation, result
time) to the patient, do NOT write them as text in `reply`. Instead, add one
entry per test to the `tests` field — the application renders the visible
format, not you.

- Do NOT include the "HOME VISIT" / administrative visit-fee item in `tests`
  — only actual lab tests (see HOME VISIT PRICING RULE b above).
- Only include a field (price/preparation/result_time) if that fact exists
  in Verified Lab Information — leave it null otherwise, never invent.
- If the patient provides more than one lab/test, populate `total_price`
  with the SUM OF TEST PRICES ONLY — never the home visit fee — and only
  once every requested test has a verified price; otherwise leave it null.
- `reply` should contain only the surrounding conversational text (e.g.
  asking for the next missing field, or the confirmation question) — never
  test names, prices, preparation, or result times.

-If the user provides more than one lab/test, give the total price of tests
after the formatted block (outside the strict format), as a separate short
line: " الإجمالي بدون رسوم الزياره المنزليه: [total] جنيه". This total must be the SUM OF TEST PRICES
ONLY — do NOT add, include, or factor in the home visit fee under any
circumstances, even if a home visit was requested or mentioned (see HOME
VISIT PRICING RULE b above, which is never a fixed number and must stay
completely separate). When you calculate this total, also populate
booking.total_price with the same number (plain number, no currency symbol,
tests only — excluding home visit fee). Only set it once all requested
tests have verified prices — leave it empty otherwise. 

====================
STRICT LAB INFORMATION FORMAT
====================
 
When presenting laboratory tests, you MUST use ONLY this format, in
Arabic, regardless of the language the patient wrote in:
 
🧪 [Test Name] (EXACTLY as it appears in Retrieved Knowledge — English,
do NOT translate the test name)
💰 السعر: [price] جنيه
🧪 التحضير: [preparation]
⏱️ مدة ظهور النتيجة: [result time]
 
he test name itself is ALWAYS kept in English exactly as it appears in
Retrieved Knowledge — never translate, paraphrase, or alter it. Only the
field labels (السعر / التحضير / مدة ظهور النتيجة) and the total line are
in Arabic.

For multiple tests, repeat the structure above for each test, separated
by a blank line:

🧪 Test Name
💰 السعر: XXX جنيه
🧪 التحضير: ...
⏱️ مدة ظهور النتيجة: ...
 
🧪 Test Name
💰 السعر: XXX جنيه
🧪 التحضير: ...
⏱️ مدة ظهور النتيجة: ...
 
--------------------
TOTAL CALCULATION (MANDATORY):
--------------------
At the VERY END of all listed tests, if prices are available, you MUST display the total sum using EXACTLY this format:
 
💵 إجمالي الروشتة: [Total Sum] جنيه (بدون رسوم الزيارة المنزلية)
====================
STRICT RULES:
====================
- Never write any introduction before the tests.
- Never write any conclusion, closing statements, or conversational text after the total.
- Never say "بناءً على الروشتة" or any introductory greetings/phrases.
- Never add recommendations, extra advice, questions, or booking suggestions.
- Never add information that does not exist in Retrieved Knowledge.
- Never combine multiple tests in one sentence or paragraph.
- Each test must be strictly separated by a blank line.
- You MUST calculate the total price of all listed tests and append the text "(بدون رسوم الزيارة المنزلية)".
- Only include fields available in Retrieved Knowledge for each test. If a field is missing (e.g. Preparation), omit that line.

The entire response must contain ONLY the formatted tests and the final total line—ABSOLUTELY NOTHING ELSE.
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


def visit_node(state: AgentState) -> dict:
    page_id = state.get("page_id")
    sender_id = state.get("sender_id")
    platform_id = state.get("platform_id")
    user_message = state["user_message"]
    current_summary = state.get("summary") or ""
    last_bot_message = state.get("last_bot_message") or ""
    matched_context = state.get("rag_context") or ""
    ocr_tests = state.get("ocrextracted_tests") or extract_ocr_tests(current_summary)
    now = datetime.now()
    current_time_info = now.strftime("Today is %A, %B %d, %Y. Current time is %I:%M %p")

    # ── existing booking ── لازم تتجاب هنا، قبل الـ LLM call ─────────────
    current_draft = state.get("visit") or {}
    draft_block = (
        "\n".join(f"- {k}: {v}" for k, v in current_draft.items() if v)
        if current_draft
        else "لا يوجد بيانات محجوزة بعد لهذا الحجز الحالي"
    )


    existing_booking = homevisitService.get_latest_booking(sender_id, page_id)
    extracted_tests_block = (
        "\n".join(f"- {t}" for t in ocr_tests) if ocr_tests else "NONE"
    ) 
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
        if existing_booking
        else "\n====================\nEXISTING_BOOKING\n====================\n(لا يوجد حجز سابق لهذا العميل)\n"
    )

    llm = get_gemini()
    structured_llm = llm.with_structured_output(HomevisitResponse, include_raw=True)

    system_prompt = f"""
{BOOKING_SYSTEM_PROMPT}

====================
CRITICAL: CURRENT TEMPORAL CONTEXT
====================
{current_time_info}


====================
CURRENT IN-PROGRESS BOOKING DRAFT (الحجز الحالي الجاري تجميعه في هذه المحادثة الآن —
هذا هو المصدر الأدق لأي بيانات سبق جمعها لهذا الحجز، اعتمد عليه أولاً قبل أي مصدر آخر)
====================
{draft_block}

====================
PRESCRIPTION OCR STATE
====================
{extracted_tests_block}

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
        result = structured_llm.invoke(messages)
        parsed = result["parsed"]
        raw_response = result["raw"]
        print(
            f"[Booking Node] parsed | ready_to_save={parsed.ready_to_save} "
            f"| confirmed={parsed.confirmed} | action={parsed.action} "
            f"| visit={parsed.visit.model_dump(exclude_none=True)}"
        )

    except Exception as e:
        print(f"[Booking Node] LLM error: {e}")
        fallback = detect_language_fallback(
            user_message,
            arabic="عذرًا، حدث خطأ مؤقت. حاول مرة أخرى.",
            default="Sorry, a temporary error occurred. Please try again.",
        )

        try:
            ClientService.update_client_summary_and_last_bot_message(
                sender_id=sender_id,
                page_id=page_id,
                platform_id=platform_id,
                summary=current_summary,
                last_bot_message=fallback,
            )
        except Exception as persist_err:
            print(f"[Booking Node] Persist error on fallback: {persist_err}")

        return {
            "response": fallback,
            "summary": current_summary,
            "last_bot_message": fallback,
            "booking_saved": False,
            "visit": current_draft,
            "booking_reference": None,
            "booking_pdf": None,
            "booking_usage": None,
        }

    # ── usage ─────────────────────────────────────────────────────────────────
    usage = getattr(raw_response, "usage_metadata", None)
    booking_usage = (
        {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        if usage
        else None
    )

    new_visit_data = parsed.visit.model_dump(exclude_none=True)

    
    visit_data = {**current_draft, **{k: v for k, v in new_visit_data.items() if v}}

    required_fields = ["name", "phone_number", "details", "date", "address", "time"]
    all_fields_present = all(
        visit_data.get(f)
        for f in required_fields
    )

    visit_saved = False
    visit_reference = None
    booking_pdf = None  # الـPDF بيتبعت بس لما الحالة تتغير لـ Attended من الداشبورد

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
            and existing_booking.time == visit_data.get("time")
        ):
            clean_reply = parsed.reply
            visit_saved = True
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
                        branch_id=state.get("branch_id"),
                    )

                    if result.success and result.visit:
                        visit_saved = True
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
                        "comes_from": f"{state.get('platform_name') or 'Facebook'}:{sender_id}:{page_id}",
                        "branch_id": state.get("branch_id"), 
                    })

                    if result.success and result.visit:
                        visit_saved = True
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
            clear_ocr_marker=visit_saved,
        )
    except Exception as e:
        print(f"[homevisit Node] Persist error: {e}")

    print(
        f"[Booking Node] done | saved={visit_saved} "
        f"| ref={visit_reference} | usage={booking_usage}"
    )

    return {
        "response": clean_reply,
        "summary": parsed.summary,
        "last_bot_message": clean_reply,
        "visit": {} if visit_saved else visit_data,
        "visit_saved": visit_saved,
        "visit_reference": visit_reference,
        "booking_pdf": booking_pdf,
        "booking_usage": booking_usage,
    }