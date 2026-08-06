from langchain_core.messages import AIMessage

from graph.state import AgentState
from software_service.client_services import ClientService


def result_node(state: AgentState) -> dict:
    page_id = state.get("page_id")
    sender_id = state.get("sender_id")
    platform_id = state.get("platform_id")

    current_summary = state.get("summary") or ""

    result_text = """
📋 طريقة الحصول على نتيجة التحاليل

1. ادخل على الرابط:
https://results.bedewylabs.com/Login/

2. اكتب رقم المستخدم (User Name) الموجود أسفل الإيصال.

3. اكتب كلمة المرور (Password) الموجودة بجواره.

4. اضغط على زر Login.

✅ ستظهر النتيجة فور اعتمادها من المعمل.

📌 ملاحظة:
• يتم تسليم النتائج بدايةً من الساعة 9:00 مساءً في اليوم المحدد بالإيصال.
• يمكنك أيضًا مسح الـ QR Code الموجود على الإيصال للوصول إلى النتيجة مباشرة.
""".strip()

    try:
        ClientService.update_client_summary_and_last_bot_message(
            sender_id=sender_id,
            page_id=page_id,
            platform_id=platform_id,
            summary=current_summary,
            last_bot_message=result_text,
        )

    except Exception as e:
        print(f"[Result Node] Persist Error: {e}")

    return {
        "response": result_text,
        "summary": current_summary,
        "last_bot_message": result_text,
    }