import logging
import os
import uuid
import requests
from graph.graph import get_agent_graph
from graph.agent_response import AgentResponse
from graph.utils import count_request
from software_service.client_services import ClientService
from ocr.processor import process_prescription_ocr

logger = logging.getLogger(__name__)


class IncomingMessage:
    def __init__(self, sender_id, page_id, platform_id, msg_type,
                 text=None, media=None, platform_name=None):
        self.sender_id     = sender_id
        self.page_id       = page_id
        self.platform_id   = platform_id
        self.platform_name = platform_name
        self.type          = msg_type
        self.text          = text
        self.media         = media


INPUT_COST_PER_TOKEN = 0.075 / 1_000_000
OUTPUT_COST_PER_TOKEN = 0.300 / 1_000_000


def _calc_total_usage(result: dict, ocr_usage: dict = None) -> dict:
    nodes = [
        ("ocr_vision_usage", ocr_usage),
        ("intent_usage", result.get("intent_usage")),
        ("lab_info_usage", result.get("lab_info_usage")),
        ("booking_usage", result.get("booking_usage")),
        ("complaint_usage", result.get("complaint_usage")),
        ("direct_usage", result.get("direct_usage")),
        ("inquiry_usage", result.get("inquiry_usage")),
    ]
    total_in = total_out = total = 0
    breakdown = {}

    for key, u in nodes:
        if not u:
            continue
        i = u.get("input_tokens",  0) or 0
        o = u.get("output_tokens", 0) or 0
        t = u.get("total_tokens",  0) or (i + o)
        if i or o:
            node_cost = (i * INPUT_COST_PER_TOKEN) + (o * OUTPUT_COST_PER_TOKEN)
            breakdown[key] = {"input": i, "output": o, "total": t, "cost_usd": node_cost}
            total_in  += i
            total_out += o
            total     += t

    total_cost_usd = (total_in * INPUT_COST_PER_TOKEN) + (total_out * OUTPUT_COST_PER_TOKEN)
    total_cost_cents = total_cost_usd * 100
    req_per_dollar = int(1.0 / total_cost_usd) if total_cost_usd > 0 else 0

    return {
        "breakdown":        breakdown,
        "total_input":      total_in,
        "total_output":     total_out,
        "total_tokens":     total,
        "total_cost_usd":   total_cost_usd,
        "total_cost_cents": total_cost_cents,
        "req_per_dollar":   req_per_dollar,
    }


def _consume_subscription(message: "IncomingMessage", usage: dict) -> None:
    """
    Deducts one message + the real estimated cost from the laboratory's
    subscription, based on the page this message came in on.
    Silently no-ops if the page/subscription can't be resolved, so a
    billing lookup failure never breaks the reply to the user.
    """
    from software_service.subscripition_service import SubscriptionService
    from models.models import Page

    try:
        page = Page.query.filter_by(
            page_id=message.page_id,
            platform_id=message.platform_id,
        ).first()

        if not page:
            print(f"[_consume_subscription] No page found for page_id={message.page_id} platform_id={message.platform_id}")
            return

        subscription = SubscriptionService.get_by_page(page)

        if not subscription:
            print(f"[_consume_subscription] No subscription found for laboratory_id={page.laboratory_id}")
            return

        SubscriptionService.consume(
            subscription,
            cost=usage["total_cost_usd"],
        )

    except Exception as e:
        print(f"[_consume_subscription] Error: {e}")



def run_agent(message: IncomingMessage, ocr_usage: dict = None) -> tuple[str, bytes | None]:
    client = ClientService.get_or_create_client(
        message.sender_id, message.page_id, message.platform_id
    )

    platform_name = message.platform_name or str(message.platform_id)

    state = {
        "page_id":           message.page_id,
        "sender_id":         message.sender_id,
        "platform_id":       message.platform_id,
        "platform_name":     platform_name,
        "user_message":      message.text or "",
        "summary":           client.summary          or "",
        "last_bot_message":  client.last_bot_message or "",
        "intent":            None,
        "response":          None,
        "intent_usage":      None,
        "lab_info_usage":    None,
        "booking_usage":     None,
        "complaint_usage":   None,
        "direct_usage":      None,   
        "inquiry_usage":     None,
        "visit_saved":     None,
        "complaint_saved":   None,
        "inquiry_saved":     None,
    }

    logger.info(
        "[run_agent] start | sender_id=%s | platform=%s | message=%r",
        message.sender_id, platform_name, message.text,
    )

    try:
        result       = get_agent_graph().invoke(state)
        response_obj = AgentResponse.from_result(result)
    except Exception:
        logger.exception(
            "[run_agent] error | sender_id=%s | platform=%s",
            message.sender_id, platform_name,
        )
        return "Sorry, something went wrong. Please try again in a moment.", None

    usage = _calc_total_usage(result, ocr_usage=ocr_usage)
    _consume_subscription(message, usage)

    logger.info(
        "[run_agent] metrics | sender_id=%s | platform=%s | intent=%s",
        message.sender_id, platform_name, result.get("intent"),
    )
    for node, u in usage["breakdown"].items():
        logger.info(
            "[run_agent] node=%-22s in=%5d | out=%5d | total=%6d | cost=$%.8f",
            node, u["input"], u["output"], u["total"], u["cost_usd"],
        )
    logger.info(
        "[run_agent] totals | sender_id=%s | input=%d | output=%d | total=%d | cost=$%.8f (%.5fc) | req_per_$=%d",
        message.sender_id,
        usage["total_input"],
        usage["total_output"],
        usage["total_tokens"],
        usage["total_cost_usd"],
        usage["total_cost_cents"],
        usage["req_per_dollar"],
    )

    # Return reply text and any booking PDF generated
    return response_obj.response, result.get("booking_pdf")


def handle_image_message(message: IncomingMessage, page) -> tuple[str, bytes | None]:
    """
    Downloads the prescription image, processes classification and OCR extraction,
    and returns a reply response (and optional PDF) to send back to the user.
    """
    image_url = message.media.get("url") if message.media else None
    if not image_url:
        logger.warning(
            "[handle_image_message] no image url | sender_id=%s", message.sender_id,
        )
        return "برجاء إرسال صورة روشتة صالحة.", None

    try:
        logger.info(
            "[handle_image_message] downloading image | sender_id=%s | url=%s",
            message.sender_id, image_url,
        )
        img_res = requests.get(image_url, timeout=30)
        if img_res.status_code == 200:
            # Ensure static/uploads exists
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            uploads_dir = os.path.join(project_dir, "static", "uploads")
            os.makedirs(uploads_dir, exist_ok=True)

            filename = f"{uuid.uuid4().hex}.jpg"
            image_path = os.path.join(uploads_dir, filename)
            with open(image_path, "wb") as f:
                f.write(img_res.content)

            logger.info(
                "[handle_image_message] image saved | sender_id=%s | path=%s",
                message.sender_id, image_path,
            )

            # Process image OCR classification & extraction
            ocr_result = process_prescription_ocr(
                image_path=image_path,
                phone_number="",
                comes_from=f"Facebook:{message.sender_id}:{message.page_id}",
                laboratory_id=page.laboratory_id
            )

            ocr_usage = ocr_result.get("ocr_usage")

            logger.info(
                "[handle_image_message] ocr result | sender_id=%s | success=%s | classified_as=%s",
                message.sender_id, ocr_result.get("success"), ocr_result.get("classified_as"),
            )

            if ocr_result.get("success"):
                # High confidence prescription -> Send extracted text to LangGraph Agent
                extracted_text = ocr_result.get("extracted_text", "")
                message.text = f"[Prescription OCR Extracted Text]:\n{extracted_text}"
                return run_agent(message, ocr_usage=ocr_usage)
            else:
                # Low confidence prescription or spam
                if ocr_result.get("classified_as") == "prescription":
                    static_reply = "لقد استلمنا صورتك وسيقوم الطبيب بمراجعتها والرد عليك ."
                    
                    # Update Client summary in DB
                    ClientService.update_client_summary_and_last_bot_message(
                        sender_id=message.sender_id,
                        page_id=message.page_id,
                        platform_id=message.platform_id,
                        summary="User uploaded a prescription image. Waiting for manual doctor review on dashboard.",
                        last_bot_message=static_reply
                    )
                    count_request()
                    return static_reply, None
                else:
                    spam_reply = "عذراً، يبدو أن الصورة المرفقة ليست روشتة طبية واضحة. يرجى إرسال صورة روشتة صحيحة لطلب التحاليل."
                    return spam_reply, None
        else:
            logger.warning(
                "[handle_image_message] download failed | sender_id=%s | status_code=%s",
                message.sender_id, img_res.status_code,
            )
            return "عذرًا، فشل تحميل الصورة المرفقة. يرجى المحاولة مرة أخرى.", None
    except Exception:
        logger.exception(
            "[handle_image_message] error | sender_id=%s", message.sender_id,
        )
        return "عذرًا، حدث خطأ أثناء معالجة الصورة المرفقة. يرجى المحاولة مرة أخرى.", None