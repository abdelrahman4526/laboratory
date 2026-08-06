import io
import json
import logging

import requests

from platforms.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class FacebookHandler(BaseHandler):
    platform_id = 2
    API_VERSION = "v19.0"

    def __init__(self, page):
        super().__init__(page)
        self.base_url = f"https://graph.facebook.com/{self.API_VERSION}"
        self.params   = {"access_token": self.token}

    @property
    def platform_name(self) -> str:
        try:
            return self.page.platform.name
        except AttributeError:
            return "Facebook"

    # ── text ─────────────────────────────────────────────────────────────────

    def send(self, recipient_id: str, text: str):
        if not text or not text.strip():
            return None

        logger.debug("[FB SEND] to=%s", recipient_id)

        payload = {
            "messaging_type": "RESPONSE",
            "recipient": {"id": recipient_id},
            "message":   {"text": text},
        }
        return self._post_json(f"{self.base_url}/me/messages", payload)

    # ── file (PDF ticket) ─────────────────────────────────────────────────────

    def send_file(
        self,
        recipient_id: str,
        file_bytes: bytes,
        filename: str,
        mime_type: str = "application/pdf",
    ):
        logger.debug("[FB SEND FILE] to=%s file=%s", recipient_id, filename)

        recipient_json = json.dumps({"id": recipient_id})
        message_json   = json.dumps({
            "attachment": {
                "type": "file",
                "payload": {"is_reusable": False},
            }
        })

        try:
            response = requests.post(
                f"{self.base_url}/me/messages",
                params=self.params,
                data={
                    "messaging_type": "RESPONSE",
                    "recipient":      recipient_json,
                    "message":        message_json,
                },
                files={
                    "filedata": (filename, io.BytesIO(file_bytes), mime_type),
                },
                timeout=30,
            )
            if response.status_code not in [200, 201]:
                logger.error(
                    "[FB FILE ERROR] status=%s body=%s",
                    response.status_code, response.text,
                )
            return response
        except Exception as e:
            logger.error("[FB FILE ERROR] %s", e)
            return None

    # ── typing indicator ─────────────────────────────────────────────────────

    def send_typing(self, recipient_id: str):
        """Show typing indicator in Messenger."""
        logger.debug("[FB TYPING] to=%s", recipient_id)
        payload = {
            "recipient":     {"id": recipient_id},
            "sender_action": "typing_on",
        }
        return self._post_json(f"{self.base_url}/me/messages", payload)

    # ── comments ─────────────────────────────────────────────────────────────

    def handle_comment(self, comment_id: str):
        self.react_to_comment(comment_id)
        self.reply_to_comment(comment_id)
        self.send_private_reply(comment_id)

    def react_to_comment(self, comment_id: str):
        logger.debug("[FB LIKE COMMENT] comment_id=%s", comment_id)
        try:
            response = requests.post(
                f"{self.base_url}/{comment_id}/likes",
                params=self.params,
                timeout=10,
            )
            if response.status_code not in [200, 201]:
                logger.error("[FB LIKE ERROR] status=%s body=%s", response.status_code, response.text)
            return response
        except Exception as e:
            logger.error("[FB LIKE ERROR] %s", e)
            return None

    def reply_to_comment(
        self,
        comment_id: str,
        static_message: str = "شكراً على تعليقك! راسلنا خاصةً للمساعدة. 🙏",
    ):
        """Reply publicly to a comment with a static message."""
        logger.debug("[FB COMMENT REPLY] comment_id=%s", comment_id)
        payload = {"message": static_message}
        return self._post_json(
            f"{self.base_url}/{comment_id}/comments", payload
        )

    def send_private_reply(
        self,
        comment_id: str,
        text: str = "أهلاً! شكراً على تعليقك، كيف نقدر نساعدك؟",
    ):
        """Send a private reply to a comment via Facebook private_replies API."""
        logger.debug("[FB PRIVATE REPLY] comment_id=%s", comment_id)
        payload = {"message": text}
        return self._post_json(
            f"{self.base_url}/{comment_id}/private_replies", payload
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _post_json(self, url: str, payload: dict):
        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                params=self.params,
                timeout=10,
            )
            if response.status_code not in [200, 201]:
                logger.error(
                    "[FB ERROR] status=%s body=%s",
                    response.status_code, response.text,
                )
            return response
        except Exception as e:
            logger.error("[FB ERROR] Connection failed: %s", e)
            return None

    def parse_message(self, payload, page_id):
        from parsers.facebook import parse_facebook_message
        return parse_facebook_message(
            payload,
            page_id,
            platform_id=self.platform_id,
            platform_name=self.platform_name,
        )
