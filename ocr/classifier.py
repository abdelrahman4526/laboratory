"""
ocr/classifier.py

Single-call Gemini multimodal OCR engine.
Replaces the old two-step flow (classify → extract) with ONE API call
that classifies the document AND extracts lab tests simultaneously.

Returned dict shape:
{
    "document_type":       str,   # e.g. "lab_prescription", "advertisement", ...
    "is_prescription":     bool,
    "is_spam":             bool,
    "overall_confidence":  int,   # 0-100
    "process_success":     bool,  # True if overall_confidence >= CONFIDENCE_THRESHOLD
    "labs": [
        {
            "standardized_name": str,
            "matched_text":      str,
            "confidence":        int   # 0-100
        },
        ...
    ],
    "unknown_items": [str, ...],
    "notes":               str
}
"""

import os
import json
import time
import logging

from PIL import Image
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ── Threshold ──────────────────────────────────────────────────────────────────
# If overall_confidence >= this value → process_success = True → automated flow
# If overall_confidence <  this value → process_success = False → manual review
CONFIDENCE_THRESHOLD = 70

# ── Prompt ────────────────────────────────────────────────────────────────────
_PROMPT = """
You are an expert laboratory prescription OCR system.

Your task is to analyze the uploaded image.
STEP 1
-------
Determine the document type.

Possible values:
- lab_prescription
- medical_report
- radiology_request
- advertisement
- invoice
- blank
- other

STEP 2
-------
If it is NOT a laboratory prescription return:

{
    "document_type": "...",
    "is_prescription": false,
    "is_spam": true,
    "overall_confidence": 0,
    "process_success": false,
    "labs": [],
    "unknown_items": [],
    "notes": "Reason"
}

STEP 3
-------
If it IS a laboratory prescription:

Extract ONLY laboratory investigations.

Ignore:
- doctor name
- patient name
- age
- gender
- address
- diagnosis
- medications
- signatures
- dates
- phone numbers

Do NOT invent laboratory tests.

If text is unreadable, put it inside unknown_items.

Remove duplicate labs.

CRITICAL ACCURACY RULES:
- Only extract a lab test if you can actually see text in the image that
  corresponds to it. Do NOT infer, guess, or add tests based on common
  panels, typical bundles, or what "usually" goes together.
- Every "matched_text" MUST be text that is genuinely visible in the image.
  Never fabricate matched_text.
- If handwriting or print is unclear, blurry, cropped, or ambiguous, do NOT
  force a match to the closest-sounding test. Put the raw fragment in
  unknown_items instead of guessing.
- If two readings of the same handwriting are plausible, choose
  unknown_items over a low-confidence guess in labs.
- Remove duplicate labs (same test mentioned more than once).

STEP 4
-------
Each laboratory test must contain:

{
    "standardized_name": "",
    "matched_text":      "",
    "confidence":        95
}

confidence is your confidence (0-100)
that the extracted laboratory test is correct.

STEP 5
-------
Return overall_confidence (0-100).

This represents your confidence that
the COMPLETE extraction is correct.

Rules:
- overall_confidence >= 70  →  process_success = true
- overall_confidence <  70  →  process_success = false
-
Return ONLY valid JSON. No markdown fences.
"""


def analyze_prescription(image_path: str) -> dict:
    """
    Single Gemini multimodal call that classifies the image AND extracts
    lab tests in one pass.

    Args:
        image_path: Absolute path to the prescription image on disk.

    Returns:
        dict with keys: document_type, is_prescription, is_spam,
        overall_confidence, process_success, labs, unknown_items, notes.
        Always returns a safe dict even on failure.
    """
    _SAFE_SPAM = {
        "document_type":      "other",
        "is_prescription":    False,
        "is_spam":            True,
        "overall_confidence": 0,
        "process_success":    False,
        "labs":               [],
        "unknown_items":      [],
        "notes":              "",
    }

    # ── API key ───────────────────────────────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        _SAFE_SPAM["notes"] = (
            "Gemini API key not configured "
            "(set GEMINI_API_KEY or GOOGLE_API_KEY)."
        )
        logger.error("[OCR] %s", _SAFE_SPAM["notes"])
        return _SAFE_SPAM

    # ── Load image ────────────────────────────────────────────────────────────
    try:
        img = Image.open(image_path)
    except Exception as exc:
        _SAFE_SPAM["notes"] = f"Failed to open image: {exc}"
        logger.error("[OCR] %s", _SAFE_SPAM["notes"])
        return _SAFE_SPAM

    # ── Single Gemini call ────────────────────────────────────────────────────
    try:
        gemini_client = genai.Client(api_key=api_key)

        start = time.time()
        response = gemini_client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=[img, _PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )
        elapsed_ms = round((time.time() - start) * 1000, 2)

        usage = response.usage_metadata
        logger.info(
            "[OCR] done | time=%s ms | in=%s | out=%s | total=%s",
            elapsed_ms,
            usage.prompt_token_count,
            usage.candidates_token_count,
            usage.total_token_count,
        )

    except Exception as exc:
        _SAFE_SPAM["notes"] = f"Gemini API call failed: {exc}"
        logger.error("[OCR] %s", _SAFE_SPAM["notes"])
        return _SAFE_SPAM

    # ── Parse JSON ────────────────────────────────────────────────────────────
    try:
        data = json.loads(response.text)
    except Exception as exc:
        _SAFE_SPAM["notes"] = (
            f"JSON parse error: {exc} | raw: {response.text[:200]}"
        )
        logger.error("[OCR] %s", _SAFE_SPAM["notes"])
        return _SAFE_SPAM

    # ── Enforce threshold (server-side safety guard) ───────────────────────────
    overall_confidence = int(data.get("overall_confidence", 0))
    data["process_success"] = overall_confidence >= CONFIDENCE_THRESHOLD

    if usage:
        data["ocr_usage"] = {
            "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
            "total_tokens": getattr(usage, "total_token_count", 0) or 0
        }

    return data


# ---------------------------------------------------------------------------
# Backwards-compat shim so existing imports of classify_prescription still work
# ---------------------------------------------------------------------------
def classify_prescription(image_path: str) -> dict:
    """
    Legacy wrapper kept for import compatibility.
    Internally calls analyze_prescription() and maps the result to the
    old shape: { "classification", "confidence", "reason" }.
    """
    result = analyze_prescription(image_path)
    return {
        "classification": "prescription" if result.get("is_prescription") else "spam",
        "confidence":     result.get("overall_confidence", 0) / 100.0,
        "reason":         result.get("notes", ""),
    }