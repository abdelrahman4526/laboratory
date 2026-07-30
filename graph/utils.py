import re
import io
import os
from datetime import datetime, timezone

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

from models.models import RequestCounter, db

def strip_tags(text: str) -> str:
    """Remove all XML-style tags injected by LLM prompts from a reply string."""
    text = re.sub(r"<SUMMARY>.*?</SUMMARY>",                   "", text, flags=re.DOTALL)
    text = re.sub(r"<INTENT>.*?</INTENT>",                     "", text, flags=re.DOTALL)
    text = re.sub(r"<LAST_BOT_MESSAGE>.*?</LAST_BOT_MESSAGE>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>",                                  "", text)
    return text.strip()


def detect_language_fallback(user_message: str, arabic: str, default: str) -> str:
    """
    Return `arabic` if the user message contains Arabic characters,
    otherwise return `default`.
    Used for error/fallback messages in nodes that must match user language.
    """
    if any("\u0600" <= c <= "\u06ff" for c in user_message):
        return arabic
    return default


PLATFORM_MAP = {
    1: "WhatsApp",
    2: "Facebook",
}

def get_platform_name(platform_id) -> str:
    """Convert platform_id to platform name string."""
    if not platform_id:
        return "unknown"
    try:
        key = int(platform_id)
        return PLATFORM_MAP.get(key, str(platform_id))
    except ValueError:
        return str(platform_id)


def count_request():
    """Decrement the global request counter."""
    try:
        counter = RequestCounter.query.first()
        if counter:
            counter.decrement()
    except Exception as e:
        print(f"[count_request] Error decrementing counter: {e}")


# ── colours ───────────────────────────────────────────────────────────────────
NAVY      = colors.HexColor("#1B4B8A")
CREAM     = colors.HexColor("#F5F0E8")
LIGHT_ROW = colors.HexColor("#EAF0FA")
WHITE     = colors.white
MUTED     = colors.HexColor("#6B7280")
DARK      = colors.HexColor("#1F2937")

# ── font ──────────────────────────────────────────────────────────────────────
_FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Cairo.ttf")

def _register_font() -> str:
    try:
        pdfmetrics.registerFont(TTFont("Cairo", _FONT_PATH))
        return "Cairo"
    except Exception:
        return "Helvetica"

# ── Arabic helper ─────────────────────────────────────────────────────────────
def _ar(text: str) -> str:
    if not text:
        return text
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text

def _is_arabic(text: str) -> bool:
    return any("\u0600" <= c <= "\u06FF" for c in (text or ""))

# ── style factory ─────────────────────────────────────────────────────────────
def _ps(name_: str, font: str, size: int, color=DARK, align: int = 0) -> ParagraphStyle:
    return ParagraphStyle(
        name_,
        fontName=font,
        fontSize=size,
        textColor=color,
        alignment=align,
        leading=size * 1.45,
    )

def generate_booking_pdf(
    name: str,
    phone: str,
    date: str,
    details: str,
    reference_id: str,
) -> bytes:

    font     = _register_font()
    buffer   = io.BytesIO()
    margin   = 18 * mm
    usable_w = A4[0] - 2 * margin

    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=12*mm, bottomMargin=12*mm,
    )

    story = []

    # ── cream header: logo icon + laboratory name ─────────────────────────────
    clinic_ar = _ar("مختبر التحاليل الطبية التخصصي")
    
    hdr = Table(
        [[
            Paragraph("✚", _ps("icon", "Helvetica", 22, NAVY, align=0)),
            Paragraph(clinic_ar, _ps("clin", font, 15, NAVY, align=2)),
        ]],
        colWidths=[16*mm, usable_w - 16*mm],
    )
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), CREAM),
        ("TOPPADDING",    (0,0),(-1,-1), 13),
        ("BOTTOMPADDING", (0,0),(-1,-1), 13),
        ("LEFTPADDING",   (0,0),(-1,-1), 12),
        ("RIGHTPADDING",  (0,0),(-1,-1), 12),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("ROUNDEDCORNERS",[8,8,0,0]),
    ]))
    story.append(hdr)

    # ── navy title bar ────────────────────────────────────────────────────────
    title_ar = _ar("تأكيد حجز موعد التحليل")
    
    ttl = Table(
        [[
            Paragraph("Booking Confirmation", _ps("ten", font, 11, WHITE, align=0)),
            Paragraph(title_ar,               _ps("tar", font, 12, WHITE, align=2)),
        ]],
        colWidths=[usable_w/2, usable_w/2],
    )
    ttl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), NAVY),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("LEFTPADDING",   (0,0),(-1,-1), 12),
        ("RIGHTPADDING",  (0,0),(-1,-1), 12),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(ttl)

    # ── reference bar ─────────────────────────────────────────────────────────
    ref = Table(
        [[Paragraph(f"Reference: {reference_id}", _ps("ref", font, 10, NAVY, align=1))]],
        colWidths=[usable_w],
    )
    ref.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), CREAM),
        ("TOPPADDING",    (0,0),(-1,-1), 7),
        ("BOTTOMPADDING", (0,0),(-1,-1), 7),
        ("ROUNDEDCORNERS",[0,0,8,8]),
    ]))
    story.append(ref)
    story.append(Spacer(1, 8*mm))

    # ── info rows ─────────────────────────────────────────────────────────────
    fields = [
        ("Patient Name",     "اسم المريض",    name),
        ("Phone",            "رقم الهاتف",    phone),
        ("Appointment Date", "تاريخ الموعد",  date),
        ("Required Analysis","التحاليل المطلوبة", details),
    ]

    rows = []
    for i, (en_lbl, ar_lbl, val) in enumerate(fields):
        lbl_text = f"{en_lbl} / {_ar(ar_lbl)}"
        lbl_cell = Paragraph(lbl_text, _ps(f"l_{i}", font, 9, MUTED, align=0))
        
        val_text  = _ar(val) if _is_arabic(val or "") else (val or "—")
        val_align = 2 if _is_arabic(val or "") else 0
        val_cell  = Paragraph(val_text, _ps(f"v_{i}", font, 11, DARK, align=val_align))
        
        rows.append([lbl_cell, val_cell])

    info = Table(rows, colWidths=[65*mm, usable_w - 65*mm])
    info.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), WHITE),
        ("BACKGROUND",    (0,0),(-1,0),  LIGHT_ROW),
        ("BACKGROUND",    (0,2),(-1,2),  LIGHT_ROW),
        ("LINEBELOW",     (0,0),(-1,-2), 0.5, colors.HexColor("#DDE3EE")),
        ("TOPPADDING",    (0,0),(-1,-1), 12),
        ("BOTTOMPADDING", (0,0),(-1,-1), 12),
        ("LEFTPADDING",   (0,0),(-1,-1), 12),
        ("RIGHTPADDING",  (0,0),(-1,-1), 12),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("ROUNDEDCORNERS",[6,6,6,6]),
    ]))
    story.append(info)
    story.append(Spacer(1, 10*mm))

    # ── footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#C5D0E0")))
    story.append(Spacer(1, 3*mm))
    
    issued    = datetime.now(timezone.utc).strftime("%B %d, %Y  %H:%M UTC")
    footer_ar = _ar("احتفظ بهذه البطاقة للمراجعة")
    
    footer_table = Table(
        [[
            Paragraph(f"Issued: {issued}", _ps("fl", font, 8, MUTED, align=0)),
            Paragraph(footer_ar,           _ps("fr", font, 8, MUTED, align=2)),
        ]],
        colWidths=[usable_w/2, usable_w/2],
    )
    footer_table.setStyle(TableStyle([
        ("TOPPADDING",    (0,0),(-1,-1), 2),
        ("BOTTOMPADDING", (0,0),(-1,-1), 2),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 0),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(footer_table)

    doc.build(story)
    return buffer.getvalue()
