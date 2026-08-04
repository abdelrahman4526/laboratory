"""
enrich_labservices_gemini.py
------------------------------------------------------------
سكريبت بيقرأ صفوف جدول labservices اللي ناقصها:
    - alias_names
    - sample_type
    - keywords
وبيبعتها لـ Gemini API عشان يولّد المحتوى، وبعدين يحدّث الصف في MySQL.

المتطلبات:
    pip install sqlalchemy pymysql requests python-dotenv

طريقة التشغيل:
    1) اعمل ملف .env جنب السكريبت فيه:
         DATABASE_URL=mysql+pymysql://USER:PASSWORD@HOST:3306/DBNAME
         GEMINI_API_KEY=your_api_key_here
    2) شغل: python enrich_labservices_gemini.py
------------------------------------------------------------
"""

import os
import json
import time
import logging

import requests
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ------------------------------------------------------------------
# الإعدادات
# ------------------------------------------------------------------
load_dotenv()

DATABASE_URL = os.getenv("MAIN_DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# اسم الموديل - غيّره لو عايز نسخة تانية
GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

BATCH_DELAY_SECONDS = 1.5   # تأخير بسيط بين كل طلب وطلب عشان الـ rate limit
MAX_RETRIES = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

if not DATABASE_URL:
    raise RuntimeError("لازم تحدد DATABASE_URL في ملف .env")
if not GEMINI_API_KEY:
    raise RuntimeError("لازم تحدد GEMINI_API_KEY في ملف .env")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


# ------------------------------------------------------------------
# بناء الـ prompt وطلب Gemini
# ------------------------------------------------------------------
def build_prompt(name: str, description: str | None, patient_instructions: str | None) -> str:
    description = description or ""
    patient_instructions = patient_instructions or ""

    return f"""
You are an assistant specialized in medical lab tests. Analyze the following
lab test and return structured data about it as pure JSON only, with no extra
text, explanation, or Markdown.
 
Test name: {name}
Description (if any): {description}
Patient instructions (if any): {patient_instructions}
 
Return exactly this JSON shape (pure JSON only):
 
{{
  "alias_names": {{
    "alias": "the single most common alias/abbreviation for this test",
    "measurement": "a short phrase describing what this test measures/detects in the body",
"equivalent_name": "the name of a DIFFERENT, distinctly-named test that measures/detects the same thing medically (if one genuinely exists), otherwise an empty string"
  "sample_type": "the sample type required for this test, e.g. Blood, Serum, Urine, Stool, Plasma",
  "keywords": ["keyword 1", "keyword 2", "keyword 3"]
}}
 
Important rules:
- alias_names must be a SINGLE object (not an array) with exactly the three
  keys above: name, measurement, equivalent_name_egypt.
- measurement should briefly state what the test measures or detects (e.g.
  "Measures thyroid-stimulating hormone levels in blood"), not a department name.
- equivalent_name is NOT just another abbreviation of the same test name. It
  must be a genuinely different, distinctly-named test that measures or
  screens for the same medical thing (e.g. two different test names used by
  different labs/methods for the same clinical purpose). If no real
  equivalent test exists, return an empty string "" — do not invent one
- If unsure about a value, give your best reasonable answer instead of
  leaving it empty.
- sample_type must be a single, short, clear value.
- keywords should be 5 to 10 common search terms (both Arabic and English)
  a patient might use to search for this test.
- Return valid JSON only, no ```json fences and no text before or after it.
"""
 
 
def call_gemini(name: str, description: str | None, patient_instructions: str | None) -> dict:
    prompt = build_prompt(name, description, patient_instructions)
 
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "response_mime_type": "application/json",
        },
    }
 
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(GEMINI_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
 
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            # Defensive cleanup in case the model wraps output in ```json ... ```
            raw_text = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
 
            parsed = json.loads(raw_text)
 
            # Basic shape validation
            if "alias_names" not in parsed or "sample_type" not in parsed or "keywords" not in parsed:
                raise ValueError(f"Incomplete response shape: {parsed}")
            if not isinstance(parsed["alias_names"], dict):
                raise ValueError(f"alias_names must be a single object, not an array: {parsed['alias_names']}")
            required_alias_keys = {"alias", "measurement", "equivalent_name"}
            if not required_alias_keys.issubset(parsed["alias_names"].keys()):
                raise ValueError(f"alias_names is missing required keys: {parsed['alias_names']}")
 
            return parsed
 
        except Exception as e:  # noqa: BLE001
            last_error = e
            log.warning("Attempt %s failed for '%s': %s", attempt, name, e)
            time.sleep(2 * attempt)
 
    raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} attempts for '{name}': {last_error}")
 
# ------------------------------------------------------------------
# منطق قاعدة البيانات
# ------------------------------------------------------------------
def fetch_pending_rows(conn):
    query = text("""
        SELECT id, name, description, patient_instructions
        FROM labservices
        WHERE is_active = 1
          AND (
                alias_names IS NULL OR alias_names = '' OR
                sample_type IS NULL OR sample_type = '' OR
                keywords IS NULL OR keywords = ''
              )
    """)
    return conn.execute(query).mappings().all()


def update_row(conn, row_id: int, alias_names: list, sample_type: str, keywords: list):
    query = text("""
        UPDATE labservices
        SET alias_names = :alias_names,
            sample_type = :sample_type,
            keywords = :keywords
        WHERE id = :id
    """)
    conn.execute(
        query,
        {
            "alias_names": json.dumps(alias_names, ensure_ascii=False),
            "sample_type": sample_type,
            "keywords": json.dumps(keywords, ensure_ascii=False),
            "id": row_id,
        },
    )


# ------------------------------------------------------------------
# التشغيل الرئيسي
# ------------------------------------------------------------------
def main():
    with engine.connect() as conn:
        rows = fetch_pending_rows(conn)
        log.info("عدد الصفوف الناقصة: %s", len(rows))

        for i, row in enumerate(rows, start=1):
            log.info("(%s/%s) بمعالجة: %s (id=%s)", i, len(rows), row["name"], row["id"])
            try:
                result = call_gemini(
                    name=row["name"],
                    description=row["description"],
                    patient_instructions=row["patient_instructions"],
                )

                update_row(
                    conn,
                    row_id=row["id"],
                    alias_names=result["alias_names"],
                    sample_type=result["sample_type"],
                    keywords=result["keywords"],
                )
                conn.commit()
                log.info("تم التحديث بنجاح لـ id=%s", row["id"])

            except Exception as e:  # noqa: BLE001
                log.error("فشل معالجة id=%s: %s", row["id"], e)
                conn.rollback()

            time.sleep(BATCH_DELAY_SECONDS)

    log.info("انتهى السكريبت.")


if __name__ == "__main__":
    main()