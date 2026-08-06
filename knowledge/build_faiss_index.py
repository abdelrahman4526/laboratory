"""
build_faiss_index.py
يبني/يحدّث الـ FAISS index من جدول labservices بالكامل.
شغّله من روت المشروع: python build_faiss_index.py
"""

import json
import logging
import time

from sqlalchemy import text

from knowledge.utils import main_session
from knowledge.schemas import EntityType, ApprovedKnowledge, AliasNames, VectorMetadata
from knowledge.embedding import generate_embedding
from knowledge.vector_store import ensure_vector_table, upsert_vector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_json_list(value: str | None) -> list[str]:
    """يقرأ عمود JSON array (زي keywords) ويرجعه كـ list نضيف من المسافات.
    لو القيمة مش JSON صالح (بيانات قديمة CSV مثلاً)، بيرجع لتقسيم الفواصل كـ fallback."""
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _load_alias_names(value: str | None) -> AliasNames:
    """يقرأ عمود alias_names (JSON object) ويحوّله لـ AliasNames.
    لو فاضي أو تالف، بيرجع AliasNames فاضي بدل ما يفشل السكريبت كله."""
    if not value:
        return AliasNames()
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return AliasNames(**parsed)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("alias_names JSON malformed, using empty AliasNames: %s", e)
    return AliasNames()


def _fetch_all_labs() -> list:
    with main_session() as session:
        return session.execute(
            text("""
                SELECT id, name, description, alias_names, sample_type, keywords
                FROM labservices
            """)
        ).fetchall()


def build_index():
    ensure_vector_table()

    labs = _fetch_all_labs()
    logger.info("Found %d labs to index", len(labs))

    success, failed = 0, 0

    for row in labs:
        try:
            approved = ApprovedKnowledge(
                entity_id=row.id,
                entity_type=EntityType.LAB,
                description=row.description or "",
                alias_names=_load_alias_names(row.alias_names),
                sample_type=row.sample_type or "",
                keywords=_load_json_list(row.keywords),
            )

            embedding = generate_embedding(row.name, approved)

            metadata = VectorMetadata(
                id=row.id,
                type=EntityType.LAB,
                name=row.name,
            )

            upsert_vector(metadata, embedding)
            success += 1
            logger.info("[%d/%d] Indexed: %s", success + failed, len(labs), row.name)

            # مهم: بريك بسيط بين الـ calls عشان متضربش rate limit في Gemini API
            time.sleep(0.2)

        except Exception:
            failed += 1
            logger.warning("Failed to index lab id=%s (%s)", row.id, row.name, exc_info=True)

    logger.info("Done. Success: %d, Failed: %d", success, failed)


if __name__ == "__main__":
    build_index()