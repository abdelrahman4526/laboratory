from sqlalchemy import text,bindparam
from collections import defaultdict

from knowledge.schemas import EntityType
from knowledge.utils import main_session
from search.schemas import SearchResult
import logging
logger = logging.getLogger(__name__)


TABLE_BY_TYPE: dict[EntityType, str] = {
    EntityType.LAB: "labservices",
}

COLUMNS_BY_TYPE: dict[EntityType, list[str]] = {
    EntityType.LAB: [
        "id", "name", "description", "price",
        "sample_type", "durations", "patient_instructions",
    ],
}


def _fetch_rows_by_type(entity_type: EntityType, ids: list) -> dict:
    """
    يجيب كل الصفوف المطلوبة لنوع واحد في query واحدة (بدل query لكل صف).
    بيرجع dict: id -> row_dict
    """
    if not ids:
        return {}
 
    table = TABLE_BY_TYPE.get(entity_type)
    columns = COLUMNS_BY_TYPE.get(entity_type)
    if table is None or columns is None:
        logger.warning(
            "[RAG] No table/columns mapped for entity_type=%s (ids=%s)",
            entity_type, ids,
        )
        return {}
 
    query = text(
        f"SELECT {', '.join(columns)} FROM {table} WHERE id IN :ids"
    ).bindparams(bindparam("ids", expanding=True))
 
    try:
        with main_session() as session:
            rows = session.execute(query, {"ids": ids}).fetchall()
    except Exception as exc:
        logger.error(
            "[RAG] DB query failed for type=%s table=%s ids=%s | error=%s",
            entity_type, table, ids, exc,
        )
        return {}
 
    result = {row.id: dict(row._mapping) for row in rows}
 
    logger.info(
        "[RAG] fetched %d/%d rows | type=%s table=%s",
        len(result), len(ids), entity_type, table,
    )
 
    missing_ids = set(ids) - set(result.keys())
    if missing_ids:
        logger.warning(
            "[RAG] %d id(s) not found in table=%s | missing_ids=%s",
            len(missing_ids), table, missing_ids,
        )
 
    return result


def build_context(results: list[SearchResult]) -> str:
    """
    يبني نص context جاهز للـ LLM prompt من نتائج البحث.
    بيجمع كل النتائج ويعمل query واحدة (بدل query لكل نتيجة على حدة).
    ترتيب الأقسام في النص النهائي بيحافظ على ترتيب `results` الأصلي.
    """
    if not results:
        logger.info("[RAG] build_context called with no results")
        return ""

    ids_by_type: dict[EntityType, list] = defaultdict(list)
    for result in results:
        ids_by_type[result.type].append(result.id)

    rows_by_type: dict[EntityType, dict] = {
        entity_type: _fetch_rows_by_type(entity_type, ids)
        for entity_type, ids in ids_by_type.items()
    }

    sections = []
    for result in results:
        row = rows_by_type.get(result.type, {}).get(result.id)
        if not row:
            continue

        block = [f"Name: {row['name']}"]

        if row.get("description"):
            block.append(f"Description: {row['description']}")

        if row.get("price") is not None:
            block.append(f"Price: {row['price']}")

        if row.get("sample_type"):
            block.append(f"sample_type: {row['sample_type']}")

        if row.get("durations"):
            block.append(f"Duration: {row['durations']}")

        if row.get("patient_instructions"):
            block.append(f"Patient Instructions: {row['patient_instructions']}")

        sections.append("\n".join(block))

    context = "\n\n------------------------\n\n".join(sections)

    logger.info(
        "[RAG] build_context done | sections=%d/%d | chars=%d",
        len(sections), len(results), len(context),
    )

    return context