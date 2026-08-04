import json
import logging
import re
from dataclasses import dataclass
from threading import Lock
import pickle
import faiss
import numpy as np
from rapidfuzz import fuzz, process
from sqlalchemy import text
from search.preprocess.normaliztion import normalize
from search.preprocess.ngram import ngram_similarity
from knowledge.utils import main_session
from knowledge.schemas import EntityType
from .schemas import SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


FAISS_INDEX_PATH = "lads.faiss"
FAISS_METADATA_PATH = "labs_metadata.pkl"
FAISS_IDS_PATH = "labs_ids.npy"

_faiss_ids = None
_faiss_index = None
_faiss_metadata = None

TABLE_BY_TYPE: dict[EntityType, str] = {
    EntityType.LAB: "labservices",

}

# rapid (fuzz.*) و ngram_similarity كلاهما بيرجعوا قيمة على مقياس 0-1،
# فالـ final_score = (rapid + ngram) / 2 برضو بيبقى على مقياس 0-1.
# لازم MIN_SCORE يكون على نفس المقياس (0-1)، مش 0-100، وإلا
# final_score >= MIN_SCORE مستحيل يتحقق أبدًا وهيرجع نتائج فاضية دايمًا.
MIN_SCORE = 0.55  # 0-1 scale
_NGRAM_MIN, _NGRAM_MAX = 1, 3
_CHAR_NGRAM_SIZE = 3


# ---------------------------------------------------------------------------
# Cache state (per entity type)
# ---------------------------------------------------------------------------

@dataclass
class _EntityIndex:
    rows: list                     # raw DB rows, indexed by position
    keywords: list[str]            # flattened keyword list (n-grams)
    keyword_row: list[int]         # keyword_index -> row_index
    keyword_source_len: list[int]  # keyword_index -> len(normalized source name/alias it came from)


_cache: dict[EntityType, _EntityIndex] = {}
_cache_lock = Lock()


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

def fuzzy_search(
    query: str,
    aliases: list[str] | None = None,
    entity_types: list[EntityType] | None = None,
    limit: int = 5,
) -> list[SearchResult]:

    normalized_query = normalize(query)

    if not normalized_query:
        return []

    results = []

    # كان فيه هنا EntityType.BUNDLE اللي مش موجودة في الـ enum الحالي
    # (اللي فيه بس EntityType.LAB) وده كان بيعمل AttributeError ويكسر
    # كل الـ request. بنستخدم بدل كده مفاتيح TABLE_BY_TYPE نفسها، عشان
    # لو ضفت نوع جديد فعلاً في المستقبل يتضاف تلقائي هنا من غير كراش.
    types_to_search = entity_types or list(TABLE_BY_TYPE.keys())

    with main_session() as session:

        for entity_type in types_to_search:

            table = TABLE_BY_TYPE[entity_type]

            rows = session.execute(
                text(f"""
                    SELECT id,name
                    FROM {table}
                """)
            ).fetchall()

            scored = []

            for row in rows:

                normalized_name = normalize(row.name)

                rapid = max(
                    fuzz.partial_ratio(normalized_query, normalized_name),
                    fuzz.token_set_ratio(normalized_query, normalized_name),
                ) / 100

                ngram = ngram_similarity(
                    normalized_query,
                    normalized_name,
                )

                final_score = (rapid + ngram) / 2

                if final_score >= MIN_SCORE:

                    scored.append(
                        (
                            row,
                            rapid,
                            ngram,
                            final_score,
                        )
                    )

            scored.sort(
                key=lambda x: x[3],
                reverse=True,
            )

            for row, rapid, ngram, final_score in scored[:limit]:

                results.append(
                    SearchResult(
                        id=row.id,
                        type=entity_type,
                        name=row.name,
                        score=round(final_score, 3),
                        source="fuzzy",
                    )
                )

    results.sort(
        key=lambda x: x.score,
        reverse=True,
    )


    
    return results[:limit]