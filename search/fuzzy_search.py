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

MIN_SCORE = 0.45  # 0-1 scale, updated for short medical terms like T4, T3


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
# Fuzzy Search Logic with Aliases & Exact Token Matching
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
    types_to_search = entity_types or list(TABLE_BY_TYPE.keys())

    with main_session() as session:

        for entity_type in types_to_search:

            table = TABLE_BY_TYPE[entity_type]

            # Query name, alias_names, keywords, search_text from labservices
            rows = session.execute(
                text(f"""
                    SELECT id, name, alias_names, keywords, search_text
                    FROM {table}
                """)
            ).fetchall()

            scored = []

            for row in rows:
                candidates = [row.name or ""]
                
                # Add aliases, keywords, and search_text to candidate search texts
                if hasattr(row, 'alias_names') and row.alias_names:
                    candidates.append(str(row.alias_names))
                if hasattr(row, 'keywords') and row.keywords:
                    candidates.append(str(row.keywords))
                if hasattr(row, 'search_text') and row.search_text:
                    candidates.append(str(row.search_text))
                if aliases:
                    candidates.extend([str(a) for a in aliases if a])

                max_final_score = 0.0

                for cand in candidates:
                    normalized_cand = normalize(cand)
                    if not normalized_cand:
                        continue

                    # Exact word/token match (vital for short medical terms like T4, T3, TSH, PTH, CBC)
                    query_words = set(re.findall(r'\b\w+\b', normalized_query.lower()))
                    cand_words = set(re.findall(r'\b\w+\b', normalized_cand.lower()))

                    if query_words and query_words.issubset(cand_words):
                        final_score = 1.0
                    else:
                        rapid = max(
                            fuzz.partial_ratio(normalized_query, normalized_cand),
                            fuzz.token_set_ratio(normalized_query, normalized_cand),
                        ) / 100

                        ngram = ngram_similarity(
                            normalized_query,
                            normalized_cand,
                        )

                        final_score = (rapid + ngram) / 2

                    if final_score > max_final_score:
                        max_final_score = final_score

                if max_final_score >= MIN_SCORE:
                    scored.append((row, max_final_score))

            scored.sort(
                key=lambda x: x[1],
                reverse=True,
            )

            for row, final_score in scored[:limit]:
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