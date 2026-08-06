"""
semantic_search.py
بحث دلالي (semantic) باستخدام embeddings + FAISS، متوافق مع
البنية الفعلية اللي بيكتبها knowledge/vector_store.py.
"""

import logging

import numpy as np

from knowledge.utils import get_gemini_client
from knowledge.schemas import EntityType
from knowledge.vector_store import get_index_and_metadata
from .schemas import SearchResult

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768
MIN_SIMILARITY = 0.5  


def _embed_query(query: str) -> list[float]:
    client_or_genai = get_gemini_client()

    if hasattr(client_or_genai, "models"):
        res = client_or_genai.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query,
            config={"output_dimensionality": EMBEDDING_DIM, "task_type": "retrieval_query"},
        )
        embedding = res.embeddings[0].values
    else:
        result = client_or_genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query",
            output_dimensionality=EMBEDDING_DIM,
        )
        embedding = result["embedding"]

    vec = np.array(embedding, dtype="float32")
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def semantic_search(query: str, entity_types: list[EntityType] | None = None,
                     limit: int = 10) -> list[SearchResult]:
    """
    يرجّع لحد `limit` نتيجة، مرتبة تنازليًا حسب القرب الدلالي من الاستعلام.
    """
    normalized = query.strip()
    if not normalized:
        return []

    try:
        query_vec = np.array(_embed_query(normalized), dtype="float32").reshape(1, -1)
    except Exception:
        logger.warning("[Semantic Search] Embedding generation failed | query=%r", normalized, exc_info=True)
        return []

    index, metadata = get_index_and_metadata()
    if index.ntotal == 0:
        logger.warning("[Semantic Search] FAISS index is empty | query=%r", normalized)
        return []

    types_to_search = {t.value for t in (entity_types or [EntityType.LAB])}

    k = min(index.ntotal, limit * 5)
    scores, faiss_ids = index.search(query_vec, k)

    results: list[SearchResult] = []
    for score, faiss_id in zip(scores[0], faiss_ids[0]):
        if faiss_id == -1:
            continue

        meta = metadata["faiss_id_to_meta"].get(int(faiss_id))
        if meta is None or meta["type"] not in types_to_search:
            continue

        if score < MIN_SIMILARITY:
            continue

        results.append(SearchResult(
            id=meta["id"],
            type=EntityType(meta["type"]),
            name=meta["name"],
            score=round(float(score), 3),
            source="vector",
        ))

    results.sort(key=lambda r: r.score, reverse=True)

    logger.info(
        "[Semantic Search] query=%r | candidates=%d | matched=%d",
        normalized, len(faiss_ids[0]), len(results),
    )

    return results[:limit]