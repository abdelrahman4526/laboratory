import os
import logging

from knowledge.schemas import EntityType
from .fuzzy_search import fuzzy_search
from .semantic_search import semantic_search
from .merger import merge_results
from .schemas import SearchResult

logger = logging.getLogger(__name__)




CONTEXT_BUDGET = 50  # max results returned to the caller


def run_search(refined_queries):
    """
    refined_queries: list of objects exposing `.query`, `.aliases`
    (and optionally `.description`) — e.g. the RefinedQuery items
    produced by the intent node. NOT plain strings.
    """

    if not refined_queries:
        logger.info("[Search Manager] no queries provided | skipping search")
        return {
            "results": [],
            "top_score": 0.0,
        }

    per_query_result_lists = []

    for item in refined_queries:

        fuzzy_results = fuzzy_search(
            query=item.query,
            aliases=item.aliases,
            limit=2,
        )

        semantic_results = semantic_search(
            query=item.query,
            limit=3,
        )
 
        logger.info(
            "[Search Manager] query=%r | fuzzy=%d | semantic=%d",
            item.query,
            len(fuzzy_results),
            len(semantic_results),
        )
 
        per_query_result_lists.append(fuzzy_results)
        per_query_result_lists.append(semantic_results)
 
    
    results = merge_results(*per_query_result_lists)
 
    results.sort(
        key=lambda x: x.score,
        reverse=True,
    )
 
    results = results[:CONTEXT_BUDGET]
 
    logger.info(
        "[Search Manager] done | final=%d | top_score=%s",
        len(results),
        results[0].score if results else 0.0,
    )
 
    return {
        "results": results,
        "top_score": results[0].score if results else 0.0,
    }