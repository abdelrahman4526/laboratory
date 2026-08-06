import logging

from search.search_manager import run_search
from graph.state import AgentState
from .context_builder import build_context
from .schemas import RetrievalResult

logger = logging.getLogger(__name__)


def retrieve(state: AgentState):

    sender_id = state.get("sender_id")
    refined_queries = state.get("refined_queries", [])

    logger.info(
        "[Retrieval Service] start | sender_id=%s | refined_queries=%d",
        sender_id, len(refined_queries or []),
    )

    all_results = []
    top_score = 0.0

    for refined_query in refined_queries:
        search = run_search([refined_query.query])

        logger.info(
            "[Retrieval Service] query=%r | results=%d | top_score=%s",
            refined_query.query, len(search["results"]), search["top_score"],
        )

        all_results.extend(search["results"])
        top_score = max(top_score, search["top_score"])

    context = build_context(all_results)

    logger.info(
        "[Retrieval Service] done | sender_id=%s | total_results=%d | top_score=%s | context_chars=%d",
        sender_id, len(all_results), top_score, len(context),
    )

    return RetrievalResult(
        results=all_results,
        context=context,
        top_score=top_score,
    )