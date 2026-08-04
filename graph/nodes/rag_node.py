import logging
from types import SimpleNamespace
from graph.state import AgentState
from search.search_manager import run_search
from rag.context_builder import build_context

logger = logging.getLogger(__name__)


def rag_node(state: AgentState):
 
    sender_id = state.get("sender_id")
    refined_queries = state.get("refined_queries") or []
 
    if refined_queries:
 
        
        queries = [rq for rq in refined_queries if rq and rq.query]
 
    else:
 
        
        queries = [SimpleNamespace(query=state["user_message"], aliases=[])]
 
    logger.info(
        "[RAG Node] start | sender_id=%s | query=%r",
        sender_id, [q.query for q in queries],
    )
 
    search = run_search(queries)
 
    results = search["results"]
 
    logger.info(
        "[RAG Node] done | sender_id=%s | results=%d | top_score=%s",
        sender_id, len(results), search["top_score"],
    )
 
    return {
        "rag_context": build_context(results),
        "search_results": results,
        "top_score": search["top_score"],
    }
 