from graph.nodes.homevisit_node import visit_node
from graph.nodes.complaint_node import complaint_node
from graph.nodes.direct_node import direct_node
from graph.nodes.intent_node import intent_node
from graph.nodes.inquiry_node import inquiry_node
from graph.nodes.rag_node import rag_node
from graph.nodes.labresults_node import  result_node
from graph.state import AgentState

from langgraph.graph import StateGraph, END

__agent_graph = None


def route_intent(state: AgentState):
    return state.get("intent", "direct")


def build_graph():

    graph = StateGraph(AgentState)

    # 1. Nodes
    graph.add_node("intent", intent_node)
    graph.add_node("rag", rag_node)
    graph.add_node("visit", visit_node)
    graph.add_node("complaint", complaint_node)
    graph.add_node("inquiry", inquiry_node)
    graph.add_node("labresults", result_node)
    graph.add_node("direct", direct_node)

    # 2. Entry Point
    graph.set_entry_point("intent")

    # 3. Intent Routing
    graph.add_conditional_edges(
        "intent",
        route_intent,
        {
            "visit": "rag",
            "inquiry": "rag",
            "complaint": "complaint",
            "direct": "direct",
            "labresults": "labresults",
        },
    )

    # 4. After RAG Routing
    graph.add_conditional_edges(
        "rag",
        route_intent,
        {
            "visit": "visit",
            "inquiry": "inquiry",
        },
    )

    # 5. End Edges (تمت إضافة labresults هنا)
    graph.add_edge("visit", END)
    graph.add_edge("complaint", END)
    graph.add_edge("inquiry", END)
    graph.add_edge("direct", END)
    graph.add_edge("labresults", END)

    return graph.compile()


def get_agent_graph():
    global __agent_graph

    if __agent_graph is None:
        __agent_graph = build_graph()

    return __agent_graph