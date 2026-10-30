"""主图组装：把三个子助理挂到主助理上，编译出可服务的图"""
from langgraph.graph import StateGraph,START,END

from graph_fin.agent_assistant import primary_runnable, primary_tools
from graph_fin.assistant import FinAssistant
from graph_fin.build_child_graph import build_research_graph, build_market_graph, build_trade_graph
from graph_fin.get_user_profile_node import get_user_profile
from graph_fin.route_fin import route_primary_assistant, route_to_workflow
from graph_fin.state import State
from tools.tools_handler import create_tool_node_with_fallback

#需要人工审批的节点，只有它会产生真实的副作用
INTERRUPT_BEFORE = ["trade_sensitive_tools"]

def build_graph() -> StateGraph:
    """组装主图（未编译）。

    返回:
        StateGraph: 组装好的图对象。
    """
    builder = StateGraph(State)
    builder.add_node("fetch_user_profile",get_user_profile)
    builder.add_edge(START,"fetch_user_profile")
    builder = build_research_graph(builder)
    builder = build_market_graph(builder)
    builder = build_trade_graph(builder)

    builder.add_node("primary_assistant", FinAssistant(primary_runnable))
    builder.add_node("primary_assistant_tools",
                     create_tool_node_with_fallback(primary_tools))

    builder.add_conditional_edges(
        "primary_assistant", route_primary_assistant,
        ["enter_research", "enter_market", "enter_trade",
         "primary_assistant_tools", END])
    builder.add_edge("primary_assistant_tools", "primary_assistant")

    builder.add_conditional_edges("fetch_user_profile", route_to_workflow)
    return builder

