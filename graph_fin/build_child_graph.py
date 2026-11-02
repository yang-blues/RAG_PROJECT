"""三个子助理的子图构建+统一出口leave_skill
控制流是"主 → 子 → 主"的星形：只有主助理能入栈（委派），子助理只能出栈
（CompleteOrEscalate），禁止横向跳转。这样就不会出现"交易助理直接跳投研助理"
这种失控的路由。
"""
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph,END
from langgraph.types import interrupt, Command

from graph_fin.agent_assistant import research_runnable, market_runnable, trade_runnable, trade_safe_tools, \
    trade_sensitive_tools
from graph_fin.assistant import FinAssistant
from graph_fin.compliance import compliance_node, route_after_compliance
from graph_fin.entry_node import create_entry_node
from graph_fin.route_fin import route_research, route_market, route_after_trade_assistant
from rag_graph.rag_subgraph import RESEARCH_SAFE_TOOLS
from tools.fin_tools import MARKET_SAFE_TOOLS
from tools.tools_handler import create_tool_node_with_fallback


def _pop_dialog_state(state:dict)->dict:
    """弹出对话栈，交还主助理。全图只有这一个收敛点。

    返回:
        dict: 新的 dialog_state（pop）与一条给模型的提示消息。
    """
    messages = []
    if state["messages"][-1].tool_calls:
        #只处理第一个 tool_call
        messages.append(
            ToolMessage(
                content="正在恢复与主助理的对话。请回顾之前的对话并根据需要协助用户。",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {"dialog_state": "pop", "messages": messages}
# ------------------------------------------------------------------ 投研子图
def build_research_graph(builder: StateGraph) -> StateGraph:
    """投研助理子图，条款与研报问答，只有只读工具"""
    builder.add_node("enter_research",
                     create_entry_node("投研助理","research"))
    builder.add_node("research",FinAssistant(research_runnable))
    builder.add_node("research_tools",create_tool_node_with_fallback(RESEARCH_SAFE_TOOLS))
    builder.add_edge("enter_research", "research")
    builder.add_edge("research_tools", "research")
    builder.add_conditional_edges("research", route_research,
                                  ["research_tools", "leave_skill", END])

    # 统一出口：所有子助理共用这一个 leave_skill
    builder.add_node("leave_skill", _pop_dialog_state)
    builder.add_edge("leave_skill", "primary_assistant")
    return builder
# ------------------------------------------------------------------ 行情子图
def build_market_graph(builder: StateGraph) -> StateGraph:
    """行情助理子图：行情 / 持仓 / 账户，全部只读，不打断用户。"""
    builder.add_node("enter_market", create_entry_node("行情助理", "market"))
    builder.add_node("market", FinAssistant(market_runnable))
    builder.add_edge("enter_market", "market")
    builder.add_node("market_tools", create_tool_node_with_fallback(MARKET_SAFE_TOOLS))
    builder.add_edge("market_tools", "market")
    builder.add_conditional_edges("market", route_market,
                                  ["market_tools", "leave_skill", END])
    return builder
# ------------------------------------------------------------------ 交易子图
#人工介入
def approval_node(state:dict):
    #1.暂停执行，等待外部传入审批结果
    decision = interrupt({})
    #2.用户同意，执行交易工具
    if decision == "yes":
        return Command(goto="trade_sensitive_tools")
    #3.用户拒绝，找到原始工具调用
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    tool_messages = []
    for tool_call in tool_calls:
        tool_messages.append(
            ToolMessage(
                content="用户拒绝了本次交易请求，交易未执行。",
                tool_call_id=tool_call["id"]
            )
        )
    return Command(goto="trade",
                   update={"messages": tool_messages})

def build_trade_graph(builder: StateGraph) -> StateGraph:
    """交易助理子图：写操作前由条件边**强制插入**合规核查节点"""
    builder.add_node("enter_trade",create_entry_node("交易助理","trade"))
    builder.add_node("trade",FinAssistant(trade_runnable))
    builder.add_node("approval_node",approval_node)
    builder.add_edge("enter_trade", "trade")
    builder.add_node(
        "trade_safe_tools",
        create_tool_node_with_fallback(trade_safe_tools))
    builder.add_node("trade_sensitive_tools",
                     create_tool_node_with_fallback(trade_sensitive_tools))
    builder.add_node("compliance_check",compliance_node)

    builder.add_edge("trade_safe_tools", "trade")
    builder.add_edge("trade_sensitive_tools", "trade")
    builder.add_conditional_edges(
        "trade", route_after_trade_assistant,
        ["trade_safe_tools", "compliance_check",
         "leave_skill", END])
    # 合规三分支：只有 pass 进审批流，block / insufficient_evidence 直接打回助理
    builder.add_conditional_edges(
        "compliance_check", route_after_compliance,
        {"approval_node": "approval_node",
         "trade": "trade"})
    return builder



