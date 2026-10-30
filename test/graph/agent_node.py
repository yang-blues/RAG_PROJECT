from test.graph.graph_state1 import AgentState
from llm_model.all_llm import llm
from test.retriever_tools import retriever_tool
from untils.log_utils import log


#agent 节点
def agent_node(state:AgentState):
    """
    调用智能体模型基于当前状态生成响应，根据问题，
    他会决定使用检索工具检索，或者直接结束
    :param state: 当前状态
    :return: 更新后的状态，包含附加到消息中的智能体响应
    """
    log.info("---调用智能体---")
    messages = state["messages"]

    tools = [retriever_tool,]
    model = llm.bind_tools(tools)
    response = model.invoke(messages)
    #返回列表，因为这会添加到现有列表中
    return {"messages":[response]}