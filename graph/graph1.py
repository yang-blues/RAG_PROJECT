from typing import Literal

from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode, tools_condition

from graph.agent_node import agent_node
from graph.generate import generate
from graph.graph_state1 import AgentState, Grade
from graph.rewrite_node import rewrite
from llm_model.all_llm import llm
from tools.draw_png import draw_graph
from tools.get_last_humanmessage import get_lat_human_message
from tools.retriever_tools import retriever_tool
from untils.log_utils import log
from untils.print_utils import _print_event




def grade_document(state) -> Literal["generate","rewrite"]:
    """
    判断检索到的文档是否与问题相关
    :param state: 当前状态
    :return: 判断结果文档是否相关
    """
    log.info("---检查document的相关性---")
    #带有结构化输出的llm
    llm_with_structured = llm.with_structured_output(Grade)
    # 提示模板
    prompt = PromptTemplate(
        template="""你是一个评估检索文档与用户问题相关性的评分器。\n
                这是检索到的文档：\n\n {context} \n\n
                这是用户的问题：{question} \n
                如果文档包含与用户问题相关的关键词或语义含义，则评为相关。\n
                给出二元评分 'yes' 或 'no' 来表示文档是否与问题相关。""",
        input_variables=["context", "question"],
    )

    #处理链
    chain = prompt | llm_with_structured

    messages = state["messages"]
    last_message = messages[-1]

    question = get_lat_human_message(messages).content
    docs = last_message.content

    scored_result = chain.invoke({"question":question,"context":docs})
    score = scored_result.binary_score

    if score == "yes":
        log.info("---输出：文档相关---")
        return "generate"
    else:
        log.info("---输出:文档不相关---")
        return "rewrite"


#定义一个工作流
workflow = StateGraph(AgentState)

#添加节点
workflow.add_node('agent',agent_node)#进入时的第一个agent节点
workflow.add_node('retrieve',ToolNode([retriever_tool]))#rag检索的工具节点
workflow.add_node('rewrite',rewrite)#评审rag返回的检索的结果，没有通过重写用户输入的节点
workflow.add_node('generate',generate)#评审通过，结合llm输出

workflow.add_edge(START,'agent')
workflow.add_conditional_edges(
    'agent',
    tools_condition,
    {
        'tools':'retrieve',
        END:END
    })
workflow.add_conditional_edges(
    'retrieve',
    grade_document,
    {
        'generate':'generate',
        'rewrite':'rewrite'
    }
)
workflow.add_edge("rewrite","agent")
workflow.add_edge("generate",END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# draw_graph(graph,'graph_rag1-2.png')
_printed = set()#set集合避免重复打印
config = {
    "configurable":{
        "thread_id" :"123"
    }
}
#执行工作流
while True:
    question = input('用户: ')
    if question.lower() in ['q','exit','quit']:
        log.info('对话结束')
        break
    else:
        inputs = {
            "messages":[
                ("user",question),
            ]
        }
        events = graph.stream(inputs,stream_mode='values',config=config)
        #打印消息
        for event in events:
            _print_event(event,_printed)

