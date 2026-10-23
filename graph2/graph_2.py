
from langgraph.graph import StateGraph,START,END

from graph2.generate_node2 import generate
from graph2.grade_answer_chain import answer_grader_chain
from graph2.grade_documents_node import grade_documents
from graph2.grade_hallucinations_chain import hallucination_grader_chain
from graph2.graph_state2 import GraphState
from graph2.query_route import query_route
from graph2.retriever_node import retrieve
from graph2.transform_query_node import transform_query
from graph2.web_search_node import web_search
from tools.draw_png import draw_graph
from untils.log_utils import log



def decide_to_generate(state: GraphState):
    """决定生成回答还是重新优化"""
    documents = state["documents"]
    if documents:
        log.info("生成最终回答")
        return "generate"
    else:
        log.info("所有文档都与问题无关，将优化查询问题")
        return "transform_query"
def grade_generation_v_documents_and_question(state):
    """
    评估生成结果是否基于文档并正确回答问题
    Args:
        state (dict): 当前图状态，包含问题、文档和生成结果
    Returns:
        str: 下一节点的名称（useful/not useful/not supported）
    """
    log.info("---检查生成内容是否存在幻觉---")  # 阶段标识
    question = state["question"]  # 获取用户问题
    documents = state["documents"]  # 获取参考文档
    generation = state["generation"]  # 获取生成结果
    transform_count=state["transform_count"]

    # 检查生成是否基于文档
    score = hallucination_grader_chain.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    if grade == "yes":  # 如果生成基于文档
        log.info("---判定：生成内容基于参考文档---")
        # 检查是否准确回答问题
        log.info("---评估：生成回答与问题的匹配度---")
        score = answer_grader_chain.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":  # 如果正确回答问题
            log.info("---判定：生成内容准确回答问题---")
            return "useful"  # 返回有用结果
        else:  # 如果没有回答问题
            log.info("---判定：生成内容未能准确回答问题---")
            if transform_count >= 3:
                return "web_search"
            return "not useful"  # 返回无用结果
    else:  # 如果生成不基于文档
        log.info("---判定：生成内容未基于参考文档，将重新尝试---")
        return "not supported"  # 返回不支持结果

#初始化工作流图4
workflow = StateGraph(GraphState)

#定义各个节点
workflow.add_node("web_search",web_search)#网络检索的节点
workflow.add_node("retrieve",retrieve)#向量检索的节点
workflow.add_node("grade_document",grade_documents)#评审过滤符合要求的document
workflow.add_node("generate",generate)#结合问题和document生成回答的节点
workflow.add_node("transform_query",transform_query)#查询优化节点

workflow.add_conditional_edges(
    START,query_route,path_map={
        'web_search':'web_search',
        'vectorstore':'retrieve',
    })

#添加固定边
workflow.add_edge("web_search","generate")#网络检索之后直接直接生成结果
workflow.add_edge("retrieve","grade_document")#检索之后评审

workflow.add_conditional_edges("grade_document",decide_to_generate,{
    "generate":"generate",
    "transform_query":"transform_query"
})
workflow.add_edge("transform_query","retrieve")

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported":"generate",#幻觉
        "useful":END,
        "not useful":"transform_query",#生成无用的结果，优化查询
        "web_search":"web_search"
    }

)

# 编译工作流
graph = workflow.compile()

# draw_graph(graph, 'graph_rag2.png')



#执行工作流
while True:
    question = input('用户: ')
    if question.lower() in ['q','exit','quit']:
        log.info('对话结束')
        break
    else:
        inputs = {
            "question":question
        }
        #打印消息
        res = graph.invoke(inputs)
        print(res.get('generation'))



# 执行工作流

# _printed = set()  # set集合，避免重复打印
#
# # 执行工作流
# while True:
#     question = input('用户：')
#     if question.lower() in ['q', 'exit', 'quit']:
#         print('对话结束，拜拜！')
#         break
#     else:
#         inputs = {
#             "question": question
#         }
#         # 流式执行工作流
#         for output in graph.stream(inputs):
#             for key, value in output.items():
#                 # 打印当前节点名称
#                 pprint(f"Node '{key}':")  # 显示当前执行的节点名称
#                 # 可选：打印每个节点的完整状态信息
#                 # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
#             pprint("\n---\n")  # 节点分隔线
#
#         # 打印最终生成结果
#         pprint(value["generation"])  # 输出最终生成的回答内容