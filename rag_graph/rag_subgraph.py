"""自适应 RAG 子图：路由 → 检索 → 评级 → 改写 → 联网兜底 → 生成（带引用）"""
from typing import Literal

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel, Field

from rag_graph.grade_documents_node import grade_documents
from rag_graph.rag_state import GraphState
from rag_graph.transform_query_node import transform_query
from rag_graph.web_search_node import web_search
from llm_model.all_llm import llm
from search.hybrid_retriever import get_research_retriever
from untils.log_utils import log

#最多改写几次查询。再多也不会更好，只会拖慢响应
MAX_TRANSFORM = 2

#路由的架构化输出类
class RouteQuery(BaseModel):
    """把问题路由到知识库还是联网搜索"""
    datasource: Literal["vectorstore","web_search"] = Field(
        description="根据用户问题选择向量知识库或联网搜索"
    )
structured_llm_route = llm.with_structured_output(RouteQuery)
route_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是金融信息检索的路由专家。\n"
     "【走知识库】产品的静态规则与条款：费率、赎回规则、基金合同、产品要素、"
     "招募说明书、适当性与合规法规、历史定期报告。\n"
     "【走联网】时效性信息：当前净值、实时价格、最新公告、盘中新闻、宏观事件、"
     "尚未入库的新规。\n"
     "判断标准只有一个：答案会不会随时间变化。会变 → 联网；不会变 → 知识库。"),
    ("human","{question}")
])
question_route_chain = route_prompt | structured_llm_route

def route_query(state: GraphState):
    """路由节点：决定走知识库还是联网"""
    log.info("---判断走知识库还是联网")
    question = state["question"]
    res = question_route_chain.invoke({"question":question})
    log.info(f"路由结果:{res.datasource}")
    return res.datasource

#检索
def retrieve(state: GraphState):
    """检索节点：走混合检索器（dense+BM25 加权RRF+MMR）"""
    log.info("---混合检索---")
    question = state["question"]
    docs = get_research_retriever().search(question)
    log.info(f"召回{len(docs)}条文档")
    return {"documents":docs,"question":question}

#生成
generate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是金融投研助理。只依据下面提供的上下文回答，禁止使用任何上下文之外的知识。\n"
     "硬性要求：\n"
     "1. 每个结论后面用 [序号] 标注它来自哪一段上下文；\n"
     "2. 上下文里没有的信息，直接回答“未检索到相关依据”，不要推测、不要泛泛而谈；\n"
     "3. 涉及费率、日期、金额时原样引用原文数字，不要换算或四舍五入；\n"
     "4. 不做投资建议，不承诺收益。"),
    ("human","问题：{question}\n\n上下文：\n{context}"),
])

generate_chain = generate_prompt | llm | StrOutputParser()

def format_docs(docs) -> str:
    """把Document 列表拼成带序号和出处的上下文，序号就是引用标号"""
    if isinstance(docs,Document):#判断传入的和指定的数据类型是不是一样的
        return docs.page_content
    blocks = []
    for i, d in enumerate(docs or [], 1):#遍历获取下表和元素
        src = d.metadata.get("source") or d.metadata.get("fund_short") or "未知来源"
        blocks.append(f"[{i}] 来源: {src}\n{d.page_content}")
    return "\n\n".join(blocks)

def generate_with_citation(state: GraphState):
    """生成节点，如果没有从document查到数据就诚实回答
    """
    log.info("---生成回答（强制引用溯源）---")
    question = state["question"]
    documents = state.get("documents") or []
    if not documents:
        return {"generation": "未检索到相关依据，无法回答该问题。",
                "question": question, "documents": documents}
    generation = generate_chain.invoke(
        {"question": question, "context": format_docs(documents)})
    return {"generation": generation, "question": question, "documents": documents}

#条件边
def decide_to_generate(state: GraphState):
    """评级之后：有合格文档就生成，没有就改写查询重试（超过次数转联网）。"""
    if state.get("documents"):
        return "generate"
    if state.get("transform_count", 0) >= MAX_TRANSFORM:
        log.info("改写已达上限，转联网兜底")
        return "web_search"
    log.info("文档全部不合格，改写查询")
    return "transform_query"
# ------------------------------------------------------------------ 组装
def build_rag_subgraph():
    """编译自适应 RAG 子图。

    返回:
        编译后的 LangGraph 图，输入 {"question": str}，输出含 "generation"。
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate_with_citation)

    workflow.add_conditional_edges(START, route_query,
                                   {"vectorstore": "retrieve",
                                    "web_search": "web_search"})
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate,
                                   {"generate": "generate",
                                    "transform_query": "transform_query",
                                    "web_search": "web_search"})
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


# ------------------------------------------------------------------ 包成工具
@tool
def lookup_fund_knowledge(question: str) -> str:
    """检索投研知识库并给出带引用的回答。

    适用于：基金产品要素（费率、托管人、基金经理、业绩比较基准）、基金合同条款、
    招募说明书、产品资料概要、定期报告内容、金融法规条文。
    不适用于：实时净值、当前价格、盘中新闻这类时效信息（那些归行情助理管）。
    返回内容是带 [序号] 引用的答案；依据不足时会明确说"未检索到相关依据"。
    """
    graph = _get_rag_graph()
    result = graph.invoke({"question": question,
                           "documents": [], "transform_count": 0})
    return result.get("generation") or "未检索到相关依据。"


_rag_graph = None


def _get_rag_graph():
    """子图单例：编译一次即可，重复编译只是浪费。"""
    global _rag_graph#可修改使用模块级别的变量
    if _rag_graph is None:
        _rag_graph = build_rag_subgraph()
    return _rag_graph


RESEARCH_SAFE_TOOLS = [lookup_fund_knowledge]

