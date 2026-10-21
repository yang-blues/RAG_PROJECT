from langchain_core.tools import create_retriever_tool

from documents.milvus_db import MilvusVectorSave

mv = MilvusVectorSave()
mv.connection()
retriever = mv.vector_store_saved.as_retriever(
    search_type='similarity',
    search_kwargs={
        'k': 2,
        "score_threshold": 0.01,  # 阈值
        "ranker_type": "rrf",
        "ranker_params": {"k": 100},
        'filter': {"category": "content"}
    }
)

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name='rag_retriever',
    description='搜索并返回关于‘半导体和光刻胶’的信息，内容包含：半导体，光刻胶等等'
)
