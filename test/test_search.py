from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, RRFRanker

from documents.markdown_parser import MarkdownParser
from documents.milvus_db import MilvusVectorSave
from llm_model.embedings_model import qwen_embedding
from untils.env_utils import MILVUS_URI, COLLECTION_NAME


def test1():
    # 第一种
    mv = MilvusVectorSave()
    mv.connection(is_first=False)
    # result = mv.vector_store_saved.similarity_search(
    result = mv.vector_store_saved.similarity_search_with_score(
        query='光刻胶',
        k=2,
        expr='category == "content"'
    )
    #similarity_search_with_score:返回的是一个二元组的列表，二元组中的二个值：分数

    for doc in result:
        print(doc)

def test2():
    """创建一个新的collection"""
    client = MilvusClient(uri=MILVUS_URI)

    schema = client.create_schema()

    schema.add_field(field_name='id',datatype=DataType.INT64,is_primary=True,auto_id=True)
    schema.add_field(field_name='text',datatype=DataType.VARCHAR,max_length=6000,enable_analyzer=True,analyzer_params={"tokenizer":"jieba","filter":["cnalphanumonly"]})
    schema.add_field(field_name='category',datatype=DataType.VARCHAR,max_length=1000)
    schema.add_field(field_name='sparse',datatype=DataType.SPARSE_FLOAT_VECTOR)

    bm25_function = Function(
        name='text_bm25_emb',
        input_field_names=["text"],
        output_field_names=["sparse"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(function=bm25_function)
    # 创建一个索引参数配置对象
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="sparse",  # 指定在哪个字段上建立索引
        index_name="sparse_inverted_index",  # 索引的名称
        index_type="SPARSE_INVERTED_INDEX",  # 索引类型，稀疏向量倒排索引
        metric_type="BM25",  # 采用BM25相关性评分
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",  # 查找最相关的Top-k文档时，尽可能跳过不太可能进入结果集的候选文档，从而提高查询效率的算法
            "bm25_k1": 1.6,  # 控制词频对BM25评分的影响，以及词频增加时的饱和程度
            "bm25_b": 0.75  # 控制文档归一化的强度
        },
    )
    if 'demo' in client.list_collections():
        # 先释放，再删除索引，再删除collection
        client.release_collection(collection_name='demo')
        client.drop_index(collection_name='demo', index_name='dense_vector_index')
        client.drop_index(collection_name='demo', index_name='sparse_inverted_index')
        client.drop_collection(collection_name='demo')
    # 创建一张表
    client.create_collection(
        collection_name='demo',
        schema=schema,
        index_params=index_params
    )
def test3():
    """往collection中插入数据"""
    vector_store = Milvus(
        embedding_function=None,
        collection_name='demo',
        builtin_function=BM25BuiltInFunction(),
        vector_field=['sparse'],
        consistency_level = "Strong",  # 读写一致性，Strong,Session(默认),Bounded,Eventually
        auto_id = True,
        connection_args = {"uri": MILVUS_URI}
    )

    file_path = r"/datas/md/tech_report_0tfhhamx.md"
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)

    vector_store.add_documents(docs)
def test4():
    """数据检索"""
    vector_store = Milvus(
        embedding_function=None,
        collection_name='demo',
        builtin_function=BM25BuiltInFunction(),
        vector_field=['sparse'],
        consistency_level="Strong",  # 读写一致性，Strong,Session(默认),Bounded,Eventually
        auto_id=True,
        connection_args={"uri": MILVUS_URI}
    )
    res = vector_store.similarity_search_with_score(
        query='活性氧原子',
        k=2,
    )
    for doc in res:
        print(doc)
def test5():
    """使用原生api进行数据检索"""
    client = MilvusClient(uri=MILVUS_URI)
    res = client.search(
        collection_name='demo',
        data=["半导体改善表面特征"],
        anns_field="sparse",
        output_fields=['text','id'],
        limit=3,
    )
    for doc in res:
        print(doc)

def test7():
    """pymilvus的混合检索"""
    client = MilvusClient(uri=MILVUS_URI)


    search_params_1 = {
        'data':[qwen_embedding.embed_query('有机半导体')],
        'anns_field':'dense',
        "param":{
            "metric_type":"IP",
            "params":{"ef":32},
        },
        "limit":5,
    }
    req1 = AnnSearchRequest(**search_params_1)  # 密集向量搜索

    search_params_2 = {
        'data': ['有机半导体'],
        'anns_field': 'sparse',
        "param": {
            "metric_type": "BM25",
        },
        "limit": 5,
    }
    req2 = AnnSearchRequest(**search_params_2)  # 稀疏向量搜索
    res = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[req1, req2],
        ranker=RRFRanker(60),
        limit=5,
        output_fields=['text'],
    )

    for hits in res:
        print('topN的结果')
        for item in hits:
            print(item)

def test8():
    """使用langchain,对milvus进行混合检索"""
    mv = MilvusVectorSave()
    mv.connection()
    # res=mv.vector_store_saved.similarity_search_with_score(
    #     query='有机半导体材料',
    #     k=3,
    #     ranker_type='rrf',
    #     ranker_params={"k":100}
    # )

    retriever = mv.vector_store_saved.as_retriever(
        search_type='similarity',
        search_kwargs={
            'k':3,
            "score_threshold":0.01,#阈值
            "ranker_type":"rrf",
            "ranker_params" : {"k": 100},
            'filter':{"category":"content"}
        }
    )

    res = retriever.invoke('有机污染物的处理办法')
    for doc in res:
        print(doc)



if __name__ == '__main__':
    test8()