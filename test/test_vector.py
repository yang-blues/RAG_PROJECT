from pymilvus import MilvusClient, DataType, Function, FunctionType

client = MilvusClient(uri='http://127.0.0.1:19530')

# schema可以理解为数据库中一个Collection的表结构，决定了每条数据有哪些字段，以及储存哪些内容
schema = client.create_schema()
#id作为唯一标识
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
#保存原始文本，供BM25分析 enable_analyzer=True为字段启用文本分析能力，供后续相关检索功能使用
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2000, enable_analyzer=True)
#保存BM25Function自动生成的稀疏向量
schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)  # 存储稀疏嵌入后的值

#定义BM25 Function
bm25_function = Function(
    name = "text_bm25_emb",  #函数名
    input_field_names = ["text"],#输入字段
    output_field_names=["sparse"],#输出字段
    function_type=FunctionType.BM25,#指定使用BM25算法
)
#注册到Collection Schema的内置处理函数
schema.add_function(bm25_function)
#创建一个索引参数配置对象
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="sparse",#指定在哪个字段上建立索引
    index_name="sparse_inverted_index",#索引的名称
    index_type="SPARSE_INVERTED_INDEX",#索引类型，稀疏向量倒排索引
    metric_type="BM25",#采用BM25相关性评分
    params={
        "inverted_index_algo": "DAAT_MAXSCORE",#查找最相关的Top-k文档时，尽可能跳过不太可能进入结果集的候选文档，从而提高查询效率的算法
        "bm25_k1": 1.6,#控制词频对BM25评分的影响，以及词频增加时的饱和程度
        "bm25_b": 0.75#控制文档归一化的强度
    },
)
# 创建一张表
client.create_collection(
    collection_name='t_demo2',
    schema=schema,
    index_params=index_params
)

# 插入测试数据
client.insert('t_demo2', [
    {'text': 'information retrieval is a field of study.'},
    {'text': 'information retrieval focuses on finding relevant information in large datasets.'},
    {'text': 'data mining and information retrieval overlap in research.'},
])


# 开始进行匹配搜索（全文搜索）
search_params = {
    'params': {'drop_ratio_search': 0.2},  # 搜索时要忽略的低重要性词语的比例： 查询向量中最小的 20% 值将在搜索过程中被忽略。
}

resp = client.search(
    collection_name='t_demo2',
    data=['whats the focus of information retrieval?'],
    anns_field='sparse',  # 匹配的稀疏向量字段
    limit=3,
    search_params=search_params,
    output_fields=["text"]
)

print(resp)
