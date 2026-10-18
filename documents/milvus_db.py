from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import MilvusClient, IndexType
from pymilvus.client.types import MetricType

from documents.markdown_parser import MarkdownParser
from llm_model.embedings_model import qwen_embedding
from untils.env_utils import MILVUS_URI, COLLECTION_NAME


class MilvusVectorSave:
    """把新的document数据插入到数据库中"""

    def __init__(self):
        """自定义collection的索引"""
        self.index_params = [
            {  # 密集向量字段
                "field_name": "dense",
                "index_name": "dense_vector_index",
                "index_type": IndexType.HNSW,
                "metric_type": MetricType.IP,
                "params": {"M": 16, "efConstruction": 64}
            },
            {
                "field_name": "sparse",
                "index_name": "sparse_inverted_index",
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",
                "params": {
                    "inverted_index_algo": "DAAT_MAXSCORE",
                    # Algorithm for building and querying the index. Val
                    "bm25_k1": 1.6,  # 范围: [1.2 ~ 2.0]
                    "bm25_b": 0.75
                }
            },
        ]
        self.vector_store_saved = None

    def create_collection(self,is_first=True):
        """创建一个collection"""
        if is_first:
            client = MilvusClient(uri=MILVUS_URI)
            #判断是否存在Collection
            if COLLECTION_NAME in client.list_collections():
                #先释放，再删除索引，再删除collection
                client.release_collection(collection_name=COLLECTION_NAME)
                client.drop_index(collection_name=COLLECTION_NAME,index_name='dense_vector_index')
                client.drop_index(collection_name=COLLECTION_NAME,index_name='sparse_inverted_index')
                client.drop_collection(collection_name=COLLECTION_NAME)
        self.vector_store_saved = Milvus(
             embedding_function=qwen_embedding,
            collection_name=COLLECTION_NAME,
            builtin_function=BM25BuiltInFunction(),
            vector_field=['dense','sparse'],
            index_params=self.index_params,
            consistency_level="Strong",#读写一致性，Strong,Session(默认),Bounded,Eventually
            auto_id=True,
            connection_args={"uri": MILVUS_URI}
        )
    def add_documents(self,datas:List[Document]):
        """将Document添加到向量数据库"""
        self.vector_store_saved.add_documents(datas)

if __name__ == '__main__':
    file_path = r"D:\ipython\RAG_PROJECT\datas\md\tech_report_0tfhhamx.md"
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)

    mv = MilvusVectorSave()
    mv.create_collection(is_first=True)
    mv.add_documents(docs)

    client = mv.vector_store_saved.client
    #得到表的结构
    desc_collection = client.describe_collection(
        collection_name=COLLECTION_NAME,
    )
    print('表结构是:',desc_collection)

    #得到当前表的所有索引
    res = client.list_indexes(
        collection_name=COLLECTION_NAME
    )
    print('表中的所有索引:',res)
    if res:
        for i in res:
            #得到索引的描述
            desc_index = client.describe_index(
                collection_name=COLLECTION_NAME,
                index_name=i)
            print(desc_index)

    result = client.query(
            collection_name=COLLECTION_NAME,
            filter="category == 'Title'",#查询 category == 'Title'所有数据
            output_fields=['text','category']#指定要返回的字段
        )
    print('测试，过滤查询的结果:',result)

