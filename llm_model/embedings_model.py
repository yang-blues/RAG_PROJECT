from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from untils.env_utils import QWEN_API_KEY, QWEN_BASE_URL

# 阿里云百炼 embedding 单次请求有双重上限：
#   条数上限约 25~40，且整批 token 总量上限约 2 万（超出报 400 batch size is invalid）。
# chunk_size 控制 embed_documents 的自动分批，默认是 1000，必然触发 400。
# 它同时保护 SemanticChunker —— 语义切割时会对所有句子批量 embed，同样受此限制。
EMBEDDING_BATCH_SIZE = 10

qwen_embedding = OpenAIEmbeddings(
    model='qwen3.7-text-embedding-flash',
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    check_embedding_ctx_length=False,
    chunk_size=EMBEDDING_BATCH_SIZE
)


# beg_hf_embeddings = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-large-en-v1.5",
#     model_kwargs={
#         "device": "cuda"
#     },
#     encode_kwargs={
#         #是否进行向量的长度归一化，False保留原始向量
#         "normalize_embeddings": True
#     }
# )