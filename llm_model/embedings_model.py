from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from untils.env_utils import QWEN_API_KEY, QWEN_BASE_URL

qwen_embedding = OpenAIEmbeddings(
    model='qwen3.7-text-embedding-flash',
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    check_embedding_ctx_length=False
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