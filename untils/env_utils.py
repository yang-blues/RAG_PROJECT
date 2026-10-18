import os

from dotenv import load_dotenv

load_dotenv(override=True)

UNSTRUCTURED_KEY = os.getenv('UNSTRUCTURED_KEY')
QWEN_API_KEY=os.getenv("QWEN_KEY")
DEEP_API_KEY=os.getenv("DEEPSEEK_API_KEY")

MILVUS_URI='http://127.0.0.1:19530'
COLLECTION_NAME = 't_collection01'

QWEN_BASE_URL=os.getenv("QWEN_BASE_URL")
DEEPSEEK_BASE_URL=os.getenv("DEEPSEEK_BASE_URL")