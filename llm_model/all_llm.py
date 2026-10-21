from langchain_openai import ChatOpenAI

from untils.env_utils import QWEN_API_KEY, QWEN_BASE_URL

llm = ChatOpenAI(
    model='qwen3.8-max-0902',
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
)

# res = llm.invoke('你好')
# print(res)