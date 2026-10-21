from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import Field, BaseModel


class AgentState(TypedDict):
    #add_messages 函数定义了应该如何处理状态更新
    #add_messages 表示追加（append）
    messages: Annotated[list[AnyMessage],add_messages]

#数据模型
class Grade(BaseModel):
    """相关性检查的二元评分"""
    binary_score: str = Field(description="相关性评分'yes'或'no'")