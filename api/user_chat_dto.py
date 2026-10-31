from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str #用户输入的消息
    thread_id: str #langgarph会话ID
    user_id: str #业务用户ID
