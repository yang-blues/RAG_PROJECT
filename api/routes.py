import json

from fastapi import APIRouter,Request
from starlette.responses import StreamingResponse

from api.user_chat_dto import ChatRequest

router = APIRouter()

@router.post("/chat")
async def chat(data:ChatRequest,request:Request):
    #获取已经编译好的LangGraph
    graph = request.app.state.graph

    #构建LangGraph配置
    config = {
        "configurable":{
            "thread_id":data.thread_id,
            "user_id":data.user_id,
        }
    }

    inputs = {
        "messages":[
            {"role":"user","content":data.message}
        ]
    }
    #生成SSE数据流的异步生成器函数
    async def event_generator():
        #遍历LangGraph流式输出的消息
        async for message,metadata in graph.astream(
            inputs,
            config,
            stream_mode="messages"
        ):
            #只处理模型生成的文本
            if message.content:
                event_data = {
                    "content":message.content,
                }
                #yield交出控制权，挂起函数，等待下一次 `anext()` 唤醒
                #SSE标准格式：data:xxx\n\n   把python对象装换成json格式，中文不会乱码
                yield f"data:{json.dumps(event_data,ensure_ascii=False)}\n\n"
        #通知前端本次生成结束
        yield "event:done\ndata: {}\n\n"
    #返回流式响应给前端（会不断拿取数据，直到没有）
    return StreamingResponse(
        event_generator(),#异步生成器
        media_type="text/event-stream",#流式
        headers={
            "Cache-Control": "no-cache",#不要缓存数据每次都实时拉
            "X-Accel-Buffering": "no" #Nginx反向代理关闭缓冲
        }
    )
