import json

from fastapi import APIRouter, Request, HTTPException
from langgraph.types import Command
from starlette.responses import StreamingResponse

from api.user_approval_dto import ApprovalRequest
from api.user_chat_dto import ChatRequest
from graph_fin.compliance import build_approval_card

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
            if message.content or getattr(message, "tool_calls", None):
                event_data = {
                    "content": message.content,
                    "node": metadata.get("langgraph_node"),
                    "message_type": message.type,
                    "id": message.id,
                    "tool_calls": getattr(message, "tool_calls", [])
                }
                #yield交出控制权，挂起函数，等待下一次 `anext()` 唤醒
                #SSE标准格式：data:xxx\n\n   把python对象装换成json格式，中文不会乱码
                yield f"data:{json.dumps(event_data,ensure_ascii=False)}\n\n"
        #检查图是否发生人工审批中断
        #1.获取当前图状态
        snapshot = await graph.aget_state(config)
        #2.判断是否发生interrupt中断
        has_interrupt = any(
            task.interrupts
            for task in snapshot.tasks
        )
        if has_interrupt:

            #获取当前状态
            state = snapshot.values

            #组装审批卡片
            approval_card = build_approval_card(
                state,
                thread_id=data.thread_id,
            )
            yield ("event:approval\n"
                   f"data:{json.dumps(approval_card, ensure_ascii=False)}\n\n")
        else:
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

@router.post("/approval")
async def approval(data:ApprovalRequest,request:Request):

    #获取已经编译好的LangGraph
    graph = request.app.state.graph

    #构建配置，必须使用原来的tgread_id
    config = {
        "configurable":{
            "thread_id":data.thread_id,
            "user_id":data.user_id,
        }
    }
    #检查当前线程是否确实处于中断状态
    snapshot = await graph.aget_state(config)
    has_interrupt = any(
        task.interrupts
        for task in snapshot.tasks
    )
    if not has_interrupt:
        raise HTTPException(
            status_code=409,
            detail="当前会话没有待处理的审批请求",
        )

    #构建恢复命令
    command = Command(resume=data.decision)

    async def event_generator():

        # 恢复图并继续流式输出
        async for message, metadata in graph.astream(
                command,
                config=config,
                stream_mode="messages"
        ):

            if message.content or getattr(message, "tool_calls", None):
                event_data = {
                    "content": message.content,
                    "node": metadata.get("langgraph_node"),
                    "message_type": message.type,
                    "id": message.id,
                    "tool_calls": getattr(message, "tool_calls", [])
                }

                yield (
                    f"data:{json.dumps(event_data, ensure_ascii=False)}\n\n"
                )

        # 检查恢复后是否又发生了中断
        new_snapshot = await graph.aget_state(config)

        has_new_interrupt = any(
            task.interrupts
            for task in new_snapshot.tasks
        )

        if has_new_interrupt:

            approval_card = build_approval_card(
                new_snapshot.values,
                thread_id=data.thread_id,
            )

            yield (
                "event:approval\n"
                f"data:{json.dumps(approval_card, ensure_ascii=False)}\n\n"
            )

        else:
            yield "event:done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

