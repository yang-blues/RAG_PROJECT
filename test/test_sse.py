import asyncio
import httpx
from httpx_sse import aconnect_sse


async def main():

    async with httpx.AsyncClient(timeout=None) as client:

        async with aconnect_sse(
            client,
            "POST",
            "http://127.0.0.1:8000/chat",
            json={
                "message": "介绍一下基金投资需要关注哪些风险",
                "thread_id": "sse_test_001",
                "user_id": "user_001"
            }
        ) as event_source:

            async for event in event_source.aiter_sse():
                if event.event == "done":
                    print("\n生成结束")
                    break

                print(event.data, end="", flush=True)


asyncio.run(main())
