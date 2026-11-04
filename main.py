from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from api.routes import router
from fin_mcp_server.market_mcp import get_stock_tools
from untils.env_utils import POSTGRES_URI
from untils.log_utils import log

#异步上下文装饰器
@asynccontextmanager
async def lifespan(app:FastAPI):
    # 1. 获取 MCP 行情工具
    stock_tools = await get_stock_tools()
    #2.创建postgreSQL的异步连接
    async with AsyncPostgresSaver.from_conn_string(POSTGRES_URI) as checkpointer:

        #3.初始化Checkpointer所需的数据表
        # await checkpoint.setup()
        #4.延迟导入，构建并编译Langgraph
        from graph_fin.graph_build import build_graph

        builder = build_graph(stock_tools=stock_tools)
        graph = builder.compile(checkpointer=checkpointer)

        #4.将编译后的图保存到FastAPI应用状态中
        app.state.graph = graph
        log.info("Langgraph初始化成功")



        yield

    log.info("Langgraph资源已释放")
app = FastAPI(lifespan=lifespan)

app.include_router(router)
@app.get("/")
async def root():
    return {"message": "FastAPI 服务运行成功"}
