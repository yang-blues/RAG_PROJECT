import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    #创建mcp客户端
    client = MultiServerMCPClient(
        {
            "stock": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:8001/mcp"
            }
        }
    )

    #获取MCP服务提供的工具
    tools = await client.get_tools()
    print("成功连接Stock Mcp!")
    print("\n当前MCP提供的工具")
    for tool in tools:
        print(f"工具名称：{tool.name}")
        print(f"工具参数：{tool.args}")
        print(f"工具描述:{tool.description}")

if __name__ == "__main__":
    asyncio.run(main())