from langchain_mcp_adapters.client import MultiServerMCPClient

from untils.env_utils import STOCK_MCP_URL

mcp_client = MultiServerMCPClient(
    {
        "stock":{
            "transport":"streamable_http",
            "url":STOCK_MCP_URL
        }
    }
)
"""
工具名称：query_stock_quote
工具参数：{'stock': {'title': 'Stock', 'type': 'string'}}
工具描述:查询个股实时行情，支持股票代码（如 sh600519、600519）或名称（如 贵州茅台）。返回现价、涨跌额、涨跌幅、今开、昨收、最高、最低、成交量等数据。
工具名称：query_index
工具参数：{}
工具描述:查询A股三大指数实时行情，包括上证指数、深证成指、创业板指的点位、涨跌幅和市场整体趋势。测试通过，现在我要将他添加进入进入我的mcp
"""
async def get_stock_tools():
    tools = await mcp_client.get_tools()

    stock_tools = [
        tool for tool in tools
    ]
    return stock_tools
