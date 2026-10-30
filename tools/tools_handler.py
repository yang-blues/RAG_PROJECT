from langchain_core.runnables import RunnableLambda

from untils.log_utils import log


def create_tool_node_with_fallback(tools:list):
    """
    带兜底机制的工具节点调用
    参数:
        tools (list): 工具列表。
    返回:
        带回退机制的 ToolNode。
    """
    from langgraph.prebuilt import ToolNode
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def handle_tool_error(state) -> dict:
    """
    处理工具调用时发生错误，根据tool_call_id手动封装一条toolMessage
    参数:
        state (dict): 当前状态，含 error 与 messages。
    返回:
        dict: 新的消息列表。
    """
    error = state.get("error")
    tool_calls = state['messages'][-1].tool_calls
    log.warning(f"工具调用失败，回灌模型自愈：{repr(error)[:200]}")
    from langchain_core.messages import ToolMessage
    return {
        "messages":[
            ToolMessage(
                content=f"错误：{repr(error)}\n请修正您的错误",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def _print_event(event: dict, _printed: set, max_length=1500):
    """
    打印流式事件，用于本地调试与 Gradio 状态图快照。

    参数:
        event (dict): 事件字典。
        _printed (set): 已打印过的消息 id，避免重复打印。
        max_length (int): 消息最大长度，超出截断。
    """
    current_state = event.get("dialog_state")
    if current_state:
        print("当前处于: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... （已截断）"
            print(msg_repr)
            _printed.add(message.id)



