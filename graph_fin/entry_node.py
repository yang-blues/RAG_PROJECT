from typing import Callable

from langchain_core.messages import ToolMessage


def create_entry_node(assistant_name:str,new_dialog_state:str) -> Callable:
    """
    工厂函数：创建一个"进入子助理"的入口节点。

    入栈时注入一条 ToolMessage，交代三件事：你是谁、用户意图还没满足、
    搞不定就 CompleteOrEscalate。子助理是**带着上下文接管**的，不是凭空醒来。

    :param assistant_name: 子助理的名字。
    :param new_dialog_state: 要压入状态栈的新状态。
    :return: 入口节点函数。
    """
    def entry_node(state:dict)->dict:
        """
        生成入场消息并更新对话状态栈。

        :param state:当前状态，包含所有信息
        :return: 含新消息和状态字典
        """
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages":[
                ToolMessage(
                    content=(
                        f"现在由{assistant_name}接管。请回顾上述主助理与用户之间的对话，"
                        f"用户的意图尚未满足，使用提供的工具继续协助用户。记住你是{assistant_name}，"
                        "并且任务在你成功调用相应工具之前不算完成。"
                        "如果用户改变主意、缺少关键信息或工具都不适用，"
                        "请调用 CompleteOrEscalate 把对话交还主助理。"
                        "不要向用户提及你的名字。"
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state":new_dialog_state,
        }
    return entry_node
