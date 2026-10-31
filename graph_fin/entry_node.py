from typing import Callable

from langchain_core.messages import ToolMessage


def create_entry_node(assistant_name:str,new_dialog_state:str) -> Callable:
    """
    工厂函数：创建一个"进入子助理"的入口节点。

    入栈时注入一条 ToolMessage，交代三件事：你是谁、用户意图还没满足、
    什么情况下才交还。子助理是**带着上下文接管**的，不是凭空醒来。

    注意这里的措辞就是子助理的**行为契约**：早期版本写的是"缺少关键信息
    就 CompleteOrEscalate"，结果模型一碰到"没说金额"就出栈，把追问退化成
    回主助理重新分类。缺参数属于本助理的分内事，必须明确排除掉。

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
                        f"用户的意图尚未满足，使用你手上的工具继续协助用户。"
                        f"记住你是{assistant_name}。\n"
                        "只有在下面两种情况才调用 CompleteOrEscalate 交还主助理：\n"
                        "  ① 用户明确放弃、改变主意或换话题；\n"
                        "  ② 用户要办的事超出你的能力范围。\n"
                        "**缺少参数（金额、代码、期限等）不算超出能力**——"
                        "直接用一句话向用户追问，留在本助理继续。\n"
                        "任务在你成功调用相应工具之前不算完成。不要向用户提及你的名字。"
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state":new_dialog_state,
        }
    return entry_node
