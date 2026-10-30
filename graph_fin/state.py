from typing import Optional, TypedDict, Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


def update_dialog_stack(left : list[str],right:Optional[str]) -> list[str]:
    """
    更新对话状态栈。

    参数:
        left (list[str]): 当前的状态栈。
        right (Optional[str]): 想要添加到栈中的新状态或动作。None 表示不做任何更改；
                               "pop" 表示弹出栈顶元素；其他值表示入栈。
    返回:
        list[str]: 更新后的状态栈。
    """
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]

class State(TypedDict):
    """
    主图的状态定义：
    字段：
        message:对话消息列表，用add_message reducer 做追加和更新
        user_profile:用户画像（风险等级，账户，适当性）。它是**结构化事实**
                    每轮进图时重新查库写入，不让模型从对话历史回忆
        dialog_state:对话状态栈，元素被Literal锁死四个值
        compliance_decision:合规核查的三种结果pass / block / insufficient_evidence。
        compliance_evidence:合规证据文本，进审批卡片给用户核对
        compliance_ticket:合规通过时签发的一次性凭证，下单函数体内校验
        compliance_warnings:集中度提示这类不拦截但要说的警告
    """
    messages: Annotated[list[AnyMessage],add_messages]
    user_profile:str
    dialog_state:Annotated[
        list[Literal["assistant","research","market","trade"]],
        update_dialog_stack,
    ]
    compliance_decision: str
    compliance_evidence: str
    compliance_warnings: list
    compliance_ticket: dict


