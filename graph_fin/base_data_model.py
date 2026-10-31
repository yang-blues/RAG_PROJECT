"""结构化输出模型：委派靠 Pydantic，不解析模型说的话。

为什么要结构化输出：条件边读的是 tool_calls[0]["name"]，是**确定性路由**。
模型多说一句"我帮你转接"也不会让解析失败——这是文本路由做不到的。
"""
from typing import Literal

from pydantic import BaseModel, Field


class CompleteOrEscalate(BaseModel):
    """结束当前子助理的会话，把控制权交还主助理（弹出 dialog_state）。

    **只在这两种情况调用：**
    1. 用户明确放弃或改变主意：如"算了""不买了""我再想想""先不办了"；
    2. 用户要办的事超出本助理的能力范围：如交易助理听到"帮我查今天涨多少"，
       投研助理听到"帮我下单"。

    **以下情况不要调用**，留在当前助理继续处理：
    - 缺少必要参数（金额、份额、产品代码、期限等）→ 直接用一句话向用户追问。
      这是本助理的分内事，交还回主助理只会让用户重复描述需求；
    - 用户在本助理能力内追问细节、要求改方案、要求再确认一遍；
    - 工具返回业务性失败（合规拦截、余额不足、低于起购金额、无检索结果）→
      解释原因并给出替代方案，不要就此甩回主助理。

    判断标准一句话：**自己能问清楚、能自己再试一次的，就不要交还。**
    频繁交还会让 dialog_state 失去意义，也让每一轮多花两次模型调用。
    """
    cancel: bool = Field(
        default=True,
        description="是否放弃当前任务。恒为 true，不需要修改。",
    )
    reason: str = Field(
        description=("交还主助理的原因，一句话。必须属于「用户放弃/改变主意」"
                     "或「超出本助理能力」两类之一。"
                     "不要写「缺少金额」「缺少产品代码」这类本可自行追问的理由。"),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "用户改变了主意，说先不买了。",
            },
            "example2": {
                "cancel": True,
                "reason": "用户转而咨询基金合同里的费率条款，超出交易助理负责范围。",
            },
        }
class ToResearchAssistant(BaseModel):
    """把工作转交给投研助理：产品条款、费率、基金要素、研报问答。"""
    request: str = Field(description="投研助理继续之前需要澄清的后续问题。")
    product_code: str = Field(
        default="",
        description="涉及的基金/股票代码，能确定就填，用于缩小检索范围。",
    )


class ToMarketAssistant(BaseModel):
    """把工作转交给行情助理：实时价格、持仓、账户余额这类结构化数据。"""
    symbols: str = Field(
        default="",
        description="用户关心的标的代码，多个用逗号分隔，如 '510300,000198'。",
    )
    request: str = Field(description="用户关于行情或持仓的具体诉求。")


class ToTradeAssistant(BaseModel):
    """把工作转交给交易助理：下单 / 撤单 / 调仓这类会产生副作用的操作。"""
    action: Literal["buy", "sell", "cancel", "rebalance"] = Field(
        description="用户想做的交易动作。"
    )
    product_code: str = Field(default="", description="交易标的代码。")
    amount: str = Field(default="", description="金额或份额，用户没说就留空。")
    request: str = Field(description="用户对这笔交易的额外要求。")


class ComplianceDecision(BaseModel):
    """合规核查的对外形态：结构化三态，给条件边和审批流消费，不是给人读的。

    三态而非二态，是因为"检索不到依据"必须被当成一等公民
    （insufficient_evidence），而不是降级成"让模型自己看着办"。
    金融场景里检索失败时让模型自由发挥，就是事故源头。
    """
    decision: Literal["pass", "block", "insufficient_evidence"] = Field(
        description="合规判定：pass 放行 / block 违规拦截 / insufficient_evidence 证据不足。"
    )
    reason: str = Field(default="", description="判定理由，违规时要写清违反了哪一条。")
    matched: bool = Field(default=False, description="风险等级是否匹配。")
