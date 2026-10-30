"""结构化输出模型：委派靠 Pydantic，不解析模型说的话。

为什么要结构化输出：条件边读的是 tool_calls[0]["name"]，是**确定性路由**。
模型多说一句"我帮你转接"也不会让解析失败——这是文本路由做不到的。
"""
from typing import Literal

from pydantic import BaseModel, Field


class CompleteOrEscalate(BaseModel):
    """
    子助理用他把控制权交给主助理
        场景：用户改主意 / 缺关键信息 / 当前工具都不适用。
    这是**显式出栈**，不是异常兜底——子助理搞不定就交还，不要让它在里面硬编。
    """
    cancel: bool = True  # 默认取消当前任务
    reason: str  # 取消或升级的原因

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "用户改变了主意，想先看看行情再决定。",
            },
            "example2": {
                "cancel": True,
                "reason": "缺少产品代码，无法继续下单。",
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
