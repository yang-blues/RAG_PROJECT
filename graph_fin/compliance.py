import json
import time
from typing import Literal

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from graph_fin.ticket import issue_ticket, get_pending_ticket
from llm_model.all_llm import llm
from search.hybrid_retriever import get_compliance_retriever
from tools import fin_db
from tools.fin_tools import SENSITIVE_TOOL_NAMES
from untils.log_utils import log


def _amount(x) -> float:
    """把模型写出的 '5万' / '5万元' / '50000' 统一成数字。"""
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", "").replace("元", "")
    try:
        if s.endswith("万"):
            return float(s[:-1]) * 10_000
        if s.endswith("亿"):
            return float(s[:-1]) * 100_000_000
        return float(s)
    except ValueError:
        return 0.0


def extract_write_intent(messages: list):
    """从最后一条AI消息里找出危险工具调用，并且拿到调用危险工具的参数

    返回:
        dict | None: {tool, action, product_code, amount, client_order_id, tool_call_id}
    """
    last = messages[-1]
    calls = getattr(last,"tool_calls",None) or []
    for tc in calls:
        if tc.get("name") in SENSITIVE_TOOL_NAMES:
            args = tc.get("args") or {}
            return {
                "tool":tc["name"],
                "tool_call_id": tc.get("id"),
                "product_code": str(args.get("product_code")
                                    or args.get("order_id")
                                    or args.get("targets") or ""),
                "amount": _amount(args.get("amount", 0)),
                "client_order_id": args.get("client_order_id", ""),
            }
    return None

def ticket_matches(state:dict,intent:dict) -> bool:
    """路由用：State 里的票据是否还覆盖当前这次写操作。"""
    t = state.get("compliance_ticket") or {}
    if not t:
        return False
    if time.time() - t.get("issued_at", 0) > t.get("ttl", 0):
        return False
    if t.get("action") != intent.get("tool"):
        return False
    if str(t.get("product_code", "")) != str(intent.get("product_code", "")):
        return False
    if t.get("amount", 0) and intent.get("amount", 0):
        if abs(float(t["amount"]) - float(intent["amount"])) > 1.0:
            return False
    return True

def evaluate_rules(user_id: str, intent: dict):
    """规则引擎：确定性判定，不经过模型。

    返回:
        tuple: (decision, reason, matched, warnings)
        decision ∈ {"pass", "block", "insufficient_evidence"}
    """
    warnings = []
    account = fin_db.get_account(user_id)
    if not account:
        return ("insufficient_evidence",
                f"未检索到用户 {user_id} 的账户与适当性信息，无法完成匹配。",
                False, warnings)

    product_code = intent.get("product_code", "")
    # 撤单/调仓不走产品风险匹配，只校验订单存在与参数合法性
    if intent["tool"] == "cancel_order":
        return ("pass", f"撤单请求参数完整（订单 {product_code}）。", True, warnings)
    if intent["tool"] == "rebalance":
        try:
            plan = json.loads(intent.get("product_code") or "{}")
            if abs(sum(float(v) for v in plan.values()) - 1.0) > 0.01:
                return ("block", "调仓目标权重合计不等于 1.0，违反组合管理规则。",
                        False, warnings)
        except (json.JSONDecodeError, ValueError):
            return ("block", "调仓目标配置不是合法 JSON，无法校验。", False, warnings)
        return ("pass", "调仓目标权重合法。", True, warnings)

    product = fin_db.get_product(product_code)
    if not product:
        return ("insufficient_evidence",
                f"未检索到产品 {product_code} 的要素与风险等级，无法做适当性匹配。",
                False, warnings)

    # 规则一：风险等级匹配
    try:
        prod_level = int(str(product["risk_level"]).lstrip("Rr"))
    except ValueError:
        return ("insufficient_evidence",
                f"产品 {product_code} 的风险等级缺失，无法做适当性匹配。", False, warnings)
    ceiling = fin_db.RISK_CEILING.get(account["risk_level"], 0)
    if prod_level > ceiling:
        return ("block",
                f"适当性不匹配：用户风险等级 {account['risk_level']} 最高可投 "
                f"R{ceiling}，产品 {product['name']} 为 {product['risk_level']}。",
                False, warnings)

    amount = intent.get("amount", 0.0)
    # 规则二：起购金额
    if amount and amount < product["min_buy"]:
        return ("block", f"金额低于起购金额 {product['min_buy']} 元。", False, warnings)

    # 规则三：可用资金
    if amount and amount > account["cash"]:
        return ("block",
                f"可用资金不足：账户余额 {account['cash']:.2f} 元，本次需 {amount:.2f} 元。",
                False, warnings)

    # 规则四：单日申购限额
    limit = fin_db.DAILY_LIMIT.get(account["risk_level"], 0.0)
    used = fin_db.get_today_buy_amount(user_id)
    if amount and used + amount > limit:
        return ("block",
                f"超出单日申购限额：{account['risk_level']} 级限额 {limit:.0f} 元，"
                f"今日已申购 {used:.0f} 元，本次 {amount:.0f} 元。", False, warnings)

    # 规则五：集中度（不拦截，只提示——拦截会误伤正常的大额低风险配置）
    total = fin_db.get_total_asset(user_id)
    if total and amount / total > fin_db.CONCENTRATION_LIMIT:
        warnings.append(
            f"本次买入占账户总资产 {amount / total:.1%}，"
            f"超过 {fin_db.CONCENTRATION_LIMIT:.0%} 的集中度提示线。")

    return ("pass", f"适当性匹配通过（{account['risk_level']} → "
                    f"{product['risk_level']}），限额与资金校验通过。",
            True, warnings)

def retrieve_evidence(query: str, topk=3):
    """从合规库取证据原文。返回 [(来源, 条文片段), ...]。

    注意这里用的是**合规库预设**：BM25 权重 10.0、不套 MMR、只认现行有效。
    用投研库的参数来查法条会明显变差（两个库必须分开调参）。
    """
    try:
        docs = get_compliance_retriever().search(query,topk)
    except Exception as e:          # 检索挂了也必须走 fail-closed，不能让图崩
        log.warning(f"合规证据检索失败，按证据不足处理: {e}")
        return []
    out = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("doc_no") or "法规库"
        out.append((src, d.page_content[:220]))
    return out

def explain_decision(decision: str, reason: str, evidence: list, warnings: list) -> str:
    """把结构化判定说成人话。LLM 只干这一件事，失败就退回模板文本。"""
    ev = "\n".join(f"- {src}：{txt}" for src, txt in evidence[:3]) or "- （无）"
    warn = "\n".join(f"⚠ {w}" for w in warnings)
    try:
        resp = llm.invoke(
            "你是合规专员。请把下面的结构化判定改写成给用户看的两三句话，"
            "不要新增任何事实，不要给出投资建议。\n"
            f"判定：{decision}\n理由：{reason}\n依据：{ev}\n提示：{warn}")
        return (resp.content or "").strip() or reason
    except Exception as e:
        log.warning(f"合规结论润色失败，退回模板文本: {e}")
        return f"{reason}\n依据：\n{ev}\n{warn}"

def compliance_node(state:dict,config: RunnableConfig) -> dict:
    """合规核查节点，规则引擎判定 + RAG去证据 + 签发票据
    参数:
        state (dict): 主图状态。
        config (RunnableConfig): 运行配置，取 user_id 与 thread_id。
    返回:
        dict: 状态增量。pass 时不补消息（tool_call 会真的被执行）；
              block / insufficient_evidence 时必须补一条 ToolMessage，
              否则那条 tool_call 永远等不到结果，模型会卡住。
    """
    intent = extract_write_intent(state.get("messages", []))
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "")
    user_id = (config or {}).get("configurable", {}).get("user_id", "")

    if not intent:
        return {
            "compliance_decision": "insufficient_evidence",
            "compliance_evidence": "未能从模型输出中解析出写操作意图。",
            "compliance_warnings": [],
        }

    decision, reason, matched, warnings = evaluate_rules(user_id, intent)

    # RAG 取证据（fail-closed：拿不到证据一律按"证据不足"处理）
    query = f"投资者适当性管理 风险等级 匹配 {intent.get('product_code', '')}".strip()
    evidence = retrieve_evidence(query)
    if not evidence:
        decision = "insufficient_evidence"
        reason = "未在现行有效的法规中检索到可引用的适当性依据，按证据不足拒绝。"
    evidence_text = "\n".join(f"[{src}] {txt}" for src, txt in evidence)
    human_text = explain_decision(decision, reason, evidence, warnings)
    log.info(f"合规核查 {decision}：{reason}")
    if decision == "pass":
        # 签票：只写库，不回 State。定位票据全靠 thread_id，
        # 且签发前会自动吊销该会话未核销的旧票——参数变了旧票必须失效。
        ticket_id = issue_ticket(
            thread_id, user_id, intent["tool"],
            intent.get("product_code", ""), intent.get("amount", 0.0),
            evidence_text, tool_call_id=intent.get("tool_call_id", ""),
            client_order_id=intent.get("client_order_id", ""))
        log.info(f"合规票据 {ticket_id} 已签发，等待人工审批")
        # pass 时不补任何消息：让那条 tool_call 真的被执行
        return {
            "compliance_decision": "pass",
            "compliance_evidence": evidence_text,
            "compliance_warnings": warnings,
        }

    # 拦截：补一条对齐 tool_call_id 的回执，让模型自己改方案
    return {
        "compliance_decision": decision,
        "compliance_evidence": evidence_text,
        "compliance_warnings": warnings,
        "messages": [
            ToolMessage(
                content=(f"【合规核查未通过：{decision}】{human_text}\n"
                         "请调整方案（更换产品、降低金额或补充资料）后再试，"
                         "不要重复提交同样的参数。"),
                tool_call_id=intent["tool_call_id"],
            )
        ],
    }
def route_after_compliance(state: dict) -> Literal["trade_sensitive_tools", "trade"]:
    """合规之后的三分支路由：只有 pass 才能进审批流。

    block 与 insufficient_evidence 都直接打回交易助理，**根本不进审批流**——
    给一个不该做的操作弹审批卡片，等于把合规责任转嫁给用户。
    """
    if state.get("compliance_decision") == "pass":
        return "trade_sensitive_tools"
    return "trade"


def build_approval_card(state: dict, thread_id: str = "") -> dict:
    """给前端的审批卡片：人核对的是**依据**，不是盲批动作。

    票据按 thread_id 从库里现查（取该会话唯一那张 issued 票），而不是读 State 快照——
    State 里压根没有票据字段，快照会保留上一次操作的残留值，人就会批错金额。
    """
    ticket = get_pending_ticket(thread_id) or {}
    return {
        "action": ticket.get("action", ""),
        "product_code": ticket.get("product_code", ""),
        "amount": float(ticket.get("amount") or 0),
        "evidence": (state.get("compliance_evidence") or "").split("\n")[:3],
        "warnings": state.get("compliance_warnings") or [],
        "ticket_id": ticket.get("ticket_id", ""),
        "expires_at": ticket.get("expires_at", ""),
    }




