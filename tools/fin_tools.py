# -*- coding: utf-8 -*-
"""金融工具集：safe（只读）/ sensitive（写操作）两级。

三条硬规则（这是本项目区别于通用 Agent 的地方）：
1. **user_id 从 config 注入，不让模型生成**。模型可能猜错或串号，
   一旦错了就是操作他人账户。工具签名里根本不给这个参数。
2. **写操作必须带 client_order_id（幂等键）**，数据库上有唯一索引兜底。
   模型重复提议一次就重复下一单，在真实金融系统里就是资金事故。
3. **写操作第一行原子核销合规票据**（一条 UPDATE ... RETURNING，校验与作废
   同一条 SQL），核销不到票直接拒绝。合规约束从 prompt 层、路由层下沉到
   函数级——这是唯一无法被绕过的层级：票据用过即废、过期即废，
   重复提交与重放攻击都卡在这里。
"""
import json
import time
import zlib
from datetime import datetime

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from graph_fin.ticket import consume_ticket
from tools import fin_db
from untils.log_utils import log

# 敏感工具名：路由用它判断"这条 tool_call 会不会产生副作用"
SENSITIVE_TOOL_NAMES = ("place_order", "cancel_order", "rebalance")
# 安全工具名：只读查询，不打断用户
SAFE_TOOL_NAMES = ("search_quotes", "get_my_positions", "get_product_info",
                   "get_my_orders")


def _user_id(config: RunnableConfig) -> str:
    """从鉴权上下文取 user_id。取不到就拒绝，绝不退回默认值。

    为什么不放在 State 里：State 是模型可见的，模型可能在多轮里把它改掉；
    configurable 是框架注入，模型碰不到。
    """
    uid = (config or {}).get("configurable", {}).get("user_id") if config else None
    if not uid:
        raise PermissionError("未注入 user_id，拒绝执行（参数必须从鉴权上下文取）")
    return uid


def _thread_id(config: RunnableConfig) -> str:
    """取会话线程号，用于查该会话签发的合规 ticket。"""
    return (config or {}).get("configurable", {}).get("thread_id", "") if config else ""


def _to_float(x) -> float:
    """把模型可能写出的 '5万' / '5万元' / '50000' 统一成数字。"""
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", "").replace("元", "").replace("人民币", "")
    try:
        if s.endswith("万"):
            return float(s[:-1]) * 10_000
        if s.endswith("亿"):
            return float(s[:-1]) * 100_000_000
        return float(s)
    except ValueError:
        raise ValueError(f"无法解析金额: {x}")


# ------------------------------------------------------------------ safe 工具
@tool
def search_quotes(symbols: str) -> str:
    """查询一个或多个标的的最新行情（价格、涨跌幅、昨收）。

    参数 symbols 为标的代码，多个用逗号分隔，例如 '510300,000198'。
    这是只读查询，不会改变任何账户状态。

    **报价纪律**：当前行情源是本地确定性模拟值（``source: mock``），
    不是真实市场数据，报数时需说明这是演示数据；
    禁止估算、禁止沿用历史对话里的旧价格。
    """
    codes = [s.strip() for s in symbols.split(",") if s.strip()]
    if not codes:
        return "未提供标的代码。"
    rows = []
    for code in codes[:10]:     # 一次最多 10 个，防模型塞一整个列表进来
        # 确定性模拟行情：同一代码每次返回同一价，便于回归测试。
        # 接真实行情时把这里换成 akshare / 券商行情 API 即可，工具签名不变。
        seed = zlib.crc32(code.encode())
        nav = 1.0 + (seed % 8000) / 1000.0
        change = ((seed >> 8) % 400 - 200) / 100.0     # -2.00% ~ +2.00%
        prev = nav / (1 + change / 100)
        rows.append({
            "symbol": code,
            "price": round(nav, 4),
            "change_pct": round(change, 2),
            "prev_close": round(prev, 4),
            "updated_at": datetime.now().strftime("%H:%M:%S"),
        })
    return json.dumps({"quotes": rows, "source": "mock"}, ensure_ascii=False)


@tool
def get_product_info(product_code: str = "") -> str:
    """查询在售产品的要素：名称、风险等级 R1~R5、最新净值、起购金额。

    不传 product_code 时返回全部在售产品列表。只读查询。
    """
    if product_code:
        p = fin_db.get_product(product_code)
        if not p:
            return f"未找到产品 {product_code}（可能已下架或代码有误）。"
        return json.dumps(p, ensure_ascii=False)
    return json.dumps(fin_db.list_products(), ensure_ascii=False)


@tool
def get_my_positions(config: RunnableConfig) -> str:
    """查询当前登录用户的持仓与账户可用资金、风险等级。只读查询。"""
    uid = _user_id(config)
    acc = fin_db.get_account(uid)
    if not acc:
        return f"未找到账户 {uid}。"
    positions = fin_db.get_positions(uid)
    return json.dumps({
        "user_id": uid,
        "name": acc["name"],
        "risk_level": acc["risk_level"],
        "cash": round(acc["cash"], 2),
        "total_asset": round(fin_db.get_total_asset(uid), 2),
        "positions": positions,
    }, ensure_ascii=False)


@tool
def get_my_orders(limit: int = 10, config: RunnableConfig = None) -> str:
    """查询当前登录用户的历史订单（含状态与幂等键）。只读查询。"""
    uid = _user_id(config)
    return json.dumps(fin_db.list_orders(uid, limit), ensure_ascii=False)


# ------------------------------------------------------------------ sensitive 工具
@tool
def place_order(product_code: str,
                action: str,
                amount: str,
                client_order_id: str,
                config: RunnableConfig = None) -> str:
    """下单（买入/卖出）。这是会产生真实副作用的写操作，执行前必须先通过合规核查。

    参数说明：
        product_code: 产品代码，如 110020。
        action: buy 买入 / sell 卖出。
        amount: 金额，支持 '50000' 或 '5万'。
        client_order_id: 客户端生成的幂等键，同一笔交易重试时必须传同一个值，
                         否则会被当作新订单。没有它就别调用本工具。
    """
    uid, tid = _user_id(config), _thread_id(config)

    # ① 函数级 fail-closed：核销不到票据，谁都别想下单
    # 一条 SQL 同时完成"校验 + 作废"：票据按"工具名 + 标的 + 金额"三要素对齐，
    # 防止"用买 A 的票据去买 B"、"批了 5 万实际下 50 万"，也防重复提交
    ok, reason, ticket = consume_ticket(tid, action="place_order",
                                        product_code=product_code,
                                        amount=_to_float(amount))
    if not ok:
        fin_db.write_audit(uid, "place_order", json.dumps(
            {"product_code": product_code, "amount": amount}, ensure_ascii=False),
            "rejected", reason)
        return f"【拒绝下单】{reason}"

    # ② 幂等：同一个 client_order_id 直接返回原单，不重复成交
    exist = fin_db.find_order_by_client_id(client_order_id)
    if exist:
        return (f"该幂等键已存在订单，未重复下单。订单号 {exist['order_id']}，"
                f"状态 {exist['status']}。")

    amt = _to_float(amount)
    product = fin_db.get_product(product_code)
    if not product:
        return f"产品 {product_code} 不存在，下单失败。"
    if amt < product["min_buy"]:
        return f"金额 {amt} 低于起购金额 {product['min_buy']}，下单失败。"

    # ③ 落单（真实撮合在这里会调券商接口，本项目用本地撮合模拟）
    snapshot = json.dumps({"decision": "pass", "ticket_id": ticket["ticket_id"],
                           "checked_at": int(time.time())}, ensure_ascii=False)
    try:
        order_id = fin_db.place_order_row(uid, product_code, action, amt,
                                          client_order_id, snapshot)
    except fin_db.INTEGRITY_ERRORS:
        # 数据库唯一约束兜底：并发下两个请求用同一个幂等键，必有一个撞到这里
        return "幂等键冲突，订单已存在，未重复下单。"

    if action == "buy" and amt <= (fin_db.get_account(uid) or {}).get("cash", 0):
        fin_db.deduct_cash(uid, amt)
    fin_db.update_order_status(order_id, "filled")
    fin_db.write_audit(uid, "place_order",
                       json.dumps({"order_id": order_id, "product_code": product_code,
                                   "amount": amt,
                                   "ticket_id": ticket["ticket_id"]},
                                  ensure_ascii=False),
                       "filled", reason)
    log.info(f"下单成功 order_id={order_id} user={uid} {action} {product_code} {amt}")
    return (f"下单成功。订单号 {order_id}，{action} {product_code} 金额 {amt:.2f} 元，"
            f"状态 filled。")


@tool
def cancel_order(order_id: int,
                 client_order_id: str,
                 config: RunnableConfig = None) -> str:
    """撤销一笔已提交的订单。写操作，执行前必须先通过合规核查。

    参数 client_order_id 为本撤单请求自己的幂等键，与下单时的幂等键不同。
    """
    uid, tid = _user_id(config), _thread_id(config)
    ok, reason, ticket = consume_ticket(tid, action="cancel_order",
                                        product_code=str(order_id))
    if not ok:
        fin_db.write_audit(uid, "cancel_order", str(order_id), "rejected", reason)
        return f"【拒绝撤单】{reason}"

    exist = fin_db.find_order_by_client_id(client_order_id)
    if exist:
        return f"该幂等键已处理过（订单 {exist['order_id']}），未重复撤单。"

    fin_db.update_order_status(order_id, "cancelled")
    fin_db.write_audit(uid, "cancel_order",
                       json.dumps({"order_id": order_id,
                                   "ticket_id": ticket["ticket_id"]},
                                  ensure_ascii=False),
                       "cancelled", reason)
    return f"订单 {order_id} 已撤销。"


@tool
def rebalance(targets: str,
              client_order_id: str,
              config: RunnableConfig = None) -> str:
    """按目标权重调仓。写操作，执行前必须先通过合规核查。

    参数 targets 形如 '{"510300": 0.6, "000198": 0.4}'，表示各标的的目标占比。
    """
    uid, tid = _user_id(config), _thread_id(config)
    ok, reason, ticket = consume_ticket(tid, action="rebalance", product_code=targets)
    if not ok:
        fin_db.write_audit(uid, "rebalance", targets, "rejected", reason)
        return f"【拒绝调仓】{reason}"

    exist = fin_db.find_order_by_client_id(client_order_id)
    if exist:
        return f"该幂等键已处理过（订单 {exist['order_id']}），未重复调仓。"

    try:
        plan = json.loads(targets)
    except json.JSONDecodeError:
        return "targets 不是合法 JSON，调仓失败。"
    total = sum(float(v) for v in plan.values())
    if abs(total - 1.0) > 0.01:
        return f"目标权重合计 {total}，必须约等于 1.0，调仓失败。"

    fin_db.write_audit(uid, "rebalance",
                       json.dumps({"targets": plan,
                                   "ticket_id": ticket["ticket_id"]},
                                  ensure_ascii=False),
                       "filled", reason)
    return f"调仓完成，目标配置 {plan}。"


# ------------------------------------------------------------------ 工具分组
MARKET_SAFE_TOOLS = [search_quotes, get_product_info, get_my_positions, get_my_orders]
# 交易助理：只读工具全给它。下单前它要自己核对产品要素、可用资金、已有订单，
# 让模型"先查再写"，而不是拿对话里的旧数字或猜的值去填下单参数。
TRADE_SAFE_TOOLS = MARKET_SAFE_TOOLS
TRADE_SENSITIVE_TOOLS = [place_order, cancel_order, rebalance]
