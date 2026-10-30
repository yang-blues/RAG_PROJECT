# -*- coding: utf-8 -*-
"""合规票据（compliance ticket）：写操作的闸门，状态存在数据库里。

为什么落库而不是放内存
----------------------
票据是"这次写操作通过了合规"的唯一凭证。放内存 dict 有三个绕不过去的问题：
进程重启票据凭空消失（正常操作被拒）、多实例部署时每台机器各存一份
（A 机器签的票 B 机器看不到）、并发下两次核销可能同时成功。
落到 PostgreSQL 之后，这三件事都交给数据库的状态机保证。

三道闸门分别在三个层级，缺一不可
--------------------------------
1. **路由层**：模型一旦输出 sensitive 工具，条件边就**强制**插进合规节点
   —— 做成节点而不是工具，模型连"要不要合规"这个选择都没有。
2. **签票层**：合规 pass 才写一张 ``status='issued'`` 的票，
   归属写 ``thread_id``。State 里不存票据任何信息——
   thread_id 来自 ``configurable``，模型看不见也改不了。
3. **核销层**：写工具第一行调用，一条 ``UPDATE ... RETURNING``
   同时完成"校验"和"作废"。

第三道是唯一绕不过去的层级——即使路由写错、即使人误批了，没有票也下不了单。
"""
import hashlib
from datetime import datetime

from tools import fin_db
from untils.log_utils import log

# 票据有效期（秒）。太短影响正常操作，太长会放大误批的后果。
TICKET_TTL = 300
# 金额匹配容差：模型把 "5万" 解析成 50000 时可能有浮点误差
AMOUNT_TOLERANCE = 1.0


def _now_text() -> str:
    """当前时间的 ISO 文本，与库里时间字段同一格式，可直接比大小。"""
    return datetime.now().isoformat(timespec="seconds")


def hash_evidence(evidence: str) -> str:
    """证据指纹。审计时用它证明"当时确实依据这段条文做的判定"。"""
    return hashlib.sha256((evidence or "").encode("utf-8")).hexdigest()[:16]


def issue_ticket(thread_id, user_id, action, product_code, amount,
                 evidence="", tool_call_id="", client_order_id="") -> str:
    """合规通过后签发票据，只在 compliance_node 里调用。

    签发前先把该会话未核销的旧票吊销：参数变了旧票必须失效，
    否则会残留一张"上一次操作"的票被这一次误用。

    返回:
        str: ticket_id。只在日志与审计里用，不回写 State——
             后续定位票据一律走 thread_id。
    """
    revoke_ticket(thread_id)
    return fin_db.issue_ticket_row(
        thread_id=thread_id,
        user_id=user_id,
        action=action,
        product_code=product_code,
        amount=amount,
        evidence_hash=hash_evidence(evidence),
        tool_call_id=tool_call_id,
        client_order_id=client_order_id,
        ttl=TICKET_TTL,
    )


def consume_ticket(thread_id, action, product_code=None, amount=None):
    """写工具的第一行调用它：原子核销，返回 (是否放行, 理由, 票据)。

    一次调用同时完成四件事，任一不满足就拒绝：
        1. 该会话有没有 ``status='issued'`` 的票（用过 / 吊销过的票不会再命中）
        2. 有没有过期
        3. 动作对不对得上（防"用买入的票据去撤单"）
        4. 标的与金额对不对得上（防"批了 5 万，实际下 50 万"）

    返回:
        tuple: (bool, str, dict | None)。命中时第三个元素是票据本身，
               便于把 ticket_id 写进订单快照与审计日志。
    """
    ticket = fin_db.consume_ticket(thread_id, action, product_code, amount,
                                   tolerance=AMOUNT_TOLERANCE)
    if ticket:
        log.info(f"合规票据 {ticket['ticket_id']} 核销通过 | {action}"
                 f" {ticket['product_code']}")
        return True, f"合规票据 {ticket['ticket_id']} 校验通过。", ticket

    # 没命中：区分"压根没票 / 已用掉"与"过期 / 参数对不上"，回执要说清原因
    pending = fin_db.get_pending_ticket(thread_id)
    if pending:
        if pending["expires_at"] <= _now_text():
            reason = f"合规票据已过期（有效期 {TICKET_TTL}s），请重新发起合规核查。"
        else:
            reason = (f"票据与本次请求不匹配：票据为 {pending['action']}"
                      f" {pending['product_code']}，本次为 {action}"
                      f" {product_code or '-'}，请重新发起合规核查。")
        fin_db.revoke_tickets(thread_id)
    else:
        reason = "未找到有效合规票据，写操作必须先经过合规核查节点。"
    log.warning(f"合规票据核销失败 | {action} {product_code or '-'} | {reason}")
    return False, reason, None


def revoke_ticket(thread_id) -> int:
    """吊销票据：用户拒绝审批、或开始新一轮写操作时调用。返回影响行数。"""
    n = fin_db.revoke_tickets(thread_id)
    if n:
        log.info(f"吊销会话 {thread_id} 的 {n} 张未核销票据")
    return n


def get_ticket(ticket_id: str):
    """按票据号查全量信息，审批卡片与审计追溯用。"""
    return fin_db.get_ticket(ticket_id)


def get_pending_ticket(thread_id: str):
    """取会话当前未核销的票据，给审批卡片做兜底。"""
    return fin_db.get_pending_ticket(thread_id)
