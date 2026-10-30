# -*- coding: utf-8 -*-
"""业务库：账户 / 产品 / 持仓 / 订单 / 审计日志。

为什么单独一个文件：行情和持仓是**结构化数据**，走 SQL 而不是向量检索。
用 RAG 查实时价格等于主动引入幻觉——编一个价格比报错严重得多。

后端固定 PostgreSQL
-------------------
连接、探活、异常类型全部收敛在 ``untils/db_config.py``，这里只管写业务 SQL。
SQL 的参数占位符固定用 ``%s``（psycopg2 的写法）。

连接串在 ``.env`` 里配一行即可::

    FINAGENT_DB_URL=postgresql+psycopg2://postgres:yang1258@127.0.0.1:5432/finagent

建表与演示数据由建表脚本负责（``python tools/init_fin_db.py``），
但 ``init_db()`` 也保留自动建表兜底，保证 clone 下来直接能跑。
"""
import json
import os
import secrets
from datetime import datetime

from untils import db_config
from untils.db_config import insert_ignore, insert_returning_id
from untils.log_utils import log

# 兼容旧引用：现在就是连接串里的库名
DB_PATH = db_config._PARAMS["database"]
DB_BACKEND = db_config.BACKEND
DB_TARGET = db_config.safe_dsn()


def cursor(conn):
    """取游标：统一用 ``RealDictCursor``，每行都是 dict，上层不用再判断类型。"""
    import psycopg2.extras
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


def get_conn():
    """取当前线程的连接（thread-local，断线自动重连）。"""
    return db_config.get_conn()


def _q(sql: str, params=()):
    """执行一条 SQL 并返回游标（读操作）。"""
    cur = cursor(get_conn())
    cur.execute(sql, params)
    return cur


def _exec(sql: str, params=()):
    """执行一条写 SQL 并立即提交。"""
    conn = get_conn()
    cur = cursor(conn)
    cur.execute(sql, params)
    conn.commit()
    return cur


# ------------------------------------------------------------------ 风险等级
# 适当性匹配：用户等级 C1~C5 决定他能买到的最高产品风险等级 R1~R5。
# 这里写成一张显式的表而不是一句 if —— 合规规则必须能被单独单测和审计。
RISK_CEILING = {
    "C1": 1,   # 保守型：只能买 R1
    "C2": 1,   # 稳健型：R1
    "C3": 2,   # 平衡型：R2 及以下
    "C4": 3,   # 成长型：R3 及以下
    "C5": 5,   # 激进型：全等级
}

# 单日申购限额（元），按用户风险等级递减
DAILY_LIMIT = {
    "C1": 10_000.0,
    "C2": 50_000.0,
    "C3": 100_000.0,
    "C4": 300_000.0,
    "C5": 1_000_000.0,
}

# 单一产品持仓集中度上限（占账户总资产比例）
CONCENTRATION_LIMIT = 0.4


def _ddl() -> list:
    """生成建表语句（PG 方言）。

    建表脚本与这里同源（``tools/init_fin_db.py --export-sql`` 导出的
    SQL 就是调这个函数拼出来的），改表结构两边自动一致。

    幂等键的 UNIQUE 内联在列上而不是单独建索引，省一条语句也更好读。
    """
    pk = db_config.pk_col()
    s64, s32, s16 = db_config.str_col(64), db_config.str_col(32), db_config.str_col(16)
    txt, num = db_config.text_col(), db_config.num_col()
    return [
        f"""CREATE TABLE IF NOT EXISTS accounts (
            user_id    {s32} PRIMARY KEY,
            name       {s32} NOT NULL,
            risk_level {s16} NOT NULL,
            cash       {num} NOT NULL DEFAULT 0,
            status     {s16} NOT NULL DEFAULT 'active'
        )""",
        f"""CREATE TABLE IF NOT EXISTS products (
            product_code {s32} PRIMARY KEY,
            name         {s64} NOT NULL,
            risk_level   {s16} NOT NULL,
            nav          {num} NOT NULL,
            category     {s32} NOT NULL,
            min_buy      {num} NOT NULL DEFAULT 100,
            status       {s16} NOT NULL DEFAULT 'on_sale'
        )""",
        f"""CREATE TABLE IF NOT EXISTS positions (
            id           {pk},
            user_id      {s32} NOT NULL,
            product_code {s32} NOT NULL,
            shares       {num} NOT NULL DEFAULT 0,
            cost         {num} NOT NULL DEFAULT 0
        )""",
        # client_order_id 的 UNIQUE 是幂等的最后一道防线：
        # 就算应用层判断漏了，数据库也不会让它下第二单
        f"""CREATE TABLE IF NOT EXISTS orders (
            order_id          {pk},
            user_id           {s32} NOT NULL,
            product_code      {s32} NOT NULL,
            action            {s16} NOT NULL,
            amount            {num} NOT NULL,
            status            {s16} NOT NULL,
            client_order_id   {s64} UNIQUE,
            decision_snapshot {txt},
            created_at        {s32} NOT NULL
        )""",
        f"""CREATE TABLE IF NOT EXISTS audit_log (
            id       {pk},
            ts       {s32} NOT NULL,
            user_id  {s32},
            action   {s32} NOT NULL,
            payload  {txt},
            result   {s32},
            evidence {txt}
        )""",
        # 合规票据：一次写操作一张票，status 由 issued 单向流转到 consumed / revoked。
        # 状态机写在数据库里而不是内存 dict —— 进程重启票据不该凭空消失，
        # 也不该凭空出现；核销必须是原子的，否则两次下单会用掉同一张票。
        f"""CREATE TABLE IF NOT EXISTS compliance_ticket (
            ticket_id       {s64} PRIMARY KEY,
            thread_id       {s64} NOT NULL,
            user_id         {s32} NOT NULL,
            action          {s32} NOT NULL,
            product_code    {txt},
            amount          {num} NOT NULL DEFAULT 0,
            decision        {s16} NOT NULL DEFAULT 'pass',
            evidence_hash   {s64},
            status          {s16} NOT NULL DEFAULT 'issued',
            issued_at       {s32} NOT NULL,
            expires_at      {s32} NOT NULL,
            consumed_at     {s32},
            tool_call_id    {s64},
            client_order_id {s64}
        )""",
        "CREATE INDEX IF NOT EXISTS idx_ticket_session"
        " ON compliance_ticket(thread_id, status)",
    ]


# 演示数据：让代码 clone 下来就能跑，不需要额外的初始化步骤
SEED_ACCOUNTS = [
    ("u_1001", "张稳健", "C3", 200_000.0),
    ("u_1002", "李成长", "C4", 500_000.0),
    ("u_1003", "王激进", "C5", 1_200_000.0),
    ("u_1004", "赵保守", "C1", 30_000.0),
]
SEED_PRODUCTS = [
    ("000198", "天弘余额宝货币", "R1", 1.0000, "货币型", 1.0),
    ("110020", "易方达沪深300ETF联接", "R3", 1.8420, "指数型", 100.0),
    ("510300", "沪深300ETF", "R3", 3.9210, "股票型", 100.0),
    ("161725", "招商中证白酒指数", "R4", 0.9130, "行业指数", 100.0),
    ("519674", "银河创新成长混合", "R4", 5.2130, "混合型", 100.0),
    ("000001", "华夏成长混合", "R5", 2.1170, "混合型", 100.0),
]
SEED_POSITIONS = [
    ("u_1002", "110020", 10_000.0, 1.7500),
    ("u_1002", "000198", 50_000.0, 1.0000),
    ("u_1003", "510300", 20_000.0, 3.8000),
]

_initialized = False


def init_db(force=False):
    """建表并灌入演示数据。

    参数:
        force (bool): True 时清空业务表重建（审计日志保留，审计不该被清）。

    建表只做一次（模块级 ``_initialized``）：每条查询前跑一遍
    ``CREATE TABLE IF NOT EXISTS`` 是白白多一次网络往返。
    """
    global _initialized
    if _initialized and not force:
        return
    conn = get_conn()
    cur = cursor(conn)
    for stmt in _ddl():
        cur.execute(stmt)
    conn.commit()

    if force:
        for t in ("orders", "positions", "products", "accounts",
                  "compliance_ticket"):
            cur.execute(f"DELETE FROM {t}")
        conn.commit()

    cur.execute("SELECT COUNT(*) AS c FROM products")
    row = cur.fetchone()
    count = row["c"] if isinstance(row, dict) else row[0]
    if count == 0:
        cur.executemany(insert_ignore(
            "accounts", "user_id,name,risk_level,cash,status", "%s,%s,%s,%s,%s"),
            [a + ("active",) for a in SEED_ACCOUNTS])
        cur.executemany(insert_ignore(
            "products",
            "product_code,name,risk_level,nav,category,min_buy,status",
            "%s,%s,%s,%s,%s,%s,%s"),
            [p + ("on_sale",) for p in SEED_PRODUCTS])
        cur.executemany(
            "INSERT INTO positions(user_id,product_code,shares,cost)"
            " VALUES (%s,%s,%s,%s)", SEED_POSITIONS)
        conn.commit()
        log.info(f"业务库初始化完成: {DB_TARGET}")
    _initialized = True


# ------------------------------------------------------------------ 查询
def get_account(user_id: str):
    """查账户（含风险等级与可用资金）。"""
    init_db()
    row = _q("SELECT * FROM accounts WHERE user_id = %s", (user_id,)).fetchone()
    return dict(row) if row else None


def get_product(product_code: str):
    """查单个产品要素，返回 None 表示产品不存在（→ insufficient_evidence）。"""
    init_db()
    row = _q("SELECT * FROM products WHERE product_code = %s",
             (product_code,)).fetchone()
    return dict(row) if row else None


def list_products():
    """列出在售产品。"""
    init_db()
    rows = _q("SELECT * FROM products WHERE status = 'on_sale'").fetchall()
    return [dict(r) for r in rows]


def get_positions(user_id: str):
    """查持仓，带产品名称与当前市值。

    市值在 Python 里算而不是 SQL 里：PG 的 ``ROUND(double precision, 2)``
    会报类型不存在（它只认 numeric），与其写一遍 CAST，不如交给 Python。
    """
    init_db()
    rows = _q(
        """
        SELECT p.product_code, pr.name, pr.nav, pr.risk_level, p.shares, p.cost
        FROM positions p JOIN products pr ON pr.product_code = p.product_code
        WHERE p.user_id = %s
        """, (user_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["market_value"] = round(float(d["shares"]) * float(d["nav"]), 2)
        out.append(d)
    return out


def get_today_buy_amount(user_id: str) -> float:
    """当日已申购金额，用于限额校验。"""
    init_db()
    today = datetime.now().strftime("%Y-%m-%d")
    row = _q("SELECT COALESCE(SUM(amount),0) AS s FROM orders"
             " WHERE user_id = %s AND action = 'buy' AND status != 'rejected'"
             " AND substr(created_at,1,10) = %s", (user_id, today)).fetchone()
    return float(row["s"] if isinstance(row, dict) else row[0])


def get_total_asset(user_id: str) -> float:
    """账户总资产 = 现金 + 持仓市值。"""
    acc = get_account(user_id) or {}
    mv = sum(p["market_value"] for p in get_positions(user_id))
    return float(acc.get("cash", 0.0)) + mv


# ------------------------------------------------------------------ 写入
def _as_text(v) -> str:
    """把任意值转成可入库的文本（dict / list 走 JSON）。"""
    if isinstance(v, str):
        return v
    if v is None:
        return ""
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def place_order_row(user_id, product_code, action, amount, client_order_id,
                    snapshot) -> int:
    """落订单。client_order_id 上有唯一约束，重复提交会抛 IntegrityError，
    这正是幂等想要的：宁可报错，也不能下两单。

    返回:
        int: 新订单 ID。PG 没有 ``lastrowid``，靠 ``RETURNING`` 取，
             封装在 ``insert_returning_id()`` 里。
    """
    init_db()
    # 先取连接再复用：插入和提交必须在**同一个连接**上，
    # 中途再调 get_conn() 会走一遍探活，有把事务回滚掉的风险
    conn = get_conn()
    cur = cursor(conn)
    sql = ("INSERT INTO orders(user_id,product_code,action,amount,status,"
           "client_order_id,decision_snapshot,created_at)"
           " VALUES (%s,%s,%s,%s,'pending',%s,%s,%s)")
    params = (user_id, product_code, action, float(amount), client_order_id,
              _as_text(snapshot), datetime.now().isoformat(timespec="seconds"))
    oid = insert_returning_id(cur, sql, params, id_col="order_id")
    conn.commit()
    return oid


def update_order_status(order_id: int, status: str):
    """更新订单状态。"""
    _exec("UPDATE orders SET status = %s WHERE order_id = %s", (status, order_id))


def deduct_cash(user_id: str, amount: float):
    """扣减可用资金（买入成交时）。

    单独封一个函数而不是让上层拼 SQL：金额和幂等这类写操作必须走
    统一的 ``_exec``（参数绑定 + 立即提交），漏一次就是"钱扣了但没落库"。
    """
    _exec("UPDATE accounts SET cash = cash - %s WHERE user_id = %s",
          (float(amount), user_id))


# 唯一约束冲突的异常类型，幂等逻辑用
INTEGRITY_ERRORS = db_config.integrity_errors()


def find_order_by_client_id(client_order_id: str):
    """按幂等键查单，用于重复提交时返回同一张单而不是新建。"""
    init_db()
    row = _q("SELECT * FROM orders WHERE client_order_id = %s",
             (client_order_id,)).fetchone()
    return dict(row) if row else None


def list_orders(user_id: str, limit=20):
    """查历史订单。"""
    init_db()
    rows = _q("SELECT * FROM orders WHERE user_id = %s"
              " ORDER BY order_id DESC", (user_id,)).fetchall()
    # LIMIT 不参数化：psycopg2 对 LIMIT %s 也能绑，但这里是内部整型，直接切更省一次往返
    return [dict(r) for r in rows[:int(limit)]]


def write_audit(user_id, action, payload, result, evidence=""):
    """审计日志：谁发起、做了什么、依据是什么、结果如何。

    金融场景里这不是加分项而是硬要求——监管要能回溯决策过程。
    """
    init_db()
    _exec("INSERT INTO audit_log(ts,user_id,action,payload,result,evidence)"
          " VALUES (%s,%s,%s,%s,%s,%s)",
          (datetime.now().isoformat(timespec="seconds"), user_id, action,
           _as_text(payload), str(result), _as_text(evidence)))


# ------------------------------------------------------------------ 合规票据
def _now_text() -> str:
    """当前时间的 ISO 文本。

    时间统一存文本而不是时间戳：与 orders.created_at 保持一致，日志里直接可读；
    比较大小靠 ISO8601 的字典序（格式固定，所以字典序等于时间序）。
    """
    return datetime.now().isoformat(timespec="seconds")


def issue_ticket_row(thread_id, user_id, action, product_code, amount,
                     evidence_hash="", tool_call_id="", client_order_id="",
                     ttl=300) -> str:
    """签发票据并落库，返回 ticket_id。

    只有合规 pass 才会走到这里——票据存在本身就意味着"这次操作过了规则引擎"，
    所以后续任何环节只要拿不到有效票据，一律拒绝（fail-closed）。
    """
    init_db()
    conn = get_conn()
    cur = cursor(conn)
    issued = _now_text()
    expires = datetime.fromtimestamp(
        datetime.now().timestamp() + ttl).isoformat(timespec="seconds")
    ticket_id = secrets.token_hex(12)
    cur.execute(
        "INSERT INTO compliance_ticket(ticket_id,thread_id,user_id,action,"
        "product_code,amount,decision,evidence_hash,status,issued_at,expires_at,"
        "tool_call_id,client_order_id)"
        " VALUES (%s,%s,%s,%s,%s,%s,'pass',%s,'issued',%s,%s,%s,%s)",
        (ticket_id, thread_id, user_id, action, str(product_code or ""),
         float(amount or 0), evidence_hash, issued, expires,
         tool_call_id, client_order_id))
    conn.commit()
    log.info(f"签发合规票据 {ticket_id} | {action} {product_code} {amount}")
    return ticket_id


def consume_ticket(thread_id, action, product_code=None, amount=None,
                   tolerance=1.0):
    """原子核销：一条 UPDATE ... RETURNING 同时完成"校验"与"作废"。

    为什么必须合成一条 SQL：先 SELECT 看有没有票、再 UPDATE 改状态是两步，
    并发下两个请求会同时看到 issued，于是用同一张票下两单。
    把 ``status='issued'`` 写进 WHERE，由数据库保证只有一个请求能 RETURNING 到行。

    参数:
        product_code / amount: 传 None 表示不参与比对（撤单按订单号比，调仓没有金额）。
    返回:
        dict | None: 命中返回该票据（含 ticket_id），未命中返回 None。
    """
    init_db()
    conn = get_conn()
    cur = cursor(conn)
    now = _now_text()
    sql = ("UPDATE compliance_ticket SET status='consumed', consumed_at=%s"
           " WHERE thread_id=%s AND status='issued' AND expires_at > %s"
           " AND action=%s")
    params = [now, thread_id, now, action]
    if product_code is not None:
        sql += " AND product_code = %s"
        params.append(str(product_code))
    if amount is not None:
        # amount=0 的票据（撤单 / 调仓没有金额）不参与金额比对
        sql += " AND (amount = 0 OR ABS(amount - %s) <= %s)"
        params += [float(amount), float(tolerance)]
    sql += (" RETURNING ticket_id,user_id,action,product_code,amount,"
            "evidence_hash,issued_at")
    cur.execute(sql, params)
    row = cur.fetchone()
    conn.commit()
    return dict(row) if row else None


def revoke_tickets(thread_id) -> int:
    """吊销该会话所有未核销票据：用户拒绝审批 / 开启新一轮写操作。"""
    init_db()
    cur = _exec("UPDATE compliance_ticket SET status='revoked'"
                " WHERE thread_id=%s AND status='issued'", (thread_id,))
    return int(cur.rowcount or 0)


def get_ticket(ticket_id: str):
    """按票据号查，审批卡片与审计追溯用。"""
    init_db()
    row = _q("SELECT * FROM compliance_ticket WHERE ticket_id = %s",
             (ticket_id,)).fetchone()
    return dict(row) if row else None


def get_pending_ticket(thread_id: str):
    """取该会话最新一张未核销票据（审批卡片兜底与调试用）。"""
    init_db()
    row = _q("SELECT * FROM compliance_ticket"
             " WHERE thread_id = %s AND status = 'issued'"
             " ORDER BY issued_at DESC", (thread_id,)).fetchone()
    return dict(row) if row else None


if __name__ == "__main__":
    init_db(force=False)
    print(f"后端 = {DB_BACKEND} | 目标 = {DB_TARGET}")
    print("账户:", get_account("u_1002"))
    print("持仓:", get_positions("u_1002"))
    print("总资产:", get_total_asset("u_1002"))
