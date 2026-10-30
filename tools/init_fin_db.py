# -*- coding: utf-8 -*-
"""FinAgent 业务库（PostgreSQL）初始化：建库 / 建表 / 灌演示数据 / 导出建表 SQL。

四种用法
--------
1. 用 ``.env`` 里的 ``FINAGENT_DB_URL`` 建表并灌数据（最常用）::

    python tools/init_fin_db.py

2. 临时换连接串（不改 .env）::

    python tools/init_fin_db.py --url postgresql+psycopg2://postgres:yang1258@127.0.0.1:5432/finagent

3. 库还不存在时先建库（用同一份账号连到默认的 postgres 库执行 CREATE DATABASE）::

    python tools/init_fin_db.py --create-db

4. 导出建表 SQL 给运维 / 到别的机器上手动执行::

    python tools/init_fin_db.py --export-sql

``--force`` 会清空业务表重建（审计日志保留——审计记录不该被清空）。
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _lit(v) -> str:
    """把 Python 值转成 SQL 字面量（字符串单引号转义成两个单引号）。"""
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    if v is None:
        return "NULL"
    return str(v)


def build_sql() -> str:
    """生成完整建表 + 演示数据脚本（PostgreSQL）。

    这份 SQL 与 ``fin_db._ddl()`` 同源，改表结构时两边会自动保持一致。
    """
    from tools import fin_db
    from untils import db_config

    header = (
        "-- FinAgent 业务库建表脚本（PostgreSQL）\n"
        "-- 建库：psql -h 127.0.0.1 -U postgres -c \"CREATE DATABASE finagent;\"\n"
        "-- 执行：psql -h 127.0.0.1 -U postgres -d finagent -f finagent_postgres.sql\n"
    )

    lines = [
        header,
        "-- 本脚本可重复执行：建表用 IF NOT EXISTS，基础数据用冲突忽略语法",
        "",
        "-- ---------- 建表 ----------",
    ]
    for stmt in fin_db._ddl():
        lines.append(stmt + ";")
        lines.append("")

    lines.append("-- ---------- 账户 ----------")
    for row in fin_db.SEED_ACCOUNTS:
        vals = ", ".join(_lit(x) for x in row + ("active",))
        lines.append(db_config.insert_ignore(
            "accounts", "user_id,name,risk_level,cash,status", vals) + ";")

    lines.extend(["", "-- ---------- 产品 ----------"])
    for row in fin_db.SEED_PRODUCTS:
        vals = ", ".join(_lit(x) for x in row + ("on_sale",))
        lines.append(db_config.insert_ignore(
            "products",
            "product_code,name,risk_level,nav,category,min_buy,status",
            vals) + ";")

    # 持仓没有唯一键，重跑脚本会重复插入，所以先清空
    lines.extend([
        "",
        "-- ---------- 持仓（无唯一键，重跑前先清空） ----------",
        "DELETE FROM positions;",
    ])
    for row in fin_db.SEED_POSITIONS:
        vals = ", ".join(_lit(x) for x in row)
        lines.append(
            f"INSERT INTO positions(user_id,product_code,shares,cost)"
            f" VALUES ({vals});")

    lines.extend(["", "-- 完成。审计日志 audit_log 与订单 orders 表运行时自动写入。", ""])
    return "\n".join(lines)


def export_sql(sql_dir: str) -> list:
    """导出建表脚本到 datas/sql/finagent_postgres.sql。"""
    os.makedirs(sql_dir, exist_ok=True)
    path = os.path.join(sql_dir, "finagent_postgres.sql")
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_sql())
    return [path]


def create_database(url: str):
    """库不存在时先建库。

    PostgreSQL 不能在事务里执行 CREATE DATABASE，所以要先开 autocommit；
    而且必须连到一个**已存在的**库（这里连默认的 postgres 库）。
    """
    from sqlalchemy.engine.url import make_url

    u = make_url(url)
    db = (u.database or "").lstrip("/")
    import psycopg2
    conn = psycopg2.connect(host=u.host, port=u.port or 5432,
                            user=u.username, password=u.password,
                            dbname="postgres")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db,))
    if cur.fetchone():
        conn.close()
        return True, f"数据库 {db} 已存在"
    cur.execute(f'CREATE DATABASE "{db}"')
    conn.close()
    return True, f"已创建数据库 {db}"


def main():
    ap = argparse.ArgumentParser(description="FinAgent 业务库初始化")
    ap.add_argument("--url", default="", help="覆盖 FINAGENT_DB_URL")
    ap.add_argument("--create-db", action="store_true", help="库不存在时先建库")
    ap.add_argument("--force", action="store_true", help="清空业务表重建")
    ap.add_argument("--export-sql", action="store_true", help="导出建表 SQL")
    ap.add_argument("--sql-dir", default=os.path.join(_ROOT, "datas", "sql"),
                    help="SQL 导出目录")
    args = ap.parse_args()

    # env 必须在导入 db_config 之前设置：它在 import 时就解析了连接串
    if args.url:
        os.environ["FINAGENT_DB_URL"] = args.url

    from tools import fin_db
    from untils import db_config
    from untils.log_utils import log

    print(f"后端 = {db_config.BACKEND} | 目标 = {db_config.safe_dsn()}")

    ok, msg = db_config.probe()
    if not ok and args.create_db:
        print(f"库不可用（{msg}），尝试建库…")
        ok2, msg2 = create_database(db_config.DB_URL)
        print(f"  → {msg2}")
        ok, msg = db_config.probe()
    if not ok:
        print(f"连接失败：{msg}")
        print("提示：确认数据库已启动、账号密码正确，或用 --create-db 先建库")
        return 1

    if args.export_sql:
        for p in export_sql(args.sql_dir):
            print(f"已导出: {p}")

    fin_db.init_db(force=args.force)
    print(f"账户 {len(fin_db.list_products())} 个产品在售 | "
          f"示例账户 u_1002 总资产 {fin_db.get_total_asset('u_1002'):.2f}")
    log.info("业务库初始化完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
