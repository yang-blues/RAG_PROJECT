# -*- coding: utf-8 -*-
"""统一数据库配置：PostgreSQL 单后端。

为什么要有这一层
----------------
业务库（账户 / 产品 / 持仓 / 订单 / 审计）如果直接散在 ``fin_db.py`` 里裸连，
连接参数、探活、异常类型就会到处重复。这一层把它们收敛成几个函数，
业务代码只管写 SQL。

后端只保留 PostgreSQL 一个：SQLite 撑不住并发写（订单、审计都在写），
MySQL 在本项目里没有额外收益，多一个后端就多一套方言分支要维护。
统一用 psycopg2 之后，占位符固定 ``%s``、自增主键固定 ``BIGSERIAL``、
自增 ID 固定走 ``RETURNING``，SQL 里不再有任何分支。

怎么用
------
在 ``.env`` 里配一行即可，格式沿用 SQLAlchemy 的连接串::

    FINAGENT_DB_URL=postgresql+psycopg2://postgres:yang1258@127.0.0.1:5432/finagent

先建库再跑初始化（库不存在时 ::

    python tools/init_fin_db.py --create-db
"""
import os
import re
import threading

from dotenv import load_dotenv
from sqlalchemy.engine.url import make_url

from untils.log_utils import log

# 先在加载 .env 之前把命令行 / 启动脚本指定的值抓在手里
_ENV_DB_URL = (os.environ.get("FINAGENT_DB_URL") or "").strip()

# 用 override=False：已存在的环境变量优先，
# 这样临时指定 FINAGENT_DB_URL 不会被 .env 覆盖回去。
load_dotenv(override=False)

# 兜底：项目里 untils/env_utils.py 用的是 load_dotenv(override=True)，
# 它一旦先被导入，就会用 .env 里的值盖掉我们显式指定的连接串。
# 这里把显式指定的值抢回来，保证两种导入顺序下行为一致。
if _ENV_DB_URL and (os.environ.get("FINAGENT_DB_URL") or "").strip() != _ENV_DB_URL:
    os.environ["FINAGENT_DB_URL"] = _ENV_DB_URL

# 项目根：untils/ 的上一级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DB_URL = "postgresql+psycopg2://postgres:yang1258@127.0.0.1:5432/finagent"

# 后端标识（/health 与日志里用）
POSTGRES = "postgresql"


def _parse(url: str):
    """解析连接串，返回 ``(后端, 连接参数 dict, 原始 URL)``。"""
    u = make_url(url)
    if not (u.drivername or "").lower().startswith("postgres"):
        raise ValueError(
            f"本项目只支持 PostgreSQL，当前连接串是 {u.drivername!r}；"
            f"请改 .env 里的 FINAGENT_DB_URL，例如 {DEFAULT_DB_URL}")
    return POSTGRES, {
        "host": u.host or "127.0.0.1",
        "port": u.port or 5432,
        "user": u.username or "",
        "password": u.password or "",
        "database": (u.database or "").lstrip("/"),
    }, url


# ------------------------------------------------------------------ 当前后端
DB_URL = (os.getenv("FINAGENT_DB_URL", "") or "").strip() or DEFAULT_DB_URL
BACKEND, _PARAMS, _RAW_URL = _parse(DB_URL)

# 占位符：psycopg2 固定用 %s，SQL 里直接写，不需要适配层
PH = "%s"


def str_col(n: int = 64) -> str:
    """键列 / 短字符串列类型。"""
    return f"VARCHAR({n})"


def text_col() -> str:
    """长文本列类型。"""
    return "TEXT"


def num_col() -> str:
    """金额 / 份额列类型。

    PG 也可以用 NUMERIC(18,4)（金额更严谨），这里用 DOUBLE PRECISION
    是为了让上层算市值、比大小时不用处理 Decimal 与 float 混算。
    """
    return "DOUBLE PRECISION"


def pk_col() -> str:
    """自增主键列定义。"""
    return "BIGSERIAL PRIMARY KEY"


def insert_ignore(table: str, cols: str, values_sql: str) -> str:
    """生成"冲突就跳过"的插入语句（PG 用 ``ON CONFLICT DO NOTHING``）。"""
    return (f"INSERT INTO {table}({cols}) VALUES ({values_sql})"
            f" ON CONFLICT DO NOTHING")


def insert_returning_id(cur, sql: str, params, id_col: str = "order_id") -> int:
    """插入并返回自增主键。

    PG 没有 ``lastrowid``，靠在尾部追加 ``RETURNING`` 把 ID 取回来。
    """
    cur.execute(sql + f" RETURNING {id_col}", params)
    row = cur.fetchone()
    if not row:
        return 0
    # RealDictCursor 返回的是 dict-like，没有 row[0]；
    # 普通游标返回 tuple。两种都得支持，否则换游标类型就 KeyError(0)
    return int(list(row.values())[0] if isinstance(row, dict) else row[0])


def integrity_errors() -> tuple:
    """唯一约束冲突的异常类型——幂等逻辑要 catch 它。

    PG 细分到 ``psycopg2.errors.UniqueViolation``；它是 ``IntegrityError``
    的子类，两个都列上，免得只捕获一种时漏掉别的完整性冲突。
    """
    import psycopg2
    return (psycopg2.errors.UniqueViolation, psycopg2.IntegrityError)


def safe_dsn(url: str = None) -> str:
    """脱敏连接串：把密码替换成 ``***``，方便打日志和 /health 返回。"""
    return re.sub(r"//([^:/@]+):[^@]*@", r"//\1:***@", url or DB_URL)


# ------------------------------------------------------------------ 连接
_local = threading.local()


def _new_conn():
    """新建一个 psycopg2 连接。"""
    import psycopg2
    p = _PARAMS
    conn = psycopg2.connect(
        host=p["host"], port=p["port"], user=p["user"],
        password=p["password"], dbname=p["database"],
    )
    conn.autocommit = False
    return conn


def _is_alive(conn) -> bool:
    """探活 + 清掉脏事务。

    数据库重启或网络闪断后，连接对象还在但已经不可用；
    PostgreSQL 更麻烦：上一条 SQL 报错后连接会卡在 INERROR 状态，
    后面所有查询都报 ``InFailedSqlTransaction``，必须先 rollback。
    """
    try:
        if getattr(conn, "closed", 0):
            return False
        # psycopg2 事务状态：0=IDLE 1=ACTIVE 2=INTRANS 3=INERROR 4=UNKNOWN
        # 只有 INERROR（上一条 SQL 报错）才必须回滚，否则后面全是
        # InFailedSqlTransaction。
        # 千万不能对 INTRANS 回滚——那是**正在进行**的正常事务，
        # 回滚会把刚 INSERT 还没 commit 的行直接抹掉。
        if conn.info.transaction_status == 3:
            conn.rollback()
        return True
    except Exception:
        return False


def get_conn():
    """按线程取连接（thread-local）。

    PG 连接本身可以跨线程，但按线程隔离可以避免多个线程
    互相 commit 掉对方尚未提交的事务。
    """
    conn = getattr(_local, "conn", None)
    if conn is None or not _is_alive(conn):
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        conn = _new_conn()
        _local.conn = conn
    return conn


def close_conn():
    """关闭当前线程的连接。服务优雅退出时调用。"""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _local.conn = None


def probe(url: str = None) -> tuple:
    """连通性探测，返回 ``(是否可用, 说明)``。"""
    _backend, params, _ = _parse(url or DB_URL)
    try:
        import psycopg2
        c = psycopg2.connect(host=params["host"], port=params["port"],
                             user=params["user"], password=params["password"],
                             dbname=params["database"])
        c.close()
        return True, f"{params['host']}:{params['port']}/{params['database']}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ------------------------------------------------------------------ Checkpoint
# LangGraph 的检查点表和业务表放同一个库（表名不冲突，都是 checkpoint_*）。
def checkpoint_url() -> str:
    """Checkpoint 的连接串。默认跟随业务库，可用环境变量单独覆盖。"""
    return (os.getenv("FINAGENT_CHECKPOINT_URL", "") or "").strip() or DB_URL


def checkpoint_dsn() -> str:
    """给 LangGraph saver 用的连接串（去掉 driver 后缀）。

    LangGraph 的 PostgresSaver 走 psycopg3，它不认 SQLAlchemy 的
    ``postgresql+psycopg2://`` 这种带 driver 的写法。
    """
    u = make_url(checkpoint_url())
    db = (u.database or "").lstrip("/")
    pwd = u.password or ""
    return f"postgresql://{u.username}:{pwd}@{u.host}:{u.port or 5432}/{db}"


def describe() -> dict:
    """给 /health 用的后端描述（密码已脱敏）。"""
    return {
        "backend": BACKEND,
        "db": safe_dsn(),
        "checkpoint": safe_dsn(checkpoint_url()),
    }


# ------------------------------------------------------------------ 启动自检
# 不做降级：默认后端就是唯一的后端，连不上说明环境没准备好，
# 与其悄悄换成别的东西后再报一堆看不懂的错，不如启动时就说清楚。
_ok, _msg = probe()
if not _ok:
    log.warning(f"PostgreSQL 不可用（{_msg}）；请确认库已创建："
                f"python tools/init_fin_db.py --create-db")


if __name__ == "__main__":
    ok, msg = probe()
    log.info(f"当前后端: {BACKEND} | 连接可用: {ok} | {msg}")
    print(f"DB_URL      = {safe_dsn()}")
    print(f"BACKEND     = {BACKEND}")
    print(f"占位符       = {PH}")
    print(f"自增主键     = {pk_col()}")
    print(f"Checkpoint  = {safe_dsn(checkpoint_url())}")
    print(f"连通性       = {ok} ({msg})")
