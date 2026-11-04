"""Stock MCP module - A 股行情工具。

提供两个工具：
  1. query_stock_quote - 查询个股实时行情
  2. query_index        - 查询大盘指数（上证/深证/创业板）

数据源：腾讯财经实时行情接口 (qt.gtimg.cn)，免 key 公开接口。
备用源：新浪财经行情接口 (hq.sinajs.cn)，需带 Referer 头。

所有网络请求均做超时与异常降级，数据源不可用时返回友好错误。
"""

import re
import urllib.request
import urllib.error
import urllib.parse
from typing import Dict, List, Optional, Tuple


from mcp.server.fastmcp import FastMCP  # type: ignore  # fastmcp v1 (mcp sdk)


# ---------------------------------------------------------------------------
# 常量与配置
# ---------------------------------------------------------------------------

# 请求超时（秒）
REQUEST_TIMEOUT = 8

# 腾讯财经行情接口（主数据源）
TENCENT_API = "https://qt.gtimg.cn/q={codes}"

# 腾讯股票搜索接口（名称→代码）
TENCENT_SEARCH_API = "https://smartbox.gtimg.cn/s3/?v=2&q={keyword}&t=all"

# 新浪财经行情接口（备用数据源）
SINA_API = "https://hq.sinajs.cn/list={codes}"
SINA_REFERER = "https://finance.sina.com.cn"

# 三大指数代码
INDEX_CODES = {
    "上证指数": "sh000001",
    "深证成指": "sz399001",
    "创业板指": "sz399006",
}

# 常用股票名称→代码映射（内置本地缓存，减少搜索请求）
STOCK_NAME_MAP: Dict[str, str] = {
    # 大盘蓝筹
    "贵州茅台": "sh600519",
    "工商银行": "sh601398",
    "建设银行": "sh601939",
    "农业银行": "sh601288",
    "中国银行": "sh601988",
    "中国平安": "sh601318",
    "招商银行": "sh600036",
    "中国石油": "sh601857",
    "中国石化": "sh600028",
    "中国人寿": "sh601628",
    "五粮液": "sz000858",
    "宁德时代": "sz300750",
    "比亚迪": "sz002594",
    "美的集团": "sz000333",
    "格力电器": "sz000651",
    "海康威视": "sz002415",
    "中国中免": "sh601888",
    "恒瑞医药": "sh600276",
    "万华化学": "sh600309",
    "三一重工": "sh600031",
    # 科技股
    "腾讯控股": "hk0700",
    "阿里巴巴": "hk9988",
    "中芯国际": "sh688981",
    "韦尔股份": "sh603501",
    "兆易创新": "sh603986",
    "立讯精密": "sz002475",
    "歌尔股份": "sz002241",
    # 新能源
    "隆基绿能": "sh601012",
    "通威股份": "sh600438",
    "阳光电源": "sz300274",
    "天齐锂业": "sz002466",
    "赣锋锂业": "sz002460",
    "亿纬锂能": "sz300014",
    # 医药
    "药明康德": "sh603259",
    "片仔癀": "sh600436",
    "云南白药": "sz000538",
    "长春高新": "sz000661",
    # 银行
    "浦发银行": "sh600000",
    "兴业银行": "sh601166",
    "民生银行": "sh600016",
    "交通银行": "sh601328",
    "中信银行": "sh601998",
    "光大银行": "sh601818",
    "平安银行": "sz000001",
    # 地产
    "万科A": "sz000002",
    "保利发展": "sh600048",
    "招商蛇口": "sz001979",
    # 证券
    "中信证券": "sh600030",
    "东方财富": "sz300059",
    "华泰证券": "sh601688",
}


# ---------------------------------------------------------------------------
# 网络请求工具
# ---------------------------------------------------------------------------

def _http_get(url: str, headers: Optional[Dict[str, str]] = None,
              timeout: int = REQUEST_TIMEOUT, encoding: str = "gbk") -> str:
    """发起 HTTP GET 请求，返回解码后的文本。

    Args:
        url: 请求 URL
        headers: 自定义请求头
        timeout: 超时时间（秒）
        encoding: 响应编码（腾讯接口返回 GBK）

    Returns:
        解码后的响应文本

    Raises:
        RuntimeError: 请求失败时抛出，附带友好错误信息
    """
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return raw.decode(encoding, errors="replace")
    except urllib.error.URLError as e:
        raise RuntimeError(f"网络请求失败：{e.reason}") from e
    except TimeoutError as e:
        raise RuntimeError(f"请求超时（{timeout}秒）") from e
    except Exception as e:
        raise RuntimeError(f"请求异常：{e}") from e


# ---------------------------------------------------------------------------
# 股票代码解析
# ---------------------------------------------------------------------------

def _normalize_stock_code(query: str) -> Tuple[str, str]:
    """将用户输入的股票代码或名称解析为标准格式 (交易所前缀+代码, 名称)。

    支持的输入格式：
      - 完整代码：sh600519、SH600519、sz000858、600519、000858
      - 股票名称：贵州茅台、五粮液

    Returns:
        (stock_code, stock_name)
        stock_code: 标准格式，如 sh600519、sz000858
        stock_name: 股票名称（如果能从本地映射找到），否则为原查询

    Raises:
        ValueError: 无法解析或找不到对应股票
    """
    query = query.strip()
    if not query:
        raise ValueError("股票代码或名称不能为空")

    # 情况 1：已经是完整格式（带 sh/sz/hk 前缀）
    m = re.match(r'^(sh|sz|hk|SH|SZ|HK)(\d{6})$', query)
    if m:
        prefix = m.group(1).lower()
        code = m.group(2)
        full_code = prefix + code
        # 尝试反查名称
        name = query
        for n, c in STOCK_NAME_MAP.items():
            if c == full_code:
                name = n
                break
        return full_code, name

    # 情况 2：纯数字代码（6位）
    if re.match(r'^\d{6}$', query):
        code = query
        # 根据代码前缀判断交易所。
        # 先匹配两段前缀：基金/LOF 与可转债的归属只看首位会判错——
        # 深市基金是 15x(ETF)/16x(LOF)/18x(封闭式)，沪市可转债是 11x，
        # 深市可转债是 12x。15x/16x 一旦落到默认 sh（如 sh161725），
        # 腾讯和新浪都会返回空，表现为"查不到行情"。
        if code.startswith(('15', '16', '18')):
            prefix = 'sz'            # 深市基金 / LOF / 封闭式基金
        elif code.startswith('11'):
            prefix = 'sh'            # 沪市可转债（110/111/113/118）
        elif code.startswith('12'):
            prefix = 'sz'            # 深市可转债（123/127/128）
        elif code.startswith(('6', '9')):
            prefix = 'sh'            # 沪市 A 股 / 沪 B（900xxx）
        elif code.startswith(('0', '3')):
            prefix = 'sz'            # 深市主板 / 创业板
        elif code.startswith('2'):
            prefix = 'sz'            # 深 B（200xxx）
        elif code.startswith('5'):
            prefix = 'sh'            # 沪市 ETF / 场内基金（50x/51x/56x/58x）
        elif code.startswith(('4', '8')):
            prefix = 'bj'            # 北交所
        else:
            prefix = 'sh'            # 其余按上证兜底
        full_code = prefix + code
        # 尝试反查名称
        name = query
        for n, c in STOCK_NAME_MAP.items():
            if c == full_code:
                name = n
                break
        return full_code, name

    # 情况 3：股票名称 - 先查本地映射
    if query in STOCK_NAME_MAP:
        return STOCK_NAME_MAP[query], query

    # 情况 4：股票名称 - 调用腾讯搜索接口
    try:
        url = TENCENT_SEARCH_API.format(keyword=urllib.parse.quote(query.encode('utf-8')))
        text = _http_get(url, encoding='utf-8')
        # 返回格式：v_hint="sh~600519~贵州茅台~gzmt~GP-A"
        m = re.search(r'v_hint="([^"]+)"', text)
        if m:
            hint = m.group(1)
            parts = hint.split('~')
            if len(parts) >= 3 and parts[0] and parts[1]:
                prefix = parts[0]
                code = parts[1]
                name = parts[2] if len(parts) > 2 else query
                return f"{prefix}{code}", name
    except Exception:
        # 搜索失败不致命，继续往下
        pass

    raise ValueError(f"无法解析股票代码或名称：{query}（请尝试使用完整代码，如 sh600519）")


# ---------------------------------------------------------------------------
# 行情数据解析（腾讯接口）
# ---------------------------------------------------------------------------

def _parse_tencent_quote(line: str) -> Dict:
    """解析腾讯财经行情接口的单条数据。

    腾讯接口字段（以 ~ 分隔）：
      1: 股票名称
      2: 股票代码
      3: 当前价格
      4: 昨收
      5: 今开
      6: 成交量（手）
      30: 时间 (YYYYMMDDHHMMSS)
      31: 涨跌额
      32: 涨跌幅（%）
      33: 最高
      34: 最低
      36: 成交量（手，与字段 6 同值）
      37: 成交额（万元）

    Args:
        line: 单行数据，如 v_sh600519="1~贵州茅台~600519~..."

    Returns:
        解析后的行情字典
    """
    # 提取引号内的内容
    m = re.search(r'="([^"]*)"', line)
    if not m:
        raise ValueError("无法解析行情数据格式")

    data_str = m.group(1)
    parts = data_str.split('~')

    # 至少需要 38 个字段才有完整信息
    if len(parts) < 35:
        raise ValueError("行情数据字段不足")

    def _f(idx: int, default: str = "") -> str:
        """安全获取字段"""
        return parts[idx] if idx < len(parts) else default

    def _float(val: str, default: float = 0.0) -> float:
        """安全转 float"""
        try:
            return float(val) if val else default
        except (ValueError, TypeError):
            return default

    name = _f(1)
    code = _f(2)
    current_price = _float(_f(3))
    prev_close = _float(_f(4))
    open_price = _float(_f(5))
    volume_shou = _float(_f(6))  # 成交量（手）
    time_str = _f(30)
    change_amount = _float(_f(31))
    change_percent = _float(_f(32))
    high_price = _float(_f(33))
    low_price = _float(_f(34))
    amount_wan = _float(_f(37))  # 成交额（万元）

    # 格式化时间
    formatted_time = time_str
    if len(time_str) == 14 and time_str.isdigit():
        formatted_time = f"{time_str[:4]}-{time_str[4:6]}-{time_str[6:8]} {time_str[8:10]}:{time_str[10:12]}:{time_str[12:14]}"

    # 成交量换算（1 手 = 100 股）
    volume_shares = volume_shou * 100
    amount_yuan = amount_wan * 10000

    return {
        "stock_name": name,
        "stock_code": code,
        "current_price": round(current_price, 2),
        "prev_close": round(prev_close, 2),
        "open_price": round(open_price, 2),
        "high_price": round(high_price, 2),
        "low_price": round(low_price, 2),
        "change_amount": round(change_amount, 2),
        "change_percent": round(change_percent, 2),
        "volume_shares": int(volume_shares),
        "volume_hands": int(volume_shou),
        "turnover": round(amount_yuan, 2),
        "turnover_wan": round(amount_wan, 2),
        "update_time": formatted_time,
        "data_source": "腾讯财经",
    }


# ---------------------------------------------------------------------------
# 主数据源：腾讯财经
# ---------------------------------------------------------------------------

def _fetch_from_tencent(codes: List[str]) -> List[Dict]:
    """从腾讯财经接口获取行情数据。

    Args:
        codes: 股票代码列表，如 ["sh600519", "sz000858"]

    Returns:
        行情字典列表
    """
    if not codes:
        return []

    url = TENCENT_API.format(codes=",".join(codes))
    text = _http_get(url, encoding="gbk")

    results = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or "none_match" in line:
            continue
        try:
            quote = _parse_tencent_quote(line)
            results.append(quote)
        except Exception:
            continue

    return results


# ---------------------------------------------------------------------------
# 备用数据源：新浪财经
# ---------------------------------------------------------------------------

def _fetch_from_sina(codes: List[str]) -> List[Dict]:
    """从新浪财经接口获取行情数据（备用）。

    Args:
        codes: 股票代码列表

    Returns:
        行情字典列表
    """
    if not codes:
        return []

    # 新浪的代码格式和腾讯基本一致（sh600519）
    url = SINA_API.format(codes=",".join(codes))
    headers = {"Referer": SINA_REFERER}
    text = _http_get(url, headers=headers, encoding="gbk")

    results = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        m = re.search(r'="([^"]*)"', line)
        if not m:
            continue
        data_str = m.group(1)
        parts = data_str.split(',')
        if len(parts) < 32:
            continue

        try:
            name = parts[0]
            code = re.search(r'var hq_str_(\w+)=', line)
            code_str = code.group(1) if code else ""
            code_num = code_str[2:] if len(code_str) > 2 else code_str

            open_price = float(parts[1]) if parts[1] else 0.0
            prev_close = float(parts[2]) if parts[2] else 0.0
            current_price = float(parts[3]) if parts[3] else 0.0
            high_price = float(parts[4]) if parts[4] else 0.0
            low_price = float(parts[5]) if parts[5] else 0.0
            volume_shares = float(parts[8]) if parts[8] else 0.0
            turnover = float(parts[9]) if parts[9] else 0.0
            date = parts[30] if len(parts) > 30 else ""
            time = parts[31] if len(parts) > 31 else ""

            change_amount = current_price - prev_close
            change_percent = (change_amount / prev_close * 100) if prev_close else 0.0

            results.append({
                "stock_name": name,
                "stock_code": code_num,
                "current_price": round(current_price, 2),
                "prev_close": round(prev_close, 2),
                "open_price": round(open_price, 2),
                "high_price": round(high_price, 2),
                "low_price": round(low_price, 2),
                "change_amount": round(change_amount, 2),
                "change_percent": round(change_percent, 2),
                "volume_shares": int(volume_shares),
                "volume_hands": int(volume_shares / 100),
                "turnover": round(turnover, 2),
                "turnover_wan": round(turnover / 10000, 2),
                "update_time": f"{date} {time}" if date and time else f"{date}{time}",
                "data_source": "新浪财经",
            })
        except (ValueError, IndexError):
            continue

    return results


# ---------------------------------------------------------------------------
# 统一行情获取（主备源自动降级）
# ---------------------------------------------------------------------------

def _get_quotes(codes: List[str]) -> List[Dict]:
    """获取股票行情数据，自动在主备数据源间降级。

    Args:
        codes: 股票代码列表（标准格式，如 sh600519）

    Returns:
        行情字典列表

    Raises:
        RuntimeError: 所有数据源均不可用时抛出
    """
    # 主数据源：腾讯财经
    try:
        results = _fetch_from_tencent(codes)
        if results:
            return results
    except Exception as e:
        primary_error = str(e)
    else:
        primary_error = "返回数据为空"

    # 备用数据源：新浪财经
    try:
        results = _fetch_from_sina(codes)
        if results:
            return results
    except Exception as e:
        backup_error = str(e)
    else:
        backup_error = "返回数据为空"

    raise RuntimeError(
        f"行情数据获取失败\n"
        f"  主源（腾讯财经）：{primary_error}\n"
        f"  备源（新浪财经）：{backup_error}"
    )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _query_stock_quote(query: str) -> dict:
    """查询个股实时行情（内部实现）。"""
    # 解析股票代码
    stock_code, stock_name = _normalize_stock_code(query)

    # 获取行情
    quotes = _get_quotes([stock_code])

    if not quotes:
        return {
            "success": False,
            "error": f"未查询到 {query} 的行情数据",
            "query": query,
        }

    quote = quotes[0]

    # 格式化输出
    result = {
        "success": True,
        "stock_name": quote["stock_name"],
        "stock_code": quote["stock_code"],
        "current_price": quote["current_price"],
        "prev_close": quote["prev_close"],
        "open_price": quote["open_price"],
        "high_price": quote["high_price"],
        "low_price": quote["low_price"],
        "change_amount": quote["change_amount"],
        "change_percent": quote["change_percent"],
        "change_direction": "上涨" if quote["change_amount"] > 0 else (
            "下跌" if quote["change_amount"] < 0 else "平盘"
        ),
        "volume_shares": quote["volume_shares"],
        "volume_hands": quote["volume_hands"],
        "turnover": quote["turnover"],
        "turnover_wan": quote["turnover_wan"],
        "update_time": quote["update_time"],
        "data_source": quote["data_source"],
    }

    return result


def _query_index() -> dict:
    """查询大盘指数实时行情（内部实现）。"""
    codes = list(INDEX_CODES.values())
    names = list(INDEX_CODES.keys())

    quotes = _get_quotes(codes)

    if not quotes:
        return {
            "success": False,
            "error": "大盘指数数据获取失败",
        }

    # 按 INDEX_CODES 顺序整理结果
    index_list = []
    for name, code in INDEX_CODES.items():
        matched = None
        for q in quotes:
            # 匹配代码（去掉前缀）
            if code[2:] == q["stock_code"] or code == f'sh{q["stock_code"]}' or code == f'sz{q["stock_code"]}':
                matched = q
                break
        if matched:
            index_list.append({
                "index_name": matched["stock_name"] or name,
                "index_code": matched["stock_code"],
                "current_point": matched["current_price"],
                "change_amount": matched["change_amount"],
                "change_percent": matched["change_percent"],
                "change_direction": "上涨" if matched["change_amount"] > 0 else (
                    "下跌" if matched["change_amount"] < 0 else "平盘"
                ),
                "open_point": matched["open_price"],
                "prev_close": matched["prev_close"],
                "high_point": matched["high_price"],
                "low_point": matched["low_price"],
                "update_time": matched["update_time"],
            })
        else:
            index_list.append({
                "index_name": name,
                "index_code": code[2:],
                "current_point": None,
                "change_amount": None,
                "change_percent": None,
                "change_direction": "未知",
                "note": "数据暂不可用",
            })

    # 判断整体市场状态
    up_count = sum(1 for i in index_list if i["change_direction"] == "上涨")
    down_count = sum(1 for i in index_list if i["change_direction"] == "下跌")
    if up_count == 3:
        market_trend = "全面上涨"
    elif down_count == 3:
        market_trend = "全面下跌"
    elif up_count > down_count:
        market_trend = "多数上涨"
    elif down_count > up_count:
        market_trend = "多数下跌"
    else:
        market_trend = "分化"

    return {
        "success": True,
        "market_trend": market_trend,
        "indices": index_list,
        "data_source": quotes[0]["data_source"] if quotes else "未知",
        "update_time": index_list[0]["update_time"] if index_list else "",
    }


# ---------------------------------------------------------------------------
# MCP 注册
# ---------------------------------------------------------------------------

def register():
    """注册 A 股行情 MCP 模块。

    Returns:
        (path_prefix, fastmcp_instance)
    """
    mcp = FastMCP("stock-mcp")

    @mcp.tool(description="查询个股实时行情，支持股票代码（如 sh600519、600519）或名称（如 贵州茅台）。返回现价、涨跌额、涨跌幅、今开、昨收、最高、最低、成交量等数据。")
    def query_stock_quote(stock: str) -> dict:
        """查询个股实时行情。

        Args:
            stock: 股票代码或名称，如 "sh600519"、"600519"、"贵州茅台"

        Returns:
            包含股票名称、代码、现价、涨跌额、涨跌幅、今开、昨收、最高、最低、成交量、成交额、更新时间等信息
        """
        return _query_stock_quote(stock)

    @mcp.tool(description="查询A股三大指数实时行情，包括上证指数、深证成指、创业板指的点位、涨跌幅和市场整体趋势。")
    def query_index() -> dict:
        """查询大盘指数实时行情。

        Returns:
            包含上证指数、深证成指、创业板指的当前点位、涨跌额、涨跌幅、最高/最低、开盘/昨收、更新时间等信息，以及市场整体趋势判断
        """
        return _query_index()

    return "/stock-mcp", mcp
