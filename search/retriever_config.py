#
"""混合检索的参数配置。

把集合名、召回宽度、RRF 常数和两套预设集中在这里，
``hybrid_retriever.py`` 只负责检索流程本身 —— 调参不必碰检索代码。
"""
from dataclasses import dataclass
from typing import Optional

# ------------------------------------------------------------------ 集合
# 说明：env_utils 里的 COLLECTION_RESEARCH / COLLECTION_COMPLIANCE 指向旧库
# （元信息拼在正文前缀里）。这里用去前缀后重建的 v2 库，不改原有常量。
COLLECTION_RESEARCH_V2 = "fund_research_v2"
COLLECTION_COMPLIANCE_V2 = "finance_compliance_v2"

# ------------------------------------------------------------------ 检索常量
CANDIDATE = 60              # 融合前每路召回的候选池大小
RRF_K = 60                  # RRF 平滑常数，抑制 top1 的绝对优势
DENSE_EF = 128              # HNSW 查询时的搜索深度，越大越准越慢
BM25_DROP_RATIO = 0.2       # BM25 丢弃低权词的比例，砍掉长尾噪声

# 正文之外想一并取回的元数据字段（按库类型分组）
META_FIELDS_FUND = ("fund_code", "fund_short", "fund_type", "doc_type", "disclose_date")
META_FIELDS_LAW = ("law_level", "publish_date", "effective_date", "law_status", "doc_no")

# 每次检索都必带：孤立的标题节点没有正文，混进来只会稀释 top-k
BASE_FILTER = "category == 'content'"

# 合规库专用：用已废止的条款去放行一笔交易，比拒答严重得多，
# 所以在元数据层就把候选缩到"现行有效"，而不是检索完再过滤。
LAW_EFFECTIVE_FILTER = "law_status == '现行有效'"


@dataclass(frozen=True)
class RetrieverPreset:
    """一套检索参数。

    两个库的检索目标不同，参数必须分开配：
    投研是"找散落在多份文档里的相关段落"，合规是"精确定位唯一一条法条"。
    """
    collection: str
    w_dense: float = 1.0                    # 向量路在 RRF 中的权重
    w_bm25: float = 2.0                     # BM25 路在 RRF 中的权重
    mmr_lambda: Optional[float] = 0.9       # None 表示不做多样性重排
    topk: int = 5                           # 最终返回条数
    candidate: int = CANDIDATE
    rrf_k: int = RRF_K
    extra_filter: Optional[str] = None      # 附加的 Milvus 过滤表达式
    score_threshold: Optional[float] = None # 低于它的结果丢掉，实现"无依据就拒答"
    meta_fields: tuple = META_FIELDS_FUND


# 投研库：口语化提问为主，向量与关键词并重，靠 MMR 让 top-k 覆盖更多文档
RESEARCH_PRESET = RetrieverPreset(
    collection=COLLECTION_RESEARCH_V2,
    w_dense=1.0,
    w_bm25=2.0,
    mmr_lambda=0.9,
    topk=5,
    meta_fields=META_FIELDS_FUND,
)

# 合规库：查询的强信号是"法名 + 条号"，BM25 主导；答案是唯一一条，必须关 MMR
COMPLIANCE_PRESET = RetrieverPreset(
    collection=COLLECTION_COMPLIANCE_V2,
    w_dense=1.0,
    w_bm25=10.0,
    mmr_lambda=None,
    topk=3,
    extra_filter=LAW_EFFECTIVE_FILTER,
    meta_fields=META_FIELDS_LAW,
)
