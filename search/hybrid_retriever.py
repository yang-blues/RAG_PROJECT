#
"""混合检索器：dense(HNSW/IP) + sparse(BM25 中文) 双路召回 → 加权 RRF →（可选）MMR 重排。

为什么单独写这个模块
--------------------
``tools/retriever_tools.py`` 用的是 langchain 封装的 Milvus retriever，
只能开"内置 RRF"(ranker_type=rrf)，**调不了两路权重，也叠不了 MMR**；
而且它把 pk 和排序位置吃掉了，融合与重排都做不了。所以这里走原生 pymilvus。

检索参数不在这里，见 ``search/retriever_config.py``。
融合与重排的算法本身也不在这里，见 ``search/rank_ops.py``。

检索流程
--------
    query → dense 路(top60) ┐
            bm25  路(top60) ┴→ 加权 RRF 融合 → [MMR 重排] → 阈值过滤
          → 按 pk 一次性回查正文 → Document 列表

# ---------------------------------------------------------------------------
# 以下为评测相关说明，与线上运行无关，保留备查（详见 datas/eval/检索评测报告.md）
# ---------------------------------------------------------------------------
# - 参数出处：投研库 222 题网格搜索得 w_bm25=2.0、MMR λ=0.9；
#   合规库 312 题法条定位得 w_bm25=10.0、不套 MMR。
#   两套金标（宽松 / 严格）各自独立搜索，结论一致，参数没有过拟合到某一套。
# - 投研库提升（严格金标）：Recall@5 46.4% → 58.1%、MRR@10 33.8% → 44.1%。
# - 合规库必须关 MMR 的原因：法条检索是"精确定位唯一一条"，
#   多样性重排会把唯一正确答案挤下去（R@5 94.9% → 76.3%）。
# - 结论：两个库必须分开调参，共用一套权重必然牺牲一侧。
"""
import threading

import numpy as np
from langchain_core.documents import Document
from pymilvus import MilvusClient

from llm_model.embedings_model import qwen_embedding
from search.rank_ops import mmr_select, weighted_rrf
from search.retriever_config import (BASE_FILTER, BM25_DROP_RATIO, CANDIDATE,
                                     COMPLIANCE_PRESET, DENSE_EF,
                                     META_FIELDS_FUND, RESEARCH_PRESET, RRF_K,
                                     RetrieverPreset)
from untils.env_utils import MILVUS_URI
from untils.log_utils import log


class HybridRetriever:
    """Milvus 混合检索器：双路召回 → 加权 RRF →（可选）MMR → 阈值拒答。

    设计要点：
        * 检索只回 **pk 列表**，最后一次性回查正文。融合、重排全在 pk 层面做，
          避免大文本在节点之间传来传去。
        * MMR 需要候选之间的相似度矩阵，所以全库 dense 向量要拉到内存；
          这里是懒加载，不跑 MMR 就不会加载。
    """

    def __init__(self,
                 collection: str,
                 w_dense: float = 1.0,
                 w_bm25: float = 2.0,
                 mmr_lambda: float | None = None,
                 candidate: int = CANDIDATE,
                 rrf_k: int = RRF_K,
                 topk: int = 5,
                 extra_filter: str | None = None,
                 score_threshold: float | None = None,
                 meta_fields: tuple = META_FIELDS_FUND):
        """
        参数:
            collection (str): Milvus 集合名。
            w_dense (float): 向量路在 RRF 中的权重。
            w_bm25 (float): BM25 路在 RRF 中的权重。
            mmr_lambda (float | None): MMR 相关度权重 λ，None 表示不做重排。
                                       λ 越大越偏向相关度，越小越偏向多样性。
            candidate (int): 每路召回的候选池大小，融合前的召回宽度。
            rrf_k (int): RRF 平滑常数。
            topk (int): 最终返回条数。
            extra_filter (str | None): 附加的 Milvus 过滤表达式。
            score_threshold (float | None): 融合分数阈值，低于它的结果直接丢掉，
                                            用于实现"检索不到依据就拒答"。
            meta_fields (tuple): 想一并取回的元数据字段。
        """
        self.collection = collection
        self.w_dense = w_dense
        self.w_bm25 = w_bm25
        self.mmr_lambda = mmr_lambda
        self.candidate = candidate
        self.rrf_k = rrf_k
        self.topk = topk
        self.extra_filter = extra_filter
        self.score_threshold = score_threshold
        self.meta_fields = meta_fields

        self.client = MilvusClient(uri=MILVUS_URI)
        self._vecs = None        # (pks, matrix)，MMR 用，懒加载
        self._idx = {}

    @classmethod
    def from_preset(cls, preset: RetrieverPreset, topk: int | None = None):
        """按预设构造检索器。调参改 ``retriever_config.py`` 即可。"""
        return cls(
            collection=preset.collection,
            w_dense=preset.w_dense,
            w_bm25=preset.w_bm25,
            mmr_lambda=preset.mmr_lambda,
            candidate=preset.candidate,
            rrf_k=preset.rrf_k,
            topk=topk or preset.topk,
            extra_filter=preset.extra_filter,
            score_threshold=preset.score_threshold,
            meta_fields=preset.meta_fields,
        )

    # ------------------------------------------------------------ 基础能力
    def output_fields(self):
        """检索时要回查的字段：正文 + 出处 + 业务元数据。"""
        return ["pk", "text", "source", "category", *self.meta_fields]

    def filter_expr(self):
        """基础过滤 + 调用方附加的过滤（合规侧的"现行有效"）。"""
        if self.extra_filter:
            return f"{BASE_FILTER} and ({self.extra_filter})"
        return BASE_FILTER

    def embed(self, query: str):
        """query 向量化。"""
        return qwen_embedding.embed_query(query)

    # ------------------------------------------------------------ 单路检索
    def _search_one(self, data, anns_field, metric, params, k):
        """执行单路检索，返回 [(pk, score), ...]。"""
        res = self.client.search(
            collection_name=self.collection,
            data=data,
            anns_field=anns_field,
            limit=k,
            search_params={"metric_type": metric, "params": params},
            filter=self.filter_expr(),
            output_fields=["pk"],
        )
        return [(int(h["entity"]["pk"]), float(h["distance"])) for h in res[0]]

    def dense_search(self, query, k=None):
        """向量路：HNSW + 内积。擅长口语化提问、同义改写。"""
        return self._search_one([self.embed(query)], "dense", "IP",
                                {"ef": DENSE_EF}, k or self.candidate)

    def bm25_search(self, query, k=None):
        """关键词路：BM25 + 中文分析器。擅长代码、条款编号、数字。"""
        return self._search_one([query], "sparse", "BM25",
                                {"drop_ratio_search": BM25_DROP_RATIO},
                                k or self.candidate)

    # ------------------------------------------------------------ 融合与重排
    def rrf_fuse(self, rank_lists, weights=None):
        """加权 RRF：score(d) = Σ w_i / (k_rrf + rank_i(d))。

        算法实现在 ``rank_ops.weighted_rrf``，这里只补上本检索器的两路权重。
        """
        return weighted_rrf(rank_lists, weights or [self.w_dense, self.w_bm25],
                            self.rrf_k)

    def load_vectors(self):
        """把全库 dense 向量拉进内存，供 MMR 计算候选间相似度。"""
        if self._vecs is not None:
            return self._vecs
        pks, vecs = [], []
        it = self.client.query_iterator(
            collection_name=self.collection,
            batch_size=2000,
            filter=self.filter_expr(),
            output_fields=["pk", "dense"],
        )
        while True:
            batch = it.next()
            if not batch:
                break
            for x in batch:
                pks.append(int(x["pk"]))
                vecs.append(x["dense"])
        self._vecs = (pks, np.asarray(vecs, dtype="float32"))
        self._idx = {pk: i for i, pk in enumerate(pks)}
        log.info(f"已加载 {len(pks)} 条向量用于 MMR（{self.collection}）")
        return self._vecs

    def mmr_rerank(self, query, cand_pks, topk, lambda_=0.9):
        """MMR 多样性重排，返回**排好序的 pk 列表**。

        这里只负责 pk ↔ 向量矩阵的下标映射（哪个 pk 是第几行、库里有没有这条），
        贪心选择本身在 ``rank_ops.mmr_select`` 里。
        """
        _pks, mat = self.load_vectors()
        qv = np.asarray(self.embed(query), dtype="float32")
        cand = [p for p in cand_pks if p in self._idx]
        if not cand:
            return cand_pks[:topk]
        C = mat[[self._idx[p] for p in cand]]
        return [cand[i] for i in mmr_select(C, qv, topk, lambda_)]

    # ------------------------------------------------------------ 对外主入口
    def search_pks(self, query, topk=None):
        """走完整链路，返回 [(pk, fused_score), ...]（只要 pk 时用这个，最省）。"""
        topk = topk or self.topk
        fused = self.rrf_fuse([self.dense_search(query), self.bm25_search(query)])
        if self.mmr_lambda is not None:
            # MMR 只产出 pk 的先后顺序，融合分数还要回到 RRF 的结果里取
            pks = self.mmr_rerank(query, [p for p, _ in fused], topk, self.mmr_lambda)
            score_of = dict(fused)
            fused = [(p, score_of[p]) for p in pks if p in score_of]
        else:
            fused = fused[:topk]
        if self.score_threshold is not None:
            fused = [(p, s) for p, s in fused if s >= self.score_threshold]
        return fused

    def _fetch_text(self, pks):
        """按 pk 一次性回查正文与元数据。"""
        if not pks:
            return {}
        expr = f"pk in [{','.join(str(p) for p in pks)}]"
        rows = self.client.query(
            collection_name=self.collection,
            filter=expr,
            output_fields=self.output_fields(),
            limit=len(pks),
        )
        return {int(r["pk"]): r for r in rows}

    def search(self, query, topk=None) -> list[Document]:
        """检索并返回 Document 列表，metadata 里带上 pk 与融合分数，供引用溯源。"""
        fused = self.search_pks(query, topk)
        if not fused:
            return []
        rows = self._fetch_text([p for p, _ in fused])
        docs = []
        for pk, score in fused:
            r = rows.get(pk)
            if not r:
                continue
            meta = {k: v for k, v in r.items() if k not in ("pk", "text", "dense", "sparse")}
            meta["pk"] = pk
            meta["score"] = round(score, 6)
            meta["collection"] = self.collection
            docs.append(Document(page_content=r.get("text", ""), metadata=meta))
        return docs

    def invoke(self, query, topk=None) -> list[Document]:
        """与 langchain Retriever 同签名，方便直接替换掉旧的 retriever。"""
        return self.search(query, topk)


# ------------------------------------------------------------------ 单例
_research_retriever = None
_compliance_retriever = None
_singleton_lock = threading.Lock()


def get_research_retriever(topk=5) -> HybridRetriever:
    """投研检索器单例：MilvusClient 与全库向量都不便宜，进程内只建一次。"""
    global _research_retriever
    with _singleton_lock:
        if _research_retriever is None:
            _research_retriever = HybridRetriever.from_preset(RESEARCH_PRESET, topk)
        return _research_retriever


def get_compliance_retriever(topk=3) -> HybridRetriever:
    """合规检索器单例。"""
    global _compliance_retriever
    with _singleton_lock:
        if _compliance_retriever is None:
            _compliance_retriever = HybridRetriever.from_preset(COMPLIANCE_PRESET, topk)
        return _compliance_retriever
