#
"""排序算子：加权 RRF 融合 + MMR 多样性重排。

为什么单独拆一个文件
--------------------
这两个算法是**纯数学**：RRF 只吃"名次列表"，MMR 只吃"向量矩阵"。
它们不该知道 Milvus、不该知道 pk、也不知道集合名。拆出来有三个好处：

    1. 可单测 —— 塞两个假 rank list 就能验 RRF，塞个随机矩阵就能验 MMR，
       不用连 Milvus、不用发 embedding 请求；
    2. 好对齐简历 —— "加权 RRF 融合""MMR 多样性重排"是检索岗的标准术语，
       集中在一个百来行的文件里，面试最好讲；
    3. 消灭双份实现 —— ``search_tool/eval_retrieval.py`` 里有一份逐字相同的代码，
       将来把它换成 import 这里，就不会出现"评测跑一套、线上跑一套"的漂移。

它们在混合检索里的位置
----------------------
    dense 路 top60 ┐
                   ├→ 加权 RRF（本文件）→ [MMR 重排（本文件）] → 阈值过滤
    bm25  路 top60 ┘

关于 LangChain 现成的 MMR
-------------------------
有，但**上层那个用不了**：

    vectorstore.max_marginal_relevance_search(query, k, fetch_k, lambda_mult)

它是单路 dense 召回 —— 内部自己 embed 一次、自己搜 fetch_k 条，
BM25 路和加权 RRF 融合全没了；而且它只认 query 字符串或 query 向量，
喂不进我们已经融合好的候选列表。

能用的只有它的**底层算子**（``langchain_community.vectorstores.utils.maximal_marginal_relevance``），
那个函数只吃两个数组、只吐选中的下标，数学上与我们这里的 ``mmr_select`` 等价
（前提是向量已归一化，因为官方用余弦、我们库里索引的度量是内积 IP）。

但**这里仍然保留了自实现**，两个理由：

    1. 度量对齐 —— 直接用内积算，跟 Milvus 建索引时声明的 ``metric_type="IP"``
       是同一套口径，不必为了迁就官方算子再引入一次 L2 归一化；
    2. 零额外依赖 —— 只有 numpy，跟 ``search_tool/eval_retrieval.py`` 里那份
       逐字一致，评测脚本将来换成 import 这里时不会有任何行为漂移。
"""
import numpy as np

from search.retriever_config import RRF_K


def weighted_rrf(rank_lists, weights=None, rrf_k=RRF_K, topk=None):
    """加权 RRF 融合：score(d) = Σ w_i / (rrf_k + rank_i(d))。

    为什么用 RRF 而不是把两路分数归一化后相加：dense 是内积、BM25 是词频得分，
    两者尺度完全不同，直接相加等于让量纲大的那一路说了算。RRF **只用名次**，
    天然免疫尺度问题；分母的 rrf_k 用来压平 top1 的绝对优势
    （没有它，第 1 名的贡献是第 60 名的 60 倍）。

    参数:
        rank_lists (list): 每路召回结果，形如 ``[[(pk, score), ...], ...]``。
                           元组第二位的原始分数**不参与计算**，只取位置。
        weights (list[float] | None): 每路的权重，None 表示等权。
        rrf_k (int): 平滑常数，见 ``search/retriever_config.py``。
        topk (int | None): 截断条数，None 表示不截断。

    返回:
        list[(pk, score)]，按融合分数从高到低排序。
    """
    if weights is None:
        weights = [1.0] * len(rank_lists)
    scores = {}
    for w, lst in zip(weights, rank_lists):
        for rank, hit in enumerate(lst, 1):
            pk = hit[0]                      # 只认 pk 和名次，不认原始分数
            scores[pk] = scores.get(pk, 0.0) + w / (rrf_k + rank)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:topk] if topk else ranked


def mmr_select(cand_matrix, query_vector, topk, lambda_=0.9):
    """MMR 贪心选择，返回``cand_matrix``里要保留的**行下标**（顺序即最终顺序）。

        MMR = argmax[ λ·sim(d, q) − (1−λ)·max_{dᵢ ∈ 已选} sim(d, dᵢ) ]

    前半项要求"和问题相关"，后半项要求"和已选的不重复"，两者加权相减。
    λ=1 退化成按相关度排序，λ=0 退化成纯粹的多样性采样。

    解决的问题：同一份文档的相邻 chunk 语义几乎一样，会霸占 top-k 里的多个位置，
    把"另一份文档里也写了这个答案"的片段挤出去。投研场景的答案散落在
    招募说明书 / 基金合同 / 年报里，去重能让 top-5 覆盖更多文档
    （严格金标 Recall@5 46.4% → 58.1%）。

    参数:
        cand_matrix (np.ndarray): 候选向量矩阵，shape (n, dim)。
        query_vector (np.ndarray): query 向量，shape (dim,)。
        topk (int): 选几个。
        lambda_ (float): 相关度权重 λ，越大越偏相关、越小越偏多样。

    返回:
        list[int]，``cand_matrix`` 的行下标。调用方负责把下标映射回 pk。
    """
    C = np.asarray(cand_matrix, dtype="float32")
    qv = np.asarray(query_vector, dtype="float32")
    if C.ndim == 1:
        C = C.reshape(1, -1)
    rel = C @ qv                    # 各候选与 query 的相关度（内积，与 Milvus 索引同口径）
    sim = C @ C.T                   # 候选两两之间的相似度矩阵

    selected, rest = [], list(range(len(C)))
    while rest and len(selected) < topk:
        if not selected:
            j = int(np.argmax(rel[rest]))            # 第一位没有"已选"，惩罚项为 0
        else:
            div = sim[np.ix_(rest, selected)].max(axis=1)
            j = int(np.argmax(lambda_ * rel[rest] - (1 - lambda_) * div))
        selected.append(rest[j])
        rest.pop(j)
    return selected
