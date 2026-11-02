[README.md](https://github.com/user-attachments/files/34253791/README.md)
# FinAgent · 面向金融场景的多助理 Agent

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.3.30-1C3C3C)
![FastAPI](https://img.shields.io/badge/FastAPI-0.162-009688?logo=fastapi&logoColor=white)
![Milvus](https://img.shields.io/badge/Milvus-2.5-00A1EA)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)

用 LangGraph 搭的金融投顾 Agent：主助理负责理解意图，投研 / 行情 / 交易三个子助理各管一段，写操作在落库前必须通过**确定性合规核查 + 人工审批 + 一次性票据**三道闸门。

项目重点不在"能对话"，而在**金融场景下哪些事必须由代码保证、不能交给模型**：用户身份不能由模型生成、合规判定不能由模型拍板、下单不能只靠 prompt 自律、审批被绕过时必须有函数级兜底。

> 数据全部为本地模拟（行情、产品、账户、法条库），仅用于工程验证，不构成任何投资建议。

---

## 这个项目解决什么问题

| 通用 Agent 的做法 | 这里为什么不行 | 本项目的做法 |
|---|---|---|
| 一个 system prompt + 一堆工具 | 金融意图跨度大（查费率 vs 下单），一个 prompt 里规则互相打架 | 拆成主助理 + 三个子助理，各自只拿自己该有的工具 |
| 模型自己判断"要不要合规" | 模型可以跳过、可以被 prompt 注入绕过 | 条件边**强制**插入 `compliance_check` 节点，模型没有选择权 |
| 合规结论让模型说 | 判定不可复现、不可审计 | 规则引擎判定（纯代码），LLM 只负责把结论说成人话 |
| 下单前弹个确认 | 确认与执行之间存在重放 / 篡改空间 | 票据按「工具 + 标的 + 金额」绑定，一条 `UPDATE ... RETURNING` 校验即作废 |
| `user_id` 当工具参数传 | 模型可能猜错、改错，等于操作他人账户 | `user_id` 从 `configurable` 注入，工具签名里根本没有这个参数 |
| 靠"请勿重复提交" | 模型重试一次就是重复下单 | 强制 `client_order_id` 幂等键 + 数据库唯一索引 |

---

## 架构

### 主图：星形路由

控制流只有一种形状：**主 → 子 → 主**。只有主助理能委派（入栈），只有子助理能交还（出栈），禁止子助理之间横向跳转——否则 `dialog_state` 栈会失去意义，也会出现"交易助理直接跳投研助理"这种失控路由。

```mermaid
flowchart TD
    S(["START"]) --> P["fetch_user_profile<br/>每轮重查用户画像，不让模型回忆"]
    P --> M{"primary_assistant<br/>结构化输出分类"}

    M -->|"ToResearchAssistant"| ER["enter_research"]
    M -->|"ToMarketAssistant"| EM["enter_market"]
    M -->|"ToTradeAssistant"| ET["enter_trade"]
    M -->|"只读工具"| MT["primary_assistant_tools"]
    M -->|"直接回答"| E(["END"])
    MT --> M

    ER --> R["research 投研助理"]
    R -->|"lookup_fund_knowledge"| RT["research_tools<br/>自适应 RAG 子图"]
    RT --> R

    EM --> K["market 行情助理"]
    K -->|"只读查询"| KT["market_tools"]
    KT --> K

    ET --> T["trade 交易助理"]
    T -->|"只读核对"| TT["trade_safe_tools"]
    TT --> T
    T -->|"写操作，强制"| C["compliance_check<br/>规则引擎 + RAG 取证"]
    C -->|"pass：签发一次性票据"| G["approval_node<br/>interrupt()"]
    C -->|"block / 证据不足"| T
    G -->|"批准"| ST["trade_sensitive_tools<br/>原子核销票据后执行"]
    G -->|"拒绝"| T
    ST --> T

    R -->|"CompleteOrEscalate"| L["leave_skill<br/>dialog_state pop"]
    K -->|"CompleteOrEscalate"| L
    T -->|"CompleteOrEscalate"| L
    L --> M

    R -->|"回答完毕"| E
    K -->|"回答完毕"| E
    T -->|"回答完毕"| E
```

### 一次下单的完整链路

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant TA as trade 助理
    participant CC as compliance_check
    participant AN as approval_node
    participant PO as place_order

    U->>TA: 帮我买入 110020 一千元
    TA->>CC: tool_call place_order(product_code, amount)
    CC->>CC: 规则引擎：适当性 / 起购 / 余额 / 日限额
    CC->>CC: RAG 检索现行有效法条作为依据
    alt block 或证据不足
        CC-->>TA: ToolMessage（拦截原因 + 改方案建议）
    else pass
        CC->>CC: issue_ticket()，TTL 300s，签发前吊销旧票
        CC->>AN: interrupt()
        AN-->>U: SSE 事件 approval（审批卡片 + 合规依据）
        U->>AN: POST /approval  decision=yes / no
        alt yes
            AN->>PO: goto trade_sensitive_tools
            PO->>PO: consume_ticket()：校验即作废
            PO-->>TA: 订单号 / 成交状态
        else no
            AN-->>TA: ToolMessage（用户拒绝，未执行）+ 回到 trade
        end
    end
    TA-->>U: 自然语言结论
```

---

## 核心设计

### 1. 交接纪律决定栈是否成立

子助理交还控制权只有一个入口 `CompleteOrEscalate`，且必须在 prompt 里被定义成**有成本的动作**（`graph_fin/agent_assistant.py`）：

- 自检 1：用户是否**明确说出**换话题（"算了""不买了"），沉默或只回一两个字不算；
- 自检 2：是"本助理压根没有这类工具"，而不是"我调了工具但没查到"；
- 自检 3：交还前必须**真实调用过至少一次**工具。

配套的是工具分组而不是靠叮嘱：交易助理**刻意不给** `search_quotes`。一旦给了，"用户问行情就该交还"这条规则永远触发不了——模型自己查完就完事了。不给，才是让"超出能力"变成可判定事实的唯一可靠办法。

### 2. 合规：判定、取证、留痕分开

`graph_fin/compliance.py` 里三件事各司其职，任何一个失败都按**拒绝**处理（fail-closed）：

| 环节 | 由谁做 | 失败时 |
|---|---|---|
| 判定 | 纯规则引擎 `evaluate_rules()` | 账户/产品缺失 → `insufficient_evidence` |
| 取证 | 混合检索合规库，取 top3 法条 | 检索异常 → 按证据不足拒绝 |
| 表述 | LLM 只把结构化结论改写成两三句话 | LLM 失败 → 退回模板文本 |

规则覆盖：适当性匹配（C1–C5 → 最高可投 R 级，显式表 `RISK_CEILING`）、起购金额、可用资金、单日申购限额、集中度提示（只警告不拦截，避免误伤正常的大额低风险配置）。

三种结论只有 `pass` 进审批流；`block` 与 `insufficient_evidence` **根本不弹审批卡片**——给一个不该做的操作弹卡片，等于把合规责任转嫁给用户。

### 3. 票据：把合规约束下沉到函数第一行

票据是"这次写操作通过了合规"的唯一凭证，存在 PostgreSQL 而不是内存（进程重启、多实例、并发核销三个问题都交给数据库状态机）。三道闸门分布在三个层级：

1. **路由层**：模型输出敏感工具 → 条件边强制进 `compliance_check`；
2. **签票层**：`pass` 才写一张 `status='issued'` 的票，归属 `thread_id`，State 里不存票据信息；
3. **核销层**：`place_order` 第一行调用 `consume_ticket()`，一条 `UPDATE ... RETURNING` 同时完成"校验 + 作废"。

第三道是唯一绕不过去的：即使路由写错、即使人误批了，**没有票也下不了单**。票据按「工具名 + 标的 + 金额」三要素对齐，防"用买 A 的票买 B"和"批了 5 万实际下 50 万"。

### 4. 人工审批用动态 `interrupt()`

`approval_node` 里 `interrupt({})` 挂起，恢复时用 `Command(resume=...)` 把 `"yes"` / `"no"` 直接变成节点返回值，再由 `Command(goto=...)` 决定去哪（下面是简化示意，完整实现见 `graph_fin/build_child_graph.py`）：

```python
def approval_node(state):
    decision = interrupt({})                 # 挂起点
    if decision == "yes":
        return Command(goto="trade_sensitive_tools")
    # 拒绝：必须为**每一个** tool_call 补 ToolMessage，否则消息链里留下孤儿调用，
    # 下一轮调模型会直接 400，且该 thread 永久不可用
    return Command(goto="trade", update={"messages": [
        ToolMessage(content="用户拒绝了本次交易请求，交易未执行。",
                    tool_call_id=tc["id"]) for tc in state["messages"][-1].tool_calls
    ]})
```

选动态 `interrupt()` 而不是静态 `interrupt_before`，是因为**拒绝要成为正常路径**：静态断点只能"继续执行"或"不执行"，想做到"点拒绝就不下单"得靠 `update_state` 改 `next`，而 `as_node` 一旦指向出边不是条件边的节点就会静默失效——工具照常执行。另一个收益是条件审批：小额免审只需要在闸门前加一个金额判断。

**一个必须知道的约束**：恢复时整个 `approval_node` 会**重跑**。所以闸门内绝不能写库（签票在 `compliance_check`，核销在工具函数里），否则第二次执行就重复了。

### 5. 检索：两个库两套参数

`search/hybrid_retriever.py` 走原生 pymilvus 而不是 langchain 封装的 retriever，因为后者只能开内置 RRF，**调不了两路权重、也叠不了 MMR**。

```
query → dense(HNSW/IP, top60) ┐
      → sparse(BM25 中文, top60)┴→ 加权 RRF → [MMR 重排] → 阈值过滤 → 按 pk 回查正文
```

两套预设（`search/retriever_config.py`），参数不是拍的，是网格搜索出来的：

| | 投研库 `fund_research_v2` | 合规库 `finance_compliance_v2` |
|---|---|---|
| BM25 权重 | 2.0 | **10.0** |
| MMR | λ=0.9（要多样性） | **关闭**（法条答案唯一，重排会把正确答案挤下去） |
| 额外过滤 | — | `law_status == '现行有效'` |
| 检索目标 | 找散落在多份文档里的相关段落 | 精确定位唯一一条法条 |

关 MMR 不是偏好问题：合规库实测 Recall@5 从 94.9% 掉到 76.3%。用已废止的条款去放行一笔交易，比拒答严重得多，所以"现行有效"放在 Milvus 过滤表达式里，而不是检索完再筛。

投研侧还有一层自适应 RAG 子图（`rag_graph/`）：知识库 or 联网路由 → 混合检索 → 文档评级 → 不合格则改写查询（最多 2 次）→ 转联网兜底 → 带 `[序号]` 引用生成。

### 6. 工具层的三条硬规则

`tools/fin_tools.py` 开头写了三条，都是"出一次就是事故"的约束：

- `user_id` 从 `RunnableConfig` 取，取不到直接 `PermissionError`，**绝不退回默认值**；
- 写操作必须带 `client_order_id`，应用层查重 + 数据库 `UNIQUE` 双保险；
- 写操作第一行核销票据，核销不到就拒绝并写审计。

工具节点都套了 fallback（`tools/tools_handler.py`）：工具抛异常时把 `ToolMessage` 回灌给模型自愈，而不是让整轮崩掉。

---

## 目录结构

```
.
├── main.py                     FastAPI 入口：lifespan 里建 checkpointer、编译图
├── api/
│   ├── routes.py               POST /chat（SSE）与 POST /approval（恢复）
│   ├── user_chat_dto.py
│   └── user_approval_dto.py
├── graph_fin/
│   ├── graph_build.py          主图组装
│   ├── build_child_graph.py    三个子图 + approval_node + leave_skill
│   ├── agent_assistant.py      四个 prompt 与工具绑定、交接纪律
│   ├── route_fin.py            全部条件边（含最关键的交易路由）
│   ├── compliance.py           规则引擎 / 取证 / 审批卡片
│   ├── ticket.py               票据签发、原子核销、吊销
│   ├── state.py                State 定义 + dialog_state 栈 reducer
│   ├── entry_node.py           子助理入场节点（写栈 + 注入上下文）
│   ├── base_data_model.py      委派 / 交还的结构化输出模型
│   └── get_user_profile_node.py
├── rag_graph/                  自适应 RAG 子图（路由/检索/评级/改写/联网/生成）
├── search/
│   ├── hybrid_retriever.py     dense + BM25 + 加权 RRF + MMR
│   ├── rank_ops.py             融合与重排算法
│   └── retriever_config.py     两套检索预设
├── documents/                  Markdown 解析、语义切片、多进程写 Milvus
├── tools/
│   ├── fin_tools.py            只读 / 写 两级金融工具
│   ├── fin_db.py               业务库访问、建表、种子数据
│   ├── init_fin_db.py          初始化脚本（支持 --create-db / --force / --export-sql）
│   └── tools_handler.py        带 fallback 的 ToolNode
├── llm_model/                  LLM 与 embedding 客户端
├── untils/                     环境变量、日志、数据库配置（项目内历史拼写，保持原样）
└── test/                       工作流 / 接口 / 检索验证脚本
```

`datas/`、`logs/` 体积大且含语料，未纳入版本控制。

---

## 快速开始

### 1. 环境

- Python 3.11
- PostgreSQL 16（同时承担业务库与 LangGraph checkpointer）
- Milvus 2.5 Standalone（`http://127.0.0.1:19530`，需支持多进程写，因此不用 Milvus Lite）

### 2. 依赖

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 配置 `.env`

```bash
cp .env.example .env    # 然后填入真实值；.env 已在 .gitignore 中
```

```dotenv
# LLM / Embedding（阿里云百炼 OpenAI 兼容模式）
QWEN_KEY=sk-xxxxxxxx
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# 可选
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=
TAVILY_API_KEY=
UNSTRUCTURED_KEY=

# 业务库 + checkpointer
POSTGRES_URI=postgresql://postgres:your_password@127.0.0.1:5432/finagent
FINAGENT_DB_URL=postgresql+psycopg2://postgres:your_password@127.0.0.1:5432/finagent
```

两个连接串格式不同（一个给 psycopg 直连，一个给 SQLAlchemy 解析），指向同一个库。

### 4. 初始化业务库

```bash
python tools/init_fin_db.py --create-db     # 库不存在时先建库，再建表 + 灌种子数据
python tools/init_fin_db.py --force         # 清空业务表重建
python tools/init_fin_db.py --export-sql    # 只导出建表 SQL 到 datas/sql/
```

种子账户（用户名 / 风险等级 / 初始资金）：

| user_id | 姓名 | 等级 | 可投上限 | 单日限额 |
|---|---|---|---|---|
| `u_1001` | 张稳健 | C3 | R2 | 10 万 |
| `u_1002` | 李成长 | C4 | R3 | 30 万 |
| `u_1003` | 王激进 | C5 | R5 | 100 万 |
| `u_1004` | 赵保守 | C1 | R1 | 1 万 |

在售产品含 `000198`(R1) / `110020`(R3) / `510300`(R3) / `161725`(R4) / `519674`(R4) / `000001`(R5)。

### 5. 构建向量库

把语料放到 `datas/md/`（投研）与 `datas/md_law/`（法规），然后：

```bash
python -m documents.write_milvus
```

解析进程负责 Markdown → 按标题层级合并 → 超长块语义切割，写进程消费队列并批量写入；embedding 接口偶发 400/限流时按指数退避重试 3 次，最终失败会统计并报错，不静默丢数据。

> 管线本身与库无关：源目录和集合名配在 `documents/write_milvus.py` 的 `__main__` 里（默认 `datas/md` → `t_collection01`，且**每次运行会删集合重建**）。投研库与合规库各指一次目录、各跑一次即可，检索参数在 `search/retriever_config.py` 里按集合名对齐。

> 另一个容易踩的点：百炼 embedding 接口有双重上限（单批条数约 20~40、整批 token 约 2 万）。`llm_model/embedings_model.py` 里必须同时设 `check_embedding_ctx_length=False`（否则 langchain 会传 token id 数组，接口 400）和 `chunk_size`（默认 1000 必然 400）。

### 6. 启动

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Windows 上直接用 `uvicorn` 会踩一个坑：`uvicorn` 在 win32 上硬编码 `ProactorEventLoop`，而 psycopg3 的异步连接要求 `SelectorEventLoop`，表现为启动即退出。`test/run_server.py` 提供了绕过写法（`asyncio.Runner(loop_factory=SelectorEventLoop)`）：

```bash
python -m test.run_server     # 127.0.0.1:8123
```

下面的接口示例都按这个端口写。

---

## 接口

### `POST /chat`

```json
{ "message": "帮我买入 110020 一千元", "thread_id": "demo-001", "user_id": "u_1002" }
```

返回 `text/event-stream`，三种事件：

| 事件 | data | 含义 |
|---|---|---|
| 默认（`message`） | `{content, node, message_type, id, tool_calls}` | 模型 / 工具的增量输出，`node` 是产出它的节点 |
| `approval` | `{action, product_code, amount, evidence, warnings, ticket_id, expires_at}` | 图已挂起，等待人工审批 |
| `done` | `{}` | 本轮正常结束 |

```
data:{"content":"我先核对一下您的账户…","node":"market","message_type":"AIMessageChunk","id":"lc_run-...","tool_calls":[]}

event:approval
data:{"action":"place_order","product_code":"110020","amount":1000.0,"evidence":["[私募投资基金监督管理条例] …"],"warnings":[],"ticket_id":"38e4a14d…","expires_at":"2026-11-01T12:31:16"}
```

### `POST /approval`

```json
{ "thread_id": "demo-001", "user_id": "u_1002", "decision": "yes" }
```

`decision` 只接受 `"yes"` / `"no"`，直接作为 `Command(resume=...)` 的值。线程当前没有待处理的中断时返回 **409**，前端据此提示"审批已失效"。

```bash
curl -N -X POST http://127.0.0.1:8123/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"帮我买入 110020 一千元","thread_id":"demo-001","user_id":"u_1002"}'

curl -N -X POST http://127.0.0.1:8123/approval \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"demo-001","user_id":"u_1002","decision":"yes"}'
```

`thread_id` 决定会话与票据归属，跨请求、跨天恢复都靠它。反向代理（nginx）必须关掉响应缓冲并放大超时，否则 SSE 会被攒成一整块：

```nginx
proxy_buffering off;
proxy_read_timeout 300s;
proxy_set_header X-Accel-Buffering no;
```

---

## 测试

工作流与检索测试直接跑（需要先完成第 4、5 步：业务库与向量库就绪）：

```bash
# 七条主链路：直答 / 直查 / 委派投研 / 委派行情 / 合规拦截 / 全链路审批 / 幂等重放
# 注：审批那条用例仍按旧版静态断点（ainvoke(None)）写法，迁移到 interrupt() 后需同步改成 Command(resume=...)
python -m test.test_fin_workflow

# 检索效果
python -m test.test_search
```

接口测试需要服务已启动（`python -m test.run_server`）：

```bash
# 审批事件：跑到中断即停，校验卡片字段与"交易工具确实没执行"
python -m test.test_approval_sse

# 审批恢复接口契约：批准 / 拒绝 / 重复提交 409 / 无待审批 409
python -m test.test_approval_resume
```

测试都开独立 `thread_id`，互不污染。审计口径可直接查库核对：

```sql
SELECT status, count(*) FROM compliance_ticket GROUP BY status;   -- issued / consumed / revoked
SELECT result, count(*) FROM audit_log GROUP BY result;           -- filled / rejected / cancelled
```

---

## 已知限制

- **行情是确定性模拟值**（`zlib.crc32` 生成，同一代码每次同价），便于回归测试；接真实行情只需替换 `search_quotes` 内部实现，工具签名不变。
- **SSE 没有心跳**：一轮约 30s 无字节下发，局域网无碍，公网部署前建议每 15s 发一行注释帧 `: ping`。
- **`message` 事件包含节点内部输出**：子助理入场指令、工具返回的 JSON 都会下发，前端需要按 `node` / `message_type` 过滤，否则会看到不该给用户看的内容。
- **拒绝审批时不吊销票据**：票据会在库里空置到 TTL（300s）自然过期，尚未补 `revoke_ticket` 与 `cancelled` 审计记录。
- **向量库 `metadata["source"]` 存的是入库时的绝对路径**：`clean_source()` 在展示层裁剪，根治需要重建索引时改存相对路径。

## Roadmap

- [ ] 接入真实行情源与券商下单接口（当前为本地撮合模拟）
- [ ] 审批拒绝路径补 `revoke_ticket` + 审计留痕
- [ ] SSE 心跳与断连重连（`Last-Event-ID`）
- [ ] 合规规则外置为可配置规则集，支持按产品类型挂不同规则
- [ ] 检索评测纳入 CI，参数改动自动跑金标集

---

## 免责声明

本项目为工程能力演示，所有账户、产品、行情、法规文本均为演示数据，撮合为本地模拟，**不得用于真实交易**。项目不提供投资建议。
