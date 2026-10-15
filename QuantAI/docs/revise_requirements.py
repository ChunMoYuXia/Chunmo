"""Apply explicit review corrections to the project copy of the supplied PRD."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT / "reference" / "original-requirements.md"
text = source.read_text(encoding="utf-8-sig")


def replace_once(old: str, new: str) -> None:
    global text
    if text.count(old) != 1:
        raise ValueError(f"Expected one occurrence: {old[:90]!r}")
    text = text.replace(old, new, 1)


# The first incremental heading repeats the entire preceding index table.
duplicate_heading = "### 0.1.1 功能补充（多周期 / 模拟盘 / 选择式选股 / 搜索 / 动效）"
first = text.index(duplicate_heading)
second = text.index(duplicate_heading, first + len(duplicate_heading))
text = text[:first] + text[second:]

replace_once(
    "> 文档性质：PRD + 技术架构 + 接口契约 + 页面设计 + 风控合规 + 落地实施清单。",
    "> 文档性质：PRD + 技术架构 + 接口契约 + 页面设计 + 风控合规 + 落地实施清单。\n"
    "> 项目审阅修订：v2.3，2026-10-15。原件保存在 `reference/original-requirements.md`。\n"
    "> 本文件保留完整产品愿景；§18.20 为新增工程约束，开发阶段与验收范围以已批准的项目设计为准。\n"
    "> 页面、JSON、SQL 中的行情与收益数字均为示例，不代表已接入真实数据；未实现能力不得标为已完成。",
)
text = text.replace("SvelteKit 5", "SvelteKit 2 + Svelte 5")
replace_once(
    "前端技术细节：TypeScript 5、",
    "桌面构建：使用 `@sveltejs/adapter-static`，根布局 `ssr = false`，SPA fallback 为 `index.html`；"
    "Tauri 的 `frontendDist` 指向静态产物。桌面客户端不依赖 SvelteKit 服务端路由或 server actions。"
    "版本最终由锁文件固定，安装包大小必须实测。参见 [Tauri 官方 SvelteKit 指南](https://v2.tauri.app/start/frontend/sveltekit/)。\n\n"
    "前端技术细节：TypeScript 5、",
)
replace_once(
    "  transfer_fee_bps: 0.2            # 仅沪市征收，深市免费",
    "  transfer_fee_bps: 0.1            # 股票过户费双向；按交易日生效的规则版本选择，禁止固定覆盖历史",
)
text = text.replace("印花10bp", "印花5bp(卖)")
replace_once(
    "以 100 万资金、月轮为例（示意，非承诺）：\n"
    "  单笔单边成本：佣金 2.5bp + 过户费 0.2bp（仅沪市）+ 滑点 8bp ≈ 10.5bp（买入）/ 15.7bp（卖出，含印花税5bp+佣金2.5bp+过户0.2bp+滑点8bp）\n"
    "  单次换手（卖出旧的+买入新的）单边成本约 12–16bp，双边约 26bp\n"
    "  月换手 25% → 年换手 300% → 年成本 ≈ 300% × 16bp ≈ 4.8%（对初始资金，含冲击）\n\n"
    "若改为日轮：年换手可达 3000%–5000% → 年成本 ≈ 42%–70%\n"
    "  ⇒ 日频策略必须在\"成本前收益 > 60%/年\"的前提下才有意义（极难持续）\n"
    "  ⇒ 这就是为什么产品把日轮标记为\"高门槛\"，并要求成本前后对比强制展示",
    "以 100 万资金、买卖金额对称、佣金不触及最低收费为例（示意，非承诺）：\n"
    "  1bp = 0.0001 = 0.01%。金额 × bps / 10000 得到费用。\n"
    "  买入单边：佣金 2.5bp + 过户费 0.1bp + 滑点 8bp = 10.6bp。\n"
    "  卖出单边：佣金 2.5bp + 过户费 0.1bp + 滑点 8bp + 印花税 5bp = 15.6bp。\n"
    "  完整买卖一轮：26.2bp；本例不包含额外冲击与融资成本。\n"
    "  定义单边再平衡比例 T = (买入金额 + 卖出金额) / (2 × 调仓前净资产)。\n"
    "  月 T=25%，简化年 T=3 倍；年成本约 3 × 26.2 / 10000 = 0.786%。\n"
    "  年 T=30–50 倍时，同样假设下年成本约 7.86%–13.10%。\n"
    "  实际费用须逐笔计算最低佣金、成交额、费率有效期与冲击，不能套用固定收益门槛。\n"
    "  双边成交额比例 = 2T；展示与导出必须标明采用的换手口径。",
)
replace_once(
    "3) 复权：前复权计算收益，后复权（或原始价+复权因子）计算资金与股数",
    "3) 复权：特征使用在该时点可得的复权版本；成交、现金、股数一律用原始可交易价格；"
    "分红送转由公司行为账本处理，禁止复权收益与现金分红重复计入",
)
replace_once(
    "4) 成交假设：次日开盘/收盘/VWAP/自定义；一笔限制为当日成交量比例（默认 10%）",
    "4) 成交假设：明确 signal_time、submitted_at、eligible_at、filled_at；收盘信号最早次交易日执行；"
    "开盘下单容量只能使用此前已知的 ADV/盘口，禁止用当日完整成交量决定开盘订单",
)
replace_once(
    "| **现金分红** | 除息日（ex_date）持仓股数不变，现金按 `每股分红 × 股数` 计入现金（T+1～T+3 到账，可配置），同时股价自然回落 | 忽略分红 → 净值被低估 |",
    "| **现金分红** | 根据股权登记日权益，在除息日确认应收股利；按公告支付日转为可用现金。税费按账户与持有期规则记录；成交和估值使用原始价 | 把应收直接当可用现金，或复权收益与分红重复计入 |",
)
replace_once(
    "| **退市/摘牌** | 按最后交易价或退市整理期价格清算，记录损失 | 从样本中消失（幸存者偏差） |",
    "| **退市/摘牌** | 保留历史样本、持仓与损失；只有真实可成交或清算回款证据才能结清，缺乏流动性时按版本化估值规则计提减值 | 从样本消失，或假设能按最后价卖出 |",
)
replace_once(
    "| 追踪 | 头 `X-Trace-Id`（可选，未传由网关生成）、`X-Request-Id`（幂等键，POST 建议携带） |",
    "| 追踪 | `X-Trace-Id` / `X-Request-Id` 用于关联请求；写操作独立使用 `Idempotency-Key`，订单必填 |",
)
replace_once(
    "| 幂等 | 带 `X-Request-Id` 的写操作在 24h 内重放返回首次结果 |",
    "| 幂等 | 键作用域为账户 + 操作；同键同载荷返回原结果，同键异载荷返回 409。普通写操作保留至少 24h；订单幂等记录按交易审计周期持久保留 |",
)
replace_once(
    "  portfolio_id BIGINT, account_id BIGINT, code TEXT, side TEXT,",
    "  portfolio_id BIGINT, account_id BIGINT NOT NULL, code TEXT, side TEXT,",
)
replace_once(
    "  request_id TEXT,             -- X-Request-Id，防重放\n"
    "  CONSTRAINT uq_orders_request_id UNIQUE (request_id)\n"
    ");\n"
    "CREATE UNIQUE INDEX idx_orders_request_id ON orders (request_id) WHERE request_id IS NOT NULL;",
    "  request_id TEXT NOT NULL,    -- Idempotency-Key；作用域为账户\n"
    "  request_hash TEXT NOT NULL,  -- 规范化请求内容摘要，同键异内容返回409\n"
    "  CONSTRAINT uq_orders_account_request UNIQUE (account_id, request_id),\n"
    "  CONSTRAINT ck_orders_quantity CHECK (quantity > 0),\n"
    "  CONSTRAINT ck_orders_filled CHECK (filled_quantity >= 0 AND filled_quantity <= quantity)\n"
    ");",
)
replace_once(
    "  before JSONB, after JSONB, result TEXT, trace_id TEXT,\n"
    "  prev_hash TEXT, hash TEXT NOT NULL",
    '  "before" JSONB, "after" JSONB, result TEXT, trace_id TEXT,\n'
    "  chain_id TEXT NOT NULL, sequence_no BIGINT NOT NULL,\n"
    "  payload_hash TEXT NOT NULL, prev_hash TEXT, hash TEXT NOT NULL,\n"
    "  CONSTRAINT uq_audit_sequence UNIQUE (chain_id, sequence_no)",
)
replace_once(
    "**审计哈希链**：`hash = sha256(prev_hash || ts || actor_id || action || resource_type || resource_id || coalesce(before::text,'') || coalesce(after::text,''))`，其中 `payload_hash` = `before` 和 `after` 两个字段的组合 hash。\n"
    "ALTER TABLE audit_logs ADD COLUMN payload_hash TEXT GENERATED ALWAYS AS (sha256(COALESCE(before::text,'') || COALESCE(after::text,''))) STORED;\n"
    "CREATE UNIQUE INDEX ON audit_logs (hash);\n"
    "每日归档封存，任何篡改都会导致后续哈希不匹配（可用 `verify_audit_chain` 作业校验）。",
    "**审计哈希链**：业务层用固定版本的规范化 JSON 字节序列计算 `payload_hash`，内容含 before/after、actor、action、resource、result、trace_id。"
    "链摘要包含 chain_id、sequence_no、UTC 时间、prev_hash 与 payload_hash，禁止无分隔字符串拼接。\n\n"
    "追加时在同一事务中锁定链头、分配序号、写日志并更新链头，防止并发分叉。PostgreSQL 16 可使用 "
    "`encode(digest(canonical_bytes, 'sha256'), 'hex')`（先启用 pgcrypto）；不要对 text 直接调用 `sha256(text)`。"
    "生产环境定期签名链头并封存到独立只追加存储；数据库管理员可重算整条链，因此数据库内 hash 链仅提供篡改检测。"
    "见 [pgcrypto 官方文档](https://www.postgresql.org/docs/16/pgcrypto.html)。",
)
replace_once(
    "紧急停止需输入确认短语 + 二次确认；撤单走幂等号。",
    "紧急停止立即进入服务端禁止新单状态并请求撤单，显示逐单结果；恢复交易才需确认短语 + 二次确认。撤单走幂等号。",
)
text = text.replace("紧急停止（需二次确认）", "紧急停止（立即生效，恢复需确认）")
replace_once(
    "| 本地数据库 | 离线行情/报告缓存、最近查看、未发送操作队列 |",
    "| 本地数据库 | 离线行情/报告缓存、最近查看、研究草稿；离线交易仅保存草稿，重连后重验并由用户确认 |",
)
replace_once(
    "| 崩溃恢复 | 未保存编辑与未发送操作恢复 |",
    "| 崩溃恢复 | 未保存编辑恢复；交易先查询服务端状态，禁止自动重放未确认订单 |",
)
replace_once(
    "### 18.19 论文审查改进建议（v2.2 补丁）\n",
    "### 18.19 论文审查改进建议（v2.2 补丁，研究候选）\n\n"
    "> v2.3 审阅限制：本节保留研究线索，不构成默认架构或确定性能承诺。"
    "论文实验条件、理论假设、数据与模型版本须先核验；提升比例和安全成功率须由本项目评测复现。"
    "§18.19.10 中新增数值门槛属于实验目标，不替代 §16 的正式验收。"
    "CPCV 变体、DDQN、ECI、多面 Critic 与 GraphRAG 延后至独立研究阶段；安全授权与交易校验由确定性代码执行。\n",
)
replace_once(
    "FATH（认证防御，arXiv:2509.02718）：",
    "FATH（研究防御，arXiv:2410.21492，2024；原编号指向无关的多模型路由论文，已纠正）：",
)
replace_once(
    "  · 集成位置：§5.7 输出护栏之后；自适应攻击成功率降至 ≤10%",
    "  · 仅在隔离实验中评估输出过滤效果；不能据此保证本项目攻击成功率 ≤10%，不能授予工具权限或替代服务端鉴权",
)

supplement = (ROOT / "engineering-supplement.md").read_text(encoding="utf-8")
replace_once("## 19. 免责声明", supplement.rstrip() + "\n\n---\n\n## 19. 免责声明")
text = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
(ROOT / "requirements.md").write_text(text, encoding="utf-8", newline="\n")
print("Wrote docs/requirements.md from preserved source with explicit review corrections.")
