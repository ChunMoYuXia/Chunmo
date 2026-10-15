# M0 本地研究与模拟客户端 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付本地可运行、计算可复现的研究/回测/模拟客户端及可上传GitHub的项目产物。

**Architecture:** Svelte静态SPA在浏览器与Tauri复用；FastAPI模块化单体、SQLite持久化。领域独立于Web/数据库，回测与模拟盘共用成本、账本、时间和执行规则。原124条FR完整保留，本计划验收M0。

**Tech Stack:** SvelteKit2、Svelte5、TypeScript strict、Tauri2、FastAPI、Pydantic2、SQLAlchemy2、SQLite、NumPy/SciPy、pytest、Vitest、Playwright；项目局部环境与锁文件。

**Spec:** [已批准设计](../specs/2026-10-15-stock-portfolio-client-design.md)，§4、§6–11。

**Status:** 14项实施及独立评审完成；源码、便携运行时、Windows x64原生编译 / NSIS安装 / 实际窗口与退出清理验收均通过。真实AI未配置，GitHub仓库选择和上传按用户要求留待后续。

## Global Constraints

- 独立根目录：`D:\workplace\stock-portfolio-client`。
- `api → application → domain`；infrastructure实现domain定义的协议，通过组合根注入。
- domain不得依赖Web、数据库、厂商SDK；同一领域内核供回测与模拟盘调用。
- 内部标的键为security_id，包含交易所映射；金额Decimal、数量整数，JSON金额字符串，配置和返回对象带单位。
- 缺失值不填0。合成日历只用于演示，不能冒充真实节假日。
- 收盘形成的信号最早次交易日执行；开盘容量禁止引用当日最终成交量。
- 分红应收与现金支付分开，退市不产生虚构卖出。
- 所有金融写操作需Idempotency-Key和规范化载荷摘要；同账户同操作同键异内容409。
- 紧急停止立即禁新单，恢复才二次确认；离线订单不自动发送。
- 用户已明确选择“先完成本地项目，稍后确定仓库”。当前不创建或上传远程仓库。
- 某条FR在M0只实现子集时标为部分，不因页面存在就算完整通过。

## Review Focus

1. BOM/中文表头/前导零代码CSV：代码原样保留，错误带行号，Task2。
2. NaN/Infinity、空值、恒定价格、不足窗口：拒绝非法输入，指标缺失明确，Task3/5/8。
3. 同键并发与成交事务中途失败：不重复成交、资金持仓审计一致，Task9。
4. 跨年、周期末、非交易日：日历驱动次交易日执行，Task7。
5. 外部网页伪装本地客户端：缺令牌/非法Origin不能读取个人数据，Task1/14。

## 文件与接口约定

路径相对项目根。Python包为`apps/api/src/portfolio_client`，测试为`apps/api/tests`，前端为`apps/client`。所有公开函数有类型，路由只做DTO/鉴权/用例调用；前端页面只编排功能组件，不复制金融公式。

| 文件组 | 职责 |
| ------ | ---- |
| `domain/market/{models,calendar,ports}.py` | 标的、原始日线、版本、交易日历、读取协议 |
| `domain/screening/{dsl,engine}.py` | 受控DSL、筛选和剔除归因 |
| `domain/portfolio/{models,optimizers,rebalancing}.py` | 权重与约束、整股建议 |
| `domain/accounting/{models,costs,ledger,corporate_actions}.py` | Decimal账本、费用和公司行为 |
| `domain/execution/{models,rules,matching}.py` | 订单状态与日线保守撮合 |
| `domain/research/{features,metrics,backtest,reports}.py` | 研究计算与证据 |
| `application/{datasets,market,screening,portfolios,research,risk,paper,jobs,reports}.py` | 用例与事务边界 |
| `infrastructure/db/{models,repositories,migrations}.py` | 持久化与迁移 |
| `api/routes/{market,screening,portfolios,research,paper,jobs,reports,ai}.py` | 类型化HTTP边界 |
| `apps/client/src/lib/features/*` | 业务组件和局部状态 |
| `contracts/openapi.json`、`src/lib/api/schema.d.ts` | 后端导出与生成前端类型 |

统一成功`{data,meta:{trace_id}}`，失败`{error:{code,message,details,trace_id}}`；`DomainError(code,message,details)`映射HTTP状态。API时间为含时区ISO8601，日期YYYY-MM-DD，金额字符串；未知指标为null并附reason。

## Task 1：可运行API、统一错误与客户端骨架

**Files:** `apps/api/pyproject.toml`、`src/portfolio_client/{main,config}.py`、`api/{errors,security}.py`、`tests/{conftest,test_app}.py`；`apps/client/package.json`、`svelte.config.js`、`src/routes/{+layout.ts,+layout.svelte}`、`src/lib/api/{client.ts,client.test.ts}`；根`.gitignore`、`scripts/dev.ps1`。

**Interfaces:** `Settings(database_url,data_dir,session_token,allowed_origins)`；`create_app(settings: Settings) -> FastAPI`；前端`request<T>(path:string,init?:RequestInit):Promise<T>`。conftest创建临时SQLite与固定测试令牌，client是TestClient，auth为Bearer测试头。

- [x] 写失败测试与夹具：

```python
@pytest.fixture
def client(tmp_path):
    settings = Settings(database_url=f'sqlite:///{tmp_path}/test.db', data_dir=tmp_path, session_token='test-token', allowed_origins=['http://127.0.0.1:5173'])
    with TestClient(create_app(settings)) as test_client:
        yield test_client

@pytest.fixture
def auth():
    return {'Authorization': 'Bearer test-token'}

def test_local_api_boundary(client, auth):
    assert client.get('/health').status_code == 200
    assert client.get('/v1/config').status_code == 401
    assert client.get('/v1/config', headers=auth).status_code == 200
    assert client.get('/v1/config', headers={**auth, 'Origin': 'https://untrusted.example'}).status_code == 403
```

- [x] Run: `.venv/Scripts/python -m pytest apps/api/tests/test_app.py -q`，确认缺接口时失败。
- [x] 安装项目局部依赖并锁版本，实现composition root、token/Host/Origin校验、trace错误映射；SPA配置`adapter({fallback:'index.html'})`、根布局`export const ssr=false`。启动脚本只绑定127.0.0.1，不打印密钥，设置校验失败指出字段。

```python
provided = request.headers.get('Authorization', '').removeprefix('Bearer ')
if not secrets.compare_digest(provided, settings.session_token):
    raise DomainError('UNAUTHENTICATED', '需要本地会话令牌', {})
origin = request.headers.get('Origin')
if origin is not None and origin not in settings.allowed_origins:
    raise DomainError('ORIGIN_REJECTED', '请求来源不受信任', {})
```
- [x] 上述测试通过，前端`npm --prefix apps/client run check`、`run build`通过；fetch客户端把401/409/422/500转为带trace的错误，不吞异常。
- [x] 定向暂存并提交：`feat: add runnable local API and client shell`。

## Task 2：数据导入、版本与合成样本

**Files:** `domain/market/{models,calendar,ports}.py`；`application/datasets.py`；`infrastructure/imports/{csv,synthetic}.py`；`infrastructure/db/{models,repositories,migrations}.py`；`api/routes/market.py`；`tests/{test_imports,test_dataset_repository}.py`；`samples/README.md`。

**Interfaces:** frozen dataclass `Security(security_id,code,exchange,name,industry,initials)`；`DailyBar(security_id,trade_date,open,high,low,close,volume,amount,available_at)`；`DatasetSnapshot(dataset_id,version,calendar_id,source_kind)`；`preview_import(content:bytes,mapping:dict[str,str])->ImportPreview`；`commit_import(preview_id:str)->DatasetSnapshot`；`generate_dataset(seed:int,securities:int,sessions:int)->ImportPreview`；`TradingCalendar(sessions:list[date]).next_session(day:date)->date`。价格Decimal，security_id为交易所+代码。

- [x] 写失败测试：

```python
def test_csv_preserves_code_and_reports_bad_row():
    payload = '\ufeff代码,日期,开盘,最高,最低,收盘,成交量\n000001,2026-01-05,10,11,9,10.5,1000\n000002,2026-01-05,12,10,11,12,1000\n'.encode('utf-8')
    result = preview_import(payload, mapping={})
    assert result.rows[0].code == '000001'
    assert result.errors[0].row == 3
    assert result.errors[0].code == 'INVALID_OHLC'
```

- [x] Run: `.venv/Scripts/python -m pytest apps/api/tests/test_imports.py apps/api/tests/test_dataset_repository.py -q`，确认失败。
- [x] 中文映射、编码、非空/有限Decimal/OHLC/重复键/时间/日历校验；预检不改有效数据，确认后事务提交，保存原文摘要、来源、日历、ingest_time和version；相同内容复用。seed确定合成随机流，生成停牌/涨跌停/公司行为样本并标识source_kind=synthetic。

```python
prices = [Decimal(row[name]) for name in ('open', 'high', 'low', 'close')]
if any(not price.is_finite() or price <= 0 for price in prices):
    raise DomainError('INVALID_PRICE', '价格必须为有限正数', {'row': line_number})
opening, high, low, close = prices
if high < max(opening, close, low) or low > min(opening, close, high):
    raise DomainError('INVALID_OHLC', '高低价范围不合法', {'row': line_number})
```
- [x] 测试同seed同摘要、失败无半批次、SQLite重启版本保留、跨交易所代码不碰撞、未知日历拒绝；补GET数据集与导入预检/确认API。
- [x] 提交：`feat: add validated versioned datasets`。

## Task 3：时点行情、搜索与特征

**Files:** `domain/research/features.py`、`application/market.py`、`infrastructure/db/repositories.py`、`api/routes/market.py`；`tests/{test_features,test_market_api}.py`。

**Interfaces:** `MarketDataPort.bars(dataset_id:str,security_ids:list[str],as_of:datetime)->list[DailyBar]`；`compute_features(bars:list[DailyBar],window:int=20)->FeatureSnapshot`，返回close/momentum_20d/volatility_20d/average_amount_20d及证据；`search_securities(query:str,limit:int=20)->list[Security]`。

- [x] 写失败测试：

```python
def test_short_history_has_no_momentum():
    assert compute_features([]).momentum_20d is None
    assert compute_features([]).volatility_20d is None
```

- [x] Run: `.venv/Scripts/python -m pytest apps/api/tests/test_features.py apps/api/tests/test_market_api.py -q`，确认失败。
- [x] 按版本和available_at过滤；动量`close[t]/close[t-20]-1`，波动为样本std(日收益)×sqrt(244)，不足窗口None；不把缺失日插值为成交。搜索精确代码→前缀→名称/拼音首字母/行业，稳定排序与limit。结果包含来源/as_of/version。

```python
visible = [bar for bar in bars if bar.available_at <= as_of]
closes = np.asarray([float(bar.close) for bar in visible], dtype=float)
momentum = float(closes[-1] / closes[-21] - 1) if len(closes) >= 21 else None
returns = np.diff(closes) / closes[:-1] if len(closes) > 1 else np.array([])
volatility = float(np.std(returns[-20:], ddof=1) * np.sqrt(244)) if len(returns) >= 20 else None
```
- [x] 创建21条固定日线测试价格从100到120动量=.2，尾部未来价格改动不改变as_of历史结果；NaN/Infinity/零价导入拒绝、恒定价格波动0但不自动视为有效优化输入。
- [x] 提交：`feat: add time-aware search and features`。

## Task 4：筛选DSL与漏斗证据

**Files:** `domain/screening/{dsl,engine}.py`、`application/screening.py`、`api/routes/screening.py`、筛选方案DB模型；`tests/{test_screening,test_screening_api}.py`。

**Interfaces:** `ScreeningSpec(version,dataset_id,as_of,universe,filters,sort_by,descending,limit)`；Filter仅白名单字段和gt/gte/lt/lte/eq/in；`screen(spec:ScreeningSpec,snapshot:list[FeatureSnapshot])->ScreeningResult`返回steps与rows。保存/恢复方案包含原universe与条件列表。

- [x] 写失败API测试：

```python
def test_unknown_filter_is_rejected(client, auth):
    payload = {'version':1,'dataset_id':'x','as_of':'2026-01-05T15:00:00+08:00','universe':[], 'filters':[{'field':'__class__','operator':'eq','value':'x'}], 'sort_by':'close','descending':True,'limit':20}
    assert client.post('/v1/screening', json=payload, headers=auth).status_code == 422
```

- [x] Run: `.venv/Scripts/python -m pytest apps/api/tests/test_screening.py apps/api/tests/test_screening_api.py -q`，确认失败。
- [x] 用显式operator函数表比较类型化值，禁止eval/自由SQL；每步返回input_count/remaining_count/exclusions，缺失值按missing_value剔除。回退重算条件前缀；同分用security_id稳定排序。

```python
operators = {'gt': operator.gt, 'gte': operator.ge, 'lt': operator.lt, 'lte': operator.le, 'eq': operator.eq, 'in': lambda left, right: left in right}
if condition.operator not in operators:
    raise DomainError('INVALID_OPERATOR', '不支持此筛选运算', {})
value = getattr(feature, condition.field)
passed = value is not None and operators[condition.operator](value, condition.value)
```
- [x] 用3条特征：1条低于门槛、1条缺失、1条通过，验证计数3→1及剔除原因各1；测试空池、未知证券、Top-N和方案重载一致。
- [x] 提交：`feat: add validated screening and funnel evidence`。

## Task 5：组合优化、约束与整股计划

**Files:** `domain/portfolio/{models,optimizers,rebalancing}.py`、`application/portfolios.py`、`api/routes/portfolios.py`、组合DB模型；`tests/{test_optimizers,test_rebalancing,test_portfolio_api}.py`。

**Interfaces:** `optimize_weights(returns:ndarray,method:Literal['equal','inverse_volatility','minimum_variance'],max_weight:float)->WeightSolution`；`plan_rebalance(target_weights:dict[str,Decimal],holdings:dict[str,int],prices:dict[str,Decimal],cash:Decimal,lot_sizes:dict[str,int],fee_schedule:FeeSchedule)->RebalancePlan`。FeeSchedule由Task6定义，整股费用集成在Task6后完成。

- [x] 写失败测试：

```python
def test_impossible_cap_has_specific_error():
    with pytest.raises(DomainError) as error:
        optimize_weights(np.array([[.01,.02],[.02,.01],[.03,.02]]), 'equal', .4)
    assert error.value.code == 'INFEASIBLE_CONSTRAINTS'
```

- [x] Run: `.venv/Scripts/python -m pytest apps/api/tests/test_optimizers.py -q`，确认失败。
- [x] equal受上限约束；inverse_volatility拒绝零波动/不足窗口；minimum_variance用收缩协方差+SciPy有界和约束求解，检查求解状态、finite、sum≈1、0≤w≤cap。逆波动率不可命名完整风险平价。组合与目标持久化，整股前转Decimal并扣预留费用，拒绝超可卖/资金不足。

```python
assets = returns.shape[1]
if assets * max_weight < 1 - 1e-10:
    raise DomainError('INFEASIBLE_CONSTRAINTS', '个股上限不足以分配全部权重', {})
covariance = np.cov(returns, rowvar=False, ddof=1)
covariance = .9 * covariance + .1 * np.diag(np.diag(covariance))
solution = scipy.optimize.minimize(lambda w: float(w @ covariance @ w), np.full(assets, 1 / assets), method='SLSQP', bounds=[(0, max_weight)] * assets, constraints=[{'type':'eq','fun':lambda w: w.sum() - 1}])
if not solution.success:
    raise DomainError('OPTIMIZATION_FAILED', solution.message, {})
```
- [x] 验证独立二维最小方差解析基线、恒定/NaN矩阵明确错误、重启组合保留、未知价不造单。Task6后测试最低佣金与整股余款，资金不得负数。
- [x] 提交：`feat: add persistent portfolios and baseline optimization`。

## Task 6：费用、账本与公司行为内核

**Files:** `domain/accounting/{models,costs,ledger,corporate_actions}.py`；`tests/{test_costs,test_ledger,test_corporate_actions,test_rebalancing}.py`。

**Interfaces:** `FeeSchedule(version,valid_from,valid_to,commission_bps,minimum_commission,transfer_bps,stamp_bps)`；`calculate_fees(notional:Decimal,side:Side,schedule:FeeSchedule)->FeeBreakdown`；`AccountState(cash,frozen_cash,withdrawable_cash,receivables,positions)`；`apply_fill(state:AccountState,fill:Fill,fees:FeeBreakdown)->AccountState`；`apply_corporate_action(state,action,session)->AccountState`。Fill字段为security_id/side/quantity/price/session，Side为buy/sell；返回新状态与账本事件。

- [x] 写失败测试：

```python
def test_stock_fees_buy_and_sell():
    schedule = FeeSchedule('cn-demo',date(2023,8,28),None,Decimal('2.5'),Decimal('5'),Decimal('.1'),Decimal('5'))
    buy = calculate_fees(Decimal('100000'), Side.BUY, schedule)
    sell = calculate_fees(Decimal('100000'), Side.SELL, schedule)
    assert (buy.commission,buy.transfer,buy.stamp) == (Decimal('25.00'),Decimal('1.00'),Decimal('0.00'))
    assert sell.stamp == Decimal('50.00')
```

- [x] Run: `.venv/Scripts/python -m pytest apps/api/tests/test_costs.py apps/api/tests/test_ledger.py apps/api/tests/test_corporate_actions.py -q`，确认失败。
- [x] 明确`BPS_DIVISOR=Decimal('10000')`；按effective日选费率，费用项目各自舍入到分，是否含规费配置明确；滑点进入成交价，不能再扣一次。现金/冻结/可取/可卖/应收及delta事件独立；除息确认应收、pay_date释放现金，送转调整数量成本。

```python
BPS_DIVISOR = Decimal('10000')
commission = max(schedule.minimum_commission, notional * schedule.commission_bps / BPS_DIVISOR)
transfer = notional * schedule.transfer_bps / BPS_DIVISOR
stamp = notional * schedule.stamp_bps / BPS_DIVISOR if side is Side.SELL else Decimal('0')
rounded = [value.quantize(Decimal('.01'), rounding=ROUND_HALF_UP) for value in (commission, transfer, stamp)]
```
- [x] 测试100股@10加最低5元佣金的独立预期账本、历史费率切换、T+1、超资金/超可卖拒绝、股利未到账不能买、分红不双计与退市无虚构清算；完成Task5整股费用集成。
- [x] 提交：`feat: implement deterministic fees and accounting ledger`。

## Task 7：共用撮合与日周月年回测

**Files:** `domain/execution/{models,rules,matching}.py`、`domain/research/backtest.py`、`application/research.py`；`tests/{test_matching,test_backtest}.py`。

**Interfaces:** `Order(order_id,account_id,security_id,side,quantity,order_type,limit_price,submitted_at,eligible_session,state)`；`match_order(order:Order,bar:DailyBar,known_liquidity,rules)->MatchResult`；`BacktestRequest(dataset_id,calendar_id,security_ids,weights,frequency,start,end,capital,fee_version,seed)`；`run_backtest(request:BacktestRequest,market:MarketDataPort,progress:Callable[[Progress],None])->BacktestResult`，含nav/positions/trades/costs/evidence/assumptions。Progress为completed/total/message。

- [x] 写失败日历测试：

```python
def test_year_end_uses_next_session():
    calendar = TradingCalendar([date(2025,12,30),date(2025,12,31),date(2026,1,5)])
    assert calendar.next_session(date(2025,12,31)) == date(2026,1,5)
```

- [x] Run: `.venv/Scripts/python -m pytest apps/api/tests/test_matching.py apps/api/tests/test_backtest.py -q`，确认失败。
- [x] 周期末由下一session所属周/月/年变化识别；首次建仓至少有完整窗口。循环为公司行为→既有可执行单→收盘估值→次日信号。撮合验状态/最小单位/T+1/涨跌停和此前已知ADV，原始开盘价+方向滑点且不越limit，日线局限写assumptions。共用Task6账本。

```python
def next_session(self, day: date) -> date:
    index = bisect.bisect_right(self.sessions, day)
    if index == len(self.sessions):
        raise DomainError('CALENDAR_EXHAUSTED', '日历没有下一交易日', {})
    return self.sessions[index]
```
- [x] 测试跨年首单次交易日、未来bar改动不影响过去信号、当日最终volume不改变开盘容量、停牌/一字板/缺窗口/整股余款、同版本seed复现。只在无费用/整股/流动性差异的受限条件与独立向量收益基线比较，声明条件。
- [x] 提交：`feat: add calendar-driven backtesting with shared execution rules`。

## Task 8：风险绩效与可追踪指标

**Files:** `domain/research/metrics.py`、`application/risk.py`、`api/routes/research.py`；`tests/{test_metrics,test_risk_api}.py`。

**Interfaces:** `max_drawdown(nav:list[float])->float|None`；`compute_metrics(nav:list[NavPoint],benchmark:list[NavPoint]|None,annualization:int,risk_free_annual:float)->PerformanceMetrics`；`historical_tail_risk(returns:ndarray,confidence:float=.95)->TailRisk`；`exposures(holdings,prices,securities)->ExposureResult`。NavPoint含session/value，输出指标value/unit/formula_version/source/as_of/reason。

- [x] 写失败测试：

```python
def test_drawdown_and_empty_history():
    assert max_drawdown([100.0,120.0,90.0,110.0]) == pytest.approx(.25)
    assert max_drawdown([]) is None
```

- [x] Run: `.venv/Scripts/python -m pytest apps/api/tests/test_metrics.py apps/api/tests/test_risk_api.py -q`，确认失败。
- [x] 日收益nav[t]/nav[t-1]-1，drawdown=max(1-nav/running_max)，vol=样本std×sqrt(N)，rf转有效日利率；Sharpe分母0/不足样本null附reason。VaR/CVaR损失为正，分位方法固定；现金进入组合分母，基准按日期对齐并声明价格/全收益口径。

```python
def max_drawdown(nav: list[float]) -> float | None:
    if not nav:
        return None
    values = np.asarray(nav, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise DomainError('INVALID_NAV', '净值必须为有限正数', {})
    return float(np.max(1 - values / np.maximum.accumulate(values)))
```
- [x] 4点净值与固定损失数组手算比对，空/非正NAV/常数/基准缺日/全现金/未知行业测试，禁止缺数据返回零风险。
- [x] 提交：`feat: expose reproducible risk and performance metrics`。

## Task 9：模拟账户、事务幂等与暂停

**Files:** `application/paper.py`、`api/routes/paper.py`、DB账户/订单/成交/账本/幂等/审计模型与迁移；`tests/{test_paper_transactions,test_paper_api}.py`。

**Interfaces:** `PaperTradingService.create_account(request)->PaperAccount`、`submit_order(account_id:str,request:OrderRequest,key:str)->OrderResult`、`advance(account_id:str,session:date,key:str)->AccountSnapshot`、`halt(account_id:str)->HaltResult`、`resume(account_id:str,confirmation:str)->AccountSnapshot`。OrderRequest字段与Task7 Order相同，移除服务端生成的ID/state；资金状态由Task6返回。

- [x] 写失败API测试（本测试创建账户后使用返回id）：

```python
def test_idempotent_order_payload_conflict(client, auth):
    account = client.post('/v1/paper/accounts',json={'name':'test','capital':'100000','dataset_id':'synthetic'},headers=auth).json()['data']
    path = f"/v1/paper/accounts/{account['id']}/orders"
    headers = {**auth,'Idempotency-Key':'order-1'}
    body = {'security_id':'XSHE:000001','side':'buy','quantity':100,'order_type':'market'}
    first = client.post(path,json=body,headers=headers)
    assert client.post(path,json=body,headers=headers).json()['data']['id'] == first.json()['data']['id']
    assert client.post(path,json={**body,'quantity':200},headers=headers).status_code == 409
```

- [x] Run: `.venv/Scripts/python -m pytest apps/api/tests/test_paper_transactions.py apps/api/tests/test_paper_api.py -q`，确认失败；在测试conftest建立Task2已提交合成dataset别名synthetic，非产品硬编码数据源。
- [x] 同一数据库写事务读取并更新账户，account+operation+key唯一，canonical载荷摘要/原响应持久化。订单/冻结/成交/账本/审计/幂等结果原子提交；链头按chain_id/seq串行，规范化JSON含schema_version。halt立即paused并撤可撤订单，结果不确定显示unknown；resume确认短语。domain使用Task6/7，不复制撮合。

```python
canonical = json.dumps(request.model_dump(mode='json'), sort_keys=True, separators=(',', ':'), ensure_ascii=False)
request_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
if prior_record is not None and prior_record.request_hash != request_hash:
    raise DomainError('IDEMPOTENCY_CONFLICT', '相同幂等键的订单内容不同', {})
```

```sql
CREATE UNIQUE INDEX uq_idempotency_scope ON idempotency_records(account_id, operation, key);
```
- [x] 两连接并发同键仅一单，注入成交异常后资金/持仓/订单/审计均回滚，跨账户同键允许，重启不丢账，暂停后拒新单，离线草稿不能自动提交；审计篡改检测但不宣称抵抗管理员重算整链。
- [x] 提交：`feat: persist paper trading with atomic ledger and idempotency`。

## Task 10：持久化任务、取消与生成契约

**Files:** `application/jobs.py`、`infrastructure/jobs/runner.py`、DB任务模型、`api/routes/{jobs,research}.py`；`scripts/export_openapi.py`、`contracts/openapi.json`、`apps/client/src/lib/api/{schema.d.ts,jobs.ts,jobs.test.ts}`；`tests/test_jobs.py`。

**Interfaces:** `JobService.submit(kind,payload,key)->JobRecord`、`get(job_id)->JobRecord`、`cancel(job_id)->JobRecord`；状态queued/running/succeeded/failed/cancelled；前端`pollJob<T>(jobId,signal,onProgress):Promise<T>`；Task7 Progress原样保存，不另造字段。

- [x] 写失败测试：

```python
def test_cancel_unknown_job_is_explicit(client, auth):
    response = client.post('/v1/jobs/does-not-exist/cancel',headers=auth)
    assert response.status_code == 404
    assert response.json()['error']['code'] == 'JOB_NOT_FOUND'
```

- [x] Run: `.venv/Scripts/python -m pytest apps/api/tests/test_jobs.py -q`，确认失败。
- [x] 持久化载荷/版本/进度/结果，有界worker执行回测且逐步检查cancel；重启遗留running明确failed(interrupted)，queued恢复；终态CAS防止取消覆盖成功或反向覆盖。导出稳定排序OpenAPI，openapi-typescript生成前端类型，--check检测漂移。

```python
transitions = {'queued': {'running', 'cancelled'}, 'running': {'succeeded', 'failed', 'cancelled'}, 'succeeded': set(), 'failed': set(), 'cancelled': set()}
if target_state not in transitions[current_state]:
    raise DomainError('INVALID_JOB_TRANSITION', '任务状态不能这样切换', {})
```
- [x] 测试任务进度、取消/成功竞态、失败trace、重启恢复与同键不重复创建；前端停止轮询不等于取消后台，取消按钮独立发送请求。
- [x] 提交：`feat: add durable research jobs and generated API types`。

## Task 11：数据、行情、选股工作台

**Files:** `features/data/{DataImport,ImportPreview,DatasetPicker}.svelte`、`features/market/{SecuritySearch,PriceChart}.svelte`、`features/screening/{UniversePicker,FilterBuilder,Funnel}.svelte`；对应routes；`lib/components/{AppShell,DataTable,StatusPanel}.svelte`、`lib/styles/tokens.css`；`features/data/DataImport.test.ts`、`features/screening/Funnel.test.ts`。

**Interfaces:** props使用Task10生成类型；全局研究上下文只有datasetId/asOf，切换后清除旧响应；每页通过统一request调用API。数字显示来源/时间/null，合成数据持续可见。

- [x] 写失败组件测试，测试中用fetch mock明确返回带row=3错误的预检结果：

```typescript
it('does not commit an invalid preview', async () => {
  render(ImportPreview, { preview: { id:'p1', rows:[], errors:[{row:3,code:'INVALID_OHLC',message:'最高价小于开盘价'}] } });
  expect(screen.getByText(/第 3 行/)).toBeVisible();
  expect(screen.getByRole('button',{name:'提交导入'})).toBeDisabled();
});
```

- [x] Run: `npm --prefix apps/client run test -- --run`，确认缺组件/行为失败。
- [x] 深色金融工作台、左导航、顶部环境/时间/搜索；预检→确认导入、行情K线、候选池点选、DSL编辑、漏斗归因与回退。空态/加载/失败/离线有可见状态，错误保留trace；键盘导航、焦点、reduced-motion和A股红涨绿跌可配置。

```svelte
{#each preview.errors as error}
  <p role="alert">第 {error.row} 行：{error.message}</p>
{/each}
<button disabled={preview.errors.length > 0} onclick={onCommit}>提交导入</button>
```
- [x] 测试数据切换丢弃旧请求、搜索键盘、筛选回退、导入失败不显示成功、null呈现；check/lint/test/build通过，图表按视窗降采样但导出保留原数据。
- [x] 提交：`feat: connect market and screening workbench`。

## Task 12：组合、回测、风险与模拟页面

**Files:** `features/portfolio/{WeightEditor,OptimizationForm,RebalanceTable}.svelte`；`features/research/{BacktestForm,JobProgress,PerformanceChart,RiskPanel}.svelte`；`features/paper/{AccountSelector,OrderForm,LedgerTable,HaltButton}.svelte`；对应routes和总览；组件测试`WeightEditor.test.ts`、`JobProgress.test.ts`、`OrderForm.test.ts`、`HaltButton.test.ts`。

**Interfaces:** WeightEditor只输出编辑值，后端负责验证/计算；BacktestForm生成Task7 request，JobProgress消费Task10；OrderForm首次提交生成key、超时重试沿用同key、改内容用新key。

- [x] 写失败组件测试，fetch mock返回paused状态：

```typescript
it('halts without delaying confirmation', async () => {
  render(HaltButton, { accountId:'a1', paused:false });
  await userEvent.click(screen.getByRole('button',{name:'紧急停止'}));
  expect(await screen.findByText('已暂停新订单')).toBeVisible();
  expect(screen.queryByRole('dialog')).toBeNull();
});
```

- [x] Run: `npm --prefix apps/client run test -- --run`，确认失败。
- [x] 接Task5/8/9/10真实API：权重/约束/现金与费用、回测曲线/成交/假设、风险敞口证据、模拟可用/可卖/应收/账本/暂停与恢复。页面不算收益费用，重载从持久化恢复；恢复才确认短语。总览仅展示真实已有结果。

```typescript
async function haltAccount(accountId: string) {
  return request(`/v1/paper/accounts/${encodeURIComponent(accountId)}/halt`, { method: 'POST' });
}
```
- [x] 测试不可行解释、null指标、取消、409不静默新建单、未知状态可见、暂停/恢复、刷新持久化；check/lint/test/build通过。
- [x] 提交：`feat: connect portfolio research risk and paper views`。

## Task 13：证据报告、导出与可选AI

**Files:** `domain/research/reports.py`、`application/reports.py`、`infrastructure/ai/{compatible,templates}.py`、`api/routes/{reports,ai}.py`；`features/reports/ReportView.svelte`、`features/ai/Copilot.svelte`、settings页面；`tests/{test_reports,test_ai_gateway}.py`、`features/ai/Copilot.test.ts`。

**Interfaces:** `build_report(metrics:PerformanceMetrics,evidence:list[Evidence],as_of:datetime)->ResearchReport`；Evidence字段id/source/data_version/as_of/formula_version；ResearchReport含kind/model/sections，每section含evidence_ids；`AIProvider.stream(request:ExplanationRequest)->AsyncIterator[AIEvent]`；`AIEvent(run_id,seq,type,version,payload)`；`render_metric_tokens(text:str,metrics:dict[str,MetricValue])->str`。

- [x] 写失败token测试：

```python
def test_unknown_metric_token_cannot_publish():
    with pytest.raises(DomainError) as error:
        render_metric_tokens('收益 {{metric:unknown}}', {})
    assert error.value.code == 'UNKNOWN_METRIC_REFERENCE'
```

- [x] Run: `.venv/Scripts/python -m pytest apps/api/tests/test_reports.py apps/api/tests/test_ai_gateway.py -q`，确认失败。
- [x] 规则报告明确kind=rule_based、model=null，含版本/成本/指标/假设/缺口与证据，导出Markdown/CSV。可选LLM配置只在后端，未配置明确状态；只解释白名单工具结果/生成验证DSL，无交易工具。数值使用允许的指标token确定性回填，无来源数值或未知引用拒绝该段。POST fetch SSE记录seq/终态/预算/取消，断线查询run不重新付费。M0不假装已实现PDF/Word、全文RAG或自由SQL。

```python
def render_metric_tokens(text: str, metrics: dict[str, MetricValue]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in metrics:
            raise DomainError('UNKNOWN_METRIC_REFERENCE', '指标引用不存在', {'key': key})
        metric = metrics[key]
        return f'{metric.value} {metric.unit}' if metric.value is not None else '数据缺失'
    return re.sub(r'\{\{metric:([a-zA-Z0-9_]+)\}\}', replace, text)
```
- [x] 测试无密钥/provider超时/非法DSL/伪数值/未知引用/取消/断线与规则报告完整性；检查日志、响应、前端包无密钥。UI无伪AI回复。
- [x] 提交：`feat: add evidence reports and optional AI explanations`。

## Task 14：桌面打包、端到端验收与项目输出

**Files:** `apps/client/src-tauri/{Cargo.toml,tauri.conf.json,src/lib.rs,capabilities/default.json}`；`scripts/{build_sidecar.py,check.ps1,build.ps1}`；`.github/workflows/{ci.yml,windows-build.yml}`；`apps/client/e2e/research-flow.spec.ts`；`tests/test_packaged_server.py`；README、`docs/acceptance.md`、FR追踪。

**Interfaces:** Rust命令`backend_connection()->{url,token}`；sidecar监听127.0.0.1动态端口、stdout版本化握手、令牌仅内存，健康探测成功才展示主页面；退出只结束自己创建的子进程。浏览器产物由打包Python同时服务静态SPA；测试从临时数据目录启动真实后端。

- [x] 写失败端到端测试：

```typescript
test('synthetic data survives reload', async ({page}) => {
  await page.goto('/data');
  await page.getByRole('button',{name:'生成演示数据'}).click();
  await expect(page.getByText('合成数据')).toBeVisible();
  await page.reload();
  await expect(page.getByText('合成数据')).toBeVisible();
});
```

- [x] Run: `.venv/Scripts/python -m pytest apps/api/tests/test_packaged_server.py -q`、`npm --prefix apps/client run test:e2e`，确认缺运行/打包时失败。
- [x] 实现工具链检测、Python打包、Tauri握手/健康/清理、最小capability/CSP，不给WebView全局shell或任意路径/导航。WindowsCI安装明确工具链并实际构建安装包，未通过则记录确切阻塞。README写启动/样本/备份/测试/打包/限制，排除真实数据、密钥、环境与缓存。

```python
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('127.0.0.1', 0))
port = server_socket.getsockname()[1]
handshake = {'version': 1, 'url': f'http://127.0.0.1:{port}'}
print(json.dumps(handshake), flush=True)
```

握手不打印令牌；父进程通过仅该子进程可见的启动环境传入令牌。进程启动后Rust使用该令牌健康探测，健康就绪才返回连接；socket交给uvicorn而不重新绑定端口。
- [x] 完整E2E覆盖导入→筛选→建组合→回测→模拟推进→报告→重载，包含任务取消与错误状态。200标的×3年样本记录机器/耗时；运行下面全部检查。没有Windows构建证据不标桌面包通过。
- [x] 更新acceptance与FR为完整/部分/未配置/未实现及证据；独立全阶段评审后提交`feat: package and verify the local research client`。用户尚未定仓库，本阶段只输出可上传本地产物。

## 验收命令与执行顺序

Task1在项目环境安装依赖后提供以下命令；后续按1→14顺序，Task5整股费用在Task6后验证。失败先定位根因再改代码，新增变更/失败/不确定性才扩大检查。

```powershell
.venv/Scripts/python -m pytest apps/api/tests -q
.venv/Scripts/python -m ruff check apps/api
.venv/Scripts/python -m mypy apps/api/src
npm --prefix apps/client run check
npm --prefix apps/client run lint
npm --prefix apps/client run test -- --run
npm --prefix apps/client run build
npm --prefix apps/client run test:e2e
.venv/Scripts/python scripts/export_openapi.py --check
```

Windows原生包另外运行`scripts/build.ps1`。每任务记录红→绿与提交自己的文件，不暂存无关内容。关键领域测试用独立手算预期，避免测试复写实现公式。

## 自检、范围与推荐执行方式

M0设计10行均已覆盖：数据2/3、搜索3/11、选股4/11、组合5/6/12、风险8/12、回测6/7/10/12、模拟6/7/9/12、报告13、可选AI13、客户端1/11/12/14。五类Review Focus分别有明确任务和测试。M1–M3保留在124条追踪表，后续各自形成实现计划与专项连接/实验验收，不写成首版假实现。

推荐Native：由我在本会话逐项实现并测试，最后安排独立全阶段评审；领域与接口连续依赖较强，适合保持实现上下文。另一个可选方式是逐任务子代理实现与审查，独立复核更频繁但成本更高。

用户确认计划和执行方式后，不逐任务再询问是否继续；仅新的范围决策、外部凭证或实际权限阻塞请求输入。
