# 可选 AI 接入与验证边界

本地研究、回测、模拟盘与规则报告无需模型服务。AI 只有在用户点击“发起 AI 解释”或“生成选股条件”后调用，当前开发环境未配置真实密钥，真实服务商接入状态为**未配置 / 未进行真实模型验收**。测试使用受控 provider 和 HTTP transport，不产生模型费用。

启动本地后端前，在同一个 PowerShell 会话设置 `PORTFOLIO_AI_BASE_URL`、`PORTFOLIO_AI_MODEL`、`PORTFOLIO_AI_API_KEY`。基础地址包含 `/v1`，必须为 HTTPS 或本机 HTTP，禁止 URL 凭证、查询参数与重定向。`.env.example` 仅列字段，程序不会自动加载 `.env`。密钥只在后端进程持有，前端设置页仅返回是否配置和模型名称。

适配器使用兼容 Chat Completions 的 SSE 协议，发送 `max_completion_tokens` 与 `stream_options.include_usage`。请求不自动重试，也不在参数不兼容时切换协议或模型。输出参数包含可见和不可见 token；中断可能收不到最终用量，因此界面只展示服务商实际报告的用量，费用以服务商账单为准。参考：[创建 Chat Completion](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)、[流式事件与中断用量](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events)。

每次最多 2048 个输出 token、60 秒总时限、4 个活动请求；输入工具上下文最多 40000 字符，模型文本最多 24000 字符，网络响应在缓冲前限制为 200000 字节。最多发布 10 个完整段落，重复用量更新合并为一次持久化事件。

模型只能读取已保存的报告证据，或当前数据版本、历史时点、候选池与选股字段白名单。它没有交易工具、任意 SQL、网络检索或文件访问。自然语言选股只生成通过后端 `ScreeningSpec` 校验的条件，数据版本、时点与候选池由服务器固定；用户显式应用后才进入选股编辑器。

模型以 JSONL 逐段返回内容。原始 token 不直接展示；每个完整段落必须通过结构、引用和数值检查，才写入公开事件。数字仅允许引用 `{{metric:total_return}}` 等后端指标 token，再由确定性格式化回填；未知指标、未知证据、非法 DSL 或无来源数值会拒绝该段并标记运行失败，已有有效段落仍保留。数字守卫有意保守，包含中文数词、Unicode 数字和“翻倍 / 减半”等常见定量表述，因此可能拒绝普通含数词的措辞。它检查引用与结构，不等于对模型的所有语义判断作事实保证。

AI 事件包含 `run_id / seq / type / version`，与运行记录一同存入本机 SQLite。POST fetch SSE 支持从序号继续读取；断开界面连接不取消模型，显式取消接口才关闭模型连接。界面“继续查询原运行”只查询或恢复原事件流，不重新调用模型。相同幂等键和请求复用原运行，不同内容返回 409。重启遗留活动运行标记失败并保留已有事件，不自动恢复付费推理。

规则报告标记 `kind=rule_based, model=null`，直接从持久化研究结果重建，带 `rule-report-v1` 版本、数据哈希、日历、引擎、费率、知识口径、指标、成本、假设与缺口。支持 Markdown、指标 CSV、完整权益序列和成交 CSV；行情页的原始日线 CSV 在后端生成，保留高精度价格与 64 位成交量，不使用图表采样或浏览器浮点转换。M0 不包含 PDF / Word、RAG、任意 SQL 或实盘能力。
