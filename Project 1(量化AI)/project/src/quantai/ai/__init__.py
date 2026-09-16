"""AI 层：LLM 调用与提示词组装。

本包对外暴露两个核心能力：
- client.gen_report：研报生成主入口（AI 优先，失败自动降级为模板）。
- prompts.build_prompt：把指标 DataFrame 组装成结构化提示词。

子模块职责划分：
- client.py  —— 多供应商 LLM 调用、失败降级、模板兜底报告、技术面评级打分。
- prompts.py —— 指标快照抽取（snapshot_facts）与 6 段式提示词拼装（build_prompt）。
"""
