#!/usr/bin/env python3
"""Memory Node Manager — automatic summarization, embedding, and hybrid retrieval
of conversation turns as structured memory nodes.

Lifecycle (enhanced with HindSight-inspired features):
  1. After each completed conversation turn (queued in the background):
     - Extract HindSight-style narrative facts via LLM API
     - Extract keywords
     - Generate embedding via EmbeddingClient
     - Store each fact as a memory node in SessionDB (SQLite + FAISS)
     - Build temporal + semantic relation graph edges to prior nodes
     - Extract entities and relations -> knowledge graph
     - Store in memory_fact_relations + entity_nodes/edges

  2. Before each new turn:
     - Embed the user's query
     - Search for relevant memory nodes (keyword + vector + entity graph + node relations)
     - Return formatted context for system prompt injection

Usage::

    from agent.memory_node_manager import MemoryNodeManager

    mgr = MemoryNodeManager(
        session_db,
        embedding_config=None,
        memory_config=None,
    )
    mgr.store_turn("用户问了什么", "助手回答了什么")
    context = mgr.recall("用户当前问题")
    reflection = mgr.reflect()
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import re
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import requests

from agent.entity_extractor import ENTITY_EXTRACTION_GUIDANCE, is_attribute_entity
from agent.temporal_entities import is_temporal_entity

logger = logging.getLogger(__name__)
MEMORY_REFLECT_META_KEY = "memory_node_last_successful_reflect_at"
MEMORY_DECAY_META_KEY = "memory_node_last_successful_decay_at"

# ── Default LLM API endpoint ──────────────────────────────────────────────

DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

# ── Memory graph relation defaults ────────────────────────────────────────
SEMANTIC_RELATION_TYPE = "semantic"
TEMPORAL_RELATION_TYPE = "temporal"
CAUSAL_RELATION_GRAPH_TYPE = "causal"
SEMANTIC_RELATION_THRESHOLD = 0.82

INTERPRETATION_MIN_OBSERVATIONS_FOR_BATCH = 3
INTERPRETATION_MIN_CLUSTER_SIZE = 2
INTERPRETATION_MAX_LLM_CALLS_PER_REFLECT = 30
INTERPRETATION_SINGLE_OBSERVATION_CONFIDENCE_CAP = 0.75

MIN_FACTS_FOR_NEW_EVIDENCE_BUNDLE = 1
OBSERVATION_EMBEDDING_SIMILARITY_THRESHOLD = 0.72
OBSERVATION_EXACT_TYPE_SIMILARITY_THRESHOLD = 0.62
OBSERVATION_COMPATIBLE_TYPE_SIMILARITY_THRESHOLD = 0.72
OBSERVATION_MIN_FACTS_FOR_NEW_CLUSTER = 2
OBSERVATION_CLUSTER_MIN_SEMANTIC_COHESION = 0.5
OBSERVATION_CLUSTER_CENTROID_WEIGHT = 0.55
OBSERVATION_CLUSTER_MAX_SOURCE_WEIGHT = 0.25
OBSERVATION_CLUSTER_COVERAGE_WEIGHT = 0.20
OBSERVATION_TYPE_COMPATIBILITY_GROUPS = {
    "task": {
        "task_state",
        "task_progress",
        "decision",
        "problem",
    },
    "knowledge": {
        "context",
        "strategy",
    },
}
OBSERVATION_TYPE_PRIORITY = {
    "constraint": 90,
    "preference_signal": 80,
    "decision": 70,
    "problem": 60,
    "task_progress": 50,
    "task_state": 40,
    "strategy": 30,
    "behavior_pattern": 20,
    "context": 10,
}
EVIDENCE_BUNDLE_TOPIC_SIMILARITY_THRESHOLD = 0.82
EVIDENCE_BUNDLE_WEAK_TOPIC_SUFFIXES = (
    "建议",
    "方案",
    "方法",
    "策略",
    "状态",
    "情况",
    "进展",
)
EVIDENCE_BUNDLE_GENERIC_TOPICS = {
    "建议",
    "方案",
    "方法",
    "策略",
    "状态",
    "情况",
    "进展",
    "健康",
    "工作",
    "生活",
    "项目",
    "系统",
    "功能",
    "问题",
    "管理",
}
OBSERVATION_INTERPRETATION_TYPES = {
    "task_state": {"task", "project_state", "constraint"},
    "task_progress": {"task", "project_state"},
    "decision": {"project_state", "strategy", "insight"},
    "preference_signal": {"explicit_preference", "inferred_preference"},
    "constraint": {"constraint", "explicit_instruction", "task_risk"},
    "problem": {"task_risk", "project_state"},
    "strategy": {"strategy", "insight"},
    "behavior_pattern": {"behavior_pattern", "inferred_preference"},
    "context": {"insight"},
}
OBSERVATION_TYPE_GUIDANCE = {
    "task_state": (
        "任务状态：归纳主体当前想完成、被要求完成或仍待处理的目标。"
        "融合目标、范围、约束和未解决项；不要把已经完成的动作继续写成待办。"
    ),
    "task_progress": (
        "任务进展：归纳围绕同一任务已经发生的动作、里程碑和状态变化。"
        "突出时间顺序与最新进度，必要时用“从 A 变为 B”保留关键转折。"
    ),
    "decision": (
        "决策：归纳主体已经明确选择、确认、放弃或否决的方案。"
        "保留明确出现的决策对象、结果及理由；新决策覆盖旧决策时以最新结论为准。"
    ),
    "preference_signal": (
        "偏好信号：归纳主体明确表达的喜欢、不喜欢、接受、拒绝或取舍倾向。"
        "保留偏好对象、方向、适用条件和明确理由，不得把一次行为推断为稳定偏好。"
    ),
    "constraint": (
        "约束：归纳必须遵守、必须避免或客观限制可选空间的边界条件。"
        "保留约束对象、适用范围、触发条件及来源，不要改写成一般建议。"
    ),
    "problem": (
        "问题：归纳已出现的故障、困难、冲突、风险或负面状态。"
        "保留问题表现、影响、明确原因和当前处理状态，不得猜测未被证据支持的原因。"
    ),
    "strategy": (
        "策略：归纳为实现同一目标提出或采用的方法、原则和步骤。"
        "合并重复建议，组织成连贯方案；保留互补方法和适用条件，避免逐条堆砌建议。"
    ),
    "behavior_pattern": (
        "行为模式：归纳多个事件共同支持的重复行为、习惯或稳定趋势。"
        "说明反复发生的行为及典型情境；单次事件不足以形成行为模式。"
    ),
    "context": (
        "背景语境：归纳理解主体或主题所需的稳定事实、处境和关系。"
        "保留有解释力的背景及当前情况，不要混入待办、建议或高层人格推断。"
    ),
}
# DECAY
MEMORY_SEMANTIC_FACT_HALF_LIFE_DAYS = 365.0
MEMORY_EPISODIC_FACT_HALF_LIFE_DAYS = 90.0
MEMORY_OBSERVATION_HALF_LIFE_DAYS = 180.0
MEMORY_INTERPRETATION_HALF_LIFE_DAYS = 365.0


OBSERVATION_CREATE_PROMPT = """你是长期记忆系统的 observation 生成模块。observation 是 evidence_bundle 内由一组相似事实直接支持的、稳定且可独立演化的具体陈述。

三层记忆架构：
- fact：原始证据层，保存对话明确支持的事件、知识、决定、约束、错误、建议和结果。
- observation：从多个 facts 中归纳历史模式、进展或状态变化；不要在摘要中提前归纳。
- interpretation：对用户偏好、任务状态和行动策略的高层理解；不要生成无直接证据的推断。

当前 observation_type 的含义：
{observation_type_definition}

归纳步骤：
1. 找出所有 facts 共同支持的核心命题。
2. 判断其余 facts 是补充细节、限定条件、重复证据，还是对状态的更新。
3. 将这些信息融合为一个当前命题，而不是选择一条代表性 fact，也不是逐条拼接 facts。

约束：
1. 只能使用输入 facts 中明确出现的信息，不得补充推断。
2. summary 必须体现 requested_observation_type 的归纳目标，并覆盖所有与核心命题相关且不重复的证据。
3. 对重复信息进行合并；对互补信息建立清晰关系；对时间变化优先表达最新状态，必要时保留关键转折。
4. summary 必须自包含，明确主体、对象、动作或状态、必要条件与结果；通常使用 1 至 3 句连贯陈述。
5. 不要写“多条事实表明”“根据上述 facts”等证据处理过程，不要输出事实清单。
6. observation_type 必须保持为 requested_observation_type。
7. 不要生成 interpretation、额外行动建议或用户画像。
8. confidence 表示输入 facts 对该 summary 的直接支持程度，不表示内容的重要性。
9. 只返回一个合法 JSON object，不要使用 Markdown。

输出格式：
{{
  "observation_type": "{requested_observation_type}",
  "summary": "",
  "confidence": 0.0
}}

evidence_bundle:
{evidence_bundle_context}

facts:
{source_facts}
"""

OBSERVATION_UPDATE_PROMPT = """你是长期记忆系统的 observation 增量更新模块。请用新增 facts 更新既有命题，同时保持命题身份和历史语义稳定。

三层记忆架构：
- fact：原始证据层，保存对话明确支持的事件、知识、决定、约束、错误、建议和结果。
- observation：从多个 facts 中归纳历史模式、进展或状态变化；不要在摘要中提前归纳。
- interpretation：对用户偏好、任务状态和行动策略的高层理解；不要生成无直接证据的推断。

当前 observation_type 的含义：
{observation_type_definition}

更新步骤：
1. 把 existing_observation 视为历史 facts 已形成的归纳结果，把 new_facts 视为新增证据。
2. 判断新增证据是在确认、补充、限定、推进、纠正还是推翻既有命题。
3. 重新写出一条融合后的当前命题；禁止把 new_facts 机械追加到旧 summary 末尾。

约束：
1. 只能使用 existing_observation 与 new_facts 中明确出现的信息。
2. 保留仍被支持且对当前命题有用的历史信息，删除重复、过时或已被明确取代的表述。
3. 对 task_state、task_progress、decision 和 problem，优先表达最新状态；仅在关键转折有助于理解时保留历史状态。
4. 对 preference_signal、constraint、strategy、behavior_pattern 和 context，融合新增条件与证据，不得仅因一次事件扩大为稳定结论。
5. 除非新增事实明确纠正或推翻旧内容，否则不要改变命题主题。
6. summary 必须体现 observation_type 的归纳目标，自包含且通常使用 1 至 3 句连贯陈述，不要输出事实清单。
7. observation_type 必须保持不变。
8. change_summary 只简述本次语义变化，例如“补充适用条件”或“状态由待处理更新为已完成”；若只是重复确认则写“新增证据确认既有命题”。
9. 不要生成 interpretation、额外行动建议或用户画像。
10. confidence 表示全部现有证据对更新后 summary 的直接支持程度。
11. 只返回一个合法 JSON object，不要使用 Markdown。

输出格式：
{{
  "observation_type": "{observation_type}",
  "summary": "",
  "confidence": 0.0,
  "change_summary": ""
}}

existing_observation:
{existing_observation}

new_facts:
{new_facts}
"""

# ── Shared causal relation guidance ───────────────────────────────────────

CAUSAL_RELATION_TYPES = ("Cause", "Want", "React", "Changed", "SameTopic", "None")
CAUSAL_RELATION_TYPE_TEXT = "/".join(CAUSAL_RELATION_TYPES)

CAUSAL_RELATION_GUIDANCE = """【causal_relation 关系类型定义（必须共用）】

Step 1：先判断是否存在明确关联

仅当满足以下任一条件，才认为"有关联"：
- 存在明确因果触发（如：因为A所以B）
- 存在明确行为/请求延续（如：A后提出需求B）
- 存在明确情绪或意愿变化

否则：
→ relation 必须为 None

若不确定，一律选择 None，禁止猜测。

Step 2：若有关联，再按优先级选择一个关系类型：

1. Cause（因果）最高优先级
   条件：
   - A直接导致B发生
   - 常见模式：
     偏好 -> 行为
     行为 -> 请求
   示例：
     "喜欢英超" -> "请求推送英超新闻"

2. Want（意愿）
   条件：
   - A引发B中的需求/请求
   - 关键词：
     想 / 希望 / 帮我 / 推送 / 能不能

3. React（反应）
   条件：
   - A引发情绪/态度变化
   - 如：开心 / 失望 / 觉得好

4. Changed（变化）
   条件：
   - B明确改变A中的状态/偏好

5. SameTopic（同主题）
   条件：
   - 仅主题相同，无因果/意图关系

6. None
   条件：
   - 无明确证据、不确定、仅语义相似、需要脑补中间步骤

严格规则：
1. 默认输出 None，除非有明确证据
2. 禁止基于"语义相似"判断因果
3. 禁止跨步推理（不能脑补中间步骤）
4. 后发生的事实不可能导致先发生的事实
5. 优先识别：偏好 -> 请求
"""

# ── Summarisation prompt template ─────────────────────────────────────────

SUMMARY_SYSTEM_PROMPT = """你是长期记忆系统的对话摘要模块。请根据以下对话生成一条可追溯、自包含、可独立召回的证据性摘要，并提取用于检索的关键词和实体。不要把信息丰富的内容压缩成空泛的主题摘要。

三层记忆架构：
- fact：原始证据层，保存对话明确支持的事件、知识、决定、约束、错误、建议和结果。
- observation：从多个 facts 中归纳历史模式、进展或状态变化；不要在摘要中提前归纳。
- interpretation：对用户偏好、任务状态和行动策略的高层理解；不要生成无直接证据的推断。

摘要要求：
1. summary 输出一条完整摘要，可以使用复句，不要为了追求短而删除关键事实
2. summary 必须脱离原始对话后仍能独立理解，并优先保留：
   - 具体对象：项目、模块、文件、函数、配置项、参数、产品、人物或材料
   - 具体动作或论点：提出、决定、修改、删除、实现、排查、验证、解释或反驳了什么
   - 关键条件和约束：阈值、触发时机、适用范围、禁止事项、依赖和前置条件
   - 原因、证据和目的：为什么采取该方案，什么现象支持该判断
   - 结果和状态变化：成功、失败、通过、阻塞、待处理，以及修改前后的差异
3. 对知识讲解、方案讨论和技术分析，不仅记录“谁讨论了什么”，还要保留其中明确出现的核心概念、方案、取舍、结论和适用条件
4. 如果对话包含多个相互关联的高价值信息点，应在同一条 summary 中使用清晰复句完整表达；不要只保留上位主题
5. 禁止只写“讨论了某主题”“要求优化相关代码”“助手提供了建议”“处理了某个问题”这类缺少实质内容的描述
6. 只基于输入对话总结。不要把“建议修改”“计划测试”“正在排查”写成“已经修改”“测试通过”或“问题已解决”，除非对话明确提供了完成证据
7. 不要保留问候、寒暄、客套话、助手复述问题、无复用价值的流水账，或已被更具体信息覆盖的重复内容
8. 提取 2-8 个关键词，优先保留关键实体、产品、技术、动作、错误、结果和约束；输出为 JSON string 数组
9. 提取对召回有用的实体，遵守实体抽取规则；普通时间表达不要作为实体
10. 最终只返回一个合法 JSON object，不要包含 Markdown、代码块或其他说明

详细程度示例：
- 不合格："用户要求优化记忆提取逻辑。"
- 合格："用户要求将 MemoryNodeManager.store_turn 从前 N 轮跳过提取改为每累计 N 轮批量提取一次，并保留未达到阈值的历史对话，避免只处理最新一轮造成信息丢失。"
- 不合格："助手提供了测试建议。"
- 合格："助手建议为批量 fact 提取增加达到轮数阈值和提取失败后保留 pending turns 的测试，但对话没有表明测试已经执行。"

输出前进行信息保真自检：
- summary 是否包含足以区别于同主题其他对话的具体对象、动作、约束或结论？
- 删除函数名、参数、错误现象、关键条件或结果后是否会改变摘要含义？如果会，必须保留。
- 是否退化成主题名称、对话行为或没有实质内容的概述？
- 是否写入了输入对话没有明确支持的意图、原因、完成状态或结果？

""" + ENTITY_EXTRACTION_GUIDANCE + """

输出格式：
{{
    "summary": "对话的核心内容概括", 
    "keywords": ["关键词1", "关键词2"], 
    "entities": [{{"name": "实体名", "type": "CONCEPT"}}]}}

对话批次（按时间顺序）：
{dialogue_batch}"""

# ── Recall query analysis prompt template ────────────────────────────────

RECALL_QUERY_ANALYSIS_PROMPT = """你是长期记忆 recall 查询分析器。请分析用户当前查询，生成用于三层记忆检索的结构化策略。

三层记忆含义：
- interpretations: 从 observation 推导出的洞察、任务状态、偏好、策略或风险，适合回答"我应该怎么做/用户偏好是什么/当前任务状态是什么"
- observations: 由多个 fact 汇总出的稳定模式、阶段性状态、事件簇或变化，适合回答"最近/通常/整体有什么趋势或状态"
- facts: 原始事实记忆，包含 semantic 与 episodic 两类，适合回答"之前具体说过什么/什么时候/证据是什么"

recall_intent 只能是：
- action: 用户需要行动建议、任务推进、偏好约束、下一步策略
- evidence: 用户需要历史证据、原始事实、具体时间地点人物、之前是否说过
- state: 用户需要状态、趋势、长期模式、最近变化或总结
- balanced: 意图不明显，需要均衡召回

fact_type_preference 只能是：
- semantic: 更需要事实、概念、背景、长期偏好或长期规则
- episodic: 更需要具体经历、事件、时间线、上下文和证据
- both: 两者都需要或无法判断

needs_recall 判断标准：
- true: 回答依赖用户或助手过去的经历、长期偏好、持续任务、历史状态、时间线或证据；或者长期记忆可能实质性改变回答内容
- false: 当前输入已经包含完整信息，且问题只需要通用知识、当前输入中的材料、简单问候确认或与历史无关的临时操作
- 仅依赖当前会话上下文不等同于需要长期记忆；不要因为长期记忆可能提供轻微帮助就返回 true

要求：
1. needs_recall 表示是否需要查询长期记忆；recall_confidence 是该判断的置信度，0-1；recall_reason 用简短枚举式文本说明原因。
2. 即使 needs_recall 为 false，也要返回完整 JSON，便于日志和降级处理。
3. search_text 是用于 embedding 的检索表达，必须保留原始查询中的关键实体、事件、约束和意图；不要过度抽象。
4. keywords 提取 2-8 个检索关键词，保留关键实体、产品、技术、动作、约束和主题词。
5. entities 提取对召回有用的实体，遵守实体抽取规则；普通时间表达不要作为实体。
6. layer_preference 是 interpretations/observations/facts 三层的偏好权重，数值在 0-1 之间，总和尽量接近 1。
7. intent_confidence 是你对 recall_intent 判断的置信度，0-1。
8. needs_evidence 表示回答是否需要展开 observation/interpretation 背后的事实证据。
9. time_sensitivity 只能是 specific、recent、long_term、none。
10. 仅返回 JSON，不要包含其他内容。

""" + ENTITY_EXTRACTION_GUIDANCE + """

输出格式：
{{
    "needs_recall": true,
    "recall_confidence": 0.0,
    "recall_reason": "historical_context_required",
    "search_text": "用于 embedding 的检索表达", 
    "keywords": ["关键词1", "关键词2"], 
    "entities": [{{"name": "实体名", "type": "CONCEPT"}}], 
    "recall_intent": "balanced", 
    "intent_confidence": 0.0, 
    "layer_preference": {{"interpretations": 0.34, "observations": 0.33, "facts": 0.33}}, 
    "fact_type_preference": "both", 
    "interpretation_type_preference": ["insight"], 
    "needs_evidence": false, 
    "time_sensitivity": "none"}}

用户查询：
{query}"""

# ── HindSight-style retain prompt template ────────────────────────────────

RETAIN_FACT_EXTRACTION_PROMPT = """你是长期记忆系统的 fact 提取模块。请从下面一批按时间顺序排列的连续对话中提取 0-8 条可追溯、自包含、可独立召回的 facts。不要为了覆盖每一轮而强行生成 fact，也不要把信息丰富的内容压缩成空泛的主题摘要。

三层记忆架构：
- fact：原始证据层，保存对话明确支持的事件、知识、决定、约束、错误、建议和结果，回答“具体发生了什么或具体说了什么”。
- observation：从多个 facts 中归纳历史模式、进展或状态变化；不要在 fact 层提前归纳。
- interpretation：对用户偏好、任务状态和行动策略的高层理解；不要在 fact 层生成无直接证据的推断。

提取粒度和信息保真要求：
1. 一个 fact 只表达一个可独立召回的事件、结论、约束、建议或知识点，但同一件事的主体、动作、对象、关键条件、原因和结果应尽量保存在同一条 fact 中，不要拆成失去上下文的句子碎片
2. 同一批对话中不同的决定、错误、方案、知识结论或任务进展应拆成多个 facts；信息密集时宁可输出多条完整事实，不要合并成一个主题标签
3. fact 的 text 必须脱离原始对话后仍能独立理解，并优先保留：
   - 具体对象：项目、模块、文件、函数、配置项、参数、产品、人物或材料
   - 具体动作或论点：提出、决定、修改、删除、实现、排查、验证、解释或反驳了什么
   - 关键条件和约束：阈值、触发时机、适用范围、禁止事项、依赖和前置条件
   - 原因、证据和目的：为什么采取该方案，什么现象支持该判断
   - 结果和状态变化：成功、失败、通过、阻塞、待处理，以及修改前后的差异
4. 禁止只写“讨论了某主题”“要求优化相关代码”“助手提供了建议”“处理了某个问题”这类缺少实质内容的描述
5. 对知识讲解、方案讨论和技术分析，不仅记录“谁讨论了什么”，还要提取其中明确出现的核心概念、方案、取舍、结论和适用条件
6. 区分事件背景和实质内容：用户提出请求可以是一条 episodic fact；对话确认的稳定技术结论可以单独成为 semantic fact；助手实际完成的修改、测试及其结果可以成为 episodic fact
7. 只基于输入对话提取。不要把“建议修改”“计划测试”“正在排查”写成“已经修改”“测试通过”或“问题已解决”，除非对话明确提供了完成证据
8. 尽量保留用户偏好、约束、决定、失败经验、助手建议和明确原因
9. 区分 fact_type（心理学意义上的记忆性质）:
   - semantic: 语义记忆，关于事实、概念、常识、稳定背景、长期偏好或长期规则
   - episodic: 情景记忆，关于具体经历/事件，通常包含特定时间、地点、人物、行为、结果、情绪或状态变化
10. occurred_start/occurred_end 如果对话没有明确日期，填空字符串
11. time_confidence 只能是 explicit、inferred_from_turn、unknown：
   - explicit: 对话中明确给出日期/时间或可无歧义换算
   - inferred_from_turn: 只能基于当前这轮对话发生时间推断
   - unknown: 无法确定时间
12. entities 遵守下方统一实体提取规则；普通时间表达应写入 occurred_start/occurred_end，不进入 entities
13. keywords 是用于检索这条 fact 的关键词，保留关键实体、产品、技术、动作和约束
14. primary_entity 是这条 fact 主要描述的唯一主体，用于后续 observation 分桶：
   - 必须输出单个实体对象，并且该实体也必须出现在 entities 中
   - 优先选择 fact 的行为、状态、偏好、决定或经历所归属的主体
   - 仅被提及的对象、建议来源、地点、工具或上下文实体不能自动成为 primary_entity
   - 多人互动事件选择该 fact 主要描述或影响的主体
15. primary_topic 是这条 fact 唯一的核心主题，用于后续 observation 分桶：
   - 必须输出一个具体、稳定的主题字符串，不要输出数组
   - 不要把 entity name 本身当作 primary_topic
   - 同批次语义相同的 facts 应尽量使用完全一致的 primary_topic 表述
   - 不要在 primary_topic 中随意增删“关系、管理、状态、情况、问题”等后缀
16. fact_subject 只能是 user、assistant、world、project、system、other；表示这条记忆主要关于谁/什么主体
   - 如果 fact_subject 是 user 或 assistant，可以把 "用户" 或 "助手" 作为 OTHER entity 输出，便于后续按对话主体聚合
17. fact_kind 只能是 preference、decision、request、recommendation、action、error、context、instruction、other
   - instruction 只用于用户明确要求 AI 长期遵守的行为规则、格式偏好、语气偏好或工作方式
   - 临时任务要求、当前轮的一次性请求不要标为 instruction
18. priority 是 0-100 的整数，表示长期记忆价值：
   - 80-100: 长期偏好、硬约束、健康/安全/核心项目事实、明确长期指令、重要任务进展
   - 60-79: 可复用经验、一般任务事件、明确决策、失败原因
   - <60: 普通闲聊、一次性问答、无后续价值、重复弱信息；不要输出这条 fact
19. task_event_like 描述这条 fact 是否是一个可能影响任务状态或步骤的事件；它不要求已经知道具体属于哪个任务
20. task_event_subject 只能是 user、assistant、both、other；表示任务事件的主体或主要来源
21. task_relevance 只能是 none、weak、medium、strong：
   - none: 与任务状态或步骤无关
   - weak: 像一个事件，但不足以说明它会影响任务状态或步骤
   - medium: 可能影响某个任务的状态或步骤
   - strong: 明确表示用户正在发起、推进、完成、阻塞、暂停、恢复或决策某个任务
22. causal_relations 只描述本次输出 facts 之间明确存在的关系；source_index/target_index 使用 facts 数组的 0-based 下标
23. 只返回 JSON，不要 markdown，不要额外解释

fact_kind 定义和判别边界：
- preference：用户长期或反复表达的喜好、偏好、禁忌、习惯、倾向；不是一次性选择。
- decision：用户或项目已经明确做出的决定、取舍、采用方案；必须有"已决定/选择/采用/放弃"的证据。
- request：用户对 AI 或系统提出的当前任务请求，通常是"帮我/请你/能不能/现在去..."。
- instruction：用户要求 AI 以后长期遵守的行为规则、格式偏好、语气偏好或工作方式；必须具有长期性，如"以后/一直/记住/默认/每次"。
- recommendation：助手给出的具体建议、推荐方案、操作路径；必须是建议，不是普通解释。
- action：用户或助手已经执行、正在执行或计划执行的动作、实现、测试、排查、验证、修改。
- error：失败、报错、阻塞、误判、踩坑、不可用方案，以及明确的负面结果。
- context：长期有用的背景事实、项目状态、关系、约束、环境信息，但不属于 preference/decision/instruction/error。
- other：有一定保留价值但不属于以上类型；谨慎使用。

fact_type 判别边界：
- semantic：不依赖单次经历也能复用的稳定知识，例如项目结构、概念定义、用户长期偏好、长期指令、系统约定、常识。
- episodic：某次具体发生过的经历或事件，例如用户在某轮提出请求、助手执行修改/测试、一次失败/通过、某个时间点的决定/状态变化/情绪反应。
- 用户/助手并不等于 fact_type；用户长期偏好通常是 semantic + user，用户本轮请求通常是 episodic + user，助手某次执行通常是 episodic + assistant。

fact_kind 冲突和主体规则：
- 冲突时选择更具体的 kind，优先级为：instruction > preference > decision > error > action > request > recommendation > context > other。
- "帮我现在改代码" 属于 request；"以后回答都先给结论" 属于 instruction。
- "记住我喜欢简洁回答" 如果描述用户属性/偏好，属于 preference；如果要求 AI 以后如何回答，属于 instruction。
- 用户提出的当前任务需求通常是 request，不要标为 instruction。
- 助手执行了工具、测试、修改、验证，通常是 episodic + assistant + action。
- 助手提出具体方案，通常是 recommendation；助手解释概念但没有可复用建议，不要抽取，若必须抽取最多为 context/other。

硬丢弃规则：
- 不要抽取助手泛泛解释概念、复述用户问题、客套话、以及用户与助手进行的问候、寒暄
- 不要抽取没有未来复用价值的对话流水账
- 不要抽取纯主观情绪，除非它改变了用户偏好、决策或任务状态
- 不要抽取已被更高价值 fact 覆盖的重复信息
- 如果一条候选 fact 的 priority < 60，不要把它放进 facts 数组
- 不要因为信息很多就只保留上位主题；应删除低价值信息，而不是删除高价值事实中的关键细节

task_event_like 判断规则：
- true: fact 描述了一个可能影响任务生命周期的事件，包括请求、计划、推进、修改、实现、排查、验证、完成、结果、失败、阻塞、暂停、恢复或决策
- false: fact 只是偏好、背景、关系、属性、静态信息、一次性知识问答或泛泛主题讨论
- 单条 fact 不需要判断它属于哪个具体 task；只需要判断它是否可能用于更新某个 task 的状态或步骤
- 助手执行测试、修改代码、总结方案可以是 task_event_like，但如果只是助手行为，task_relevance 通常不要超过 medium
- 用户明确要求、计划、继续、完成、阻塞、暂停或恢复某个任务时，task_relevance 通常是 medium 或 strong

表达原则：
- 不要求固定句式；结构化字段已经保存时间、主体和类别，text 应优先承载具体内容
- preference/context 应写清具体偏好、稳定背景及其适用范围
- decision/request/action 应写清具体对象、采用或要求的方案、关键条件及结果
- instruction 应写清 AI 今后需要长期遵守的具体行为
- assistant episodic 只有在对话明确表明助手实际执行、建议或验证了具体内容时才提取，并保留执行对象和结果

详细程度示例：
- 不合格："用户要求优化记忆提取逻辑。"
- 合格："用户要求将 MemoryNodeManager.store_turn 从前 N 轮跳过提取改为每累计 N 轮批量提取一次，并保留未达到阈值的历史对话，避免只处理最新一轮造成信息丢失。"
- 不合格："双方讨论了 observation 和 interpretation 的匹配。"
- 合格拆分 1："用户认为 observation 与 interpretation 的匹配不应仅依赖关键词，建议把 embedding 相似度作为候选匹配指标。"
- 合格拆分 2："用户要求 observation 匹配已有 interpretation 后更新 interpretation 内容，而不是只建立证据关联。"
- 证据保守："助手建议增加测试"不能写成"助手已经增加测试并验证通过"；只有对话明确出现执行和测试结果时才能记录完成状态。

输出前进行信息保真自检：
- text 是否包含足以区别于同主题其他事实的具体对象、动作、约束或结论？
- 删除函数名、参数、错误现象、关键条件或结果后是否会改变事实含义？如果会，必须保留。
- 是否把多个可独立召回的决定、错误、方案或知识结论错误合并成了一条？
- 是否退化成主题名称、对话行为或没有实质内容的概述？
- 是否写入了输入对话没有明确支持的意图、原因、完成状态或结果？

""" + ENTITY_EXTRACTION_GUIDANCE + """

""" + CAUSAL_RELATION_GUIDANCE + """

输出格式：
{{
  "facts": [
    {{
      "text": "完整叙事事实",
      "keywords": ["关键词1", "关键词2"],
      "primary_entity": {{"name": "主要主体实体", "type": "PERSON"}},
      "primary_topic": "唯一核心主题",
      "fact_type": "semantic/episodic",
      "fact_subject": "user/assistant/world/project/system/other",
      "fact_kind": "preference/decision/request/recommendation/action/error/context/instruction/other",
      "priority": 80,
      "priority_reason": "为什么这条 fact 值得长期保留",
      "task_event_like": true,
      "task_event_subject": "user/assistant/both/other",
      "task_relevance": "none/weak/medium/strong",
      "occurred_start": "",
      "occurred_end": "",
      "time_confidence": "explicit/inferred_from_turn/unknown",
      "where": "",
      "entities": [
        {{"name": "实体名", "type": "CONCEPT"}}
      ]
    }}
  ],
  "causal_relations": [
    {{"source_index": 0, "target_index": 1, "relation": \"""" + CAUSAL_RELATION_TYPE_TEXT + """\", "confidence": 0.0}}
  ]
}}

对话批次（按时间顺序）：
{dialogue_batch}"""

# ── Causal relation extraction prompt template ────────────────────────────
# Adapted from AI_Glass_Agent relation_prompt.md

RELATION_PROMPT_TEMPLATE = """你是"AI眼镜记忆关系抽取模块"。

你的任务是：判断【摘要A】与【摘要B】之间的关系。

【基本设定】
- 【摘要A】发生在【摘要B】之前（严格时序，不可颠倒）
- 主体通常为"用户"，少数为"AI"
- 目标：服务于"用户偏好建模 + 主动推送"

--------------------------------------------------
""" + CAUSAL_RELATION_GUIDANCE + """

--------------------------------------------------

【输出格式（极其重要）】

{{
  "relation": \"""" + CAUSAL_RELATION_TYPE_TEXT + """\",
  "confidence": 0.0-1.0,
  "reason": ""
}}

--------------------------------------------------
【输出约束（必须遵守）】

1. 只能输出 JSON
2. 不允许输出任何额外文字
3. 不允许使用 markdown
4. 不允许添加解释
5. reason 必须是"基于摘要的直接证据"，一句话

--------------------------------------------------
【自检（输出前必须检查）】

在输出前，请确认：

- 是否严格是 JSON（无多余字符）
- relation 是否在候选类型中：""" + CAUSAL_RELATION_TYPE_TEXT + """
- 是否存在过度推断
- 若不确定 → 改为 None

--------------------------------------------------

【输入】

【摘要A】：
{summary1}

【摘要B】：
{summary2}"""

INTERPRETATION_GENERATION_PROMPT = """你是长期记忆 interpretation 生成模块。

你需要基于一条 observation 及其 supporting facts，判断是否值得生成一条 Agent 对当前世界状态的解释。

这里的 interpretation 不是用户原话，也不是原始事实；它是 Agent 基于记忆证据形成的 current best interpretation，用于后续召回时指导如何理解和行动。

四层记忆架构：
- fact：原始证据，表示对话中提取出的事实。
- evidence_bundle：按相同 entity/topic 组织 facts 的容器，只负责限定证据范围。
- observation：evidence_bundle 内由一组相似 facts 直接支持的具体命题，是 interpretation 匹配和生成的主要语义单元。
- interpretation：当前解释，表示 Agent 现在如何理解这些 observations，以及后续应该如何行动。

entity: {entity_name}
topic: {topic_label}
observation_type: {observation_type}
observation_summary: {observation_summary}
observation_metadata: {observation_metadata}

supporting facts:
{source_facts}

只在 observation 对未来行为有明确指导价值时生成 interpretation。适合生成的情况：
- 用户明确偏好、长期指令、稳定工作方式、决策倾向
- 项目当前状态、任务策略、风险、约束或冲突解决结论
- 多个事实共同支持的行为模式或当前解释
- observation 背后体现出可复用的 insight、task、策略、偏好、风险或状态判断

单条 observation 的证据门槛：
- task 可以由单条 observation 生成，但必须明确指向请求、目标、进展、阻塞、结果或下一步行动。
- explicit_preference/explicit_instruction 可以由单条 observation 生成，但必须来自用户明确表达的偏好或指令；单次行为暗示只能作为 inferred_preference，且证据不足时 should_create=false。
- insight、project_state、task_risk、strategy、behavior_pattern 默认需要多个 observation 或多个 supporting facts 支撑；如果只有单条 observation 且 evidence_shape=single_event，通常 should_create=false。
- 不要把一次性事件、普通上下文或孤立事实上升为稳定洞察、长期偏好或行为模式。
- 单条 observation 生成的 inferred 类型 confidence 不要超过 0.75。

不要生成 interpretation 的情况：
- observation 只是普通事实摘要，缺少未来行动含义
- observation 只描述一次性任务步骤，没有可复用解释
- 证据不足，只能靠猜测用户心理
- 只是复述 observation，没有形成新的 current interpretation

字段要求：
- should_create=false 时，只输出 {{"should_create": false}}。
- claim 是 Agent 当前解释，必须谨慎、可证据支持；不要写成用户原话。
- interpretation_type 只能是 insight、task、explicit_preference、explicit_instruction、inferred_preference、behavior_pattern、project_state、task_risk、constraint、conflict_resolution、strategy、other。
- observation_metadata 如果包含 observation_type、evidence_mode 和 allowed_interpretation_types，必须遵守其类型门禁；interpretation_type 必须来自 allowed_interpretation_types。
- fact_type 描述证据性质，observation_type 描述证据直接支持的命题，interpretation_type 描述 Agent 的高层理解；禁止跳过 observation_type 把 episodic 行为直接写成显式偏好。
- insight 表示从 observation 提炼出的当前可用洞察；task 表示 Agent 当前认为用户正在推进的任务或目标。
- 如果 interpretation_type=task，metadata 中填写 task_status、task_source、goal、evidence、steps；task_status 只能是 active、blocked、paused、stale，task_source 固定为 inferred_from_interpretation。
- 如果 interpretation_type 不是 task，不要填写 task_status、goal、steps、next_action。
- status 只能是 current、conflicted；新生成的 interpretation 不要输出 superseded 或 archived。
- conflict_status 只能是 none、resolved、unresolved。
- polarity 只能是 positive、negative、mixed、neutral。
- strength/confidence 是 0.0-1.0；inferred 类型如果证据少，confidence 不要超过 0.75。
- scope 是这条解释适用的范围，尽量短，如 "memory-system-design"。
- target_text 是解释指向的对象、方案、习惯、项目状态或风险。
- action_implication 描述这条解释未来如何影响 Agent 行为；如果没有明确行动含义，应 should_create=false。
- evidence_fact_ids 是直接支持该 interpretation 的底层 fact id，表示“为什么 Agent 相信这个解释”；只能使用 supporting facts 中出现的 id。
- evidence_observation_ids 是支持该 interpretation 的 observation id，表示“哪些中层归纳支撑这个解释”；只能使用输入 observation 的 id。
- counter_evidence_fact_ids 是反驳、削弱、限定或造成冲突的底层 fact id；只有存在明确反证、例外、边界条件或 unresolved conflict 时填写。
- counter_evidence_observation_ids 是反驳、削弱、限定或造成冲突的 observation id；只有存在明确反证、例外、边界条件或 unresolved conflict 时填写。
- 如果只是证据不足，不要把无关事实放入 counter_evidence_*；应降低 confidence 或 should_create=false。

只返回合法 JSON，不要 markdown，不要额外解释。格式如下：
{{
  "should_create": true,
  "claim": "Agent 当前解释为...",
  "target_text": "解释对象",
  "scope": "适用范围",
  "interpretation_type": "insight | task | explicit_preference | explicit_instruction | inferred_preference | behavior_pattern | project_state | task_risk | constraint | conflict_resolution | strategy | other",
  "polarity": "positive | negative | mixed | neutral",
  "strength": 0.0,
  "confidence": 0.0,
  "status": "current | conflicted",
  "conflict_status": "none | resolved | unresolved",
  "resolution": "可选，冲突如何被解决",
  "action_implication": "未来 Agent 应如何使用这个解释",
  "evidence_fact_ids": [1, 2],
  "evidence_observation_ids": [3],
  "counter_evidence_fact_ids": [],
  "counter_evidence_observation_ids": [],
  "metadata": {{"source": "interpretation_generation"}}
}}"""

INTERPRETATION_OBSERVATION_VALUE_PROMPT = """你是 observation 到已有 interpretation 的价值判断模块。

你需要独立判断一条新 observation 是否与候选 interpretation 相关，以及它是否值得改变已有 interpretation。
这一步同时负责匹配确认、信息价值判断和冲突识别，但不直接生成或改写 interpretation。

observation:
{observation}

supporting facts:
{source_facts}

candidate interpretations:
{candidate_interpretations}

决策定义：
- update：observation 带来进展、修正、范围变化、状态变化或冲突，已有 interpretation 的内容需要更新。
- evidence_only：observation 只是支持或确认已有 interpretation，只需增加证据，不需要改写内容。
- unmatched：与候选 interpretation 均不匹配，但可能与其他 observation 组合生成新的 interpretation。
- defer：当前独立价值不足，但未来或与其他 observation 组合后可能有价值。
- ignore：琐碎、重复、无行动意义，并且没有继续组合分析的价值。

relationship 定义：
- support：支持或确认已有解释。
- extend：增加新的进展、条件、范围或行动含义。
- revise：已有解释需要被修正。
- contradict：明确反驳、削弱或使已有解释产生冲突。
- unrelated：与候选解释无关。

判断要求：
- update 或 evidence_only 时，target_interpretation_id 必须来自候选列表。
- contradict 通常应选择 update，不能仅因语义方向相反而判为 unrelated。
- 候选为空时，只能选择 unmatched、defer 或 ignore。
- 单条 observation 独立价值低，不代表应该 ignore；若可能形成重复模式、时间进展或组合证据，应选择 unmatched 或 defer。
- 不要仅凭 entity/topic 相同认定匹配，必须判断命题、对象、范围和时间状态是否相关。
- conflict_level 只能是 none、partial、strong。
- intrinsic_value 只能是 low、medium、high。

只返回合法 JSON，不要 markdown，不要额外解释：
{{
  "decision": "update | evidence_only | unmatched | defer | ignore",
  "target_interpretation_id": 1,
  "relationship": "support | extend | revise | contradict | unrelated",
  "conflict_level": "none | partial | strong",
  "intrinsic_value": "low | medium | high",
  "reason": "简洁说明判断依据"
}}"""

INTERPRETATION_UPDATE_PROMPT = """你是长期记忆 interpretation 更新模块。

你需要根据新的 observation 和 supporting facts，更新一条已经存在的 interpretation。

这里的 interpretation 是 Agent 当前对记忆证据的 current best interpretation。更新时要让它吸收新 observation 带来的进展、确认、修正、冲突或范围变化，而不是只追加证据。

四层记忆架构：
- fact：原始证据，表示对话中提取出的事实。
- evidence_bundle：按相同 entity/topic 组织 facts 的容器，只负责限定证据范围。
- observation：evidence_bundle 内由一组相似 facts 直接支持的具体命题，是 interpretation 匹配和更新的主要语义单元。
- interpretation：当前解释，表示 Agent 现在如何理解这些 observations，以及后续应该如何行动。

entity: {entity_name}
topic: {topic_label}

已有 interpretation：
id: {interpretation_id}
interpretation_type: {interpretation_type}
status: {status}
conflict_status: {conflict_status}
polarity: {polarity}
strength: {strength}
confidence: {confidence}
target_text: {target_text}
scope: {scope}
claim: {claim}
resolution: {resolution}
action_implication: {action_implication}
metadata: {interpretation_metadata}

新的 observation：
id: {observation_id}
observation_type: {observation_type}
summary: {observation_summary}
metadata: {observation_metadata}

supporting facts:
{source_facts}

更新要求：
- 如果新的 observation 只是确认旧 interpretation，可以小幅提高 confidence/strength，并让 metadata 反映确认来源。
- 如果新的 observation 表示进展、状态变化、结果、冲突或修正，应更新 claim、action_implication、status/conflict_status、resolution 或 task metadata。
- 如果 interpretation_type=task，metadata 中填写 task_status、task_source、goal、evidence、steps；task_status 只能是 active、blocked、paused、stale，task_source 固定为 inferred_from_interpretation。
- 如果 interpretation_type 不是 task，不要填写 task_status、goal、steps、next_action。
- 不要改变 interpretation_type，除非原类型明显错误；若必须改变，只能使用合法类型。
- 新输入 metadata 如果包含 allowed_interpretation_types，更新后的 interpretation_type 必须位于该列表；不兼容时输出 {{"should_update": false}}。
- 不要编造没有证据支持的新目标、偏好或风险。
- evidence_fact_ids 是直接支持更新后 interpretation 的底层 fact id，表示“为什么 Agent 现在仍然相信这个解释”；只能使用 supporting facts 中出现的 id。
- evidence_observation_ids 是支持更新后 interpretation 的 observation id，表示“哪些中层归纳支撑这个解释”；通常应包含新的 observation id，只能使用输入 observation 的 id。
- counter_evidence_fact_ids 是反驳、削弱、限定或造成冲突的底层 fact id；只有新 observation 或 supporting facts 提供明确反证、例外、边界条件或 unresolved conflict 时填写。
- counter_evidence_observation_ids 是反驳、削弱、限定或造成冲突的 observation id；只有存在明确反证、例外、边界条件或 unresolved conflict 时填写。
- 如果新 evidence 只是范围变窄或条件更明确，可以更新 claim/action_implication/resolution，不必一定放入 counter_evidence_*。
- 如果没有任何内容需要更新，只输出 {{"should_update": false}}。

字段要求：
- claim 是更新后的 Agent 当前解释，必须谨慎、可证据支持；不要写成用户原话。
- interpretation_type 只能是 insight、task、explicit_preference、explicit_instruction、inferred_preference、behavior_pattern、project_state、task_risk、constraint、conflict_resolution、strategy、other。
- status 只能是 current、conflicted；不要输出 superseded 或 archived。
- conflict_status 只能是 none、resolved、unresolved。
- polarity 只能是 positive、negative、mixed、neutral。
- strength/confidence 是 0.0-1.0。
- action_implication 描述这条解释未来如何影响 Agent 行为；如果旧内容仍然准确，可以保留但应吸收新 observation 的变化。

只返回合法 JSON，不要 markdown，不要额外解释。格式如下：
{{
  "should_update": true,
  "claim": "更新后的 Agent 当前解释...",
  "target_text": "解释对象",
  "scope": "适用范围",
  "interpretation_type": "insight | task | explicit_preference | explicit_instruction | inferred_preference | behavior_pattern | project_state | task_risk | constraint | conflict_resolution | strategy | other",
  "polarity": "positive | negative | mixed | neutral",
  "strength": 0.0,
  "confidence": 0.0,
  "status": "current | conflicted",
  "conflict_status": "none | resolved | unresolved",
  "resolution": "可选，冲突如何被解决",
  "action_implication": "未来 Agent 应如何使用这个解释",
  "evidence_fact_ids": [1, 2],
  "evidence_observation_ids": [3],
  "counter_evidence_fact_ids": [],
  "counter_evidence_observation_ids": [],
  "metadata": {{"source": "interpretation_update"}}
}}"""

INTERPRETATION_BATCH_UPDATE_PROMPT = """你是长期记忆 interpretation 批量更新模块。

你需要使用一组已经确认匹配的 observations，一次性更新已有 interpretation。
这些 observations 可能分别提供支持、进展、修正或冲突。请形成统一的 current best interpretation，不要按输入顺序机械追加。

已有 interpretation：
{interpretation}

matched observations:
{observations}

supporting facts:
{source_facts}

更新要求：
- 综合全部 observations 后再更新 claim、状态、范围、置信度和行动含义。
- support 可以增强证据；extend 应吸收新进展或条件；revise 应修正原解释；contradict 应体现冲突或明确的新结论。
- 明确支持当前解释的 observation id 放入 evidence_observation_ids。
- 明确反驳、削弱或限制当前解释的 observation id 放入 counter_evidence_observation_ids。
- 对应的底层 fact id 分别放入 evidence_fact_ids 或 counter_evidence_fact_ids。
- 如果 interpretation_type=task，metadata 中填写 task_status、task_source、goal、evidence、steps；task_source 固定为 inferred_from_interpretation。
- 不要编造输入中没有的目标、偏好、风险或结论。
- 如果综合后内容不需要改变，只输出 {{"should_update": false}}。

字段约束：
- interpretation_type 只能是 insight、task、explicit_preference、explicit_instruction、inferred_preference、behavior_pattern、project_state、task_risk、constraint、conflict_resolution、strategy、other。
- status 只能是 current、conflicted。
- conflict_status 只能是 none、resolved、unresolved。
- polarity 只能是 positive、negative、mixed、neutral。
- strength/confidence 必须是 0.0-1.0。
- evidence_* 和 counter_evidence_* 只能引用输入中的 id。

只返回合法 JSON，不要 markdown，不要额外解释：
{{
  "should_update": true,
  "claim": "更新后的 Agent 当前解释",
  "target_text": "解释对象",
  "scope": "适用范围",
  "interpretation_type": "insight | task | explicit_preference | explicit_instruction | inferred_preference | behavior_pattern | project_state | task_risk | constraint | conflict_resolution | strategy | other",
  "polarity": "positive | negative | mixed | neutral",
  "strength": 0.0,
  "confidence": 0.0,
  "status": "current | conflicted",
  "conflict_status": "none | resolved | unresolved",
  "resolution": "可选，冲突如何被解决",
  "action_implication": "未来 Agent 应如何使用这个解释",
  "evidence_fact_ids": [1, 2],
  "evidence_observation_ids": [3],
  "counter_evidence_fact_ids": [],
  "counter_evidence_observation_ids": [],
  "metadata": {{"source": "interpretation_batch_update"}}
}}"""

INTERPRETATION_FEEDBACK_ANALYSIS_PROMPT = """你是长期记忆系统的 interpretation feedback 分析模块。

你需要判断 current_user_message 是否在反馈上一轮 assistant 回答中召回的 interpretations。

重要边界：
- 只处理用户明确确认、否认、修正、延期或表示过时的信息。
- 用户开启新话题、提出新任务、普通追问或没有明显指向这些 interpretations 时，has_feedback=false。
- 不要把沉默、换话题或没有接话当作负反馈。
- feedback 必须绑定到输入中的 interpretation_id。
- recall 只是候选召回，不代表每个 recalled_interpretation 都和 current_user_message 有关；如果某条 interpretation 与当前用户消息没有直接反馈关系，feedback_type=unrelated。
- has_feedback=true 仅表示至少存在一条 accept/reject/modify/defer/outdated；如果全部是 unrelated 或 none，has_feedback=false。

previous_user_query:
{previous_user_query}

previous_assistant_response:
{previous_assistant_response}

recalled_interpretations:
{interpretations}

current_user_message:
{current_user_message}

feedback_type 定义：
- accept：用户明确确认 interpretation 成立。
- reject：用户明确否认 interpretation。
- modify：用户修正 interpretation 的对象、原因、范围、状态或行动含义。
- defer：用户表示现在不想处理、以后再说；不代表 interpretation 错误。
- outdated：用户表示该 interpretation 已不适用或已经结束。
- unrelated：current_user_message 与该 interpretation 没有直接反馈关系，通常是误召回、新话题或只与其他 interpretation 有关。
- none：无法判断是否有可用反馈，或没有足够证据归类。

只返回合法 JSON，不要 markdown，不要额外解释：
{{
  "has_feedback": true,
  "feedback_items": [
    {{
      "interpretation_id": 1,
      "feedback_type": "accept | reject | modify | defer | outdated | unrelated | none",
      "confidence": 0.0,
      "evidence_text": "用户原文中支持该判断的短句",
      "correction": "如果 feedback_type=modify，写出用户修正后的含义；否则可为空"
    }}
  ]
}}"""

INTERPRETATION_UPDATE_USING_FEEDBACK_PROMPT = """你是长期记忆 interpretation 反馈校准模块。

你需要根据用户对 interpretation 的明确反馈，更新这条 interpretation。
反馈是直接来自用户的校准信号，优先级高于一般 observation，但仍然不能编造用户没有表达的新事实。

existing_interpretation:
{interpretation}

user_feedback:
{feedback}

更新要求：
- accept：通常只小幅提高 confidence/strength，claim/action_implication 可保持不变。
- reject：降低 confidence；如果用户明确否认核心判断，可将 status 设为 conflicted 或 archived，并在 resolution 中说明用户否认点。
- modify：吸收用户 correction，更新 claim/action_implication/scope/resolution；不要保留已被用户纠正的错误推断。
- defer：不改变真假判断，主要在 metadata 中体现暂缓，通常不必改 claim。
- outdated：如果用户表示已不适用，可将 status 设为 archived 或 superseded。

字段约束：
- status 只能是 current、conflicted、archived、superseded。
- conflict_status 只能是 none、resolved、unresolved。
- polarity 只能是 positive、negative、mixed、neutral。
- strength/confidence 必须是 0.0-1.0。
- metadata 应包含 source="interpretation_feedback_update"。
- 如果不需要更新内容，只输出 {{"should_update": false}}。

只返回合法 JSON，不要 markdown，不要额外解释：
{{
  "should_update": true,
  "claim": "更新后的 Agent 当前解释",
  "target_text": "解释对象",
  "scope": "适用范围",
  "interpretation_type": "insight | task | explicit_preference | explicit_instruction | inferred_preference | behavior_pattern | project_state | task_risk | constraint | conflict_resolution | strategy | other",
  "polarity": "positive | negative | mixed | neutral",
  "strength": 0.0,
  "confidence": 0.0,
  "status": "current | conflicted | archived | superseded",
  "conflict_status": "none | resolved | unresolved",
  "resolution": "反馈如何改变或确认这条解释",
  "action_implication": "未来 Agent 应如何使用这个解释",
  "metadata": {{"source": "interpretation_feedback_update"}}
}}"""

# ── Reflect prompt template ───────────────────────────────────────────────

# ── Memory node context block template ───────────────────────────────────
# NOTE: recall() returns RAW text (no wrapper). Callers (run_agent.py) use
# build_memory_context_block() from agent/memory_manager.py to wrap in
# <memory-context> tags.  This avoids double-wrapping issues where the
# sanitize_context regex strips pre-wrapped content entirely.

MEMORY_NODE_HEADER = "[Memory recall — past conversation summaries relevant to the current query]"
WORLD_FACT_SECTION_HEADER = (
    "[Semantic memories — stable facts, concepts, preferences, project background, and common knowledge]"
)
EXPERIENCE_SECTION_HEADER = (
    "[Episodic memories — specific user/assistant experiences, events, decisions, actions, and outcomes]"
)
OBSERVATION_SECTION_HEADER = (
    "[Observations — consolidated patterns inferred from related facts and experiences]"
)
OBSERVATION_SUPPORT_SECTION_HEADER = (
    "[Supporting facts for observations]"
)
INTERPRETATION_SECTION_HEADER = (
    "[Current interpretations — agent's current best interpretation of memory]"
)

MEMORY_CONTEXT_BLOCK = """<memory-context>
[System note: The following are relevant past conversation memories, NOT new user input. Treat as informational background data.]

{memory_text}
</memory-context>"""


def _llm_base_host(base_url: str) -> str:
    try:
        parsed = urlparse(str(base_url or ""))
    except Exception:
        return ""
    return (parsed.hostname or "").lower()


def _llm_prefers_max_completion_tokens(base_url: str) -> bool:
    host = _llm_base_host(base_url)
    return host == "api.openai.com" or host.endswith(".openai.azure.com")


def _llm_model_requires_responses_api(model: str) -> bool:
    text = str(model or "").strip().lower()
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return text.startswith("gpt-5")


def _llm_error_text(exc: Exception) -> str:
    return str(exc or "").lower()


def _llm_temperature_rejected(exc: Exception) -> bool:
    text = _llm_error_text(exc)
    return "temperature" in text and (
        "unsupported" in text
        or "not support" in text
        or "invalid" in text
        or "only the default" in text
    )


def _llm_max_tokens_rejected(exc: Exception) -> bool:
    text = _llm_error_text(exc)
    return (
        "max_tokens" in text
        or "max_completion_tokens" in text
        or "unsupported_parameter" in text
    )


def _llm_unsupported_chat_api(exc: Exception) -> bool:
    text = _llm_error_text(exc)
    return "unsupported_api_for_model" in text or "responses api" in text


def _response_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    parts: List[str] = []
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    for item in output or []:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        for block in content or []:
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            if text:
                parts.append(str(text))
    return "\n".join(parts)


def _call_llm_responses_api(prompt: str, model: str, base_url: str, api_key: str,
                            timeout: int = 120) -> Optional[str]:
    url = f"{base_url.rstrip('/')}/responses"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "input": prompt,
        "max_output_tokens": 2048,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return _response_output_text(resp.json()) or None
    except requests.exceptions.RequestException as e:
        logger.debug("Responses API LLM call failed: %s", e)
        return None


def _call_llm_api(prompt: str, model: str, base_url: str, api_key: str,
                  timeout: int = 120) -> Optional[str]:
    """Call an OpenAI-compatible chat completions API with a single user message.

    Supports OpenAI, OpenRouter, DeepSeek, vLLM, and any provider that exposes
    a ``/v1/chat/completions`` endpoint.

    The entire *prompt* is sent as a ``user`` message (system instructions are
    embedded directly in the prompt text).  Returns the response text, or
    ``None`` on failure.
    """
    if _llm_model_requires_responses_api(model):
        response_text = _call_llm_responses_api(
            prompt,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        if response_text:
            return response_text

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "stream": False,
    }
    data.update(
        {"max_completion_tokens": 2048}
        if _llm_prefers_max_completion_tokens(base_url)
        else {"max_tokens": 2048}
    )

    attempts = [data]
    temperature_stripped = dict(data)
    temperature_stripped.pop("temperature", None)
    attempts.append(temperature_stripped)
    token_swapped = dict(temperature_stripped)
    if "max_tokens" in token_swapped:
        token_swapped["max_completion_tokens"] = token_swapped.pop("max_tokens")
    elif "max_completion_tokens" in token_swapped:
        token_swapped["max_tokens"] = token_swapped.pop("max_completion_tokens")
    attempts.append(token_swapped)

    seen_payloads = set()
    last_error: Optional[Exception] = None
    for payload in attempts:
        marker = json.dumps(sorted(payload.keys()), ensure_ascii=False)
        if marker in seen_payloads:
            continue
        seen_payloads.add(marker)
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            result = resp.json()
            choices = result.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return None
        except requests.exceptions.RequestException as e:
            last_error = e
            if not (_llm_temperature_rejected(e) or _llm_max_tokens_rejected(e)):
                break
    if last_error is not None:
        logger.debug("LLM API call failed: %s", last_error)
    return None


class MemoryNodeManager:
    """Manages automatic creation, storage, and retrieval of summarized memory nodes.

    Summarisation uses the project-wide OpenAI client (``llm_client``) or
    falls back to raw HTTP via ``_call_llm_api``.
    Embedding uses ``EmbeddingClient``.
    Storage/search uses ``SessionDB.memory_*`` methods.

    Usage::

        # With shared AIAgent client:
        mgr = MemoryNodeManager(
            session_db,
            embedding_config=embedding_config,
            memory_config=memory_config,
            llm_client=agent.client,
        )

        # Standalone (raw HTTP):
        mgr = MemoryNodeManager(
            session_db,
            embedding_config=embedding_config,
            memory_config=memory_config,
        )
    """

    def __init__(
        self,
        session_db: Any,  # SessionDB instance
        embedding_config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        llm_client: Any = None,  # OpenAI-compatible client (shares the project's API infra)
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        memory_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._db = session_db
        self._enabled = enabled and bool(session_db)
        self._embedding_client: Any = None  # lazy init
        self._llm_client = llm_client

        embedding_cfg = embedding_config or {}
        memory_cfg = memory_config or {}

        # LLM model config (used regardless of client or raw HTTP). This is
        # supplied by the owning agent so memory extraction uses the same model
        # that answers the user's query, not embedding_config.
        self._llm_model = str(llm_model or DEFAULT_LLM_MODEL)
        self._llm_timeout = int(memory_cfg.get("llm_timeout", 120))

        # Raw HTTP fallback config (only used when llm_client is None). These
        # are supplied by the owning agent, not embedding_config.
        self._llm_base_url = str(llm_base_url or DEFAULT_LLM_BASE_URL)
        self._llm_api_key = "" if llm_api_key is None else str(llm_api_key)

        # Retrieval config
        self._top_k = int(memory_cfg.get("retrieval_top_k", 8))
        self._recall_observation_min_embedding_similarity = self._clip_unit_float(
            memory_cfg.get("recall_observation_min_embedding_similarity"),
            0.35,
        )
        self._recall_interpretation_min_embedding_similarity = self._clip_unit_float(
            memory_cfg.get("recall_interpretation_min_embedding_similarity"),
            0.45,
        )
        self._min_turns_before_store = max(
            1,
            int(memory_cfg.get("min_turns_before_store", 1) or 1),
        )
        self._max_chars_before_store = max(
            1,
            int(memory_cfg.get("max_chars_before_store", 2000) or 2000),
        )
        self._pending_store_turns: List[Dict[str, Any]] = []

        # Default recall budget: "mid"
        self._recall_budget = memory_cfg.get("recall_budget", "mid")

        # Enable entity extraction (default: True if session_db available)
        self._enable_entity_extraction = memory_cfg.get(
            "enable_entity_extraction",
            True,
        )

        self._turn_count = 0
        self._embedding_cfg = embedding_cfg
        self._memory_cfg = memory_cfg

        self._store_queue_maxsize = max(
            1,
            int(memory_cfg.get("store_queue_maxsize", 100) or 100),
        )
        self._reflect_interval_seconds = max(
            1.0,
            float(memory_cfg.get("reflect_interval_seconds", 3600) or 3600),
        )
        self._decay_interval_seconds = max(
            1.0,
            float(memory_cfg.get("decay_interval_seconds", 86400) or 86400),
        )
        self._reflect_observation_interpretation_min_embedding_similarity = self._clip_unit_float(
            memory_cfg.get("reflect_observation_interpretation_min_embedding_similarity"),
            0.45,
        )
        self._store_queue: queue.Queue[Dict[str, Any]] = queue.Queue(
            maxsize=self._store_queue_maxsize,
        )
        self._store_worker_thread: Optional[threading.Thread] = None
        self._store_worker_lock = threading.Lock()
        self._store_shutdown_event = threading.Event()
        self._llm_thread_context = threading.local()
        self._reflect_queued_or_running = False
        self._decay_queued_or_running = False
        try:
            self._last_successful_reflect_at = float(
                self._db.get_meta(MEMORY_REFLECT_META_KEY) or 0.0
            )
        except Exception:
            self._last_successful_reflect_at = 0.0
        try:
            self._last_successful_decay_at = float(
                self._db.get_meta(MEMORY_DECAY_META_KEY) or 0.0
            )
        except Exception:
            self._last_successful_decay_at = 0.0

    def configure_llm(
        self,
        *,
        llm_client: Any = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
    ) -> None:
        """Refresh the shared LLM client/model from the owning agent."""
        if llm_client is not None:
            self._llm_client = llm_client
        if llm_model:
            self._llm_model = str(llm_model)
        if llm_base_url:
            self._llm_base_url = str(llm_base_url)
        if llm_api_key is not None:
            self._llm_api_key = str(llm_api_key)

    def _ensure_embedding_client(self, *, optional: bool = False) -> bool:
        if self._embedding_client is not None:
            return True
        try:
            from agent.embedding_client import EmbeddingClient
            # Don't pass llm_client — embedding backends use their own provider
            # config, which may differ from the chat provider (e.g. DeepSeek
            # doesn't support embeddings, but OpenRouter / OpenAI do).
            self._embedding_client = EmbeddingClient(self._embedding_cfg)
            return True
        except Exception as e:
            logger.error("Failed to init EmbeddingClient: %s", e)
            if not optional:
                self._enabled = False
            return False

    def _embed_memory_layer_text(self, text: str) -> Optional[np.ndarray]:
        """Embed observation/interpretation text for later cheap recall reranking."""
        clean_text = str(text or "").strip()
        if not clean_text:
            return None
        try:
            if self._embedding_client is None and not self._ensure_embedding_client(optional=True):
                return None
            return self._embedding_client.embed_text(clean_text)
        except Exception as exc:
            logger.debug("Memory layer embedding failed: %s", exc)
            return None

    @staticmethod
    def _build_interpretation_embedding_text(
        *,
        entity_name: str = "",
        target_text: str = "",
        scope: str = "",
        interpretation_type: str = "",
        claim: str = "",
        action_implication: str = "",
        resolution: str = "",
    ) -> str:
        return "\n".join(
            part
            for part in [
                f"entity: {entity_name}" if entity_name else "",
                f"target: {target_text}" if target_text else "",
                f"scope: {scope}" if scope else "",
                f"type: {interpretation_type}" if interpretation_type else "",
                f"claim: {claim}" if claim else "",
                f"action: {action_implication}" if action_implication else "",
                f"resolution: {resolution}" if resolution else "",
            ]
            if part
        )

    # ── LLM call (shared client when available) ─────────────────────────

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM using the shared OpenAI client, falling back to raw HTTP.

        When ``self._llm_client`` is set (passed from ``AIAgent``), uses the
        project-wide OpenAI client — same connection pool, same API endpoint,
        same authentication.  Otherwise falls back to ``_call_llm_api`` (raw
        ``requests.post`` to an OpenAI-compatible ``/v1/chat/completions``).
        """
        thread_config = getattr(self._llm_thread_context, "config", None) or {}
        llm_client = thread_config.get("llm_client", self._llm_client)
        llm_model = str(thread_config.get("llm_model") or self._llm_model)
        llm_base_url = str(thread_config.get("llm_base_url") or self._llm_base_url)
        llm_api_key = str(thread_config.get("llm_api_key", self._llm_api_key) or "")

        if llm_client is not None:
            if _llm_model_requires_responses_api(llm_model):
                responses = getattr(llm_client, "responses", None)
                create = getattr(responses, "create", None)
                if create is not None:
                    try:
                        resp = create(
                            model=llm_model,
                            input=prompt,
                            max_output_tokens=2048,
                            timeout=self._llm_timeout,
                        )
                        text = _response_output_text(resp)
                        if text:
                            return text
                    except Exception as e:
                        logger.debug("Shared Responses API LLM call failed: %s", e)

            base_kwargs = {
                "model": llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "timeout": self._llm_timeout,
            }
            base_kwargs.update(
                {"max_completion_tokens": 2048}
                if _llm_prefers_max_completion_tokens(llm_base_url)
                else {"max_tokens": 2048}
            )
            attempts = [base_kwargs]
            temperature_stripped = dict(base_kwargs)
            temperature_stripped.pop("temperature", None)
            attempts.append(temperature_stripped)
            token_swapped = dict(temperature_stripped)
            if "max_tokens" in token_swapped:
                token_swapped["max_completion_tokens"] = token_swapped.pop("max_tokens")
            elif "max_completion_tokens" in token_swapped:
                token_swapped["max_tokens"] = token_swapped.pop("max_completion_tokens")
            attempts.append(token_swapped)

            seen_payloads = set()
            last_error: Optional[Exception] = None
            for kwargs in attempts:
                marker = json.dumps(sorted(kwargs.keys()), ensure_ascii=False)
                if marker in seen_payloads:
                    continue
                seen_payloads.add(marker)
                try:
                    resp = llm_client.chat.completions.create(**kwargs)
                    return getattr(resp.choices[0].message, "content", "") or ""
                except Exception as e:
                    last_error = e
                    if _llm_unsupported_chat_api(e):
                        responses = getattr(llm_client, "responses", None)
                        create = getattr(responses, "create", None)
                        if create is not None:
                            try:
                                resp = create(
                                    model=llm_model,
                                    input=prompt,
                                    max_output_tokens=2048,
                                    timeout=self._llm_timeout,
                                )
                                text = _response_output_text(resp)
                                if text:
                                    return text
                            except Exception as responses_error:
                                logger.debug(
                                    "Shared Responses API retry failed: %s",
                                    responses_error,
                                )
                        break
                    if _llm_temperature_rejected(e) or _llm_max_tokens_rejected(e):
                        logger.debug("Shared LLM client rejected params, retrying: %s", e)
                        continue
                    break
            logger.debug("Shared LLM client call failed: %s", last_error)
            return None
        return _call_llm_api(
            prompt,
            model=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            timeout=self._llm_timeout,
        )

    @staticmethod
    def _build_dialogue_batch_for_prompt(
        source_turns: List[Dict[str, Any]],
        fallback_timestamp: Optional[Any] = None,
    ) -> str:
        """Format source turns without losing user/assistant pairing."""
        if isinstance(fallback_timestamp, datetime):
            fallback_timestamp_text = fallback_timestamp.isoformat()
        else:
            fallback_timestamp_text = str(fallback_timestamp or "")

        def _turn_timestamp(turn: Dict[str, Any]) -> str:
            value = turn.get("turn_timestamp")
            if isinstance(value, datetime):
                return value.isoformat()
            return str(value or fallback_timestamp_text)

        return "\n\n".join(
            (
                f"[Turn {index}]\n"
                f"对话发生时间：{_turn_timestamp(turn)}\n"
                f"用户：{turn.get('user_message') or ''}\n"
                f"助手：{turn.get('assistant_response') or ''}"
            )
            for index, turn in enumerate(source_turns, start=1)
        )

    def _summarize_turn(
        self,
        source_turns: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Summarise a batch of paired conversation turns via LLM API call.

        Retries once on parse failure to handle transient API errors.
        """
        if not source_turns:
            return None
        dialogue_batch = self._build_dialogue_batch_for_prompt(source_turns)
        prompt = SUMMARY_SYSTEM_PROMPT.format(
            dialogue_batch=dialogue_batch,
        )
        
        for attempt in range(2):
            result = self._call_llm(prompt)
            if not result:
                if attempt == 0:
                    logger.debug("Summarisation attempt %d returned empty, retrying...", attempt)
                    continue
                return None
            # Parse JSON from the response
            text = result.strip()
            # Strip code fences if present
            if "```" in text:
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    text = text[start:end + 1]
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                if attempt == 0:
                    logger.debug("Summarisation JSON parse failed on attempt %d, retrying...", attempt)
                    continue
                logger.debug("Summarisation JSON parse failed after 2 attempts: %.120s", result)
                return None

            summary = data.get("summary", "").strip()
            keywords = data.get("keywords", [])
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(",") if k.strip()]
            entities = self._normalize_fact_entities(data.get("entities", []))
            if not summary:
                if attempt == 0:
                    continue
                return None
            return {"summary": summary, "keywords": keywords, "entities": entities}

        return None

    # ── Recall query analysis ────────────────────────────────────────────

    @staticmethod
    def _clip_unit_float(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, number))

    @classmethod
    def _normalize_recall_layer_preference(cls, value: Any) -> Dict[str, float]:
        if not isinstance(value, dict):
            return {}
        raw = {
            "interpretations": cls._clip_unit_float(value.get("interpretations"), 0.0),
            "observations": cls._clip_unit_float(value.get("observations"), 0.0),
            "facts": cls._clip_unit_float(value.get("facts"), 0.0),
        }
        total = sum(raw.values())
        if total <= 0:
            return {}
        return {key: round(score / total, 4) for key, score in raw.items() if score > 0}

    @staticmethod
    def _normalize_recall_intent(value: Any) -> str:
        intent = str(value or "").strip().lower()
        return intent if intent in {"action", "evidence", "state", "balanced"} else "balanced"

    @staticmethod
    def _normalize_fact_type_preference(value: Any) -> str:
        fact_type = str(value or "").strip().lower()
        return fact_type if fact_type in {"semantic", "episodic", "both"} else "both"

    @staticmethod
    def _normalize_time_sensitivity(value: Any) -> str:
        sensitivity = str(value or "").strip().lower()
        return sensitivity if sensitivity in {"specific", "recent", "long_term", "none"} else "none"

    @staticmethod
    def _normalize_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1"}:
                return True
            if normalized in {"false", "no", "0"}:
                return False
        return default

    @staticmethod
    def _normalize_string_list(value: Any, *, limit: int = 8) -> List[str]:
        if isinstance(value, str):
            raw = value.split(",")
        elif isinstance(value, list):
            raw = value
        else:
            raw = []
        out: List[str] = []
        seen = set()
        for item in raw:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _fallback_query_keywords(query: str, *, limit: int = 8) -> List[str]:
        keywords: List[str] = []
        seen = set()
        for item in re.split(r"\s+|,|，|;|；|。|\?|？", str(query or "")):
            text = item.strip()
            if len(text) <= 1 or text in seen:
                continue
            seen.add(text)
            keywords.append(text)
            if len(keywords) >= limit:
                break
        return keywords

    def _normalize_recall_query_analysis(
        self,
        data: Dict[str, Any],
        query: str,
    ) -> Optional[Dict[str, Any]]:
        search_text = str(data.get("search_text") or data.get("summary") or query or "").strip()
        if not search_text:
            return None
        keywords = self._normalize_keywords(data.get("keywords"))
        if not keywords:
            keywords = self._fallback_query_keywords(query)
        return {
            # Older analysis payloads did not contain a recall gate. Defaulting
            # to True preserves their behavior and avoids accidental misses.
            "needs_recall": self._normalize_bool(data.get("needs_recall"), True),
            "recall_confidence": self._clip_unit_float(data.get("recall_confidence"), 0.0),
            "recall_reason": str(data.get("recall_reason") or "").strip(),
            "search_text": search_text,
            "keywords": keywords[:8],
            "entities": self._normalize_fact_entities(data.get("entities", [])),
            "recall_intent": self._normalize_recall_intent(data.get("recall_intent")),
            "intent_confidence": self._clip_unit_float(data.get("intent_confidence"), 0.0),
            "layer_preference": self._normalize_recall_layer_preference(data.get("layer_preference")),
            "fact_type_preference": self._normalize_fact_type_preference(data.get("fact_type_preference")),
            "interpretation_type_preference": self._normalize_string_list(
                data.get("interpretation_type_preference"),
                limit=5,
            ),
            "needs_evidence": bool(data.get("needs_evidence", False)),
            "time_sensitivity": self._normalize_time_sensitivity(data.get("time_sensitivity")),
        }

    def _analyze_recall_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Analyze a recall query in one LLM call.

        Accepts the older ``{"summary", "keywords", "entities"}`` shape as a
        compatibility fallback so recall remains best-effort if a model returns
        the previous summary schema.
        """
        prompt = RECALL_QUERY_ANALYSIS_PROMPT.format(query=query)
        for attempt in range(2):
            result = self._call_llm(prompt)
            if not result:
                if attempt == 0:
                    logger.debug("Recall query analysis attempt %d returned empty, retrying...", attempt)
                    continue
                return None
            data = self._parse_json_object_from_llm_text(result)
            if data is None:
                if attempt == 0:
                    logger.debug("Recall query analysis JSON parse failed on attempt %d, retrying...", attempt)
                    continue
                logger.debug("Recall query analysis JSON parse failed after 2 attempts: %.120s", result)
                return None
            normalized = self._normalize_recall_query_analysis(data, query)
            if normalized:
                return normalized
            if attempt == 0:
                continue
            return None
        return None

    @staticmethod
    def _rule_based_recall_gate(query: str) -> Dict[str, str]:
        """Handle only high-confidence recall decisions without an LLM call."""
        text = str(query or "").strip()
        normalized = re.sub(r"[\s，。！？、,.!?;；:：~～]+", "", text).lower()
        if not normalized:
            return {"decision": "skip", "reason": "empty_query"}

        trivial_queries = {
            "你好", "您好", "嗨", "哈喽", "hello", "hi", "hey",
            "谢谢", "感谢", "多谢", "thankyou", "thanks",
            "好的", "好", "可以", "明白了", "知道了", "收到", "没问题",
            "ok", "okay", "gotit",
            "再见", "拜拜", "bye", "goodbye",
        }
        if normalized in trivial_queries:
            return {"decision": "skip", "reason": "trivial_social_query"}

        explicit_history_terms = (
            "你还记得", "还记得我", "之前我", "我之前", "我们之前",
            "上次我", "我们上次", "以前我", "过去我", "曾经我",
            "我说过", "我提到过", "历史记录", "根据你对我的了解",
            "继续之前", "接着之前", "what did i", "do you remember",
            "last time", "previously", "my history",
        )
        lowered = text.lower()
        if any(term in lowered for term in explicit_history_terms):
            return {"decision": "recall", "reason": "explicit_history_reference"}
        return {"decision": "analyze", "reason": "semantic_judgment_required"}

    @staticmethod
    def _parse_json_object_from_llm_text(text: str) -> Optional[Dict[str, Any]]:
        """Parse a JSON object from an LLM response, tolerating code fences."""
        raw = (text or "").strip()
        if not raw:
            return None
        if "```" in raw:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                raw = raw[start:end + 1]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _normalize_keywords(value: Any) -> List[str]:
        if isinstance(value, str):
            raw = value.split(",")
        elif isinstance(value, list):
            raw = value
        else:
            raw = []
        out: List[str] = []
        seen = set()
        for item in raw:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    @staticmethod
    def _normalize_fact_entities(value: Any) -> List[Dict[str, str]]:
        if not isinstance(value, list):
            return []
        entities: List[Dict[str, str]] = []
        seen = set()
        for item in value:
            if isinstance(item, str):
                name = item.strip()
                etype = "CONCEPT"
            elif isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                etype = str(item.get("type", "CONCEPT")).strip().upper() or "CONCEPT"
            else:
                continue
            if not name or name in seen:
                continue
            if is_temporal_entity(name, etype):
                continue
            if is_attribute_entity(name, etype):
                continue
            seen.add(name)
            entities.append({"name": name, "type": etype})
        return entities

    @staticmethod
    def _subject_entity_name(fact_subject: Any) -> str:
        subject = MemoryNodeManager._normalize_fact_subject(fact_subject)
        if subject == "user":
            return "用户"
        if subject == "assistant":
            return "助手"
        return ""

    @classmethod
    def _fact_entities_with_subject(
        cls,
        entities: List[Dict[str, str]],
        fact_subject: Any,
    ) -> List[Dict[str, str]]:
        subject_name = cls._subject_entity_name(fact_subject)
        if not subject_name:
            return entities
        if any(str(entity.get("name") or "").strip() == subject_name for entity in entities):
            return entities
        return [{"name": subject_name, "type": "OTHER"}, *entities]

    def _fallback_fact_from_summary(
        self,
        summary_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        keywords = self._normalize_keywords(summary_data.get("keywords", []))
        entities = self._normalize_fact_entities(summary_data.get("entities", []))
        return {
            "text": str(summary_data.get("summary", "")).strip(),
            "keywords": keywords,
            "primary_entity": entities[0] if entities else None,
            "primary_topic": keywords[0] if keywords else "general",
            "topic": [keywords[0] if keywords else "general"],
            "fact_type": "episodic",
            "fact_subject": "other",
            "fact_kind": "conversation_summary",
            "priority": 60,
            "priority_reason": "fallback conversation summary",
            "task_event_like": None,
            "task_event_subject": "",
            "task_relevance": "",
            "occurred_start": "",
            "occurred_end": "",
            "time_confidence": "unknown",
            "where": "",
            "entities": entities,
        }

    @staticmethod
    def _normalize_fact_type(value: Any) -> str:
        fact_type = str(value or "semantic").strip().lower().replace("-", "_").replace(" ", "_")
        if fact_type in {"episodic", "episodic_memory", "experience", "experience_fact"}:
            return "episodic"
        if fact_type in {"semantic", "semantic_memory", "world", "world_fact"}:
            return "semantic"
        return "semantic"

    @staticmethod
    def _normalize_fact_subject(value: Any) -> str:
        fact_subject = str(value or "other").strip().lower().replace("-", "_").replace(" ", "_")
        allowed = {"user", "assistant", "world", "project", "system", "other"}
        return fact_subject if fact_subject in allowed else "other"

    @staticmethod
    def _normalize_fact_kind(value: Any) -> str:
        allowed = {
            "preference", "decision", "request", "recommendation",
            "action", "error", "context", "instruction",
            "conversation_summary", "other",
        }
        fact_kind = str(value or "other").strip().lower()
        return fact_kind if fact_kind in allowed else "other"

    @staticmethod
    def _normalize_fact_priority(value: Any) -> int:
        try:
            priority = int(float(value))
        except (TypeError, ValueError):
            return 70
        return max(0, min(100, priority))

    @staticmethod
    def _normalize_time_confidence(value: Any) -> str:
        time_confidence = str(value or "unknown").strip().lower()
        allowed = {"explicit", "inferred_from_turn", "unknown"}
        return time_confidence if time_confidence in allowed else "unknown"

    def _extract_retain_facts(
        self,
        source_turns: List[Dict[str, Any]],
        turn_timestamp: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Extract HindSight-style narrative facts for retain.

        The preferred path asks the LLM for structured narrative facts. If the
        model fails or returns malformed JSON, we fall back to the older single
        summary so memory retention remains best-effort instead of all-or-none.
        """
        if not source_turns:
            return None
        if turn_timestamp is None:
            turn_timestamp_text = datetime.now().astimezone().isoformat()
        elif isinstance(turn_timestamp, datetime):
            turn_timestamp_text = turn_timestamp.astimezone().isoformat()
        else:
            turn_timestamp_text = str(turn_timestamp)

        dialogue_batch = self._build_dialogue_batch_for_prompt(
            source_turns,
            fallback_timestamp=turn_timestamp_text,
        )
        prompt = RETAIN_FACT_EXTRACTION_PROMPT.format(
            dialogue_batch=dialogue_batch,
        )

        data: Optional[Dict[str, Any]] = None
        for attempt in range(2):
            result = self._call_llm(prompt)
            data = self._parse_json_object_from_llm_text(result or "")
            if data is not None:
                break
            if attempt == 0:
                logger.debug("Retain fact extraction parse failed, retrying")

        facts: List[Dict[str, Any]] = []
        skipped_low_priority = False
        raw_to_fact_index: Dict[int, int] = {}
        if data is not None and isinstance(data.get("facts"), list):
            for raw_index, raw_fact in enumerate(data.get("facts", [])):
                if not isinstance(raw_fact, dict):
                    continue
                text = str(raw_fact.get("text") or raw_fact.get("summary") or "").strip()
                if not text:
                    continue
                priority = self._normalize_fact_priority(raw_fact.get("priority", 70))
                if priority < 60:
                    skipped_low_priority = True
                    continue
                fact_subject = self._normalize_fact_subject(raw_fact.get("fact_subject", "other"))
                entities = self._fact_entities_with_subject(
                    self._normalize_fact_entities(raw_fact.get("entities", [])),
                    fact_subject,
                )
                primary_entity_candidates = self._normalize_fact_entities(
                    [raw_fact.get("primary_entity")]
                )
                if primary_entity_candidates:
                    primary_entity = primary_entity_candidates[0]
                else:
                    subject_name = self._subject_entity_name(fact_subject)
                    primary_entity = next(
                        (
                            entity
                            for entity in entities
                            if str(entity.get("name") or "").strip() == subject_name
                        ),
                        entities[0] if entities else None,
                    )
                if primary_entity and not any(
                    str(entity.get("name") or "").strip()
                    == str(primary_entity.get("name") or "").strip()
                    for entity in entities
                ):
                    entities = [primary_entity, *entities]
                keywords = self._normalize_keywords(raw_fact.get("keywords", []))
                legacy_topics = self._normalize_keywords(raw_fact.get("topic", []))
                primary_topic = str(
                    raw_fact.get("primary_topic")
                    or (legacy_topics[0] if legacy_topics else "")
                ).strip()
                if not keywords:
                    keywords = [primary_topic] if primary_topic else []
                if not keywords:
                    keywords = [e["name"] for e in entities[:5]]
                if not primary_topic:
                    primary_topic = keywords[0] if keywords else "general"
                task_event_like_raw = raw_fact.get("task_event_like")
                task_event_like: Optional[bool]
                if isinstance(task_event_like_raw, bool):
                    task_event_like = task_event_like_raw
                elif isinstance(task_event_like_raw, (int, float)):
                    task_event_like = bool(task_event_like_raw)
                elif isinstance(task_event_like_raw, str):
                    lowered_action = task_event_like_raw.strip().lower()
                    if lowered_action in {"true", "yes", "1"}:
                        task_event_like = True
                    elif lowered_action in {"false", "no", "0"}:
                        task_event_like = False
                    else:
                        task_event_like = None
                else:
                    task_event_like = None
                task_event_subject = str(raw_fact.get("task_event_subject", "") or "").strip().lower()
                if task_event_subject not in {"user", "assistant", "both", "other"}:
                    task_event_subject = ""
                task_relevance = str(raw_fact.get("task_relevance", "") or "").strip().lower()
                if task_relevance not in {"none", "weak", "medium", "strong"}:
                    task_relevance = ""
                if task_event_like is True and not task_relevance:
                    task_relevance = "medium"
                if task_event_like is False and not task_relevance:
                    task_relevance = "none"
                raw_to_fact_index[raw_index] = len(facts)
                facts.append({
                    "text": text,
                    "keywords": keywords,
                    "primary_entity": primary_entity,
                    "primary_topic": primary_topic,
                    "topic": [primary_topic],
                    "fact_type": self._normalize_fact_type(raw_fact.get("fact_type", "semantic")),
                    "fact_subject": fact_subject,
                    "fact_kind": self._normalize_fact_kind(raw_fact.get("fact_kind", "other")),
                    "priority": priority,
                    "priority_reason": str(raw_fact.get("priority_reason", "") or "").strip(),
                    "task_event_like": task_event_like,
                    "task_event_subject": task_event_subject,
                    "task_relevance": task_relevance,
                    "occurred_start": str(raw_fact.get("occurred_start", "") or "").strip(),
                    "occurred_end": str(raw_fact.get("occurred_end", "") or "").strip(),
                    "time_confidence": self._normalize_time_confidence(raw_fact.get("time_confidence", "unknown")),
                    "where": str(raw_fact.get("where", "") or "").strip(),
                    "entities": entities,
                })

        if not facts:
            if skipped_low_priority:
                return {"facts": [], "causal_relations": []}
            summary_data = self._summarize_turn(source_turns)
            if not summary_data:
                return None
            fallback = self._fallback_fact_from_summary(summary_data)
            if not fallback["text"]:
                return None
            facts = [fallback]
            relations: List[Dict[str, Any]] = []
        else:
            relations = []
            if data is not None and isinstance(data.get("causal_relations"), list):
                for item in data.get("causal_relations", []):
                    if not isinstance(item, dict):
                        continue
                    try:
                        raw_source_index = int(item.get("source_index"))
                        raw_target_index = int(item.get("target_index"))
                    except (TypeError, ValueError):
                        continue
                    source_index = raw_to_fact_index.get(raw_source_index)
                    target_index = raw_to_fact_index.get(raw_target_index)
                    if source_index is None or target_index is None:
                        continue
                    relation = str(item.get("relation", "") or "").strip()
                    if not relation or relation == "None":
                        continue
                    if source_index == target_index:
                        continue
                    if not (0 <= source_index < len(facts) and 0 <= target_index < len(facts)):
                        continue
                    try:
                        confidence = float(item.get("confidence", 1.0) or 1.0)
                    except (TypeError, ValueError):
                        confidence = 1.0
                    relations.append({
                        "source_index": source_index,
                        "target_index": target_index,
                        "relation": relation,
                        "confidence": confidence,
                    })

        return {"facts": facts, "causal_relations": relations}

    # ── Causal relation extraction ───────────────────────────────────────

    def _extract_causal_relations(
        self,
        cur_summary: str,
        similar_nodes: List[Dict[str, Any]],
    ) -> List[Optional[str]]:
        """Determine causal relations between a new summary and similar existing nodes.

        For each similar node, calls the LLM API with the relation prompt template
        to determine the type of relation (Cause/Want/React/Changed/SameTopic/None).

        Follows the same approach as AI_Glass_Agent's ``_extract_causal_relations``.

        Args:
            cur_summary: The summary of the new (current) conversation turn.
            similar_nodes:  List of existing memory node dicts, each with at
                least a ``"summary"`` key.

        Returns:
            A list of relation strings (same length as *similar_nodes*).
            ``None`` indicates no causal relation was found.
        """
        if not similar_nodes:
            return []

        relations: List[Optional[str]] = []

        for node in similar_nodes:
            prompt = RELATION_PROMPT_TEMPLATE.format(
                summary1=cur_summary,
                summary2=node.get("summary", ""),
            )
            result = self._call_llm(prompt)

            if not result:
                relations.append(None)
                continue

            try:
                text = result.strip()
                # Strip code fences if present
                if "```" in text:
                    start = text.find("{")
                    end = text.rfind("}")
                    if start != -1 and end != -1:
                        text = text[start:end + 1]
                data = json.loads(text)
                relation = data.get("relation", "None")
                if relation == "None":
                    relations.append(None)
                else:
                    relations.append(relation)
            except (json.JSONDecodeError, KeyError):
                logger.debug("Failed to parse causal relation: %.120s", result)
                relations.append(None)

        return relations

    # ── Entity extraction (lazy init) ─────────────────────────────────────

    def _ensure_entity_extractor(self) -> Any:
        """Lazy-init and return EntityExtractor, or None if unavailable."""
        if not self._enable_entity_extraction or not self._db:
            return None
        try:
            from agent.entity_extractor import EntityExtractor
            return EntityExtractor(
                session_db=self._db,
                llm_client=self._llm_client,
                llm_model=self._llm_model,
                llm_base_url=self._llm_base_url,
                llm_api_key=self._llm_api_key,
                llm_timeout=self._llm_timeout,
            )
        except Exception as e:
            logger.debug("EntityExtractor unavailable: %s", e)
            return None

    @staticmethod
    def _memory_time_key(fact_index: int = 0, turn_timestamp: Optional[Any] = None) -> str:
        """Return a lexicographically sortable, unique-ish local timestamp key."""
        if turn_timestamp is None:
            now = datetime.now().astimezone()
        elif isinstance(turn_timestamp, datetime):
            now = turn_timestamp.astimezone()
        else:
            try:
                now = datetime.fromisoformat(str(turn_timestamp).replace("Z", "+00:00")).astimezone()
            except ValueError:
                now = datetime.now().astimezone()
        base = now.strftime("%Y-%m-%d %H:%M:%S.%f")
        offset = now.strftime("%z")
        if len(offset) == 5:
            offset = f"{offset[:3]}:{offset[3:]}"
        return f"{base}{offset}#{fact_index:02d}"

    @staticmethod
    def _build_original_dialog_payload(
        fact: Dict[str, Any],
        source_turns: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Store source dialog plus structured retain metadata in one field.

        The current DB schema has no dedicated metadata column for memory
        nodes, so retain metadata is encoded alongside the source transcript in
        a JSON payload. Existing readers treat this as opaque text.
        """
        payload = {
            "source_dialog": {
                "turns": [
                    {
                        "user_message": str(turn.get("user_message") or ""),
                        "assistant_response": str(
                            turn.get("assistant_response") or ""
                        ),
                        "turn_timestamp": (
                            turn.get("turn_timestamp").isoformat()
                            if isinstance(turn.get("turn_timestamp"), datetime)
                            else str(turn.get("turn_timestamp") or "")
                        ),
                        "tags": list(turn.get("tags") or []),
                    }
                    for turn in (source_turns or [])
                ],
            },
            "retain_fact": {
                "text": fact.get("text", ""),
                "fact_type": MemoryNodeManager._normalize_fact_type(fact.get("fact_type", "semantic")),
                "fact_subject": MemoryNodeManager._normalize_fact_subject(fact.get("fact_subject", "other")),
                "fact_kind": fact.get("fact_kind", "other"),
                "priority": fact.get("priority", 70),
                "priority_reason": fact.get("priority_reason", ""),
                "task_event_like": fact.get("task_event_like"),
                "task_event_subject": fact.get("task_event_subject", ""),
                "task_relevance": fact.get("task_relevance", ""),
                "primary_entity": fact.get("primary_entity"),
                "primary_topic": fact.get("primary_topic", "general"),
                "topic": fact.get("topic", [fact.get("primary_topic", "general")]),
                "occurred_start": fact.get("occurred_start", ""),
                "occurred_end": fact.get("occurred_end", ""),
                "time_confidence": fact.get("time_confidence", "unknown"),
                "where": fact.get("where", ""),
                "entities": fact.get("entities", []),
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    def _fact_tags(
        self,
        fact: Dict[str, Any],
        tags: Optional[List[str]] = None,
    ) -> List[str]:
        out: List[str] = []
        for tag in tags or []:
            if tag and tag not in out:
                out.append(tag)
        for tag in (
            f"fact_type:{self._normalize_fact_type(fact.get('fact_type', 'semantic'))}",
            f"fact_subject:{self._normalize_fact_subject(fact.get('fact_subject', 'other'))}",
            f"fact_kind:{fact.get('fact_kind', 'other')}",
            f"priority:{self._normalize_fact_priority(fact.get('priority', 70))}",
            "source:memory_node_manager",
        ):
            if tag not in out:
                out.append(tag)
        priority = self._normalize_fact_priority(fact.get("priority", 70))
        if priority >= 80:
            priority_band = "high"
        elif priority >= 60:
            priority_band = "medium"
        else:
            priority_band = "low"
        priority_band_tag = f"priority_band:{priority_band}"
        if priority_band_tag not in out:
            out.append(priority_band_tag)
        time_confidence = self._normalize_time_confidence(fact.get("time_confidence", "unknown"))
        time_confidence_tag = f"time_confidence:{time_confidence}"
        if time_confidence_tag not in out:
            out.append(time_confidence_tag)
        if fact.get("task_event_like") is True and "task_event_like:true" not in out:
            out.append("task_event_like:true")
        task_event_subject = str(fact.get("task_event_subject") or "").strip()
        if task_event_subject:
            tag = f"task_event_subject:{task_event_subject}"
            if tag not in out:
                out.append(tag)
        task_relevance = str(fact.get("task_relevance") or "").strip()
        if task_relevance:
            tag = f"task_relevance:{task_relevance}"
            if tag not in out:
                out.append(tag)
        return out

    def _link_fact_entities(self, fact_id: int, entities: List[Dict[str, str]]) -> List[Tuple[int, str]]:
        linked_entities: List[Tuple[int, str]] = []
        if not entities or not self._db:
            return linked_entities
        for entity in entities:
            name = entity.get("name", "").strip()
            if not name:
                continue
            etype = entity.get("type", "CONCEPT").strip().upper() or "CONCEPT"
            try:
                entity_id = self._db.entity_add_entity(name=name, entity_type=etype)
                self._db.entity_link_fact(fact_id, entity_id)
                linked_entities.append((entity_id, name))
            except Exception as exc:
                logger.debug("Failed to link retain entity %r to fact %d: %s", name, fact_id, exc)
        return linked_entities

    @staticmethod
    def _topic_key(topic: Any) -> str:
        text = str(topic or "").strip().lower()
        text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "-", text).strip("-")
        return text or "general"

    @classmethod
    def _topic_keys(cls, topics: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for topic in topics:
            topic_key = cls._topic_key(topic)
            if not topic_key or topic_key in seen:
                continue
            seen.add(topic_key)
            out.append(topic_key)
        return out or ["general"]

    @classmethod
    def _generalize_topic_key(cls, topic: Any) -> str:
        """Remove only safe, weak suffixes from a topic key."""
        topic_key = cls._topic_key(topic)
        for suffix in EVIDENCE_BUNDLE_WEAK_TOPIC_SUFFIXES:
            if not topic_key.endswith(suffix):
                continue
            candidate = topic_key[:-len(suffix)].rstrip("-")
            if candidate.endswith("的"):
                candidate = candidate[:-1].rstrip("-")
            if (
                len(candidate) >= 2
                and candidate not in EVIDENCE_BUNDLE_GENERIC_TOPICS
            ):
                return candidate
        return topic_key

    @classmethod
    def _topic_embedding_match_allowed(
        cls,
        left_topic: Any,
        right_topic: Any,
    ) -> bool:
        """Require a lexical anchor and reject broad-to-specific absorption."""
        left = cls._generalize_topic_key(left_topic)
        right = cls._generalize_topic_key(right_topic)
        if left == right:
            return True
        if (
            left in EVIDENCE_BUNDLE_GENERIC_TOPICS
            or right in EVIDENCE_BUNDLE_GENERIC_TOPICS
        ):
            return False
        left_chars = {char for char in left if char != "-"}
        right_chars = {char for char in right if char != "-"}
        if not left_chars or not right_chars:
            return False
        lexical_affinity = len(left_chars & right_chars) / len(
            left_chars | right_chars
        )
        return lexical_affinity >= 0.5

    def _topic_only_embedding(
        self,
        topic: Any,
        cache: Dict[str, Optional[np.ndarray]],
    ) -> Optional[np.ndarray]:
        topic_key = self._topic_key(topic)
        if topic_key not in cache:
            cache[topic_key] = self._as_embedding_vector(
                self._embed_memory_layer_text(topic_key)
            )
        return cache[topic_key]

    def _canonical_topic_entries_for_entity(
        self,
        entity_id: int,
        embedding_cache: Dict[str, Optional[np.ndarray]],
    ) -> List[Dict[str, Any]]:
        if not self._db:
            return []
        try:
            bundles = self._db.get_evidence_bundles_for_entity(entity_id)
        except Exception:
            return []
        entries: List[Dict[str, Any]] = []
        for bundle in bundles:
            topic_key = self._topic_key(bundle.get("topic_key") or "general")
            topic_label = str(
                bundle.get("topic_label") or topic_key
            ).strip() or topic_key
            metadata = self._json_dict(bundle.get("metadata", {}))
            aliases = {
                str(alias).strip()
                for alias in metadata.get("topic_aliases", [])
                if str(alias or "").strip()
            }
            aliases.add(topic_label)
            embedding = self._as_embedding_vector(
                bundle.get("canonical_topic_embedding")
            )
            if embedding is None:
                embedding = self._topic_only_embedding(
                    topic_label,
                    embedding_cache,
                )
                if embedding is not None:
                    metadata.update({
                        "canonical_topic": topic_label,
                        "topic_aliases": sorted(aliases),
                        "topic_embedding_text": topic_label,
                    })
                    try:
                        self._db.memory_update_evidence_bundle_topic(
                            int(bundle["id"]),
                            canonical_topic_embedding=embedding,
                            metadata=metadata,
                        )
                    except Exception:
                        pass
            entries.append({
                "bundle_id": int(bundle["id"]),
                "topic_key": topic_key,
                "topic_label": topic_label,
                "embedding": embedding,
                "aliases": aliases,
            })
        return entries

    def _canonicalize_topics_for_entity(
        self,
        entity_id: int,
        raw_topics: List[str],
        embedding_cache: Dict[str, Optional[np.ndarray]],
    ) -> Dict[str, Dict[str, Any]]:
        """Resolve raw topics to stable existing or provisional canonical topics."""
        entries = self._canonical_topic_entries_for_entity(
            entity_id,
            embedding_cache,
        )
        exact_lookup: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            exact_lookup.setdefault(entry["topic_key"], entry)
            for alias in entry["aliases"]:
                exact_lookup.setdefault(self._topic_key(alias), entry)

        topic_counts = Counter(
            self._topic_key(topic)
            for topic in raw_topics
            if str(topic or "").strip()
        )
        raw_labels: Dict[str, str] = {}
        for topic in raw_topics:
            raw_label = str(topic or "").strip()
            if raw_label:
                raw_labels.setdefault(self._topic_key(raw_label), raw_label)

        resolved: Dict[str, Dict[str, Any]] = {}
        ordered_raw_keys = sorted(
            topic_counts,
            key=lambda key: (
                -topic_counts[key],
                len(self._generalize_topic_key(key)),
                key,
            ),
        )
        for raw_key in ordered_raw_keys:
            raw_label = raw_labels.get(raw_key, raw_key)
            generalized_key = self._generalize_topic_key(raw_key)
            entry = exact_lookup.get(raw_key)
            match_reason = "exact_existing_topic"
            if entry is None:
                entry = exact_lookup.get(generalized_key)
                match_reason = "generalized_exact_topic"

            topic_embedding = self._topic_only_embedding(
                generalized_key,
                embedding_cache,
            )
            similarity = 1.0 if entry is not None else 0.0
            if entry is None and topic_embedding is not None:
                scored_entries = [
                    (
                        self._cal_embedding_similarity(
                            topic_embedding,
                            candidate.get("embedding"),
                        ),
                        candidate,
                    )
                    for candidate in entries
                    if candidate.get("embedding") is not None
                    and self._topic_embedding_match_allowed(
                        generalized_key,
                        candidate.get("topic_key"),
                    )
                ]
                scored_entries.sort(key=lambda item: item[0], reverse=True)
                if (
                    scored_entries
                    and scored_entries[0][0]
                    >= EVIDENCE_BUNDLE_TOPIC_SIMILARITY_THRESHOLD
                ):
                    similarity, entry = scored_entries[0]
                    match_reason = "topic_embedding"

            if entry is None:
                canonical_key = generalized_key
                entry = {
                    "bundle_id": None,
                    "topic_key": canonical_key,
                    "topic_label": canonical_key,
                    "embedding": topic_embedding,
                    "aliases": set(),
                }
                entries.append(entry)
                exact_lookup.setdefault(canonical_key, entry)
                match_reason = (
                    "weak_suffix_generalization"
                    if canonical_key != raw_key
                    else "new_canonical_topic"
                )

            entry["aliases"].add(raw_label)
            exact_lookup.setdefault(raw_key, entry)
            resolved[raw_key] = {
                "topic_key": entry["topic_key"],
                "topic_label": entry["topic_label"],
                "canonical_topic_embedding": entry.get("embedding"),
                "topic_alias": raw_label,
                "topic_match_reason": match_reason,
                "topic_similarity": similarity,
            }
        return resolved

    @staticmethod
    def _string_list(value: Any, *, limit: int = 5) -> List[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        out = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _reflect_log_text(value: Any, *, limit: int = 500) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return f"{text[:limit]}..."

    @classmethod
    def _fact_entity_items(cls, fact: Dict[str, Any]) -> List[Dict[str, Any]]:
        entity_items: List[Dict[str, Any]] = []
        linked_entities = fact.get("linked_entities", [])
        if not isinstance(linked_entities, list):
            linked_entities = []
        for entity in linked_entities:
            if isinstance(entity, dict):
                entity_items.append({
                    "id": entity.get("id") or entity.get("entity_id"),
                    "name": entity.get("name") or entity.get("entity_name"),
                })
            elif isinstance(entity, (list, tuple)) and len(entity) >= 2:
                entity_items.append({"id": entity[0], "name": entity[1]})
        return entity_items

    @classmethod
    def _fact_entity_pairs(cls, fact: Dict[str, Any]) -> List[Tuple[int, str]]:
        pairs: List[Tuple[int, str]] = []
        for item in cls._fact_entity_items(fact):
            try:
                entity_id = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            name = str(item.get("name") or "").strip()
            pairs.append((entity_id, name))
        return pairs

    @classmethod
    def _reflect_fact_log_items(
        cls,
        facts: List[Dict[str, Any]],
        *,
        limit: int = 12,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for fact in facts[:limit]:
            items.append({
                "fact_id": fact.get("fact_id", fact.get("id")),
                "time_key": fact.get("time_key"),
                "fact_type": fact.get("fact_type"),
                "fact_subject": fact.get("fact_subject"),
                "fact_kind": fact.get("fact_kind"),
                "summary": cls._reflect_log_text(fact.get("summary")),
                "primary_entity_id": fact.get("primary_entity_id"),
                "primary_entity_name": fact.get("primary_entity_name"),
                "primary_topic": fact.get("primary_topic"),
                "topics": fact.get("topics", []),
                "keywords": fact.get("keywords", []),
                "task_event_like": fact.get("task_event_like"),
                "task_event_subject": fact.get("task_event_subject"),
                "task_relevance": fact.get("task_relevance"),
                "linked_entities": cls._fact_entity_items(fact),
            })
        return items

    @classmethod
    def _reflect_observation_log_item(cls, observation: Dict[str, Any]) -> Dict[str, Any]:
        metadata = observation.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata or "{}")
            except (TypeError, ValueError):
                metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        return {
            "id": observation.get("id"),
            "entity_id": observation.get("entity_id"),
            "entity_name": observation.get("entity_name"),
            "topic_key": observation.get("topic_key"),
            "topic_label": observation.get("topic_label"),
            "observation_type": observation.get("observation_type"),
            "summary": cls._reflect_log_text(observation.get("summary")),
            "keywords": observation.get("keywords"),
            "confidence": observation.get("confidence"),
            "status": observation.get("status"),
            "metadata": metadata,
        }

    @classmethod
    def _reflect_evidence_bundle_log_item(
        cls,
        evidence_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        metadata = evidence_bundle.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata or "{}")
            except (TypeError, ValueError):
                metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        return {
            "id": evidence_bundle.get("id"),
            "entity_id": evidence_bundle.get("entity_id"),
            "entity_name": evidence_bundle.get("entity_name"),
            "topic_key": evidence_bundle.get("topic_key"),
            "topic_label": evidence_bundle.get("topic_label"),
            "bundle_type": evidence_bundle.get("bundle_type"),
            "source_time_start": evidence_bundle.get("source_time_start"),
            "source_time_end": evidence_bundle.get("source_time_end"),
            "metadata": metadata,
        }

    @classmethod
    def _log_info(cls, scope: str, event: str, payload: Dict[str, Any]) -> None:
        record = {
            "scope": scope,
            "event": event,
            "payload": payload,
        }
        try:
            body = json.dumps(record, ensure_ascii=False, sort_keys=False, indent=2, default=str)
        except (TypeError, ValueError):
            body = json.dumps({
                "scope": scope,
                "event": event,
                "payload": str(payload),
            }, ensure_ascii=False, sort_keys=True, indent=2)
        logger.error("\n%s", body)

    @classmethod
    def _normalize_task_steps(cls, value: Any, *, limit: int = 12) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        allowed_statuses = {"todo", "active", "done", "blocked", "skipped"}
        steps: List[Dict[str, Any]] = []
        seen_titles = set()
        for raw_step in value:
            if isinstance(raw_step, str):
                raw_step = {"title": raw_step}
            if not isinstance(raw_step, dict):
                continue
            title = str(raw_step.get("title") or "").strip()
            if not title:
                continue
            title_key = title.casefold()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            status = str(raw_step.get("status", "active") or "active").strip().lower()
            if status not in allowed_statuses:
                status = "active"
            step: Dict[str, Any] = {
                "title": title,
                "status": status,
            }
            evidence = cls._string_list(raw_step.get("evidence", []), limit=5)
            if evidence:
                step["evidence"] = evidence
            updated_at = str(raw_step.get("updated_at") or "").strip()
            if updated_at:
                step["updated_at"] = updated_at[:32]
            notes = str(raw_step.get("notes") or "").strip()
            if notes:
                step["notes"] = notes
            steps.append(step)
            if len(steps) >= limit:
                break
        return steps

    @classmethod
    def _normalize_task_metadata(
        cls,
        metadata: Dict[str, Any],
        *,
        allow_stale: bool = False,
    ) -> Dict[str, Any]:
        task_status = str(metadata.get("task_status", "active") or "active").strip().lower()
        allowed_statuses = {"active", "blocked", "paused"}
        if allow_stale:
            allowed_statuses.add("stale")
        if task_status not in allowed_statuses:
            task_status = "active"

        normalized: Dict[str, Any] = {
            "task_status": task_status,
            "task_source": "inferred_from_interpretation",
        }
        goal = str(metadata.get("goal") or "").strip()
        if goal:
            normalized["goal"] = goal
        evidence = cls._string_list(metadata.get("evidence", []), limit=5)
        if evidence:
            normalized["evidence"] = evidence
        steps = cls._normalize_task_steps(metadata.get("steps", []), limit=12)
        if steps:
            normalized["steps"] = steps
        next_action = str(metadata.get("next_action") or "").strip()
        if next_action:
            normalized["next_action"] = next_action
        return normalized

    @staticmethod
    def _json_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str):
            try:
                data = json.loads(value or "{}")
            except (TypeError, ValueError):
                return {}
            return data if isinstance(data, dict) else {}
        return {}

    @classmethod
    def _task_status(cls, task: Dict[str, Any]) -> str:
        metadata = cls._json_dict(task.get("metadata", {}))
        status = str(metadata.get("task_status", "active") or "active").strip().lower()
        if status not in {"active", "blocked", "paused", "stale"}:
            status = "active"
        return status

    @classmethod
    def _task_profile_text(cls, task: Dict[str, Any]) -> str:
        metadata = cls._json_dict(task.get("metadata", {}))
        lines = [
            f"Task summary: {task.get('summary', '')}",
            f"Goal: {metadata.get('goal', '')}",
            f"Current status: {metadata.get('task_status', 'active')}",
        ]
        steps = metadata.get("steps", [])
        if isinstance(steps, list) and steps:
            lines.append("Steps:")
            for step in steps[:12]:
                if isinstance(step, dict):
                    title = str(step.get("title") or "").strip()
                    status = str(step.get("status") or "active").strip()
                else:
                    title = str(step or "").strip()
                    status = "active"
                if title:
                    lines.append(f"- {status}: {title}")
        next_action = str(metadata.get("next_action") or "").strip()
        if next_action:
            lines.append(f"Next action: {next_action}")
        lines.extend([
            f"Keywords: {task.get('keywords', '')}",
            f"Entity: {task.get('entity_name', '')}",
            f"Topic: {task.get('topic_label') or task.get('topic_key', '')}",
        ])
        return "\n".join(line for line in lines if str(line).strip())

    @staticmethod
    def _as_embedding_vector(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        try:
            if isinstance(value, (bytes, bytearray, memoryview)):
                arr = np.frombuffer(bytes(value), dtype=np.float32)
            else:
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        if arr.size == 0:
            return None
        norm = float(np.linalg.norm(arr))
        if norm <= 0:
            return None
        return arr / norm

    @classmethod
    def _cal_embedding_similarity(cls, left: Any, right: Any) -> float:
        left_vec = cls._as_embedding_vector(left)
        right_vec = cls._as_embedding_vector(right)
        if left_vec is None or right_vec is None:
            return 0.0
        if left_vec.shape != right_vec.shape:
            return 0.0
        return float(np.dot(left_vec, right_vec))

    @classmethod
    def _embedding_centroid(
        cls,
        embeddings: Iterable[Any],
    ) -> Optional[np.ndarray]:
        vectors = [
            vector
            for value in embeddings
            if (vector := cls._as_embedding_vector(value)) is not None
        ]
        if not vectors:
            return None
        first_shape = vectors[0].shape
        compatible = [vector for vector in vectors if vector.shape == first_shape]
        if not compatible:
            return None
        return cls._as_embedding_vector(
            np.mean(np.stack(compatible), axis=0)
        )

    def _evidence_centroid_for_sources(
        self,
        source_facts: List[Dict[str, Any]],
    ) -> Optional[np.ndarray]:
        source_ids = [
            int(self._fact_id(fact))
            for fact in source_facts
            if self._fact_id(fact) is not None
        ]
        embeddings = self._db.memory_fact_embeddings(source_ids)
        return self._embedding_centroid(
            embeddings.get(source_id)
            for source_id in source_ids
        )

    @classmethod
    def _score_fact_cluster_against_observation_evidence(
        cls,
        cluster_source_fact_ids: List[int],
        cluster_centroid: Any,
        observation: Dict[str, Any],
        fact_embeddings: Dict[int, np.ndarray],
    ) -> Tuple[float, float, float, float]:
        """Score how completely one fact cluster supports an observation."""
        cluster_embeddings = [
            fact_embeddings.get(int(source_fact_id))
            for source_fact_id in cluster_source_fact_ids
        ]
        cluster_embeddings = [
            embedding
            for embedding in cluster_embeddings
            if cls._as_embedding_vector(embedding) is not None
        ]
        observation_source_embeddings = [
            fact_embeddings.get(int(source_fact_id))
            for source_fact_id in observation.get("source_fact_ids", [])
        ]
        observation_source_embeddings = [
            embedding
            for embedding in observation_source_embeddings
            if cls._as_embedding_vector(embedding) is not None
        ]
        observation_centroid = observation.get(
            "evidence_centroid_embedding"
        )
        if observation_centroid is None:
            observation_centroid = cls._embedding_centroid(
                observation_source_embeddings
            )
        normalized_cluster_centroid = cls._as_embedding_vector(
            cluster_centroid
        )
        if normalized_cluster_centroid is None:
            normalized_cluster_centroid = cls._embedding_centroid(
                cluster_embeddings
            )
        centroid_similarity = cls._cal_embedding_similarity(
            normalized_cluster_centroid,
            observation_centroid,
        )
        cross_source_similarities = [
            cls._cal_embedding_similarity(
                cluster_embedding,
                observation_source_embedding,
            )
            for cluster_embedding in cluster_embeddings
            for observation_source_embedding in observation_source_embeddings
        ]
        max_source_similarity = max(
            cross_source_similarities,
            default=0.0,
        )
        fact_coverage_similarities = [
            max(
                cls._cal_embedding_similarity(
                    cluster_embedding,
                    observation_centroid,
                ),
                max(
                    (
                        cls._cal_embedding_similarity(
                            cluster_embedding,
                            observation_source_embedding,
                        )
                        for observation_source_embedding
                        in observation_source_embeddings
                    ),
                    default=0.0,
                ),
            )
            for cluster_embedding in cluster_embeddings
        ]
        coverage_similarity = (
            sum(fact_coverage_similarities)
            / len(fact_coverage_similarities)
            if fact_coverage_similarities
            else 0.0
        )
        score = (
            OBSERVATION_CLUSTER_CENTROID_WEIGHT * centroid_similarity
            + OBSERVATION_CLUSTER_MAX_SOURCE_WEIGHT * max_source_similarity
            + OBSERVATION_CLUSTER_COVERAGE_WEIGHT * coverage_similarity
        )
        return (
            score,
            centroid_similarity,
            max_source_similarity,
            coverage_similarity,
        )

    @staticmethod
    def _fact_match_text(fact: Dict[str, Any]) -> str:
        keywords = fact.get("keywords", [])
        topics = fact.get("topics", [])
        if not isinstance(keywords, list):
            keywords = [keywords]
        if not isinstance(topics, list):
            topics = [topics]
        return "\n".join([
            f"Fact: {fact.get('summary', '')}",
            f"Keywords: {' '.join(str(item) for item in keywords if str(item or '').strip())}",
            f"Topics: {' '.join(str(item) for item in topics if str(item or '').strip())}",
        ])

    @staticmethod
    def _fact_kind_excluded_from_task(fact: Dict[str, Any]) -> bool:
        fact_kind = str(fact.get("fact_kind") or "").strip().lower()
        return fact_kind in {"preference", "instruction", "context", "other"}

    @staticmethod
    def _fact_id(value: Dict[str, Any]) -> Optional[int]:
        raw = value.get("fact_id", value.get("id"))
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _metadata_candidate_types(value: Any) -> List[str]:
        if isinstance(value, str):
            raw = re.split(r"[,/|\s]+", value)
        elif isinstance(value, list):
            raw = value
        else:
            raw = []
        out: List[str] = []
        for item in raw:
            text = str(item or "").strip().lower().replace("-", "_").replace(" ", "_")
            if text in {"insight", "task", "preference"} and text not in out:
                out.append(text)
        return out

    @classmethod
    def _metadata_fact_type_distribution(cls, value: Any) -> Dict[str, int]:
        if isinstance(value, str):
            try:
                value = json.loads(value or "{}")
            except (TypeError, ValueError):
                value = {}
        if not isinstance(value, dict):
            return {"semantic": 0, "episodic": 0}
        out = {"semantic": 0, "episodic": 0}
        for key in ("semantic", "episodic"):
            try:
                out[key] = max(0, int(value.get(key, 0) or 0))
            except (TypeError, ValueError):
                out[key] = 0
        return out

    @classmethod
    def _fact_type_distribution_from_facts(cls, facts: List[Dict[str, Any]]) -> Dict[str, int]:
        out = {"semantic": 0, "episodic": 0}
        for fact in facts or []:
            fact_type = cls._normalize_fact_type(fact.get("fact_type", "semantic"))
            out[fact_type] = out.get(fact_type, 0) + 1
        return out

    @staticmethod
    def _fact_type_evidence_summary(distribution: Dict[str, int]) -> Tuple[str, str]:
        semantic = max(0, int(distribution.get("semantic", 0) or 0))
        episodic = max(0, int(distribution.get("episodic", 0) or 0))
        if semantic <= 0 and episodic <= 0:
            return "unknown", "unknown"
        if semantic > 0 and episodic <= 0:
            return "semantic", "semantic_only"
        if episodic > 0 and semantic <= 0:
            return "episodic", "episodic_only"
        if semantic == episodic:
            return "mixed", "balanced_mixed"
        if semantic > episodic:
            return "semantic", "semantic_dominant"
        return "episodic", "episodic_dominant"

    @classmethod
    def _implicit_observation_type_for_fact(cls, fact: Dict[str, Any]) -> str:
        """Map fact evidence semantics to the claim it can directly support."""
        fact_kind = str(fact.get("fact_kind") or "other").strip().lower()
        fact_type = cls._normalize_fact_type(fact.get("fact_type", "semantic"))
        if fact_kind == "instruction":
            return "constraint"
        if fact_kind == "preference":
            return "preference_signal"
        if fact_kind == "error":
            return "problem"
        if fact_kind == "decision":
            return "decision"
        if fact_kind == "recommendation":
            return "strategy"
        if fact_kind == "request":
            return "task_state"
        if fact_kind == "action":
            if cls._is_task_event_like_fact(fact) or fact_type == "episodic":
                return "task_progress"
            return "behavior_pattern"
        if cls._is_task_event_like_fact(fact):
            return "task_state"
        return "context"

    @staticmethod
    def _observation_type_match_rule(
        left_type: str,
        right_type: str,
    ) -> Tuple[str, Optional[float]]:
        """Return the compatibility class and similarity threshold."""
        left = str(left_type or "context").strip().lower()
        right = str(right_type or "context").strip().lower()
        if left == right:
            return "exact", OBSERVATION_EXACT_TYPE_SIMILARITY_THRESHOLD
        for compatible_types in OBSERVATION_TYPE_COMPATIBILITY_GROUPS.values():
            if left in compatible_types and right in compatible_types:
                return (
                    "compatible",
                    OBSERVATION_COMPATIBLE_TYPE_SIMILARITY_THRESHOLD,
                )
        return "incompatible", None

    @classmethod
    def _observation_type_for_fact_cluster(
        cls,
        facts: List[Dict[str, Any]],
    ) -> str:
        """Choose a stable claim type from all facts in a semantic cluster."""
        candidate_types = [
            cls._implicit_observation_type_for_fact(fact)
            for fact in facts
        ]
        if not candidate_types:
            return "context"
        return max(
            candidate_types,
            key=lambda observation_type: (
                OBSERVATION_TYPE_PRIORITY.get(observation_type, 0),
                candidate_types.count(observation_type),
            ),
        )

    @staticmethod
    def _observation_type_prompt_guidance(observation_type: str) -> str:
        normalized_type = str(
            observation_type or "context"
        ).strip().lower()
        return OBSERVATION_TYPE_GUIDANCE.get(
            normalized_type,
            OBSERVATION_TYPE_GUIDANCE["context"],
        )

    @staticmethod
    def _observation_candidate_families(observation_type: str) -> List[str]:
        allowed = OBSERVATION_INTERPRETATION_TYPES.get(
            str(observation_type or "context"),
            {"insight"},
        )
        families: List[str] = []
        if allowed & {"task", "project_state", "task_risk"}:
            families.append("task")
        if allowed & {
            "explicit_preference",
            "explicit_instruction",
            "inferred_preference",
            "behavior_pattern",
        }:
            families.append("preference")
        if allowed & {"insight", "strategy", "constraint", "conflict_resolution"}:
            families.append("insight")
        return families or ["insight"]

    @staticmethod
    def _observation_allowed_interpretation_types(
        observation_type: str,
        evidence_mode: str,
    ) -> List[str]:
        allowed = set(OBSERVATION_INTERPRETATION_TYPES.get(
            str(observation_type or "context"),
            {"insight"},
        ))
        if observation_type == "preference_signal":
            if evidence_mode == "explicit":
                allowed = {"explicit_preference"}
            else:
                allowed = {"inferred_preference"}
        return sorted(allowed)

    @classmethod
    def _observation_evidence_mode(
        cls,
        observation_type: str,
        source_facts: List[Dict[str, Any]],
    ) -> str:
        kinds = {
            str(fact.get("fact_kind") or "other").strip().lower()
            for fact in source_facts
        }
        fact_types = {
            cls._normalize_fact_type(fact.get("fact_type", "semantic"))
            for fact in source_facts
        }
        if observation_type == "constraint" and "instruction" in kinds:
            return "explicit"
        if (
            observation_type == "preference_signal"
            and "preference" in kinds
            and "semantic" in fact_types
        ):
            return "explicit"
        if observation_type == "behavior_pattern" or (
            observation_type == "preference_signal"
            and len(source_facts) >= 2
            and fact_types == {"episodic"}
        ):
            return "behavioral"
        if len(source_facts) >= 2:
            return "aggregated"
        if fact_types == {"episodic"}:
            return "episodic"
        return "semantic"

    def _cluster_evidence_bundle_facts_into_observations(
        self,
        source_facts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build typed semantic fact clusters inside one evidence bundle."""
        facts = [
            dict(fact)
            for fact in source_facts
            if self._fact_id(fact) is not None
            and str(fact.get("summary") or "").strip()
        ]
        if not facts:
            return []
        fact_ids = [int(self._fact_id(fact)) for fact in facts]
        embeddings = self._db.memory_fact_embeddings(fact_ids)
        clusters: List[Dict[str, Any]] = []
        for fact in facts:
            fact_id = int(self._fact_id(fact))
            implicit_observation_type = self._implicit_observation_type_for_fact(fact)
            fact_embedding_vector = self._as_embedding_vector(embeddings.get(fact_id))
            best_cluster = None
            best_similarity = -1.0
            best_match_rule = "incompatible"
            best_pair_similarity_sum = 0.0
            best_pair_count = 0
            for cluster in clusters:
                match_rule, threshold = self._observation_type_match_rule(
                    implicit_observation_type,
                    cluster["observation_type"],
                )
                if threshold is None:
                    continue
                similarity = self._cal_embedding_similarity(
                    fact_embedding_vector,
                    cluster.get("centroid"),
                )
                if similarity < threshold:
                    continue
                new_pair_similarities = [
                    self._cal_embedding_similarity(fact_embedding_vector, existing_vector)
                    for existing_vector in cluster["vectors"]
                ] if fact_embedding_vector is not None else []
                projected_pair_similarity_sum = (
                    cluster["pair_similarity_sum"]
                    + sum(new_pair_similarities)
                )
                projected_pair_count = (
                    cluster["pair_count"] + len(new_pair_similarities)
                )
                semantic_cohesion = (
                    projected_pair_similarity_sum / projected_pair_count
                    if projected_pair_count
                    else 1.0
                )
                if (
                    semantic_cohesion
                    < OBSERVATION_CLUSTER_MIN_SEMANTIC_COHESION
                ):
                    continue
                if (
                    similarity > best_similarity
                    or (
                        similarity == best_similarity
                        and match_rule == "exact"
                        and best_match_rule != "exact"
                    )
                ):
                    best_cluster = cluster
                    best_similarity = similarity
                    best_match_rule = match_rule
                    best_pair_similarity_sum = projected_pair_similarity_sum
                    best_pair_count = projected_pair_count
            if best_cluster is None:
                clusters.append({
                    "observation_type": implicit_observation_type,
                    "source_facts": [fact],
                    "vectors": [fact_embedding_vector] if fact_embedding_vector is not None else [],
                    "centroid": fact_embedding_vector,
                    "pair_similarity_sum": 0.0,
                    "pair_count": 0,
                })
                continue
            best_cluster["source_facts"].append(fact)
            best_cluster["pair_similarity_sum"] = best_pair_similarity_sum
            best_cluster["pair_count"] = best_pair_count
            if fact_embedding_vector is not None:
                best_cluster["vectors"].append(fact_embedding_vector)
                centroid = np.mean(
                    np.stack(best_cluster["vectors"]),
                    axis=0,
                )
                best_cluster["centroid"] = self._as_embedding_vector(centroid)
            best_cluster["observation_type"] = (
                self._observation_type_for_fact_cluster(
                    best_cluster["source_facts"]
                )
            )

        candidate_fact_clusters: List[Dict[str, Any]] = []
        for cluster in clusters:
            cluster_facts = cluster["source_facts"]
            centroid = cluster.get("centroid")
            candidate_fact_clusters.append({
                "observation_type": cluster["observation_type"],
                "source_fact_ids": [
                    int(self._fact_id(fact)) for fact in cluster_facts
                ],
                "evidence_centroid_embedding": centroid,
            })
        return candidate_fact_clusters

    @staticmethod
    def _observation_fact_payload(source_facts: List[Dict[str, Any]]) -> str:
        return json.dumps(
            [
                {
                    "fact_id": fact.get("id", fact.get("fact_id")),
                    "fact_type": fact.get("fact_type", "semantic"),
                    "fact_kind": fact.get("fact_kind", "other"),
                    "fact_text": str(fact.get("summary") or "").strip(),
                    "timestamp": fact.get("time_key"),
                }
                for fact in source_facts
                if str(fact.get("summary") or "").strip()
            ],
            ensure_ascii=False,
            indent=2,
        )

    def _generate_observation_using_llm(
        self,
        *,
        evidence_bundle: Dict[str, Any],
        observation_type: str,
        source_facts: List[Dict[str, Any]],
        existing_observation: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create or incrementally revise one stable observation."""
        if not source_facts:
            return None
        if existing_observation:
            prompt = OBSERVATION_UPDATE_PROMPT.format(
                observation_type=observation_type,
                observation_type_definition=(
                    self._observation_type_prompt_guidance(observation_type)
                ),
                existing_observation=json.dumps(
                    {
                        "observation_id": existing_observation.get("id"),
                        "observation_type": observation_type,
                        "summary": existing_observation.get("summary", ""),
                        "confidence": existing_observation.get("confidence", 0.5),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                new_facts=self._observation_fact_payload(source_facts),
            )
        else:
            prompt = OBSERVATION_CREATE_PROMPT.format(
                requested_observation_type=observation_type,
                observation_type_definition=(
                    self._observation_type_prompt_guidance(observation_type)
                ),
                evidence_bundle_context=json.dumps(
                    {
                        "evidence_bundle_id": evidence_bundle.get("id"),
                        "entity": evidence_bundle.get("entity_name", ""),
                        "topic": evidence_bundle.get("topic_label")
                        or evidence_bundle.get("topic_key", ""),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                source_facts=self._observation_fact_payload(source_facts),
            )
        data = self._parse_json_object_from_llm_text(self._call_llm(prompt) or "")
        if not data:
            return None
        returned_type = str(
            data.get("observation_type") or observation_type
        ).strip().lower()
        summary = str(data.get("summary") or "").strip()
        if returned_type != observation_type or not summary:
            return None
        try:
            confidence = float(data.get("confidence", 0.7) or 0.7)
        except (TypeError, ValueError):
            confidence = 0.7
        return {
            "summary": summary,
            "confidence": max(0.0, min(1.0, confidence)),
            "change_summary": str(data.get("change_summary") or "").strip(),
        }

    @staticmethod
    def _fallback_observation_text(
        source_facts: List[Dict[str, Any]],
        *,
        existing_text: str = "",
    ) -> str:
        summaries = list(dict.fromkeys(
            str(fact.get("summary") or "").strip()
            for fact in source_facts
            if str(fact.get("summary") or "").strip()
        ))
        existing = str(existing_text or "").strip()
        if existing:
            additions = [summary for summary in summaries if summary not in existing]
            return "；".join([existing, *additions[:2]])
        return max(summaries, key=len) if summaries else ""

    def _observation_record_from_sources(
        self,
        *,
        observation_type: str,
        summary: str,
        source_facts: List[Dict[str, Any]],
        confidence: float,
        previous_metadata: Optional[Dict[str, Any]] = None,
        change_summary: str = "",
    ) -> Dict[str, Any]:
        evidence_mode = self._observation_evidence_mode(
            observation_type,
            source_facts,
        )
        metadata = dict(previous_metadata or {})
        metadata.update({
            "source": "evidence_bundle_fact_incremental_observation",
            "allowed_interpretation_types": (
                self._observation_allowed_interpretation_types(
                    observation_type,
                    evidence_mode,
                )
            ),
            "candidate_interpretation_types": (
                self._observation_candidate_families(observation_type)
            ),
            "fact_type_distribution": (
                self._fact_type_distribution_from_facts(source_facts)
            ),
            "fact_kind_distribution": dict(Counter(
                str(fact.get("fact_kind") or "other").strip().lower()
                for fact in source_facts
            )),
            "source_count": len(source_facts),
            "revision": int(metadata.get("revision") or 0) + 1,
        })
        if change_summary:
            metadata["last_change_summary"] = change_summary
        embedding_text = "\n".join([
            f"Observation type: {observation_type}",
            f"Summary text: {summary}",
        ])
        return {
            "observation_type": observation_type,
            "summary": summary,
            "evidence_mode": evidence_mode,
            "confidence": confidence,
            "source_fact_ids": [
                int(self._fact_id(fact))
                for fact in source_facts
                if self._fact_id(fact) is not None
            ],
            "embedding": self._embed_memory_layer_text(embedding_text),
            "embedding_text": embedding_text,
            "evidence_centroid_embedding": (
                self._evidence_centroid_for_sources(source_facts)
            ),
            "metadata": metadata,
        }

    def _update_observations_for_evidence_bundles(
        self,
        evidence_bundle_ids: List[int],
    ) -> List[int]:
        """Incrementally match new facts to stable observations."""
        if not self._db:
            return []
        clean_ids = list(dict.fromkeys(
            int(evidence_bundle_id)
            for evidence_bundle_id in evidence_bundle_ids
            if evidence_bundle_id is not None
        ))
        evidence_bundles = {
            int(item["id"]): item
            for item in self._db.get_evidence_bundles_by_ids(clean_ids)
        }
        supporting_facts = self._db.get_evidence_bundle_supporting_facts(
            clean_ids,
            per_evidence_bundle=1000,
        )
        existing_by_bundle: Dict[int, List[Dict[str, Any]]] = {}
        for observation in self._db.get_observations_for_evidence_bundles(clean_ids):
            existing_by_bundle.setdefault(
                int(observation["evidence_bundle_id"]),
                [],
            ).append(observation)

        touched_observation_ids: List[int] = []
        for evidence_bundle_id in clean_ids:
            bundle_observation_ids: List[int] = []
            evidence_bundle = evidence_bundles.get(evidence_bundle_id)
            if not evidence_bundle:
                continue
            all_facts = supporting_facts.get(evidence_bundle_id, [])
            existing_observations = existing_by_bundle.get(
                evidence_bundle_id,
                [],
            )
            assigned_fact_ids = {
                int(fact_id)
                for observation in existing_observations
                for fact_id in observation.get("source_fact_ids", [])
            }
            self._db.memory_set_evidence_bundle_sources_pending_observation(
                evidence_bundle_id,
                sorted(assigned_fact_ids),
                pending=False,
            )
            new_facts = [
                fact for fact in all_facts
                if (
                    self._fact_id(fact) is not None
                    and int(self._fact_id(fact)) not in assigned_fact_ids
                )
            ]
            relevant_fact_ids = {
                int(self._fact_id(fact))
                for fact in new_facts
                if self._fact_id(fact) is not None
            }
            relevant_fact_ids.update(
                int(fact_id)
                for observation in existing_observations
                for fact_id in observation.get("source_fact_ids", [])
            )
            fact_embeddings = self._db.memory_fact_embeddings(
                sorted(relevant_fact_ids)
            )
            fact_clusters = (
                self._cluster_evidence_bundle_facts_into_observations(
                    new_facts
                )
            )
            matched: Dict[int, List[Dict[str, Any]]] = {}
            unmatched_clusters: List[Dict[str, Any]] = []
            new_facts_by_id = {
                int(self._fact_id(fact)): fact
                for fact in new_facts
                if self._fact_id(fact) is not None
            }
            for cluster in fact_clusters:
                cluster_source_ids = [
                    int(fact_id)
                    for fact_id in cluster.get("source_fact_ids", [])
                    if fact_id is not None
                ]
                cluster_observation_type = str(
                    cluster.get("observation_type") or "context"
                )
                scored = []
                for observation in existing_observations:
                    match_rule, threshold = self._observation_type_match_rule(
                        cluster_observation_type,
                        str(observation.get("observation_type") or ""),
                    )
                    if threshold is None:
                        continue
                    (
                        match_similarity,
                        centroid_similarity,
                        max_source_similarity,
                        coverage_similarity,
                    ) = self._score_fact_cluster_against_observation_evidence(
                        cluster_source_ids,
                        cluster.get("evidence_centroid_embedding"),
                        observation,
                        fact_embeddings,
                    )
                    scored.append((
                        match_similarity,
                        centroid_similarity,
                        max_source_similarity,
                        coverage_similarity,
                        observation,
                        match_rule,
                        threshold,
                    ))
                scored.sort(
                    key=lambda item: (
                        item[0] >= item[6],
                        item[0],
                        item[5] == "exact",
                    ),
                    reverse=True,
                )
                best_match = scored[0] if scored else None
                self._log_info(
                    "memory_reflect",
                    "observation_cluster_evidence_similarity_scored",
                    {
                        "evidence_bundle_id": evidence_bundle_id,
                        "cluster_source_fact_ids": cluster_source_ids,
                        "cluster_source_count": len(cluster_source_ids),
                        "observation_type": cluster_observation_type,
                        "best_observation_id": (
                            int(best_match[4]["id"])
                            if best_match
                            else None
                        ),
                        "match_similarity": round(best_match[0], 4)
                        if best_match
                        else 0.0,
                        "centroid_similarity": round(best_match[1], 4)
                        if best_match
                        else 0.0,
                        "max_source_similarity": round(best_match[2], 4)
                        if best_match
                        else 0.0,
                        "coverage_similarity": round(best_match[3], 4)
                        if best_match
                        else 0.0,
                        "type_match": best_match[5] if best_match else "none",
                        "threshold": best_match[6] if best_match else None,
                    },
                )
                if (
                    best_match
                    and best_match[0] >= best_match[6]
                ):
                    observation_id = int(best_match[4]["id"])
                    matched_facts = matched.setdefault(observation_id, [])
                    matched_fact_ids = {
                        int(self._fact_id(fact))
                        for fact in matched_facts
                        if self._fact_id(fact) is not None
                    }
                    matched_facts.extend(
                        new_facts_by_id[fact_id]
                        for fact_id in cluster_source_ids
                        if (
                            fact_id in new_facts_by_id
                            and fact_id not in matched_fact_ids
                        )
                    )
                else:
                    unmatched_clusters.append(cluster)

            by_id = {
                int(observation["id"]): observation
                for observation in existing_observations
            }
            for observation_id, added_facts in matched.items():
                existing = by_id[observation_id]
                historical_facts = self._db.memory_facts_by_ids(
                    existing.get("source_fact_ids", [])
                )
                combined_facts = historical_facts + added_facts
                generated = self._generate_observation_using_llm(
                    evidence_bundle=evidence_bundle,
                    observation_type=str(existing["observation_type"]),
                    source_facts=added_facts,
                    existing_observation=existing,
                )
                summary = (
                    generated["summary"]
                    if generated
                    else self._fallback_observation_text(
                        added_facts,
                        existing_text=str(existing.get("summary") or ""),
                    )
                )
                record = self._observation_record_from_sources(
                    observation_type=str(existing["observation_type"]),
                    summary=summary,
                    source_facts=combined_facts,
                    confidence=(
                        generated["confidence"]
                        if generated
                        else float(existing.get("confidence") or 0.5)
                    ),
                    previous_metadata=self._json_dict(
                        existing.get("metadata", {})
                    ),
                    change_summary=(
                        generated.get("change_summary", "")
                        if generated
                        else "deterministic fallback after LLM failure"
                    ),
                )
                self._db.memory_update_observation(
                    observation_id,
                    summary=record["summary"],
                    evidence_mode=record["evidence_mode"],
                    confidence=record["confidence"],
                    source_fact_ids=record["source_fact_ids"],
                    embedding=record["embedding"],
                    embedding_text=record["embedding_text"],
                    evidence_centroid_embedding=record[
                        "evidence_centroid_embedding"
                    ],
                    metadata=record["metadata"],
                )
                self._db.memory_set_evidence_bundle_sources_pending_observation(
                    evidence_bundle_id,
                    [
                        int(self._fact_id(fact))
                        for fact in added_facts
                        if self._fact_id(fact) is not None
                    ],
                    pending=False,
                )
                touched_observation_ids.append(observation_id)
                bundle_observation_ids.append(observation_id)

            for candidate in unmatched_clusters:
                candidate_source_ids = [
                    int(fact_id)
                    for fact_id in candidate.get("source_fact_ids", [])
                    if fact_id is not None
                ]
                if (
                    len(candidate_source_ids)
                    < OBSERVATION_MIN_FACTS_FOR_NEW_CLUSTER
                ):
                    self._db.memory_set_evidence_bundle_sources_pending_observation(
                        evidence_bundle_id,
                        candidate_source_ids,
                        pending=True,
                    )
                    self._log_info(
                        "memory_reflect",
                        "observation_fact_cluster_deferred",
                        {
                            "evidence_bundle_id": evidence_bundle_id,
                            "observation_type": candidate.get(
                                "observation_type"
                            ),
                            "source_fact_ids": candidate_source_ids,
                            "minimum_fact_count": (
                                OBSERVATION_MIN_FACTS_FOR_NEW_CLUSTER
                            ),
                        },
                    )
                    continue
                candidate_facts = self._db.memory_facts_by_ids(
                    candidate_source_ids
                )
                generated = self._generate_observation_using_llm(
                    evidence_bundle=evidence_bundle,
                    observation_type=str(candidate["observation_type"]),
                    source_facts=candidate_facts,
                )
                summary = (
                    generated["summary"]
                    if generated
                    else self._fallback_observation_text(candidate_facts)
                )
                record = self._observation_record_from_sources(
                    observation_type=str(candidate["observation_type"]),
                    summary=summary,
                    source_facts=candidate_facts,
                    confidence=(
                        generated["confidence"]
                        if generated
                        else float(candidate.get("confidence") or 0.5)
                    ),
                    change_summary=(
                        generated.get("change_summary", "")
                        if generated
                        else "deterministic fallback after LLM failure"
                    ),
                )
                observation_id = self._db.memory_create_observation(
                    evidence_bundle_id,
                    record,
                )
                if observation_id is not None:
                    self._db.memory_set_evidence_bundle_sources_pending_observation(
                        evidence_bundle_id,
                        candidate_source_ids,
                        pending=False,
                    )
                    touched_observation_ids.append(observation_id)
                    bundle_observation_ids.append(observation_id)

            pending_observation_fact_ids = (
                self._db
                .memory_evidence_bundle_pending_observation_source_ids(
                    evidence_bundle_id
                )
            )
            self._log_info(
                "memory_reflect",
                "evidence_bundle_observations_incrementally_updated",
                {
                    "evidence_bundle_id": evidence_bundle_id,
                    "new_fact_ids": [
                        int(self._fact_id(fact)) for fact in new_facts
                    ],
                    "touched_observation_ids": bundle_observation_ids,
                    "pending_observation_fact_ids": (
                        pending_observation_fact_ids
                    ),
                },
            )
        return list(dict.fromkeys(touched_observation_ids))

    @staticmethod
    def _is_task_event_like_fact(fact: Dict[str, Any]) -> bool:
        if MemoryNodeManager._fact_kind_excluded_from_task(fact):
            return False
        task_event_like_raw = fact.get("task_event_like")
        task_relevance = str(fact.get("task_relevance") or "").strip().lower()
        if isinstance(task_event_like_raw, bool):
            if not task_event_like_raw:
                return False
            if task_relevance in {"none", "weak"}:
                return False
            return True
        if isinstance(task_event_like_raw, (int, float)):
            if int(task_event_like_raw) <= 0:
                return False
            if task_relevance in {"none", "weak"}:
                return False
            return True
        if isinstance(task_event_like_raw, str) and task_event_like_raw.strip():
            lowered = task_event_like_raw.strip().lower()
            if lowered in {"false", "no", "0"}:
                return False
            if lowered in {"true", "yes", "1"}:
                if task_relevance in {"none", "weak"}:
                    return False
                return True

        text = " ".join([
            str(fact.get("summary", "") or ""),
            " ".join(str(item) for item in fact.get("keywords", []) if str(item or "").strip()),
            " ".join(str(item) for item in fact.get("topics", []) if str(item or "").strip()),
        ]).lower()
        if not text.strip():
            return False
        action_patterns = [
            "修改", "实现", "排查", "验证", "测试", "设计", "讨论", "优化", "补充",
            "合并", "接入", "调用", "迁移", "重构", "修复", "继续", "先不要",
            "可以修改", "需要", "计划", "准备", "下一步", "更新", "生成", "新增",
            "完成", "通过", "失败", "阻塞", "卡住", "暂停", "恢复", "解决", "放一下",
            "implement", "fix", "debug", "investigate", "test", "verify",
            "design", "update", "refactor", "migrate", "add", "remove",
            "continue", "plan", "next step", "complete", "completed", "done",
            "pass", "passed", "fail", "failed", "block", "blocked", "pause",
            "paused", "resume", "resumed", "resolve", "resolved",
        ]
        return any(pattern in text for pattern in action_patterns)

    def _fact_anchor_entity(self, facts: List[Dict[str, Any]]) -> Optional[Tuple[int, str]]:
        counts: Dict[int, Dict[str, Any]] = {}
        order = 0
        for fact in facts:
            for entity_id, entity_name in self._fact_entity_pairs(fact):
                item = counts.setdefault(
                    entity_id,
                    {"entity_id": entity_id, "entity_name": entity_name, "count": 0, "order": order},
                )
                item["count"] += 1
                order += 1
        if not counts:
            return None
        winner = sorted(counts.values(), key=lambda item: (-item["count"], item["order"]))[0]
        return int(winner["entity_id"]), str(winner["entity_name"] or "")
        
    @staticmethod
    def _filter_int_ids(values: Any, allowed: set[int]) -> List[int]:
        if not isinstance(values, list):
            return []
        out: List[int] = []
        seen = set()
        for value in values:
            try:
                int_value = int(value)
            except (TypeError, ValueError):
                continue
            if int_value not in allowed or int_value in seen:
                continue
            seen.add(int_value)
            out.append(int_value)
        return out

    def _generate_interpretation(
        self,
        *,
        observation: Dict[str, Any],
        source_facts: List[Dict[str, Any]],
        observation_id: int,
        observation_ids: Optional[List[int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generate an optional current interpretation from an observation."""
        fact_lines = []
        allowed_fact_ids: set[int] = set()
        for index, fact in enumerate(source_facts[:8], 1):
            fact_id = fact.get("id", fact.get("fact_id"))
            try:
                int_fact_id = int(fact_id)
            except (TypeError, ValueError):
                continue
            summary = str(fact.get("summary") or "").strip()
            if not summary:
                continue
            allowed_fact_ids.add(int_fact_id)
            fact_type = str(fact.get("fact_type") or "semantic")
            fact_subject = str(fact.get("fact_subject") or "other")
            fact_kind = str(fact.get("fact_kind") or "other")
            fact_lines.append(f"{index}. id={int_fact_id} [{fact_type}/{fact_subject}/{fact_kind}] {summary}")
        if not fact_lines:
            return None

        metadata = observation.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata or "{}")
            except (TypeError, ValueError):
                metadata = {}
        prompt = INTERPRETATION_GENERATION_PROMPT.format(
            entity_name=observation.get("entity_name", ""),
            topic_label=observation.get("topic_label") or observation.get("topic_key") or "",
            observation_type=observation.get("observation_type", "insight"),
            observation_summary=observation.get("summary", ""),
            observation_metadata=json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
            source_facts="\n".join(fact_lines),
        )
        result = self._call_llm(prompt)
        data = self._parse_json_object_from_llm_text(result or "")
        if not data or not bool(data.get("should_create")):
            return None

        claim = str(data.get("claim") or "").strip()
        action_implication = str(data.get("action_implication") or "").strip()
        if not claim or not action_implication:
            return None
        allowed_types = {
            "insight", "task",
            "explicit_preference", "explicit_instruction", "inferred_preference",
            "behavior_pattern", "project_state", "task_risk", "constraint",
            "conflict_resolution", "strategy", "other",
        }
        interpretation_type = str(
            data.get("interpretation_type") or "behavior_pattern"
        ).strip().lower().replace("-", "_").replace(" ", "_")
        if interpretation_type not in allowed_types:
            interpretation_type = "behavior_pattern"
        allowed_observation_types = {
            str(value).strip().lower()
            for value in metadata.get("allowed_interpretation_types", [])
            if str(value or "").strip()
        }
        if (
            allowed_observation_types
            and interpretation_type not in allowed_observation_types
        ):
            self._log_info(
                "memory_reflect",
                "interpretation_generation_rejected",
                {
                    "observation_id": observation.get("id"),
                    "observation_type": metadata.get("observation_type"),
                    "interpretation_type": interpretation_type,
                    "reason": "observation_type_gate",
                    "allowed_interpretation_types": sorted(
                        allowed_observation_types
                    ),
                },
            )
            return None
        status = str(data.get("status") or "current").strip().lower()
        if status not in {"current", "conflicted"}:
            status = "current"
        conflict_status = str(data.get("conflict_status") or "none").strip().lower()
        if conflict_status not in {"none", "resolved", "unresolved"}:
            conflict_status = "none"
        polarity = str(data.get("polarity") or "neutral").strip().lower()
        if polarity not in {"positive", "negative", "mixed", "neutral"}:
            polarity = "neutral"
        try:
            strength = float(data.get("strength", 0.5) or 0.5)
        except (TypeError, ValueError):
            strength = 0.5
        try:
            confidence = float(data.get("confidence", 0.5) or 0.5)
        except (TypeError, ValueError):
            confidence = 0.5
        allowed_observation_ids = {
            int(item)
            for item in (observation_ids or [observation_id])
            if item is not None
        }
        if not allowed_observation_ids:
            allowed_observation_ids = {int(observation_id)}
        evidence_observation_ids = self._filter_int_ids(
            data.get("evidence_observation_ids", sorted(allowed_observation_ids)),
            allowed_observation_ids,
        ) or sorted(allowed_observation_ids)
        if len(evidence_observation_ids) == 1:
            allowed, single_reason = self._single_observation_generation_allowed(
                observation=observation,
                source_facts=source_facts,
                interpretation_family=self._interpretation_family(interpretation_type),
            )
            if not allowed:
                self._log_info(
                    "memory_reflect",
                    "interpretation_generation_rejected",
                    {
                        "observation_id": evidence_observation_ids[0],
                        "interpretation_type": interpretation_type,
                        "reason": single_reason,
                    },
                )
                return None
            if interpretation_type not in {"task", "explicit_preference", "explicit_instruction"}:
                confidence = min(confidence, INTERPRETATION_SINGLE_OBSERVATION_CONFIDENCE_CAP)
        metadata_out = data.get("metadata", {})
        if not isinstance(metadata_out, dict):
            metadata_out = {}
        if interpretation_type == "task":
            task_metadata = self._normalize_task_metadata(metadata_out, allow_stale=True)
            task_metadata["task_source"] = "inferred_from_interpretation"
            metadata_out = task_metadata
        else:
            metadata_out = {
                key: value
                for key, value in metadata_out.items()
                if key not in {"task_status", "task_source", "goal", "steps", "next_action"}
            }
        metadata_out = {
            "source": "interpretation_generation",
            **metadata_out,
        }
        if len(evidence_observation_ids) == 1:
            metadata_out.setdefault("evidence_shape", "single_observation")
            if interpretation_type not in {"task", "explicit_preference", "explicit_instruction"}:
                metadata_out.setdefault("stability", "tentative")
        return {
            "claim": claim,
            "subject_text": str(data.get("subject_text") or "agent").strip() or "agent",
            "target_text": str(data.get("target_text") or "").strip(),
            "scope": str(data.get("scope") or "general").strip() or "general",
            "interpretation_type": interpretation_type,
            "polarity": polarity,
            "strength": max(0.0, min(1.0, strength)),
            "confidence": max(0.0, min(1.0, confidence)),
            "status": status,
            "conflict_status": conflict_status,
            "resolution": str(data.get("resolution") or "").strip(),
            "action_implication": action_implication,
            "evidence_fact_ids": self._filter_int_ids(
                data.get("evidence_fact_ids", []),
                allowed_fact_ids,
            ),
            "evidence_observation_ids": evidence_observation_ids,
            "counter_evidence_fact_ids": self._filter_int_ids(
                data.get("counter_evidence_fact_ids", []),
                allowed_fact_ids,
            ),
            "counter_evidence_observation_ids": self._filter_int_ids(
                data.get("counter_evidence_observation_ids", []),
                allowed_observation_ids,
            ),
            "metadata": metadata_out,
        }

    def _update_existing_interpretation_from_observation(
        self,
        *,
        interpretation: Dict[str, Any],
        observation: Dict[str, Any],
        source_facts: List[Dict[str, Any]],
        observation_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Update a matched current interpretation using a new observation."""
        fact_lines = []
        allowed_fact_ids: set[int] = set()
        for index, fact in enumerate(source_facts[:8], 1):
            fact_id = fact.get("id", fact.get("fact_id"))
            try:
                int_fact_id = int(fact_id)
            except (TypeError, ValueError):
                continue
            summary = str(fact.get("summary") or "").strip()
            if not summary:
                continue
            allowed_fact_ids.add(int_fact_id)
            fact_type = str(fact.get("fact_type") or "semantic")
            fact_subject = str(fact.get("fact_subject") or "other")
            fact_kind = str(fact.get("fact_kind") or "other")
            fact_lines.append(f"{index}. id={int_fact_id} [{fact_type}/{fact_subject}/{fact_kind}] {summary}")
        if not fact_lines:
            return None

        interpretation_metadata = self._json_dict(interpretation.get("metadata", {}))
        observation_metadata = self._json_dict(observation.get("metadata", {}))
        prompt = INTERPRETATION_UPDATE_PROMPT.format(
            entity_name=observation.get("entity_name", ""),
            topic_label=observation.get("topic_label") or observation.get("topic_key") or "",
            interpretation_id=interpretation.get("id"),
            interpretation_type=interpretation.get("interpretation_type", "insight"),
            status=interpretation.get("status", "current"),
            conflict_status=interpretation.get("conflict_status", "none"),
            polarity=interpretation.get("polarity", "neutral"),
            strength=interpretation.get("strength", 0.5),
            confidence=interpretation.get("confidence", 0.5),
            target_text=interpretation.get("target_text", ""),
            scope=interpretation.get("scope", "general"),
            claim=interpretation.get("claim", ""),
            resolution=interpretation.get("resolution", ""),
            action_implication=interpretation.get("action_implication", ""),
            interpretation_metadata=json.dumps(interpretation_metadata or {}, ensure_ascii=False, sort_keys=True),
            observation_id=int(observation_id),
            observation_type=observation.get("observation_type", "observation"),
            observation_summary=observation.get("summary", ""),
            observation_metadata=json.dumps(observation_metadata or {}, ensure_ascii=False, sort_keys=True),
            source_facts="\n".join(fact_lines),
        )
        result = self._call_llm(prompt)
        data = self._parse_json_object_from_llm_text(result or "")
        if not data or not bool(data.get("should_update")):
            return None

        allowed_types = {
            "insight", "task",
            "explicit_preference", "explicit_instruction", "inferred_preference",
            "behavior_pattern", "project_state", "task_risk", "constraint",
            "conflict_resolution", "strategy", "other",
        }
        existing_type = str(interpretation.get("interpretation_type") or "insight")
        interpretation_type = str(
            data.get("interpretation_type") or existing_type
        ).strip().lower().replace("-", "_").replace(" ", "_")
        if interpretation_type not in allowed_types:
            interpretation_type = existing_type if existing_type in allowed_types else "insight"
        allowed_observation_types = {
            str(value).strip().lower()
            for value in observation_metadata.get(
                "allowed_interpretation_types",
                [],
            )
            if str(value or "").strip()
        }
        if (
            allowed_observation_types
            and interpretation_type not in allowed_observation_types
        ):
            return None

        claim = str(data.get("claim") or interpretation.get("claim") or "").strip()
        action_implication = str(
            data.get("action_implication") or interpretation.get("action_implication") or ""
        ).strip()
        if not claim or not action_implication:
            return None

        status = str(data.get("status") or interpretation.get("status") or "current").strip().lower()
        if status not in {"current", "conflicted"}:
            status = "current"
        conflict_status = str(
            data.get("conflict_status") or interpretation.get("conflict_status") or "none"
        ).strip().lower()
        if conflict_status not in {"none", "resolved", "unresolved"}:
            conflict_status = "none"
        polarity = str(data.get("polarity") or interpretation.get("polarity") or "neutral").strip().lower()
        if polarity not in {"positive", "negative", "mixed", "neutral"}:
            polarity = "neutral"
        try:
            strength = float(data.get("strength", interpretation.get("strength", 0.5)) or 0.5)
        except (TypeError, ValueError):
            strength = float(interpretation.get("strength") or 0.5)
        try:
            confidence = float(data.get("confidence", interpretation.get("confidence", 0.5)) or 0.5)
        except (TypeError, ValueError):
            confidence = float(interpretation.get("confidence") or 0.5)

        allowed_observation_ids = {int(observation_id)}
        metadata_out = data.get("metadata", {})
        if not isinstance(metadata_out, dict):
            metadata_out = {}
        if interpretation_type == "task":
            task_metadata = self._normalize_task_metadata(metadata_out, allow_stale=True)
            task_metadata["task_source"] = "inferred_from_interpretation"
            metadata_out = task_metadata
        else:
            metadata_out = {
                key: value
                for key, value in metadata_out.items()
                if key not in {"task_status", "task_source", "goal", "steps", "next_action"}
            }
        metadata_out = {
            "source": "interpretation_update",
            **metadata_out,
        }
        return {
            "claim": claim,
            "subject_text": str(interpretation.get("subject_text") or "agent").strip() or "agent",
            "target_text": str(data.get("target_text") or interpretation.get("target_text") or "").strip(),
            "scope": str(data.get("scope") or interpretation.get("scope") or "general").strip() or "general",
            "interpretation_type": interpretation_type,
            "polarity": polarity,
            "strength": max(0.0, min(1.0, strength)),
            "confidence": max(0.0, min(1.0, confidence)),
            "status": status,
            "conflict_status": conflict_status,
            "resolution": str(data.get("resolution") or interpretation.get("resolution") or "").strip(),
            "action_implication": action_implication,
            "evidence_fact_ids": self._filter_int_ids(
                data.get("evidence_fact_ids", list(allowed_fact_ids)),
                allowed_fact_ids,
            ),
            "evidence_observation_ids": self._filter_int_ids(
                data.get("evidence_observation_ids", [observation_id]),
                allowed_observation_ids,
            ) or [int(observation_id)],
            "counter_evidence_fact_ids": self._filter_int_ids(
                data.get("counter_evidence_fact_ids", []),
                allowed_fact_ids,
            ),
            "counter_evidence_observation_ids": self._filter_int_ids(
                data.get("counter_evidence_observation_ids", []),
                allowed_observation_ids,
            ),
            "metadata": metadata_out,
        }

    def _update_existing_interpretation_from_observations(
        self,
        *,
        interpretation: Dict[str, Any],
        assignments: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Update one interpretation from all matched observations at once."""
        if not assignments:
            return None

        observation_payloads: List[Dict[str, Any]] = []
        source_facts = self._dedupe_source_facts([
            fact
            for assignment in assignments
            for fact in assignment.get("item", {}).get("source_facts", [])
        ])
        allowed_fact_ids = {
            int(fact["id"])
            for fact in source_facts
            if fact.get("id") is not None
        }
        allowed_observation_ids = {
            int(assignment["item"]["observation_id"])
            for assignment in assignments
        }
        for assignment in assignments:
            item = assignment["item"]
            observation = item["observation"]
            observation_payloads.append({
                "id": int(item["observation_id"]),
                "observation_type": observation.get("observation_type"),
                "summary": observation.get("summary"),
                "metadata": self._json_dict(observation.get("metadata", {})),
                "relationship": assignment.get("relationship", "extend"),
                "conflict_level": assignment.get("conflict_level", "none"),
                "intrinsic_value": assignment.get("intrinsic_value", "medium"),
                "reason": assignment.get("reason", ""),
            })

        fact_lines = []
        for index, fact in enumerate(source_facts[:24], 1):
            fact_id = int(fact["id"])
            summary = str(fact.get("summary") or "").strip()
            if not summary:
                continue
            fact_lines.append(
                f"{index}. id={fact_id} "
                f"[{fact.get('fact_type', 'semantic')}/"
                f"{fact.get('fact_subject', 'other')}/"
                f"{fact.get('fact_kind', 'other')}] {summary}"
            )
        if not fact_lines:
            return None

        interpretation_payload = {
            key: interpretation.get(key)
            for key in (
                "id", "claim", "target_text", "scope", "interpretation_type",
                "polarity", "strength", "confidence", "status",
                "conflict_status", "resolution", "action_implication",
                "evidence_fact_ids", "evidence_observation_ids",
                "counter_evidence_fact_ids",
                "counter_evidence_observation_ids", "metadata",
            )
        }
        prompt = INTERPRETATION_BATCH_UPDATE_PROMPT.format(
            interpretation=json.dumps(
                interpretation_payload,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
            observations=json.dumps(
                observation_payloads,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
            source_facts="\n".join(fact_lines),
        )
        data = self._parse_json_object_from_llm_text(self._call_llm(prompt) or "")
        if not data or not bool(data.get("should_update")):
            return None

        allowed_types = {
            "insight", "task",
            "explicit_preference", "explicit_instruction", "inferred_preference",
            "behavior_pattern", "project_state", "task_risk", "constraint",
            "conflict_resolution", "strategy", "other",
        }
        existing_type = str(
            interpretation.get("interpretation_type") or "insight"
        ).strip().lower()
        interpretation_type = str(
            data.get("interpretation_type") or existing_type
        ).strip().lower().replace("-", "_").replace(" ", "_")
        if interpretation_type not in allowed_types:
            interpretation_type = (
                existing_type if existing_type in allowed_types else "insight"
            )

        for assignment in assignments:
            allowed_for_observation = {
                str(value).strip().lower()
                for value in self._json_dict(
                    assignment["item"]["observation"].get("metadata", {})
                ).get("allowed_interpretation_types", [])
                if str(value or "").strip()
            }
            if (
                allowed_for_observation
                and interpretation_type not in allowed_for_observation
            ):
                return None

        claim = str(
            data.get("claim") or interpretation.get("claim") or ""
        ).strip()
        action_implication = str(
            data.get("action_implication")
            or interpretation.get("action_implication")
            or ""
        ).strip()
        if not claim or not action_implication:
            return None

        status = str(
            data.get("status") or interpretation.get("status") or "current"
        ).strip().lower()
        if status not in {"current", "conflicted"}:
            status = "current"
        conflict_status = str(
            data.get("conflict_status")
            or interpretation.get("conflict_status")
            or "none"
        ).strip().lower()
        if conflict_status not in {"none", "resolved", "unresolved"}:
            conflict_status = "none"
        polarity = str(
            data.get("polarity")
            or interpretation.get("polarity")
            or "neutral"
        ).strip().lower()
        if polarity not in {"positive", "negative", "mixed", "neutral"}:
            polarity = "neutral"
        try:
            strength = float(
                data.get("strength", interpretation.get("strength", 0.5))
                or 0.5
            )
        except (TypeError, ValueError):
            strength = float(interpretation.get("strength") or 0.5)
        try:
            confidence = float(
                data.get("confidence", interpretation.get("confidence", 0.5))
                or 0.5
            )
        except (TypeError, ValueError):
            confidence = float(interpretation.get("confidence") or 0.5)

        metadata_out = data.get("metadata", {})
        if not isinstance(metadata_out, dict):
            metadata_out = {}
        if interpretation_type == "task":
            metadata_out = self._normalize_task_metadata(
                metadata_out,
                allow_stale=True,
            )
            metadata_out["task_source"] = "inferred_from_interpretation"
        else:
            metadata_out = {
                key: value
                for key, value in metadata_out.items()
                if key not in {
                    "task_status", "task_source", "goal", "steps",
                    "next_action",
                }
            }
        metadata_out = {
            "source": "interpretation_batch_update",
            **metadata_out,
        }
        return {
            "claim": claim,
            "subject_text": str(
                interpretation.get("subject_text") or "agent"
            ).strip() or "agent",
            "target_text": str(
                data.get("target_text")
                or interpretation.get("target_text")
                or ""
            ).strip(),
            "scope": str(
                data.get("scope") or interpretation.get("scope") or "general"
            ).strip() or "general",
            "interpretation_type": interpretation_type,
            "polarity": polarity,
            "strength": max(0.0, min(1.0, strength)),
            "confidence": max(0.0, min(1.0, confidence)),
            "status": status,
            "conflict_status": conflict_status,
            "resolution": str(
                data.get("resolution")
                or interpretation.get("resolution")
                or ""
            ).strip(),
            "action_implication": action_implication,
            "evidence_fact_ids": self._filter_int_ids(
                data.get("evidence_fact_ids", sorted(allowed_fact_ids)),
                allowed_fact_ids,
            ),
            "evidence_observation_ids": self._filter_int_ids(
                data.get(
                    "evidence_observation_ids",
                    sorted(allowed_observation_ids),
                ),
                allowed_observation_ids,
            ),
            "counter_evidence_fact_ids": self._filter_int_ids(
                data.get("counter_evidence_fact_ids", []),
                allowed_fact_ids,
            ),
            "counter_evidence_observation_ids": self._filter_int_ids(
                data.get("counter_evidence_observation_ids", []),
                allowed_observation_ids,
            ),
            "metadata": metadata_out,
        }

    @staticmethod
    def _interpretation_family(interpretation_type: Any) -> str:
        text = str(interpretation_type or "").strip().lower().replace("-", "_").replace(" ", "_")
        if text == "task":
            return "task"
        if text in {"explicit_preference", "explicit_instruction", "inferred_preference", "behavior_pattern"}:
            return "preference"
        return "insight"

    @staticmethod
    def _unique_source_node_count(source_facts: List[Dict[str, Any]]) -> int:
        fact_ids: set[int] = set()
        fallback_count = 0
        for fact in source_facts:
            fact_id = fact.get("id", fact.get("fact_id"))
            try:
                fact_ids.add(int(fact_id))
            except (TypeError, ValueError):
                fallback_count += 1
        return len(fact_ids) or fallback_count

    @classmethod
    def _single_observation_generation_allowed(
        cls,
        *,
        observation: Dict[str, Any],
        source_facts: List[Dict[str, Any]],
        interpretation_family: str,
    ) -> Tuple[bool, str]:
        metadata = cls._json_dict(observation.get("metadata", {}))
        observation_type = str(
            observation.get("observation_type") or "context"
        ).strip().lower()
        evidence_shape = str(metadata.get("evidence_shape") or "single_event").strip().lower()
        temporal_scope = str(metadata.get("temporal_scope") or "recent").strip().lower()
        dominant_fact_type = str(metadata.get("dominant_fact_type") or "unknown").strip().lower()
        evidence_mixture = str(metadata.get("evidence_mixture") or "unknown").strip().lower()
        source_kinds = {
            str(fact.get("fact_kind") or "").strip().lower()
            for fact in source_facts
        }
        source_count = cls._unique_source_node_count(source_facts)

        if interpretation_family == "task":
            if observation_type in {"task_state", "task_progress", "decision"}:
                return True, "task_observation_type"
            if any(cls._is_task_event_like_fact(fact) for fact in source_facts):
                return True, "task_event_evidence"
            if source_kinds & {"request", "action", "decision", "error", "recommendation"}:
                return True, "task_fact_kind"
            if dominant_fact_type == "episodic" and temporal_scope in {"momentary", "recent", "ongoing"}:
                return True, "episodic_task_context"
            return False, "weak_task_signal"

        if interpretation_family == "preference":
            if source_kinds & {"instruction"}:
                return True, "explicit_instruction"
            if source_kinds & {"preference"} and observation_type in {
                "preference_signal",
                "constraint",
                "behavior_pattern",
            }:
                return True, "explicit_preference_signal"
            if source_count >= 2 and evidence_shape in {"repeated_pattern", "confirmation"} and source_kinds & {"preference"}:
                return True, "repeated_preference_evidence"
            if source_count >= 2 and temporal_scope in {"ongoing", "recurring"} and source_kinds & {"preference", "instruction"}:
                return True, "stable_preference_scope"
            if (
                source_count >= 2
                and evidence_mixture in {"semantic_dominant", "balanced_mixed"}
                and evidence_shape in {"repeated_pattern", "confirmation"}
            ):
                return True, "stable_fact_type_preference_evidence"
            return False, "weak_preference_signal"

        if source_count < 2:
            return False, "single_fact_insight_signal"
        if evidence_shape in {"repeated_pattern", "contrast", "progression", "correction", "confirmation"}:
            return True, "structured_insight_evidence"
        if observation_type in {
            "problem",
            "task_progress",
            "decision",
            "behavior_pattern",
        } and temporal_scope != "momentary":
            return True, "material_insight_observation_type"
        if evidence_mixture in {"semantic_dominant", "episodic_dominant", "balanced_mixed"}:
            return True, "mixed_fact_type_insight"
        return False, "weak_insight_signal"

    @classmethod
    def _candidate_interpretation_families(
        cls,
        observation: Dict[str, Any],
        source_facts: List[Dict[str, Any]],
    ) -> List[str]:
        metadata = cls._json_dict(observation.get("metadata", {}))
        families = cls._metadata_candidate_types(metadata.get("candidate_interpretation_types"))
        if not families:
            families = cls._observation_candidate_families(
                str(observation.get("observation_type") or "context"),
            )
        return [family for family in ("insight", "task", "preference") if family in set(families)]

    @classmethod
    def _observation_cluster_interpretation_family(cls, observation: Dict[str, Any], source_facts: List[Dict[str, Any]]) -> str:
        observation_type = str(
            observation.get("observation_type") or "context"
        ).strip().lower()
        candidate_families = cls._candidate_interpretation_families(observation, source_facts)
        source_kinds = {
            str(fact.get("fact_kind") or "").strip().lower()
            for fact in source_facts
        }
        if "task" in candidate_families and (
            any(cls._is_task_event_like_fact(fact) for fact in source_facts)
            or observation_type in {"task_state", "task_progress", "decision"}
        ):
            return "task"
        if "preference" in candidate_families and (
            source_kinds & {"preference", "instruction"}
            or observation_type in {
                "preference_signal",
                "constraint",
                "behavior_pattern",
            }
        ):
            return "preference"
        if "task" in candidate_families and observation_type in {
            "task_state",
            "task_progress",
            "decision",
        } and source_kinds & {
            "request", "action", "decision", "error",
        }:
            return "task"
        return "insight"

    @staticmethod
    def _match_terms(*values: Any) -> set[str]:
        terms: set[str] = set()
        for value in values:
            if isinstance(value, list):
                iterable = value
            else:
                iterable = [value]
            for item in iterable:
                text = str(item or "").strip().lower()
                if not text:
                    continue
                for part in re.split(r"[^0-9a-zA-Z\u4e00-\u9fff]+", text):
                    if len(part) >= 2:
                        terms.add(part)
                compact = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", text)
                if len(compact) >= 2:
                    terms.add(compact)
        return terms

    @classmethod
    def _term_overlap_score(cls, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        overlap = left & right
        if not overlap:
            return 0.0
        return len(overlap) / max(1, min(len(left), len(right)))

    @staticmethod
    def _task_interpretation_status(interpretation: Dict[str, Any]) -> str:
        metadata = interpretation.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata or "{}")
            except (TypeError, ValueError):
                metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        status = str(metadata.get("task_status") or "").strip().lower()
        return status if status in {"active", "blocked", "paused", "stale"} else ""

    @classmethod
    def _interpretation_entity_ids(cls, interpretation: Dict[str, Any]) -> set[int]:
        metadata = cls._json_dict(interpretation.get("metadata", {}))
        values = [
            interpretation.get("entity_id"),
            metadata.get("entity_id"),
        ]
        out: set[int] = set()
        for value in values:
            try:
                if value is not None:
                    out.add(int(value))
            except (TypeError, ValueError):
                continue
        return out

    @classmethod
    def _observation_interpretation_temporal_score(
        cls,
        *,
        observation_metadata: Dict[str, Any],
        interpretation: Dict[str, Any],
        interpretation_family: str,
    ) -> Tuple[float, str]:
        temporal_scope = str(observation_metadata.get("temporal_scope") or "").strip().lower()
        evidence_shape = str(observation_metadata.get("evidence_shape") or "").strip().lower()
        if interpretation_family == "preference":
            if temporal_scope in {"ongoing", "recurring"}:
                return 0.08, "preference_scope"
            if evidence_shape in {"repeated_pattern", "confirmation"}:
                return 0.06, "preference_evidence_shape"
        if interpretation_family == "task":
            task_status = cls._task_interpretation_status(interpretation)
            if task_status in {"active", "blocked", "paused"} and temporal_scope in {"momentary", "recent", "ongoing"}:
                return 0.08, "task_temporal_state"
            if evidence_shape in {"progression", "correction", "confirmation"}:
                return 0.05, "task_evidence_shape"
        if interpretation_family == "insight":
            if evidence_shape in {"repeated_pattern", "contrast", "progression", "correction", "confirmation"}:
                return 0.08, "insight_evidence_shape"
            if temporal_scope in {"ongoing", "recurring", "historical"}:
                return 0.05, "insight_scope"
        return 0.0, ""

    @classmethod
    def _interpretation_type_specific_score(
        cls,
        *,
        observation: Dict[str, Any],
        source_facts: List[Dict[str, Any]],
        interpretation: Dict[str, Any],
        interpretation_family: str,
        observation_terms: set[str],
        interpretation_terms: set[str],
        observation_metadata: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []
        observation_type = str(
            observation.get("observation_type") or "context"
        ).strip().lower()
        evidence_shape = str(observation_metadata.get("evidence_shape") or "").strip().lower()
        dominant_fact_type = str(observation_metadata.get("dominant_fact_type") or "").strip().lower()
        evidence_mixture = str(observation_metadata.get("evidence_mixture") or "").strip().lower()
        source_kinds = {
            str(fact.get("fact_kind") or "").strip().lower()
            for fact in source_facts
        }
        source_subjects = {
            str(fact.get("fact_subject") or "").strip().lower()
            for fact in source_facts
        }

        if interpretation_family == "preference":
            if observation_type in {
                "preference_signal",
                "constraint",
                "behavior_pattern",
            }:
                score += 0.08
                reasons.append("preference_type")
            if source_kinds & {"preference", "instruction"}:
                score += 0.10
                reasons.append("preference_evidence")
            if source_subjects & {"user"}:
                score += 0.04
                reasons.append("user_subject")
            if dominant_fact_type == "semantic" or evidence_mixture in {"semantic_only", "semantic_dominant"}:
                score += 0.04
                reasons.append("semantic_preference_memory")
            elif evidence_mixture in {"balanced_mixed", "episodic_dominant"} and evidence_shape in {"repeated_pattern", "confirmation"}:
                score += 0.04
                reasons.append("episodic_preference_pattern")
            target_terms = cls._match_terms(interpretation.get("target_text"), interpretation.get("scope"))
            target_overlap = cls._term_overlap_score(observation_terms, target_terms)
            if target_overlap:
                score += min(0.08, target_overlap * 0.08)
                reasons.append("preference_target")

        elif interpretation_family == "task":
            if observation_type in {"task_state", "task_progress", "decision"}:
                score += 0.08
                reasons.append("task_type")
            if any(cls._is_task_event_like_fact(fact) for fact in source_facts) or source_kinds & {
                "request", "action", "decision", "error", "recommendation",
            }:
                score += 0.10
                reasons.append("task_evidence")
            task_status = cls._task_interpretation_status(interpretation)
            if task_status and task_status != "stale":
                score += 0.04
                reasons.append("task_active_state")
            if dominant_fact_type == "episodic" or evidence_mixture in {"episodic_only", "episodic_dominant", "balanced_mixed"}:
                score += 0.04
                reasons.append("episodic_task_memory")
            goal_terms = cls._match_terms(cls._json_dict(interpretation.get("metadata", {})).get("goal"))
            goal_overlap = cls._term_overlap_score(observation_terms, goal_terms)
            if goal_overlap:
                score += min(0.08, goal_overlap * 0.08)
                reasons.append("task_goal")

        else:
            if observation_type in {
                "behavior_pattern",
                "task_progress",
                "decision",
                "problem",
                "strategy",
                "context",
            }:
                score += 0.06
                reasons.append("insight_type")
            if evidence_shape in {"repeated_pattern", "contrast", "progression", "correction", "confirmation"}:
                score += 0.08
                reasons.append("insight_shape")
            if evidence_mixture in {"semantic_dominant", "episodic_dominant", "balanced_mixed"}:
                score += 0.04
                reasons.append("mixed_fact_type_evidence")
            claim_overlap = cls._term_overlap_score(
                observation_terms,
                cls._match_terms(interpretation.get("claim"), interpretation.get("resolution")),
            )
            if claim_overlap:
                score += min(0.08, claim_overlap * 0.08)
                reasons.append("insight_claim")

        temporal_score, temporal_reason = cls._observation_interpretation_temporal_score(
            observation_metadata=observation_metadata,
            interpretation=interpretation,
            interpretation_family=interpretation_family,
        )
        if temporal_score:
            score += temporal_score
            reasons.append(temporal_reason)
        return score, reasons

    def _calculate_interpretation_candidate_score_for_observation(
        self,
        *,
        observation: Dict[str, Any],
        source_facts: List[Dict[str, Any]],
        interpretation: Dict[str, Any],
        observation_id: int,
    ) -> Tuple[float, str]:
        metadata = self._json_dict(interpretation.get("metadata", {}))
        evidence_observation_ids = interpretation.get("evidence_observation_ids", [])
        linked_observation_ids = {
            int(value)
            for value in metadata.get("observation_ids", [])
            if str(value).isdigit()
        }
        if int(observation_id) in linked_observation_ids:
            return 1.0, "existing_observation_evidence"
        if (
            int(observation_id) in evidence_observation_ids
            or metadata.get("observation_id") == int(observation_id)
        ):
            return 1.0, "existing_observation_evidence"

        score = 0.0
        reasons: List[str] = []
        observation_metadata = self._json_dict(observation.get("metadata", {}))
        raw_observation_metadata = self._json_dict(observation.get("metadata", {}))
        allowed_interpretation_types = {
            str(value).strip().lower()
            for value in raw_observation_metadata.get(
                "allowed_interpretation_types",
                [],
            )
            if str(value or "").strip()
        }
        interpretation_type = str(
            interpretation.get("interpretation_type") or ""
        ).strip().lower()
        if (
            allowed_interpretation_types
            and interpretation_type not in allowed_interpretation_types
        ):
            return 0.0, "observation_type_gate"
        candidate_families = self._candidate_interpretation_families(observation, source_facts)
        observation_family = self._observation_cluster_interpretation_family(observation, source_facts)
        interpretation_family = self._interpretation_family(interpretation.get("interpretation_type"))
        if candidate_families and interpretation_family not in candidate_families:
            return 0.0, "candidate_type_gate"
        if observation_family == interpretation_family:
            score += 0.24
            reasons.append("family")
        elif observation_family == "preference" and interpretation_family == "insight":
            score += 0.08
            reasons.append("preference_insight")

        observation_entity_id = observation.get("entity_id")
        try:
            observation_entity_id_int = int(observation_entity_id) if observation_entity_id is not None else None
        except (TypeError, ValueError):
            observation_entity_id_int = None
        if observation_entity_id_int is not None and observation_entity_id_int in self._interpretation_entity_ids(interpretation):
            score += 0.18
            reasons.append("entity")
        observation_topic = self._topic_key(observation.get("topic_key") or observation.get("topic_label") or "")
        interpretation_topic = self._topic_key(
            metadata.get("topic_key")
            or interpretation.get("scope")
            or interpretation.get("target_text")
            or ""
        )
        if observation_topic and interpretation_topic and observation_topic == interpretation_topic:
            score += 0.20
            reasons.append("topic")

        observation_terms = self._match_terms(
            observation.get("summary"),
            observation.get("keywords", []),
            observation.get("topic_key"),
            observation.get("topic_label"),
        )
        interpretation_terms = self._match_terms(
            interpretation.get("claim"),
            interpretation.get("action_implication"),
            interpretation.get("target_text"),
            interpretation.get("scope"),
        )
        overlap = self._term_overlap_score(observation_terms, interpretation_terms)
        if overlap:
            score += min(0.20, overlap * 0.20)
            reasons.append("term_overlap")

        type_score, type_reasons = self._interpretation_type_specific_score(
            observation=observation,
            source_facts=source_facts,
            interpretation=interpretation,
            interpretation_family=interpretation_family,
            observation_terms=observation_terms,
            interpretation_terms=interpretation_terms,
            observation_metadata=observation_metadata,
        )
        score += type_score
        reasons.extend(type_reasons)

        try:
            confidence = float(interpretation.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        score += min(0.05, confidence * 0.05)
        return min(1.0, score), "+".join(reasons) or "weak"

    def _search_interpretation_candidates_for_observation(
        self,
        observation: Dict[str, Any],
        observation_id: int,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        seen: set[int] = set()

        try:
            existing = self._db.get_interpretations_for_observation(
                int(observation_id),
                limit=10,
            )
        except Exception:
            existing = []
        for item in existing:
            try:
                item_id = int(item["id"])
            except (TypeError, ValueError, KeyError):
                continue
            candidates.append(item)
            seen.add(item_id)

        query = " ".join(
            str(part or "").strip()
            for part in [
                observation.get("summary"),
                observation.get("keywords"),
                observation.get("topic_label"),
                observation.get("topic_key"),
            ]
            if str(part or "").strip()
        )
        entities = [observation.get("entity_name")] if observation.get("entity_name") else []
        try:
            searched = self._db.search_memory_interpretations(
                query,
                entities=entities,
                top_k=50,
                statuses=["current", "conflicted"],
                min_confidence=0.35,
            )
        except Exception:
            searched = []
        searched = self._rank_interpretation_search_candidates(
            searched,
            keyword=query,
            entities=entities,
            top_k=12,
            query_embedding=None,
            min_embedding_similarity=None,
        )
        for item in searched:
            try:
                item_id = int(item["id"])
            except (TypeError, ValueError, KeyError):
                continue
            if item_id in seen:
                continue
            candidates.append(item)
            seen.add(item_id)
        return candidates

    def _judge_observation_value_for_interpretation(
        self,
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Judge one observation against existing interpretations."""
        observation = item["observation"]
        observation_id = int(item["observation_id"])
        source_facts = item.get("source_facts", [])
        interpretation_candidates = self._search_interpretation_candidates_for_observation(
            observation,
            observation_id,
        )
        scored_interpretation_candidates: List[Tuple[float, str, Dict[str, Any]]] = []
        for candidate in interpretation_candidates:
            score, score_reason = self._calculate_interpretation_candidate_score_for_observation(
                observation=observation,
                source_facts=source_facts,
                interpretation=candidate,
                observation_id=observation_id,
            )
            if score <= 0.0:
                continue
            scored_interpretation_candidates.append((score, score_reason, candidate))
        scored_interpretation_candidates.sort(
            key=lambda entry: (
                entry[0],
                entry[2].get("updated_at") or "",
            ),
            reverse=True,
        )
        scored_interpretation_candidates = scored_interpretation_candidates[:6]

        if (
            scored_interpretation_candidates
            and scored_interpretation_candidates[0][1] == "existing_observation_evidence"
        ):
            score, score_reason, candidate = scored_interpretation_candidates[0]
            return {
                "decision": "evidence_only",
                "target_interpretation_id": int(candidate["id"]),
                "target_interpretation": candidate,
                "relationship": "support",
                "conflict_level": "none",
                "intrinsic_value": "medium",
                "reason": score_reason,
                "candidate_score": score,
            }

        candidate_payloads = []
        candidate_by_id: Dict[int, Dict[str, Any]] = {}
        for score, score_reason, candidate in scored_interpretation_candidates:
            candidate_id = int(candidate["id"])
            candidate_by_id[candidate_id] = candidate
            candidate_payloads.append({
                "id": candidate_id,
                "claim": candidate.get("claim"),
                "target_text": candidate.get("target_text"),
                "scope": candidate.get("scope"),
                "interpretation_type": candidate.get("interpretation_type"),
                "status": candidate.get("status"),
                "conflict_status": candidate.get("conflict_status"),
                "action_implication": candidate.get("action_implication"),
                "retrieval_score": round(float(score), 4),
                "retrieval_reason": score_reason,
            })

        fact_lines = []
        for index, fact in enumerate(source_facts[:12], 1):
            fact_id = fact.get("id", fact.get("fact_id"))
            summary = str(fact.get("summary") or "").strip()
            if fact_id is None or not summary:
                continue
            fact_lines.append(
                f"{index}. id={int(fact_id)} "
                f"[{fact.get('fact_type', 'semantic')}/"
                f"{fact.get('fact_subject', 'other')}/"
                f"{fact.get('fact_kind', 'other')}] {summary}"
            )
        observation_payload = {
            "id": observation_id,
            "entity_id": observation.get("entity_id"),
            "entity_name": observation.get("entity_name"),
            "topic_key": observation.get("topic_key"),
            "topic_label": observation.get("topic_label"),
            "observation_type": observation.get("observation_type"),
            "summary": observation.get("summary"),
            "metadata": self._json_dict(observation.get("metadata", {})),
        }
        prompt = INTERPRETATION_OBSERVATION_VALUE_PROMPT.format(
            observation=json.dumps(
                observation_payload,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
            source_facts="\n".join(fact_lines) or "(none)",
            candidate_interpretations=json.dumps(
                candidate_payloads,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
        )
        data = self._parse_json_object_from_llm_text(self._call_llm(prompt) or "")
        decision = str(data.get("decision") or "").strip().lower() if data else ""
        relationship = (
            str(data.get("relationship") or "unrelated").strip().lower()
            if data else "unrelated"
        )
        conflict_level = (
            str(data.get("conflict_level") or "none").strip().lower()
            if data else "none"
        )
        intrinsic_value = (
            str(data.get("intrinsic_value") or "medium").strip().lower()
            if data else "medium"
        )
        if decision not in {
            "update", "evidence_only", "unmatched", "defer", "ignore",
        }:
            if scored_interpretation_candidates and scored_interpretation_candidates[0][0] >= 0.78:
                decision = "update"
                relationship = "extend"
            else:
                decision = "unmatched"
                relationship = "unrelated"
        if relationship not in {
            "support", "extend", "revise", "contradict", "unrelated",
        }:
            relationship = "unrelated"
        if conflict_level not in {"none", "partial", "strong"}:
            conflict_level = "none"
        if intrinsic_value not in {"low", "medium", "high"}:
            intrinsic_value = "medium"

        try:
            target_id = int(data.get("target_interpretation_id")) if data else None
        except (TypeError, ValueError):
            target_id = None
        if decision in {"update", "evidence_only"}:
            if target_id not in candidate_by_id:
                decision = "unmatched"
                target_id = None
                relationship = "unrelated"
                conflict_level = "none"
            elif relationship == "unrelated":
                decision = "unmatched"
                target_id = None
        else:
            target_id = None
        if relationship == "contradict" and decision == "evidence_only":
            decision = "update"

        result = {
            "decision": decision,
            "target_interpretation_id": target_id,
            "target_interpretation": (
                candidate_by_id.get(target_id) if target_id is not None else None
            ),
            "relationship": relationship,
            "conflict_level": conflict_level,
            "intrinsic_value": intrinsic_value,
            "reason": str(data.get("reason") or "")[:512] if data else "",
            "candidate_score": next(
                (
                    score
                    for score, _, candidate in scored_interpretation_candidates
                    if int(candidate["id"]) == target_id
                ),
                0.0,
            ),
        }
        self._log_info(
            "memory_reflect",
            "observation_interpretation_value_judged",
            {
                "observation_id": observation_id,
                "candidate_interpretation_ids": [
                    payload["id"] for payload in candidate_payloads
                ],
                **{
                    key: value
                    for key, value in result.items()
                    if key != "target_interpretation"
                },
            },
        )
        return result

    def _persist_interpretation_assignments(
        self,
        *,
        interpretation: Dict[str, Any],
        assignments: List[Dict[str, Any]],
        updated_interpretation: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Persist links and optional content update for matched observations."""
        support_fact_ids: List[int] = []
        support_observation_ids: List[int] = []
        counter_fact_ids: List[int] = []
        counter_observation_ids: List[int] = []
        for assignment in assignments:
            item = assignment["item"]
            observation_id = int(item["observation_id"])
            source_fact_ids = [
                int(fact_id)
                for fact_id in item.get("source_fact_ids", [])
                if fact_id is not None
            ]
            if assignment.get("relationship") == "contradict":
                counter_fact_ids.extend(source_fact_ids)
                counter_observation_ids.append(observation_id)
            else:
                support_fact_ids.extend(source_fact_ids)
                support_observation_ids.append(observation_id)

        updated = updated_interpretation or {}
        evidence_fact_ids = list(dict.fromkeys([
            *interpretation.get("evidence_fact_ids", []),
            *support_fact_ids,
            *updated.get("evidence_fact_ids", []),
        ]))
        evidence_observation_ids = list(dict.fromkeys([
            *interpretation.get("evidence_observation_ids", []),
            *support_observation_ids,
            *updated.get("evidence_observation_ids", []),
        ]))
        counter_evidence_fact_ids = list(dict.fromkeys([
            *interpretation.get("counter_evidence_fact_ids", []),
            *counter_fact_ids,
            *updated.get("counter_evidence_fact_ids", []),
        ]))
        counter_evidence_observation_ids = list(dict.fromkeys([
            *interpretation.get("counter_evidence_observation_ids", []),
            *counter_observation_ids,
            *updated.get("counter_evidence_observation_ids", []),
        ]))
        metadata = {
            **self._json_dict(interpretation.get("metadata", {})),
            **self._json_dict(updated.get("metadata", {})),
        }
        metadata["observation_ids"] = list(dict.fromkeys([
            *[
                int(value)
                for value in metadata.get("observation_ids", [])
                if str(value).isdigit()
            ],
            *[
                int(assignment["item"]["observation_id"])
                for assignment in assignments
            ],
        ]))
        metadata["value_judgement"] = {
            "last_observation_ids": [
                int(assignment["item"]["observation_id"])
                for assignment in assignments
            ],
            "relationships": {
                str(assignment["item"]["observation_id"]): assignment.get(
                    "relationship",
                    "support",
                )
                for assignment in assignments
            },
            "content_updated": bool(updated_interpretation),
        }

        claim = updated.get("claim", interpretation.get("claim", ""))
        target_text = updated.get(
            "target_text",
            interpretation.get("target_text", ""),
        )
        scope = updated.get("scope", interpretation.get("scope", "general"))
        interpretation_type = updated.get(
            "interpretation_type",
            interpretation.get("interpretation_type", "insight"),
        )
        action_implication = updated.get(
            "action_implication",
            interpretation.get("action_implication", ""),
        )
        resolution = updated.get(
            "resolution",
            interpretation.get("resolution", ""),
        )
        embedding_text = self._build_interpretation_embedding_text(
            entity_name=(
                interpretation.get("entity_name")
                or metadata.get("entity_name")
                or assignments[0]["item"]["observation"].get("entity_name")
                or ""
            ),
            target_text=target_text,
            scope=scope,
            interpretation_type=interpretation_type,
            claim=claim,
            action_implication=action_implication,
            resolution=resolution,
        )
        interpretation_id = self._db.memory_upsert_interpretation(
            interpretation_id=int(interpretation["id"]),
            claim=claim,
            entity_id=(
                interpretation.get("entity_id")
                or metadata.get("entity_id")
                or assignments[0]["item"]["observation"].get("entity_id")
            ),
            subject_text=updated.get(
                "subject_text",
                interpretation.get("subject_text", ""),
            ),
            target_text=target_text,
            scope=scope,
            interpretation_type=interpretation_type,
            polarity=updated.get(
                "polarity",
                interpretation.get("polarity", "neutral"),
            ),
            strength=updated.get(
                "strength",
                interpretation.get("strength", 0.5),
            ),
            confidence=updated.get(
                "confidence",
                interpretation.get("confidence", 0.5),
            ),
            status=updated.get(
                "status",
                interpretation.get("status", "current"),
            ),
            conflict_status=updated.get(
                "conflict_status",
                interpretation.get("conflict_status", "none"),
            ),
            resolution=resolution,
            action_implication=action_implication,
            evidence_fact_ids=evidence_fact_ids,
            evidence_observation_ids=evidence_observation_ids,
            counter_evidence_fact_ids=counter_evidence_fact_ids,
            counter_evidence_observation_ids=counter_evidence_observation_ids,
            embedding=self._embed_memory_layer_text(embedding_text),
            embedding_text=embedding_text,
            metadata=metadata,
        )
        for assignment in assignments:
            relationship = assignment.get("relationship", "support")
            relation = (
                "contradict"
                if relationship == "contradict"
                else ("refine" if relationship in {"extend", "revise"} else "support")
            )
            self._db.memory_link_interpretation_observation(
                interpretation_id,
                int(assignment["item"]["observation_id"]),
                relation=relation,
                confidence=float(assignment.get("candidate_score") or 0.75),
            )
        return int(interpretation_id)

    @classmethod
    def _interpretation_basis_hash(
        cls,
        observation: Dict[str, Any],
        source_facts: List[Dict[str, Any]],
    ) -> str:
        """Hash the observation fields that matter for interpretation decisions."""
        normalized_metadata = cls._json_dict(observation.get("metadata", {}))
        source_fact_ids = []
        for fact in source_facts:
            fact_id = fact.get("id", fact.get("fact_id"))
            try:
                source_fact_ids.append(int(fact_id))
            except (TypeError, ValueError):
                continue
        basis = {
            "summary": str(observation.get("summary") or "").strip(),
            "keywords": cls._string_list(observation.get("keywords", []), limit=20),
            "entity_id": observation.get("entity_id"),
            "topic_key": observation.get("topic_key"),
            "topic_label": observation.get("topic_label"),
            "observation_type": observation.get("observation_type", "observation"),
            "metadata": {
                "evidence_shape": normalized_metadata.get("evidence_shape"),
                "temporal_scope": normalized_metadata.get("temporal_scope"),
                "candidate_interpretation_types": normalized_metadata.get("candidate_interpretation_types", []),
                "has_conflict": normalized_metadata.get("has_conflict", False),
                "source_fact_type_distribution": normalized_metadata.get("source_fact_type_distribution", {}),
                "dominant_fact_type": normalized_metadata.get("dominant_fact_type"),
                "evidence_mixture": normalized_metadata.get("evidence_mixture"),
            },
            "source_fact_ids": sorted(set(source_fact_ids)),
        }
        payload = json.dumps(basis, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _interpretation_state(metadata: Dict[str, Any]) -> str:
        state = str(metadata.get("interpretation_status") or "pending").strip().lower()
        if state not in {"pending", "deferred", "linked", "generated", "ignored"}:
            return "pending"
        return state

    @classmethod
    def _interpretation_state_is_final_for_basis(
        cls,
        metadata: Dict[str, Any],
        basis_hash: str,
    ) -> bool:
        return (
            cls._interpretation_state(metadata) in {"linked", "generated", "ignored"}
            and str(metadata.get("interpretation_basis_hash") or "") == basis_hash
        )

    def _update_observation_interpretation_state(
        self,
        observation: Dict[str, Any],
        *,
        status: str,
        basis_hash: str,
        reason: str,
        interpretation_id: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._db or not observation.get("id"):
            return
        metadata = self._json_dict(observation.get("metadata", {}))
        status = status if status in {"pending", "deferred", "linked", "generated", "ignored"} else "pending"
        metadata.update({
            "interpretation_status": status,
            "interpretation_basis_hash": basis_hash,
            "interpretation_checked_at": datetime.now(timezone.utc).isoformat(),
            "interpretation_reason": str(reason or "")[:256],
        })
        if interpretation_id is not None:
            linked_ids = []
            for value in metadata.get("linked_interpretation_ids", []):
                try:
                    linked_ids.append(int(value))
                except (TypeError, ValueError):
                    continue
            if int(interpretation_id) not in linked_ids:
                linked_ids.append(int(interpretation_id))
            metadata["linked_interpretation_ids"] = linked_ids
            if status == "generated":
                metadata["generated_interpretation_id"] = int(interpretation_id)
        if extra:
            metadata.update(extra)
        try:
            self._db.memory_update_observation_metadata(
                int(observation["id"]),
                metadata,
            )
        except AttributeError:
            self._log_info(
                "memory_reflect",
                "interpretation_state_update_unsupported", 
                {
                    "observation_id": observation.get("id"),
                    "status": status,
                    "reason": reason,
                }
            )
            return
        observation["metadata"] = metadata

    @classmethod
    def _interpretation_generation_trigger_priority(
        cls,
        observation: Dict[str, Any],
        source_facts: List[Dict[str, Any]],
        family: str,
    ) -> Tuple[str, str]:
        metadata = cls._json_dict(observation.get("metadata", {}))
        observation_type = str(
            observation.get("observation_type") or "context"
        ).strip().lower()
        evidence_shape = str(metadata.get("evidence_shape") or "single_event")
        temporal_scope = str(metadata.get("temporal_scope") or "recent")
        dominant_fact_type = str(metadata.get("dominant_fact_type") or "unknown")
        evidence_mixture = str(metadata.get("evidence_mixture") or "unknown")
        source_kinds = {
            str(fact.get("fact_kind") or "").strip().lower()
            for fact in source_facts
        }
        if family == "preference":
            if source_kinds & {"instruction"}:
                return "high", "explicit_instruction"
            if observation_type in {"preference_signal", "constraint"}:
                return "high", "preference_signal"
            if evidence_shape in {"repeated_pattern", "confirmation"} or temporal_scope in {"ongoing", "recurring"}:
                return "high", "stable_preference_signal"
            if evidence_mixture in {"semantic_dominant", "balanced_mixed"}:
                return "medium", "mixed_preference_evidence"
            return "medium", "weak_preference_signal"
        if family == "task":
            if observation_type in {"task_state", "task_progress", "decision"}:
                return "high", "task_state_signal"
            if any(cls._is_task_event_like_fact(fact) for fact in source_facts):
                return "high", "task_event_evidence"
            if dominant_fact_type == "episodic" or evidence_mixture in {"episodic_only", "episodic_dominant"}:
                return "medium", "episodic_task_context"
            return "medium", "weak_task_signal"
        if observation_type in {"problem", "task_progress", "decision"}:
            return "high", "material_insight_change"
        if evidence_shape in {"repeated_pattern", "contrast", "progression", "correction", "confirmation"}:
            return "medium", "structured_insight_evidence"
        if evidence_mixture in {"semantic_dominant", "episodic_dominant", "balanced_mixed"}:
            return "medium", "mixed_fact_type_evidence"
        return "low", "ordinary_insight"

    def _get_similar_deferred_observations(
        self,
        item: Dict[str, Any],
        seen_observation_ids: set[int],
    ) -> List[Dict[str, Any]]:
        if not self._db:
            return []
        semantic_observation = item["observation"]
        try:
            candidates = self._db.get_deferred_observations_for_interpretation(
                entity_id=semantic_observation.get("entity_id"),
                exclude_observation_ids=list(seen_observation_ids),
                limit=16,
            )
        except Exception:
            return []

        deferred_items: List[Dict[str, Any]] = []
        scored_candidates: List[Tuple[float, Dict[str, Any]]] = []
        for candidate in candidates:
            try:
                candidate_id = int(candidate["id"])
            except (TypeError, ValueError, KeyError):
                continue
            if candidate_id in seen_observation_ids:
                continue
            candidate_semantic_observation = self._build_semantic_observation(
                candidate
            )
            source_facts = self._db.memory_facts_by_ids(
                candidate.get("source_fact_ids", [])
            )
            family = self._observation_cluster_interpretation_family(
                candidate_semantic_observation,
                source_facts,
            )
            if family != item.get("family"):
                continue
            if str(candidate.get("observation_type") or "") != str(
                semantic_observation.get("observation_type") or ""
            ):
                continue
            similarity = self._cal_embedding_similarity(
                semantic_observation.get("embedding"),
                candidate_semantic_observation.get("embedding"),
            )
            if similarity < OBSERVATION_EMBEDDING_SIMILARITY_THRESHOLD:
                continue
            scored_candidates.append((similarity, {
                "candidate": candidate,
                "observation": candidate_semantic_observation,
                "source_facts": source_facts,
                "family": family,
            }))

        scored_candidates.sort(key=lambda entry: entry[0], reverse=True)
        for similarity, entry in scored_candidates[:8]:
            candidate = entry["candidate"]
            candidate_id = int(candidate["id"])
            semantic_candidate = entry["observation"]
            source_facts = entry["source_facts"]
            metadata = self._json_dict(semantic_candidate.get("metadata", {}))
            basis_hash = self._interpretation_basis_hash(
                semantic_candidate,
                source_facts,
            )
            if str(metadata.get("interpretation_basis_hash") or "") != basis_hash:
                continue
            seen_observation_ids.add(candidate_id)
            deferred_items.append({
                "observation": semantic_candidate,
                "observation_id": candidate_id,
                "evidence_bundle_id": int(candidate["evidence_bundle_id"]),
                "source_facts": source_facts,
                "source_fact_ids": [int(fact["id"]) for fact in source_facts if fact.get("id") is not None],
                "basis_hash": basis_hash,
                "family": entry["family"],
                "priority": "deferred",
                "priority_reason": (
                    f"previously_deferred_similarity_{similarity:.3f}"
                ),
                "is_deferred_context": True,
            })
        return deferred_items

    def _cluster_observation_items_for_interpretation(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        buckets: Dict[Tuple[str, Any, str], Dict[str, Any]] = {}
        for item in items:
            observation = item["observation"]
            source_facts = item.get("source_facts", [])
            family = self._observation_cluster_interpretation_family(observation, source_facts)
            topic_key = self._topic_key(observation.get("topic_key") or observation.get("topic_label") or "general")
            cluster_topic = topic_key or "general"
            observation_type = str(
                observation.get("observation_type") or "context"
            )
            key = (
                family,
                observation.get("entity_id"),
                cluster_topic,
            )
            bucket = buckets.setdefault(key, {
                "family": family,
                "entity_id": observation.get("entity_id"),
                "topic_key": cluster_topic,
                "observation_types": [],
                "items": [],
            })
            if observation_type not in bucket["observation_types"]:
                bucket["observation_types"].append(observation_type)
            bucket["items"].append(item)
        clusters = list(buckets.values())
        clusters.sort(key=lambda cluster: (len(cluster["items"]), cluster["family"], cluster["topic_key"]), reverse=True)
        return clusters

    @staticmethod
    def _dedupe_source_facts(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set[int] = set()
        for fact in facts:
            fact_id = fact.get("id", fact.get("fact_id"))
            try:
                int_fact_id = int(fact_id)
            except (TypeError, ValueError):
                continue
            if int_fact_id in seen:
                continue
            seen.add(fact_id)
            out.append(fact)
        return out

    def _generate_interpretation_from_observation_cluster(
        self,
        cluster: Dict[str, Any],
    ) -> Optional[int]:
        items = cluster.get("items") or []
        if not items:
            return None
        family = str(cluster.get("family") or "insight")
        all_source_facts = self._dedupe_source_facts([
            fact
            for item in items
            for fact in item.get("source_facts", [])
        ])
        observation_ids = [
            int(item["observation_id"])
            for item in items
            if item.get("observation_id") is not None
        ]
        observation_ids = list(dict.fromkeys(observation_ids))
        if not observation_ids:
            return None

        if len(items) == 1:
            observation = items[0]["observation"]
            interpretation = self._generate_interpretation(
                observation=observation,
                source_facts=all_source_facts,
                observation_id=observation_ids[0],
                observation_ids=observation_ids,
            )
            representative = observation
        else:
            summaries = []
            for index, item in enumerate(items, 1):
                observation = item["observation"]
                summary = str(observation.get("summary") or "").strip()
                if summary:
                    summaries.append(f"{index}. id={item['observation_id']} {summary}")
            representative = dict(items[0]["observation"])
            representative_metadata = self._json_dict(
                representative.get("metadata", {})
            )
            representative["summary"] = "Clustered observations:\n" + "\n".join(summaries)
            representative["metadata"] = {
                **representative_metadata,
                "interpretation_cluster_family": family,
                "clustered_observation_ids": observation_ids,
            }
            interpretation = self._generate_interpretation(
                observation=representative,
                source_facts=all_source_facts,
                observation_id=observation_ids[0],
                observation_ids=observation_ids,
            )
        if not interpretation:
            return None

        source_fact_ids = [int(fact["id"]) for fact in all_source_facts if fact.get("id") is not None]
        metadata = {
            **(interpretation["metadata"] or {}),
            "observation_id": int(observation_ids[0]),
            "observation_ids": observation_ids,
            "entity_id": representative.get("entity_id"),
            "entity_name": representative.get("entity_name"),
            "topic_key": representative.get("topic_key"),
            "topic_label": representative.get("topic_label"),
            "observation_type": representative.get("observation_type", "observation"),
            "interpretation_cluster_family": family,
        }
        embedding_text = self._build_interpretation_embedding_text(
            entity_name=representative.get("entity_name") or "",
            target_text=interpretation["target_text"],
            scope=interpretation["scope"],
            interpretation_type=interpretation["interpretation_type"],
            claim=interpretation["claim"],
            action_implication=interpretation["action_implication"],
            resolution=interpretation["resolution"],
        )
        interpretation_id = self._db.memory_upsert_interpretation(
            claim=interpretation["claim"],
            entity_id=representative.get("entity_id"),
            subject_text=interpretation["subject_text"],
            target_text=interpretation["target_text"],
            scope=interpretation["scope"],
            interpretation_type=interpretation["interpretation_type"],
            polarity=interpretation["polarity"],
            strength=interpretation["strength"],
            confidence=interpretation["confidence"],
            status=interpretation["status"],
            conflict_status=interpretation["conflict_status"],
            resolution=interpretation["resolution"],
            action_implication=interpretation["action_implication"],
            evidence_fact_ids=interpretation["evidence_fact_ids"] or source_fact_ids,
            evidence_observation_ids=interpretation["evidence_observation_ids"] or observation_ids,
            counter_evidence_fact_ids=interpretation["counter_evidence_fact_ids"],
            counter_evidence_observation_ids=interpretation["counter_evidence_observation_ids"],
            embedding=self._embed_memory_layer_text(embedding_text),
            embedding_text=embedding_text,
            metadata=metadata,
        )
        for observation_id in observation_ids:
            self._db.memory_link_interpretation_observation(
                interpretation_id,
                observation_id,
                confidence=interpretation["confidence"],
            )
        self._log_info(
            "memory_reflect",
            "interpretation_generated", 
            {
                "interpretation_id": interpretation_id,
                "observation_ids": observation_ids,
                "source_fact_ids": source_fact_ids,
                "cluster_family": family,
                "generated_interpretation": interpretation,
            }
        )
        return int(interpretation_id)

    @classmethod
    def _judge_observation_clustering_should_run(
        cls,
        cluster: Dict[str, Any],
    ) -> Tuple[bool, str]:
        items = cluster.get("items") or []
        changed_items = [item for item in items if not item.get("is_deferred_context")]
        if not changed_items:
            return False, "no_changed_observation"
        if len(items) == 1 and len(changed_items) == 1:
            item = changed_items[0]
            return cls._single_observation_generation_allowed(
                observation=item["observation"],
                source_facts=item.get("source_facts", []),
                interpretation_family=str(item.get("family") or cluster.get("family") or "insight"),
            )
        if any(item.get("priority") == "high" for item in changed_items):
            return True, "high_priority_observation"
        if len(changed_items) >= INTERPRETATION_MIN_OBSERVATIONS_FOR_BATCH:
            return True, "cluster_changed_batch_threshold"
        if len(items) >= INTERPRETATION_MIN_CLUSTER_SIZE:
            return True, "cluster_size_threshold"
        return False, "trigger_threshold_not_met"

    def _build_semantic_observation(
        self,
        observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the semantic unit consumed by interpretation generation."""
        observation_type = str(
            observation.get("observation_type") or "context"
        ).strip().lower()
        metadata = self._json_dict(observation.get("metadata", {}))
        source_count = max(
            1,
            int(metadata.get("source_count") or len(
                observation.get("source_fact_ids", [])
            ) or 1),
        )
        evidence_mode = str(
            observation.get("evidence_mode") or "aggregated"
        ).strip().lower()
        evidence_shape = (
            "repeated_pattern"
            if observation_type in {"behavior_pattern", "preference_signal"}
            and source_count >= 2
            else (
                "progression"
                if observation_type in {
                    "task_state",
                    "task_progress",
                    "decision",
                }
                and source_count >= 2
                else "single_event"
            )
        )
        temporal_scope = (
            "recurring"
            if evidence_mode == "behavioral"
            else ("ongoing" if evidence_mode in {"explicit", "semantic"} else "recent")
        )
        fact_type_distribution = metadata.get(
            "fact_type_distribution",
            {"semantic": 0, "episodic": 0},
        )
        dominant_fact_type, evidence_mixture = (
            self._fact_type_evidence_summary(
                self._metadata_fact_type_distribution(
                    fact_type_distribution
                )
            )
        )
        allowed_interpretation_types = metadata.get(
            "allowed_interpretation_types",
            self._observation_allowed_interpretation_types(
                observation_type,
                evidence_mode,
            ),
        )
        summary = str(observation.get("summary") or "").strip()
        return {
            "id": int(observation["id"]),
            "evidence_bundle_id": int(observation["evidence_bundle_id"]),
            "entity_id": observation.get("entity_id"),
            "entity_name": observation.get("entity_name"),
            "topic_key": observation.get("topic_key"),
            "topic_label": observation.get("topic_label"),
            "observation_type": observation_type,
            "source_fact_ids": [
                int(fact_id)
                for fact_id in observation.get("source_fact_ids", [])
            ],
            "summary": summary,
            "keywords": summary,
            "confidence": observation.get("confidence", 0.5),
            "embedding": observation.get("embedding"),
            "embedding_text": observation.get("embedding_text") or summary,
            "metadata": {
                **metadata,
                "observation_id": int(observation["id"]),
                "observation_type": observation_type,
                "evidence_mode": evidence_mode,
                "evidence_shape": evidence_shape,
                "temporal_scope": temporal_scope,
                "candidate_interpretation_types": (
                    self._observation_candidate_families(observation_type)
                ),
                "allowed_interpretation_types": allowed_interpretation_types,
                "source_fact_type_distribution": fact_type_distribution,
                "dominant_fact_type": dominant_fact_type,
                "evidence_mixture": evidence_mixture,
            },
        }

    def _reflect_generate_interpretations_using_observations(
        self,
        evidence_bundle_ids: List[int],
    ) -> int:
        if not self._db:
            return 0
        clean_ids = list(dict.fromkeys(
            int(evidence_bundle_id)
            for evidence_bundle_id in evidence_bundle_ids
            if evidence_bundle_id is not None
        ))
        if not clean_ids:
            return 0
        observations = self._db.get_observations_for_evidence_bundles(clean_ids)
        generated = 0
        candidate_items: List[Dict[str, Any]] = []
        semantic_observations = [
            self._build_semantic_observation(observation)
            for observation in observations
        ]
        for semantic_observation in semantic_observations:
            observation_id = int(semantic_observation["id"])
            source_facts = self._db.memory_facts_by_ids(
                semantic_observation.get("source_fact_ids", [])
            )
            source_fact_ids = [
                int(fact["id"])
                for fact in source_facts
                if fact.get("id") is not None
            ]
            metadata = self._json_dict(
                semantic_observation.get("metadata", {})
            )
            basis_hash = self._interpretation_basis_hash(
                semantic_observation,
                source_facts,
            )
            if self._interpretation_state_is_final_for_basis(
                metadata,
                basis_hash,
            ):
                self._log_info(
                    "memory_reflect",
                    "interpretation_semantic_unit_skipped",
                    {
                        "observation_id": observation_id,
                        "status": self._interpretation_state(metadata),
                        "reason": "basis_already_processed",
                    },
                )
                continue
            if (
                self._interpretation_state(metadata) == "deferred"
                and str(metadata.get("interpretation_basis_hash") or "")
                == basis_hash
            ):
                self._log_info(
                    "memory_reflect",
                    "interpretation_semantic_unit_skipped",
                    {
                        "observation_id": observation_id,
                        "status": "deferred",
                        "reason": "unchanged_deferred_context",
                    },
                )
                continue
            semantic_observation["metadata"] = metadata
            family = self._observation_cluster_interpretation_family(
                semantic_observation,
                source_facts,
            )
            priority, priority_reason = (
                self._interpretation_generation_trigger_priority(
                    observation=semantic_observation,
                    source_facts=source_facts,
                    family=family,
                )
            )
            candidate_items.append({
                "observation": semantic_observation,
                "observation_id": int(semantic_observation["id"]),
                "source_facts": source_facts,
                "source_fact_ids": source_fact_ids,
                "basis_hash": basis_hash,
                "family": family,
                "priority": priority,
                "priority_reason": priority_reason,
                "is_deferred_context": False,
            })

        if not candidate_items:
            return 0

        llm_calls_used = 0
        assignments_by_interpretation: Dict[int, Dict[str, Any]] = {}
        remaining_items: List[Dict[str, Any]] = []
        for item in candidate_items:
            judgement = self._judge_observation_value_for_interpretation(item)
            decision = str(judgement.get("decision") or "unmatched")
            judgement_extra = {
                "interpretation_value_decision": decision,
                "interpretation_value_relationship": judgement.get(
                    "relationship",
                ),
                "interpretation_value_conflict_level": judgement.get(
                    "conflict_level",
                ),
                "interpretation_value_intrinsic": judgement.get(
                    "intrinsic_value",
                ),
                "interpretation_value_reason": judgement.get("reason"),
            }
            if decision in {"update", "evidence_only"}:
                target = judgement.get("target_interpretation")
                target_id = judgement.get("target_interpretation_id")
                if target is None or target_id is None:
                    remaining_items.append(item)
                    continue
                group = assignments_by_interpretation.setdefault(
                    int(target_id),
                    {
                        "interpretation": target,
                        "assignments": [],
                        "needs_update": False,
                    },
                )
                group["assignments"].append({
                    **judgement,
                    "item": item,
                    "state_extra": judgement_extra,
                })
                if decision == "update":
                    group["needs_update"] = True
                continue
            if decision == "ignore":
                self._update_observation_interpretation_state(
                    item["observation"],
                    status="ignored",
                    basis_hash=item["basis_hash"],
                    reason=str(judgement.get("reason") or "value_judgement_ignore"),
                    extra=judgement_extra,
                )
                continue
            remaining_items.append({
                **item,
                "value_judgement": judgement,
            })

        for target_id, group in assignments_by_interpretation.items():
            assignments = group["assignments"]
            updated_interpretation = None
            if group["needs_update"]:
                if llm_calls_used >= INTERPRETATION_MAX_LLM_CALLS_PER_REFLECT:
                    for assignment in assignments:
                        item = assignment["item"]
                        self._update_observation_interpretation_state(
                            item["observation"],
                            status="deferred",
                            basis_hash=item["basis_hash"],
                            reason="llm_budget_exhausted_before_batch_update",
                            extra=assignment["state_extra"],
                        )
                    continue
                updated_interpretation = (
                    self._update_existing_interpretation_from_observations(
                        interpretation=group["interpretation"],
                        assignments=assignments,
                    )
                )
                llm_calls_used += 1
                if updated_interpretation is None:
                    for assignment in assignments:
                        item = assignment["item"]
                        self._update_observation_interpretation_state(
                            item["observation"],
                            status="deferred",
                            basis_hash=item["basis_hash"],
                            reason="batch_update_not_applied",
                            extra=assignment["state_extra"],
                        )
                    continue

            interpretation_id = self._persist_interpretation_assignments(
                interpretation=group["interpretation"],
                assignments=assignments,
                updated_interpretation=updated_interpretation,
            )
            for assignment in assignments:
                item = assignment["item"]
                self._update_observation_interpretation_state(
                    item["observation"],
                    status="linked",
                    basis_hash=item["basis_hash"],
                    reason=(
                        "value_judgement_batch_update"
                        if updated_interpretation
                        else "value_judgement_evidence_only"
                    ),
                    interpretation_id=interpretation_id,
                    extra=assignment["state_extra"],
                )
            generated += len(assignments)
            self._log_info(
                "memory_reflect",
                "interpretation_assignments_applied",
                {
                    "interpretation_id": target_id,
                    "observation_ids": [
                        int(assignment["item"]["observation_id"])
                        for assignment in assignments
                    ],
                    "content_updated": bool(updated_interpretation),
                },
            )

        if not remaining_items:
            return generated

        seen_observation_ids = {
            int(item["observation_id"]) for item in candidate_items
        }
        cluster_context_items = list(remaining_items)
        for item in list(remaining_items):
            cluster_context_items.extend(
                self._get_similar_deferred_observations(
                    item,
                    seen_observation_ids,
                )
            )

        for cluster in self._cluster_observation_items_for_interpretation(
            cluster_context_items
        ):
            should_run, reason = self._judge_observation_clustering_should_run(cluster)
            changed_items = [
                item
                for item in cluster.get("items") or []
                if not item.get("is_deferred_context")
            ]
            if not should_run:
                for item in changed_items:
                    judgement = item.get("value_judgement") or {}
                    self._update_observation_interpretation_state(
                        item["observation"],
                        status="deferred",
                        basis_hash=item["basis_hash"],
                        reason=reason,
                        extra={
                            "interpretation_priority": item.get("priority"),
                            "interpretation_priority_reason": item.get("priority_reason"),
                            "interpretation_value_decision": judgement.get("decision"),
                            "interpretation_value_relationship": judgement.get("relationship"),
                            "interpretation_value_reason": judgement.get("reason"),
                        },
                    )
                continue

            if llm_calls_used >= INTERPRETATION_MAX_LLM_CALLS_PER_REFLECT:
                for item in changed_items:
                    judgement = item.get("value_judgement") or {}
                    self._update_observation_interpretation_state(
                        item["observation"],
                        status="deferred",
                        basis_hash=item["basis_hash"],
                        reason="llm_budget_exhausted",
                        extra={
                            "interpretation_value_decision": judgement.get("decision"),
                            "interpretation_value_relationship": judgement.get("relationship"),
                            "interpretation_value_reason": judgement.get("reason"),
                        },
                    )
                continue
            
            interpretation_id = self._generate_interpretation_from_observation_cluster(cluster)
            llm_calls_used += 1
            if interpretation_id is not None:
                generated += 1
                for item in cluster.get("items") or []:
                    judgement = item.get("value_judgement") or {}
                    self._update_observation_interpretation_state(
                        item["observation"],
                        status="generated",
                        basis_hash=item["basis_hash"],
                        reason="generated_from_observation_cluster",
                        interpretation_id=interpretation_id,
                        extra={
                            "interpretation_cluster_family": cluster.get("family"),
                            "interpretation_cluster_topic": cluster.get("topic_key"),
                            "interpretation_value_decision": judgement.get("decision"),
                            "interpretation_value_relationship": judgement.get("relationship"),
                            "interpretation_value_reason": judgement.get("reason"),
                        },
                    )
            else:
                for item in changed_items:
                    judgement = item.get("value_judgement") or {}
                    self._update_observation_interpretation_state(
                        item["observation"],
                        status="deferred",
                        basis_hash=item["basis_hash"],
                        reason="generation_not_created",
                        extra={
                            "interpretation_cluster_family": cluster.get("family"),
                            "interpretation_cluster_topic": cluster.get("topic_key"),
                            "interpretation_value_decision": judgement.get("decision"),
                            "interpretation_value_relationship": judgement.get("relationship"),
                            "interpretation_value_reason": judgement.get("reason"),
                        },
                    )
        return generated

    def _candidate_evidence_bundles_for_fact_cluster(
        self,
        cluster: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not self._db:
            return []
        try:
            entity_id = int(cluster["entity_id"])
        except (KeyError, TypeError, ValueError):
            return []
        try:
            return self._db.get_evidence_bundles_using_entity_topic(
                entity_id=entity_id,
                topic_key=self._topic_key(cluster.get("topic_key") or "general"),
            )
        except Exception:
            return []

    def _match_fact_cluster_to_existing_evidence_bundle(
        self,
        cluster: Dict[str, Any],
    ) -> Optional[Tuple[Dict[str, Any], float, str, List[Dict[str, Any]]]]:
        candidates = self._candidate_evidence_bundles_for_fact_cluster(cluster)
        if not candidates:
            return None
        evidence_bundle = candidates[0]
        evidence_bundle_id = int(evidence_bundle["id"])
        supporting_facts = self._db.get_evidence_bundle_supporting_facts(
            [evidence_bundle_id],
            per_evidence_bundle=8,
        ).get(evidence_bundle_id, [])
        return evidence_bundle, 1.0, "exact_entity_topic", supporting_facts

    def _update_existing_evidence_bundle_from_fact_cluster(
        self,
        cluster: Dict[str, Any],
        *,
        consumed_fact_ids: set[int],
        changed_evidence_bundle_ids: Optional[List[int]] = None,
    ) -> Optional[int]:
        if not self._db:
            return None
        source_facts = list(cluster.get("source_facts") or [])
        source_fact_ids = [
            int(fact_id)
            for fact_id in cluster.get("source_fact_ids", [])
            if fact_id is not None
        ]
        overlapping_fact_ids = consumed_fact_ids.intersection(source_fact_ids)
        cluster_for_match = cluster
        if overlapping_fact_ids:
            source_fact_ids = [
                fact_id
                for fact_id in source_fact_ids
                if fact_id not in overlapping_fact_ids
            ]
            remaining_fact_ids = set(source_fact_ids)
            source_facts = [
                fact
                for fact in source_facts
                if self._fact_id(fact) in remaining_fact_ids
            ]
            cluster_for_match = {
                **cluster,
                "source_facts": source_facts,
                "source_fact_ids": source_fact_ids,
            }
        if not source_fact_ids:
            return None
        match = self._match_fact_cluster_to_existing_evidence_bundle(cluster_for_match)
        if not match:
            return None
        existing_bundle, score, reason, supporting_nodes = match
        evidence_bundle_id = int(existing_bundle["id"])
        existing_source_ids = self._db.memory_evidence_bundle_source_ids(
            evidence_bundle_id
        )
        pending_source_ids = [
            fact_id
            for fact_id in source_fact_ids
            if fact_id not in existing_source_ids
        ]
        if not pending_source_ids:
            consumed_fact_ids.update(source_fact_ids)
            return evidence_bundle_id

        metadata = self._json_dict(existing_bundle.get("metadata", {}))
        topic_aliases = {
            str(alias).strip()
            for alias in metadata.get("topic_aliases", [])
            if str(alias or "").strip()
        }
        topic_aliases.update(
            str(alias).strip()
            for alias in cluster.get("topic_aliases", [])
            if str(alias or "").strip()
        )
        canonical_topic = str(
            existing_bundle.get("topic_label")
            or existing_bundle.get("topic_key")
            or cluster.get("topic_label")
            or cluster.get("topic_key")
            or "general"
        )
        topic_aliases.add(canonical_topic)
        metadata.update({
            "canonical_topic": canonical_topic,
            "topic_aliases": sorted(topic_aliases),
            "topic_embedding_text": canonical_topic,
        })
        stored_source_ids = list(dict.fromkeys(existing_source_ids + pending_source_ids))
        self._db.memory_replace_evidence_bundle_group(
            keep_evidence_bundle_id=evidence_bundle_id,
            remove_evidence_bundle_ids=[],
            bundle_type=str(existing_bundle.get("bundle_type") or "entity_topic"),
            source_fact_ids=stored_source_ids,
            metadata=metadata,
            source_roles={
                fact_id: "matched"
                for fact_id in pending_source_ids
            },
        )
        canonical_topic_embedding = (
            existing_bundle.get("canonical_topic_embedding")
            if existing_bundle.get("canonical_topic_embedding") is not None
            else cluster.get("canonical_topic_embedding")
        )
        self._db.memory_update_evidence_bundle_topic(
            evidence_bundle_id,
            canonical_topic_embedding=canonical_topic_embedding,
            metadata=metadata,
        )
        if changed_evidence_bundle_ids is not None:
            changed_evidence_bundle_ids.append(evidence_bundle_id)
        consumed_fact_ids.update(source_fact_ids)
        self._log_info(
            "memory_reflect",
            "fact_cluster_evidence_bundle_matched", {
            "evidence_bundle_id": evidence_bundle_id,
            "entity_id": cluster.get("entity_id"),
            "topic_key": cluster.get("topic_key"),
            "topic_aliases": sorted(topic_aliases),
            "topic_match_reasons": cluster.get("topic_match_reasons", []),
            "source_fact_ids": source_fact_ids,
            "score": score,
            "reason": reason,
            "supporting_facts": self._reflect_fact_log_items(supporting_nodes + source_facts),
            "updated_evidence_bundle": {
                **self._reflect_evidence_bundle_log_item(existing_bundle),
                "metadata": metadata,
            },
        })
        return evidence_bundle_id

    def _cluster_unprocessed_facts(
        self,
        facts: List[Dict[str, Any]],
        *,
        excluded_fact_ids: set[int],
    ) -> List[Dict[str, Any]]:
        prepared_facts: List[Dict[str, Any]] = []
        topics_by_entity: Dict[int, List[str]] = {}
        embedding_cache: Dict[str, Optional[np.ndarray]] = {}
        for fact in facts:
            fact_id = self._fact_id(fact)
            if fact_id is None or fact_id in excluded_fact_ids:
                continue
            try:
                entity_id = int(
                    fact.get("primary_entity_id")
                    or next(iter(self._fact_entity_pairs(fact)))[0]
                )
            except (StopIteration, TypeError, ValueError):
                continue
            entity_name = str(
                fact.get("primary_entity_name")
                or next(
                    (
                        name
                        for candidate_id, name in self._fact_entity_pairs(fact)
                        if int(candidate_id) == entity_id
                    ),
                    "",
                )
            )
            raw_topic = str(
                fact.get("primary_topic")
                or next(iter(fact.get("topics", []) or []), "general")
            ).strip() or "general"
            prepared_facts.append({
                "fact": fact,
                "fact_id": fact_id,
                "entity_id": entity_id,
                "entity_name": entity_name,
                "raw_topic": raw_topic,
            })
            topics_by_entity.setdefault(entity_id, []).append(raw_topic)

        topic_resolutions = {
            entity_id: self._canonicalize_topics_for_entity(
                entity_id,
                raw_topics,
                embedding_cache,
            )
            for entity_id, raw_topics in topics_by_entity.items()
        }
        buckets: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for prepared in prepared_facts:
            fact = prepared["fact"]
            fact_id = prepared["fact_id"]
            entity_id = prepared["entity_id"]
            entity_name = prepared["entity_name"]
            raw_topic = prepared["raw_topic"]
            resolution = topic_resolutions[entity_id].get(
                self._topic_key(raw_topic),
                {
                    "topic_key": self._topic_key(raw_topic),
                    "topic_label": self._topic_key(raw_topic),
                    "canonical_topic_embedding": None,
                    "topic_alias": raw_topic,
                    "topic_match_reason": "fallback_exact_topic",
                    "topic_similarity": 0.0,
                },
            )
            topic_key = str(resolution["topic_key"])
            key = (entity_id, topic_key)
            bucket = buckets.setdefault(
                key,
                {
                    "entity_id": entity_id,
                    "entity_name": entity_name,
                    "topic_key": topic_key,
                    "topic_label": str(resolution["topic_label"]),
                    "canonical_topic_embedding": resolution.get(
                        "canonical_topic_embedding"
                    ),
                    "topic_aliases": set(),
                    "topic_match_reasons": set(),
                    "facts": [],
                    "fact_ids": set(),
                },
            )
            bucket["topic_aliases"].add(str(resolution["topic_alias"]))
            bucket["topic_match_reasons"].add(
                str(resolution["topic_match_reason"])
            )
            if fact_id not in bucket["fact_ids"]:
                bucket["fact_ids"].add(fact_id)
                bucket["facts"].append(fact)

        clusters: List[Dict[str, Any]] = []
        for bucket in buckets.values():
            facts_for_cluster = sorted(
                bucket.get("facts", []),
                key=lambda fact: (str(fact.get("time_key") or ""), self._fact_id(fact) or 0),
            )
            clusters.append({
                **{
                    key: value
                    for key, value in bucket.items()
                    if key not in {
                        "facts",
                        "fact_ids",
                        "topic_aliases",
                        "topic_match_reasons",
                    }
                },
                "topic_aliases": sorted(bucket["topic_aliases"]),
                "topic_match_reasons": sorted(
                    bucket["topic_match_reasons"]
                ),
                "source_facts": facts_for_cluster,
                "source_fact_ids": [
                    self._fact_id(fact)
                    for fact in facts_for_cluster
                    if self._fact_id(fact) is not None
                ],
                "can_create_evidence_bundle": len(facts_for_cluster) >= MIN_FACTS_FOR_NEW_EVIDENCE_BUNDLE,
            })

        clusters.sort(
            key=lambda item: (
                1 if item.get("can_create_evidence_bundle") else 0,
                len(item.get("source_fact_ids", [])),
                str(item.get("topic_key") or ""),
            ),
            reverse=True,
        )
        return clusters

    def _generate_evidence_bundle_using_unmatched_fact_clusters(
        self,
        cluster: Dict[str, Any],
        *,
        consumed_fact_ids: set[int],
        changed_evidence_bundle_ids: Optional[List[int]] = None,
    ) -> Optional[int]:
        if not self._db:
            return None
        if not cluster.get("can_create_evidence_bundle", True):
            return None
        source_facts = list(cluster.get("source_facts") or [])
        source_fact_ids = [
            int(fact_id)
            for fact_id in cluster.get("source_fact_ids", [])
            if fact_id is not None
        ]
        overlapping_fact_ids = consumed_fact_ids.intersection(source_fact_ids)
        if overlapping_fact_ids:
            source_fact_ids = [
                fact_id
                for fact_id in source_fact_ids
                if fact_id not in overlapping_fact_ids
            ]
            remaining_fact_ids = set(source_fact_ids)
            source_facts = [
                fact
                for fact in source_facts
                if self._fact_id(fact) in remaining_fact_ids
            ]

        entity_id = int(cluster["entity_id"])
        topic_key = str(cluster.get("topic_key") or "general")
        topic_label = str(cluster.get("topic_label") or topic_key)
        topic_aliases = {
            str(alias).strip()
            for alias in cluster.get("topic_aliases", [])
            if str(alias or "").strip()
        }
        topic_aliases.add(topic_label)
        bundle_metadata = {
            "canonical_topic": topic_label,
            "topic_aliases": sorted(topic_aliases),
            "topic_embedding_text": topic_label,
        }
        evidence_bundle_id = self._db.memory_upsert_evidence_bundle(
            entity_id=entity_id,
            topic_key=topic_key,
            topic_label=topic_label,
            canonical_topic_embedding=cluster.get(
                "canonical_topic_embedding"
            ),
            source_fact_ids=source_fact_ids,
            bundle_type="entity_topic",
            metadata=bundle_metadata,
            source_role="initial",
        )
        if changed_evidence_bundle_ids is not None:
            changed_evidence_bundle_ids.append(int(evidence_bundle_id))
        consumed_fact_ids.update(source_fact_ids)
        self._log_info(
            "memory_reflect",
            "fact_cluster_evidence_bundle_generated",
            {
                "evidence_bundle_id": evidence_bundle_id,
                "entity_id": entity_id,
                "entity_name": cluster.get("entity_name"),
                "topic_key": topic_key,
                "topic_aliases": bundle_metadata["topic_aliases"],
                "topic_match_reasons": cluster.get(
                    "topic_match_reasons",
                    [],
                ),
                "source_fact_ids": source_fact_ids,
                "source_facts": self._reflect_fact_log_items(source_facts),
                "generated_evidence_bundle": {
                    "bundle_type": "entity_topic",
                    "metadata": bundle_metadata,
                },
            }
        )
        return int(evidence_bundle_id)
    
    def _reflect_generate_evidence_bundles_using_facts(
        self,
        *,
        limit: int,
        date_key: Optional[str] = None,
        changed_evidence_bundle_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Generate or update evidence bundles from unprocessed facts."""
        if not self._db:
            return {"candidate_count": 0, "consolidated": 0}

        unprocessed_fact_candidates = self._db.get_unprocessed_facts_for_evidence_bundle(
            date_key=date_key,
            limit=limit,
        )
        touched_entity_ids = list(dict.fromkeys(
            int(entity_id)
            for item in unprocessed_fact_candidates
            for entity_id, _entity_name in item.get("linked_entities", [])
        ))
        self._log_info(
            "memory_reflect",
            "fact_candidates_for_evidence_bundle",
            {
                "limit": limit,
                "candidate_count": len(unprocessed_fact_candidates),
                "touched_entity_ids": touched_entity_ids,
                "facts": self._reflect_fact_log_items(unprocessed_fact_candidates, limit=limit),
            }
        )
        consolidated = 0
        entity_topic_updates = 0
        fact_cluster_evidence_bundle_matches = 0
        fact_cluster_evidence_bundle_fact_ids: set[int] = set()
        fact_clusters_consolidated = 0
        fact_cluster_fact_ids: set[int] = set()
        consumed_fact_ids: set[int] = set()
        entity_topic_fact_ids: set[int] = set()
        changed_ids = (
            changed_evidence_bundle_ids
            if changed_evidence_bundle_ids is not None
            else []
        )
        clusters = self._cluster_unprocessed_facts(
            unprocessed_fact_candidates,
            excluded_fact_ids=consumed_fact_ids,
        )
        self._log_info(
            "memory_reflect",
            "fact_cluster_candidates_for_evidence_bundle",
            {
                "cluster_count": len(clusters),
                "clusters": [
                    {
                        "entity_id": cluster.get("entity_id"),
                        "entity_name": cluster.get("entity_name"),
                        "topic_key": cluster.get("topic_key"),
                        "can_create_evidence_bundle": cluster.get("can_create_evidence_bundle"),
                        "source_fact_ids": cluster.get("source_fact_ids", []),
                    }
                    for cluster in clusters
                ],
            })
        for cluster in clusters:
            before_cluster_consumed_fact_ids = set(consumed_fact_ids)
            try:
                matched_bundle_id = self._update_existing_evidence_bundle_from_fact_cluster(
                    cluster,
                    consumed_fact_ids=consumed_fact_ids,
                    changed_evidence_bundle_ids=changed_ids,
                )
            except Exception as exc:
                logger.debug(
                    "Failed to match fact cluster entity %s topic %s to evidence bundle: %s",
                    cluster.get("entity_id"),
                    cluster.get("topic_key"),
                    exc,
                )
                matched_bundle_id = None
            if matched_bundle_id is not None:
                consolidated += 1
                current_consumed_fact_ids = consumed_fact_ids - before_cluster_consumed_fact_ids
                fact_cluster_evidence_bundle_matches += 1
                fact_cluster_evidence_bundle_fact_ids.update(current_consumed_fact_ids)
                entity_topic_fact_ids.update(current_consumed_fact_ids)
                continue

            try:
                evidence_bundle_id = self._generate_evidence_bundle_using_unmatched_fact_clusters(
                    cluster,
                    consumed_fact_ids=consumed_fact_ids,
                    changed_evidence_bundle_ids=changed_ids,
                )
            except Exception as exc:
                logger.debug(
                    "Failed to generate evidence bundle from fact cluster entity %s topic %s: %s",
                    cluster.get("entity_id"),
                    cluster.get("topic_key"),
                    exc,
                )
                evidence_bundle_id = None
            if evidence_bundle_id is None:
                continue
            consolidated += 1
            fact_clusters_consolidated += 1

            current_consumed_fact_ids = consumed_fact_ids - before_cluster_consumed_fact_ids
            fact_cluster_fact_ids.update(current_consumed_fact_ids)
            entity_topic_fact_ids.update(current_consumed_fact_ids)
                
        return {
            "candidate_count": len(unprocessed_fact_candidates),
            "consolidated": consolidated,
            "entity_topic_updates": entity_topic_updates,
            "entity_topic_node_count": len(entity_topic_fact_ids),
            "fact_cluster_evidence_bundle_matches": fact_cluster_evidence_bundle_matches,
            "fact_cluster_evidence_bundle_node_count": len(
                fact_cluster_evidence_bundle_fact_ids
            ),
            "fact_evidence_bundle_matches": fact_cluster_evidence_bundle_matches,
            "fact_evidence_bundle_node_count": len(
                fact_cluster_evidence_bundle_fact_ids
            ),
            "fact_clusters_considered": len(clusters),
            "fact_clusters_consolidated": fact_clusters_consolidated,
            "fact_cluster_node_count": len(fact_cluster_fact_ids),
            "changed_evidence_bundle_ids": list(dict.fromkeys(changed_ids)),
            "touched_entity_ids": touched_entity_ids,
        }

    def _link_fact_relations(
        self,
        fact_ids: List[int],
        relations: List[Dict[str, Any]],
    ) -> None:
        for relation in relations:
            try:
                source_id = fact_ids[int(relation["source_index"])]
                target_id = fact_ids[int(relation["target_index"])]
                relation_type = str(relation["relation"])
                confidence = float(relation.get("confidence", 1.0) or 1.0)
                self._db.memory_add_fact_relation(
                    source_fact_id=source_id,
                    target_fact_id=target_id,
                    relation_type=relation_type,
                    confidence=confidence,
                )
            except Exception as exc:
                logger.debug("Failed to link retain relation %s: %s", relation, exc)

    def _link_temporal_relations(self, fact_id: int) -> int:
        """Link the new fact node to all prior nodes from the same calendar day."""
        prior_ids = self._db.memory_prior_fact_ids(fact_id, same_day=True)
        linked = 0
        for prior_id in prior_ids:
            try:
                self._db.memory_add_fact_relation(
                    source_fact_id=fact_id,
                    target_fact_id=prior_id,
                    relation_type=TEMPORAL_RELATION_TYPE,
                    confidence=1.0,
                )
                linked += 1
            except Exception as exc:
                logger.debug(
                    "Failed to link temporal relation %d -> %d: %s",
                    fact_id, prior_id, exc,
                )
        return linked

    def _link_semantic_relations(self, fact_id: int, embedding: np.ndarray) -> int:
        """Link the new node to all prior nodes above semantic similarity threshold."""
        prior_ids = set(self._db.memory_prior_fact_ids(fact_id))
        if not prior_ids:
            return 0
        neighbors = self._db.memory_semantic_neighbors(
            embedding,
            exclude_fact_id=fact_id,
            allowed_ids=prior_ids,
            threshold=SEMANTIC_RELATION_THRESHOLD,
        )
        linked = 0
        for prior_id, similarity in neighbors.items():
            try:
                self._db.memory_add_fact_relation(
                    source_fact_id=fact_id,
                    target_fact_id=prior_id,
                    relation_type=SEMANTIC_RELATION_TYPE,
                    confidence=float(similarity),
                )
                linked += 1
            except Exception as exc:
                logger.debug(
                    "Failed to link semantic relation %d -> %d: %s",
                    fact_id, prior_id, exc,
                )
        return linked

    def _link_causal_relations(
        self,
        fact_id: int,
        summary: str,
        embedding: np.ndarray,
        keywords: Optional[List[str]] = None,
    ) -> int:
        """Future extension point for LLM-classified causal graph edges.

        HindSight-style causal relation classification is intentionally left
        disabled for now.  Keep this method as the single place to plug it
        back in when causal relation quality and cost controls are ready.
        """
        return 0

    def _build_relation_graph(
        self,
        fact_id: int,
        summary: str,
        embedding: np.ndarray,
        keywords: Optional[List[str]] = None,
    ) -> None:
        temporal_count = self._link_temporal_relations(fact_id)
        semantic_count = self._link_semantic_relations(fact_id, embedding)
        causal_count = self._link_causal_relations(
            fact_id=fact_id,
            summary=summary,
            embedding=embedding,
            keywords=keywords,
        )
        logger.debug(
            "Graph linked node %d temporal=%d semantic=%d causal=%d",
            fact_id, temporal_count, semantic_count, causal_count,
        )

    def _store_worker_loop(self) -> None:
        while not self._store_shutdown_event.is_set() or not self._store_queue.empty():
            try:
                task = self._store_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            task_kind = str(task.get("kind") or "store")
            try:
                self._llm_thread_context.config = task["llm_config"]
                if task_kind == "reflect":
                    candidates = self._db.get_unprocessed_facts_for_evidence_bundle(
                        limit=1,
                    )
                    pending_feedback = self._db.memory_pending_interpretation_feedback(
                        limit=1,
                    )
                    if not candidates and not pending_feedback:
                        logger.debug(
                            "Memory reflect due but skipped: no unobserved facts or pending feedback",
                        )
                        continue
                    report = self.reflect(
                        limit=int(task.get("limit") or 100),
                        reflect_timestamp=task.get("reflect_timestamp"),
                    )
                    if not report.get("error"):
                        completed_at = time.time()
                        with self._store_worker_lock:
                            self._last_successful_reflect_at = completed_at
                        try:
                            self._db.set_meta(
                                MEMORY_REFLECT_META_KEY,
                                str(completed_at),
                            )
                        except Exception as exc:
                            logger.debug(
                                "Could not persist memory reflect timestamp: %s",
                                exc,
                            )
                    logger.debug("MemoryNodeManager reflect report: %s", report)
                elif task_kind == "decay":
                    report = self.decay(
                        decay_timestamp=task.get("decay_timestamp"),
                    )
                    if not report.get("error"):
                        completed_at = time.time()
                        with self._store_worker_lock:
                            self._last_successful_decay_at = completed_at
                        try:
                            self._db.set_meta(
                                MEMORY_DECAY_META_KEY,
                                str(completed_at),
                            )
                        except Exception as exc:
                            logger.debug(
                                "Could not persist memory decay timestamp: %s",
                                exc,
                            )
                    logger.debug("MemoryNodeManager decay report: %s", report)
                elif task_kind == "feedback":
                    count = self.analyze_feedback_for_pending_interpretations(
                        task.get("user_message", ""),
                        recall_event_id=task.get("recall_event_id"),
                    )
                    logger.debug(
                        "MemoryNodeManager feedback analysis stored %d item(s)",
                        count,
                    )
                else:
                    self.store_turn(
                        user_message=task["user_message"],
                        assistant_response=task["assistant_response"],
                        tags=task["tags"],
                        turn_timestamp=task["turn_timestamp"],
                    )
            except Exception as exc:
                logger.info(
                    "Async memory %s failed (non-fatal): %s",
                    task_kind,
                    exc,
                )
            finally:
                self._llm_thread_context.config = None
                if task_kind == "reflect":
                    with self._store_worker_lock:
                        self._reflect_queued_or_running = False
                if task_kind == "decay":
                    with self._store_worker_lock:
                        self._decay_queued_or_running = False
                self._store_queue.task_done()

    def _llm_config_snapshot(
        self,
        *,
        llm_client: Any = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "llm_client": (
                llm_client if llm_client is not None else self._llm_client
            ),
            "llm_model": str(llm_model or self._llm_model),
            "llm_base_url": str(llm_base_url or self._llm_base_url),
            "llm_api_key": (
                str(llm_api_key)
                if llm_api_key is not None
                else self._llm_api_key
            ),
        }

    @staticmethod
    def _normalize_interpretation_feedback_type(value: Any) -> str:
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        allowed = {"accept", "reject", "modify", "defer", "outdated", "unrelated", "none"}
        return text if text in allowed else "none"

    @classmethod
    def _feedback_interpretation_payload(
        cls,
        interpretations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for item in interpretations or []:
            try:
                interpretation_id = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            metadata = cls._json_dict(item.get("metadata", {}))
            payload.append({
                "id": interpretation_id,
                "claim": item.get("claim"),
                "interpretation_type": item.get("interpretation_type"),
                "status": item.get("status"),
                "confidence": item.get("confidence"),
                "scope": item.get("scope"),
                "target_text": item.get("target_text"),
                "action_implication": item.get("action_implication"),
                "resolution": item.get("resolution"),
                "entity_name": item.get("entity_name") or metadata.get("entity_name"),
                "recall_rank": item.get("recall_rank"),
                "recall_score": item.get("recall_score"),
            })
        return payload

    def analyze_feedback_for_pending_interpretations(
        self,
        user_message: str,
        *,
        recall_event_id: Optional[int] = None,
    ) -> int:
        """Analyze whether a new user message gives feedback on recalled interpretations."""
        if not self._enabled or not self._db or not str(user_message or "").strip():
            return 0
        try:
            if recall_event_id is not None:
                event = self._db.memory_recall_event_by_id(
                    int(recall_event_id),
                    limit_interpretations=8,
                )
            else:
                event = self._db.memory_latest_pending_recall_event(
                    limit_interpretations=8,
                )
        except Exception as exc:
            logger.debug("Memory interpretation feedback lookup failed: %s", exc)
            return 0
        if not event or not event.get("interpretations"):
            return 0

        interpretations = event.get("interpretations") or []
        interpretation_payload = self._feedback_interpretation_payload(interpretations)
        if not interpretation_payload:
            try:
                self._db.memory_mark_recall_event_status(
                    int(event["id"]),
                    "feedback_analyzed",
                )
            except Exception:
                pass
            return 0
        self._log_info(
            "memory_feedback",
            "raw_data_info",
            {
                "interpretations": json.dumps(
                    interpretation_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
                "current_user_message": self._reflect_log_text(user_message, limit=1600),
            }
        )
        prompt = INTERPRETATION_FEEDBACK_ANALYSIS_PROMPT.format(
            previous_user_query=self._reflect_log_text(event.get("query"), limit=1200),
            previous_assistant_response=self._reflect_log_text(
                event.get("assistant_response"),
                limit=2400,
            ),
            interpretations=json.dumps(
                interpretation_payload,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
            current_user_message=self._reflect_log_text(user_message, limit=1600),
        )
        data = self._parse_json_object_from_llm_text(self._call_llm(prompt) or "")
        if not data:
            self._log_info(
                "memory_feedback",
                "analysis_failed",
                {
                    "recall_event_id": event.get("id"),
                    "reason": "llm_json_empty",
                },
            )
            return 0

        by_id = {
            int(item["id"]): item
            for item in interpretation_payload
            if item.get("id") is not None
        }
        feedback_items = data.get("feedback_items")
        if not isinstance(feedback_items, list):
            feedback_items = []
        written = 0
        for raw_item in feedback_items:
            if not isinstance(raw_item, dict):
                continue
            try:
                interpretation_id = int(raw_item.get("interpretation_id"))
            except (TypeError, ValueError):
                continue
            if interpretation_id not in by_id:
                continue
            feedback_type = self._normalize_interpretation_feedback_type(
                raw_item.get("feedback_type")
            )
            if feedback_type in {"none", "unrelated"}:
                self._log_info(
                    "memory_feedback",
                    "analysis_skipped",
                    {
                        "interpretation_id": interpretation_id,
                        "feedback_type": feedback_type,
                        "confidence": raw_item.get("confidence"),
                    },
                )
                continue
            confidence = self._clip_unit_float(raw_item.get("confidence"), 0.0)
            if confidence < 0.4:
                continue
            try:
                self._log_info(
                    "memory_feedback",
                    "analysis_finished",
                    {
                        "interpretation_id": interpretation_id,
                        "feedback_type": feedback_type,
                        **by_id[interpretation_id]
                    }
                )
                self._db.memory_add_interpretation_feedback(
                    recall_event_id=int(event["id"]),
                    interpretation_id=interpretation_id,
                    feedback_type=feedback_type,
                    confidence=confidence,
                    user_message=str(user_message or ""),
                    evidence_text=str(raw_item.get("evidence_text") or "").strip(),
                    correction=str(raw_item.get("correction") or "").strip(),
                    metadata={
                        "source": "feedback_analysis",
                        "previous_query": event.get("query"),
                        "analysis_has_feedback": bool(data.get("has_feedback")),
                    },
                )
                written += 1
            except Exception as exc:
                logger.debug("Could not store interpretation feedback: %s", exc)

        try:
            self._db.memory_mark_recall_event_status(
                int(event["id"]),
                "feedback_analyzed",
            )
        except Exception:
            pass
        self._log_info(
            "memory_feedback",
            "analysis_finished",
            {
                "recall_event_id": event.get("id"),
                "has_feedback": bool(data.get("has_feedback")),
                "feedback_items": written,
            },
        )
        return written

    def analyze_feedback_for_pending_interpretations_async(
        self,
        user_message: str,
        *,
        llm_client: Any = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
    ) -> bool:
        """Queue interpretation feedback analysis without blocking the current turn."""
        if not self._enabled or not str(user_message or "").strip():
            return False
        if not self._db:
            return False
        try:
            event = self._db.memory_latest_pending_recall_event(
                limit_interpretations=1,
            )
        except Exception as exc:
            logger.debug("Memory interpretation feedback async lookup failed: %s", exc)
            return False
        if not event:
            return False
        task = {
            "kind": "feedback",
            "user_message": str(user_message),
            "recall_event_id": int(event["id"]),
            "llm_config": self._llm_config_snapshot(
                llm_client=llm_client,
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
            ),
        }
        with self._store_worker_lock:
            if self._store_shutdown_event.is_set():
                return False
            try:
                self._store_queue.put_nowait(task)
            except queue.Full:
                logger.warning(
                    "Memory store queue is full; dropping feedback analysis (maxsize=%d)",
                    self._store_queue_maxsize,
                )
                return False
            self._ensure_store_worker_locked()
        return True

    def _ensure_store_worker_locked(self) -> None:
        if self._store_worker_thread and self._store_worker_thread.is_alive():
            return
        self._store_worker_thread = threading.Thread(
            target=self._store_worker_loop,
            daemon=True,
            name="memory-node-store",
        )
        self._store_worker_thread.start()

    def store_turn_async(
        self,
        user_message: str,
        assistant_response: str,
        tags: Optional[List[str]] = None,
        turn_timestamp: Optional[Any] = None,
        *,
        llm_client: Any = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
    ) -> bool:
        """Queue a completed turn for ordered background retention.

        The return value only indicates whether the queue accepted the turn.
        """
        if not self._enabled or not user_message or not assistant_response:
            return False

        if self._db:
            try:
                self._db.memory_attach_latest_recall_event_response(
                    query=str(user_message),
                    assistant_response=str(assistant_response),
                )
            except Exception as exc:
                logger.debug("Could not attach response to memory recall event: %s", exc)

        task = {
            "kind": "store",
            "user_message": str(user_message),
            "assistant_response": str(assistant_response),
            "tags": list(tags or []),
            "turn_timestamp": turn_timestamp,
            "llm_config": self._llm_config_snapshot(
                llm_client=llm_client,
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
            ),
        }
        with self._store_worker_lock:
            if self._store_shutdown_event.is_set():
                return False
            try:
                self._store_queue.put_nowait(task)
            except queue.Full:
                logger.warning(
                    "Memory store queue is full; dropping turn (maxsize=%d)",
                    self._store_queue_maxsize,
                )
                return False
            self._ensure_store_worker_locked()
        return True

    def reflect_if_due_async(
        self,
        *,
        limit: int = 100,
        reflect_timestamp: Optional[Any] = None,
        llm_client: Any = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
    ) -> bool:
        """Queue reflection after earlier store tasks when its time window is due."""
        if not self._enabled or not self._db:
            return False

        now = time.time()
        with self._store_worker_lock:
            if self._store_shutdown_event.is_set():
                return False
            if self._reflect_queued_or_running:
                return False
            try:
                persisted_last_reflect = float(
                    self._db.get_meta(MEMORY_REFLECT_META_KEY) or 0.0
                )
                self._last_successful_reflect_at = max(
                    self._last_successful_reflect_at,
                    persisted_last_reflect,
                )
            except Exception:
                pass
            if (
                now - self._last_successful_reflect_at
                < self._reflect_interval_seconds
            ):
                return False

            task = {
                "kind": "reflect",
                "limit": max(1, int(limit or 100)),
                "reflect_timestamp": reflect_timestamp,
                "llm_config": self._llm_config_snapshot(
                    llm_client=llm_client,
                    llm_model=llm_model,
                    llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key,
                ),
            }
            try:
                self._store_queue.put_nowait(task)
            except queue.Full:
                logger.warning(
                    "Memory store queue is full; reflection was not queued "
                    "(maxsize=%d)",
                    self._store_queue_maxsize,
                )
                return False
            self._reflect_queued_or_running = True
            self._ensure_store_worker_locked()
        return True

    def decay_if_due_async(
        self,
        *,
        decay_timestamp: Optional[Any] = None,
    ) -> bool:
        """Queue time-decay maintenance after earlier memory tasks when due."""
        if not self._enabled or not self._db:
            return False
        now = time.time()
        with self._store_worker_lock:
            if self._store_shutdown_event.is_set():
                return False
            if self._decay_queued_or_running:
                return False
            try:
                persisted_last_decay = float(
                    self._db.get_meta(MEMORY_DECAY_META_KEY) or 0.0
                )
                self._last_successful_decay_at = max(
                    self._last_successful_decay_at,
                    persisted_last_decay,
                )
            except Exception:
                pass
            if (
                now - self._last_successful_decay_at
                < self._decay_interval_seconds
            ):
                return False

            task = {
                "kind": "decay",
                "decay_timestamp": decay_timestamp,
                "llm_config": self._llm_config_snapshot(),
            }
            try:
                self._store_queue.put_nowait(task)
            except queue.Full:
                logger.warning(
                    "Memory store queue is full; decay was not queued "
                    "(maxsize=%d)",
                    self._store_queue_maxsize,
                )
                return False
            self._decay_queued_or_running = True
            self._ensure_store_worker_locked()
        return True

    def flush_store_queue(self, timeout: Optional[float] = None) -> bool:
        """Wait until all accepted background memory tasks finish."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while self._store_queue.unfinished_tasks:
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def shutdown_store_worker(
        self,
        *,
        wait: bool = True,
        timeout: Optional[float] = 5.0,
    ) -> bool:
        """Stop accepting store tasks and optionally drain the worker."""
        with self._store_worker_lock:
            self._store_shutdown_event.set()
            worker = self._store_worker_thread
        if not wait or worker is None:
            return not self._store_queue.unfinished_tasks
        worker.join(timeout=None if timeout is None else max(0.0, timeout))
        return not worker.is_alive() and not self._store_queue.unfinished_tasks

    # ── Store turn as memory fact node ─────────────────────────────────────────

    @staticmethod
    def _cal_store_turns_character_count(source_turns: List[Dict[str, Any]]) -> int:
        return sum(
            len(str(turn.get("user_message") or ""))
            + len(str(turn.get("assistant_response") or ""))
            for turn in source_turns
        )

    def store_turn(
        self,
        user_message: str,
        assistant_response: str,
        tags: Optional[List[str]] = None,
        turn_timestamp: Optional[Any] = None,
    ) -> bool:
        """Retain a turn as one or more narrative memory nodes.

        The work follows the HindSight retain shape:
        extract narrative facts → embed each fact → store nodes → link
        entities and explicit intra-retain causal relations → build the
        cross-turn relation graph. Callers that need non-blocking behavior
        should use store_turn_async().

        Returns True if at least one fact was stored, False otherwise.
        """
        if not self._enabled:
            return False
        if not user_message or not assistant_response:
            return False

        self._turn_count += 1
        self._pending_store_turns.append({
            "user_message": user_message,
            "assistant_response": assistant_response,
            "turn_timestamp": turn_timestamp,
            "tags": list(tags or []),
        })
        pending_character_count = self._cal_store_turns_character_count(
            self._pending_store_turns,
        )
        turn_threshold_reached = (
            len(self._pending_store_turns) >= self._min_turns_before_store
        )
        character_threshold_exceeded = (
            pending_character_count >= self._max_chars_before_store
        )
        if not turn_threshold_reached and not character_threshold_exceeded:
            return False

        if not self._ensure_embedding_client():
            return False

        source_turns = list(self._pending_store_turns)
        batch_user_message = "\n\n".join(
            str(turn.get("user_message") or "")
            for turn in source_turns
        )
        batch_assistant_response = "\n\n".join(
            str(turn.get("assistant_response") or "")
            for turn in source_turns
        )
        batch_tags = list(dict.fromkeys(
            tag
            for turn in source_turns
            for tag in (turn.get("tags") or [])
            if tag
        ))
        batch_timestamp = source_turns[-1].get("turn_timestamp")

        try:
            # ── Step 1: Extract narrative facts (SYNC) ──
            retain_data = self._extract_retain_facts(
                source_turns,
                turn_timestamp=batch_timestamp,
            )
            if not retain_data:
                logger.debug("Skipping memory fact node — retain extraction returned no data")
                return False
            self._pending_store_turns.clear()
            facts = retain_data.get("facts", [])
            stored_nodes: List[Tuple[int, str, np.ndarray, List[str]]] = []
            fact_ids: List[int] = []

            for idx, fact in enumerate(facts):
                summary = str(fact.get("text", "")).strip()
                if not summary:
                    continue
                keywords = self._normalize_keywords(fact.get("keywords", []))
                primary_topic = self._topic_key(
                    fact.get("primary_topic")
                    or next(iter(self._normalize_keywords(fact.get("topic", []))), "")
                    or (keywords[0] if keywords else "general")
                )
                topics = [primary_topic]
                primary_entity = fact.get("primary_entity")
                primary_entity_id: Optional[int] = None
                if isinstance(primary_entity, dict):
                    primary_entity_name = str(primary_entity.get("name") or "").strip()
                    primary_entity_type = (
                        str(primary_entity.get("type") or "CONCEPT").strip().upper()
                        or "CONCEPT"
                    )
                    if primary_entity_name:
                        primary_entity_id = self._db.entity_add_entity(
                            name=primary_entity_name,
                            entity_type=primary_entity_type,
                        )

                if idx == 0:
                    self._log_info(
                        "memory_store",
                        "extract_facts",
                        {
                            "user_message": batch_user_message,
                            "assistant_response": batch_assistant_response,
                            "source_turn_count": len(source_turns),
                        },
                    )
                self._log_info(
                    "memory_store",
                    "extract_facts",
                    {
                        "summary": summary,
                        "keywords": keywords,
                        "topics": topics,
                        "primary_entity": primary_entity,
                        "primary_entity_id": primary_entity_id,
                        "primary_topic": primary_topic,
                        "entity_names": [
                            str(entity.get("name", "")).strip()
                            for entity in fact.get("entities", [])
                            if isinstance(entity, dict) and str(entity.get("name", "")).strip()
                            ],
                        "fact_type": fact.get("fact_type", "semantic"),
                        "fact_subject": fact.get("fact_subject", "other"),
                        "fact_kind": fact.get("fact_kind", "other"),
                        "task_event_like": fact.get("task_event_like"),
                        "task_event_subject": fact.get("task_event_subject", ""),
                        "task_relevance": fact.get("task_relevance", ""),
                    }
                )
                # ── Step 2: Generate embedding (SYNC) ──
                embedding = self._embedding_client.embed_text(summary)
                if embedding is None:
                    logger.info("Skipping memory fact — embedding generation failed")
                    continue
                
                # ── Step 3: Store the new fact node (SYNC) ──
                fact_id = self._db.memory_add_fact(
                    time_key=self._memory_time_key(
                        idx,
                        turn_timestamp=batch_timestamp,
                    ),
                    summary=summary,
                    keywords=keywords,
                    topic=topics,
                    original_dialog=self._build_original_dialog_payload(
                        fact=fact,
                        source_turns=source_turns,
                    ),
                    query_embedding=embedding,
                    tags=self._fact_tags(fact, batch_tags),
                    fact_type=fact.get("fact_type", "semantic"),
                    fact_subject=fact.get("fact_subject", "other"),
                    fact_kind=fact.get("fact_kind", "other"),
                    task_event_like=fact.get("task_event_like"),
                    task_event_subject=fact.get("task_event_subject", ""),
                    task_relevance=fact.get("task_relevance", ""),
                    primary_entity_id=primary_entity_id,
                    primary_topic=primary_topic,
                    entity_names=[
                        str(entity.get("name", "")).strip()
                        for entity in fact.get("entities", [])
                        if isinstance(entity, dict) and str(entity.get("name", "")).strip()
                    ],
                )

                fact_entities = fact.get("entities", [])
                linked_entities = self._link_fact_entities(fact_id, fact_entities)
                if primary_entity_id is not None and all(
                    entity_id != primary_entity_id
                    for entity_id, _entity_name in linked_entities
                ):
                    self._db.entity_link_fact(fact_id, primary_entity_id)
                
                stored_nodes.append((fact_id, summary, embedding, keywords))
                fact_ids.append(fact_id)

            if not stored_nodes:
                return False

            # ── Step 4: Link explicit relations between newly retained facts ──
            self._link_fact_relations(fact_ids, retain_data.get("causal_relations", []))

            # ── Step 5: Build cross-turn relation graph ──
            for fact_id, summary, embedding, keywords in stored_nodes:
                try:
                    self._build_relation_graph(
                        fact_id=fact_id,
                        summary=summary,
                        embedding=embedding,
                        keywords=keywords,
                    )
                except Exception as exc:
                    logger.debug(
                        "Relation graph construction failed for fact node %d: %s",
                        fact_id,
                        exc,
                    )

            logger.debug(
                "Retained %d memory fact node(s) from %d turn(s)",
                len(stored_nodes),
                len(source_turns),
            )
            return True

        except Exception as e:
            logger.info("Failed to store memory node (non-fatal): %s", e)
            return False

    def _augment_evidence_bundle_merge_group_with_pending_sources(
        self,
        group: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Add same entity/topic facts to a merge group so overlap updates use one LLM call."""
        if not self._db:
            return group
        try:
            entity_id = int(group.get("entity_id"))
            topic_key = str(group.get("topic_key") or "")
        except (TypeError, ValueError):
            return group
        if not topic_key:
            return group

        group_source_facts = [
            dict(fact)
            for fact in group.get("source_facts", [])
            if fact.get("id") is not None
        ]
        group_source_ids = {int(fact["id"]) for fact in group_source_facts}
        topic_source_facts = self._db.get_fact_nodes_using_entity_topic(
            entity_id=entity_id,
            topic_key=topic_key,
            limit=12,
        )
        pending_source_facts = [
            dict(fact)
            for fact in topic_source_facts
            if fact.get("id") is not None and int(fact["id"]) not in group_source_ids
        ]

        combined_source_facts: List[Dict[str, Any]] = []
        seen_source_ids: set[int] = set()
        for fact in pending_source_facts + group_source_facts + topic_source_facts:
            if fact.get("id") is None:
                continue
            fact_id = int(fact["id"])
            if fact_id in seen_source_ids:
                continue
            seen_source_ids.add(fact_id)
            combined_source_facts.append(dict(fact))

        augmented = dict(group)
        augmented["source_facts"] = combined_source_facts
        augmented["pending_source_facts"] = pending_source_facts
        augmented["pending_source_fact_ids"] = [
            int(fact["id"]) for fact in pending_source_facts
        ]
        return augmented

    def _merge_duplicated_evidence_bundle_group(
        self,
        group: Dict[str, Any],
        *,
        changed_evidence_bundle_ids: Optional[List[int]] = None,
    ) -> bool:
        evidence_bundles = group.get("evidence_bundles") or []
        source_facts = group.get("source_facts") or []
        if len(evidence_bundles) < 2:
            return False
        source_ids = [int(fact["id"]) for fact in source_facts]
        if not source_ids:
            return False
        keep_bundle = evidence_bundles[0]
        bundle_type = str(keep_bundle.get("bundle_type") or "entity_topic")
        metadata = self._json_dict(keep_bundle.get("metadata", {}))
        remove_ids = [
            int(evidence_bundle["id"])
            for evidence_bundle in evidence_bundles[1:]
        ]
        self._log_info(
            "memory_reflect",
            "evidence_bundle_merge",
            {
                "entity_id": group.get("entity_id"),
                "entity_name": group.get("entity_name", ""),
                "topic_key": group.get("topic_key"),
                "topic_label": group.get("topic_label", group.get("topic_key", "")),
                "bundle_type": bundle_type,
                "keep_evidence_bundle_id": int(keep_bundle["id"]),
                "remove_evidence_bundle_ids": remove_ids,
                "input_evidence_bundles": [
                    self._reflect_evidence_bundle_log_item(evidence_bundle)
                    for evidence_bundle in evidence_bundles
                ],
                "supporting_facts": self._reflect_fact_log_items(source_facts),
                "merged_evidence_bundle": {
                    "bundle_type": bundle_type,
                    "metadata": metadata,
                },
            })
        self._db.memory_replace_evidence_bundle_group(
            keep_evidence_bundle_id=int(keep_bundle["id"]),
            remove_evidence_bundle_ids=remove_ids,
            bundle_type=bundle_type,
            source_fact_ids=source_ids,
            metadata=metadata,
            source_roles={
                int(fact_id): "matched"
                for fact_id in group.get("pending_source_fact_ids", [])
            },
        )
        if changed_evidence_bundle_ids is not None:
            changed_evidence_bundle_ids.append(int(keep_bundle["id"]))
        return True

    def _reflect_merging_duplicated_entities(
        self,
        *,
        limit: int,
        date_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Merge duplicate entities and repair observations affected by the merge."""
        if not self._db:
            return {
                "candidates": [],
                "merged": 0,
                "candidate_count": 0,
                "merge_candidates": 0,
                "evidence_bundle_groups_merged": 0,
                "changed_evidence_bundle_ids": [],
            }

        unprocessed_fact_candidates = self._db.get_unprocessed_facts_for_evidence_bundle(
            date_key=date_key,
            limit=limit,
        )
        anchor_entity_ids = list(dict.fromkeys(
            int(entity_id)
            for item in unprocessed_fact_candidates
            for entity_id, _entity_name in item.get("linked_entities", [])
        ))

        entity_report = self._db.merge_similar_entities(
            limit=limit,
            anchor_entity_ids=anchor_entity_ids,
        )
        self._log_info(
            "memory_reflect",
            "entity_merge_candidates",
            {
                "anchor_entity_ids": anchor_entity_ids,
                "candidate_count": entity_report.get("candidate_count", 0),
                "merge_candidates": entity_report.get("merge_candidates", 0),
                "merged": entity_report.get("merged", 0),
                "candidates": entity_report.get("candidates", []),
            },
        )

        merged_entity_ids: List[int] = []
        for candidate in entity_report.get("candidates", []):
            if candidate.get("action") != "merge":
                continue
            canonical_id = int(candidate["canonical_id"])
            merged_entity_ids.append(canonical_id)
            self._log_info(
                "memory_reflect",
                "entity_merge",
                {
                    "canonical_id": canonical_id,
                    "canonical_name": candidate.get("canonical_name"),
                    "duplicate_id": candidate.get("duplicate_id"),
                    "duplicate_name": candidate.get("duplicate_name"),
                    "confidence": candidate.get("confidence"),
                    "reason": candidate.get("reason"),
                    "risk": candidate.get("risk"),
                    "name_score": candidate.get("name_score"),
                    "type_score": candidate.get("type_score"),
                    "co_entities_score": candidate.get("co_entities_score"),
                },
            )

        merged_entity_ids = list(dict.fromkeys(merged_entity_ids))
        groups = (
            self._db.find_duplicated_evidence_bundle_groups(
                entity_ids=merged_entity_ids,
            )
            if merged_entity_ids
            else []
        )
        augmented_groups = [
            self._augment_evidence_bundle_merge_group_with_pending_sources(group)
            for group in groups
        ]
        self._log_info(
            "memory_reflect",
            "evidence_bundle_merge_candidates",
            {
                "entity_ids": merged_entity_ids,
                "group_count": len(augmented_groups),
                "groups": [
                    {
                        "entity_id": group.get("entity_id"),
                        "entity_name": group.get("entity_name"),
                        "topic_key": group.get("topic_key"),
                        "topic_label": group.get("topic_label"),
                        "bundle_type": group.get("bundle_type"),
                        "evidence_bundle_ids": [
                            evidence_bundle.get("id")
                            for evidence_bundle in group.get("evidence_bundles", [])
                        ],
                        "source_fact_ids": [
                            fact.get("id")
                            for fact in group.get("source_facts", [])
                        ],
                        "pending_source_fact_ids": group.get(
                            "pending_source_fact_ids",
                            [],
                        ),
                    }
                    for group in augmented_groups
                ],
            },
        )

        evidence_bundle_groups_merged = 0
        changed_evidence_bundle_ids: List[int] = []
        for group in augmented_groups:
            try:
                if not self._merge_duplicated_evidence_bundle_group(
                    group,
                    changed_evidence_bundle_ids=changed_evidence_bundle_ids,
                ):
                    continue
                evidence_bundle_groups_merged += 1
            except Exception as exc:
                logger.debug(
                    "Failed to merge observations for entity %s topic %s: %s",
                    group.get("entity_id"),
                    group.get("topic_key"),
                    exc,
                )

        return {
            **entity_report,
            "merged_entity_ids": merged_entity_ids,
            "evidence_bundle_groups_merged": evidence_bundle_groups_merged,
            "changed_evidence_bundle_ids": list(dict.fromkeys(changed_evidence_bundle_ids)),
        }

    @staticmethod
    def _normalize_feedback_update_status(value: Any, fallback: str = "current") -> str:
        text = str(value or fallback).strip().lower().replace("-", "_").replace(" ", "_")
        allowed = {"current", "conflicted", "archived", "superseded"}
        return text if text in allowed else fallback

    def _update_interpretation_using_feedback(
        self,
        *,
        interpretation: Dict[str, Any],
        feedback: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        interpretation_payload = {
            key: interpretation.get(key)
            for key in (
                "id",
                "claim",
                "target_text",
                "scope",
                "interpretation_type",
                "polarity",
                "strength",
                "confidence",
                "status",
                "conflict_status",
                "resolution",
                "action_implication",
                "metadata",
            )
        }
        feedback_payload = {
            "id": feedback.get("id"),
            "feedback_type": feedback.get("feedback_type"),
            "confidence": feedback.get("confidence"),
            "user_message": feedback.get("user_message"),
            "evidence_text": feedback.get("evidence_text"),
            "correction": feedback.get("correction"),
            "created_at": feedback.get("created_at"),
        }
        prompt = INTERPRETATION_UPDATE_USING_FEEDBACK_PROMPT.format(
            interpretation=json.dumps(
                interpretation_payload,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
            feedback=json.dumps(
                feedback_payload,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
        )
        data = self._parse_json_object_from_llm_text(self._call_llm(prompt) or "")
        if not data or not bool(data.get("should_update")):
            return None
        allowed_types = {
            "insight", "task",
            "explicit_preference", "explicit_instruction", "inferred_preference",
            "behavior_pattern", "project_state", "task_risk", "constraint",
            "conflict_resolution", "strategy", "other",
        }
        existing_type = str(interpretation.get("interpretation_type") or "insight")
        interpretation_type = str(
            data.get("interpretation_type") or existing_type
        ).strip().lower().replace("-", "_").replace(" ", "_")
        if interpretation_type not in allowed_types:
            interpretation_type = existing_type if existing_type in allowed_types else "insight"
        conflict_status = str(
            data.get("conflict_status") or interpretation.get("conflict_status") or "none"
        ).strip().lower()
        if conflict_status not in {"none", "resolved", "unresolved"}:
            conflict_status = "none"
        polarity = str(data.get("polarity") or interpretation.get("polarity") or "neutral").strip().lower()
        if polarity not in {"positive", "negative", "mixed", "neutral"}:
            polarity = "neutral"
        return {
            "claim": str(data.get("claim") or interpretation.get("claim") or "").strip(),
            "target_text": str(data.get("target_text") or interpretation.get("target_text") or "").strip(),
            "scope": str(data.get("scope") or interpretation.get("scope") or "general").strip() or "general",
            "interpretation_type": interpretation_type,
            "polarity": polarity,
            "strength": self._clip_unit_float(data.get("strength"), float(interpretation.get("strength") or 0.5)),
            "confidence": self._clip_unit_float(data.get("confidence"), float(interpretation.get("confidence") or 0.5)),
            "status": self._normalize_feedback_update_status(
                data.get("status"),
                str(interpretation.get("status") or "current"),
            ),
            "conflict_status": conflict_status,
            "resolution": str(data.get("resolution") or interpretation.get("resolution") or "").strip(),
            "action_implication": str(
                data.get("action_implication")
                or interpretation.get("action_implication")
                or ""
            ).strip(),
            "metadata": self._json_dict(data.get("metadata", {})),
        }

    def _fallback_update_interpretation_using_feedback(
        self,
        *,
        interpretation: Dict[str, Any],
        feedback: Dict[str, Any],
    ) -> Dict[str, Any]:
        feedback_type = self._normalize_interpretation_feedback_type(
            feedback.get("feedback_type")
        )
        confidence = self._clip_unit_float(interpretation.get("confidence"), 0.5)
        strength = self._clip_unit_float(interpretation.get("strength"), 0.5)
        status = str(interpretation.get("status") or "current")
        conflict_status = str(interpretation.get("conflict_status") or "none")
        resolution = str(interpretation.get("resolution") or "").strip()
        if feedback_type == "accept":
            confidence = min(1.0, confidence + 0.06)
            strength = min(1.0, strength + 0.03)
        elif feedback_type == "reject":
            confidence = max(0.0, confidence - 0.25)
            status = "conflicted" if confidence >= 0.25 else "archived"
            conflict_status = "unresolved"
            resolution = str(feedback.get("evidence_text") or "User rejected this interpretation.").strip()
        elif feedback_type == "modify":
            confidence = max(0.0, confidence - 0.08)
            status = "conflicted"
            conflict_status = "unresolved"
            resolution = str(feedback.get("correction") or feedback.get("evidence_text") or "").strip()
        elif feedback_type == "outdated":
            status = "archived"
            resolution = str(feedback.get("evidence_text") or "User indicated this interpretation is outdated.").strip()
        elif feedback_type == "defer":
            resolution = str(feedback.get("evidence_text") or "User deferred this interpretation.").strip()
        return {
            "claim": interpretation.get("claim", ""),
            "target_text": interpretation.get("target_text", ""),
            "scope": interpretation.get("scope", "general"),
            "interpretation_type": interpretation.get("interpretation_type", "insight"),
            "polarity": interpretation.get("polarity", "neutral"),
            "strength": strength,
            "confidence": confidence,
            "status": status,
            "conflict_status": conflict_status,
            "resolution": resolution,
            "action_implication": interpretation.get("action_implication", ""),
            "metadata": {},
        }

    def _persist_interpretation_feedback_update(
        self,
        *,
        interpretation: Dict[str, Any],
        feedback: Dict[str, Any],
        update: Dict[str, Any],
    ) -> int:
        metadata = {
            **self._json_dict(interpretation.get("metadata", {})),
            **self._json_dict(update.get("metadata", {})),
        }
        feedback_type = self._normalize_interpretation_feedback_type(
            feedback.get("feedback_type")
        )
        feedback_meta = self._json_dict(metadata.get("feedback", {}))
        counts = self._json_dict(feedback_meta.get("counts", {}))
        counts[feedback_type] = int(counts.get(feedback_type, 0) or 0) + 1
        feedback_meta.update({
            "counts": counts,
            "last_feedback_id": feedback.get("id"),
            "last_feedback_type": feedback_type,
            "last_feedback_confidence": feedback.get("confidence"),
            "last_feedback_evidence": feedback.get("evidence_text"),
            "last_feedback_correction": feedback.get("correction"),
            "last_feedback_at": feedback.get("created_at"),
        })
        metadata["feedback"] = feedback_meta
        metadata["source"] = metadata.get("source") or "interpretation_feedback_update"

        claim = str(update.get("claim") or interpretation.get("claim") or "").strip()
        action_implication = str(
            update.get("action_implication")
            or interpretation.get("action_implication")
            or ""
        ).strip()
        target_text = str(update.get("target_text") or interpretation.get("target_text") or "").strip()
        scope = str(update.get("scope") or interpretation.get("scope") or "general").strip() or "general"
        interpretation_type = str(
            update.get("interpretation_type")
            or interpretation.get("interpretation_type")
            or "insight"
        )
        resolution = str(update.get("resolution") or interpretation.get("resolution") or "").strip()
        embedding_text = self._build_interpretation_embedding_text(
            entity_name=interpretation.get("entity_name") or metadata.get("entity_name") or "",
            target_text=target_text,
            scope=scope,
            interpretation_type=interpretation_type,
            claim=claim,
            action_implication=action_implication,
            resolution=resolution,
        )
        return self._db.memory_upsert_interpretation(
            interpretation_id=int(interpretation["id"]),
            claim=claim,
            entity_id=interpretation.get("entity_id") or metadata.get("entity_id"),
            subject_text=interpretation.get("subject_text", ""),
            target_text=target_text,
            scope=scope,
            interpretation_type=interpretation_type,
            polarity=update.get("polarity", interpretation.get("polarity", "neutral")),
            strength=update.get("strength", interpretation.get("strength", 0.5)),
            confidence=update.get("confidence", interpretation.get("confidence", 0.5)),
            status=update.get("status", interpretation.get("status", "current")),
            conflict_status=update.get(
                "conflict_status",
                interpretation.get("conflict_status", "none"),
            ),
            resolution=resolution,
            action_implication=action_implication,
            evidence_fact_ids=interpretation.get("evidence_fact_ids", []),
            evidence_observation_ids=interpretation.get("evidence_observation_ids", []),
            counter_evidence_fact_ids=interpretation.get("counter_evidence_fact_ids", []),
            counter_evidence_observation_ids=interpretation.get(
                "counter_evidence_observation_ids",
                [],
            ),
            embedding=self._embed_memory_layer_text(embedding_text),
            embedding_text=embedding_text,
            metadata=metadata,
        )

    def _reflect_apply_interpretation_feedback(
        self,
        *,
        limit: int = 50,
    ) -> Dict[str, Any]:
        if not self._db:
            return {"processed": 0, "applied": 0, "ignored": 0}
        try:
            feedback_items = self._db.memory_pending_interpretation_feedback(
                limit=limit,
            )
        except Exception as exc:
            return {"processed": 0, "applied": 0, "ignored": 0, "error": str(exc)}
        report = {"processed": 0, "applied": 0, "ignored": 0, "feedback_ids": []}
        for feedback in feedback_items:
            report["processed"] += 1
            feedback_id = int(feedback.get("id"))
            interpretation = feedback.get("interpretation")
            if not interpretation:
                self._db.memory_mark_interpretation_feedback_applied(
                    feedback_id,
                    status="ignored",
                )
                report["ignored"] += 1
                continue
            feedback_type = self._normalize_interpretation_feedback_type(
                feedback.get("feedback_type")
            )
            if feedback_type in {"none", "unrelated"}:
                self._db.memory_mark_interpretation_feedback_applied(
                    feedback_id,
                    status="ignored",
                )
                report["ignored"] += 1
                continue
            update: Optional[Dict[str, Any]] = None
            if feedback_type in {"reject", "modify", "outdated"}:
                update = self._update_interpretation_using_feedback(
                    interpretation=interpretation,
                    feedback=feedback,
                )
            if update is None:
                update = self._fallback_update_interpretation_using_feedback(
                    interpretation=interpretation,
                    feedback=feedback,
                )
            try:
                interpretation_id = self._persist_interpretation_feedback_update(
                    interpretation=interpretation,
                    feedback=feedback,
                    update=update,
                )
                self._db.memory_mark_interpretation_feedback_applied(
                    feedback_id,
                    status="applied",
                )
                report["applied"] += 1
                report["feedback_ids"].append(feedback_id)
                self._log_info(
                    "memory_reflect",
                    "interpretation_feedback_applied",
                    {
                        "feedback_id": feedback_id,
                        "interpretation_id": interpretation_id,
                        "feedback_type": feedback_type,
                    },
                )
            except Exception as exc:
                logger.debug("Could not apply interpretation feedback: %s", exc)
                report["ignored"] += 1
        return report

    @staticmethod
    def _parse_timestamp(
        reflect_timestamp: Optional[Any] = None,
    ) -> datetime:
        if reflect_timestamp is None:
            reflect_now = datetime.now().astimezone()
        elif isinstance(reflect_timestamp, datetime):
            reflect_now = (
                reflect_timestamp.astimezone()
                if reflect_timestamp.tzinfo is None
                else reflect_timestamp
            )
        else:
            try:
                parsed_reflect_timestamp = datetime.fromisoformat(
                    str(reflect_timestamp).replace("Z", "+00:00")
                )
                reflect_now = (
                    parsed_reflect_timestamp.astimezone()
                    if parsed_reflect_timestamp.tzinfo is None
                    else parsed_reflect_timestamp
                )
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid memory reflect timestamp %r; using current time",
                    reflect_timestamp,
                )
                reflect_now = datetime.now().astimezone()
        return reflect_now

    def decay(
        self,
        *,
        decay_timestamp: Optional[Any] = None,
        semantic_fact_half_life_days: Optional[float] = None,
        episodic_fact_half_life_days: Optional[float] = None,
        observation_half_life_days: Optional[float] = None,
        interpretation_half_life_days: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run memory time-decay maintenance independently from reflection."""
        if not self._db:
            return {
                "error": "memory database unavailable",
            }
        decay_timestamp = self._parse_timestamp(decay_timestamp)
        report = self._db.memory_reflect_node_decay(
            semantic_fact_half_life_days=(
                semantic_fact_half_life_days or MEMORY_SEMANTIC_FACT_HALF_LIFE_DAYS
            ),
            episodic_fact_half_life_days=(
                episodic_fact_half_life_days or MEMORY_EPISODIC_FACT_HALF_LIFE_DAYS
            ),
            observation_half_life_days=(
                observation_half_life_days or MEMORY_OBSERVATION_HALF_LIFE_DAYS
            ),
            interpretation_half_life_days=(
                interpretation_half_life_days or MEMORY_INTERPRETATION_HALF_LIFE_DAYS
            ),
            decay_timestamp=decay_timestamp,
        )
        self._log_info(
            "memory_decay",
            "finish",
            {
                "evaluated": report.get("evaluated", 0),
                "updated": report.get("updated", 0),
                "facts": len(report.get("facts", [])),
                "observations": len(report.get("observations", [])),
                "interpretations": len(report.get("interpretations", [])),
                "evaluated_at": report.get("evaluated_at"),
            },
        )
        return report

    def reflect(
        self,
        *,
        limit: int = 100,
        reflect_timestamp: Optional[Any] = None,
        task_active_to_paused_days: Optional[float] = None,
        task_stale_days: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run memory reflection maintenance.

        It selects unprocessed facts, merges newly introduced entities, updates
        or creates observations, and generates interpretations.
        ``run_agent.py`` schedules this method through the
        ordered background queue when the configured time interval is due.
        ``reflect_timestamp`` selects the local calendar day to process and is
        defaults to the current time.
        """
        if not self._db:
            return {
                "candidates": [],
                "merged": 0,
                "candidate_count": 0,
                "evidence_bundles_consolidated": 0,
                "error": "memory database unavailable",
            }
        reflect_now = self._parse_timestamp(reflect_timestamp)
        reflect_date_key = reflect_now.date().isoformat()
        self._log_info(
            "memory_reflect",
            "start", 
            {
                "limit": limit,
                "reflect_timestamp": reflect_now.isoformat(),
                "reflect_date_key": reflect_date_key,
                "task_active_to_paused_days": task_active_to_paused_days,
                "task_stale_days": task_stale_days,
            })

        feedback_report = self._reflect_apply_interpretation_feedback(
            limit=limit,
        )

        entity_merging_report = self._reflect_merging_duplicated_entities(
            limit=limit,
            date_key=reflect_date_key,
        )
        changed_evidence_bundle_ids = list(
            entity_merging_report.get("changed_evidence_bundle_ids", [])
        )
        # build fact-evidence_bundle matching
        evidence_bundle_report = self._reflect_generate_evidence_bundles_using_facts(
            limit=limit,
            date_key=reflect_date_key,
            changed_evidence_bundle_ids=changed_evidence_bundle_ids,
        )
        
        report = dict(entity_merging_report)
        report["interpretation_feedback"] = feedback_report
        report["evidence_bundle_reflect"] = evidence_bundle_report
        report["evidence_bundles_consolidated"] = evidence_bundle_report.get("consolidated", 0)
        report["changed_evidence_bundle_ids"] = list(dict.fromkeys(
            changed_evidence_bundle_ids
        ))
        # Cluster facts inside each evidence bundle into semantic observations.
        report["observation_ids"] = self._update_observations_for_evidence_bundles(
            report["changed_evidence_bundle_ids"]
        )
        report["observations_updated"] = len(report["observation_ids"])
        report["observations_generated"] = report["observations_updated"]
        report["interpretations_generated"] = 0
        report["interpretations_generated"] = self._reflect_generate_interpretations_using_observations(
            report["changed_evidence_bundle_ids"]
        )
        
        self._log_info(
            "memory_reflect",
            "finish", 
            {
                "evidence_bundles_consolidated": report.get("evidence_bundles_consolidated", 0),
                "task_matched": evidence_bundle_report.get("task_matched", 0),
                "task_updates": evidence_bundle_report.get("task_updates", 0),
                "entity_merged": report.get("merged", 0),
                "evidence_bundle_groups_merged": report.get("evidence_bundle_groups_merged", 0),
                "observations_updated": report.get("observations_updated", 0),
                "interpretations_generated": report.get("interpretations_generated", 0),
                "interpretation_feedback_applied": feedback_report.get("applied", 0),
                "tasks_paused": report.get("tasks_paused", 0),
                "tasks_stale": report.get("tasks_stale", 0),
            })
        return report

    # ── Recall relevant memory nodes ──────────────────────────────────────

    @staticmethod
    def _infer_recall_intent(query: str, keywords: List[str]) -> str:
        """Classify recall needs without an extra LLM call."""
        haystack = " ".join([query or "", *(keywords or [])]).lower()
        action_terms = {
            "偏好", "喜欢", "倾向", "应该", "怎么做", "继续", "任务", "todo",
            "task", "preference", "prefer", "should", "plan", "next",
        }
        evidence_terms = {
            "之前", "上次", "什么时候", "哪里", "哪次", "说过", "提到过", "记录",
            "历史", "具体", "原文", "证据", "when", "where", "what did", "history",
            "specific", "quote", "evidence",
        }
        state_terms = {
            "最近", "通常", "一直", "经常", "模式", "趋势", "变化", "状态", "总结",
            "recent", "usually", "often", "pattern", "trend", "status", "summary",
        }
        if any(term in haystack for term in action_terms):
            return "action"
        if any(term in haystack for term in evidence_terms):
            return "evidence"
        if any(term in haystack for term in state_terms):
            return "state"
        return "balanced"

    @classmethod
    def _resolve_recall_intent(
        cls,
        query: str,
        keywords: List[str],
        llm_intent: str,
        intent_confidence: float,
    ) -> str:
        """Combine cheap rule intent with LLM query-analysis intent."""
        rule_intent = cls._infer_recall_intent(query, keywords)
        normalized_llm_intent = cls._normalize_recall_intent(llm_intent)
        confidence = cls._clip_unit_float(intent_confidence, 0.0)
        if rule_intent == "evidence" and normalized_llm_intent != "evidence":
            return "evidence"
        if confidence >= 0.65:
            return normalized_llm_intent
        return rule_intent

    @staticmethod
    def _recall_layer_limits(k: int, intent: str) -> Dict[str, int]:
        """Allocate a small recall budget across the three memory layers."""
        base = max(1, int(k or 1))
        if intent == "action":
            return {
                "interpretations": max(3, min(6, base)),
                "observations": max(2, min(5, base)),
                "facts": max(2, base // 2),
            }
        if intent == "evidence":
            return {
                "interpretations": max(1, min(3, base // 2 or 1)),
                "observations": max(2, min(4, base // 2 or 1)),
                "facts": base,
            }
        if intent == "state":
            return {
                "interpretations": max(2, min(4, base // 2 or 1)),
                "observations": max(3, min(6, base)),
                "facts": max(2, base // 2),
            }
        return {
            "interpretations": max(2, min(4, base // 2)),
            "observations": max(2, min(4, base // 2)),
            "facts": base,
        }

    @classmethod
    def _apply_recall_layer_preference(
        cls,
        layer_limits: Dict[str, int],
        layer_preference: Dict[str, float],
    ) -> Dict[str, int]:
        """Blend LLM layer preference into the rule-derived recall budget."""
        if not layer_preference:
            return dict(layer_limits)
        keys = ("interpretations", "observations", "facts")
        total_budget = max(1, sum(max(0, int(layer_limits.get(key, 0) or 0)) for key in keys))
        base_total = max(1, sum(max(0, int(layer_limits.get(key, 0) or 0)) for key in keys))
        base_share = {
            key: max(0, int(layer_limits.get(key, 0) or 0)) / base_total
            for key in keys
        }
        normalized_preference = cls._normalize_recall_layer_preference(layer_preference)
        if not normalized_preference:
            return dict(layer_limits)
        blended = {
            key: (base_share.get(key, 0.0) * 0.55) + (normalized_preference.get(key, 0.0) * 0.45)
            for key in keys
        }
        limits = {
            key: max(1, int(round(total_budget * blended.get(key, 0.0))))
            for key in keys
        }
        diff = total_budget - sum(limits.values())
        while diff != 0:
            if diff > 0:
                key = max(keys, key=lambda item: blended.get(item, 0.0))
                limits[key] += 1
                diff -= 1
                continue
            removable = [
                key for key in keys
                if limits.get(key, 0) > 1
            ]
            if not removable:
                break
            key = min(removable, key=lambda item: blended.get(item, 0.0))
            limits[key] -= 1
            diff += 1
        return limits

    @staticmethod
    def _recall_embedding_text(query: str, analysis: Dict[str, Any]) -> str:
        pieces = [str(query or "").strip(), str(analysis.get("search_text") or "").strip()]
        keywords = analysis.get("keywords") or []
        if keywords:
            pieces.append("keywords: " + ", ".join(str(keyword) for keyword in keywords))
        entities = analysis.get("entities") or []
        entity_names = [
            str(entity.get("name") or "").strip()
            for entity in entities
            if isinstance(entity, dict) and str(entity.get("name") or "").strip()
        ]
        if entity_names:
            pieces.append("entities: " + ", ".join(entity_names))
        text = "\n".join(piece for piece in pieces if piece)
        return text or str(query or "")

    @staticmethod
    def _recall_log_interpretation_items(items: List[Dict[str, Any]], *, limit: int = 20) -> Dict[str, str]:
        items_info: Dict[str, str] = {}
        for item in items or []:
            item_id = item.get("id")
            items_info[str(item_id)] = item.get("claim")

        return items_info

    @staticmethod
    def _recall_log_observation_items(items: List[Dict[str, Any]], *, limit: int = 20) -> Dict[str, str]:
        items_info: Dict[str, str] = {}
        for item in items or []:
            item_id = item.get("id")
            items_info[str(item_id)] = item.get("summary")

        return items_info

    @staticmethod
    def _recall_log_item_ids(items: List[Dict[str, Any]], *, limit: int = 20) -> List[Any]:
        ids: List[Any] = []
        for item in items or []:
            item_id = item.get("id")
            if item_id is None:
                continue
            ids.append(item_id)
            if len(ids) >= limit:
                break
        return ids

    @staticmethod
    def _recall_log_entities(entities: List[Any], *, limit: int = 12) -> List[str]:
        out: List[str] = []
        seen = set()
        for entity in entities or []:
            if isinstance(entity, dict):
                text = str(entity.get("name") or "").strip()
            else:
                text = str(entity or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _merge_recall_items(*groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge recall records by id while preserving first-seen order."""
        merged: List[Dict[str, Any]] = []
        seen = set()
        for group in groups:
            for item in group or []:
                item_id = item.get("id")
                if item_id in seen:
                    continue
                seen.add(item_id)
                merged.append(item)
        return merged

    @classmethod
    def _recall_search_terms(
        cls,
        query: str,
        keywords: List[str],
        entities: List[Any],
    ) -> List[str]:
        """Build compact terms for cheap cross-layer recall reranking."""
        raw_terms: List[str] = []
        raw_terms.extend(str(keyword or "") for keyword in keywords or [])
        for entity in entities or []:
            if isinstance(entity, dict):
                raw_terms.append(str(entity.get("name") or ""))
            else:
                raw_terms.append(str(entity or ""))
        raw_terms.extend(re.split(r"\s+|,|，|;|；", str(query or "")))

        terms: List[str] = []
        seen = set()
        for term in raw_terms:
            text = str(term or "").strip().lower()
            if not text or len(text) <= 1 or text in seen:
                continue
            seen.add(text)
            terms.append(text)
            if len(terms) >= 16:
                break
        return terms

    @staticmethod
    def _recall_item_text(layer: str, item: Dict[str, Any]) -> str:
        if layer == "interpretation":
            keys = (
                "claim", "action_implication", "subject_text", "target_text",
                "scope", "interpretation_type", "resolution", "entity_name",
            )
        elif layer == "observation":
            keys = (
                "summary", "keywords", "topic_label", "topic_key",
                "observation_type", "entity_name",
            )
        else:
            keys = (
                "summary", "keywords", "topic", "fact_type", "fact_kind",
                "fact_subject", "task_relevance", "entity_names",
            )
        return " ".join(str(item.get(key) or "") for key in keys).lower()

    @staticmethod
    def _recall_layer_intent_weight(layer: str, intent: str, item: Dict[str, Any]) -> float:
        if intent == "action":
            if layer == "interpretation":
                return 1.25
            if layer == "observation":
                return 1.0
            return 0.8
        if intent == "evidence":
            if layer == "fact":
                return 1.25
            if layer == "observation":
                return 1.0
            return 0.75
        if intent == "state":
            if layer == "observation":
                return 1.25
            if layer == "interpretation":
                return 1.0
            return 0.85
        return 1.0

    @classmethod
    def _recall_intent_bonus(cls, layer: str, intent: str, item: Dict[str, Any]) -> float:
        if layer == "interpretation":
            interpretation_type = str(item.get("interpretation_type") or "").lower()
            if intent == "action" and interpretation_type in {
                "task", "preference", "explicit_preference", "inferred_preference",
            }:
                return 0.45
            if intent == "state" and interpretation_type in {
                "insight", "behavior_pattern", "project_state", "strategy", "task_risk",
            }:
                return 0.35
            if intent == "evidence":
                return -0.2
        if layer == "observation":
            metadata = cls._json_dict(item.get("metadata", {}))
            observation_type = str(item.get("observation_type") or "context").lower()
            temporal_scope = str(metadata.get("temporal_scope") or "").lower()
            if intent == "action" and observation_type in {
                "task_state",
                "task_progress",
                "decision",
                "preference_signal",
                "constraint",
            }:
                return 0.35
            if intent == "state" and (
                observation_type in {
                    "task_state",
                    "task_progress",
                    "decision",
                    "problem",
                    "behavior_pattern",
                    "context",
                }
                or temporal_scope in {"ongoing", "recurring", "recent"}
            ):
                return 0.35
            if intent == "evidence" and observation_type in {
                "task_progress",
                "decision",
                "problem",
                "context",
            }:
                return 0.25
        if layer == "fact":
            fact_type = str(item.get("fact_type") or "").lower()
            fact_kind = str(item.get("fact_kind") or "").lower()
            if intent == "evidence" and fact_type == "episodic":
                return 0.35
            if intent == "action" and fact_kind in {"preference", "instruction", "request", "task"}:
                return 0.3
            if intent == "state" and fact_type == "semantic":
                return 0.2
        return 0.0

    @classmethod
    def _recall_candidate_score(
        cls,
        *,
        layer: str,
        item: Dict[str, Any],
        rank: int,
        terms: List[str],
        intent: str,
    ) -> float:
        haystack = cls._recall_item_text(layer, item)
        matched_terms = [term for term in terms if term and term in haystack]
        keyword_score = min(2.0, len(matched_terms) * 0.35)
        rank_score = 1.0 / max(1, rank)
        embedding_score = max(0.0, float(item.get("embedding_similarity") or 0.0))
        if layer in {"interpretation", "observation"}:
            confidence = cls._clip_unit_float(item.get("confidence"), 0.0)
            decay_score = cls._clip_unit_float(item.get("decay_score"), 1.0)
            reliability = (confidence * 0.75) + (decay_score * 0.25)
        else:
            reliability = cls._clip_unit_float(item.get("decay_score"), 1.0)
        score = (
            keyword_score
            + (rank_score * 0.9)
            + (embedding_score * 1.4)
            + (reliability * 0.45)
            + cls._recall_intent_bonus(layer, intent, item)
        )
        score *= cls._recall_layer_intent_weight(layer, intent, item)
        return round(float(score), 4)

    @staticmethod
    def _recall_keyword_terms(keyword: Any) -> List[str]:
        keyword_query = " ".join(keyword) if isinstance(keyword, list) else str(keyword or "")
        return [
            term.strip().lower()
            for term in re.split(r"\s+|OR", keyword_query)
            if term.strip()
        ]

    @staticmethod
    def _recall_entity_terms(entities: Optional[List[Any]]) -> List[str]:
        terms: List[str] = []
        for entity in entities or []:
            if isinstance(entity, dict):
                name = str(entity.get("name", "")).strip()
            else:
                name = str(entity or "").strip()
            if name:
                terms.append(name.lower())
        return terms

    @classmethod
    def _recall_embedding_similarity(
        cls,
        query_embedding: Optional[np.ndarray],
        stored_embedding: Any,
    ) -> Optional[float]:
        if query_embedding is None or stored_embedding is None:
            return None
        similarity = cls._cal_embedding_similarity(query_embedding, stored_embedding)
        return max(0.0, min(1.0, float(similarity)))

    @classmethod
    def _rank_interpretation_search_candidates(
        cls,
        candidates: List[Dict[str, Any]],
        *,
        keyword: Any,
        entities: Optional[List[Any]],
        top_k: int,
        query_embedding: Optional[np.ndarray],
        min_embedding_similarity: Optional[float],
    ) -> List[Dict[str, Any]]:
        """Filter and rank interpretation rows fetched by SessionDB."""
        terms = cls._recall_keyword_terms(keyword)
        entity_terms = cls._recall_entity_terms(entities)
        threshold = (
            None
            if min_embedding_similarity is None
            else max(0.0, min(1.0, float(min_embedding_similarity)))
        )
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for candidate in candidates or []:
            item = dict(candidate)
            content_haystack = " ".join(
                str(item.get(key) or "")
                for key in (
                    "claim", "action_implication", "subject_text", "target_text",
                    "scope", "interpretation_type", "resolution",
                )
            ).lower()
            entity_haystack = str(item.get("entity_name") or "").lower()
            matched_terms = [term for term in terms if term in content_haystack]
            matched_entity_name_terms = [
                term
                for term in terms
                if term in entity_haystack and term not in matched_terms
            ]
            entity_matches = sum(1 for term in entity_terms if term in entity_haystack)
            embedding_similarity = cls._recall_embedding_similarity(
                query_embedding,
                item.get("embedding"),
            )
            if (
                threshold is not None
                and query_embedding is not None
                and (
                    embedding_similarity is None
                    or embedding_similarity < threshold
                )
            ):
                continue
            embedding_match = embedding_similarity is not None and embedding_similarity >= 0.35
            strong_embedding_match = embedding_similarity is not None and embedding_similarity >= 0.55
            if terms or entity_terms:
                if entity_terms:
                    if (
                        entity_matches <= 0
                        and not matched_terms
                        and not matched_entity_name_terms
                        and not strong_embedding_match
                    ):
                        continue
                elif not matched_terms and not matched_entity_name_terms and not embedding_match:
                    continue
            else:
                matched_terms = ["_"]
            keyword_score = (len(matched_terms) * 1.2) + (len(matched_entity_name_terms) * 0.6)
            embedding_score = max(0.0, float(embedding_similarity or 0.0))
            confidence_score = cls._clip_unit_float(item.get("confidence"), 0.0)
            decay_score = cls._clip_unit_float(item.get("decay_score"), 1.0)
            score = (
                keyword_score
                + (entity_matches * 1.5)
                + (embedding_score * 1.4)
                + confidence_score
                + (decay_score * 0.4)
                + (0.5 if item.get("status") == "current" else 0.0)
            )
            if embedding_similarity is not None:
                item["embedding_similarity"] = round(float(embedding_similarity), 4)
            item.pop("embedding", None)
            scored.append((score, item))

        scored.sort(
            key=lambda pair: (
                pair[0],
                pair[1].get("last_supported_at")
                or pair[1].get("updated_at")
                or pair[1].get("created_at")
                or "",
            ),
            reverse=True,
        )
        return [item for _, item in scored[:max(1, int(top_k or 3))]]

    @classmethod
    def _rank_observation_search_candidates(
        cls,
        candidates: List[Dict[str, Any]],
        *,
        keyword: Any,
        entities: Optional[List[Any]],
        top_k: int,
        query_embedding: Optional[np.ndarray],
        min_embedding_similarity: Optional[float],
    ) -> List[Dict[str, Any]]:
        """Filter and rank observation rows fetched by SessionDB."""
        terms = cls._recall_keyword_terms(keyword)
        entity_terms = cls._recall_entity_terms(entities)
        threshold = (
            None
            if min_embedding_similarity is None
            else max(0.0, min(1.0, float(min_embedding_similarity)))
        )
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for candidate in candidates or []:
            item = dict(candidate)
            metadata = cls._json_dict(item.get("metadata", {}))
            item["metadata"] = metadata
            entity_name = str(item.get("entity_name") or "").lower()
            haystack = " ".join([
                str(item.get("summary") or ""),
                str(item.get("observation_type") or ""),
                str(item.get("topic_label") or ""),
                entity_name,
            ]).lower()
            matched_terms = [term for term in terms if term in haystack]
            entity_matches = sum(
                1
                for term in entity_terms
                if term in entity_name or term in haystack
            )
            embedding_similarity = cls._recall_embedding_similarity(
                query_embedding,
                item.get("embedding"),
            )
            if (
                threshold is not None
                and query_embedding is not None
                and (
                    embedding_similarity is None
                    or embedding_similarity < threshold
                )
            ):
                continue
            embedding_match = embedding_similarity is not None and embedding_similarity >= 0.35
            if terms or entity_terms:
                if entity_terms:
                    if entity_matches <= 0 and not matched_terms and not embedding_match:
                        continue
                elif not matched_terms and not embedding_match:
                    continue
            score = (
                len(matched_terms)
                + (entity_matches * 1.5)
                + (max(0.0, float(embedding_similarity or 0.0)) * 1.4)
                + cls._clip_unit_float(item.get("confidence"), 0.0)
                + (cls._clip_unit_float(item.get("decay_score"), 1.0) * 0.4)
            )
            if embedding_similarity is not None:
                item["embedding_similarity"] = round(float(embedding_similarity), 4)
            item.pop("embedding", None)
            item.pop("evidence_centroid_embedding", None)
            scored.append((score, item))

        scored.sort(
            key=lambda pair: (
                pair[0],
                pair[1].get("last_supported_at")
                or pair[1].get("updated_at")
                or pair[1].get("created_at")
                or "",
            ),
            reverse=True,
        )
        return [item for _, item in scored[:max(1, int(top_k or 3))]]

    @classmethod
    def _rank_recall_raw_candidates(
        cls,
        *,
        interpretations: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
        semantic_facts: List[Dict[str, Any]],
        episodic_facts: List[Dict[str, Any]],
        terms: List[str],
        intent: str,
        layer_limits: Dict[str, int],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Select final recall candidates while preserving each layer's own order."""
        total_budget = max(1, sum(max(0, int(value or 0)) for value in layer_limits.values()))
        flexible_cap = max(1, int(total_budget * 0.6))
        caps = {
            "interpretations": max(layer_limits.get("interpretations", 0), flexible_cap),
            "observations": max(layer_limits.get("observations", 0), flexible_cap),
            "semantic_facts": max(1, min(total_budget, flexible_cap)),
            "episodic_facts": max(1, min(total_budget, flexible_cap)),
        }
        candidates: List[Tuple[float, int, str, Dict[str, Any]]] = []
        sequence = 0
        groups = [
            ("interpretations", "interpretation", interpretations),
            ("observations", "observation", observations),
            ("semantic_facts", "fact", semantic_facts),
            ("episodic_facts", "fact", episodic_facts),
        ]
        for bucket, layer, items in groups:
            for rank, item in enumerate(items or [], 1):
                sequence += 1
                score = cls._recall_candidate_score(
                    layer=layer,
                    item=item,
                    rank=rank,
                    terms=terms,
                    intent=intent,
                )
                item["_recall_score"] = score
                item["_recall_rank"] = rank
                candidates.append((score, sequence, bucket, item))

        candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
        selected = {
            "interpretations": [],
            "observations": [],
            "semantic_facts": [],
            "episodic_facts": [],
        }
        selected_ids = {key: set() for key in selected}

        for _score, _sequence, bucket, item in candidates:
            if sum(len(values) for values in selected.values()) >= total_budget:
                break
            if len(selected[bucket]) >= caps[bucket]:
                continue
            item_id = item.get("id")
            if item_id in selected_ids[bucket]:
                continue
            selected_ids[bucket].add(item_id)
            selected[bucket].append(item)
        for items in selected.values():
            items.sort(key=lambda item: int(item.get("_recall_rank") or 0))
        return selected

    @classmethod
    def _rank_recall_candidates(cls, **kwargs: Any) -> Dict[str, List[Dict[str, Any]]]:
        return cls._rank_recall_raw_candidates(**kwargs)

    def _retrieve_recall_raw_candidates(
        self,
        *,
        keywords: List[str],
        entities: List[Any],
        query_embedding: np.ndarray,
        candidate_limits: Dict[str, int],
        raw_candidate_limits: Dict[str, int],
        layer_limits: Dict[str, int],
        fact_type_preference: str,
        budget: str,
        time_start: Optional[str],
        time_end: Optional[str],
        tags: Optional[List[str]],
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        """Retrieve and rank first-pass recall candidates for each memory layer."""
        raw_interpretation_candidates = self._db.search_memory_interpretations(
            entities=entities,
            top_k=raw_candidate_limits["interpretations"],
        )
        interpretation_candidates = self._rank_interpretation_search_candidates(
            raw_interpretation_candidates,
            keyword=keywords,
            entities=entities,
            top_k=candidate_limits["interpretations"],
            query_embedding=query_embedding,
            min_embedding_similarity=self._recall_interpretation_min_embedding_similarity,
        )

        raw_observation_candidates = self._db.search_memory_observations(
            entities=entities,
            top_k=raw_candidate_limits["observations"],
        )
        observation_candidates = self._rank_observation_search_candidates(
            raw_observation_candidates,
            keyword=keywords,
            entities=entities,
            top_k=candidate_limits["observations"],
            query_embedding=query_embedding,
            min_embedding_similarity=self._recall_observation_min_embedding_similarity,
        )

        fact_candidate_limit = max(
            candidate_limits["facts"],
            layer_limits["facts"] * 3,
            layer_limits["facts"] + 4,
        )
        semantic_candidate_limit = max(1, fact_candidate_limit)
        episodic_candidate_limit = max(1, fact_candidate_limit)
        if fact_type_preference == "semantic":
            episodic_candidate_limit = max(1, layer_limits["facts"])
        elif fact_type_preference == "episodic":
            semantic_candidate_limit = max(1, layer_limits["facts"])
        semantic_candidates = self._db.search_memory_facts(
            keywords,
            query_embedding,
            top_k=semantic_candidate_limit,
            budget=budget,
            time_start=time_start, time_end=time_end,
            tags=tags,
            fact_types=["semantic"],
        )

        episodic_candidates = self._db.search_memory_facts(
            keywords,
            query_embedding,
            top_k=episodic_candidate_limit,
            budget=budget,
            time_start=time_start, time_end=time_end,
            tags=tags,
            fact_types=["episodic"],
        )
        self._log_info("memory_recall", "candidates_found", {
            "interpretations": {
                **self._recall_log_interpretation_items(interpretation_candidates)
            },
            "observations": {
                **self._recall_log_observation_items(observation_candidates)
            },
            "semantic_facts": {
                **self._recall_log_observation_items(semantic_candidates, limit=semantic_candidate_limit),
                "top_k": semantic_candidate_limit,
            },
            "episodic_facts": {
                **self._recall_log_observation_items(episodic_candidates, limit=episodic_candidate_limit),
                "top_k": episodic_candidate_limit,
            },
        })
        return interpretation_candidates, observation_candidates, semantic_candidates, episodic_candidates

    def _retrieve_recall_support_candidates(
        self,
        interpretation_candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Collect ranked linked observations and facts for interpretation candidates."""
        candidate_observation_ids_from_interpretation: List[int] = []
        candidate_fact_ids_from_interpretation: List[int] = []
        interpretation_observation_ids: Dict[int, List[int]] = {}
        interpretation_fact_ids: Dict[int, List[int]] = {}
        interpretation_by_id: Dict[int, Dict[str, Any]] = {}
        for interpretation in interpretation_candidates:
            try:
                interpretation_id = int(interpretation.get("id"))
            except (TypeError, ValueError):
                continue
            interpretation_by_id[interpretation_id] = interpretation
            observation_ids: List[int] = []
            for value in (
                interpretation.get("evidence_observation_ids", []) or []
            ) + (
                interpretation.get("counter_evidence_observation_ids", []) or []
            ):
                try:
                    observation_ids.append(int(value))
                except (TypeError, ValueError):
                    continue
            fact_ids: List[int] = []
            for value in (
                interpretation.get("evidence_fact_ids", []) or []
            ) + (
                interpretation.get("counter_evidence_fact_ids", []) or []
            ):
                try:
                    fact_ids.append(int(value))
                except (TypeError, ValueError):
                    continue
            observation_ids = list(dict.fromkeys(observation_ids))
            fact_ids = list(dict.fromkeys(fact_ids))
            interpretation_observation_ids[interpretation_id] = observation_ids
            interpretation_fact_ids[interpretation_id] = fact_ids
            candidate_observation_ids_from_interpretation.extend(observation_ids)
            candidate_fact_ids_from_interpretation.extend(fact_ids)
        candidate_observation_ids_from_interpretation = list(
            dict.fromkeys(candidate_observation_ids_from_interpretation)
        )
        candidate_fact_ids_from_interpretation = list(
            dict.fromkeys(candidate_fact_ids_from_interpretation)
        )
        candidate_observations_from_interpretation = self._db.get_observations_by_ids(
            candidate_observation_ids_from_interpretation
        )
        candidate_observations_by_id = {
            int(observation["id"]): observation
            for observation in candidate_observations_from_interpretation
            if observation.get("id") is not None
        }
        candidate_facts_by_id = {
            int(fact["id"]): fact
            for fact in self._db.memory_facts_by_ids(
                candidate_fact_ids_from_interpretation
            )
            if fact.get("id") is not None
        }
        facts_by_observation = self._db.get_observation_supporting_facts(
            candidate_observation_ids_from_interpretation,
            per_observation=3,
        ) if candidate_observation_ids_from_interpretation else {}

        linked_observations: List[Dict[str, Any]] = []
        linked_facts: List[Dict[str, Any]] = []
        seen_fact_ids = set()
        max_observations_per_interpretation = 2
        max_facts_per_interpretation = 3
        max_facts_per_observation = 2

        def _fact_sort_key(item: Dict[str, Any]) -> Tuple[float, str, int]:
            return (
                float(item.get("_recall_support_score") or 0.0),
                item.get("time_key") or item.get("updated_at") or "",
                int(item.get("id") or 0),
            )

        for interpretation_id, interpretation in interpretation_by_id.items():
            parent_score = float(interpretation.get("_recall_score") or 0.0)
            ranked_observations: List[Dict[str, Any]] = []
            for observation_id in interpretation_observation_ids.get(interpretation_id, []):
                observation = candidate_observations_by_id.get(int(observation_id))
                if not observation:
                    continue
                support_item = dict(observation)
                support_item["_recall_support_parent_layer"] = "interpretation"
                support_item["_recall_support_parent_id"] = interpretation_id
                support_item["_recall_support_score"] = round(
                    (parent_score * 0.75)
                    + (float(support_item.get("confidence") or 0.0) * 0.2),
                    4,
                )
                ranked_observations.append(support_item)
            ranked_observations.sort(
                key=lambda item: (
                    float(item.get("_recall_support_score") or 0.0),
                    item.get("last_supported_at")
                    or item.get("updated_at")
                    or "",
                ),
                reverse=True,
            )
            selected_observations = ranked_observations[:max_observations_per_interpretation]
            linked_observations.extend(selected_observations)

            direct_facts: List[Dict[str, Any]] = []
            for fact_id in interpretation_fact_ids.get(interpretation_id, []):
                fact = candidate_facts_by_id.get(int(fact_id))
                if not fact:
                    continue
                support_item = dict(fact)
                support_item["_recall_support_parent_layer"] = "interpretation"
                support_item["_recall_support_parent_id"] = interpretation_id
                support_item["_recall_support_score"] = round(parent_score * 0.8, 4)
                direct_facts.append(support_item)
            direct_facts.sort(key=_fact_sort_key, reverse=True)
            for fact in direct_facts[:max_facts_per_interpretation]:
                fact_id = fact.get("id")
                if fact_id in seen_fact_ids:
                    continue
                seen_fact_ids.add(fact_id)
                linked_facts.append(fact)

            for observation in selected_observations:
                observation_id = int(observation["id"])
                observation_score = float(observation.get("_recall_support_score") or 0.0)
                observation_facts: List[Dict[str, Any]] = []
                for fact in facts_by_observation.get(observation_id, []):
                    support_item = dict(fact)
                    support_item["_recall_support_parent_layer"] = "observation"
                    support_item["_recall_support_parent_id"] = observation_id
                    support_item["_recall_support_score"] = round(observation_score * 0.85, 4)
                    observation_facts.append(support_item)
                observation_facts.sort(key=_fact_sort_key, reverse=True)
                for fact in observation_facts[:max_facts_per_observation]:
                    fact_id = fact.get("id")
                    if fact_id in seen_fact_ids:
                        continue
                    seen_fact_ids.add(fact_id)
                    linked_facts.append(fact)

        self._log_info("memory_recall", "support_candidates_built", {
            "interpretation_observation_ids": candidate_observation_ids_from_interpretation,
            "interpretation_fact_ids": candidate_fact_ids_from_interpretation,
            "linked_observations": {
                "count": len(linked_observations),
                "ids": self._recall_log_item_ids(linked_observations),
            },
            "linked_facts": {
                "count": len(linked_facts),
                "ids": self._recall_log_item_ids(linked_facts),
            },
        })
        return {
            "linked_observations": linked_observations,
            "linked_facts": linked_facts,
        }

    def recall(
        self,
        query: str,
        top_k: int = None,
        budget: str = None,
        tags: Optional[List[str]] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
    ) -> str:
        """Search for memory nodes relevant to *query*.

        Uses hybrid search (keyword + vector + entity graph + node relations),
        with budget controlling entity graph traversal depth.

        Supports time range filtering:
        - Pass *time_start* and/or *time_end* explicitly (ISO timestamp strings).
        - If the *query* contains Chinese time expressions like "最近一周",
          "上个月", "2025年3月到6月", they are automatically parsed and applied,
          and the time expression is stripped from the search query.

        Args:
            query: The user's current query / context.
            top_k: Override default retrieval count.
            budget: "low", "mid" (default), or "high".
            tags: Optional list of tags to filter by.
            time_start: Optional ISO timestamp start filter.
            time_end: Optional ISO timestamp end filter.

        Returns formatted markdown text, or empty string if nothing relevant.
        """
        if not self._enabled or not query:
            return ""

        started_at = time.monotonic()
        try:
            k = top_k or self._top_k
            b = budget or self._recall_budget
            self._log_info("memory_recall", "start", {
                "query": self._reflect_log_text(query, limit=300),
                "top_k": k,
                "budget": b,
                "tags": tags or [],
                "time_start": time_start,
                "time_end": time_end,
            })

            # Detect and extract time expressions from the query
            _parsed_time_start, _parsed_time_end, clean_query = self._parse_time_expression(query)
            ts = time_start or _parsed_time_start
            te = time_end or _parsed_time_end
            search_query = clean_query or query
            self._log_info("memory_recall", "query_prepared", {
                "search_query": self._reflect_log_text(search_query, limit=300),
                "clean_query": self._reflect_log_text(clean_query, limit=300),
                "parsed_time_start": _parsed_time_start,
                "parsed_time_end": _parsed_time_end,
                "effective_time_start": ts,
                "effective_time_end": te,
            })

            gate = self._rule_based_recall_gate(search_query)
            self._log_info("memory_recall", "gate_decided", {
                "decision": gate["decision"],
                "reason": gate["reason"],
            })
            if gate["decision"] == "skip":
                self._log_info("memory_recall", "skip", {
                    "reason": gate["reason"],
                    "query": self._reflect_log_text(search_query, limit=300),
                    "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
                })
                return ""

            # Both explicit recall requests and ambiguous queries need the LLM
            # analysis to produce retrieval-oriented search text and strategy.
            query_analysis = self._analyze_recall_query(search_query)
            if not query_analysis:
                self._log_info("memory_recall", "skip", {
                    "reason": "query_analysis_empty",
                    "gate_decision": gate["decision"],
                    "gate_reason": gate["reason"],
                    "query": self._reflect_log_text(search_query, limit=300),
                    "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
                })
                return ""
            analysis_source = "llm"

            if (
                gate["decision"] != "recall"
                and not query_analysis.get("needs_recall", True)
            ):
                self._log_info("memory_recall", "skip", {
                    "reason": query_analysis.get("recall_reason") or "llm_recall_not_needed",
                    "decision_source": analysis_source,
                    "recall_confidence": query_analysis.get("recall_confidence"),
                    "query": self._reflect_log_text(search_query, limit=300),
                    "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
                })
                return ""

            if not self._ensure_embedding_client():
                self._log_info("memory_recall", "skip", {
                    "reason": "embedding_client_unavailable",
                    "query": self._reflect_log_text(query, limit=300),
                    "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
                })
                return ""

            # Generate embedding from a retrieval-oriented query expression.
            query_embedding_text = self._recall_embedding_text(search_query, query_analysis)
            query_embedding = self._embedding_client.embed_text(query_embedding_text)
            if query_embedding is None:
                self._log_info("memory_recall", "skip", {
                    "reason": "query_embedding_empty",
                    "query": self._reflect_log_text(search_query, limit=300),
                    "embedding_text": self._reflect_log_text(query_embedding_text, limit=300),
                    "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
                })
                return ""

            keywords = query_analysis["keywords"]
            entities = query_analysis.get("entities", [])
            recall_intent = self._resolve_recall_intent(
                search_query,
                keywords,
                query_analysis.get("recall_intent", "balanced"),
                float(query_analysis.get("intent_confidence") or 0.0),
            )
            layer_limits = self._apply_recall_layer_preference(
                self._recall_layer_limits(k, recall_intent),
                query_analysis.get("layer_preference", {}),
            )
            recall_terms = self._recall_search_terms(search_query, keywords, entities)
            candidate_limits = {
                layer: max(limit * 3, limit + 4)
                for layer, limit in layer_limits.items()
            }
            raw_candidate_limits = {
                "interpretations": max(candidate_limits["interpretations"] * 10, 50),
                "observations": max(candidate_limits["observations"] * 10, 50),
            }
            self._log_info("memory_recall", "query_analyzed", {
                "search_text": self._reflect_log_text(query_analysis.get("search_text"), limit=300),
                "analysis_source": analysis_source,
                "needs_recall": query_analysis.get("needs_recall"),
                "recall_confidence": query_analysis.get("recall_confidence"),
                "recall_reason": query_analysis.get("recall_reason"),
                "keywords": keywords,
                "entities": self._recall_log_entities(entities),
                "llm_recall_intent": query_analysis.get("recall_intent"),
                "intent_confidence": query_analysis.get("intent_confidence"),
                "resolved_intent": recall_intent,
                "layer_preference": query_analysis.get("layer_preference", {}),
                "fact_type_preference": query_analysis.get("fact_type_preference", "both"),
                "time_sensitivity": query_analysis.get("time_sensitivity"),
                "needs_evidence": query_analysis.get("needs_evidence"),
                "layer_limits": layer_limits,
                "candidate_limits": candidate_limits,
                "raw_candidate_limits": raw_candidate_limits,
                "embedding_text": self._reflect_log_text(query_embedding_text, limit=300),
            })

            (
                interpretation_candidates,
                observation_candidates,
                semantic_candidates,
                episodic_candidates,
            ) = self._retrieve_recall_raw_candidates(
                keywords=keywords,
                entities=entities,
                query_embedding=query_embedding,
                candidate_limits=candidate_limits,
                raw_candidate_limits=raw_candidate_limits,
                layer_limits=layer_limits,
                fact_type_preference=query_analysis.get("fact_type_preference", "both"),
                budget=b,
                time_start=ts,
                time_end=te,
                tags=tags,
            )

            ranked_recall = self._rank_recall_raw_candidates(
                interpretations=interpretation_candidates,
                observations=observation_candidates,
                semantic_facts=semantic_candidates,
                episodic_facts=episodic_candidates,
                terms=recall_terms,
                intent=recall_intent,
                layer_limits=layer_limits,
            )

            interpretation_nodes = ranked_recall["interpretations"]
            observation_nodes = ranked_recall["observations"]
            semantic_fact_nodes = ranked_recall["semantic_facts"]
            episodic_fact_nodes = ranked_recall["episodic_facts"]

            self._log_info("memory_recall", "ranked", {
                "interpretations": {
                    "count": len(interpretation_nodes),
                    "ids": self._recall_log_item_ids(interpretation_nodes),
                },
                "observations": {
                    "count": len(observation_nodes),
                    "ids": self._recall_log_item_ids(observation_nodes),
                },
                "semantic_facts": {
                    "count": len(semantic_fact_nodes),
                    "ids": self._recall_log_item_ids(semantic_fact_nodes),
                },
                "episodic_facts": {
                    "count": len(episodic_fact_nodes),
                    "ids": self._recall_log_item_ids(episodic_fact_nodes),
                },
            })

            support_candidates = self._retrieve_recall_support_candidates(
                interpretation_nodes
            )
            observation_nodes_from_interpretation = support_candidates["linked_observations"]
            fact_nodes_from_interpretation = support_candidates["linked_facts"]
            observation_nodes = self._merge_recall_items(
                observation_nodes,
                observation_nodes_from_interpretation,
            )
            fact_nodes_from_observation = self._db.get_observation_supporting_facts(
                [int(obs["id"]) for obs in observation_nodes],
                per_observation=2,
            ) if observation_nodes else {}

            fact_ids_from_observation = {
                fact["id"]
                for facts in fact_nodes_from_observation.values()
                for fact in facts
            }
            fact_ids_from_observation.update(fact["id"] for fact in fact_nodes_from_interpretation)
            semantic_fact_nodes = [fact for fact in semantic_fact_nodes if fact.get("id") not in fact_ids_from_observation]
            episodic_fact_nodes = [fact for fact in episodic_fact_nodes if fact.get("id") not in fact_ids_from_observation]

            self._log_info("memory_recall", "evidence_expanded", {
                "observation_ids_from_interpretations": [
                    item.get("id") for item in observation_nodes_from_interpretation
                ],
                "fact_ids_from_interpretations": [
                    item.get("id") for item in fact_nodes_from_interpretation
                ],
                "observations_from_interpretations": {
                    "count": len(observation_nodes_from_interpretation),
                    "ids": self._recall_log_item_ids(observation_nodes_from_interpretation),
                },
                "supporting_facts_from_observations": {
                    "observation_count": len(fact_nodes_from_observation),
                    "fact_count": sum(len(nodes) for nodes in fact_nodes_from_observation.values()),
                },
                "direct_facts_removed_as_support": len(fact_ids_from_observation),
                "remaining_semantic_facts": len(semantic_fact_nodes),
                "remaining_episodic_facts": len(episodic_fact_nodes),
            })

            if not interpretation_nodes and not observation_nodes and not semantic_fact_nodes and not episodic_fact_nodes:
                self._log_info("memory_recall", "finish", {
                    "status": "empty",
                    "reason": "no_relevant_nodes",
                    "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
                })
                return ""

            # Format results as raw text (no <memory-context> wrapper)
            lines: List[str] = []
            lines.append(MEMORY_NODE_HEADER)
            lines.append("")
            if interpretation_nodes:
                lines.append(INTERPRETATION_SECTION_HEADER)
                lines.append("System note: These are agent interpretations derived from memory evidence, not direct user quotes. Treat low-confidence or inferred claims cautiously.")
                for i, interpretation in enumerate(interpretation_nodes, 1):
                    lines.append(self._format_interpretation(i, interpretation))
                lines.append("")
            if observation_nodes:
                lines.append(OBSERVATION_SECTION_HEADER)
                lines.append("System note: These are consolidated long-term patterns. Treat them as high-level guidance supported by the facts below.")
                for i, observation in enumerate(observation_nodes, 1):
                    lines.append(self._format_observation(i, observation))
                lines.append("")
            support_lines: List[str] = []
            seen_support = set()
            support_index = 1
            for fact in fact_nodes_from_interpretation:
                if fact.get("id") in seen_support:
                    continue
                seen_support.add(fact.get("id"))
                support_lines.append(self._format_recall_fact_node(support_index, fact))
                support_index += 1
            for observation in observation_nodes:
                for fact in fact_nodes_from_observation.get(int(observation["id"]), []):
                    if fact.get("id") in seen_support:
                        continue
                    seen_support.add(fact.get("id"))
                    support_lines.append(self._format_recall_fact_node(support_index, fact))
                    support_index += 1
            if support_lines:
                lines.append(OBSERVATION_SUPPORT_SECTION_HEADER)
                lines.append("System note: These are source facts supporting the interpretations and observations above.")
                lines.extend(support_lines)
                lines.append("")
            if semantic_fact_nodes:
                lines.append(WORLD_FACT_SECTION_HEADER)
                lines.append("System note: These are semantic memories: stable facts, concepts, preferences, and background knowledge. Use them as background state, not as a new user request.")
                for i, fact in enumerate(semantic_fact_nodes, 1):
                    lines.append(self._format_recall_fact_node(i, fact))
                lines.append("")
            if episodic_fact_nodes:
                lines.append(EXPERIENCE_SECTION_HEADER)
                lines.append("System note: These are episodic memories: specific user/assistant experiences and events. Use them for timeline, prior attempts, outcomes, and context.")
                for i, fact in enumerate(episodic_fact_nodes, 1):
                    lines.append(self._format_recall_fact_node(i, fact))

            memory_text = "\n".join(lines)
            memory_text = memory_text.strip()
            self._log_info("memory_recall", "finish", {
                "status": "ok",
                "counts": {
                    "interpretations": len(interpretation_nodes),
                    "observations": len(observation_nodes),
                    "support_facts": len(support_lines),
                    "semantic_facts": len(semantic_fact_nodes),
                    "episodic_facts": len(episodic_fact_nodes),
                },
                "output_chars": len(memory_text),
                "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
            })
            if interpretation_nodes:
                try:
                    recall_event_id = self._db.memory_record_interpretation_recall_event(
                        query=search_query,
                        interpretations=interpretation_nodes,
                        metadata={
                            "source": "memory_recall",
                            "resolved_intent": recall_intent,
                            "query_analysis": {
                                "keywords": keywords,
                                "entities": self._recall_log_entities(entities),
                                "needs_evidence": query_analysis.get("needs_evidence"),
                            },
                        },
                    )
                    self._log_info(
                        "memory_recall",
                        "interpretation_recall_event_recorded",
                        {
                            "recall_event_id": recall_event_id,
                            "interpretation_ids": self._recall_log_item_ids(
                                interpretation_nodes
                            ),
                        },
                    )
                except Exception as exc:
                    logger.debug("Could not record interpretation recall event: %s", exc)
            return memory_text

        except Exception as e:
            self._log_info("memory_recall", "error", {
                "error": str(e),
                "query": self._reflect_log_text(query, limit=300),
                "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
            })
            return ""

    @staticmethod
    def _format_recall_fact_node(index: int, fact: Dict[str, Any]) -> str:
        node_summary = fact.get("summary", "")
        time_key = fact.get("time_key", "")
        kw = ", ".join(fact.get("keywords", []))
        line = f"{index}. [{time_key}] {node_summary}"
        if kw:
            line += f"  (关键词: {kw})"
        return line

    @staticmethod
    def _format_observation(index: int, observation: Dict[str, Any]) -> str:
        summary = observation.get("summary", "")
        entity = observation.get("entity_name", "")
        topic = observation.get("topic_label", "")
        updated = observation.get("last_supported_at") or observation.get("updated_at") or ""
        prefix = f"{entity} / {topic}".strip(" /")
        line = f"{index}. "
        if prefix:
            line += f"[{prefix}] "
        line += summary
        if updated:
            line += f"  (last supported: {updated})"
        return line

    @staticmethod
    def _format_interpretation(index: int, interpretation: Dict[str, Any]) -> str:
        claim = str(interpretation.get("claim") or "").strip()
        interpretation_type = str(interpretation.get("interpretation_type") or "behavior_pattern")
        status = str(interpretation.get("status") or "current")
        confidence = interpretation.get("confidence", 0.0)
        scope = str(interpretation.get("scope") or "general").strip()
        action = str(interpretation.get("action_implication") or "").strip()
        conflict_status = str(interpretation.get("conflict_status") or "none").strip()
        prefix_parts = [interpretation_type, f"status={status}", f"confidence={confidence}"]
        if scope:
            prefix_parts.append(f"scope={scope}")
        if conflict_status and conflict_status != "none":
            prefix_parts.append(f"conflict={conflict_status}")
        line = f"{index}. [{'; '.join(prefix_parts)}] {claim}"
        if action:
            line += f"  (action implication: {action})"
        return line

    # ── Time expression parser ───────────────────────────────────────────

    @staticmethod
    def _parse_time_expression(query: str) -> tuple:
        """Parse Chinese time expressions from *query*.

        Returns ``(time_start, time_end, clean_query)`` where *time_start* and
        *time_end* are ISO timestamp strings (``\"2026-04-20 00:00:00\"`` format)
        or ``None``, and *clean_query* is the query with the time expression
        stripped.

        Supported expressions:
          - ``最近N天/周/月/年`` → last N days/weeks/months/years
          - ``最近`` (alone) → last 7 days
          - ``上个月/上周/昨天/前天/去年`` → relative periods
          - ``本周/这个月/今年/今天`` → current periods
          - ``2025年3月到6月``, ``2025年3月至6月``
          - ``从2025年3月到2025年6月``
          - ``过去N天``, ``近N天``, ``近N周``, ``近N个月``
        """
        import datetime
        import re

        now = datetime.datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        time_start: Optional[str] = None
        time_end: Optional[str] = None
        clean_query = query

        # Pattern: 最近N天/周/月/年  or  近N天/周/月  or  过去N天
        m = re.search(r'(?:最近|近|过去)\s*(\d+)\s*(天|日|周|星期|个月|月|年)', query)
        if m:
            num = int(m.group(1))
            unit = m.group(2)
            if unit in ('天', '日'):
                delta = datetime.timedelta(days=num)
            elif unit in ('周', '星期'):
                delta = datetime.timedelta(weeks=num)
            elif unit in ('个月', '月'):
                delta = datetime.timedelta(days=num * 30)
            elif unit == '年':
                delta = datetime.timedelta(days=num * 365)
            else:
                delta = datetime.timedelta(days=num)
            time_start = (now - delta).strftime("%Y-%m-%d %H:%M:%S")
            time_end = now.strftime("%Y-%m-%d %H:%M:%S")
            clean_query = query[:m.start()] + query[m.end():]
            return time_start, time_end, clean_query.strip()

        # Pattern: 最近 (standalone) → last 7 days
        m = re.search(r'最近\s*', query)
        if m:
            time_start = (now - datetime.timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
            time_end = now.strftime("%Y-%m-%d %H:%M:%S")
            clean_query = query[:m.start()] + query[m.end():]
            return time_start, time_end, clean_query.strip()

        # Pattern: 从2025年3月到2025年6月  or  2025年3月到6月  or  2025年3月至6月
        m = re.search(r'(?:从)?\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(?:到|至)\s*(?:(\d{4})\s*年)?\s*(\d{1,2})\s*月', query)
        if m:
            year1 = int(m.group(1))
            month1 = int(m.group(2))
            year2 = int(m.group(3)) if m.group(3) else year1
            month2 = int(m.group(4))
            try:
                dt1 = datetime.datetime(year1, month1, 1)
                dt2 = datetime.datetime(year2, month2 + 1, 1) - datetime.timedelta(days=1) if month2 < 12 else datetime.datetime(year2, 12, 31, 23, 59, 59)
                time_start = dt1.strftime("%Y-%m-%d %H:%M:%S")
                time_end = dt2.strftime("%Y-%m-%d %H:%M:%S")
                clean_query = query[:m.start()] + query[m.end():]
                return time_start, time_end, clean_query.strip()
            except ValueError:
                pass

        # Pattern: 2025年3月 (specific month)
        m = re.search(r'(\d{4})\s*年\s*(\d{1,2})\s*月', query)
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
            try:
                dt1 = datetime.datetime(year, month, 1)
                dt2 = datetime.datetime(year, month + 1, 1) - datetime.timedelta(days=1) if month < 12 else datetime.datetime(year, 12, 31, 23, 59, 59)
                time_start = dt1.strftime("%Y-%m-%d %H:%M:%S")
                time_end = dt2.strftime("%Y-%m-%d %H:%M:%S")
                clean_query = query[:m.start()] + query[m.end():]
                return time_start, time_end, clean_query.strip()
            except ValueError:
                pass

        # Pattern: 上个月/上星期/上周 → last month/week
        m = re.search(r'上(?:个)?(?:月|星期|周)', query)
        if m:
            unit = m.group()[1:]
            if '月' in unit:
                first_of_month = today_start.replace(day=1)
                end_of_last_month = first_of_month - datetime.timedelta(days=1)
                start_of_last_month = end_of_last_month.replace(day=1)
                time_start = start_of_last_month.strftime("%Y-%m-%d %H:%M:%S")
                time_end = end_of_last_month.strftime("%Y-%m-%d %H:%M:%S")
            else:  # 周/星期
                start_of_this_week = today_start - datetime.timedelta(days=today_start.weekday())
                start_of_last_week = start_of_this_week - datetime.timedelta(days=7)
                time_start = start_of_last_week.strftime("%Y-%m-%d %H:%M:%S")
                time_end = start_of_this_week.strftime("%Y-%m-%d %H:%M:%S")
            clean_query = query[:m.start()] + query[m.end():]
            return time_start, time_end, clean_query.strip()

        # Pattern: 这个月/本月 → this month
        m = re.search(r'(?:这个月|本月)', query)
        if m:
            time_start = today_start.replace(day=1).strftime("%Y-%m-%d %H:%M:%S")
            time_end = now.strftime("%Y-%m-%d %H:%M:%S")
            clean_query = query[:m.start()] + query[m.end():]
            return time_start, time_end, clean_query.strip()

        # Pattern: 本周/这一周 → this week
        m = re.search(r'(?:本周|这一周)', query)
        if m:
            time_start = (today_start - datetime.timedelta(days=today_start.weekday())).strftime("%Y-%m-%d %H:%M:%S")
            time_end = now.strftime("%Y-%m-%d %H:%M:%S")
            clean_query = query[:m.start()] + query[m.end():]
            return time_start, time_end, clean_query.strip()

        # Pattern: 昨天 → yesterday
        m = re.search(r'昨天|昨日', query)
        if m:
            yesterday = today_start - datetime.timedelta(days=1)
            time_start = yesterday.strftime("%Y-%m-%d %H:%M:%S")
            time_end = (yesterday + datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
            clean_query = query[:m.start()] + query[m.end():]
            return time_start, time_end, clean_query.strip()

        # Pattern: 前天 → day before yesterday
        m = re.search(r'前天|前日', query)
        if m:
            day_before = today_start - datetime.timedelta(days=2)
            time_start = day_before.strftime("%Y-%m-%d %H:%M:%S")
            time_end = (day_before + datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
            clean_query = query[:m.start()] + query[m.end():]
            return time_start, time_end, clean_query.strip()

        # Pattern: 今天 → today
        m = re.search(r'今天|今日', query)
        if m:
            time_start = today_start.strftime("%Y-%m-%d %H:%M:%S")
            time_end = now.strftime("%Y-%m-%d %H:%M:%S")
            clean_query = query[:m.start()] + query[m.end():]
            return time_start, time_end, clean_query.strip()

        return None, None, query

    def turn_count(self) -> int:
        """Return the number of turns processed by this manager."""
        return self._turn_count
