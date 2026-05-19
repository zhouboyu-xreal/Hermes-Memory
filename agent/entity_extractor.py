#!/usr/bin/env python3
"""Entity Extractor — LLM-driven entity and relation extraction for memory knowledge graph.

Extracts structured entities (people, projects, concepts, technologies, etc.)
and their relations from conversation turns, then stores them in SessionDB's
knowledge graph tables (entity_nodes, entity_edges, memory_node_entities).

Usage::

    from agent.entity_extractor import EntityExtractor

    extractor = EntityExtractor(session_db, llm_client=agent.client)
    await extractor.extract_from_turn(
        node_id=42,
        user_message="帮我写一个FAISS向量搜索的代码",
        assistant_response="好的，我来实现一个FAISS向量搜索...",
    )
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from agent.temporal_entities import is_temporal_entity

logger = logging.getLogger(__name__)

# ── Shared entity extraction guidance ────────────────────────────────────

ENTITY_TYPE_VALUES = (
    "PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "PROJECT",
    "TECHNOLOGY", "CONCEPT", "TOPIC", "PREFERENCE", "OTHER",
)

ENTITY_EXTRACTION_GUIDANCE = """实体提取规则:

实体不是只限传统 NER。这里的实体指后续可以跨 facts 聚合、检索、建图的语义锚点。
优先抽取对长期记忆有复用价值的名词或短名词短语，而不是只抽取专有名词。

实体类型(type 可选值):
- PERSON(人): 对话中提到的具体人名、称呼
- ORGANIZATION(组织): 公司、团队、机构
- LOCATION(地点): 地理位置、场所
- PRODUCT(产品): 产品名、服务名
- PROJECT(项目): 项目名、产品名
- TECHNOLOGY(技术): 技术栈、框架、库、工具
- CONCEPT(概念): 抽象概念、方法论、理论
- TOPIC(主题): 讨论的话题领域
- PREFERENCE(偏好): 用户的偏好、喜好、习惯
- OTHER(其他): 明确提到但不适合上述类型的实体

应该抽取的实体包括：
- 对话主体或角色：用户、助手、妻子、孩子、团队、客户等
- 用户长期相关的领域、问题、任务、状态或场景：健康管理、身体状态、工作、商务活动、应酬、家庭教育、夫妻沟通、疲劳感等
- 可复用的方案、方法、工具、活动或对象：健康饮食、家庭会议、统一规则、野餐、瑜伽垫等
- 明确影响用户选择的约束对象或条件：经济负担、固定作息、时间不足、工作压力等

不要抽取普通时间表达作为实体，例如：今天、昨天、上周、最近三天、2026-05-07、10:30、三个月。
时间应作为事实的时间元数据处理，不进入 entity graph。
只有有语义身份的命名时间概念才可作为实体，例如：春节、Q3 财报季、Sprint 42。

不要抽取纯属性、纯形容词短语、孤立程度词或泛化标签作为实体；它们应保留在 fact text、keywords、topic 或 observation 中。
例如：低场地依赖、低强度、高优先级、低成本、强隐私、轻量级。
但如果短语中包含可复用的核心对象或场景，应抽取核心对象，例如：
- "长期高强度工作带来的疲劳感" 可抽取 "工作"、"疲劳感"
- "高频商务活动" 可抽取 "商务活动"
- "经济负担太重" 可抽取 "经济负担"

实体应来自对话中明确出现或由角色/事实主体直接确定的内容，不要过度推断。
每条长期记忆 fact 通常至少包含主体实体（如 用户/助手）和 1-4 个核心语义锚点。"""

_ATTRIBUTE_ENTITY_PATTERNS = (
    re.compile(r"^(低|高|中|中等|较低|较高|轻|重|强|弱|少|多).{0,12}(依赖|强度|成本|优先级|门槛|复杂度|风险|约束|要求|活动|方案)$"),
    re.compile(r".*(依赖|强度|成本|优先级|门槛|复杂度|风险|约束|要求|特征|属性)$"),
)


def is_attribute_entity(name: str, entity_type: str = "") -> bool:
    """Return True when an extracted entity is really an attribute phrase."""
    text = str(name or "").strip()
    if not text:
        return False
    etype = str(entity_type or "").strip().upper()
    if etype in {"PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "PROJECT", "TECHNOLOGY"}:
        return False
    compact = re.sub(r"[\s\-_/]+", "", text)
    return any(pattern.match(compact) for pattern in _ATTRIBUTE_ENTITY_PATTERNS)

# ── Prompt templates ─────────────────────────────────────────────────────

ENTITY_EXTRACTION_PROMPT = """你是实体和关系提取助手。从对话中提取实体和它们之间的关系。

""" + ENTITY_EXTRACTION_GUIDANCE + """

关系类型:
- works_on(从事): 某人从事某个项目
- uses(使用): 某人使用某项技术
- mentions(提及): 提到某个实体
- prefers(偏好): 偏好某事物
- relates_to(相关): 两个实体相关
- is_a(是): 实体属于某个类别
- has_property(具有属性): 实体具有某个属性

规则:
1. 仅提取明确提到的实体，不要过度推断
2. 每个实体有且仅有一个 type
3. 关系要有明确证据，不确定则不输出
4. 只输出 JSON，不要包含其他内容

对话内容:
用户: {user_message}
助手: {assistant_response}

输出格式(严格 JSON):
{{
  "entities": [
    {{"name": "实体名", "type": "实体类型"}}
  ],
  "relations": [
    {{"source": "实体名A", "relation": "关系类型", "target": "实体名B"}}
  ]
}}"""

# ── Entity type priorities for deduplication ─────────────────────────────
_ENTITY_TYPE_PRIORITY = [
    *ENTITY_TYPE_VALUES,
]


class EntityExtractor:
    """LLM-driven entity and relation extraction.

    Extracts entities from conversation turns and persists them to
    the knowledge graph tables in SessionDB.
    """

    def __init__(
        self,
        session_db: Any,
        llm_client: Any = None,
        llm_model: str = "gpt-4o-mini",
        llm_base_url: str = "",
        llm_api_key: str = "",
        llm_timeout: int = 120,
    ):
        self._db = session_db
        self._llm_client = llm_client
        self._llm_model = llm_model
        self._llm_base_url = llm_base_url
        self._llm_api_key = llm_api_key
        self._llm_timeout = llm_timeout

    # ── Public API ───────────────────────────────────────────────────────

    def extract_from_turn(
        self,
        node_id: int,
        user_message: str,
        assistant_response: str,
    ) -> bool:
        """Extract entities and relations from a conversation turn.

        Calls the LLM to extract entities/relations, then persists them
        to the knowledge graph tables linked to *node_id*.

        Returns True if at least one entity was extracted, False otherwise.
        All failures are logged and non-fatal.
        """
        if not self._db or not node_id:
            return False

        try:
            raw = self._call_llm_for_extraction(user_message, assistant_response)
            if not raw:
                return False

            data = self._parse_response(raw)
            if not data:
                return False

            entities = data.get("entities", [])
            relations = data.get("relations", [])

            if not entities:
                return False

            # Store entities and build lookup
            entity_id_map: Dict[str, int] = {}
            for ent in entities:
                name = ent.get("name", "").strip()
                etype = ent.get("type", "CONCEPT").upper()
                if not name:
                    continue
                if is_temporal_entity(name, etype):
                    continue
                if is_attribute_entity(name, etype):
                    continue
                eid = self._db.entity_add_entity(
                    name=name,
                    entity_type=etype if etype in _ENTITY_TYPE_PRIORITY else "CONCEPT",
                )
                entity_id_map[name] = eid
                # Link memory node to entity
                self._db.entity_link_node(node_id, eid)

            # Store relations
            for rel in relations:
                source = rel.get("source", "").strip()
                rtype = rel.get("relation", "").strip()
                target = rel.get("target", "").strip()
                if not source or not rtype or not target:
                    continue
                src_id = entity_id_map.get(source)
                tgt_id = entity_id_map.get(target)
                if src_id is not None and tgt_id is not None and src_id != tgt_id:
                    self._db.entity_add_edge(
                        source_entity_id=src_id,
                        target_entity_id=tgt_id,
                        relation_type=rtype,
                    )

            logger.debug(
                "EntityExtractor: extracted %d entities, %d relations for node %d",
                len(entities), len(relations), node_id,
            )
            return True

        except Exception as e:
            logger.debug("EntityExtractor failed for node %d: %s", node_id, e)
            return False

    # ── Internal ─────────────────────────────────────────────────────────

    def _call_llm_for_extraction(
        self, user_message: str, assistant_response: str,
    ) -> Optional[str]:
        """Call LLM for entity/relation extraction."""
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            user_message=user_message,
            assistant_response=assistant_response,
        )

        if self._llm_client is not None:
            try:
                resp = self._llm_client.chat.completions.create(
                    model=self._llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2048,
                    timeout=self._llm_timeout,
                )
                return getattr(resp.choices[0].message, "content", "") or ""
            except Exception as e:
                logger.debug("Shared LLM client failed for entity extraction: %s", e)
                return None

        # Fallback to raw HTTP
        return self._call_llm_api(prompt)

    def _call_llm_api(self, prompt: str) -> Optional[str]:
        """Raw HTTP fallback to OpenAI-compatible endpoint."""
        import requests

        base_url = self._llm_base_url or "https://api.openai.com/v1"
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._llm_api_key:
            headers["Authorization"] = f"Bearer {self._llm_api_key}"

        data = {
            "model": self._llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2048,
            "stream": False,
        }
        try:
            resp = requests.post(url, json=data, headers=headers, timeout=self._llm_timeout)
            resp.raise_for_status()
            result = resp.json()
            choices = result.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return None
        except Exception as e:
            logger.debug("HTTP entity extraction call failed: %s", e)
            return None

    @staticmethod
    def _parse_response(raw: str) -> Optional[Dict[str, Any]]:
        """Parse LLM JSON response, stripping code fences if present."""
        text = raw.strip()
        if "```" in text:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start:end + 1]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.debug("EntityExtractor: failed to parse LLM response: %.120s", raw)
            return None
        if not isinstance(data.get("entities"), list):
            return None
        return data
