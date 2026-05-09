#!/usr/bin/env python3
"""Memory Node Manager — automatic summarization, embedding, and hybrid retrieval
of conversation turns as structured memory nodes.

Lifecycle (enhanced with HindSight-inspired features):
  1. After each completed conversation turn (SYNC + ASYNC):
     - Extract HindSight-style narrative facts via LLM API
     - Extract keywords
     - Generate embedding via EmbeddingClient
     - Store each fact as a memory node in SessionDB (SQLite + FAISS)
     - Start background thread for causal + entity extraction

  2. Background (ASYNC, non-blocking):
     - Extract causal relations to similar nodes
     - Extract entities and relations -> knowledge graph
     - Store in memory_node_relations + entity_nodes/edges

  3. Before each new turn:
     - Embed the user's query
     - Search for relevant memory nodes (keyword + vector + entity graph + node relations)
     - Return formatted context for system prompt injection

Usage::

    from agent.memory_node_manager import MemoryNodeManager

    mgr = MemoryNodeManager(session_db, embedding_config=None)
    mgr.store_turn("用户问了什么", "助手回答了什么")
    context = mgr.recall("用户当前问题")
    reflection = mgr.reflect("总结用户偏好")
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from agent.entity_extractor import ENTITY_EXTRACTION_GUIDANCE
from agent.temporal_entities import is_temporal_entity

logger = logging.getLogger(__name__)

# ── Default LLM API endpoint ──────────────────────────────────────────────

DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

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

SUMMARY_SYSTEM_PROMPT = """你是一个对话摘要助手。请总结以下对话，提取关键的主题和信息。

要求：
1. 用一句话精炼概括对话的核心内容
2. 提取2-5个关键词（用逗号分隔）
3. 仅返回JSON格式，不要包含其他内容

输出格式：
{{"summary": "对话的核心内容概括", "keywords": ["关键词1", "关键词2"]}}

对话内容：
用户：{user_message}
助手：{assistant_response}"""

# ── HindSight-style retain prompt template ────────────────────────────────

RETAIN_FACT_EXTRACTION_PROMPT = """你是一个长期记忆 retain 管道。请把下面一轮对话转成 1-3 条自包含的叙事事实，用于 AI agent 的长期记忆。

要求：
1. 不要按句子碎片化；每条 fact 必须能独立说明 who/what/when/where/why
2. 尽量保留用户偏好、约束、决定、失败经验、助手建议和明确原因
3. 区分 fact_type:
   - world: 客观世界/用户/项目事实
   - experience: 助手自己的行为、建议、推荐、执行经历
4. occurred_start/occurred_end 如果对话没有明确日期，填空字符串
5. entities 遵守下方统一实体提取规则；普通时间表达应写入 occurred_start/occurred_end，不进入 entities
6. causal_relations 只描述本次输出 facts 之间明确存在的关系；source_index/target_index 使用 facts 数组的 0-based 下标
7. 只返回 JSON，不要 markdown，不要额外解释

""" + ENTITY_EXTRACTION_GUIDANCE + """

""" + CAUSAL_RELATION_GUIDANCE + """

输出格式：
{{
  "facts": [
    {{
      "text": "完整叙事事实",
      "keywords": ["关键词1", "关键词2"],
      "fact_type": "world/experience",
      "fact_kind": "preference/decision/request/recommendation/action/error/context/other",
      "occurred_start": "",
      "occurred_end": "",
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

对话内容：
用户：{user_message}
助手：{assistant_response}"""

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

# ── Reflect prompt template ───────────────────────────────────────────────

# ── Memory node context block template ───────────────────────────────────
# NOTE: recall() returns RAW text (no wrapper). Callers (run_agent.py) use
# build_memory_context_block() from agent/memory_manager.py to wrap in
# <memory-context> tags.  This avoids double-wrapping issues where the
# sanitize_context regex strips pre-wrapped content entirely.

MEMORY_NODE_HEADER = "[Memory recall — past conversation summaries relevant to the current query]"
WORLD_FACT_SECTION_HEADER = (
    "[World facts — stable facts about the user, projects, preferences, and external state]"
)
EXPERIENCE_SECTION_HEADER = (
    "[Experience memories — prior assistant actions, recommendations, decisions, and outcomes]"
)

MEMORY_CONTEXT_BLOCK = """<memory-context>
[System note: The following are relevant past conversation memories, NOT new user input. Treat as informational background data.]

{memory_text}
</memory-context>"""


def _call_llm_api(prompt: str, model: str, base_url: str, api_key: str,
                  timeout: int = 120) -> Optional[str]:
    """Call an OpenAI-compatible chat completions API with a single user message.

    Supports OpenAI, OpenRouter, DeepSeek, vLLM, and any provider that exposes
    a ``/v1/chat/completions`` endpoint.

    The entire *prompt* is sent as a ``user`` message (system instructions are
    embedded directly in the prompt text).  Returns the response text, or
    ``None`` on failure.
    """
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
        "max_tokens": 2048,
        "stream": False,
    }
    try:
        resp = requests.post(url, json=data, headers=headers, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()
        choices = result.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return None
    except requests.exceptions.RequestException as e:
        logger.debug("LLM API call failed: %s", e)
        return None


class MemoryNodeManager:
    """Manages automatic creation, storage, and retrieval of summarized memory nodes.

    Summarisation uses the project-wide OpenAI client (``llm_client``) or
    falls back to raw HTTP via ``_call_llm_api``.
    Embedding uses ``EmbeddingClient``.
    Storage/search uses ``SessionDB.memory_*`` methods.

    Usage::

        # With shared AIAgent client:
        mgr = MemoryNodeManager(session_db, embedding_config, llm_client=agent.client)

        # Standalone (raw HTTP):
        mgr = MemoryNodeManager(session_db, embedding_config)
    """

    def __init__(
        self,
        session_db: Any,  # SessionDB instance
        embedding_config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        llm_client: Any = None,  # OpenAI-compatible client (shares the project's API infra)
    ) -> None:
        self._db = session_db
        self._enabled = enabled and bool(session_db)
        self._embedding_client: Any = None  # lazy init
        self._llm_client = llm_client

        cfg = embedding_config or {}

        # LLM model config (used regardless of client or raw HTTP)
        self._llm_model = cfg.get("llm_model", cfg.get("summary_model", DEFAULT_LLM_MODEL))
        self._llm_timeout = int(cfg.get("llm_timeout", cfg.get("timeout", 120)))

        # Raw HTTP fallback config (only used when llm_client is None)
        self._llm_base_url = cfg.get("llm_base_url", cfg.get("base_url", DEFAULT_LLM_BASE_URL))
        self._llm_api_key = cfg.get("llm_api_key", cfg.get("api_key", ""))

        # Retrieval config
        self._top_k = int(cfg.get("retrieval_top_k", 8))
        self._min_turns_before_store = int(cfg.get("min_turns_before_store", 0))

        # Default recall budget: "mid"
        self._recall_budget = cfg.get("recall_budget", "mid")

        # Enable entity extraction (default: True if session_db available)
        self._enable_entity_extraction = cfg.get("enable_entity_extraction", True)

        self._turn_count = 0
        self._embedding_cfg = cfg

        # Async background thread for non-critical work (causal + entity extraction)
        self._async_thread: Optional[threading.Thread] = None

    # ── Lazy init ─────────────────────────────────────────────────────────

    def _ensure_embedding_client(self) -> bool:
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
            self._enabled = False
            return False

    # ── LLM call (shared client when available) ─────────────────────────

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM using the shared OpenAI client, falling back to raw HTTP.

        When ``self._llm_client`` is set (passed from ``AIAgent``), uses the
        project-wide OpenAI client — same connection pool, same API endpoint,
        same authentication.  Otherwise falls back to ``_call_llm_api`` (raw
        ``requests.post`` to an OpenAI-compatible ``/v1/chat/completions``).
        """
        if self._llm_client is not None:
            try:
                resp = self._llm_client.chat.completions.create(
                    model=self._llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2048,
                    timeout=self._llm_timeout,
                )
                return getattr(resp.choices[0].message, "content", "") or ""
            except Exception as e:
                logger.debug("Shared LLM client call failed: %s", e)
                return None
        return _call_llm_api(
            prompt,
            model=self._llm_model,
            base_url=self._llm_base_url,
            api_key=self._llm_api_key,
            timeout=self._llm_timeout,
        )

    # ── Summarisation ────────────────────────────────────────────────────

    def _summarize_turn(
        self, user_message: str, assistant_response: str
    ) -> Optional[Dict[str, Any]]:
        """Summarise a conversation turn via LLM API call.

        Retries once on parse failure to handle transient API errors.
        """
        prompt = SUMMARY_SYSTEM_PROMPT.format(
            user_message=user_message,
            assistant_response=assistant_response,
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
            if not summary:
                if attempt == 0:
                    continue
                return None
            return {"summary": summary, "keywords": keywords}

        return None

    @staticmethod
    def _json_object_from_llm_text(text: str) -> Optional[Dict[str, Any]]:
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
            seen.add(name)
            entities.append({"name": name, "type": etype})
        return entities

    def _fallback_fact_from_summary(
        self,
        summary_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        keywords = self._normalize_keywords(summary_data.get("keywords", []))
        return {
            "text": str(summary_data.get("summary", "")).strip(),
            "keywords": keywords,
            "fact_type": "world",
            "fact_kind": "conversation_summary",
            "occurred_start": "",
            "occurred_end": "",
            "where": "",
            "entities": [],
        }

    def _extract_retain_facts(
        self,
        user_message: str,
        assistant_response: str,
    ) -> Optional[Dict[str, Any]]:
        """Extract HindSight-style narrative facts for retain.

        The preferred path asks the LLM for structured narrative facts. If the
        model fails or returns malformed JSON, we fall back to the older single
        summary so memory retention remains best-effort instead of all-or-none.
        """
        prompt = RETAIN_FACT_EXTRACTION_PROMPT.format(
            user_message=user_message,
            assistant_response=assistant_response,
        )

        data: Optional[Dict[str, Any]] = None
        for attempt in range(2):
            result = self._call_llm(prompt)
            logger.error("output from LLM \n" + result)
            data = self._json_object_from_llm_text(result or "")
            if data is not None:
                break
            if attempt == 0:
                logger.debug("Retain fact extraction parse failed, retrying")

        facts: List[Dict[str, Any]] = []
        if data is not None and isinstance(data.get("facts"), list):
            for raw_fact in data.get("facts", []):
                if not isinstance(raw_fact, dict):
                    continue
                text = str(raw_fact.get("text") or raw_fact.get("summary") or "").strip()
                if not text:
                    continue
                entities = self._normalize_fact_entities(raw_fact.get("entities", []))
                keywords = self._normalize_keywords(raw_fact.get("keywords", []))
                if not keywords:
                    keywords = [e["name"] for e in entities[:5]]
                facts.append({
                    "text": text,
                    "keywords": keywords,
                    "fact_type": str(raw_fact.get("fact_type", "world") or "world").strip().lower(),
                    "fact_kind": str(raw_fact.get("fact_kind", "other") or "other").strip().lower(),
                    "occurred_start": str(raw_fact.get("occurred_start", "") or "").strip(),
                    "occurred_end": str(raw_fact.get("occurred_end", "") or "").strip(),
                    "where": str(raw_fact.get("where", "") or "").strip(),
                    "entities": entities,
                })

        if not facts:
            summary_data = self._summarize_turn(user_message, assistant_response)
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
                        source_index = int(item.get("source_index"))
                        target_index = int(item.get("target_index"))
                    except (TypeError, ValueError):
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
            logger.error("cur chosen similar node, summary, " + node.get("summary", ""))
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
    def _memory_time_key(fact_index: int = 0) -> str:
        """Return a lexicographically sortable, unique-ish timestamp key."""
        now = datetime.now(timezone.utc)
        base = now.strftime("%Y-%m-%d %H:%M:%S.%f")
        return f"{base}+00:00#{fact_index:02d}"

    @staticmethod
    def _original_dialog_payload(
        user_message: str,
        assistant_response: str,
        fact: Dict[str, Any],
    ) -> str:
        """Store source dialog plus structured retain metadata in one field.

        The current DB schema has no dedicated metadata column for memory
        nodes, so retain metadata is encoded alongside the source transcript in
        a JSON payload. Existing readers treat this as opaque text.
        """
        payload = {
            "source_dialog": {
                "user": user_message,
                "assistant": assistant_response,
            },
            "retain_fact": {
                "text": fact.get("text", ""),
                "fact_type": fact.get("fact_type", "world"),
                "fact_kind": fact.get("fact_kind", "other"),
                "occurred_start": fact.get("occurred_start", ""),
                "occurred_end": fact.get("occurred_end", ""),
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
            f"fact_type:{fact.get('fact_type', 'world')}",
            f"fact_kind:{fact.get('fact_kind', 'other')}",
            "source:memory_node_manager",
        ):
            if tag not in out:
                out.append(tag)
        return out

    def _link_fact_entities(self, node_id: int, entities: List[Dict[str, str]]) -> None:
        if not entities or not self._db:
            return
        for entity in entities:
            name = entity.get("name", "").strip()
            if not name:
                continue
            etype = entity.get("type", "CONCEPT").strip().upper() or "CONCEPT"
            try:
                entity_id = self._db.entity_add_entity(name=name, entity_type=etype)
                self._db.entity_link_node(node_id, entity_id)
            except Exception as exc:
                logger.debug("Failed to link retain entity %r to node %d: %s", name, node_id, exc)

    def _link_retain_relations(
        self,
        node_ids: List[int],
        relations: List[Dict[str, Any]],
    ) -> None:
        for relation in relations:
            try:
                source_id = node_ids[int(relation["source_index"])]
                target_id = node_ids[int(relation["target_index"])]
                relation_type = str(relation["relation"])
                confidence = float(relation.get("confidence", 1.0) or 1.0)
                self._db.memory_add_node_relation(
                    source_node_id=source_id,
                    target_node_id=target_id,
                    relation_type=relation_type,
                    confidence=confidence,
                )
            except Exception as exc:
                logger.debug("Failed to link retain relation %s: %s", relation, exc)

    # ── Store turn as memory node ─────────────────────────────────────────

    def store_turn(
        self,
        user_message: str,
        assistant_response: str,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Retain a turn as one or more narrative memory nodes.

        The synchronous part follows the HindSight retain shape:
        extract narrative facts → embed each fact → store nodes → link
        entities and explicit intra-retain causal relations. Additional
        cross-turn causal extraction still runs in the background.

        Returns True if at least one fact was stored, False otherwise.
        """
        if not self._enabled:
            return False
        if not user_message or not assistant_response:
            return False

        self._turn_count += 1
        if self._turn_count <= self._min_turns_before_store:
            return False

        if not self._ensure_embedding_client():
            return False

        try:
            # ── Step 1: Extract narrative facts (SYNC) ──
            retain_data = self._extract_retain_facts(user_message, assistant_response)
            if not retain_data:
                logger.debug("Skipping memory node — retain extraction returned no data")
                return False

            facts = retain_data.get("facts", [])
            stored_nodes: List[Tuple[int, str, np.ndarray, bool, List[str]]] = []
            node_ids: List[int] = []

            for idx, fact in enumerate(facts):
                summary = str(fact.get("text", "")).strip()
                if not summary:
                    continue
                keywords = self._normalize_keywords(fact.get("keywords", []))

                # ── Step 2: Generate embedding (SYNC) ──
                embedding = self._embedding_client.embed_text(summary)
                if embedding is None:
                    logger.info("Skipping memory fact — embedding generation failed")
                    continue

                # ── Step 3: Store the new node (SYNC) ──
                node_id = self._db.memory_add_node(
                    time_key=self._memory_time_key(idx),
                    summary=summary,
                    keywords=keywords,
                    original_dialog=self._original_dialog_payload(
                        user_message=user_message,
                        assistant_response=assistant_response,
                        fact=fact,
                    ),
                    query_embedding=embedding,
                    tags=self._fact_tags(fact, tags),
                    fact_type=fact.get("fact_type", "world"),
                )
                fact_entities = fact.get("entities", [])
                self._link_fact_entities(node_id, fact_entities)
                # If retain extraction already produced structured entities,
                # trust that output and avoid a second whole-turn extraction.
                run_entity_extraction = not bool(fact_entities)
                stored_nodes.append((node_id, summary, embedding, run_entity_extraction, keywords))
                node_ids.append(node_id)

            if not stored_nodes:
                return False

            # ── Step 4: Link explicit relations between newly retained facts ──
            self._link_retain_relations(node_ids, retain_data.get("causal_relations", []))

            # ── Step 5: Start async background work ──
            # (cross-turn causal relation extraction + legacy whole-turn entity extraction)
            for node_id, summary, embedding, run_entity_extraction, keywords in stored_nodes:
                self._start_async_work(
                    node_id=node_id,
                    summary=summary,
                    user_message=user_message,
                    assistant_response=assistant_response,
                    embedding=embedding,
                    keywords=keywords,
                    wait_previous=False,
                    run_entity_extraction=run_entity_extraction,
                )

            logger.debug(
                "Retained %d memory fact node(s) from turn",
                len(stored_nodes),
            )
            return True

        except Exception as e:
            logger.info("Failed to store memory node (non-fatal): %s", e)
            return False

    def _start_async_work(
        self,
        node_id: int,
        summary: str,
        user_message: str,
        assistant_response: str,
        embedding: np.ndarray,
        keywords: Optional[List[str]] = None,
        wait_previous: bool = True,
        run_entity_extraction: bool = True,
    ) -> None:
        """Start background thread for causal + entity extraction."""
        def _run_async():
            try:
                # ── A. Causal relation extraction ──
                similar_nodes, similar_ids = self._db.memory_relation_candidates(
                    node_id=node_id,
                    query_embedding=embedding,
                    keywords=keywords or [],
                    top_k=getattr(self._db, "MEMORY_TOP_K_CAUSAL", 5),
                    budget=self._recall_budget,
                )
                if similar_nodes:
                    logger.error("cur_chosen_node, summary, " + summary)
                    relations = self._extract_causal_relations(summary, similar_nodes)
                    for similar_id, relation in zip(similar_ids, relations):                        
                        if relation is not None:
                            # Write to both normalized table + legacy JSON
                            self._db.memory_update_causal(
                                node_id, similar_id,
                                relation_ab=relation,
                                relation_ba=None,
                            )
                            self._db.memory_add_node_relation(
                                source_node_id=node_id,
                                target_node_id=similar_id,
                                relation_type=relation,
                            )
                    logger.debug(
                        "Async: linked node %d to %d similar node(s)",
                        node_id, len(similar_ids),
                    )

                # ── B. Entity extraction ──
                extractor = self._ensure_entity_extractor() if run_entity_extraction else None
                if extractor is not None:
                    extractor.extract_from_turn(
                        node_id=node_id,
                        user_message=user_message,
                        assistant_response=assistant_response,
                    )

            except Exception as e:
                logger.debug("Async background work failed for node %d: %s", node_id, e)

        # Wait for previous async thread to finish, then start new one
        if wait_previous and self._async_thread and self._async_thread.is_alive():
            self._async_thread.join(timeout=5.0)
        self._async_thread = threading.Thread(
            target=_run_async, daemon=True, name="memory-node-async"
        )
        self._async_thread.start()

    # ── Recall relevant memory nodes ──────────────────────────────────────

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

        if not self._ensure_embedding_client():
            return ""

        try:
            k = top_k or self._top_k
            b = budget or self._recall_budget

            # Detect and extract time expressions from the query
            _parsed_time_start, _parsed_time_end, clean_query = self._parse_time_expression(query)
            ts = time_start or _parsed_time_start
            te = time_end or _parsed_time_end
            search_query = clean_query or query

            # Generate embedding from the clean query
            query_embedding = self._embedding_client.embed_text(search_query)
            if query_embedding is None:
                logger.debug("Query embedding is None")
                return ""

            # Generate summary for the query (for keyword extraction)
            summary_data = self._summarize_turn(search_query, "")
            if not summary_data:
                logger.debug("Skipping recall — summarisation returned no data")
                return ""

            keywords = summary_data["keywords"]

            # Hybrid search is run separately per fact type so stable world
            # facts and assistant experiences stay distinct through recall.
            world_nodes = self._db.memory_search(
                keywords, query_embedding, top_k=k, budget=b,
                time_start=ts, time_end=te,
                tags=tags,
                fact_types=["world"],
            )
            experience_nodes = self._db.memory_search(
                keywords, query_embedding, top_k=k, budget=b,
                time_start=ts, time_end=te,
                tags=tags,
                fact_types=["experience"],
            )

            if not world_nodes and not experience_nodes:
                logger.debug("No relevant memory nodes found for query")
                return ""

            # Format results as raw text (no <memory-context> wrapper)
            lines: List[str] = []
            lines.append(MEMORY_NODE_HEADER)
            lines.append("")
            if world_nodes:
                lines.append(WORLD_FACT_SECTION_HEADER)
                lines.append("System note: These are durable world facts. Use them as background state, not as a new user request.")
                for i, node in enumerate(world_nodes, 1):
                    lines.append(self._format_recall_node(i, node))
                lines.append("")
            if experience_nodes:
                lines.append(EXPERIENCE_SECTION_HEADER)
                lines.append("System note: These are prior assistant experiences. Use them to avoid repeating failed approaches and to reuse successful patterns.")
                for i, node in enumerate(experience_nodes, 1):
                    lines.append(self._format_recall_node(i, node))

            memory_text = "\n".join(lines)
            return memory_text.strip()

        except Exception as e:
            logger.debug("Memory recall failed (non-fatal): %s", e)
            return ""

    @staticmethod
    def _format_recall_node(index: int, node: Dict[str, Any]) -> str:
        node_summary = node.get("summary", "")
        time_key = node.get("time_key", "")
        kw = ", ".join(node.get("keywords", []))
        line = f"{index}. [{time_key}] {node_summary}"
        if kw:
            line += f"  (关键词: {kw})"
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
