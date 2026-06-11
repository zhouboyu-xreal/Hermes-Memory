import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pytest

import hermes_state
from agent.memory_node_manager import MemoryNodeManager
from agent.memory_node_manager import (
    CAUSAL_RELATION_TYPE_TEXT,
    OBSERVATION_SUPPORT_SECTION_HEADER,
    INTERPRETATION_GENERATION_PROMPT,
    INTERPRETATION_UPDATE_PROMPT,
    INTERPRETATION_SECTION_HEADER,
    EXPERIENCE_SECTION_HEADER,
    OBSERVATION_SECTION_HEADER,
    RELATION_PROMPT_TEMPLATE,
    RETAIN_FACT_EXTRACTION_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    WORLD_FACT_SECTION_HEADER,
)
from hermes_state import SessionDB


class _FakeEmbeddingClient:
    def embed_text(self, text):
        if not text:
            return None
        return np.ones((1, 1536), dtype=np.float32)


class _CapturingEmbeddingClient:
    def __init__(self):
        self.texts = []

    def embed_text(self, text):
        self.texts.append(text)
        if not text:
            return None
        return np.ones((1, 1536), dtype=np.float32)


class _RejectingChatCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "temperature" in kwargs:
            raise RuntimeError("Unsupported parameter: temperature")
        if "max_tokens" in kwargs:
            raise RuntimeError("Unsupported parameter: max_tokens")

        class _Message:
            pass

        class _Choice:
            pass

        class _Response:
            pass

        message = _Message()
        message.content = self.content
        choice = _Choice()
        choice.message = message
        response = _Response()
        response.choices = [choice]
        return response


class _RejectingLLMClient:
    def __init__(self, content):
        self.completions = _RejectingChatCompletions(content)

        class _Chat:
            pass

        self.chat = _Chat()
        self.chat.completions = self.completions


class _ResponsesOnlyClient:
    def __init__(self, content):
        self.calls = []
        self.content = content

        class _ChatCompletions:
            def create(_self, **_kwargs):
                raise RuntimeError("unsupported_api_for_model: use Responses API")

        class _Chat:
            pass

        class _Responses:
            pass

        self.chat = _Chat()
        self.chat.completions = _ChatCompletions()
        self.responses = _Responses()
        self.responses.create = self._create_response

    def _create_response(self, **kwargs):
        self.calls.append(kwargs)

        class _Response:
            pass

        response = _Response()
        response.output_text = self.content
        return response


class _KeywordEmbeddingClient:
    def embed_text(self, text):
        lowered = str(text or "").lower()
        vec = np.zeros((1, 3), dtype=np.float32)
        if "memory" in lowered or "reflect" in lowered:
            vec[0, 0] = 1.0
        elif "travel" in lowered:
            vec[0, 1] = 1.0
        else:
            vec[0, 2] = 1.0
        return vec


class _OrthogonalTaskEmbeddingClient:
    def embed_text(self, text):
        lowered = str(text or "").lower()
        vec = np.zeros((1, 2), dtype=np.float32)
        if lowered.startswith("task summary"):
            vec[0, 0] = 1.0
        else:
            vec[0, 1] = 1.0
        return vec


class _NoAsyncMemoryNodeManager(MemoryNodeManager):
    def __init__(self, *args, llm_outputs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._embedding_client = _FakeEmbeddingClient()
        self._llm_outputs = list(llm_outputs or [])
        self.llm_prompts = []
        self.graph_calls = []

    def _call_llm(self, prompt):
        self.llm_prompts.append(prompt)
        if self._llm_outputs:
            return self._llm_outputs.pop(0)
        return None

    def _build_relation_graph(self, **kwargs):
        self.graph_calls.append(kwargs)
        return super()._build_relation_graph(**kwargs)


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


def test_store_turn_retains_multiple_hindsight_facts(db):
    retain_payload = {
        "facts": [
            {
                "text": "Alice prefers Slack over email for urgent team communication.",
                "keywords": ["Alice", "Slack", "email"],
                "primary_entity": {"name": "Alice", "type": "PERSON"},
                "primary_topic": "urgent-team-communication",
                "fact_type": "semantic",
                "fact_subject": "user",
                "fact_kind": "preference",
                "priority": 92,
                "priority_reason": "stable user communication preference",
                "task_event_like": False,
                "task_event_subject": "user",
                "task_relevance": "none",
                "occurred_start": "2026-05-01 00:00:00",
                "occurred_end": "2026-05-01 23:59:59",
                "time_confidence": "explicit",
                "where": "work",
                "entities": [
                    {"name": "Alice", "type": "PERSON"},
                    {"name": "Slack", "type": "PRODUCT"},
                ],
            },
            {
                "text": "Hermes recommended configuring alerts to notify Alice in Slack.",
                "keywords": ["Hermes", "alerts", "Slack"],
                "primary_entity": {"name": "Hermes", "type": "AGENT"},
                "primary_topic": "alert-routing",
                "fact_type": "episodic",
                "fact_subject": "assistant",
                "fact_kind": "recommendation",
                "priority": 78,
                "priority_reason": "reusable assistant recommendation",
                "task_event_like": True,
                "task_event_subject": "assistant",
                "task_relevance": "medium",
                "time_confidence": "inferred_from_turn",
                "entities": [
                    {"name": "Alice", "type": "PERSON"},
                    {"name": "Slack", "type": "PRODUCT"},
                ],
            },
        ],
        "causal_relations": [
            {"source_index": 0, "target_index": 1, "relation": "Cause", "confidence": 0.8}
        ],
    }
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[json.dumps(retain_payload)],
    )

    assert mgr.store_turn("Alice hates email for urgent alerts", "Use Slack alerts.") is True

    rows = db._conn.execute(
        "SELECT id, summary, keywords, topic, primary_entity_id, primary_topic, "
        "tags, fact_type, fact_subject, fact_kind, entity_names, "
        "task_event_like, task_event_subject, task_relevance "
        "FROM memory_nodes ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["summary"] == retain_payload["facts"][0]["text"]
    assert rows[1]["summary"] == retain_payload["facts"][1]["text"]
    assert "Alice Slack email" == rows[0]["keywords"]
    assert "urgent-team-communication" == rows[0]["topic"]
    assert rows[0]["primary_topic"] == "urgent-team-communication"
    assert rows[0]["primary_entity_id"] is not None
    assert rows[0]["fact_type"] == "semantic"
    assert rows[0]["fact_subject"] == "user"
    assert rows[0]["fact_kind"] == "preference"
    assert rows[1]["fact_type"] == "episodic"
    assert rows[1]["fact_subject"] == "assistant"
    assert rows[1]["fact_kind"] == "recommendation"
    assert json.loads(rows[0]["entity_names"]) == ["用户", "Alice", "Slack"]
    assert json.loads(rows[1]["entity_names"]) == ["Hermes", "助手", "Alice", "Slack"]
    assert "fact_type:semantic" in json.loads(rows[0]["tags"])
    assert "fact_subject:user" in json.loads(rows[0]["tags"])
    assert "fact_type:episodic" in json.loads(rows[1]["tags"])
    assert "fact_subject:assistant" in json.loads(rows[1]["tags"])
    assert "fact_kind:preference" in json.loads(rows[0]["tags"])
    assert "priority:92" in json.loads(rows[0]["tags"])
    assert "priority_band:high" in json.loads(rows[0]["tags"])
    assert "time_confidence:explicit" in json.loads(rows[0]["tags"])
    assert "priority:78" in json.loads(rows[1]["tags"])
    assert "priority_band:medium" in json.loads(rows[1]["tags"])
    assert "time_confidence:inferred_from_turn" in json.loads(rows[1]["tags"])
    assert rows[0]["task_event_like"] == 0
    assert rows[0]["task_event_subject"] == "user"
    assert rows[0]["task_relevance"] == "none"
    assert rows[1]["task_event_like"] == 1
    assert rows[1]["task_event_subject"] == "assistant"
    assert rows[1]["task_relevance"] == "medium"

    detail = db._conn.execute(
        "SELECT original_dialog FROM memory_nodes WHERE id = ?",
        (rows[0]["id"],),
    ).fetchone()
    original_payload = json.loads(detail["original_dialog"])
    assert original_payload["retain_fact"]["fact_type"] == "semantic"
    assert original_payload["retain_fact"]["fact_subject"] == "user"
    assert original_payload["retain_fact"]["fact_kind"] == "preference"
    assert original_payload["retain_fact"]["priority"] == 92
    assert original_payload["retain_fact"]["priority_reason"] == "stable user communication preference"
    assert original_payload["retain_fact"]["task_event_like"] is False
    assert original_payload["retain_fact"]["task_relevance"] == "none"
    assert original_payload["retain_fact"]["occurred_start"] == "2026-05-01 00:00:00"
    assert original_payload["retain_fact"]["time_confidence"] == "explicit"

    alice = db._conn.execute("SELECT id, type FROM entity_nodes WHERE name = 'Alice'").fetchone()
    assert alice is not None
    assert alice["type"] == "PERSON"
    linked_nodes = db._conn.execute(
        "SELECT node_id FROM memory_node_entities WHERE entity_id = ? ORDER BY node_id",
        (alice["id"],),
    ).fetchall()
    assert [r["node_id"] for r in linked_nodes] == [rows[0]["id"], rows[1]["id"]]

    relation = db._conn.execute(
        "SELECT source_node_id, target_node_id, relation_type, confidence "
        "FROM memory_node_relations WHERE relation_type = 'Cause'"
    ).fetchone()
    assert relation["source_node_id"] == rows[0]["id"]
    assert relation["target_node_id"] == rows[1]["id"]
    assert relation["relation_type"] == "Cause"
    assert relation["confidence"] == pytest.approx(0.8)
    assert len(mgr.graph_calls) == 2
    assert mgr.graph_calls[0]["keywords"] == ["Alice", "Slack", "email"]


def test_memory_node_details_live_on_memory_nodes_table(db):
    tables = {
        row["name"]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }

    assert "memory_nodes" in tables
    assert "memory_leaf_details" not in tables
    columns = {
        row["name"]
        for row in db._conn.execute("PRAGMA table_info(memory_nodes)").fetchall()
    }
    assert "fact_type" in columns
    assert "fact_subject" in columns
    assert "fact_kind" in columns
    assert "entity_names" in columns


def test_memory_interpretations_store_current_agent_interpretations(db):
    tables = {
        row["name"]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "memory_interpretations" in tables
    columns = {
        row["name"]
        for row in db._conn.execute("PRAGMA table_info(memory_interpretations)").fetchall()
    }
    assert "entity_id" in columns
    assert "embedding" in columns
    assert "embedding_text" in columns
    assert "embedding_updated_at" in columns
    assert "subject_entity_id" not in columns
    assert "target_entity_id" not in columns

    hermes = db.entity_add_entity("Hermes Agent", "PROJECT")
    interpretation_id = db.memory_upsert_interpretation(
        claim="用户当前倾向先用 heuristic 控制 task fact 选择，再谨慎修改 prompt。",
        entity_id=hermes,
        subject_text="user",
        target_text="memory task fact selection",
        scope="memory-system-design",
        interpretation_type="task",
        confidence=0.86,
        strength=0.8,
        action_implication="后续先讨论 heuristic/data-flow，再考虑 prompt guidance。",
        evidence_node_ids=[1, "2", "bad", 2],
        evidence_observation_ids=[3],
        metadata={"source": "agent_interpretation"},
    )

    row = db._conn.execute(
        "SELECT claim, entity_id, interpretation_type, status, confidence, evidence_node_ids, metadata "
        "FROM memory_interpretations WHERE id = ?",
        (interpretation_id,),
    ).fetchone()
    assert row["entity_id"] == hermes
    assert row["interpretation_type"] == "task"
    assert row["status"] == "current"
    assert row["confidence"] == pytest.approx(0.86)
    assert json.loads(row["evidence_node_ids"]) == [1, 2]
    assert json.loads(row["metadata"]) == {"source": "agent_interpretation"}

    updated_id = db.memory_upsert_interpretation(
        claim="用户当前更偏好 deterministic heuristic 控制 task fact 选择。",
        entity_id=hermes,
        subject_text="user",
        target_text="memory task fact selection",
        scope="memory-system-design",
        interpretation_type="task",
        confidence=0.9,
    )
    assert updated_id == interpretation_id

    results = db.search_memory_interpretations(["heuristic", "task"], top_k=5)

    assert [item["id"] for item in results] == [interpretation_id]
    assert results[0]["claim"] == "用户当前更偏好 deterministic heuristic 控制 task fact 选择。"
    assert results[0]["evidence_node_ids"] == [1, 2]


def test_search_memory_interpretations_uses_embedding_similarity(db):
    matching_id = db.memory_upsert_interpretation(
        claim="Design calibration should happen before implementation.",
        target_text="implementation workflow",
        scope="workflow",
        interpretation_type="inferred_preference",
        confidence=0.75,
        action_implication="Discuss design before editing code.",
        embedding=np.array([[1.0, 0.0]], dtype=np.float32),
        embedding_text="design calibration before implementation",
    )
    db.memory_upsert_interpretation(
        claim="Calendar cleanup is unrelated.",
        target_text="calendar cleanup",
        scope="calendar",
        interpretation_type="insight",
        confidence=0.95,
        action_implication="Use calendar context.",
        embedding=np.array([[0.0, 1.0]], dtype=np.float32),
        embedding_text="calendar cleanup",
    )

    results = db.search_memory_interpretations(
        ["alignment"],
        top_k=5,
        query_embedding=np.array([[1.0, 0.0]], dtype=np.float32),
    )

    assert [item["id"] for item in results] == [matching_id]
    assert results[0]["embedding_similarity"] == pytest.approx(1.0)


def test_search_memory_interpretations_separates_content_and_entity_matches(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    content_match = db.memory_upsert_interpretation(
        claim="The alert workflow prefers Slack escalation.",
        target_text="alert routing",
        scope="alert-workflow",
        interpretation_type="insight",
        confidence=0.7,
        action_implication="Use Slack when alert routing comes up.",
    )
    entity_only = db.memory_upsert_interpretation(
        claim="User has a current collaboration preference.",
        entity_id=alice,
        target_text="collaboration",
        scope="workflow",
        interpretation_type="inferred_preference",
        confidence=0.95,
        action_implication="Consider person-specific collaboration context.",
    )

    keyword_results = db.search_memory_interpretations(["Alice", "Slack"], top_k=5)

    assert [item["id"] for item in keyword_results[:2]] == [content_match, entity_only]

    entity_results = db.search_memory_interpretations(
        "collaboration",
        entities=[{"name": "Alice"}],
        top_k=5,
    )

    assert entity_results[0]["id"] == entity_only


def test_store_turn_falls_back_to_summary_when_retain_json_is_bad(db):
    summary_payload = {
        "summary": "The user decided to use PostgreSQL 16 for the project.",
        "keywords": ["PostgreSQL", "project"],
    }
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=["not json", "still not json", json.dumps(summary_payload)],
    )

    assert mgr.store_turn("Use PostgreSQL 16", "Good choice.") is True

    row = db._conn.execute(
        "SELECT summary, keywords, topic, tags, fact_kind FROM memory_nodes"
    ).fetchone()
    assert row["summary"] == summary_payload["summary"]
    assert row["keywords"] == "PostgreSQL project"
    assert row["topic"] == "postgresql"
    assert row["fact_kind"] == "conversation_summary"
    assert "fact_kind:conversation_summary" in json.loads(row["tags"])
    assert len(mgr.graph_calls) == 1


def test_retain_and_relation_prompts_share_relation_type_contract():
    assert CAUSAL_RELATION_TYPE_TEXT in RETAIN_FACT_EXTRACTION_PROMPT
    assert CAUSAL_RELATION_TYPE_TEXT in RELATION_PROMPT_TEMPLATE
    assert "Reason/HinderedBy" not in RETAIN_FACT_EXTRACTION_PROMPT
    assert "Reason/HinderedBy" not in RELATION_PROMPT_TEMPLATE
    assert '"keywords": ["关键词1", "关键词2"]' in RETAIN_FACT_EXTRACTION_PROMPT
    assert '"primary_entity": {{"name": "主要主体实体", "type": "PERSON"}}' in RETAIN_FACT_EXTRACTION_PROMPT
    assert '"primary_topic": "唯一核心主题"' in RETAIN_FACT_EXTRACTION_PROMPT
    assert '"fact_type": "semantic/episodic"' in RETAIN_FACT_EXTRACTION_PROMPT
    assert '"fact_subject": "user/assistant/world/project/system/other"' in RETAIN_FACT_EXTRACTION_PROMPT
    assert '"fact_kind": "preference/decision/request/recommendation/action/error/context/instruction/other"' in RETAIN_FACT_EXTRACTION_PROMPT
    assert '"priority": 80' in RETAIN_FACT_EXTRACTION_PROMPT
    assert '"time_confidence": "explicit/inferred_from_turn/unknown"' in RETAIN_FACT_EXTRACTION_PROMPT
    assert '"task_event_like": true' in RETAIN_FACT_EXTRACTION_PROMPT
    assert "{dialogue_batch}" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "fact_kind 定义和判别边界" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "fact_type 判别边界" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "preference：用户长期或反复表达的喜好" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "decision：用户或项目已经明确做出的决定" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "request：用户对 AI 或系统提出的当前任务请求" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "instruction：用户要求 AI 以后长期遵守" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "recommendation：助手给出的具体建议" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "action：用户或助手已经执行" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "error：失败、报错、阻塞" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "context：长期有用的背景事实" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "冲突时选择更具体的 kind" in RETAIN_FACT_EXTRACTION_PROMPT
    assert '"帮我现在改代码" 属于 request' in RETAIN_FACT_EXTRACTION_PROMPT
    assert '"以后回答都先给结论" 属于 instruction' in RETAIN_FACT_EXTRACTION_PROMPT
    assert "助手执行了工具、测试、修改、验证" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "硬丢弃规则" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "0-8 条可追溯、自包含、可独立召回的 facts" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "fact：原始证据层" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "不要把信息丰富的内容压缩成空泛的主题摘要" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "信息密集时宁可输出多条完整事实" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "禁止只写“讨论了某主题”" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "表达原则" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "不要求固定句式" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "输出前进行信息保真自检" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "MemoryNodeManager.store_turn" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "不能写成\"助手已经增加测试并验证通过\"" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "可能影响任务状态或步骤的事件" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "不要求已经知道具体属于哪个任务" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "实体不是只限传统 NER" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "健康管理" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "商务活动" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "经济负担" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "每条长期记忆 fact 通常至少包含主体实体" in RETAIN_FACT_EXTRACTION_PROMPT


def test_summary_prompt_preserves_recall_worthy_details():
    assert "可追溯、自包含、可独立召回的证据性摘要" in SUMMARY_SYSTEM_PROMPT
    assert "不要把信息丰富的内容压缩成空泛的主题摘要" in SUMMARY_SYSTEM_PROMPT
    assert "summary 输出一条完整摘要，可以使用复句" in SUMMARY_SYSTEM_PROMPT
    assert "项目、模块、文件、函数、配置项、参数" in SUMMARY_SYSTEM_PROMPT
    assert "禁止只写“讨论了某主题”" in SUMMARY_SYSTEM_PROMPT
    assert "提取 2-8 个关键词" in SUMMARY_SYSTEM_PROMPT
    assert "MemoryNodeManager.store_turn" in SUMMARY_SYSTEM_PROMPT
    assert "对话没有表明测试已经执行" in SUMMARY_SYSTEM_PROMPT
    assert "输出前进行信息保真自检" in SUMMARY_SYSTEM_PROMPT
    assert "{dialogue_batch}" in SUMMARY_SYSTEM_PROMPT
    assert "{user_message}" not in SUMMARY_SYSTEM_PROMPT
    assert "{assistant_response}" not in SUMMARY_SYSTEM_PROMPT


def test_retain_fact_prompt_includes_turn_timestamp_context(db):
    retain_payload = {
        "facts": [
            {
                "text": "助手曾在当前对话时间围绕测试执行验证，结果是测试通过。",
                "keywords": ["测试", "验证"],
                "topic": ["测试"],
                "fact_type": "episodic",
                "fact_kind": "action",
                "priority": 70,
                "time_confidence": "inferred_from_turn",
                "entities": [],
            }
        ],
        "causal_relations": [],
    }
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[json.dumps(retain_payload)],
    )

    assert mgr.store_turn("跑测试", "测试通过。") is True
    assert "对话发生时间：" in mgr.llm_prompts[0]
    assert "用户：跑测试" in mgr.llm_prompts[0]
    assert "助手：测试通过。" in mgr.llm_prompts[0]


def test_store_turn_uses_explicit_turn_timestamp_for_prompt_and_node_time(db):
    retain_payload = {
        "facts": [
            {
                "text": "助手在指定样本时间完成测试验证。",
                "keywords": ["测试", "验证"],
                "topic": ["测试"],
                "fact_type": "episodic",
                "fact_kind": "action",
                "priority": 70,
                "time_confidence": "inferred_from_turn",
                "entities": [],
            }
        ],
        "causal_relations": [],
    }
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[json.dumps(retain_payload)],
    )
    turn_timestamp = datetime(2026, 5, 19, 12, 30, 0, tzinfo=timezone.utc)

    assert mgr.store_turn("跑测试", "测试通过。", turn_timestamp=turn_timestamp) is True
    assert "对话发生时间：2026-05-19T12:30:00+00:00" in mgr.llm_prompts[0]
    row = db._conn.execute("SELECT time_key FROM memory_nodes").fetchone()
    assert row["time_key"].startswith("2026-05-19 12:30:00.000000+00:00#")


def test_store_turn_adds_subject_entity_names_when_llm_omits_entities(db):
    retain_payload = {
        "facts": [
            {
                "text": "用户明确偏好先给结论。",
                "keywords": ["结论", "偏好"],
                "topic": ["回答方式"],
                "fact_type": "semantic",
                "fact_subject": "user",
                "fact_kind": "preference",
                "priority": 80,
                "entities": [],
            },
            {
                "text": "助手建议先总结再展开。",
                "keywords": ["总结", "建议"],
                "topic": ["回答方式"],
                "fact_type": "episodic",
                "fact_subject": "assistant",
                "fact_kind": "recommendation",
                "priority": 65,
                "entities": [],
            },
        ],
        "causal_relations": [],
    }
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[json.dumps(retain_payload)],
    )

    assert mgr.store_turn("以后先给结论", "可以，我会先总结。") is True
    rows = db._conn.execute(
        "SELECT entity_names, original_dialog FROM memory_nodes ORDER BY id"
    ).fetchall()
    assert json.loads(rows[0]["entity_names"]) == ["用户"]
    assert json.loads(rows[1]["entity_names"]) == ["助手"]
    assert json.loads(rows[0]["original_dialog"])["retain_fact"]["entities"][0]["name"] == "用户"
    assert json.loads(rows[1]["original_dialog"])["retain_fact"]["entities"][0]["name"] == "助手"


def test_store_turn_discards_low_priority_retain_facts(db):
    retain_payload = {
        "facts": [
            {
                "text": "The user made a one-off small talk comment with no future utility.",
                "keywords": ["small", "talk"],
                "topic": ["small talk"],
                "fact_type": "semantic",
                "fact_kind": "context",
                "priority": 40,
                "priority_reason": "one-off low value comment",
                "time_confidence": "inferred_from_turn",
                "entities": [],
            }
        ],
        "causal_relations": [],
    }
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[json.dumps(retain_payload)],
    )

    assert mgr.store_turn("随便聊一句", "好的。") is False
    assert db._conn.execute("SELECT COUNT(*) FROM memory_nodes").fetchone()[0] == 0


def test_store_turn_extracts_facts_every_configured_turn_batch(db):
    retain_payload = {
        "facts": [
            {
                "text": "用户连续推进了批量记忆提取方案。",
                "keywords": ["批量提取", "记忆"],
                "topic": ["记忆系统"],
                "fact_type": "episodic",
                "fact_subject": "user",
                "fact_kind": "action",
                "priority": 70,
                "entities": [],
            }
        ],
        "causal_relations": [],
    }
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        memory_config={"min_turns_before_store": 3},
        llm_outputs=[json.dumps(retain_payload)],
    )

    assert mgr.store_turn("第一轮需求", "第一轮回答") is False
    assert mgr.store_turn("第二轮补充", "第二轮回答") is False
    assert mgr.llm_prompts == []

    assert mgr.store_turn("第三轮确认", "第三轮回答") is True
    assert len(mgr.llm_prompts) == 1
    assert "[Turn 1]" in mgr.llm_prompts[0]
    assert "第一轮需求" in mgr.llm_prompts[0]
    assert "[Turn 2]" in mgr.llm_prompts[0]
    assert "第二轮补充" in mgr.llm_prompts[0]
    assert "[Turn 3]" in mgr.llm_prompts[0]
    assert "第三轮确认" in mgr.llm_prompts[0]
    assert mgr._pending_store_turns == []

    original_dialog = json.loads(
        db._conn.execute(
            "SELECT original_dialog FROM memory_nodes"
        ).fetchone()["original_dialog"]
    )
    assert len(original_dialog["source_dialog"]["turns"]) == 3


def test_store_turn_extracts_facts_when_pending_characters_exceed_limit(db):
    retain_payload = {
        "facts": [
            {
                "text": "用户提供了一段超过字符阈值的长文本。",
                "keywords": ["长文本", "字符阈值"],
                "topic": ["记忆提取"],
                "fact_type": "episodic",
                "fact_subject": "user",
                "fact_kind": "action",
                "priority": 70,
                "entities": [],
            }
        ],
        "causal_relations": [],
    }
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        memory_config={
            "min_turns_before_store": 5,
            "max_chars_before_store": 2000,
        },
        llm_outputs=[json.dumps(retain_payload)],
    )

    assert mgr.store_turn("用" * 1000, "答" * 998) is False
    assert len(mgr._pending_store_turns) == 1
    assert mgr.llm_prompts == []

    assert mgr.store_turn("补", "充") is True
    assert len(mgr.llm_prompts) == 1
    assert mgr._pending_store_turns == []


def test_store_turn_keeps_pending_batch_when_extraction_fails(db):
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        memory_config={"min_turns_before_store": 2},
        llm_outputs=[],
    )

    assert mgr.store_turn("第一轮", "回答一") is False
    assert mgr.store_turn("第二轮", "回答二") is False
    assert len(mgr._pending_store_turns) == 2


def test_store_turn_async_queues_and_processes_turns_in_order(db):
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
    )
    calls = []
    first_started = threading.Event()
    release_first = threading.Event()

    def fake_store_turn(user_message, assistant_response, tags=None, turn_timestamp=None):
        if not calls:
            first_started.set()
            release_first.wait(timeout=2.0)
        calls.append((user_message, assistant_response, list(tags or []), turn_timestamp))
        return True

    mgr.store_turn = fake_store_turn

    assert mgr.store_turn_async("第一轮", "回答一", tags=["one"]) is True
    assert first_started.wait(timeout=1.0)
    assert mgr.store_turn_async("第二轮", "回答二", tags=["two"]) is True
    release_first.set()

    assert mgr.flush_store_queue(timeout=2.0) is True
    assert calls == [
        ("第一轮", "回答一", ["one"], None),
        ("第二轮", "回答二", ["two"], None),
    ]
    assert mgr.shutdown_store_worker(timeout=1.0) is True


def test_store_turn_async_uses_enqueued_llm_config_snapshot(db):
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
    )
    seen_configs = []

    def fake_store_turn(user_message, assistant_response, tags=None, turn_timestamp=None):
        seen_configs.append(dict(mgr._llm_thread_context.config))
        return True

    client = object()
    mgr.store_turn = fake_store_turn

    assert mgr.store_turn_async(
        "用户消息",
        "助手回复",
        llm_client=client,
        llm_model="turn-model",
        llm_base_url="https://turn.example/v1",
        llm_api_key="turn-key",
    ) is True
    mgr.configure_llm(
        llm_model="later-model",
        llm_base_url="https://later.example/v1",
        llm_api_key="later-key",
    )

    assert mgr.flush_store_queue(timeout=2.0) is True
    assert seen_configs == [{
        "llm_client": client,
        "llm_model": "turn-model",
        "llm_base_url": "https://turn.example/v1",
        "llm_api_key": "turn-key",
    }]
    assert mgr.shutdown_store_worker(timeout=1.0) is True


def test_store_turn_async_drops_when_bounded_queue_is_full(db):
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        memory_config={"store_queue_maxsize": 1},
    )
    first_started = threading.Event()
    release_first = threading.Event()

    def fake_store_turn(user_message, assistant_response, tags=None, turn_timestamp=None):
        first_started.set()
        release_first.wait(timeout=2.0)
        return True

    mgr.store_turn = fake_store_turn

    assert mgr.store_turn_async("第一轮", "回答一") is True
    assert first_started.wait(timeout=1.0)
    assert mgr.store_turn_async("第二轮", "回答二") is True
    assert mgr.store_turn_async("第三轮", "回答三") is False

    release_first.set()
    assert mgr.flush_store_queue(timeout=2.0) is True
    assert mgr.shutdown_store_worker(timeout=1.0) is True


def test_reflect_if_due_async_waits_for_interval(db):
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        memory_config={"reflect_interval_seconds": 3600},
    )
    mgr._last_successful_reflect_at = time.time()

    assert mgr.reflect_if_due_async() is False
    assert mgr._store_queue.unfinished_tasks == 0


def test_reflect_if_due_async_runs_after_queued_store(db):
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        memory_config={"reflect_interval_seconds": 3600},
    )
    events = []
    release_store = threading.Event()
    store_started = threading.Event()

    def fake_store_turn(*_args, **_kwargs):
        events.append("store-start")
        store_started.set()
        release_store.wait(timeout=2.0)
        events.append("store-finish")
        return True

    def fake_reflect(*_args, **_kwargs):
        events.append("reflect")
        return {"merged": 0}

    mgr.store_turn = fake_store_turn
    mgr.reflect = fake_reflect
    db.get_unobserved_nodes_for_observation = lambda **_kwargs: [{"node_id": 1}]
    mgr._last_successful_reflect_at = time.time() - 3601

    assert mgr.store_turn_async("第一轮", "回答一") is True
    assert store_started.wait(timeout=1.0)
    assert mgr.reflect_if_due_async() is True
    assert mgr.reflect_if_due_async() is False
    release_store.set()
    assert mgr.flush_store_queue(timeout=2.0) is True

    assert events == ["store-start", "store-finish", "reflect"]
    assert mgr._reflect_queued_or_running is False
    assert float(db.get_meta("memory_node_last_successful_reflect_at")) > 0


def test_reflect_if_due_async_skips_full_reflect_without_new_facts(db):
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        memory_config={"reflect_interval_seconds": 1},
    )
    reflect_calls = []
    mgr.reflect = lambda **kwargs: reflect_calls.append(kwargs) or {"merged": 0}
    db.get_unobserved_nodes_for_observation = lambda **_kwargs: []
    mgr._last_successful_reflect_at = time.time() - 2

    assert mgr.reflect_if_due_async() is True
    assert mgr.flush_store_queue(timeout=2.0) is True

    assert reflect_calls == []
    assert mgr._reflect_queued_or_running is False


def test_store_turn_filters_plain_time_expressions_from_fact_entities(db):
    retain_payload = {
        "facts": [
            {
                "text": "Alice wants Slack alerts during the Spring Festival launch window.",
                "keywords": ["Alice", "Slack", "春节"],
                "topic": ["launch", "alerts"],
                "fact_type": "semantic",
                "fact_kind": "preference",
                "occurred_start": "2026-05-07 00:00:00",
                "entities": [
                    {"name": "Alice", "type": "PERSON"},
                    {"name": "今天", "type": "TIME"},
                    {"name": "2026-05-07", "type": "CONCEPT"},
                    {"name": "最近三天", "type": "OTHER"},
                    {"name": "春节", "type": "CONCEPT"},
                ],
            }
        ],
        "causal_relations": [],
    }
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[json.dumps(retain_payload)],
    )

    assert mgr.store_turn("今天 Alice 提到春节发布窗口", "Use Slack alerts.") is True

    names = {
        row["name"]
        for row in db._conn.execute("SELECT name FROM entity_nodes").fetchall()
    }
    assert "Alice" in names
    assert "春节" in names
    assert "今天" not in names
    assert "2026-05-07" not in names
    assert "最近三天" not in names


def test_store_turn_filters_attribute_phrases_from_fact_entities(db):
    retain_payload = {
        "facts": [
            {
                "text": "Alice prefers low-intensity outdoor activities with low venue dependency.",
                "keywords": ["Alice", "户外活动", "低场地依赖"],
                "topic": ["户外活动"],
                "fact_type": "semantic",
                "entities": [
                    {"name": "Alice", "type": "PERSON"},
                    {"name": "低场地依赖", "type": "CONCEPT"},
                    {"name": "低强度户外活动", "type": "TOPIC"},
                ],
            }
        ],
        "causal_relations": [],
    }
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[json.dumps(retain_payload)],
    )

    assert mgr.store_turn("Alice wants low-intensity outdoor activities.", "ok") is True

    names = {
        row["name"]
        for row in db._conn.execute("SELECT name FROM entity_nodes").fetchall()
    }
    assert "Alice" in names
    assert "低场地依赖" not in names
    assert "低强度户外活动" not in names


def test_memory_time_key_uses_local_timezone_offset():
    key = MemoryNodeManager._memory_time_key(0)
    local_offset = __import__("datetime").datetime.now().astimezone().strftime("%z")
    local_offset = f"{local_offset[:3]}:{local_offset[3:]}"

    assert key.endswith("#00")
    assert local_offset in key


def _add_memory_node(
    db,
    *,
    time_key,
    summary,
    keywords,
    fact_type="semantic",
    fact_kind="other",
    task_event_like=None,
    task_event_subject="",
    task_relevance="",
):
    return db.memory_add_node(
        time_key=time_key,
        summary=summary,
        keywords=keywords,
        topic=keywords,
        original_dialog="{}",
        query_embedding=np.ones((1, 1536), dtype=np.float32),
        fact_type=fact_type,
        fact_kind=fact_kind,
        task_event_like=task_event_like,
        task_event_subject=task_event_subject,
        task_relevance=task_relevance,
    )


def test_memory_node_embeddings_reconstruct_requested_vectors(db, monkeypatch):
    class _FakeFaissIndex:
        ntotal = 2
        d = 3

        @staticmethod
        def reconstruct(index):
            return np.array([float(index), 1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(hermes_state, "_HAS_FAISS", True)
    db._memory_faiss_index = _FakeFaissIndex()
    db._memory_faiss_id_map = [11, 22]

    vectors = db.memory_node_embeddings([22, 99999])

    assert set(vectors) == {22}
    assert np.allclose(vectors[22], np.array([1.0, 1.0, 0.0], dtype=np.float32))


def _add_task_interpretation(
    db,
    *,
    entity_id,
    topic_key,
    topic_label=None,
    summary,
    keywords=None,
    source_node_ids=None,
    metadata=None,
):
    source_node_ids = source_node_ids or []
    topic_label = topic_label or topic_key
    observation_id = None
    if source_node_ids:
        observation_id = db.memory_upsert_evidence_bundle(
            entity_id=entity_id,
            topic_key=topic_key,
            topic_label=topic_label,
            bundle_type="observation",
            summary=summary,
            keywords=keywords or [topic_label],
            source_node_ids=source_node_ids,
            metadata={"observation_kind": "timeline"},
        )
    task_metadata = {
        "task_status": "active",
        "task_source": "inferred_from_interpretation",
        "entity_id": entity_id,
        "topic_key": topic_key,
        "topic_label": topic_label,
        **(metadata or {}),
    }
    if observation_id is not None:
        task_metadata["observation_id"] = observation_id
    interpretation_id = db.memory_upsert_interpretation(
        claim=summary,
        entity_id=entity_id,
        target_text=topic_label,
        scope=topic_key,
        interpretation_type="task",
        metadata=task_metadata,
        evidence_node_ids=source_node_ids,
        evidence_observation_ids=[observation_id] if observation_id is not None else [],
    )
    return interpretation_id, observation_id


def test_memory_search_uses_temporal_channel_when_semantic_and_keyword_are_empty(db, monkeypatch):
    older = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice discussed Slack alerts.",
        keywords=["Slack"],
    )
    newer = _add_memory_node(
        db,
        time_key="2026-05-02 10:00:00",
        summary="Bob discussed email digests.",
        keywords=["email"],
    )
    _add_memory_node(
        db,
        time_key="2026-06-01 10:00:00",
        summary="Outside the requested range.",
        keywords=["outside"],
    )
    monkeypatch.setattr(db, "_search_memory_vector", lambda *args, **kwargs: {})
    monkeypatch.setattr(db, "_search_memory_keyword", lambda *args, **kwargs: {})

    nodes = db.search_memory_facts(
        "no-match",
        np.ones((1, 1536), dtype=np.float32),
        top_k=5,
        time_start="2026-05-01 00:00:00",
        time_end="2026-05-31 23:59:59",
    )

    assert [n["id"] for n in nodes] == [newer, older]


def test_memory_search_facts_excludes_graph_neighbors_by_default(db, monkeypatch):
    slack = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Slack", "alerts"],
    )
    calendar = _add_memory_node(
        db,
        time_key="2026-05-02 10:00:00",
        summary="Alice wants calendar reminders in the morning.",
        keywords=["calendar"],
    )
    alice_id = db.entity_add_entity("Alice", "PERSON")
    db.entity_link_node(slack, alice_id)
    db.entity_link_node(calendar, alice_id)
    db.memory_add_node_relation(slack, calendar, "semantic", confidence=0.9)
    monkeypatch.setattr(db, "_search_memory_vector", lambda *args, **kwargs: {})

    nodes = db.search_memory_facts(
        ["Slack"],
        np.ones((1, 1536), dtype=np.float32),
        top_k=3,
        budget="mid",
    )

    assert [n["id"] for n in nodes] == [slack]
    assert "embedding_similarity" in nodes[0]
    assert nodes[0]["keyword_score"] is not None


def test_memory_search_rrf_includes_graph_neighbors_when_enabled(db, monkeypatch):
    slack = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Slack", "alerts"],
    )
    calendar = _add_memory_node(
        db,
        time_key="2026-05-02 10:00:00",
        summary="Alice wants calendar reminders in the morning.",
        keywords=["calendar"],
    )
    unrelated = _add_memory_node(
        db,
        time_key="2026-05-03 10:00:00",
        summary="Charlie prefers email newsletters.",
        keywords=["email"],
    )
    alice_id = db.entity_add_entity("Alice", "PERSON")
    db.entity_link_node(slack, alice_id)
    db.entity_link_node(calendar, alice_id)
    db.memory_add_node_relation(slack, calendar, "semantic", confidence=0.9)
    charlie_id = db.entity_add_entity("Charlie", "PERSON")
    db.entity_link_node(unrelated, charlie_id)
    monkeypatch.setattr(db, "_search_memory_vector", lambda *args, **kwargs: {})

    nodes = db.search_memory_facts(
        ["Slack"],
        np.ones((1, 1536), dtype=np.float32),
        top_k=3,
        budget="mid",
        include_graph=True,
    )

    ids = [n["id"] for n in nodes]
    assert ids[:2] == [slack, calendar]
    assert unrelated not in ids


def test_memory_search_filters_by_fact_type(db, monkeypatch):
    semantic = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Alice", "Slack"],
        fact_type="semantic",
    )
    episodic = _add_memory_node(
        db,
        time_key="2026-05-01 11:00:00",
        summary="Hermes recommended Slack alert routing for Alice.",
        keywords=["Alice", "Slack"],
        fact_type="episodic",
    )
    monkeypatch.setattr(db, "_search_memory_vector", lambda *args, **kwargs: {})

    semantic_nodes = db.search_memory_facts(
        ["Alice", "Slack"],
        np.ones((1, 1536), dtype=np.float32),
        top_k=5,
        fact_types=["semantic"],
    )
    episodic_nodes = db.search_memory_facts(
        ["Alice", "Slack"],
        np.ones((1, 1536), dtype=np.float32),
        top_k=5,
        fact_types=["episodic"],
    )

    assert [n["id"] for n in semantic_nodes] == [semantic]
    assert [n["id"] for n in episodic_nodes] == [episodic]
    assert semantic_nodes[0]["fact_type"] == "semantic"
    assert episodic_nodes[0]["fact_type"] == "episodic"


def test_memory_keyword_search_keeps_cjk_terms_intact(db):
    exact = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="用户喜欢简洁回答。",
        keywords=["简洁回答"],
    )
    _add_memory_node(
        db,
        time_key="2026-05-01 11:00:00",
        summary="助手需要回答详细问题。",
        keywords=["回答"],
    )
    _add_memory_node(
        db,
        time_key="2026-05-01 12:00:00",
        summary="用户喜欢简洁的摘要。",
        keywords=["简洁"],
    )

    results = db._search_memory_keyword("简洁回答", limit=10)

    assert list(results) == [exact]


def test_memory_keyword_search_cjk_falls_back_to_all_characters(db):
    expected = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="用户喜欢先给结论再给依据。",
        keywords=["结论", "依据"],
    )
    _add_memory_node(
        db,
        time_key="2026-05-01 11:00:00",
        summary="用户喜欢简洁回答。",
        keywords=["简洁回答"],
    )
    _add_memory_node(
        db,
        time_key="2026-05-01 12:00:00",
        summary="助手需要回答详细问题。",
        keywords=["回答"],
    )

    results = db._search_memory_keyword("喜欢结论", limit=10)

    assert list(results) == [expected]


def test_memory_keyword_search_applies_time_range_before_ranking(db):
    in_range = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Alice", "Slack"],
    )
    _add_memory_node(
        db,
        time_key="2026-04-01 10:00:00",
        summary="Alice also mentioned Slack before the requested window.",
        keywords=["Alice", "Slack"],
    )
    _add_memory_node(
        db,
        time_key="2026-06-01 10:00:00",
        summary="Alice also mentioned Slack after the requested window.",
        keywords=["Alice", "Slack"],
    )

    results = db._search_memory_keyword(
        "Slack",
        limit=10,
        time_start="2026-05-01 00:00:00",
        time_end="2026-05-31 23:59:59",
    )

    assert list(results) == [in_range]


def test_memory_search_pushes_time_candidate_ids_to_vector_channel(db, monkeypatch):
    in_range = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Alice", "Slack"],
    )
    out_of_range = _add_memory_node(
        db,
        time_key="2026-04-01 10:00:00",
        summary="Alice mentioned Slack before the requested window.",
        keywords=["Alice", "Slack"],
    )
    seen = {}

    def fake_vector_search(*args, **kwargs):
        seen["allowed_ids"] = set(kwargs.get("allowed_ids") or [])
        return {out_of_range: 1.0, in_range: 0.9}

    monkeypatch.setattr(db, "_search_memory_vector", fake_vector_search)

    nodes = db.search_memory_facts(
        "Slack",
        np.ones((1, 1536), dtype=np.float32),
        top_k=5,
        time_start="2026-05-01 00:00:00",
        time_end="2026-05-31 23:59:59",
    )

    assert seen["allowed_ids"] == {in_range}
    assert [node["id"] for node in nodes] == [in_range]


def test_memory_search_reranks_final_candidates_by_decay_score(db, monkeypatch):
    stale = _add_memory_node(
        db,
        time_key="2025-01-01 10:00:00",
        summary="Alice once preferred email alerts.",
        keywords=["Alice", "alerts"],
    )
    fresh = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice now prefers Slack alerts.",
        keywords=["Alice", "alerts"],
    )
    db._conn.execute(
        "UPDATE memory_nodes SET decay_score = CASE id WHEN ? THEN 0.0 WHEN ? THEN 1.0 ELSE decay_score END",
        (stale, fresh),
    )
    monkeypatch.setattr(db, "_search_memory_vector", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        db,
        "_search_memory_keyword",
        lambda *args, **kwargs: {stale: 0.01, fresh: 0.02},
    )

    nodes = db.search_memory_facts(
        "Alice alerts",
        np.ones((1, 1536), dtype=np.float32),
        top_k=2,
    )

    assert [node["id"] for node in nodes] == [fresh, stale]
    assert nodes[0]["decay_score"] == 1.0


def test_summarize_turn_returns_entities(db):
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "summary": "Alice wants Slack alerts.",
                "keywords": ["Slack", "alerts"],
                "entities": [
                    {"name": "Alice", "type": "PERSON"},
                    {"name": "今天", "type": "TIME"},
                ],
            })
        ],
    )

    summary = mgr._summarize_turn([
        {
            "user_message": "Alice wants Slack alerts",
            "assistant_response": "",
            "turn_timestamp": "2026-05-01T09:00:00+00:00",
        }
    ])

    assert summary["keywords"] == ["Slack", "alerts"]
    assert summary["entities"] == [{"name": "Alice", "type": "PERSON"}]
    assert "[Turn 1]" in mgr.llm_prompts[0]
    assert "对话发生时间：2026-05-01T09:00:00+00:00" in mgr.llm_prompts[0]
    assert "用户：Alice wants Slack alerts" in mgr.llm_prompts[0]
    assert "助手：" in mgr.llm_prompts[0]


def test_retain_fallback_summary_receives_paired_source_turns(db):
    summary_payload = {
        "summary": "用户先提出批量提取需求，随后补充必须保留各轮问答对应关系。",
        "keywords": ["批量提取", "轮次对应"],
        "entities": [],
    }
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        memory_config={"min_turns_before_store": 2},
        llm_outputs=["not json", "still not json", json.dumps(summary_payload)],
    )

    assert mgr.store_turn(
        "第一轮需求",
        "第一轮回答",
        turn_timestamp="2026-05-01T09:00:00+00:00",
    ) is False
    assert mgr.store_turn(
        "第二轮补充",
        "第二轮回答",
        turn_timestamp="2026-05-01T09:05:00+00:00",
    ) is True

    summary_prompt = mgr.llm_prompts[2]
    assert (
        "[Turn 1]\n"
        "对话发生时间：2026-05-01T09:00:00+00:00\n"
        "用户：第一轮需求\n"
        "助手：第一轮回答"
    ) in summary_prompt
    assert (
        "[Turn 2]\n"
        "对话发生时间：2026-05-01T09:05:00+00:00\n"
        "用户：第二轮补充\n"
        "助手：第二轮回答"
    ) in summary_prompt


def test_memory_node_manager_llm_config_comes_from_agent_not_embedding_config(db):
    mgr = MemoryNodeManager(
        db,
        embedding_config={
            "llm_model": "stale-memory-model",
            "summary_model": "stale-summary-model",
            "llm_base_url": "https://stale-memory.example/v1",
            "base_url": "https://stale-base.example/v1",
            "llm_api_key": "stale-memory-key",
            "api_key": "stale-api-key",
        },
        llm_model="main-agent-model",
        llm_base_url="https://main-agent.example/v1",
        llm_api_key="main-agent-key",
    )

    assert mgr._llm_model == "main-agent-model"
    assert mgr._llm_base_url == "https://main-agent.example/v1"
    assert mgr._llm_api_key == "main-agent-key"

    mgr.configure_llm(
        llm_model="fallback-agent-model",
        llm_base_url="https://fallback-agent.example/v1",
        llm_api_key="fallback-agent-key",
    )

    assert mgr._llm_model == "fallback-agent-model"
    assert mgr._llm_base_url == "https://fallback-agent.example/v1"
    assert mgr._llm_api_key == "fallback-agent-key"


def test_memory_node_manager_separates_embedding_and_memory_config(db):
    embedding_config = {
        "provider": "ollama",
        "model": "qwen3-embedding:8b",
        "timeout": 15,
        "retrieval_top_k": 99,
        "min_turns_before_store": 99,
        "llm_timeout": 99,
    }
    memory_config = {
        "retrieval_top_k": 12,
        "min_turns_before_store": 3,
        "max_chars_before_store": 2400,
        "reflect_interval_seconds": 1800,
        "recall_budget": "high",
        "enable_entity_extraction": False,
        "llm_timeout": 45,
    }

    mgr = MemoryNodeManager(
        db,
        embedding_config=embedding_config,
        memory_config=memory_config,
    )

    assert mgr._embedding_cfg == embedding_config
    assert mgr._memory_cfg == memory_config
    assert mgr._top_k == 12
    assert mgr._min_turns_before_store == 3
    assert mgr._max_chars_before_store == 2400
    assert mgr._reflect_interval_seconds == 1800
    assert mgr._recall_budget == "high"
    assert mgr._enable_entity_extraction is False
    assert mgr._llm_timeout == 45


def test_analyze_recall_query_accepts_legacy_summary_shape(db):
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "summary": "Alice Slack alerts",
                "keywords": ["Alice", "Slack"],
                "entities": [{"name": "Alice", "type": "PERSON"}],
            })
        ],
    )

    analysis = mgr._analyze_recall_query("Alice Slack alerts")

    assert analysis["search_text"] == "Alice Slack alerts"
    assert analysis["keywords"] == ["Alice", "Slack"]
    assert analysis["entities"] == [{"name": "Alice", "type": "PERSON"}]
    assert analysis["needs_recall"] is True
    assert analysis["recall_intent"] == "balanced"
    assert analysis["fact_type_preference"] == "both"


def test_recall_gate_skips_trivial_query_without_llm_or_embedding(db):
    capture = _CapturingEmbeddingClient()
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})
    mgr._embedding_client = capture

    assert mgr.recall("谢谢！") == ""
    assert mgr.llm_prompts == []
    assert capture.texts == []


def test_recall_gate_explicit_history_reference_still_uses_llm_analysis(db, monkeypatch):
    _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Alice", "Slack"],
        fact_type="semantic",
    )
    monkeypatch.setattr(db, "_search_memory_vector", lambda *args, **kwargs: {})
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "needs_recall": True,
                "recall_confidence": 0.96,
                "recall_reason": "explicit_history_reference",
                "search_text": "Alice Slack urgent alert preference",
                "keywords": ["Alice", "Slack", "alerts"],
                "recall_intent": "evidence",
            })
        ],
    )

    context = mgr.recall("你还记得我之前说过的 Slack 告警偏好吗？")

    assert "Alice prefers Slack for urgent alerts." in context
    assert len(mgr.llm_prompts) == 1
    assert "你还记得我之前说过的 Slack 告警偏好吗" in mgr.llm_prompts[0]


def test_recall_gate_respects_llm_skip_regardless_of_confidence(db):
    capture = _CapturingEmbeddingClient()
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "needs_recall": False,
                "recall_confidence": 0.2,
                "recall_reason": "self_contained_general_knowledge",
                "search_text": "Python list comprehension syntax",
                "keywords": ["Python", "list comprehension"],
            })
        ],
    )
    mgr._embedding_client = capture

    assert mgr.recall("Python 列表推导式的语法是什么？") == ""
    assert len(mgr.llm_prompts) == 1
    assert capture.texts == []


def test_recall_gate_analysis_failure_stops_recall(db, monkeypatch):
    _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Alice", "Slack"],
        fact_type="semantic",
    )
    monkeypatch.setattr(db, "_search_memory_vector", lambda *args, **kwargs: {})
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})

    capture = _CapturingEmbeddingClient()
    mgr._embedding_client = capture

    assert mgr.recall("Alice Slack alerts") == ""
    assert len(mgr.llm_prompts) == 2
    assert capture.texts == []


def test_analyze_recall_query_retries_when_model_rejects_chat_params(db):
    payload = json.dumps({
        "search_text": "Alice Slack alert preference",
        "keywords": ["Alice", "Slack"],
        "recall_intent": "state",
        "intent_confidence": 0.8,
    })
    client = _RejectingLLMClient(payload)
    mgr = MemoryNodeManager(
        db,
        embedding_config={},
        llm_client=client,
        llm_model="gpt-5-test",
        llm_base_url="https://api.openai.com/v1",
    )

    analysis = mgr._analyze_recall_query("Alice Slack alerts")

    assert analysis["search_text"] == "Alice Slack alert preference"
    assert analysis["keywords"] == ["Alice", "Slack"]
    assert len(client.completions.calls) == 2
    assert "temperature" in client.completions.calls[0]
    assert "temperature" not in client.completions.calls[1]
    assert "max_completion_tokens" in client.completions.calls[1]


def test_analyze_recall_query_uses_responses_api_for_gpt5_models(db):
    payload = json.dumps({
        "search_text": "Alice Slack alert preference",
        "keywords": ["Alice", "Slack"],
        "recall_intent": "state",
        "intent_confidence": 0.8,
    })
    client = _ResponsesOnlyClient(payload)
    mgr = MemoryNodeManager(
        db,
        embedding_config={},
        llm_client=client,
        llm_model="gpt-5.4",
        llm_base_url="https://api.openai.com/v1",
    )

    analysis = mgr._analyze_recall_query("Alice Slack alerts")

    assert analysis["search_text"] == "Alice Slack alert preference"
    assert analysis["keywords"] == ["Alice", "Slack"]
    assert len(client.calls) == 1
    assert client.calls[0]["model"] == "gpt-5.4"
    assert "max_output_tokens" in client.calls[0]


def test_resolve_recall_intent_prefers_confident_llm_but_keeps_evidence_override():
    assert MemoryNodeManager._resolve_recall_intent(
        "我之后应该怎么处理 Slack 告警？",
        ["Slack", "告警"],
        "action",
        0.9,
    ) == "action"

    assert MemoryNodeManager._resolve_recall_intent(
        "之前我具体什么时候提到过 Slack 告警？",
        ["Slack", "告警"],
        "action",
        0.9,
    ) == "evidence"


def test_search_memory_evidence_bundles_uses_entities(db):
    columns = {
        row["name"]
        for row in db._conn.execute("PRAGMA table_info(memory_evidence_bundles)").fetchall()
    }
    assert {"embedding", "embedding_text", "embedding_updated_at"}.issubset(columns)
    node_id = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack alerts.",
        keywords=["Slack"],
    )
    alice = db.entity_add_entity("Alice", "PERSON")
    db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="alerts",
        topic_label="alerts",
        bundle_type="insight",
        summary="The user prefers concise escalation notes.",
        keywords=["escalation"],
        source_node_ids=[node_id],
        confidence=0.8,
    )

    results = db.search_memory_evidence_bundles(
        ["unrelated"],
        entities=[{"name": "Alice", "type": "PERSON"}],
        top_k=3,
    )

    assert [row["entity_name"] for row in results] == ["Alice"]


def test_search_memory_evidence_bundles_uses_embedding_similarity(db):
    source_id = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="The user asked to discuss architecture before code.",
        keywords=["architecture"],
    )
    hermes = db.entity_add_entity("Hermes Agent", "PROJECT")
    matching_id = db.memory_upsert_evidence_bundle(
        entity_id=hermes,
        topic_key="workflow",
        topic_label="workflow",
        bundle_type="preference",
        summary="The workflow favors design calibration before implementation.",
        keywords=["architecture"],
        source_node_ids=[source_id],
        confidence=0.8,
        embedding=np.array([[1.0, 0.0]], dtype=np.float32),
        embedding_text="design calibration before implementation",
    )
    db.memory_upsert_evidence_bundle(
        entity_id=hermes,
        topic_key="calendar",
        topic_label="calendar",
        bundle_type="insight",
        summary="Calendar cleanup is unrelated.",
        keywords=["calendar"],
        source_node_ids=[source_id],
        confidence=0.95,
        embedding=np.array([[0.0, 1.0]], dtype=np.float32),
        embedding_text="calendar cleanup",
    )

    results = db.search_memory_evidence_bundles(
        ["alignment"],
        top_k=5,
        query_embedding=np.array([[1.0, 0.0]], dtype=np.float32),
    )

    assert [item["id"] for item in results] == [matching_id]
    assert results[0]["embedding_similarity"] == pytest.approx(1.0)


def test_search_memory_evidence_bundles_rejects_weak_family_term_entity_mismatch(db):
    node_id = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="小明父亲退休后保留了自驾游偏好。",
        keywords=["小明父亲", "退休", "自驾游"],
    )
    xiaoming_father = db.entity_add_entity("小明父亲", "PERSON")
    db.memory_upsert_evidence_bundle(
        entity_id=xiaoming_father,
        topic_key="家庭",
        topic_label="家庭",
        bundle_type="insight",
        summary="小明父亲退休后保留了对开车自驾游和苹果的偏好。",
        keywords=["退休", "自驾游", "苹果"],
        source_node_ids=[node_id],
        confidence=0.7,
    )

    results = db.search_memory_evidence_bundles(
        ["父亲", "中学", "校长"],
        entities=[{"name": "小张父亲", "type": "PERSON"}],
        top_k=3,
    )

    assert results == []


def test_entity_link_records_co_entities(db):
    node_id = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack alerts.",
        keywords=["Alice", "Slack"],
    )
    alice = db.entity_add_entity("Alice", "PERSON")
    slack = db.entity_add_entity("Slack", "PRODUCT")

    db.entity_link_node(node_id, alice)
    db.entity_link_node(node_id, slack)

    rows = {
        row["name"]: json.loads(row["co_entities"])
        for row in db._conn.execute(
            "SELECT name, co_entities FROM entity_nodes WHERE id IN (?, ?)",
            (alice, slack),
        ).fetchall()
    }
    assert rows["Alice"][str(slack)]["name"] == "Slack"
    assert rows["Alice"][str(slack)]["count"] == 1
    assert rows["Slack"][str(alice)]["name"] == "Alice"


def test_memory_reflect_reports_entity_merge_conditions(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    alice_spaced = db.entity_add_entity(" alice ", "PERSON")
    hermes = db.entity_add_entity("Hermes", "PRODUCT")
    hermes_agent = db.entity_add_entity("Hermes Agent", "PRODUCT")
    _ = alice
    _ = hermes

    report = db.reflect_merging_entities(limit=10)

    by_duplicate = {item["duplicate_id"]: item for item in report["candidates"]}
    assert by_duplicate[alice_spaced]["action"] == "merge"
    assert by_duplicate[alice_spaced]["reason"] == "normalized_name_match"
    hermes_candidates = [
        item for item in report["candidates"]
        if {item["canonical_id"], item["duplicate_id"]} == {hermes, hermes_agent}
    ]
    assert hermes_candidates
    assert hermes_candidates[0]["action"] == "candidate"
    assert hermes_candidates[0]["reason"] in {"token_subset", "name_substring"}
    assert report["rules"]["score_weights"]["co_entities"] > 0


def test_memory_reflect_entities_can_scope_candidates_to_anchor_entities(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    alice_spaced = db.entity_add_entity(" alice ", "PERSON")
    hermes = db.entity_add_entity("Hermes", "PRODUCT")
    hermes_agent = db.entity_add_entity("Hermes Agent", "PRODUCT")
    _ = alice

    report = db.reflect_merging_entities(
        limit=10,
        anchor_entity_ids=[alice_spaced],
    )

    pairs = {
        frozenset((item["canonical_id"], item["duplicate_id"]))
        for item in report["candidates"]
    }
    assert frozenset((alice, alice_spaced)) in pairs
    assert frozenset((hermes, hermes_agent)) not in pairs
    assert report["anchor_entity_count"] == 1


def test_memory_node_manager_reflect_delegates_to_db(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    alice_spaced = db.entity_add_entity(" alice ", "PERSON")
    node_id = _add_memory_node(
        db,
        time_key=MemoryNodeManager._memory_time_key(0),
        summary="Alice prefers Slack alerts.",
        keywords=["Alice", "Slack"],
    )
    db.entity_link_node(node_id, alice_spaced)
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})

    report = mgr.reflect(limit=5)

    assert report["evidence_bundle_reflect"]["touched_entity_ids"] == [alice]
    assert report["merge_candidates"] == 1


def test_memory_node_manager_reflect_uses_requested_timestamp_for_fact_day(db):
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})
    requested_at = datetime(2031, 4, 5, 23, 30, tzinfo=timezone(timedelta(hours=8)))
    requested_date_keys = []
    original_get_candidates = db.get_unobserved_nodes_for_observation

    def capture_candidates(**kwargs):
        requested_date_keys.append(kwargs.get("date_key"))
        return original_get_candidates(**kwargs)

    db.get_unobserved_nodes_for_observation = capture_candidates

    report = mgr.reflect(limit=5, reflect_timestamp=requested_at)

    assert requested_date_keys == ["2031-04-05", "2031-04-05"]
    assert report["node_decay"]["evaluated_at"] == requested_at.isoformat()
    assert report["evidence_bundle_reflect"]["candidate_count"] == 0


def test_memory_reflect_can_merge_normalized_entity_duplicates(db):
    node_id = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack alerts.",
        keywords=["Alice", "Slack"],
    )
    alice = db.entity_add_entity("Alice", "PERSON")
    alice_spaced = db.entity_add_entity(" alice ", "PERSON")
    db.entity_link_node(node_id, alice_spaced)

    report = db.reflect_merging_entities(limit=10)

    assert report["merged"] == 1
    assert db._conn.execute(
        "SELECT COUNT(*) FROM entity_nodes WHERE id = ?",
        (alice_spaced,),
    ).fetchone()[0] == 0
    link = db._conn.execute(
        "SELECT entity_id FROM memory_node_entities WHERE node_id = ?",
        (node_id,),
    ).fetchone()
    assert link["entity_id"] == alice
    metadata = json.loads(db._conn.execute(
        "SELECT metadata FROM entity_nodes WHERE id = ?",
        (alice,),
    ).fetchone()["metadata"])
    assert " alice " in metadata["aliases"]


def test_reflect_merges_same_topic_observations_with_llm(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    alice_spaced = db.entity_add_entity(" alice ", "PERSON")
    first_node = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Slack", "alerts"],
    )
    second_node = _add_memory_node(
        db,
        time_key="2026-05-02 10:00:00",
        summary="Alice wants incident notifications in Slack.",
        keywords=["Slack", "notifications"],
    )
    touched_node = _add_memory_node(
        db,
        time_key=MemoryNodeManager._memory_time_key(0),
        summary="Alice still discusses Slack alerts.",
        keywords=["alerts"],
    )
    db.entity_link_node(first_node, alice)
    db.entity_link_node(second_node, alice_spaced)
    db.entity_link_node(touched_node, alice_spaced)
    first_observation = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="alerts",
        topic_label="alerts",
        bundle_type="entity_topic",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Slack", "alerts"],
        source_node_ids=[first_node],
        confidence=0.8,
        metadata={"observation_kind": "context"},
    )
    second_observation = db.memory_upsert_evidence_bundle(
        entity_id=alice_spaced,
        topic_key="alerts",
        topic_label="alerts",
        bundle_type="entity_topic",
        summary="Alice routes incident notifications through Slack.",
        keywords=["Slack", "notifications"],
        source_node_ids=[second_node],
        confidence=0.75,
        metadata={"observation_kind": "context"},
    )
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "category": "observation",
                "summary": "Alice consistently wants urgent and incident alerts routed through Slack.",
                "keywords": ["Slack", "alerts", "notifications"],
                "confidence": 0.9,
                "metadata": {"observation_kind": "context"},
            }),
            json.dumps({
                "category": "observation",
                "summary": "Alice consistently wants urgent and incident alerts routed through Slack.",
                "keywords": ["Slack", "alerts", "notifications"],
                "confidence": 0.9,
                "metadata": {"observation_kind": "context"},
            }),
            json.dumps({"should_create": False}),
        ],
    )

    report = mgr.reflect(limit=10)

    assert report["merged"] == 1
    assert report["evidence_bundle_groups_merged"] == 1
    merge_prompt = next(
        prompt
        for prompt in mgr.llm_prompts
        if "相关既有 observation" in prompt
        and "Alice routes incident notifications through Slack." in prompt
    )
    assert "已有 observation" in merge_prompt
    assert "相关既有 observation" in merge_prompt
    assert "Alice prefers Slack for urgent alerts." in merge_prompt
    assert "Alice routes incident notifications through Slack." in merge_prompt
    assert "Alice still discusses Slack alerts." in merge_prompt
    source_facts_section = merge_prompt.split("新的来源事实：", 1)[1]
    assert "Alice still discusses Slack alerts." in source_facts_section
    assert "Alice prefers Slack for urgent alerts." not in source_facts_section
    assert "Alice wants incident notifications in Slack." not in source_facts_section
    observation_prompts = [
        prompt
        for prompt in mgr.llm_prompts
        if "observation consolidation 模块" in prompt
    ]
    assert observation_prompts == [merge_prompt]
    rows = db._conn.execute(
        "SELECT id, entity_id, topic_key, observation_type, summary, keywords "
        "FROM memory_evidence_bundles ORDER BY id"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["id"] in {first_observation, second_observation}
    assert rows[0]["entity_id"] == alice
    assert rows[0]["topic_key"] == "alerts"
    assert rows[0]["observation_type"] == "observation"
    assert "urgent and incident alerts" in rows[0]["summary"]
    assert "notifications" in rows[0]["keywords"]
    source_ids = {
        row["node_id"]
        for row in db._conn.execute(
            "SELECT node_id FROM memory_evidence_bundle_sources WHERE observation_id = ?",
            (rows[0]["id"],),
        ).fetchall()
    }
    assert source_ids == {first_node, second_node, touched_node}
    source_roles = {
        row["node_id"]: row["role"]
        for row in db._conn.execute(
            "SELECT node_id, role FROM memory_evidence_bundle_sources WHERE observation_id = ?",
            (rows[0]["id"],),
        ).fetchall()
    }
    assert source_roles == {
        first_node: "initial",
        second_node: "initial",
        touched_node: "matched",
    }


def test_reflect_keeps_insight_and_task_observations_separate(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    node_ids = []
    for idx, summary in enumerate(
        [
            "Alice prefers Slack for urgent alerts.",
            "Alice is actively implementing Slack alert routing.",
        ],
        1,
    ):
        node_id = _add_memory_node(
            db,
            time_key=f"2026-05-0{idx} 10:00:00",
            summary=summary,
            keywords=["Slack", "alerts"],
        )
        db.entity_link_node(node_id, alice)
        node_ids.append(node_id)
    insight_id = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="alerts",
        topic_label="alerts",
        bundle_type="insight",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Slack", "alerts"],
        source_node_ids=[node_ids[0]],
    )
    task_id = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="alerts",
        topic_label="alerts",
        bundle_type="task",
        summary="Alice is actively implementing Slack alert routing.",
        keywords=["Slack", "alerts", "routing"],
        source_node_ids=[node_ids[1]],
        metadata={
            "task_status": "active",
            "task_source": "inferred_from_observation",
        },
    )

    groups = db.find_duplicated_evidence_bundle_groups(entity_ids=[alice])

    assert groups == []
    rows = db._conn.execute(
        "SELECT id, observation_type FROM memory_evidence_bundles ORDER BY id"
    ).fetchall()
    assert [(row["id"], row["observation_type"]) for row in rows] == [
        (insight_id, "insight"),
        (task_id, "task"),
    ]


def test_reflect_observation_decay_uses_fact_type_half_lives(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    world_node = _add_memory_node(
        db,
        time_key="2026-01-01 10:00:00",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Alice", "Slack"],
        fact_type="semantic",
    )
    experience_node = _add_memory_node(
        db,
        time_key="2026-01-01 11:00:00",
        summary="Hermes previously routed Alice's alerts through Slack.",
        keywords=["Alice", "Slack"],
        fact_type="episodic",
    )
    world_observation = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="world-alerts",
        topic_label="world alerts",
        bundle_type="insight",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Slack", "alerts"],
        source_node_ids=[world_node],
    )
    experience_observation = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="experience-alerts",
        topic_label="experience alerts",
        bundle_type="insight",
        summary="Hermes has prior Slack alert routing experience for Alice.",
        keywords=["Slack", "alerts"],
        source_node_ids=[experience_node],
    )

    node_report = db.memory_reflect_node_decay(
        fact_half_life_days=365,
        experience_half_life_days=30,
        now=datetime(2026, 4, 1, 0, 0, 0),
    )
    report = db.memory_reflect_evidence_bundle_decay(
        threshold=0.3,
        now=datetime(2026, 4, 1, 0, 0, 0),
    )

    rows = {
        row["id"]: row["status"]
        for row in db._conn.execute(
            "SELECT id, status FROM memory_evidence_bundles WHERE id IN (?, ?)",
            (world_observation, experience_observation),
        ).fetchall()
    }
    node_scores = {
        row["id"]: row["decay_score"]
        for row in db._conn.execute(
            "SELECT id, decay_score FROM memory_nodes WHERE id IN (?, ?)",
            (world_node, experience_node),
        ).fetchall()
    }
    assert node_report["updated"] == 2
    assert node_scores[world_node] > node_scores[experience_node]
    assert report["inactivated"] == 1
    assert rows[world_observation] == "active"
    assert rows[experience_observation] == "inactive"


def test_task_inactivity_policy_pauses_and_stales_idle_tasks(db):
    now = datetime(2026, 5, 15, 12, 0, 0)
    entity_id = db.entity_add_entity("Hermes Agent", "PROJECT")
    node_ids = [
        _add_memory_node(
            db,
            time_key=f"2026-05-15 10:0{idx}:00",
            summary=f"Task source {idx}",
            keywords=["memory"],
        )
        for idx in range(4)
    ]
    for node_id in node_ids:
        db.entity_link_node(node_id, entity_id)

    def add_task(status, last_supported_at, source_node_id):
        interpretation_id = db.memory_upsert_interpretation(
            claim=f"Task {source_node_id}",
            target_text=f"task-{source_node_id}",
            scope=f"task-{source_node_id}",
            interpretation_type="task",
            metadata={
                "task_status": status,
                "task_source": "inferred_from_interpretation",
                "entity_id": entity_id,
                "entity_name": "Hermes Agent",
                "topic_key": f"task-{source_node_id}",
                "topic_label": f"task-{source_node_id}",
            },
        )
        db._conn.execute(
            "UPDATE memory_interpretations SET last_supported_at = ?, updated_at = ? WHERE id = ?",
            (last_supported_at.isoformat(), last_supported_at.isoformat(), interpretation_id),
        )
        return interpretation_id

    active_to_paused = add_task("active", now - timedelta(days=10), node_ids[0])
    active_to_stale = add_task("active", now - timedelta(days=40), node_ids[1])
    blocked_to_stale = add_task("blocked", now - timedelta(days=40), node_ids[2])
    recent_active = add_task("active", now - timedelta(days=1), node_ids[3])

    report = db.memory_reflect_task_inactivity(
        active_to_paused_days=7,
        stale_days=30,
        now=now,
    )

    assert report["checked"] == 4
    assert report["paused"] == 1
    assert report["stale"] == 2
    rows = {
        row["id"]: (row["status"], json.loads(row["metadata"]))
        for row in db._conn.execute(
            "SELECT id, status, metadata FROM memory_interpretations ORDER BY id"
        ).fetchall()
    }
    assert rows[active_to_paused][0] == "current"
    assert rows[active_to_paused][1]["task_status"] == "paused"
    assert rows[active_to_paused][1]["previous_task_status"] == "active"
    assert rows[active_to_paused][1]["status_updated_by"] == "reflect_task_inactivity_policy"
    assert rows[active_to_stale][1]["task_status"] == "stale"
    assert rows[blocked_to_stale][1]["task_status"] == "stale"
    assert rows[recent_active][1]["task_status"] == "active"


def test_memory_node_manager_reflect_reports_task_inactivity(db):
    now = datetime.now().astimezone()
    entity_id = db.entity_add_entity("Hermes Agent", "PROJECT")
    node_id = _add_memory_node(
        db,
        time_key=MemoryNodeManager._memory_time_key(0),
        summary="用户正在完善 Hermes Agent 的记忆系统。",
        keywords=["memory"],
    )
    db.entity_link_node(node_id, entity_id)
    interpretation_id = db.memory_upsert_interpretation(
        claim="用户正在完善 Hermes Agent 的记忆系统。",
        target_text="memory-system",
        scope="memory-system",
        interpretation_type="task",
        metadata={
            "task_status": "active",
            "task_source": "inferred_from_interpretation",
            "entity_id": entity_id,
            "entity_name": "Hermes Agent",
            "topic_key": "memory-system",
            "topic_label": "memory-system",
        },
    )
    idle_at = now - timedelta(days=10)
    db._conn.execute(
        "UPDATE memory_interpretations SET last_supported_at = ?, updated_at = ? WHERE id = ?",
        (idle_at.isoformat(), idle_at.isoformat(), interpretation_id),
    )
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})

    report = mgr.reflect(
        limit=10,
        task_active_to_paused_days=7,
        task_stale_days=30,
    )

    assert report["tasks_paused"] == 1
    assert report["tasks_stale"] == 0
    assert report["task_inactivity"]["changed"] == 1
    metadata = json.loads(db._conn.execute(
        "SELECT metadata FROM memory_interpretations WHERE id = ?",
        (interpretation_id,),
    ).fetchone()["metadata"])
    assert metadata["task_status"] == "paused"


def test_search_memory_evidence_bundles_ignores_inactive_observations(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    active_node = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice currently prefers Slack for urgent alerts.",
        keywords=["Alice", "Slack"],
    )
    stale_node = _add_memory_node(
        db,
        time_key="2000-01-01 10:00:00",
        summary="Alice once preferred email alerts.",
        keywords=["Alice", "email"],
    )
    active_observation = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="slack-alerts",
        topic_label="Slack alerts",
        bundle_type="insight",
        summary="Alice currently prefers Slack for urgent alerts.",
        keywords=["Slack", "alerts"],
        source_node_ids=[active_node],
    )
    inactive_observation = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="email-alerts",
        topic_label="email alerts",
        bundle_type="insight",
        summary="Alice once preferred email alerts.",
        keywords=["email", "alerts"],
        source_node_ids=[stale_node],
    )
    db._conn.execute(
        "UPDATE memory_evidence_bundles SET status = 'inactive' WHERE id = ?",
        (inactive_observation,),
    )

    results = db.search_memory_evidence_bundles(["Alice", "alerts"], top_k=5)

    assert [item["id"] for item in results] == [active_observation]


def test_memory_node_manager_reflect_inactivates_stale_observations(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    old_node = _add_memory_node(
        db,
        time_key="2000-01-01 10:00:00",
        summary="Alice used email alerts long ago.",
        keywords=["Alice", "email"],
        fact_type="episodic",
    )
    observation_id = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="email-alerts",
        topic_label="email alerts",
        bundle_type="insight",
        summary="Alice used email alerts long ago.",
        keywords=["email", "alerts"],
        source_node_ids=[old_node],
    )
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})

    report = mgr.reflect(
        experience_half_life_days=7,
        observation_decay_threshold=0.9,
    )

    row = db._conn.execute(
        "SELECT mo.status, mn.decay_score "
        "FROM memory_evidence_bundles mo "
        "JOIN memory_evidence_bundle_sources mos ON mos.observation_id = mo.id "
        "JOIN memory_nodes mn ON mn.id = mos.node_id "
        "WHERE mo.id = ?",
        (observation_id,),
    ).fetchone()
    assert report["node_decay"]["updated"] == 1
    assert report["evidence_bundles_inactivated"] == 1
    assert row["status"] == "inactive"
    assert row["decay_score"] < 0.9


def test_recall_formats_semantic_and_episodic_sections(db, monkeypatch):
    _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Alice", "Slack"],
        fact_type="semantic",
    )
    _add_memory_node(
        db,
        time_key="2026-05-01 11:00:00",
        summary="Hermes recommended Slack alert routing for Alice.",
        keywords=["Alice", "Slack"],
        fact_type="episodic",
    )
    monkeypatch.setattr(db, "_search_memory_vector", lambda *args, **kwargs: {})
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[json.dumps({"summary": "Alice Slack alerts", "keywords": ["Alice", "Slack"]})],
    )

    context = mgr.recall("Alice Slack alerts")

    assert WORLD_FACT_SECTION_HEADER in context
    assert EXPERIENCE_SECTION_HEADER in context
    assert "semantic memories" in context
    assert "episodic memories" in context
    assert "Alice prefers Slack for urgent alerts." in context
    assert "Hermes recommended Slack alert routing for Alice." in context


def test_recall_uses_query_analysis_search_text_for_embedding(db, monkeypatch):
    _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Alice", "Slack"],
        fact_type="semantic",
    )
    monkeypatch.setattr(db, "_search_memory_vector", lambda *args, **kwargs: {})
    capture = _CapturingEmbeddingClient()
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "search_text": "Alice urgent Slack alert routing preference",
                "keywords": ["Alice", "Slack", "alerts"],
                "recall_intent": "action",
                "intent_confidence": 0.91,
                "layer_preference": {
                    "interpretations": 0.6,
                    "observations": 0.3,
                    "facts": 0.1,
                },
            })
        ],
    )
    mgr._embedding_client = capture

    context = mgr.recall("What should I remember about Alice alerts?")

    assert "Alice prefers Slack for urgent alerts." in context
    assert capture.texts
    assert "Alice urgent Slack alert routing preference" in capture.texts[0]
    assert "keywords: Alice, Slack, alerts" in capture.texts[0]


def test_recall_emits_structured_stage_logs(db, monkeypatch, caplog):
    _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Alice", "Slack"],
        fact_type="semantic",
    )
    monkeypatch.setattr(db, "_search_memory_vector", lambda *args, **kwargs: {})
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "search_text": "Alice Slack alerts preference",
                "keywords": ["Alice", "Slack"],
                "recall_intent": "state",
                "intent_confidence": 0.8,
            })
        ],
    )

    with caplog.at_level(logging.INFO, logger="agent.memory_node_manager"):
        context = mgr.recall("Alice Slack alerts")

    assert "Alice prefers Slack for urgent alerts." in context
    records = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "agent.memory_node_manager"
        and '"scope": "memory_recall"' in record.getMessage()
    ]
    events = [record["event"] for record in records]
    assert events[:4] == ["start", "query_prepared", "gate_decided", "query_analyzed"]
    assert "candidates_found" in events
    assert "ranked" in events
    assert events[-1] == "finish"
    finish = records[-1]
    assert finish["payload"]["status"] == "ok"
    assert finish["payload"]["counts"]["semantic_facts"] == 1


def test_recall_rerank_preserves_layer_order_after_selection():
    ranked = MemoryNodeManager._rank_recall_candidates(
        interpretations=[],
        observations=[
            {
                "id": 1,
                "summary": "Layer-local search ranked this observation first.",
                "keywords": ["workflow"],
                "confidence": 0.4,
            },
            {
                "id": 2,
                "summary": "Layer-local search ranked this observation second but it mentions calibration.",
                "keywords": ["calibration"],
                "confidence": 0.95,
            },
        ],
        semantic_facts=[],
        episodic_facts=[],
        terms=["calibration"],
        intent="balanced",
        layer_limits={"interpretations": 0, "observations": 2, "facts": 0},
    )

    assert [item["id"] for item in ranked["observations"]] == [1, 2]
    assert ranked["observations"][1]["_recall_score"] > ranked["observations"][0]["_recall_score"]


def test_recall_formats_current_interpretations_before_evidence(db, monkeypatch):
    db.memory_upsert_interpretation(
        claim="用户当前倾向先用 heuristic 控制 task fact 选择，再谨慎修改 prompt。",
        subject_text="user",
        target_text="memory task fact selection",
        scope="memory-system-design",
        interpretation_type="inferred_preference",
        confidence=0.86,
        action_implication="后续先讨论 heuristic/data-flow，再考虑 prompt guidance。",
    )
    _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="用户要求先过滤 preference/instruction/context/other 类型的 fact。",
        keywords=["heuristic", "task", "fact"],
        fact_type="semantic",
    )
    monkeypatch.setattr(db, "_search_memory_vector", lambda *args, **kwargs: {})
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[json.dumps({"summary": "heuristic task fact", "keywords": ["heuristic", "task", "fact"]})],
    )

    context = mgr.recall("heuristic task fact")

    assert INTERPRETATION_SECTION_HEADER in context
    assert "agent interpretations derived from memory evidence" in context
    assert "用户当前倾向先用 heuristic 控制 task fact 选择" in context
    assert "action implication: 后续先讨论 heuristic/data-flow，再考虑 prompt guidance。" in context
    assert context.index(INTERPRETATION_SECTION_HEADER) < context.index(WORLD_FACT_SECTION_HEADER)


def test_recall_expands_interpretation_to_evidence_observations(db, monkeypatch):
    alice = db.entity_add_entity("Alice", "PERSON")
    source_id = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice repeatedly asked to discuss architecture before code changes.",
        keywords=["architecture", "discussion"],
        fact_type="episodic",
    )
    db.entity_link_node(source_id, alice)
    evidence_bundle_id = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="architecture-first",
        topic_label="Architecture-first workflow",
        bundle_type="entity_topic",
        summary="Architecture-first workflow evidence.",
        keywords=["architecture", "discussion"],
        source_node_ids=[source_id],
    )
    observation_id = db.memory_create_observation(
        evidence_bundle_id,
        {
            "observation_type": "preference_signal",
            "summary": (
                "Alice's workflow favors design discussion before implementation."
            ),
            "confidence": 0.8,
            "source_node_ids": [source_id],
        },
    )
    assert observation_id is not None
    db.memory_upsert_interpretation(
        claim="Alice currently prefers calibration before implementation.",
        entity_id=alice,
        target_text="implementation workflow",
        scope="workflow",
        interpretation_type="preference",
        confidence=0.8,
        evidence_observation_ids=[observation_id],
    )
    monkeypatch.setattr(db, "_search_memory_vector", lambda *args, **kwargs: {})
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[json.dumps({"summary": "calibration implementation", "keywords": ["calibration", "implementation"]})],
    )

    context = mgr.recall("calibration implementation")

    assert "Alice currently prefers calibration before implementation." in context
    assert "Alice's workflow favors design discussion before implementation." in context
    assert "Alice repeatedly asked to discuss architecture before code changes." in context


def test_recall_formats_interpretation_direct_evidence_once(db, monkeypatch):
    evidence_id = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="The recall router should connect interpretations to evidence facts.",
        keywords=["recall", "router", "evidence"],
        fact_type="semantic",
    )
    db.memory_upsert_interpretation(
        claim="Recall routing should expose evidence facts for interpretation hits.",
        target_text="recall routing",
        scope="memory-recall",
        interpretation_type="insight",
        confidence=0.82,
        evidence_node_ids=[evidence_id],
    )
    monkeypatch.setattr(db, "_search_memory_vector", lambda *args, **kwargs: {})
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[json.dumps({"summary": "recall router evidence", "keywords": ["recall", "router", "evidence"]})],
    )

    context = mgr.recall("recall router evidence")

    assert INTERPRETATION_SECTION_HEADER in context
    assert OBSERVATION_SUPPORT_SECTION_HEADER in context
    assert context.count("The recall router should connect interpretations to evidence facts.") == 1


def test_store_turn_consolidates_observation_for_entity_topic_bucket(db):
    retain_payload = {
        "facts": [
            {
                "text": "Alice prefers Slack for urgent alerts.",
                "keywords": ["Alice", "Slack", "alerts"],
                "topic": ["Slack alerts"],
                "fact_type": "semantic",
                "entities": [{"name": "Alice", "type": "PERSON"}],
            },
            {
                "text": "Alice dislikes email for urgent alerts.",
                "keywords": ["Alice", "email", "alerts"],
                "topic": ["Slack alerts"],
                "fact_type": "semantic",
                "entities": [{"name": "Alice", "type": "PERSON"}],
            },
            {
                "text": "Hermes previously recommended Slack alert routing for Alice.",
                "keywords": ["Hermes", "Slack", "routing", "Alice"],
                "topic": ["Slack alerts"],
                "fact_type": "episodic",
                "entities": [{"name": "Alice", "type": "PERSON"}],
            },
        ],
        "causal_relations": [],
    }
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps(retain_payload),
            json.dumps({
                "category": "observation",
                "summary": "Alice's urgent alert workflow is Slack-centered.",
                "keywords": ["Slack", "alerts"],
                "confidence": 0.86,
                "metadata": {"observation_kind": "context"},
            }),
        ],
    )
    assert mgr.store_turn("Alice urgent alerts", "Use Slack.") is True
    assert db._conn.execute("SELECT COUNT(*) FROM memory_evidence_bundles").fetchone()[0] == 0

    report = mgr.reflect(limit=10)
    assert report["evidence_bundle_reflect"]["candidate_count"] == 3
    assert report["evidence_bundles_consolidated"] == 1
    assert report["evidence_bundle_reflect"]["fact_clusters_consolidated"] == 1

    observation = db._conn.execute(
        "SELECT mo.summary, mo.topic_key, mo.observation_type, mo.metadata, en.name AS entity_name "
        "FROM memory_evidence_bundles mo "
        "JOIN entity_nodes en ON en.id = mo.entity_id"
    ).fetchone()
    assert observation["entity_name"] == "Alice"
    assert observation["topic_key"] == "slack-alerts"
    assert observation["observation_type"] == "observation"
    assert json.loads(observation["metadata"])["observation_kind"] == "context"
    assert observation["summary"] == "Alice's urgent alert workflow is Slack-centered."
    sources = db._conn.execute("SELECT node_id FROM memory_evidence_bundle_sources").fetchall()
    assert len(sources) == 3
    source_roles = db._conn.execute(
        "SELECT node_id, role FROM memory_evidence_bundle_sources ORDER BY node_id"
    ).fetchall()
    assert {row["node_id"]: row["role"] for row in source_roles} == {
        source["node_id"]: "initial"
        for source in sources
    }


def test_unmatched_fact_clusters_keep_exact_topics_separate(db):
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})
    facts = [
        {
            "node_id": 1,
            "time_key": "2026-05-01 10:00:00+00:00#00",
            "summary": "用户讨论家庭情况。",
            "topics": ["家庭"],
            "linked_entities": [(7, "用户")],
            "fact_type": "semantic",
            "fact_subject": "user",
            "fact_kind": "context",
        },
        {
            "node_id": 2,
            "time_key": "2026-05-01 10:30:00+00:00#00",
            "summary": "用户讨论家庭关系。",
            "topics": ["家庭关系"],
            "linked_entities": [(7, "用户")],
            "fact_type": "semantic",
            "fact_subject": "user",
            "fact_kind": "context",
        },
    ]

    clusters = mgr._cluster_unmatched_facts(facts, excluded_node_ids=set())

    assert {
        (cluster["topic_key"], tuple(cluster["source_node_ids"]))
        for cluster in clusters
    } == {
        ("家庭", (1,)),
        ("家庭关系", (2,)),
    }
    assert all(cluster["topic_match"] == "exact" for cluster in clusters)


def test_unmatched_fact_clusters_do_not_match_different_specific_family_topics(db):
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})
    facts = [
        {
            "node_id": 1,
            "time_key": "2026-05-01 10:00:00+00:00#00",
            "summary": "用户讨论家庭关系。",
            "topics": ["家庭关系"],
            "linked_entities": [(7, "用户")],
            "fact_type": "semantic",
            "fact_subject": "user",
            "fact_kind": "context",
        },
        {
            "node_id": 2,
            "time_key": "2026-05-01 10:30:00+00:00#00",
            "summary": "用户讨论家庭互动。",
            "topics": ["家庭互动"],
            "linked_entities": [(7, "用户")],
            "fact_type": "semantic",
            "fact_subject": "user",
            "fact_kind": "context",
        },
    ]

    clusters = mgr._cluster_unmatched_facts(facts, excluded_node_ids=set())

    assert not [
        cluster for cluster in clusters
        if cluster.get("topic_key") == "家庭" and cluster.get("topic_match") == "normalized"
    ]


def test_unmatched_fact_clusters_require_time_window_for_generalized_topic_match(db):
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})
    facts = [
        {
            "node_id": 1,
            "time_key": "2026-05-01 10:00:00+00:00#00",
            "summary": "用户讨论健康。",
            "topics": ["健康"],
            "linked_entities": [(7, "用户")],
            "fact_type": "semantic",
            "fact_subject": "user",
            "fact_kind": "context",
        },
        {
            "node_id": 2,
            "time_key": "2026-05-01 13:30:00+00:00#00",
            "summary": "用户讨论健康管理。",
            "topics": ["健康管理"],
            "linked_entities": [(7, "用户")],
            "fact_type": "semantic",
            "fact_subject": "user",
            "fact_kind": "context",
        },
    ]

    clusters = mgr._cluster_unmatched_facts(facts, excluded_node_ids=set())

    assert not [
        cluster for cluster in clusters
        if cluster.get("topic_key") == "健康" and cluster.get("topic_match") == "normalized"
    ]


def test_unmatched_fact_clusters_ignore_time_when_topics_are_exact(db):
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})
    facts = [
        {
            "node_id": 1,
            "time_key": "2026-05-01 10:00:00+00:00#00",
            "summary": "用户讨论健康。",
            "topics": ["健康"],
            "linked_entities": [(7, "用户")],
            "fact_type": "semantic",
            "fact_subject": "user",
            "fact_kind": "context",
        },
        {
            "node_id": 2,
            "time_key": "2026-05-01 10:30:00+00:00#00",
            "summary": "用户讨论健康管理。",
            "topics": ["健康管理"],
            "linked_entities": [(7, "用户")],
            "fact_type": "semantic",
            "fact_subject": "user",
            "fact_kind": "context",
        },
        {
            "node_id": 3,
            "time_key": "2026-05-01 13:30:00+00:00#00",
            "summary": "用户很晚后再次讨论健康管理。",
            "topics": ["健康管理"],
            "linked_entities": [(7, "用户")],
            "fact_type": "semantic",
            "fact_subject": "user",
            "fact_kind": "context",
        },
    ]

    clusters = mgr._cluster_unmatched_facts(facts, excluded_node_ids=set())

    assert {
        (cluster["topic_key"], tuple(cluster["source_node_ids"]))
        for cluster in clusters
    } == {
        ("健康", (1,)),
        ("健康管理", (2, 3)),
    }


def test_unmatched_fact_clusters_do_not_generalize_topic_suffixes(db):
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})
    facts = [
        {
            "node_id": 1,
            "time_key": "2026-05-01 10:00:00+00:00#00",
            "summary": "用户早上讨论健康。",
            "topics": ["健康"],
            "linked_entities": [(7, "用户")],
            "fact_type": "semantic",
            "fact_subject": "user",
            "fact_kind": "context",
        },
        {
            "node_id": 2,
            "time_key": "2026-05-01 16:00:00+00:00#00",
            "summary": "用户下午再次讨论健康。",
            "topics": ["健康"],
            "linked_entities": [(7, "用户")],
            "fact_type": "semantic",
            "fact_subject": "user",
            "fact_kind": "context",
        },
        {
            "node_id": 3,
            "time_key": "2026-05-01 16:30:00+00:00#00",
            "summary": "用户下午讨论健康管理。",
            "topics": ["健康管理"],
            "linked_entities": [(7, "用户")],
            "fact_type": "semantic",
            "fact_subject": "user",
            "fact_kind": "context",
        },
    ]

    clusters = mgr._cluster_unmatched_facts(facts, excluded_node_ids=set())

    assert {
        (cluster["topic_key"], tuple(cluster["source_node_ids"]))
        for cluster in clusters
    } == {
        ("健康", (1, 2)),
        ("健康管理", (3,)),
    }


def test_unmatched_fact_clusters_assign_each_fact_to_one_primary_cluster(db):
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})
    facts = [
        {
            "node_id": 1,
            "time_key": "2026-05-01 10:00:00+00:00#00",
            "summary": "用户讨论家庭关系以及孩子的教育安排。",
            "keywords": ["用户", "家庭关系", "教育"],
            "topics": ["家庭", "家庭关系"],
            "linked_entities": [(7, "用户"), (8, "孩子")],
            "fact_type": "semantic",
            "fact_subject": "user",
            "fact_kind": "context",
        },
        {
            "node_id": 2,
            "time_key": "2026-05-01 10:20:00+00:00#00",
            "summary": "用户补充了家庭关系的情况。",
            "keywords": ["用户", "家庭关系"],
            "topics": ["家庭关系"],
            "linked_entities": [(7, "用户")],
            "fact_type": "semantic",
            "fact_subject": "user",
            "fact_kind": "context",
        },
        {
            "node_id": 3,
            "time_key": "2026-05-01 10:30:00+00:00#00",
            "summary": "用户继续讨论家庭。",
            "keywords": ["用户", "家庭"],
            "topics": ["家庭"],
            "linked_entities": [(7, "用户")],
            "fact_type": "semantic",
            "fact_subject": "user",
            "fact_kind": "context",
        },
    ]

    clusters = mgr._cluster_unmatched_facts(facts, excluded_node_ids=set())

    assigned_node_ids = [
        node_id
        for cluster in clusters
        for node_id in cluster["source_node_ids"]
    ]
    assert sorted(assigned_node_ids) == [1, 2, 3]
    assert len(assigned_node_ids) == len(set(assigned_node_ids))
    assert {
        (cluster["entity_id"], cluster["topic_key"], tuple(cluster["source_node_ids"]))
        for cluster in clusters
    } == {
        (7, "家庭", (1, 3)),
        (7, "家庭关系", (2,)),
    }


def test_unmatched_fact_clusters_use_family_profile_without_duplicate_mixed_bucket(db):
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})
    facts = [
        {
            "node_id": 1,
            "time_key": "2026-05-01 10:00:00+00:00#00",
            "summary": "用户偏好使用邮件接收日报。",
            "topics": ["日报通知"],
            "linked_entities": [(7, "用户")],
            "fact_type": "semantic",
            "fact_subject": "user",
            "fact_kind": "preference",
        },
        {
            "node_id": 2,
            "time_key": "2026-05-01 10:10:00+00:00#00",
            "summary": "用户要求日报通知保持简洁。",
            "topics": ["日报通知"],
            "linked_entities": [(7, "用户")],
            "fact_type": "semantic",
            "fact_subject": "user",
            "fact_kind": "instruction",
        },
    ]

    clusters = mgr._cluster_unmatched_facts(facts, excluded_node_ids=set())

    assert len(clusters) == 1
    assert clusters[0]["cluster_family"] == "preference"
    assert clusters[0]["family_distribution"] == {"preference": 2}
    assert clusters[0]["source_node_ids"] == [1, 2]
    assert clusters[0]["can_create_evidence_bundle"] is True


def test_unmatched_fact_clusters_keep_singleton_for_matching_only(db):
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})
    facts = [
        {
            "node_id": 1,
            "time_key": "2026-05-01 10:00:00+00:00#00",
            "summary": "用户提到一个尚未重复出现的新主题。",
            "topics": ["新主题"],
            "linked_entities": [(7, "用户")],
            "fact_type": "semantic",
            "fact_subject": "user",
            "fact_kind": "context",
        },
    ]

    clusters = mgr._cluster_unmatched_facts(facts, excluded_node_ids=set())

    assert len(clusters) == 1
    assert clusters[0]["source_node_ids"] == [1]
    assert clusters[0]["is_singleton"] is True
    assert clusters[0]["can_create_evidence_bundle"] is False
    assert clusters[0]["cluster_reason"] == "single_fact_deferred"


def test_cluster_unprocessed_facts_returns_only_structural_fields(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    facts = []
    for index, summary in enumerate(
        [
            "Alice implemented the alert routing fallback.",
            "Alice verified the alert routing fallback.",
        ],
    ):
        node_id = _add_memory_node(
            db,
            time_key=f"2026-05-01 10:{index:02d}:00",
            summary=summary,
            keywords=["alert-routing"],
            fact_type="episodic",
            fact_kind="action",
            task_event_like=True,
        )
        facts.append({
            "node_id": node_id,
            "time_key": f"2026-05-01 10:{index:02d}:00",
            "summary": summary,
            "keywords": ["alert-routing"],
            "topics": ["alert-routing"],
            "primary_entity_id": alice,
            "primary_entity_name": "Alice",
            "primary_topic": "alert-routing",
            "linked_entities": [(alice, "Alice")],
            "fact_type": "episodic",
            "fact_kind": "action",
        })
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})

    clusters = mgr._cluster_unprocessed_facts(
        facts,
        excluded_node_ids=set(),
    )

    assert len(clusters) == 1
    assert set(clusters[0]) == {
        "entity_id",
        "entity_name",
        "topic_key",
        "topic_label",
        "source_nodes",
        "source_node_ids",
        "can_create_evidence_bundle",
    }
    assert clusters[0]["can_create_evidence_bundle"] is True


def test_reflect_generates_interpretation_from_consolidated_observation(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    node_ids = []
    for idx, summary in enumerate(
        [
            "Alice prefers Slack for urgent alerts.",
            "Alice dislikes email for urgent alerts.",
            "Hermes recommended Slack alert routing for Alice.",
        ],
        1,
    ):
        node_id = _add_memory_node(
            db,
            time_key=MemoryNodeManager._memory_time_key(idx),
            summary=summary,
            keywords=["Slack alerts"],
            fact_type="episodic" if idx == 3 else "semantic",
            fact_kind="recommendation" if idx == 3 else "preference",
        )
        db.entity_link_node(node_id, alice)
        node_ids.append(node_id)
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "observation_type": "strategy",
                "summary": "Hermes recommended Slack alert routing for Alice.",
                "confidence": 0.82,
            }),
            json.dumps({
                "observation_type": "preference_signal",
                "summary": (
                    "Alice explicitly prefers Slack and dislikes email for "
                    "urgent alerts."
                ),
                "confidence": 0.9,
            }),
            json.dumps({
                "should_create": True,
                "claim": "Agent 当前解释为 Alice 的紧急告警协作应优先使用 Slack。",
                "target_text": "urgent alert routing",
                "scope": "alert-workflow",
                "interpretation_type": "explicit_preference",
                "polarity": "positive",
                "strength": 0.9,
                "confidence": 0.88,
                "status": "current",
                "conflict_status": "none",
                "action_implication": "后续涉及 Alice 的紧急告警时优先建议 Slack 路由。",
                "evidence_node_ids": [node_ids[0], node_ids[1], 99999],
                "evidence_observation_ids": [1],
                "counter_evidence_node_ids": [],
                "counter_evidence_observation_ids": [],
            }),
        ],
    )

    report = mgr.reflect(limit=10)

    assert report["evidence_bundles_consolidated"] == 1
    interpretation = db._conn.execute(
        "SELECT claim, entity_id, target_text, scope, interpretation_type, confidence, "
        "action_implication, evidence_node_ids, evidence_observation_ids, metadata "
        "FROM memory_interpretations"
    ).fetchone()
    evidence_bundle_id = db._conn.execute(
        "SELECT id FROM memory_evidence_bundles"
    ).fetchone()["id"]
    evidence_bundle = db.get_evidence_bundles_by_ids(
        [evidence_bundle_id]
    )[0]
    bundle_metadata = json.loads(evidence_bundle["metadata"])
    assert evidence_bundle["embedding_text"] == evidence_bundle["summary"]
    assert "[strategy]" not in evidence_bundle["summary"]
    assert "[preference_signal]" not in evidence_bundle["summary"]
    assert "Hermes recommended Slack alert routing" in evidence_bundle["summary"]
    assert "Alice explicitly prefers Slack" in evidence_bundle["summary"]
    assert set(bundle_metadata) <= {"decay"}
    assert interpretation["entity_id"] == alice
    assert interpretation["interpretation_type"] == "explicit_preference"
    assert interpretation["target_text"] == "urgent alert routing"
    assert interpretation["scope"] == "alert-workflow"
    assert interpretation["confidence"] == pytest.approx(0.88)
    assert "优先使用 Slack" in interpretation["claim"]
    assert "优先建议 Slack" in interpretation["action_implication"]
    assert json.loads(interpretation["evidence_node_ids"]) == node_ids[:2]
    evidence_observation_ids = json.loads(
        interpretation["evidence_observation_ids"]
    )
    assert evidence_observation_ids == [
        row["id"]
        for row in db._conn.execute(
            "SELECT id FROM memory_observations "
            "WHERE observation_type = 'preference_signal'"
        ).fetchall()
    ]
    metadata = json.loads(interpretation["metadata"])
    assert metadata["source"] == "interpretation_generation"
    assert metadata["evidence_shape"] == "single_observation"
    assert len(metadata["observation_ids"]) == 1
    linked_observation = db._conn.execute(
        "SELECT observation_id FROM memory_interpretation_observations "
        "WHERE interpretation_id = 1"
    ).fetchone()
    assert linked_observation["observation_id"] == metadata["observation_ids"][0]
    assert any("interpretation 生成模块" in prompt for prompt in mgr.llm_prompts)


def test_incremental_observation_update_preserves_identity_and_rebuilds_summary(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    first_node = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice implemented the alert routing fallback.",
        keywords=["alert-routing", "fallback"],
        fact_type="episodic",
        fact_kind="action",
        task_event_like=True,
    )
    db.entity_link_node(first_node, alice)
    evidence_bundle_id = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="alert-routing",
        topic_label="alert routing",
        bundle_type="entity_topic",
        summary="Temporary container summary.",
        keywords=["alerts"],
        source_node_ids=[first_node],
    )
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "observation_type": "task_progress",
                "summary": "Alice implemented the alert routing fallback.",
                "confidence": 0.82,
            }),
        ],
    )
    embedding_client = _CapturingEmbeddingClient()
    mgr._embedding_client = embedding_client

    first_ids = mgr._update_observations_for_evidence_bundles(
        [evidence_bundle_id]
    )
    assert len(first_ids) == 1
    stable_observation_id = first_ids[0]
    interpretation_id = db.memory_upsert_interpretation(
        claim="Alice is progressing the alert routing fallback.",
        entity_id=alice,
        target_text="alert routing fallback",
        scope="alert-routing",
        interpretation_type="task",
        evidence_node_ids=[first_node],
        evidence_observation_ids=[stable_observation_id],
    )
    db.memory_link_interpretation_observation(
        interpretation_id,
        stable_observation_id,
    )

    second_node = _add_memory_node(
        db,
        time_key="2026-05-01 11:00:00",
        summary="Alice verified the alert routing fallback tests.",
        keywords=["alert-routing", "tests"],
        fact_type="episodic",
        fact_kind="action",
        task_event_like=True,
    )
    db.entity_link_node(second_node, alice)
    db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="alert-routing",
        topic_label="alert routing",
        bundle_type="entity_topic",
        summary="This value must be replaced from observations.",
        keywords=["alerts"],
        source_node_ids=[first_node, second_node],
    )
    mgr._llm_outputs.append(json.dumps({
        "observation_type": "task_progress",
        "summary": (
            "Alice implemented the alert routing fallback and verified its "
            "tests."
        ),
        "confidence": 0.9,
        "change_summary": "Added verification result.",
    }))

    second_ids = mgr._update_observations_for_evidence_bundles(
        [evidence_bundle_id]
    )

    assert second_ids == [stable_observation_id]
    observations = db.get_observations_for_evidence_bundles(
        [evidence_bundle_id]
    )
    assert len(observations) == 1
    assert observations[0]["id"] == stable_observation_id
    assert observations[0]["source_node_ids"] == [first_node, second_node]
    assert observations[0]["metadata"]["revision"] == 2
    linked_observation_id = db._conn.execute(
        "SELECT observation_id FROM memory_interpretation_observations "
        "WHERE interpretation_id = ?",
        (interpretation_id,),
    ).fetchone()["observation_id"]
    assert linked_observation_id == stable_observation_id
    evidence_bundle = db.get_evidence_bundles_by_ids(
        [evidence_bundle_id]
    )[0]
    assert evidence_bundle["summary"] == (
        "Alice implemented the alert routing fallback and verified its tests."
    )
    assert evidence_bundle["embedding_text"] == evidence_bundle["summary"]
    assert embedding_client.texts[-1] == evidence_bundle["summary"]
    assert json.loads(evidence_bundle["metadata"]) == {}


def test_interpretation_generation_skips_observation_without_observations(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    node_id = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice discussed alert routing.",
        keywords=["alert-routing"],
    )
    db.entity_link_node(node_id, alice)
    evidence_bundle_id = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="alert-routing",
        topic_label="alert routing",
        bundle_type="observation",
        summary="Alice discussed alert routing.",
        keywords=["alert-routing"],
        source_node_ids=[node_id],
        metadata={},
    )
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[json.dumps({"should_create": True})],
    )

    generated = mgr._reflect_generate_interpretations_using_observations(
        [evidence_bundle_id]
    )

    assert generated == 0
    assert mgr.llm_prompts == []
    assert db._conn.execute(
        "SELECT COUNT(*) FROM memory_interpretations"
    ).fetchone()[0] == 0


def test_deferred_interpretation_context_is_loaded_from_observations(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})
    evidence_bundle_ids = []
    observation_ids = []
    for index, summary in enumerate(
        [
            "Alice is evaluating Slack alert routing.",
            "Alice is comparing Slack routing behavior.",
        ],
        1,
    ):
        node_id = _add_memory_node(
            db,
            time_key=f"2026-05-0{index} 10:00:00",
            summary=summary,
            keywords=["Slack", "routing"],
            fact_kind="context",
        )
        db.entity_link_node(node_id, alice)
        evidence_bundle_id = db.memory_upsert_evidence_bundle(
            entity_id=alice,
            topic_key=f"slack-routing-{index}",
            topic_label=f"Slack routing {index}",
            bundle_type="entity_topic",
            summary=f"parent observation text {index}",
            keywords=["parent-only-keyword"],
            source_node_ids=[node_id],
            metadata={"parent_only": True},
        )
        observation_id = db.memory_create_observation(
            evidence_bundle_id,
            {
                "observation_type": "context",
                "summary": summary,
                "evidence_mode": "semantic",
                "confidence": 0.8,
                "source_node_ids": [node_id],
                "embedding": np.asarray([1.0, 0.01 * index], dtype=np.float32),
                "embedding_text": summary,
                "metadata": {
                    "allowed_interpretation_types": ["insight"],
                    "candidate_interpretation_types": ["insight"],
                    "source_count": 1,
                },
            },
        )
        evidence_bundle_ids.append(evidence_bundle_id)
        observation_ids.append(observation_id)

    first_observation = db.get_observations_for_evidence_bundles(
        [evidence_bundle_ids[0]]
    )[0]
    first_semantic_observation = mgr._build_semantic_observation(first_observation)
    assert first_semantic_observation["summary"] == first_observation["summary"]
    assert "parent observation text" not in first_semantic_observation["summary"]
    assert "parent_only" not in first_semantic_observation["metadata"]
    first_sources = db.memory_nodes_by_ids(
        first_observation["source_node_ids"]
    )
    first_basis_hash = mgr._interpretation_basis_hash(
        first_semantic_observation,
        first_sources,
    )
    db.memory_update_observation_metadata(
        observation_ids[0],
        {
            **first_observation["metadata"],
            "interpretation_status": "deferred",
            "interpretation_basis_hash": first_basis_hash,
        },
    )

    second_observation = db.get_observations_for_evidence_bundles(
        [evidence_bundle_ids[1]]
    )[0]
    second_semantic_observation = mgr._build_semantic_observation(
        second_observation
    )
    second_sources = db.memory_nodes_by_ids(
        second_observation["source_node_ids"]
    )
    family = mgr._observation_interpretation_cluster_family(
        second_semantic_observation,
        second_sources,
    )
    deferred_items = mgr._get_similar_deferred_observations(
        {
            "observation": second_semantic_observation,
            "family": family,
        },
        {observation_ids[1]},
    )

    assert [item["observation_id"] for item in deferred_items] == [
        observation_ids[0]
    ]
    assert deferred_items[0]["evidence_bundle_id"] == evidence_bundle_ids[0]
    assert deferred_items[0]["observation"]["summary"] == (
        first_observation["summary"]
    )


def test_interpretation_linker_reuses_existing_observation_evidence_without_llm(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    first = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Slack alerts"],
        fact_kind="preference",
    )
    second = _add_memory_node(
        db,
        time_key="2026-05-02 10:00:00",
        summary="Alice reiterated that urgent alerts should use Slack.",
        keywords=["Slack alerts"],
        fact_kind="preference",
    )
    db.entity_link_node(first, alice)
    db.entity_link_node(second, alice)
    observation_id = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="slack-alerts",
        topic_label="Slack alerts",
        bundle_type="observation",
        summary="Alice's urgent alert workflow is Slack-centered.",
        keywords=["Slack", "alerts"],
        source_node_ids=[first, second],
        metadata={"observation_kind": "context"},
    )
    interpretation_id = db.memory_upsert_interpretation(
        claim="Agent 当前解释为 Alice 的紧急告警协作应优先使用 Slack。",
        target_text="urgent alert routing",
        scope="slack-alerts",
        interpretation_type="insight",
        confidence=0.88,
        action_implication="后续涉及 Alice 的紧急告警时优先建议 Slack 路由。",
        evidence_node_ids=[first],
        evidence_observation_ids=[observation_id],
        metadata={"observation_id": observation_id},
    )
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[json.dumps({"should_create": True})],
    )

    generated = mgr._generate_interpretations_using_observations([observation_id])

    assert generated == 1
    assert mgr.llm_prompts == []
    row = db._conn.execute(
        "SELECT evidence_node_ids, evidence_observation_ids, metadata "
        "FROM memory_interpretations WHERE id = ?",
        (interpretation_id,),
    ).fetchone()
    assert json.loads(row["evidence_node_ids"]) == [first, second]
    assert json.loads(row["evidence_observation_ids"]) == [observation_id]
    assert json.loads(row["metadata"])["cheap_linker"]["last_match_reason"] == "existing_observation_evidence"


def test_interpretation_linker_matches_preference_by_entity_topic_without_llm(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    source = _add_memory_node(
        db,
        time_key="2026-05-03 10:00:00",
        summary="Alice repeatedly prefers discussing architecture before implementation.",
        keywords=["architecture planning"],
        fact_kind="preference",
    )
    db.entity_link_node(source, alice)
    observation_id = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="architecture-planning",
        topic_label="architecture planning",
        bundle_type="observation",
        summary="Alice has repeatedly preferred architecture discussion before implementation.",
        keywords=["architecture", "planning"],
        source_node_ids=[source],
        metadata={"observation_kind": "event_pattern"},
    )
    interpretation_id = db.memory_upsert_interpretation(
        claim="Alice prefers architecture discussion before implementation on complex coding work.",
        target_text="architecture planning",
        scope="architecture-planning",
        interpretation_type="explicit_preference",
        confidence=0.84,
        action_implication="Start complex implementation requests with a brief design pass when appropriate.",
        metadata={
            "entity_id": alice,
            "entity_name": "Alice",
            "topic_key": "architecture-planning",
            "topic_label": "architecture planning",
        },
    )
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "should_update": True,
                "claim": "Alice has a reinforced preference for architecture discussion before implementation.",
                "target_text": "architecture planning",
                "scope": "architecture-planning",
                "interpretation_type": "explicit_preference",
                "polarity": "positive",
                "strength": 0.88,
                "confidence": 0.9,
                "status": "current",
                "conflict_status": "none",
                "action_implication": "Start complex implementation requests with architecture discussion before editing code.",
                "evidence_node_ids": [source],
                "evidence_observation_ids": [observation_id],
                "metadata": {"source": "interpretation_update", "preference_domain": "coding-workflow"},
            })
        ],
    )

    generated = mgr._generate_interpretations_using_observations([observation_id])

    assert generated == 1
    assert len([prompt for prompt in mgr.llm_prompts if "interpretation 更新模块" in prompt]) == 1
    row = db._conn.execute(
        "SELECT claim, action_implication, confidence, evidence_node_ids, evidence_observation_ids, metadata "
        "FROM memory_interpretations WHERE id = ?",
        (interpretation_id,),
    ).fetchone()
    assert "reinforced preference" in row["claim"]
    assert "before editing code" in row["action_implication"]
    assert row["confidence"] == pytest.approx(0.9)
    assert json.loads(row["evidence_node_ids"]) == [source]
    assert json.loads(row["evidence_observation_ids"]) == [observation_id]
    metadata = json.loads(row["metadata"])
    assert metadata["cheap_linker"]["last_match_score"] >= 0.72
    assert "preference_evidence" in metadata["cheap_linker"]["last_match_reason"]
    assert metadata["cheap_linker"]["content_updated"] is True
    assert metadata["source"] == "interpretation_update"
    assert metadata["preference_domain"] == "coding-workflow"


def test_interpretation_linker_respects_observation_candidate_type_gate(db):
    hermes = db.entity_add_entity("Hermes Agent", "PROJECT")
    source = _add_memory_node(
        db,
        time_key="2026-05-04 10:00:00",
        summary="用户要求实现 observation 与 interpretation 的 task 匹配逻辑。",
        keywords=["memory matching"],
        fact_kind="request",
        task_event_like=True,
        task_event_subject="user",
        task_relevance="strong",
    )
    db.entity_link_node(source, hermes)
    observation_id = db.memory_upsert_evidence_bundle(
        entity_id=hermes,
        topic_key="memory-matching",
        topic_label="memory matching",
        bundle_type="observation",
        summary="用户正在推进 observation 与 interpretation 的匹配实现任务。",
        keywords=["memory", "matching"],
        source_node_ids=[source],
        metadata={
            "observation_kind": "task_signal",
            "candidate_interpretation_types": ["task"],
        },
    )
    preference_id = db.memory_upsert_interpretation(
        claim="用户偏好围绕 memory matching 先讨论方案再实现。",
        target_text="memory matching",
        scope="memory-matching",
        interpretation_type="explicit_preference",
        confidence=0.94,
        action_implication="处理 memory matching 改动时先讨论方案。",
        metadata={
            "entity_id": hermes,
            "entity_name": "Hermes Agent",
            "topic_key": "memory-matching",
            "topic_label": "memory matching",
        },
    )
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})
    source_nodes = db.get_evidence_bundle_supporting_nodes([observation_id])[observation_id]

    linked = mgr._link_observation_to_existing_interpretation(
        observation=dict(db._conn.execute("SELECT * FROM memory_evidence_bundles WHERE id = ?", (observation_id,)).fetchone()),
        source_nodes=source_nodes,
        observation_id=observation_id,
        source_node_ids=[source],
    )

    assert linked is None
    metadata = json.loads(
        db._conn.execute(
            "SELECT metadata FROM memory_interpretations WHERE id = ?",
            (preference_id,),
        ).fetchone()["metadata"]
    )
    assert "cheap_linker" not in metadata


def test_interpretation_generation_clusters_unmatched_observations(db):
    hermes = db.entity_add_entity("Hermes Agent", "PROJECT")
    first_source = _add_memory_node(
        db,
        time_key="2026-05-05 10:00:00",
        summary="用户要求设计 observation 与 interpretation 的匹配策略。",
        keywords=["memory interpretation"],
        fact_kind="request",
        task_event_like=True,
        task_event_subject="user",
        task_relevance="strong",
    )
    second_source = _add_memory_node(
        db,
        time_key="2026-05-05 11:00:00",
        summary="助手开始实现 observation 与 interpretation 的 cheap matcher。",
        keywords=["memory interpretation"],
        fact_kind="action",
        task_event_like=True,
        task_event_subject="assistant",
        task_relevance="strong",
    )
    db.entity_link_node(first_source, hermes)
    db.entity_link_node(second_source, hermes)
    first_observation = db.memory_upsert_evidence_bundle(
        entity_id=hermes,
        topic_key="memory-interpretation",
        topic_label="memory interpretation",
        bundle_type="observation",
        summary="用户要求设计 observation 与 interpretation 的匹配策略。",
        keywords=["memory", "interpretation"],
        source_node_ids=[first_source],
        metadata={
            "observation_kind": "task_signal",
            "evidence_shape": "single_event",
            "temporal_scope": "recent",
            "candidate_interpretation_types": ["task"],
        },
    )
    second_observation = db.memory_upsert_evidence_bundle(
        entity_id=hermes,
        topic_key="memory-implementation",
        topic_label="memory implementation",
        bundle_type="observation",
        summary="助手开始实现 observation 与 interpretation 的 cheap matcher。",
        keywords=["memory", "interpretation"],
        source_node_ids=[second_source],
        metadata={
            "observation_kind": "state_change",
            "evidence_shape": "progression",
            "temporal_scope": "recent",
            "candidate_interpretation_types": ["task"],
        },
    )
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "should_create": True,
                "claim": "用户当前正在推进 observation 与 interpretation 的匹配实现。",
                "target_text": "observation interpretation matching",
                "scope": "memory-interpretation",
                "interpretation_type": "task",
                "polarity": "neutral",
                "strength": 0.8,
                "confidence": 0.82,
                "status": "current",
                "conflict_status": "none",
                "action_implication": "后续 memory interpretation 工作应保留该任务上下文。",
                "evidence_node_ids": [first_source, second_source],
                "evidence_observation_ids": [first_observation, second_observation],
                "metadata": {
                    "task_status": "active",
                    "goal": "实现 observation 与 interpretation 的匹配逻辑。",
                    "steps": [{"title": "实现 cheap matcher", "status": "active"}],
                },
            })
        ],
    )

    generated = mgr._generate_interpretations_using_observations([first_observation, second_observation])

    assert generated == 1
    interpretation_prompts = [
        prompt for prompt in mgr.llm_prompts if "interpretation 生成模块" in prompt
    ]
    assert len(interpretation_prompts) == 1
    assert "Clustered observations" in interpretation_prompts[0]
    assert f"id={first_observation}" in interpretation_prompts[0]
    assert f"id={second_observation}" in interpretation_prompts[0]
    row = db._conn.execute(
        "SELECT interpretation_type, evidence_node_ids, evidence_observation_ids, metadata "
        "FROM memory_interpretations"
    ).fetchone()
    assert row["interpretation_type"] == "task"
    assert json.loads(row["evidence_node_ids"]) == [first_source, second_source]
    assert json.loads(row["evidence_observation_ids"]) == [first_observation, second_observation]
    metadata = json.loads(row["metadata"])
    assert metadata["observation_ids"] == [first_observation, second_observation]
    assert metadata["interpretation_cluster_family"] == "task"


def test_interpretation_generation_defers_weak_single_observation(db):
    hermes = db.entity_add_entity("Hermes Agent", "PROJECT")
    source = _add_memory_node(
        db,
        time_key="2026-05-06 10:00:00",
        summary="Hermes memory recall discussion mentioned indexing context.",
        keywords=["memory", "indexing"],
        fact_kind="context",
    )
    db.entity_link_node(source, hermes)
    observation_id = db.memory_upsert_evidence_bundle(
        entity_id=hermes,
        topic_key="memory-indexing",
        topic_label="memory indexing",
        bundle_type="observation",
        summary="Hermes memory recall discussion has indexing context.",
        keywords=["memory", "indexing"],
        source_node_ids=[source],
        metadata={
            "observation_kind": "context",
            "evidence_shape": "single_event",
            "temporal_scope": "recent",
            "candidate_interpretation_types": ["insight"],
        },
    )
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})

    generated = mgr._generate_interpretations_using_observations([observation_id])

    assert generated == 0
    assert mgr.llm_prompts == []
    metadata = json.loads(
        db._conn.execute(
            "SELECT metadata FROM memory_evidence_bundles WHERE id = ?",
            (observation_id,),
        ).fetchone()["metadata"]
    )
    assert metadata["interpretation_status"] == "deferred"
    assert metadata["interpretation_reason"] == "single_fact_insight_signal"
    assert metadata["interpretation_basis_hash"]


def test_interpretation_generation_defers_single_fact_insight_before_llm(db):
    hermes = db.entity_add_entity("Hermes Agent", "PROJECT")
    source = _add_memory_node(
        db,
        time_key="2026-05-06 10:00:00",
        summary="Hermes memory system changed one recall label.",
        keywords=["memory", "recall"],
        fact_kind="context",
    )
    db.entity_link_node(source, hermes)
    observation_id = db.memory_upsert_evidence_bundle(
        entity_id=hermes,
        topic_key="memory-recall",
        topic_label="memory recall",
        bundle_type="observation",
        summary="Hermes memory system changed one recall label.",
        keywords=["memory", "recall"],
        source_node_ids=[source],
        metadata={
            "observation_kind": "state_change",
            "evidence_shape": "progression",
            "temporal_scope": "recent",
            "candidate_interpretation_types": ["insight"],
        },
    )
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "should_create": True,
                "claim": "Hermes recall labels are now a stable design signal.",
                "target_text": "memory recall labels",
                "scope": "memory-recall",
                "interpretation_type": "insight",
                "polarity": "neutral",
                "strength": 0.8,
                "confidence": 0.88,
                "status": "current",
                "conflict_status": "none",
                "action_implication": "Use recall label changes as a stable design signal.",
                "evidence_node_ids": [source],
                "evidence_observation_ids": [observation_id],
                "counter_evidence_node_ids": [],
                "counter_evidence_observation_ids": [],
            })
        ],
    )

    generated = mgr._generate_interpretations_using_observations([observation_id])

    assert generated == 0
    assert mgr.llm_prompts == []
    assert db._conn.execute("SELECT COUNT(*) AS count FROM memory_interpretations").fetchone()["count"] == 0
    metadata = json.loads(
        db._conn.execute(
            "SELECT metadata FROM memory_evidence_bundles WHERE id = ?",
            (observation_id,),
        ).fetchone()["metadata"]
    )
    assert metadata["interpretation_status"] == "deferred"
    assert metadata["interpretation_reason"] == "single_fact_insight_signal"


def test_interpretation_generation_does_not_use_global_batch_threshold_for_weak_singletons(db):
    observation_ids = []
    for index in range(3):
        entity = db.entity_add_entity(f"Hermes Area {index}", "PROJECT")
        source = _add_memory_node(
            db,
            time_key=f"2026-05-06 1{index}:00:00",
            summary=f"Hermes area {index} discussion mentioned ordinary context.",
            keywords=[f"area-{index}", "ordinary"],
            fact_kind="context",
        )
        db.entity_link_node(source, entity)
        observation_ids.append(
            db.memory_upsert_evidence_bundle(
                entity_id=entity,
                topic_key=f"ordinary-context-{index}",
                topic_label=f"ordinary context {index}",
                bundle_type="observation",
                summary=f"Hermes area {index} has ordinary context.",
                keywords=[f"area-{index}", "ordinary"],
                source_node_ids=[source],
                metadata={
                    "observation_kind": "context",
                    "evidence_shape": "single_event",
                    "temporal_scope": "recent",
                    "candidate_interpretation_types": ["insight"],
                },
            )
        )
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})

    generated = mgr._generate_interpretations_using_observations(observation_ids)

    assert generated == 0
    assert mgr.llm_prompts == []
    rows = db._conn.execute(
        "SELECT metadata FROM memory_evidence_bundles WHERE id IN (?, ?, ?) ORDER BY id",
        tuple(observation_ids),
    ).fetchall()
    for row in rows:
        metadata = json.loads(row["metadata"])
        assert metadata["interpretation_status"] == "deferred"
        assert metadata["interpretation_reason"] == "single_fact_insight_signal"


def test_interpretation_generation_skips_final_observation_when_basis_unchanged(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    source = _add_memory_node(
        db,
        time_key="2026-05-07 10:00:00",
        summary="Alice prefers brief architecture notes before code changes.",
        keywords=["architecture", "brief"],
        fact_kind="preference",
    )
    db.entity_link_node(source, alice)
    observation_id = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="architecture-notes",
        topic_label="architecture notes",
        bundle_type="observation",
        summary="Alice prefers brief architecture notes before code changes.",
        keywords=["architecture", "brief"],
        source_node_ids=[source],
        metadata={
            "observation_kind": "preference_signal",
            "evidence_shape": "single_event",
            "temporal_scope": "ongoing",
            "candidate_interpretation_types": ["preference"],
        },
    )
    interpretation_id = db.memory_upsert_interpretation(
        claim="Alice prefers brief architecture notes before code changes.",
        target_text="architecture notes",
        scope="architecture-notes",
        interpretation_type="explicit_preference",
        confidence=0.86,
        action_implication="Start complex code changes with brief architecture notes.",
        evidence_node_ids=[source],
        evidence_observation_ids=[observation_id],
        metadata={"entity_id": alice, "topic_key": "architecture-notes"},
    )
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})

    first = mgr._generate_interpretations_using_observations([observation_id])
    second = mgr._generate_interpretations_using_observations([observation_id])

    assert first == 1
    assert second == 0
    assert mgr.llm_prompts == []
    row = db._conn.execute(
        "SELECT metadata FROM memory_evidence_bundles WHERE id = ?",
        (observation_id,),
    ).fetchone()
    metadata = json.loads(row["metadata"])
    assert metadata["interpretation_status"] == "linked"
    assert metadata["linked_interpretation_ids"] == [interpretation_id]


def test_interpretation_generation_reuses_deferred_observation_in_new_cluster(db):
    hermes = db.entity_add_entity("Hermes Agent", "PROJECT")
    first_source = _add_memory_node(
        db,
        time_key="2026-05-08 10:00:00",
        summary="Hermes memory recall discussion mentioned indexing context.",
        keywords=["memory", "indexing"],
        fact_kind="context",
    )
    second_source = _add_memory_node(
        db,
        time_key="2026-05-08 11:00:00",
        summary="Hermes memory recall discussion later connected indexing context to recall quality.",
        keywords=["memory", "indexing"],
        fact_kind="context",
    )
    db.entity_link_node(first_source, hermes)
    db.entity_link_node(second_source, hermes)
    first_observation = db.memory_upsert_evidence_bundle(
        entity_id=hermes,
        topic_key="memory-indexing",
        topic_label="memory indexing",
        bundle_type="observation",
        summary="Hermes memory recall discussion has indexing context.",
        keywords=["memory", "indexing"],
        source_node_ids=[first_source],
        metadata={
            "observation_kind": "context",
            "evidence_shape": "single_event",
            "temporal_scope": "recent",
            "candidate_interpretation_types": ["insight"],
        },
    )
    first_mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})
    assert first_mgr._generate_interpretations_using_observations([first_observation]) == 0

    second_observation = db.memory_upsert_evidence_bundle(
        entity_id=hermes,
        topic_key="memory-indexing",
        topic_label="memory indexing followup",
        bundle_type="observation_followup",
        summary="Hermes memory recall discussion connected indexing context to recall quality.",
        keywords=["memory", "indexing", "recall"],
        source_node_ids=[second_source],
        metadata={
            "observation_kind": "context",
            "evidence_shape": "single_event",
            "temporal_scope": "recent",
            "candidate_interpretation_types": ["insight"],
        },
    )
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "should_create": True,
                "claim": "Hermes memory recall quality is currently tied to indexing context.",
                "target_text": "memory indexing",
                "scope": "memory-indexing",
                "interpretation_type": "insight",
                "polarity": "neutral",
                "strength": 0.72,
                "confidence": 0.76,
                "status": "current",
                "conflict_status": "none",
                "action_implication": "Use indexing context when reasoning about memory recall quality.",
                "evidence_node_ids": [first_source, second_source],
                "evidence_observation_ids": [second_observation, first_observation],
                "counter_evidence_node_ids": [],
                "counter_evidence_observation_ids": [],
            })
        ],
    )

    generated = mgr._generate_interpretations_using_observations([second_observation])

    assert generated == 1
    prompt = next(prompt for prompt in mgr.llm_prompts if "interpretation 生成模块" in prompt)
    assert "Clustered observations" in prompt
    assert f"id={first_observation}" in prompt
    assert f"id={second_observation}" in prompt
    rows = db._conn.execute(
        "SELECT id, metadata FROM memory_evidence_bundles WHERE id IN (?, ?) ORDER BY id",
        (first_observation, second_observation),
    ).fetchall()
    metadata_by_id = {row["id"]: json.loads(row["metadata"]) for row in rows}
    assert metadata_by_id[first_observation]["interpretation_status"] == "generated"
    assert metadata_by_id[second_observation]["interpretation_status"] == "generated"


def test_store_turn_can_consolidate_task_observation(db):
    retain_payload = {
        "facts": [
            {
                "text": "用户正在排查 Hermes memory recall 的匹配问题。",
                "keywords": ["Hermes", "memory", "recall", "排查"],
                "topic": ["memory recall"],
                "fact_type": "semantic",
                "fact_kind": "action",
                "entities": [{"name": "Hermes Agent", "type": "PROJECT"}],
            },
            {
                "text": "用户计划修改 observation 生成逻辑以区分 insight 和 task。",
                "keywords": ["observation", "insight", "task", "修改"],
                "topic": ["memory recall"],
                "fact_type": "semantic",
                "fact_kind": "action",
                "entities": [{"name": "Hermes Agent", "type": "PROJECT"}],
            },
            {
                "text": "助手帮助用户实现记忆系统 reflect 和 observation decay 相关改动。",
                "keywords": ["reflect", "observation", "decay", "实现"],
                "topic": ["memory recall"],
                "fact_type": "episodic",
                "fact_kind": "action",
                "entities": [{"name": "Hermes Agent", "type": "PROJECT"}],
            },
        ],
        "causal_relations": [],
    }
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps(retain_payload),
            json.dumps({
                "category": "observation",
                "summary": "用户正在迭代 Hermes Agent 的记忆系统，当前聚焦 recall、reflect 与 observation 生成逻辑。",
                "keywords": ["Hermes Agent", "memory", "recall", "observation"],
                "confidence": 0.82,
                "metadata": {
                    "observation_kind": "timeline",
                    "has_conflict": False,
                },
            }),
            json.dumps({
                "should_create": True,
                "claim": "用户当前正在迭代 Hermes Agent 的记忆系统。",
                "target_text": "Hermes Agent memory system",
                "scope": "memory-recall",
                "interpretation_type": "task",
                "polarity": "neutral",
                "strength": 0.82,
                "confidence": 0.82,
                "status": "current",
                "conflict_status": "none",
                "action_implication": "后续围绕 Hermes Agent 记忆系统改动时应保留当前任务上下文。",
                "evidence_node_ids": [],
                "evidence_observation_ids": [1],
                "metadata": {
                    "task_status": "active",
                    "task_source": "inferred_from_interpretation",
                    "goal": "完善 Hermes Agent 的长期记忆系统。",
                    "evidence": ["排查 memory recall", "修改 observation 生成逻辑"],
                    "steps": [
                        {
                            "title": "排查 memory recall 匹配问题",
                            "status": "done",
                            "evidence": ["排查 memory recall"],
                            "updated_at": "2026-05-14",
                        },
                        {
                            "title": "将 observation 分类为 insight 和 task",
                            "status": "active",
                            "evidence": ["修改 observation 生成逻辑"],
                        },
                    ],
                    "next_action": "继续验证 observation 分类逻辑",
                },
            }),
        ],
    )

    assert mgr.store_turn("继续改记忆系统", "我们来调整 observation 分类。") is True
    assert db._conn.execute("SELECT COUNT(*) FROM memory_evidence_bundles").fetchone()[0] == 0

    report = mgr.reflect(limit=10)
    assert report["evidence_bundles_consolidated"] == 1

    observation = db._conn.execute(
        "SELECT observation_type, summary, metadata FROM memory_evidence_bundles"
    ).fetchone()
    metadata = json.loads(observation["metadata"])
    assert observation["observation_type"] == "observation"
    assert "正在迭代 Hermes Agent" in observation["summary"]
    assert metadata["observation_kind"] == "event_cluster"
    assert metadata["evidence_shape"] == "progression"
    assert metadata["temporal_scope"] == "recent"
    assert metadata["candidate_interpretation_types"] == ["insight", "task"]
    assert metadata["source_fact_type_distribution"] == {"semantic": 2, "episodic": 1}
    assert metadata["dominant_fact_type"] == "semantic"
    assert metadata["evidence_mixture"] == "semantic_dominant"
    interpretation = db._conn.execute(
        "SELECT interpretation_type, claim, metadata FROM memory_interpretations"
    ).fetchone()
    interpretation_metadata = json.loads(interpretation["metadata"])
    assert interpretation["interpretation_type"] == "task"
    assert "正在迭代 Hermes Agent" in interpretation["claim"]
    assert interpretation_metadata["task_status"] == "active"
    assert interpretation_metadata["task_source"] == "inferred_from_interpretation"
    assert interpretation_metadata["goal"] == "完善 Hermes Agent 的长期记忆系统。"
    assert interpretation_metadata["steps"][0]["title"] == "排查 memory recall 匹配问题"


def test_reflect_updates_observation_by_entity_and_topic(db):
    hermes = db.entity_add_entity("Hermes Agent", "PROJECT")
    source_node = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="用户正在完善 Hermes Agent 的 reflect 机制。",
        keywords=["memory-reflect"],
    )
    db.entity_link_node(source_node, hermes)
    task_id, observation_id = _add_task_interpretation(
        db,
        entity_id=hermes,
        topic_key="memory-reflect",
        topic_label="memory-reflect",
        summary="用户正在完善 Hermes Agent 的长期记忆 reflect 机制。",
        keywords=["Hermes Agent", "memory", "reflect"],
        source_node_ids=[source_node],
        metadata={
            "steps": [{"title": "设计 reflect 机制", "status": "active"}],
        },
    )
    new_node = _add_memory_node(
        db,
        time_key=MemoryNodeManager._memory_time_key(0),
        summary="用户要求在 run_agent 中每 5 轮调用 reflect。",
        keywords=["memory-reflect"],
        fact_kind="request",
    )
    db.entity_link_node(new_node, hermes)
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "category": "observation",
                "summary": "用户正在完善 Hermes Agent 的 reflect 调度机制。",
                "keywords": ["Hermes Agent", "reflect"],
                "confidence": 0.9,
                "metadata": {
                    "observation_kind": "timeline",
                    "has_conflict": False,
                },
            })
        ],
    )

    report = mgr.reflect(limit=10)

    assert report["evidence_bundle_reflect"].get("task_matched", 0) == 0
    assert report["evidence_bundle_reflect"]["fact_evidence_bundle_matches"] == 1
    assert report["evidence_bundle_reflect"]["fact_evidence_bundle_node_count"] == 1
    assert report["evidence_bundle_reflect"]["entity_topic_updates"] == 0
    assert report["evidence_bundle_reflect"]["entity_topic_node_count"] == 1
    assert db.memory_evidence_bundle_source_ids(observation_id) == [source_node, new_node]
    row = db._conn.execute(
        "SELECT summary, metadata FROM memory_evidence_bundles WHERE id = ?",
        (observation_id,),
    ).fetchone()
    assert "reflect 调度机制" in row["summary"]

    task_row = db._conn.execute(
        "SELECT metadata FROM memory_interpretations WHERE id = ?",
        (task_id,),
    ).fetchone()
    metadata = json.loads(task_row["metadata"])
    assert metadata["task_status"] == "active"


def test_reflect_excludes_non_task_fact_kind_from_existing_task_match(db):
    hermes = db.entity_add_entity("Hermes Agent", "PROJECT")
    source_node = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="用户正在完善 Hermes Agent 的 reflect 机制。",
        keywords=["memory-reflect"],
        fact_kind="action",
    )
    db.entity_link_node(source_node, hermes)
    task_id, observation_id = _add_task_interpretation(
        db,
        entity_id=hermes,
        topic_key="memory-reflect",
        topic_label="memory-reflect",
        summary="用户正在完善 Hermes Agent 的长期记忆 reflect 机制。",
        keywords=["Hermes Agent", "memory", "reflect"],
        source_node_ids=[source_node],
    )
    new_node = _add_memory_node(
        db,
        time_key=MemoryNodeManager._memory_time_key(0),
        summary="用户要求记录自己偏好讨论 reflect 调度机制。",
        keywords=["memory-reflect"],
        fact_kind="preference",
        task_event_like=True,
        task_event_subject="user",
        task_relevance="strong",
    )
    db.entity_link_node(new_node, hermes)
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "category": "observation",
                "summary": "用户偏好讨论 reflect 调度机制，但这不是 task 进展。",
                "keywords": ["memory-reflect", "preference"],
                "confidence": 0.78,
                "metadata": {"observation_kind": "context"},
            }),
            json.dumps({"should_create": False}),
        ],
    )

    report = mgr.reflect(limit=10)

    assert report["evidence_bundle_reflect"].get("task_matched", 0) == 0
    assert report["evidence_bundle_reflect"]["fact_evidence_bundle_matches"] == 0
    assert report["evidence_bundle_reflect"]["fact_clusters_consolidated"] == 0
    assert db.memory_evidence_bundle_source_ids(observation_id) == [source_node]
    assert mgr.llm_prompts == []


def test_reflect_no_longer_matches_observation_by_high_task_embedding_similarity(db):
    hermes = db.entity_add_entity("Hermes Agent", "PROJECT")
    alice = db.entity_add_entity("Alice", "PERSON")
    source_node = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="用户正在设计 memory reflect 机制。",
        keywords=["memory-system"],
    )
    db.entity_link_node(source_node, hermes)
    task_id, observation_id = _add_task_interpretation(
        db,
        entity_id=hermes,
        topic_key="memory-system",
        topic_label="memory-system",
        summary="用户正在完善 Hermes Agent memory reflect 任务。",
        keywords=["memory", "reflect"],
        source_node_ids=[source_node],
    )
    new_node = _add_memory_node(
        db,
        time_key=MemoryNodeManager._memory_time_key(0),
        summary="用户继续讨论 reflect 的低成本 task matching 方案。",
        keywords=["unrelated-topic"],
        fact_kind="action",
    )
    db.entity_link_node(new_node, alice)
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "category": "observation",
                "summary": "用户正在完善 memory reflect 的 task matching 方案。",
                "keywords": ["memory", "reflect", "task matching"],
                "confidence": 0.88,
                "metadata": {
                    "observation_kind": "timeline",
                    "has_conflict": False,
                },
            })
        ],
    )
    mgr._embedding_client = _KeywordEmbeddingClient()

    report = mgr.reflect(limit=10)

    assert report["evidence_bundle_reflect"].get("task_matched", 0) == 0
    assert report["evidence_bundle_reflect"]["entity_topic_updates"] == 0
    assert db.memory_evidence_bundle_source_ids(observation_id) == [source_node]


def test_reflect_no_longer_matches_fact_to_single_recent_active_task(db):
    hermes = db.entity_add_entity("Hermes Agent", "PROJECT")
    alice = db.entity_add_entity("Alice", "PERSON")
    source_node = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="用户正在完善 Hermes Agent 的记忆系统。",
        keywords=["memory-system"],
    )
    db.entity_link_node(source_node, hermes)
    task_id, observation_id = _add_task_interpretation(
        db,
        entity_id=hermes,
        topic_key="memory-system",
        topic_label="memory-system",
        summary="用户正在完善 Hermes Agent 的记忆系统。",
        keywords=["memory"],
        source_node_ids=[source_node],
    )
    new_node = _add_memory_node(
        db,
        time_key=MemoryNodeManager._memory_time_key(0),
        summary="用户要求继续修改 prompt。",
        keywords=["prompt-work"],
        fact_kind="request",
    )
    db.entity_link_node(new_node, alice)
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "category": "observation",
                "summary": "用户正在继续完善记忆系统 prompt。",
                "keywords": ["memory", "prompt"],
                "confidence": 0.78,
                "metadata": {
                    "observation_kind": "timeline",
                    "has_conflict": False,
                },
            })
        ],
    )
    mgr._embedding_client = _OrthogonalTaskEmbeddingClient()

    report = mgr.reflect(limit=10)

    assert report["evidence_bundle_reflect"].get("task_matched", 0) == 0
    assert report["evidence_bundle_reflect"]["entity_topic_updates"] == 0
    assert db.memory_evidence_bundle_source_ids(observation_id) == [source_node]


def test_reflect_leaves_unmatched_non_action_fact_for_regular_observation_flow(db):
    hermes = db.entity_add_entity("Hermes Agent", "PROJECT")
    zhang = db.entity_add_entity("小张父亲", "PERSON")
    source_node = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="用户正在完善 Hermes Agent 的记忆系统。",
        keywords=["memory-system"],
    )
    db.entity_link_node(source_node, hermes)
    task_id, observation_id = _add_task_interpretation(
        db,
        entity_id=hermes,
        topic_key="memory-system",
        topic_label="memory-system",
        summary="用户正在完善 Hermes Agent 的记忆系统。",
        keywords=["memory"],
        source_node_ids=[source_node],
    )
    new_node = _add_memory_node(
        db,
        time_key=MemoryNodeManager._memory_time_key(0),
        summary="小张父亲是中学的校长。",
        keywords=["家庭"],
    )
    db.entity_link_node(new_node, zhang)
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})
    mgr._embedding_client = _OrthogonalTaskEmbeddingClient()

    report = mgr.reflect(limit=10)

    assert report["evidence_bundle_reflect"].get("task_matched", 0) == 0
    assert db.memory_evidence_bundle_source_ids(observation_id) == [source_node]
    assert mgr.llm_prompts == []


def test_reflect_updates_existing_observation_by_entity_topic(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    bob = db.entity_add_entity("Bob", "PERSON")
    old_node = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice previously routed alerts through Slack.",
        keywords=["alert-routing"],
        fact_kind="action",
    )
    db.entity_link_node(old_node, alice)
    observation_id = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="alert-routing",
        topic_label="alert-routing",
        bundle_type="observation",
        summary="Alice has an alert routing history.",
        keywords=["alert-routing"],
        source_node_ids=[old_node],
        metadata={"observation_kind": "timeline"},
    )
    new_matching_node = _add_memory_node(
        db,
        time_key=MemoryNodeManager._memory_time_key(0),
        summary="Alice continued refining alert routing today.",
        keywords=["alert-routing"],
        fact_kind="action",
        task_event_like=True,
        task_event_subject="user",
        task_relevance="strong",
    )
    unrelated_task_node = _add_memory_node(
        db,
        time_key=MemoryNodeManager._memory_time_key(1),
        summary="Bob also discussed alert routing implementation today.",
        keywords=["alert-routing"],
        fact_kind="action",
        task_event_like=True,
        task_event_subject="user",
        task_relevance="strong",
    )
    db.entity_link_node(new_matching_node, alice)
    db.entity_link_node(unrelated_task_node, bob)
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "category": "observation",
                "summary": "Alice has continued refining alert routing over time.",
                "keywords": ["alert-routing", "Slack"],
                "confidence": 0.88,
                "metadata": {"observation_kind": "timeline"},
            }),
            json.dumps({"should_create": False}),
        ],
    )
    mgr._embedding_client = _OrthogonalTaskEmbeddingClient()

    report = mgr.reflect(limit=10)

    assert report["evidence_bundle_reflect"]["fact_evidence_bundle_matches"] == 1
    assert report["evidence_bundle_reflect"]["fact_evidence_bundle_node_count"] == 1
    assert report["evidence_bundle_reflect"]["entity_topic_updates"] == 0
    assert report["evidence_bundle_reflect"]["entity_topic_node_count"] == 1
    assert db.memory_evidence_bundle_source_ids(observation_id) == [old_node, new_matching_node]
    assert unrelated_task_node not in db.find_bundled_source_node_ids(
        [unrelated_task_node]
    )
    row = db._conn.execute(
        "SELECT summary, metadata FROM memory_evidence_bundles WHERE id = ?",
        (observation_id,),
    ).fetchone()
    assert row["summary"] == "Alice has continued refining alert routing over time."
    metadata = json.loads(row["metadata"])
    assert metadata["source_fact_type_distribution"] == {"semantic": 2, "episodic": 0}
    assert metadata["dominant_fact_type"] == "semantic"
    assert metadata["evidence_mixture"] == "semantic_only"


def test_reflect_clusters_facts_before_updating_existing_observation(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    old_node = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice initially routed urgent alerts through Slack.",
        keywords=["alert-routing"],
        fact_kind="action",
    )
    db.entity_link_node(old_node, alice)
    observation_id = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="alert-routing",
        topic_label="alert-routing",
        bundle_type="observation",
        summary="Alice has an established alert routing workflow.",
        keywords=["alert-routing", "Slack"],
        source_node_ids=[old_node],
        confidence=0.82,
        metadata={"observation_kind": "event_cluster"},
    )
    new_nodes = []
    for index, summary in enumerate(
        [
            "Alice refined the Slack escalation order for urgent alerts.",
            "Alice added a fallback channel to the urgent alert routing workflow.",
        ],
    ):
        node_id = _add_memory_node(
            db,
            time_key=MemoryNodeManager._memory_time_key(index),
            summary=summary,
            keywords=["alert-routing"],
            fact_kind="action",
        )
        db.entity_link_node(node_id, alice)
        new_nodes.append(node_id)
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "category": "observation",
                "summary": "Alice progressively refined the urgent alert routing workflow.",
                "keywords": ["alert-routing", "Slack", "fallback"],
                "confidence": 0.9,
                "metadata": {
                    "observation_kind": "event_cluster",
                    "evidence_shape": "progression",
                },
            }),
        ],
    )

    report = mgr.reflect(limit=10)

    evidence_bundle_report = report["evidence_bundle_reflect"]
    assert evidence_bundle_report["fact_cluster_evidence_bundle_matches"] == 1
    assert evidence_bundle_report["fact_cluster_evidence_bundle_node_count"] == 2
    assert evidence_bundle_report["fact_evidence_bundle_matches"] == 1
    assert evidence_bundle_report["fact_evidence_bundle_node_count"] == 2
    assert evidence_bundle_report["fact_clusters_consolidated"] == 0
    assert db.memory_evidence_bundle_source_ids(observation_id) == [old_node, *new_nodes]
    source_roles = {
        row["node_id"]: row["role"]
        for row in db._conn.execute(
            "SELECT node_id, role FROM memory_evidence_bundle_sources WHERE observation_id = ?",
            (observation_id,),
        ).fetchall()
    }
    assert source_roles == {
        old_node: "initial",
        new_nodes[0]: "matched",
        new_nodes[1]: "matched",
    }
    observation_prompts = [
        prompt
        for prompt in mgr.llm_prompts
        if "observation consolidation 模块" in prompt
    ]
    assert observation_prompts == []
    row = db._conn.execute(
        "SELECT metadata FROM memory_evidence_bundles WHERE id = ?",
        (observation_id,),
    ).fetchone()
    assert set(json.loads(row["metadata"])) <= {"decay"}


def test_fact_cluster_candidates_always_include_exact_entity_topic_observation(db, monkeypatch):
    alice = db.entity_add_entity("Alice", "PERSON")
    old_node = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice has an alert routing workflow.",
        keywords=["alert-routing"],
    )
    db.entity_link_node(old_node, alice)
    observation_id = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="alert-routing",
        topic_label="alert-routing",
        bundle_type="observation",
        summary="Alice has an alert routing workflow.",
        keywords=["alert-routing"],
        source_node_ids=[old_node],
    )
    other_topic_observation_id = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="health-management",
        topic_label="health-management",
        bundle_type="observation",
        summary="Alice has a health management routine.",
        keywords=["health"],
        source_node_ids=[old_node],
    )
    new_node = _add_memory_node(
        db,
        time_key=MemoryNodeManager._memory_time_key(0),
        summary="Alice refined alert routing.",
        keywords=["alert-routing"],
    )
    db.entity_link_node(new_node, alice)
    monkeypatch.setattr(
        db,
        "search_memory_evidence_bundles",
        lambda *args, **kwargs: pytest.fail("exact cluster matching must not use broad search"),
    )
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})

    candidates = mgr._candidate_evidence_bundles_for_fact_cluster({
        "entity_id": alice,
        "entity_name": "Alice",
        "topic_key": "alert-routing",
        "source_node_ids": [new_node],
        "source_nodes": [],
    })

    assert [candidate["id"] for candidate in candidates] == [observation_id]
    assert other_topic_observation_id not in [candidate["id"] for candidate in candidates]
    assert candidates[0]["entity_name"] == "Alice"


def test_observation_observations_preserve_fact_to_interpretation_type_mapping(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    task_node = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice implemented the alert routing fallback.",
        keywords=["alert-routing", "fallback"],
        fact_type="episodic",
        fact_kind="action",
        task_event_like=True,
    )
    preference_node = _add_memory_node(
        db,
        time_key="2026-05-01 10:10:00",
        summary="Alice explicitly prefers Slack for urgent alerts.",
        keywords=["Slack", "preference"],
        fact_type="semantic",
        fact_kind="preference",
    )
    constraint_node = _add_memory_node(
        db,
        time_key="2026-05-01 10:20:00",
        summary="Alice instructed that urgent alerts must not use email.",
        keywords=["email", "constraint"],
        fact_type="semantic",
        fact_kind="instruction",
    )
    for node_id in (task_node, preference_node, constraint_node):
        db.entity_link_node(node_id, alice)
    observation_id = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="alert-routing",
        topic_label="alert routing",
        bundle_type="observation",
        summary="Alice has several alert routing requirements and activities.",
        keywords=["alerts"],
        source_node_ids=[task_node, preference_node, constraint_node],
    )
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})

    observation_ids = mgr._update_observations_for_evidence_bundles([observation_id])

    assert len(observation_ids) == 3
    observations = db.get_observations_for_evidence_bundles([observation_id])
    by_type = {item["observation_type"]: item for item in observations}
    assert set(by_type) == {"task_progress", "preference_signal", "constraint"}
    assert by_type["task_progress"]["source_node_ids"] == [task_node]
    assert by_type["preference_signal"]["evidence_mode"] == "explicit"
    assert by_type["constraint"]["evidence_mode"] == "explicit"
    assert set(
        by_type["task_progress"]["metadata"]["allowed_interpretation_types"]
    ) == {"task", "project_state"}
    assert set(
        by_type["preference_signal"]["metadata"]["allowed_interpretation_types"]
    ) == {"explicit_preference"}
    assert set(
        by_type["constraint"]["metadata"]["allowed_interpretation_types"]
    ) == {"constraint", "explicit_instruction", "task_risk"}


def test_observation_type_gate_rejects_incompatible_interpretation(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    node_id = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice explicitly prefers Slack for urgent alerts.",
        keywords=["Slack", "preference"],
        fact_type="semantic",
        fact_kind="preference",
    )
    db.entity_link_node(node_id, alice)
    observation_id = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="alert-routing",
        topic_label="alert routing",
        bundle_type="observation",
        summary="Alice has an alert routing preference.",
        keywords=["alerts"],
        source_node_ids=[node_id],
    )
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})
    mgr._update_observations_for_evidence_bundles([observation_id])
    observation = db.get_observations_for_evidence_bundles([observation_id])[0]
    semantic_observation = mgr._build_semantic_observation(observation)
    task_interpretation = {
        "id": 1,
        "entity_id": alice,
        "interpretation_type": "task",
        "claim": "Alice is implementing alert routing.",
        "target_text": "alert routing",
        "scope": "alert-routing",
        "confidence": 0.9,
        "metadata": {},
        "evidence_observation_ids": [],
    }

    score, reason = mgr._calculate_interpretation_candidate_score(
        observation=semantic_observation,
        source_nodes=db.memory_nodes_by_ids([node_id]),
        interpretation=task_interpretation,
        observation_id=observation_id,
    )

    assert score == 0.0
    assert reason == "observation_type_gate"


def test_behavioral_preference_observation_only_allows_inferred_preference(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    node_ids = []
    for index in range(2):
        node_id = _add_memory_node(
            db,
            time_key=f"2026-05-0{index + 1} 10:00:00",
            summary="Alice again selected Slack for urgent alerts.",
            keywords=["Slack", "alerts"],
            fact_type="episodic",
            fact_kind="preference",
        )
        db.entity_link_node(node_id, alice)
        node_ids.append(node_id)
    observation_id = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="alert-routing",
        topic_label="alert routing",
        bundle_type="observation",
        summary="Alice repeatedly selected Slack for urgent alerts.",
        keywords=["Slack", "alerts"],
        source_node_ids=node_ids,
    )
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})

    mgr._update_observations_for_evidence_bundles([observation_id])

    observation = db.get_observations_for_evidence_bundles([observation_id])[0]
    assert observation["observation_type"] == "preference_signal"
    assert observation["evidence_mode"] == "behavioral"
    assert observation["metadata"]["allowed_interpretation_types"] == [
        "inferred_preference"
    ]


def test_observation_metadata_tracks_fact_type_mixture():
    metadata = MemoryNodeManager._normalize_observation_metadata(
        {"observation_kind": "context"},
        [
            {"fact_type": "semantic", "fact_kind": "context"},
            {"fact_type": "episodic", "fact_kind": "action"},
            {"fact_type": "episodic", "fact_kind": "error"},
        ],
    )

    assert metadata["source_fact_type_distribution"] == {"semantic": 1, "episodic": 2}
    assert metadata["dominant_fact_type"] == "episodic"
    assert metadata["evidence_mixture"] == "episodic_dominant"


def test_task_metadata_normalizes_steps():
    metadata = MemoryNodeManager._normalize_task_metadata({
        "task_status": "stale",
        "task_source": "model_output",
        "goal": "  Ship memory task tracking.  ",
        "evidence": "用户要求 task 存步骤",
        "steps": [
            {
                "title": "  设计 task steps 结构  ",
                "status": "finished",
                "evidence": "讨论 goal 和 steps",
                "notes": "keep compact",
            },
            "补充测试",
            {"title": ""},
        ],
        "next_action": "  验证 prompt 解析  ",
    })

    assert metadata["task_status"] == "active"
    assert metadata["task_source"] == "inferred_from_interpretation"
    assert metadata["goal"] == "Ship memory task tracking."
    assert metadata["evidence"] == ["用户要求 task 存步骤"]
    assert metadata["steps"] == [
        {
            "title": "设计 task steps 结构",
            "status": "active",
            "evidence": ["讨论 goal 和 steps"],
            "notes": "keep compact",
        },
        {
            "title": "补充测试",
            "status": "active",
        },
    ]
    assert metadata["next_action"] == "验证 prompt 解析"

    stale_metadata = MemoryNodeManager._normalize_task_metadata(
        {"task_status": "stale"},
        allow_stale=True,
    )
    assert stale_metadata["task_status"] == "stale"


def test_task_status_prompt_definitions_scope_stale_by_prompt_role():
    assert "interpretation_type 只能是 insight、task" in INTERPRETATION_GENERATION_PROMPT
    assert "task_status 只能是 active、blocked、paused、stale" in INTERPRETATION_GENERATION_PROMPT
    assert "task_source 固定为 inferred_from_interpretation" in INTERPRETATION_GENERATION_PROMPT
    assert "interpretation 更新模块" in INTERPRETATION_UPDATE_PROMPT
    assert "should_update" in INTERPRETATION_UPDATE_PROMPT
    assert "task_status 只能是 active、blocked、paused、stale" in INTERPRETATION_UPDATE_PROMPT

def test_interpretation_generation_prompt_defines_interpretation_contract():
    assert "四层记忆架构" in INTERPRETATION_GENERATION_PROMPT
    assert "observation" in INTERPRETATION_GENERATION_PROMPT
    assert "allowed_interpretation_types" in INTERPRETATION_GENERATION_PROMPT
    assert "current best interpretation" in INTERPRETATION_GENERATION_PROMPT
    assert "不是用户原话" in INTERPRETATION_GENERATION_PROMPT
    assert "单条 observation 的证据门槛" in INTERPRETATION_GENERATION_PROMPT
    assert "evidence_shape=single_event，通常 should_create=false" in INTERPRETATION_GENERATION_PROMPT
    assert "confidence 不要超过 0.75" in INTERPRETATION_GENERATION_PROMPT
    assert "should_create=false" in INTERPRETATION_GENERATION_PROMPT
    assert "action_implication" in INTERPRETATION_GENERATION_PROMPT
    assert "evidence_node_ids" in INTERPRETATION_GENERATION_PROMPT
    assert "evidence_observation_ids 是支持该 interpretation 的 observation id" in INTERPRETATION_GENERATION_PROMPT
    assert "counter_evidence_node_ids 是反驳、削弱、限定或造成冲突的底层 fact id" in INTERPRETATION_GENERATION_PROMPT
    assert "如果只是证据不足，不要把无关事实放入 counter_evidence_*" in INTERPRETATION_GENERATION_PROMPT
    assert "interpretation_type 只能是 insight、task" in INTERPRETATION_GENERATION_PROMPT
    assert "conflict_resolution" in INTERPRETATION_GENERATION_PROMPT
    assert "更新一条已经存在的 interpretation" in INTERPRETATION_UPDATE_PROMPT
    assert "新的 observation" in INTERPRETATION_UPDATE_PROMPT
    assert "evidence_node_ids 是直接支持更新后 interpretation 的底层 fact id" in INTERPRETATION_UPDATE_PROMPT
    assert "evidence_observation_ids 是支持更新后 interpretation 的 observation id" in INTERPRETATION_UPDATE_PROMPT
    assert "counter_evidence_observation_ids 是反驳、削弱、限定或造成冲突的 observation id" in INTERPRETATION_UPDATE_PROMPT


def test_structured_memory_log_message_is_json(caplog):
    with caplog.at_level(logging.INFO, logger="agent.memory_node_manager"):
        MemoryNodeManager._log_info("memory_reflect", "sample_event", {"fact": "用户修改 prompt"})

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "agent.memory_node_manager"
    ]
    assert messages
    assert messages[-1].startswith("\n")
    assert "\n  \"event\": \"sample_event\"" in messages[-1]
    assert "\n  \"payload\": {" in messages[-1]
    data = json.loads(messages[-1])
    assert data == {
        "scope": "memory_reflect",
        "event": "sample_event",
        "payload": {"fact": "用户修改 prompt"},
    }


def test_observation_source_nodes_match_topic_key_exactly(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    outdoor = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice wants outdoor activities.",
        keywords=["户外活动"],
    )
    low_intensity = db.memory_add_node(
        time_key="2026-05-01 11:00:00",
        summary="Alice wants low-intensity outdoor activities.",
        keywords=["低强度户外活动"],
        topic=["低强度户外活动"],
        original_dialog="{}",
        query_embedding=np.ones((1, 1536), dtype=np.float32),
    )
    db.entity_link_node(outdoor, alice)
    db.entity_link_node(low_intensity, alice)

    source_nodes = db.memory_observation_source_nodes(
        entity_id=alice,
        topic_key="户外活动",
        limit=12,
    )

    assert [node["id"] for node in source_nodes] == [outdoor]


def test_recall_includes_observations_and_supporting_facts(db, monkeypatch):
    alice = db.entity_add_entity("Alice", "PERSON")
    source_ids = []
    for idx, summary in enumerate(
        [
            "Alice prefers Slack for urgent alerts.",
            "Alice dislikes email for urgent alerts.",
            "Hermes recommended Slack alert routing for Alice.",
        ],
        1,
    ):
        node_id = _add_memory_node(
            db,
            time_key=f"2026-05-0{idx} 10:00:00",
            summary=summary,
            keywords=["Slack alerts"],
            fact_type="semantic" if idx < 3 else "episodic",
        )
        db.entity_link_node(node_id, alice)
        source_ids.append(node_id)
    evidence_bundle_id = db.memory_upsert_evidence_bundle(
        entity_id=alice,
        topic_key="slack-alerts",
        topic_label="Slack alerts",
        bundle_type="entity_topic",
        summary="Slack alert workflow evidence.",
        keywords=["Slack", "alerts"],
        source_node_ids=source_ids,
    )
    observation_id = db.memory_create_observation(
        evidence_bundle_id,
        {
            "observation_type": "preference_signal",
            "summary": "Alice's urgent alert workflow is Slack-centered.",
            "confidence": 0.85,
            "source_node_ids": source_ids,
        },
    )
    assert observation_id is not None
    monkeypatch.setattr(db, "_search_memory_vector", lambda *args, **kwargs: {})
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[json.dumps({"summary": "Alice Slack alerts", "keywords": ["Alice", "Slack", "alerts"]})],
    )

    context = mgr.recall("Alice Slack alerts")

    assert OBSERVATION_SECTION_HEADER in context
    assert "Alice's urgent alert workflow is Slack-centered." in context
    assert "[Supporting facts for observations]" in context
    assert "Alice prefers Slack for urgent alerts." in context


def test_topic_list_creates_standardized_topic_keys(db):
    topic_keys = MemoryNodeManager._topic_keys(["Slack", "alerts"])

    assert topic_keys == ["slack", "alerts"]


def test_memory_relation_candidates_use_entity_keyword_and_temporal_signals(db, monkeypatch):
    preference = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Slack", "alerts"],
    )
    keyword_match = _add_memory_node(
        db,
        time_key="2026-05-01 11:00:00",
        summary="The team configured Slack notification routing.",
        keywords=["Slack", "routing"],
    )
    recent_context = _add_memory_node(
        db,
        time_key="2026-05-01 12:00:00",
        summary="Bob mentioned calendar reminders.",
        keywords=["calendar"],
    )
    current = _add_memory_node(
        db,
        time_key="2026-05-01 13:00:00",
        summary="Alice asked Hermes to send urgent alerts through Slack.",
        keywords=["Alice", "Slack", "alerts"],
    )
    later = _add_memory_node(
        db,
        time_key="2026-05-01 14:00:00",
        summary="This future node should not be a causal candidate.",
        keywords=["Slack"],
    )
    alice_id = db.entity_add_entity("Alice", "PERSON")
    db.entity_link_node(preference, alice_id)
    db.entity_link_node(current, alice_id)
    monkeypatch.setattr(db, "_search_memory_vector", lambda *args, **kwargs: {})

    nodes, ids = db.memory_relation_candidates(
        node_id=current,
        query_embedding=np.ones((1, 1536), dtype=np.float32),
        keywords=["Slack", "alerts"],
        top_k=4,
        budget="mid",
    )

    assert preference in ids
    assert keyword_match in ids
    assert recent_context in ids
    assert later not in ids
    assert [node["id"] for node in nodes] == ids


def test_memory_relation_candidates_stay_within_same_fact_type(db, monkeypatch):
    world_prior = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Alice", "Slack"],
        fact_type="semantic",
    )
    experience_prior = _add_memory_node(
        db,
        time_key="2026-05-01 11:00:00",
        summary="Hermes recommended Slack alert routing for Alice.",
        keywords=["Alice", "Slack"],
        fact_type="episodic",
    )
    current_world = _add_memory_node(
        db,
        time_key="2026-05-01 12:00:00",
        summary="Alice wants urgent alerts in Slack.",
        keywords=["Alice", "Slack"],
        fact_type="semantic",
    )
    alice_id = db.entity_add_entity("Alice", "PERSON")
    for node_id in (world_prior, experience_prior, current_world):
        db.entity_link_node(node_id, alice_id)
    monkeypatch.setattr(db, "_search_memory_vector", lambda *args, **kwargs: {})

    _nodes, ids = db.memory_relation_candidates(
        node_id=current_world,
        query_embedding=np.ones((1, 1536), dtype=np.float32),
        keywords=["Alice", "Slack"],
        top_k=5,
        budget="mid",
    )

    assert world_prior in ids
    assert experience_prior not in ids


def test_memory_node_relations_store_multidimensional_scores(db):
    source = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Alice", "Slack"],
    )
    target = _add_memory_node(
        db,
        time_key="2026-05-01 12:00:00",
        summary="Alice wants Slack incident routing.",
        keywords=["Alice", "Slack"],
    )
    alice_id = db.entity_add_entity("Alice", "PERSON")
    db.entity_link_node(source, alice_id)
    db.entity_link_node(target, alice_id)

    db.memory_add_node_relation(
        source_node_id=source,
        target_node_id=target,
        relation_type="semantic",
        confidence=0.91,
    )

    relation = db._conn.execute(
        "SELECT semantic_score, causal_score, temporal_score, entity_score, weight "
        "FROM memory_node_relations"
    ).fetchone()
    assert relation["semantic_score"] == pytest.approx(0.91)
    assert relation["causal_score"] == pytest.approx(0.0)
    assert relation["temporal_score"] > 0.0
    assert relation["entity_score"] == pytest.approx(1.0)
    assert relation["weight"] > relation["temporal_score"] * 0.15


def test_memory_graph_expand_uses_priority_beam_search(db):
    seed = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Seed memory.",
        keywords=["seed"],
    )
    weak_direct = _add_memory_node(
        db,
        time_key="2026-05-01 11:00:00",
        summary="Weak direct neighbor.",
        keywords=["weak"],
    )
    strong_bridge = _add_memory_node(
        db,
        time_key="2026-05-01 12:00:00",
        summary="Strong bridge neighbor.",
        keywords=["bridge"],
    )
    strong_second_hop = _add_memory_node(
        db,
        time_key="2026-05-01 13:00:00",
        summary="Strong second-hop neighbor.",
        keywords=["target"],
    )

    db.memory_add_node_relation(seed, weak_direct, "semantic", confidence=0.2, weight=0.2)
    db.memory_add_node_relation(seed, strong_bridge, "semantic", confidence=0.9, weight=0.9)
    db.memory_add_node_relation(strong_bridge, strong_second_hop, "semantic", confidence=0.9, weight=0.9)

    ranked = db._memory_graph_expand_ranked(
        [seed],
        depth=2,
        limit=3,
    )

    assert ranked == [strong_bridge, strong_second_hop, weak_direct]


def test_relation_graph_links_temporal_same_day_and_semantic_prior_nodes(db, monkeypatch):
    previous_same_day = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Alice", "Slack"],
    )
    previous_other_day = _add_memory_node(
        db,
        time_key="2026-04-30 10:00:00",
        summary="Alice likes concise status reports.",
        keywords=["Alice", "reports"],
    )
    current = _add_memory_node(
        db,
        time_key="2026-05-01 12:00:00",
        summary="Alice wants Slack alerts for incidents.",
        keywords=["Alice", "Slack"],
    )
    seen = {}

    def fake_semantic_neighbors(embedding, *, exclude_node_id=None, allowed_ids=None, threshold=0.0):
        seen["exclude_node_id"] = exclude_node_id
        seen["allowed_ids"] = set(allowed_ids or [])
        seen["threshold"] = threshold
        return {previous_other_day: 0.91}

    monkeypatch.setattr(db, "memory_semantic_neighbors", fake_semantic_neighbors)
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})

    mgr._build_relation_graph(
        node_id=current,
        summary="Alice wants Slack alerts for incidents.",
        embedding=np.ones((1, 1536), dtype=np.float32),
        keywords=["Alice", "Slack"],
    )

    rows = db._conn.execute(
        "SELECT source_node_id, target_node_id, relation_type, confidence "
        "FROM memory_node_relations ORDER BY relation_type, target_node_id"
    ).fetchall()
    relations = {
        (row["source_node_id"], row["target_node_id"], row["relation_type"])
        for row in rows
    }
    assert (current, previous_same_day, "temporal") in relations
    assert (current, previous_other_day, "semantic") in relations
    assert seen["exclude_node_id"] == current
    assert seen["allowed_ids"] == {previous_same_day, previous_other_day}
    assert seen["threshold"] == pytest.approx(0.82)

    causal_edges = db._conn.execute(
        "SELECT relation_type FROM memory_node_relations "
        "WHERE source_node_id = ? AND relation_type NOT IN ('semantic', 'temporal')",
        (current,),
    ).fetchall()
    assert causal_edges == []
