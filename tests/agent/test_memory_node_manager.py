import json
import logging
from datetime import datetime, timedelta, timezone
import numpy as np
import pytest

from agent.memory_node_manager import MemoryNodeManager
from agent.memory_node_manager import (
    CAUSAL_RELATION_TYPE_TEXT,
    INSIGHT_CONSOLIDATION_PROMPT,
    OBSERVATION_CANDIDATE_INTERPRETATION_GUIDANCE,
    OBSERVATION_CONSOLIDATION_PROMPT,
    OBSERVATION_METADATA_GUIDANCE,
    OBSERVATION_SOURCE_FACT_GUIDANCE,
    OBSERVATION_TIME_GUIDANCE,
    OBSERVATION_MERGE_PROMPT,
    OBSERVATION_UPDATE_PROMPT,
    INTERPRETATION_GENERATION_PROMPT,
    INTERPRETATION_UPDATE_PROMPT,
    INTERPRETATION_SECTION_HEADER,
    RELATION_PROMPT_TEMPLATE,
    RETAIN_FACT_EXTRACTION_PROMPT,
    TASK_CONSOLIDATION_PROMPT,
)
from hermes_state import SessionDB


class _FakeEmbeddingClient:
    def embed_text(self, text):
        if not text:
            return None
        return np.ones((1, 1536), dtype=np.float32)


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
        self.async_calls = []

    def _call_llm(self, prompt):
        self.llm_prompts.append(prompt)
        if self._llm_outputs:
            return self._llm_outputs.pop(0)
        return None

    def _start_async_work(self, **kwargs):
        self.async_calls.append(kwargs)


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
                "topic": ["urgent", "team", "communication"],
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
                "topic": ["alert", "routing"],
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
        "SELECT id, summary, keywords, topic, tags, fact_type, fact_subject, fact_kind, entity_names, "
        "task_event_like, task_event_subject, task_relevance "
        "FROM memory_nodes ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["summary"] == retain_payload["facts"][0]["text"]
    assert rows[1]["summary"] == retain_payload["facts"][1]["text"]
    assert "Alice Slack email" == rows[0]["keywords"]
    assert "urgent team communication" == rows[0]["topic"]
    assert rows[0]["fact_type"] == "semantic"
    assert rows[0]["fact_subject"] == "user"
    assert rows[0]["fact_kind"] == "preference"
    assert rows[1]["fact_type"] == "episodic"
    assert rows[1]["fact_subject"] == "assistant"
    assert rows[1]["fact_kind"] == "recommendation"
    assert json.loads(rows[0]["entity_names"]) == ["用户", "Alice", "Slack"]
    assert json.loads(rows[1]["entity_names"]) == ["助手", "Alice", "Slack"]
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
        "FROM memory_node_relations"
    ).fetchone()
    assert relation["source_node_id"] == rows[0]["id"]
    assert relation["target_node_id"] == rows[1]["id"]
    assert relation["relation_type"] == "Cause"
    assert relation["confidence"] == pytest.approx(0.8)
    assert len(mgr.async_calls) == 2
    assert "run_entity_extraction" not in mgr.async_calls[0]
    assert "run_entity_extraction" not in mgr.async_calls[1]
    assert mgr.async_calls[0]["keywords"] == ["Alice", "Slack", "email"]


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

    interpretation_id = db.memory_upsert_interpretation(
        claim="用户当前倾向先用 heuristic 控制 task fact 选择，再谨慎修改 prompt。",
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
        "SELECT claim, interpretation_type, status, confidence, evidence_node_ids, metadata "
        "FROM memory_interpretations WHERE id = ?",
        (interpretation_id,),
    ).fetchone()
    assert row["interpretation_type"] == "task"
    assert row["status"] == "current"
    assert row["confidence"] == pytest.approx(0.86)
    assert json.loads(row["evidence_node_ids"]) == [1, 2]
    assert json.loads(row["metadata"]) == {"source": "agent_interpretation"}

    updated_id = db.memory_upsert_interpretation(
        claim="用户当前更偏好 deterministic heuristic 控制 task fact 选择。",
        subject_text="user",
        target_text="memory task fact selection",
        scope="memory-system-design",
        interpretation_type="task",
        confidence=0.9,
    )
    assert updated_id == interpretation_id

    results = db.memory_search_interpretations(["heuristic", "task"], top_k=5)

    assert [item["id"] for item in results] == [interpretation_id]
    assert results[0]["claim"] == "用户当前更偏好 deterministic heuristic 控制 task fact 选择。"
    assert results[0]["evidence_node_ids"] == [1, 2]


def test_memory_search_interpretations_separates_content_and_entity_matches(db):
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
        subject_entity_id=alice,
        target_text="collaboration",
        scope="workflow",
        interpretation_type="inferred_preference",
        confidence=0.95,
        action_implication="Consider person-specific collaboration context.",
    )

    keyword_results = db.memory_search_interpretations(["Alice", "Slack"], top_k=5)

    assert [item["id"] for item in keyword_results[:2]] == [content_match, entity_only]

    entity_results = db.memory_search_interpretations(
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
    assert row["topic"] == "postgresql project"
    assert row["fact_kind"] == "conversation_summary"
    assert "fact_kind:conversation_summary" in json.loads(row["tags"])
    assert "run_entity_extraction" not in mgr.async_calls[0]


def test_retain_and_relation_prompts_share_relation_type_contract():
    assert CAUSAL_RELATION_TYPE_TEXT in RETAIN_FACT_EXTRACTION_PROMPT
    assert CAUSAL_RELATION_TYPE_TEXT in RELATION_PROMPT_TEMPLATE
    assert "Reason/HinderedBy" not in RETAIN_FACT_EXTRACTION_PROMPT
    assert "Reason/HinderedBy" not in RELATION_PROMPT_TEMPLATE
    assert '"keywords": ["关键词1", "关键词2"]' in RETAIN_FACT_EXTRACTION_PROMPT
    assert '"topic": ["主题1", "主题2"]' in RETAIN_FACT_EXTRACTION_PROMPT
    assert '"fact_type": "semantic/episodic"' in RETAIN_FACT_EXTRACTION_PROMPT
    assert '"fact_subject": "user/assistant/world/project/system/other"' in RETAIN_FACT_EXTRACTION_PROMPT
    assert '"fact_kind": "preference/decision/request/recommendation/action/error/context/instruction/other"' in RETAIN_FACT_EXTRACTION_PROMPT
    assert '"priority": 80' in RETAIN_FACT_EXTRACTION_PROMPT
    assert '"time_confidence": "explicit/inferred_from_turn/unknown"' in RETAIN_FACT_EXTRACTION_PROMPT
    assert '"task_event_like": true' in RETAIN_FACT_EXTRACTION_PROMPT
    assert "对话发生时间：{turn_timestamp}" in RETAIN_FACT_EXTRACTION_PROMPT
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
    assert "固定句式" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "用户要求 AI 以后回答/执行任务时" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "可能影响任务状态或步骤的事件" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "不要求已经知道具体属于哪个任务" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "实体不是只限传统 NER" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "健康管理" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "商务活动" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "经济负担" in RETAIN_FACT_EXTRACTION_PROMPT
    assert "每条长期记忆 fact 通常至少包含主体实体" in RETAIN_FACT_EXTRACTION_PROMPT


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
        observation_id = db.memory_upsert_observation(
            entity_id=entity_id,
            topic_key=topic_key,
            topic_label=topic_label,
            observation_type="observation",
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
    monkeypatch.setattr(db, "_memory_search_vector", lambda *args, **kwargs: {})
    monkeypatch.setattr(db, "_memory_search_keyword", lambda *args, **kwargs: {})

    nodes = db.memory_search(
        "no-match",
        np.ones((1, 1536), dtype=np.float32),
        top_k=5,
        time_start="2026-05-01 00:00:00",
        time_end="2026-05-31 23:59:59",
    )

    assert [n["id"] for n in nodes] == [newer, older]


def test_memory_search_rrf_includes_graph_neighbors(db, monkeypatch):
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
    monkeypatch.setattr(db, "_memory_search_vector", lambda *args, **kwargs: {})

    nodes = db.memory_search(
        ["Slack"],
        np.ones((1, 1536), dtype=np.float32),
        top_k=3,
        budget="mid",
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
    monkeypatch.setattr(db, "_memory_search_vector", lambda *args, **kwargs: {})

    semantic_nodes = db.memory_search(
        ["Alice", "Slack"],
        np.ones((1, 1536), dtype=np.float32),
        top_k=5,
        fact_types=["semantic"],
    )
    episodic_nodes = db.memory_search(
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

    results = db._memory_search_keyword("简洁回答", limit=10)

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

    results = db._memory_search_keyword("喜欢结论", limit=10)

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

    results = db._memory_search_keyword(
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

    monkeypatch.setattr(db, "_memory_search_vector", fake_vector_search)

    nodes = db.memory_search(
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
    monkeypatch.setattr(db, "_memory_search_vector", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        db,
        "_memory_search_keyword",
        lambda *args, **kwargs: {stale: 0.01, fresh: 0.02},
    )

    nodes = db.memory_search(
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

    summary = mgr._summarize_turn("Alice wants Slack alerts", "")

    assert summary["keywords"] == ["Slack", "alerts"]
    assert summary["entities"] == [{"name": "Alice", "type": "PERSON"}]


def test_memory_search_observations_uses_entities(db):
    node_id = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack alerts.",
        keywords=["Slack"],
    )
    alice = db.entity_add_entity("Alice", "PERSON")
    db.memory_upsert_observation(
        entity_id=alice,
        topic_key="alerts",
        topic_label="alerts",
        observation_type="insight",
        summary="The user prefers concise escalation notes.",
        keywords=["escalation"],
        source_node_ids=[node_id],
        confidence=0.8,
    )

    results = db.memory_search_observations(
        ["unrelated"],
        entities=[{"name": "Alice", "type": "PERSON"}],
        top_k=3,
    )

    assert [row["entity_name"] for row in results] == ["Alice"]


def test_memory_search_observations_rejects_weak_family_term_entity_mismatch(db):
    node_id = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="小明父亲退休后保留了自驾游偏好。",
        keywords=["小明父亲", "退休", "自驾游"],
    )
    xiaoming_father = db.entity_add_entity("小明父亲", "PERSON")
    db.memory_upsert_observation(
        entity_id=xiaoming_father,
        topic_key="家庭",
        topic_label="家庭",
        observation_type="insight",
        summary="小明父亲退休后保留了对开车自驾游和苹果的偏好。",
        keywords=["退休", "自驾游", "苹果"],
        source_node_ids=[node_id],
        confidence=0.7,
    )

    results = db.memory_search_observations(
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

    report = db.memory_reflect_entities(dry_run=True, limit=10)

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

    report = db.memory_reflect_entities(
        dry_run=True,
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
    db.entity_add_entity("Alice", "PERSON")
    alice_spaced = db.entity_add_entity(" alice ", "PERSON")
    node_id = _add_memory_node(
        db,
        time_key=MemoryNodeManager._memory_time_key(0),
        summary="Alice prefers Slack alerts.",
        keywords=["Alice", "Slack"],
    )
    db.entity_link_node(node_id, alice_spaced)
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})

    report = mgr.reflect(dry_run=True, limit=5)

    assert report["dry_run"] is True
    assert report["observation_reflect"]["touched_entity_ids"] == [alice_spaced]
    assert report["merge_candidates"] == 1


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

    report = db.memory_reflect_entities(dry_run=False, limit=10)

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
    first_observation = db.memory_upsert_observation(
        entity_id=alice,
        topic_key="alerts",
        topic_label="alerts",
        observation_type="observation",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Slack", "alerts"],
        source_node_ids=[first_node],
        confidence=0.8,
        metadata={"observation_kind": "context"},
    )
    second_observation = db.memory_upsert_observation(
        entity_id=alice_spaced,
        topic_key="alerts",
        topic_label="alerts",
        observation_type="observation",
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

    report = mgr.reflect(dry_run=False, limit=10)

    assert report["merged"] == 1
    assert report["observation_groups_merged"] == 1
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
        "FROM memory_observations ORDER BY id"
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
            "SELECT node_id FROM memory_observation_sources WHERE observation_id = ?",
            (rows[0]["id"],),
        ).fetchall()
    }
    assert source_ids == {first_node, second_node, touched_node}


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
    insight_id = db.memory_upsert_observation(
        entity_id=alice,
        topic_key="alerts",
        topic_label="alerts",
        observation_type="insight",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Slack", "alerts"],
        source_node_ids=[node_ids[0]],
    )
    task_id = db.memory_upsert_observation(
        entity_id=alice,
        topic_key="alerts",
        topic_label="alerts",
        observation_type="task",
        summary="Alice is actively implementing Slack alert routing.",
        keywords=["Slack", "alerts", "routing"],
        source_node_ids=[node_ids[1]],
        metadata={
            "task_status": "active",
            "task_source": "inferred_from_observation",
        },
    )

    groups = db.memory_duplicate_observation_groups(entity_ids=[alice])

    assert groups == []
    rows = db._conn.execute(
        "SELECT id, observation_type FROM memory_observations ORDER BY id"
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
    world_observation = db.memory_upsert_observation(
        entity_id=alice,
        topic_key="world-alerts",
        topic_label="world alerts",
        observation_type="insight",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Slack", "alerts"],
        source_node_ids=[world_node],
    )
    experience_observation = db.memory_upsert_observation(
        entity_id=alice,
        topic_key="experience-alerts",
        topic_label="experience alerts",
        observation_type="insight",
        summary="Hermes has prior Slack alert routing experience for Alice.",
        keywords=["Slack", "alerts"],
        source_node_ids=[experience_node],
    )

    node_report = db.memory_reflect_node_decay(
        dry_run=False,
        fact_half_life_days=365,
        experience_half_life_days=30,
        now=datetime(2026, 4, 1, 0, 0, 0),
    )
    report = db.memory_reflect_observation_decay(
        dry_run=False,
        threshold=0.3,
        now=datetime(2026, 4, 1, 0, 0, 0),
    )

    rows = {
        row["id"]: row["status"]
        for row in db._conn.execute(
            "SELECT id, status FROM memory_observations WHERE id IN (?, ?)",
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
        dry_run=False,
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
        dry_run=False,
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


def test_memory_search_observations_ignores_inactive_observations(db):
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
    active_observation = db.memory_upsert_observation(
        entity_id=alice,
        topic_key="slack-alerts",
        topic_label="Slack alerts",
        observation_type="insight",
        summary="Alice currently prefers Slack for urgent alerts.",
        keywords=["Slack", "alerts"],
        source_node_ids=[active_node],
    )
    inactive_observation = db.memory_upsert_observation(
        entity_id=alice,
        topic_key="email-alerts",
        topic_label="email alerts",
        observation_type="insight",
        summary="Alice once preferred email alerts.",
        keywords=["email", "alerts"],
        source_node_ids=[stale_node],
    )
    db._conn.execute(
        "UPDATE memory_observations SET status = 'inactive' WHERE id = ?",
        (inactive_observation,),
    )

    results = db.memory_search_observations(["Alice", "alerts"], top_k=5)

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
    observation_id = db.memory_upsert_observation(
        entity_id=alice,
        topic_key="email-alerts",
        topic_label="email alerts",
        observation_type="insight",
        summary="Alice used email alerts long ago.",
        keywords=["email", "alerts"],
        source_node_ids=[old_node],
    )
    mgr = _NoAsyncMemoryNodeManager(db, embedding_config={})

    report = mgr.reflect(
        dry_run=False,
        experience_half_life_days=7,
        observation_decay_threshold=0.9,
    )

    row = db._conn.execute(
        "SELECT mo.status, mn.decay_score "
        "FROM memory_observations mo "
        "JOIN memory_observation_sources mos ON mos.observation_id = mo.id "
        "JOIN memory_nodes mn ON mn.id = mos.node_id "
        "WHERE mo.id = ?",
        (observation_id,),
    ).fetchone()
    assert report["node_decay"]["updated"] == 1
    assert report["observations_inactivated"] == 1
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
    monkeypatch.setattr(db, "_memory_search_vector", lambda *args, **kwargs: {})
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[json.dumps({"summary": "Alice Slack alerts", "keywords": ["Alice", "Slack"]})],
    )

    context = mgr.recall("Alice Slack alerts")

    assert "[Semantic memories" in context
    assert "[Episodic memories" in context
    assert "semantic memories" in context
    assert "episodic memories" in context
    assert "Alice prefers Slack for urgent alerts." in context
    assert "Hermes recommended Slack alert routing for Alice." in context


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
    monkeypatch.setattr(db, "_memory_search_vector", lambda *args, **kwargs: {})
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
    assert context.index(INTERPRETATION_SECTION_HEADER) < context.index("[Semantic memories")


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
    assert db._conn.execute("SELECT COUNT(*) FROM memory_observations").fetchone()[0] == 0

    report = mgr.reflect(dry_run=False, limit=10)
    assert report["observation_reflect"]["candidate_count"] == 3
    assert report["observations_consolidated"] == 1
    assert report["observation_reflect"]["fact_clusters_consolidated"] == 1
    assert report["observation_reflect"]["fact_cluster_node_count"] == 3

    observation = db._conn.execute(
        "SELECT mo.summary, mo.topic_key, mo.observation_type, mo.metadata, en.name AS entity_name "
        "FROM memory_observations mo "
        "JOIN entity_nodes en ON en.id = mo.entity_id"
    ).fetchone()
    assert observation["entity_name"] == "Alice"
    assert observation["topic_key"] == "slack-alerts"
    assert observation["observation_type"] == "observation"
    assert json.loads(observation["metadata"])["observation_kind"] == "context"
    assert observation["summary"] == "Alice's urgent alert workflow is Slack-centered."
    sources = db._conn.execute("SELECT node_id FROM memory_observation_sources").fetchall()
    assert len(sources) == 3


def test_unmatched_fact_clusters_match_generalized_topic_with_time_window(db):
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

    clusters = mgr._unmatched_fact_clusters(facts, excluded_node_ids=set())

    normalized = [
        cluster for cluster in clusters
        if cluster.get("topic_key") == "家庭" and cluster.get("topic_match") == "normalized"
    ]
    assert normalized
    assert normalized[0]["raw_topic_keys"] == ["家庭", "家庭关系"]
    assert normalized[0]["source_node_ids"] == [1, 2]


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

    clusters = mgr._unmatched_fact_clusters(facts, excluded_node_ids=set())

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

    clusters = mgr._unmatched_fact_clusters(facts, excluded_node_ids=set())

    assert not [
        cluster for cluster in clusters
        if cluster.get("topic_key") == "健康" and cluster.get("topic_match") == "normalized"
    ]


def test_unmatched_fact_clusters_filter_suffix_facts_outside_time_window(db):
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

    clusters = mgr._unmatched_fact_clusters(facts, excluded_node_ids=set())

    normalized = [
        cluster for cluster in clusters
        if cluster.get("topic_key") == "健康" and cluster.get("topic_match") == "normalized"
    ]
    assert normalized
    assert normalized[0]["source_node_ids"] == [1, 2]


def test_unmatched_fact_clusters_match_suffix_fact_to_nearest_bare_fact(db):
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

    clusters = mgr._unmatched_fact_clusters(facts, excluded_node_ids=set())

    normalized = [
        cluster for cluster in clusters
        if cluster.get("topic_key") == "健康" and cluster.get("topic_match") == "normalized"
    ]
    assert normalized
    assert normalized[0]["source_node_ids"] == [1, 2, 3]


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
                "category": "observation",
                "summary": "Alice's urgent alert workflow is Slack-centered.",
                "keywords": ["Slack", "alerts"],
                "confidence": 0.86,
                "metadata": {"observation_kind": "context"},
            }),
            json.dumps({
                "should_create": True,
                "claim": "Agent 当前解释为 Alice 的紧急告警协作应优先使用 Slack。",
                "target_text": "urgent alert routing",
                "scope": "alert-workflow",
                "interpretation_type": "insight",
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

    report = mgr.reflect(dry_run=False, limit=10)

    assert report["observations_consolidated"] == 1
    interpretation = db._conn.execute(
        "SELECT claim, target_text, scope, interpretation_type, confidence, "
        "action_implication, evidence_node_ids, evidence_observation_ids, metadata "
        "FROM memory_interpretations"
    ).fetchone()
    observation_id = db._conn.execute("SELECT id FROM memory_observations").fetchone()["id"]
    assert interpretation["interpretation_type"] == "insight"
    assert interpretation["target_text"] == "urgent alert routing"
    assert interpretation["scope"] == "alert-workflow"
    assert interpretation["confidence"] == pytest.approx(0.88)
    assert "优先使用 Slack" in interpretation["claim"]
    assert "优先建议 Slack" in interpretation["action_implication"]
    assert json.loads(interpretation["evidence_node_ids"]) == node_ids[:2]
    assert json.loads(interpretation["evidence_observation_ids"]) == [observation_id]
    assert json.loads(interpretation["metadata"])["source"] == "interpretation_generation"
    assert any("interpretation 生成模块" in prompt for prompt in mgr.llm_prompts)


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
    observation_id = db.memory_upsert_observation(
        entity_id=alice,
        topic_key="slack-alerts",
        topic_label="Slack alerts",
        observation_type="observation",
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

    generated = mgr._generate_interpretations_for_observations([observation_id])

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
    observation_id = db.memory_upsert_observation(
        entity_id=alice,
        topic_key="architecture-planning",
        topic_label="architecture planning",
        observation_type="observation",
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

    generated = mgr._generate_interpretations_for_observations([observation_id])

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
    observation_id = db.memory_upsert_observation(
        entity_id=hermes,
        topic_key="memory-matching",
        topic_label="memory matching",
        observation_type="observation",
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
    source_nodes = db.memory_observation_supporting_nodes([observation_id])[observation_id]

    linked = mgr._link_observation_to_existing_interpretation(
        observation=dict(db._conn.execute("SELECT * FROM memory_observations WHERE id = ?", (observation_id,)).fetchone()),
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
    first_observation = db.memory_upsert_observation(
        entity_id=hermes,
        topic_key="memory-interpretation",
        topic_label="memory interpretation",
        observation_type="observation",
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
    second_observation = db.memory_upsert_observation(
        entity_id=hermes,
        topic_key="memory-implementation",
        topic_label="memory implementation",
        observation_type="observation",
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

    generated = mgr._generate_interpretations_for_observations([first_observation, second_observation])

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
    observation_id = db.memory_upsert_observation(
        entity_id=hermes,
        topic_key="memory-indexing",
        topic_label="memory indexing",
        observation_type="observation",
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

    generated = mgr._generate_interpretations_for_observations([observation_id])

    assert generated == 0
    assert mgr.llm_prompts == []
    metadata = json.loads(
        db._conn.execute(
            "SELECT metadata FROM memory_observations WHERE id = ?",
            (observation_id,),
        ).fetchone()["metadata"]
    )
    assert metadata["interpretation_status"] == "deferred"
    assert metadata["interpretation_reason"] == "trigger_threshold_not_met"
    assert metadata["interpretation_basis_hash"]


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
            db.memory_upsert_observation(
                entity_id=entity,
                topic_key=f"ordinary-context-{index}",
                topic_label=f"ordinary context {index}",
                observation_type="observation",
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

    generated = mgr._generate_interpretations_for_observations(observation_ids)

    assert generated == 0
    assert mgr.llm_prompts == []
    rows = db._conn.execute(
        "SELECT metadata FROM memory_observations WHERE id IN (?, ?, ?) ORDER BY id",
        tuple(observation_ids),
    ).fetchall()
    for row in rows:
        metadata = json.loads(row["metadata"])
        assert metadata["interpretation_status"] == "deferred"
        assert metadata["interpretation_reason"] == "trigger_threshold_not_met"


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
    observation_id = db.memory_upsert_observation(
        entity_id=alice,
        topic_key="architecture-notes",
        topic_label="architecture notes",
        observation_type="observation",
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

    first = mgr._generate_interpretations_for_observations([observation_id])
    second = mgr._generate_interpretations_for_observations([observation_id])

    assert first == 1
    assert second == 0
    assert mgr.llm_prompts == []
    row = db._conn.execute(
        "SELECT metadata FROM memory_observations WHERE id = ?",
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
    first_observation = db.memory_upsert_observation(
        entity_id=hermes,
        topic_key="memory-indexing",
        topic_label="memory indexing",
        observation_type="observation",
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
    assert first_mgr._generate_interpretations_for_observations([first_observation]) == 0

    second_observation = db.memory_upsert_observation(
        entity_id=hermes,
        topic_key="memory-indexing",
        topic_label="memory indexing followup",
        observation_type="observation_followup",
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

    generated = mgr._generate_interpretations_for_observations([second_observation])

    assert generated == 1
    prompt = next(prompt for prompt in mgr.llm_prompts if "interpretation 生成模块" in prompt)
    assert "Clustered observations" in prompt
    assert f"id={first_observation}" in prompt
    assert f"id={second_observation}" in prompt
    rows = db._conn.execute(
        "SELECT id, metadata FROM memory_observations WHERE id IN (?, ?) ORDER BY id",
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
    assert db._conn.execute("SELECT COUNT(*) FROM memory_observations").fetchone()[0] == 0

    report = mgr.reflect(dry_run=False, limit=10)
    assert report["observations_consolidated"] == 1

    observation = db._conn.execute(
        "SELECT observation_type, summary, metadata FROM memory_observations"
    ).fetchone()
    metadata = json.loads(observation["metadata"])
    assert observation["observation_type"] == "observation"
    assert "正在迭代 Hermes Agent" in observation["summary"]
    assert metadata["observation_kind"] == "event_cluster"
    assert metadata["evidence_shape"] == "progression"
    assert metadata["temporal_scope"] == "recent"
    assert metadata["candidate_interpretation_types"] == ["insight", "task"]
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

    report = mgr.reflect(dry_run=False, limit=10)

    assert report["observation_reflect"].get("task_matched", 0) == 0
    assert report["observation_reflect"]["fact_observation_matches"] == 1
    assert report["observation_reflect"]["fact_observation_node_count"] == 1
    assert report["observation_reflect"]["entity_topic_updates"] == 0
    assert report["observation_reflect"]["entity_topic_node_count"] == 1
    assert db.memory_observation_source_ids(observation_id) == [source_node, new_node]
    row = db._conn.execute(
        "SELECT summary, metadata FROM memory_observations WHERE id = ?",
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

    report = mgr.reflect(dry_run=False, limit=10)

    assert report["observation_reflect"].get("task_matched", 0) == 0
    assert report["observation_reflect"]["fact_observation_matches"] == 0
    assert report["observation_reflect"]["fact_clusters_consolidated"] == 0
    assert db.memory_observation_source_ids(observation_id) == [source_node]
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

    report = mgr.reflect(dry_run=False, limit=10)

    assert report["observation_reflect"].get("task_matched", 0) == 0
    assert report["observation_reflect"]["entity_topic_updates"] == 0
    assert db.memory_observation_source_ids(observation_id) == [source_node]


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

    report = mgr.reflect(dry_run=False, limit=10)

    assert report["observation_reflect"].get("task_matched", 0) == 0
    assert report["observation_reflect"]["entity_topic_updates"] == 0
    assert db.memory_observation_source_ids(observation_id) == [source_node]


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

    report = mgr.reflect(dry_run=False, limit=10)

    assert report["observation_reflect"].get("task_matched", 0) == 0
    assert db.memory_observation_source_ids(observation_id) == [source_node]
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
    observation_id = db.memory_upsert_observation(
        entity_id=alice,
        topic_key="alert-routing",
        topic_label="alert-routing",
        observation_type="observation",
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

    report = mgr.reflect(dry_run=False, limit=10)

    assert report["observation_reflect"]["fact_observation_matches"] == 1
    assert report["observation_reflect"]["fact_observation_node_count"] == 1
    assert report["observation_reflect"]["entity_topic_updates"] == 0
    assert report["observation_reflect"]["entity_topic_node_count"] == 1
    assert db.memory_observation_source_ids(observation_id) == [old_node, new_matching_node]
    assert unrelated_task_node not in db.memory_observed_source_node_ids([unrelated_task_node])
    row = db._conn.execute(
        "SELECT summary FROM memory_observations WHERE id = ?",
        (observation_id,),
    ).fetchone()
    assert row["summary"] == "Alice has continued refining alert routing over time."
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
    assert INSIGHT_CONSOLIDATION_PROMPT == OBSERVATION_CONSOLIDATION_PROMPT
    assert TASK_CONSOLIDATION_PROMPT == OBSERVATION_CONSOLIDATION_PROMPT
    for prompt in (OBSERVATION_CONSOLIDATION_PROMPT, OBSERVATION_UPDATE_PROMPT, OBSERVATION_MERGE_PROMPT):
        assert '"category": "observation"' in prompt
        assert "task_status、goal、steps、next_action、insight_type" in prompt
        assert '"task_status": "active | blocked | paused | stale"' not in prompt
        assert '"category": "task"' not in prompt
        assert '"category": "insight"' not in prompt

    assert "interpretation_type 只能是 insight、task" in INTERPRETATION_GENERATION_PROMPT
    assert "task_status 只能是 active、blocked、paused、stale" in INTERPRETATION_GENERATION_PROMPT
    assert "task_source 固定为 inferred_from_interpretation" in INTERPRETATION_GENERATION_PROMPT
    assert "interpretation 更新模块" in INTERPRETATION_UPDATE_PROMPT
    assert "should_update" in INTERPRETATION_UPDATE_PROMPT
    assert "task_status 只能是 active、blocked、paused、stale" in INTERPRETATION_UPDATE_PROMPT


def test_observation_prompts_explain_fact_type_and_kind_labels():
    assert "semantic（语义记忆）" in OBSERVATION_SOURCE_FACT_GUIDANCE
    assert "episodic（情景记忆）" in OBSERVATION_SOURCE_FACT_GUIDANCE
    assert "fact_subject 表示记忆主体" in OBSERVATION_SOURCE_FACT_GUIDANCE
    assert "fact_kind 类别说明" in OBSERVATION_SOURCE_FACT_GUIDANCE
    assert "preference：用户长期或反复表达的偏好" in OBSERVATION_SOURCE_FACT_GUIDANCE
    assert "instruction：用户要求 AI 以后长期遵守" in OBSERVATION_SOURCE_FACT_GUIDANCE
    assert "fact_type 决定记忆性质" in OBSERVATION_SOURCE_FACT_GUIDANCE
    assert "observation_kind 表示 observation 的信息性质" in OBSERVATION_METADATA_GUIDANCE
    assert "evidence_shape 表示支撑 observation 的证据形态" in OBSERVATION_METADATA_GUIDANCE
    assert "temporal_scope 表示 observation 的时间范围" in OBSERVATION_METADATA_GUIDANCE
    assert "source facts 行中的 time 表示该事实的证据时间" in OBSERVATION_TIME_GUIDANCE
    assert "source_time_start/source_time_end 表示已有 observation 的证据覆盖范围" in OBSERVATION_TIME_GUIDANCE
    assert "created_at/updated_at 表示 observation 记录的存储生命周期" in OBSERVATION_TIME_GUIDANCE
    assert "candidate_interpretation_types 是给 interpretation 生成/匹配阶段使用的粗粒度路由提示" in OBSERVATION_CANDIDATE_INTERPRETATION_GUIDANCE
    assert "最终 interpretation_type 仍由 interpretation prompt" in OBSERVATION_CANDIDATE_INTERPRETATION_GUIDANCE
    assert "explicit_preference、explicit_instruction、inferred_preference、behavior_pattern" in OBSERVATION_CANDIDATE_INTERPRETATION_GUIDANCE
    assert "task 表示 observation 可能支持 Agent 当前认为用户正在推进的任务或目标" in OBSERVATION_CANDIDATE_INTERPRETATION_GUIDANCE

    for prompt in (
        INSIGHT_CONSOLIDATION_PROMPT,
        TASK_CONSOLIDATION_PROMPT,
        OBSERVATION_UPDATE_PROMPT,
        OBSERVATION_MERGE_PROMPT,
    ):
        assert OBSERVATION_SOURCE_FACT_GUIDANCE in prompt
        assert OBSERVATION_METADATA_GUIDANCE in prompt
        assert OBSERVATION_TIME_GUIDANCE in prompt
        assert OBSERVATION_CANDIDATE_INTERPRETATION_GUIDANCE in prompt
        assert "candidate_interpretation_types" in prompt
        assert "evidence_shape" in prompt
        assert "temporal_scope" in prompt
        assert "preference_signal" in prompt


def test_interpretation_generation_prompt_defines_interpretation_contract():
    assert "三层记忆架构" in INTERPRETATION_GENERATION_PROMPT
    assert "current best interpretation" in INTERPRETATION_GENERATION_PROMPT
    assert "不是用户原话" in INTERPRETATION_GENERATION_PROMPT
    assert "should_create=false" in INTERPRETATION_GENERATION_PROMPT
    assert "action_implication" in INTERPRETATION_GENERATION_PROMPT
    assert "evidence_node_ids" in INTERPRETATION_GENERATION_PROMPT
    assert "interpretation_type 只能是 insight、task" in INTERPRETATION_GENERATION_PROMPT
    assert "conflict_resolution" in INTERPRETATION_GENERATION_PROMPT
    assert "更新一条已经存在的 interpretation" in INTERPRETATION_UPDATE_PROMPT
    assert "新的 observation" in INTERPRETATION_UPDATE_PROMPT


def test_reflect_error_log_message_is_json(caplog):
    with caplog.at_level(logging.ERROR, logger="agent.memory_node_manager"):
        MemoryNodeManager._log_reflect_error("sample_event", {"fact": "用户修改 prompt"})

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
    db.memory_upsert_observation(
        entity_id=alice,
        topic_key="slack-alerts",
        topic_label="Slack alerts",
        observation_type="insight",
        summary="Alice's urgent alert workflow is Slack-centered.",
        keywords=["Slack", "alerts"],
        source_node_ids=source_ids,
    )
    monkeypatch.setattr(db, "_memory_search_vector", lambda *args, **kwargs: {})
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[json.dumps({"summary": "Alice Slack alerts", "keywords": ["Alice", "Slack", "alerts"]})],
    )

    context = mgr.recall("Alice Slack alerts")

    assert "[Observations" in context
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
    monkeypatch.setattr(db, "_memory_search_vector", lambda *args, **kwargs: {})

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
    monkeypatch.setattr(db, "_memory_search_vector", lambda *args, **kwargs: {})

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
