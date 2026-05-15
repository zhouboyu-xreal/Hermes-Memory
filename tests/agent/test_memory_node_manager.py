import json
from datetime import datetime, timedelta
import numpy as np
import pytest

from agent.memory_node_manager import MemoryNodeManager
from agent.memory_node_manager import (
    CAUSAL_RELATION_TYPE_TEXT,
    OBSERVATION_CONSOLIDATION_PROMPT,
    OBSERVATION_MERGE_PROMPT,
    OBSERVATION_UPDATE_PROMPT,
    RELATION_PROMPT_TEMPLATE,
    RETAIN_FACT_EXTRACTION_PROMPT,
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
                "fact_type": "world",
                "fact_kind": "preference",
                "occurred_start": "2026-05-01 00:00:00",
                "occurred_end": "2026-05-01 23:59:59",
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
                "fact_type": "experience",
                "fact_kind": "recommendation",
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
        "SELECT id, summary, keywords, topic, tags FROM memory_nodes ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["summary"] == retain_payload["facts"][0]["text"]
    assert rows[1]["summary"] == retain_payload["facts"][1]["text"]
    assert "Alice Slack email" == rows[0]["keywords"]
    assert "urgent team communication" == rows[0]["topic"]
    assert "fact_type:world" in json.loads(rows[0]["tags"])
    assert "fact_type:experience" in json.loads(rows[1]["tags"])

    detail = db._conn.execute(
        "SELECT original_dialog FROM memory_nodes WHERE id = ?",
        (rows[0]["id"],),
    ).fetchone()
    original_payload = json.loads(detail["original_dialog"])
    assert original_payload["retain_fact"]["fact_kind"] == "preference"
    assert original_payload["retain_fact"]["occurred_start"] == "2026-05-01 00:00:00"

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
        "SELECT summary, keywords, topic, tags FROM memory_nodes"
    ).fetchone()
    assert row["summary"] == summary_payload["summary"]
    assert row["keywords"] == "PostgreSQL project"
    assert row["topic"] == "postgresql project"
    assert "fact_kind:conversation_summary" in json.loads(row["tags"])
    assert "run_entity_extraction" not in mgr.async_calls[0]


def test_retain_and_relation_prompts_share_relation_type_contract():
    assert CAUSAL_RELATION_TYPE_TEXT in RETAIN_FACT_EXTRACTION_PROMPT
    assert CAUSAL_RELATION_TYPE_TEXT in RELATION_PROMPT_TEMPLATE
    assert "Reason/HinderedBy" not in RETAIN_FACT_EXTRACTION_PROMPT
    assert "Reason/HinderedBy" not in RELATION_PROMPT_TEMPLATE
    assert '"keywords": ["关键词1", "关键词2"]' in RETAIN_FACT_EXTRACTION_PROMPT
    assert '"topic": ["主题1", "主题2"]' in RETAIN_FACT_EXTRACTION_PROMPT


def test_store_turn_filters_plain_time_expressions_from_fact_entities(db):
    retain_payload = {
        "facts": [
            {
                "text": "Alice wants Slack alerts during the Spring Festival launch window.",
                "keywords": ["Alice", "Slack", "春节"],
                "topic": ["launch", "alerts"],
                "fact_type": "world",
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
                "fact_type": "world",
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


def _add_memory_node(db, *, time_key, summary, keywords, fact_type="world"):
    return db.memory_add_node(
        time_key=time_key,
        summary=summary,
        keywords=keywords,
        topic=keywords,
        original_dialog="{}",
        query_embedding=np.ones((1, 1536), dtype=np.float32),
        fact_type=fact_type,
    )


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
    world = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Alice", "Slack"],
        fact_type="world",
    )
    experience = _add_memory_node(
        db,
        time_key="2026-05-01 11:00:00",
        summary="Hermes recommended Slack alert routing for Alice.",
        keywords=["Alice", "Slack"],
        fact_type="experience",
    )
    monkeypatch.setattr(db, "_memory_search_vector", lambda *args, **kwargs: {})

    world_nodes = db.memory_search(
        ["Alice", "Slack"],
        np.ones((1, 1536), dtype=np.float32),
        top_k=5,
        fact_types=["world"],
    )
    experience_nodes = db.memory_search(
        ["Alice", "Slack"],
        np.ones((1, 1536), dtype=np.float32),
        top_k=5,
        fact_types=["experience"],
    )

    assert [n["id"] for n in world_nodes] == [world]
    assert [n["id"] for n in experience_nodes] == [experience]
    assert world_nodes[0]["fact_type"] == "world"
    assert experience_nodes[0]["fact_type"] == "experience"


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
        observation_type="insight",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Slack", "alerts"],
        source_node_ids=[first_node],
        confidence=0.8,
    )
    second_observation = db.memory_upsert_observation(
        entity_id=alice_spaced,
        topic_key="alerts",
        topic_label="alerts",
        observation_type="insight",
        summary="Alice routes incident notifications through Slack.",
        keywords=["Slack", "notifications"],
        source_node_ids=[second_node],
        confidence=0.75,
    )
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "category": "insight",
                "summary": "Alice consistently wants urgent and incident alerts routed through Slack.",
                "keywords": ["Slack", "alerts", "notifications"],
                "confidence": 0.9,
                "metadata": {"insight_type": "preference"},
            })
        ],
    )

    report = mgr.reflect(dry_run=False, limit=10)

    assert report["merged"] == 1
    assert report["observation_groups_merged"] == 1
    assert "current_category: insight" in mgr.llm_prompts[-1]
    rows = db._conn.execute(
        "SELECT id, entity_id, topic_key, observation_type, summary, keywords "
        "FROM memory_observations ORDER BY id"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["id"] in {first_observation, second_observation}
    assert rows[0]["entity_id"] == alice
    assert rows[0]["topic_key"] == "alerts"
    assert rows[0]["observation_type"] == "insight"
    assert "urgent and incident alerts" in rows[0]["summary"]
    assert "notifications" in rows[0]["keywords"]
    source_ids = {
        row["node_id"]
        for row in db._conn.execute(
            "SELECT node_id FROM memory_observation_sources WHERE observation_id = ?",
            (rows[0]["id"],),
        ).fetchall()
    }
    assert source_ids == {first_node, second_node}


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
        fact_type="world",
    )
    experience_node = _add_memory_node(
        db,
        time_key="2026-01-01 11:00:00",
        summary="Hermes previously routed Alice's alerts through Slack.",
        keywords=["Alice", "Slack"],
        fact_type="experience",
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
        obs_id = db.memory_upsert_observation(
            entity_id=entity_id,
            topic_key=f"task-{source_node_id}",
            topic_label=f"task-{source_node_id}",
            observation_type="task",
            summary=f"Task {source_node_id}",
            keywords=["task"],
            source_node_ids=[source_node_id],
            metadata={
                "task_status": status,
                "task_source": "inferred_from_observation",
            },
        )
        db._conn.execute(
            "UPDATE memory_observations SET last_supported_at = ?, updated_at = ? WHERE id = ?",
            (last_supported_at.isoformat(), last_supported_at.isoformat(), obs_id),
        )
        return obs_id

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
            "SELECT id, status, metadata FROM memory_observations ORDER BY id"
        ).fetchall()
    }
    assert rows[active_to_paused][0] == "active"
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
    obs_id = db.memory_upsert_observation(
        entity_id=entity_id,
        topic_key="memory-system",
        topic_label="memory-system",
        observation_type="task",
        summary="用户正在完善 Hermes Agent 的记忆系统。",
        keywords=["memory"],
        source_node_ids=[node_id],
        metadata={"task_status": "active", "task_source": "inferred_from_observation"},
    )
    idle_at = now - timedelta(days=10)
    db._conn.execute(
        "UPDATE memory_observations SET last_supported_at = ?, updated_at = ? WHERE id = ?",
        (idle_at.isoformat(), idle_at.isoformat(), obs_id),
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
        "SELECT metadata FROM memory_observations WHERE id = ?",
        (obs_id,),
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
        fact_type="experience",
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


def test_recall_formats_world_and_experience_sections(db, monkeypatch):
    _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="Alice prefers Slack for urgent alerts.",
        keywords=["Alice", "Slack"],
        fact_type="world",
    )
    _add_memory_node(
        db,
        time_key="2026-05-01 11:00:00",
        summary="Hermes recommended Slack alert routing for Alice.",
        keywords=["Alice", "Slack"],
        fact_type="experience",
    )
    monkeypatch.setattr(db, "_memory_search_vector", lambda *args, **kwargs: {})
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[json.dumps({"summary": "Alice Slack alerts", "keywords": ["Alice", "Slack"]})],
    )

    context = mgr.recall("Alice Slack alerts")

    assert "[World facts" in context
    assert "[Experience memories" in context
    assert "durable world facts" in context
    assert "prior assistant experiences" in context
    assert "Alice prefers Slack for urgent alerts." in context
    assert "Hermes recommended Slack alert routing for Alice." in context


def test_store_turn_consolidates_observation_for_entity_topic_bucket(db):
    retain_payload = {
        "facts": [
            {
                "text": "Alice prefers Slack for urgent alerts.",
                "keywords": ["Alice", "Slack", "alerts"],
                "topic": ["Slack alerts"],
                "fact_type": "world",
                "entities": [{"name": "Alice", "type": "PERSON"}],
            },
            {
                "text": "Alice dislikes email for urgent alerts.",
                "keywords": ["Alice", "email", "alerts"],
                "topic": ["Slack alerts"],
                "fact_type": "world",
                "entities": [{"name": "Alice", "type": "PERSON"}],
            },
            {
                "text": "Hermes previously recommended Slack alert routing for Alice.",
                "keywords": ["Hermes", "Slack", "routing", "Alice"],
                "topic": ["Slack alerts"],
                "fact_type": "experience",
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
                "category": "insight",
                "summary": "Alice's urgent alert workflow is Slack-centered.",
                "keywords": ["Slack", "alerts"],
                "confidence": 0.86,
                "metadata": {"insight_type": "workflow"},
            }),
        ],
    )

    assert mgr.store_turn("Alice urgent alerts", "Use Slack.") is True
    assert db._conn.execute("SELECT COUNT(*) FROM memory_observations").fetchone()[0] == 0

    report = mgr.reflect(dry_run=False, limit=10)
    assert report["observation_reflect"]["candidate_count"] == 3
    assert report["observations_consolidated"] == 1

    observation = db._conn.execute(
        "SELECT mo.summary, mo.topic_key, mo.observation_type, mo.metadata, en.name AS entity_name "
        "FROM memory_observations mo "
        "JOIN entity_nodes en ON en.id = mo.entity_id"
    ).fetchone()
    assert observation["entity_name"] == "Alice"
    assert observation["topic_key"] == "slack-alerts"
    assert observation["observation_type"] == "insight"
    assert json.loads(observation["metadata"])["insight_type"] == "workflow"
    assert observation["summary"] == "Alice's urgent alert workflow is Slack-centered."
    sources = db._conn.execute("SELECT node_id FROM memory_observation_sources").fetchall()
    assert len(sources) == 3


def test_store_turn_can_consolidate_task_observation(db):
    retain_payload = {
        "facts": [
            {
                "text": "用户正在排查 Hermes memory recall 的匹配问题。",
                "keywords": ["Hermes", "memory", "recall", "排查"],
                "topic": ["memory recall"],
                "fact_type": "world",
                "entities": [{"name": "Hermes Agent", "type": "PROJECT"}],
            },
            {
                "text": "用户计划修改 observation 生成逻辑以区分 insight 和 task。",
                "keywords": ["observation", "insight", "task", "修改"],
                "topic": ["memory recall"],
                "fact_type": "world",
                "entities": [{"name": "Hermes Agent", "type": "PROJECT"}],
            },
            {
                "text": "助手帮助用户实现记忆系统 reflect 和 observation decay 相关改动。",
                "keywords": ["reflect", "observation", "decay", "实现"],
                "topic": ["memory recall"],
                "fact_type": "experience",
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
                "category": "task",
                "summary": "用户正在迭代 Hermes Agent 的记忆系统，当前聚焦 recall、reflect 与 observation 生成逻辑。",
                "keywords": ["Hermes Agent", "memory", "recall", "observation"],
                "confidence": 0.82,
                "metadata": {
                    "task_status": "active",
                    "task_source": "inferred_from_observation",
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
    assert observation["observation_type"] == "task"
    assert "正在迭代 Hermes Agent" in observation["summary"]
    assert metadata["task_status"] == "active"
    assert metadata["task_source"] == "inferred_from_observation"
    assert metadata["goal"] == "完善 Hermes Agent 的长期记忆系统。"
    assert metadata["evidence"] == ["排查 memory recall", "修改 observation 生成逻辑"]
    assert metadata["steps"][0]["title"] == "排查 memory recall 匹配问题"
    assert metadata["steps"][0]["status"] == "done"
    assert metadata["steps"][0]["updated_at"] == "2026-05-14"
    assert metadata["steps"][1]["title"] == "将 observation 分类为 insight 和 task"
    assert metadata["steps"][1]["status"] == "active"


def test_reflect_matches_fact_to_task_by_entity_and_topic(db):
    hermes = db.entity_add_entity("Hermes Agent", "PROJECT")
    source_node = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="用户正在完善 Hermes Agent 的 reflect 机制。",
        keywords=["memory-reflect"],
    )
    db.entity_link_node(source_node, hermes)
    task_id = db.memory_upsert_observation(
        entity_id=hermes,
        topic_key="memory-reflect",
        topic_label="memory-reflect",
        observation_type="task",
        summary="用户正在完善 Hermes Agent 的长期记忆 reflect 机制。",
        keywords=["Hermes Agent", "memory", "reflect"],
        source_node_ids=[source_node],
        metadata={
            "task_status": "active",
            "task_source": "inferred_from_observation",
            "steps": [{"title": "设计 reflect 机制", "status": "active"}],
        },
    )
    new_node = _add_memory_node(
        db,
        time_key=MemoryNodeManager._memory_time_key(0),
        summary="用户要求在 run_agent 中每 5 轮调用 reflect。",
        keywords=["memory-reflect"],
    )
    db.entity_link_node(new_node, hermes)
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "category": "task",
                "summary": "用户正在完善 Hermes Agent 的 reflect 调度机制。",
                "keywords": ["Hermes Agent", "reflect"],
                "confidence": 0.9,
                "metadata": {
                    "task_status": "active",
                    "task_source": "inferred_from_observation",
                    "steps": [
                        {"title": "设计 reflect 机制", "status": "done"},
                        {"title": "每 5 轮对话调用 reflect", "status": "done"},
                    ],
                },
            })
        ],
    )

    report = mgr.reflect(dry_run=False, limit=10)

    assert report["observation_reflect"]["task_matched"] == 1
    assert report["observation_reflect"]["task_match_methods"] == {"entity_topic": 1}
    assert db.memory_observation_source_ids(task_id) == [source_node, new_node]
    row = db._conn.execute(
        "SELECT summary, metadata FROM memory_observations WHERE id = ?",
        (task_id,),
    ).fetchone()
    assert "reflect 调度机制" in row["summary"]
    metadata = json.loads(row["metadata"])
    assert metadata["task_match_methods"] == ["entity_topic"]
    assert metadata["steps"][1]["title"] == "每 5 轮对话调用 reflect"


def test_reflect_matches_fact_to_task_by_high_embedding_similarity(db):
    hermes = db.entity_add_entity("Hermes Agent", "PROJECT")
    alice = db.entity_add_entity("Alice", "PERSON")
    source_node = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="用户正在设计 memory reflect 机制。",
        keywords=["memory-system"],
    )
    db.entity_link_node(source_node, hermes)
    task_id = db.memory_upsert_observation(
        entity_id=hermes,
        topic_key="memory-system",
        topic_label="memory-system",
        observation_type="task",
        summary="用户正在完善 Hermes Agent memory reflect 任务。",
        keywords=["memory", "reflect"],
        source_node_ids=[source_node],
        metadata={"task_status": "active", "task_source": "inferred_from_observation"},
    )
    new_node = _add_memory_node(
        db,
        time_key=MemoryNodeManager._memory_time_key(0),
        summary="用户继续讨论 reflect 的低成本 task matching 方案。",
        keywords=["unrelated-topic"],
    )
    db.entity_link_node(new_node, alice)
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "category": "task",
                "summary": "用户正在完善 memory reflect 的 task matching 方案。",
                "keywords": ["memory", "reflect", "task matching"],
                "confidence": 0.88,
                "metadata": {
                    "task_status": "active",
                    "task_source": "inferred_from_observation",
                    "steps": [{"title": "设计低成本 task matching", "status": "active"}],
                },
            })
        ],
    )
    mgr._embedding_client = _KeywordEmbeddingClient()

    report = mgr.reflect(dry_run=False, limit=10)

    assert report["observation_reflect"]["task_matched"] == 1
    assert report["observation_reflect"]["task_match_methods"] == {"embedding": 1}
    assert db.memory_observation_source_ids(task_id) == [source_node, new_node]


def test_reflect_matches_action_like_fact_to_single_recent_active_task(db):
    hermes = db.entity_add_entity("Hermes Agent", "PROJECT")
    alice = db.entity_add_entity("Alice", "PERSON")
    source_node = _add_memory_node(
        db,
        time_key="2026-05-01 10:00:00",
        summary="用户正在完善 Hermes Agent 的记忆系统。",
        keywords=["memory-system"],
    )
    db.entity_link_node(source_node, hermes)
    task_id = db.memory_upsert_observation(
        entity_id=hermes,
        topic_key="memory-system",
        topic_label="memory-system",
        observation_type="task",
        summary="用户正在完善 Hermes Agent 的记忆系统。",
        keywords=["memory"],
        source_node_ids=[source_node],
        metadata={"task_status": "active", "task_source": "inferred_from_observation"},
    )
    new_node = _add_memory_node(
        db,
        time_key=MemoryNodeManager._memory_time_key(0),
        summary="用户要求继续修改 prompt。",
        keywords=["prompt-work"],
    )
    db.entity_link_node(new_node, alice)
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "category": "task",
                "summary": "用户正在继续完善记忆系统 prompt。",
                "keywords": ["memory", "prompt"],
                "confidence": 0.78,
                "metadata": {
                    "task_status": "active",
                    "task_source": "inferred_from_observation",
                    "steps": [{"title": "继续修改 prompt", "status": "active"}],
                },
            })
        ],
    )
    mgr._embedding_client = _OrthogonalTaskEmbeddingClient()

    report = mgr.reflect(dry_run=False, limit=10)

    assert report["observation_reflect"]["task_matched"] == 1
    assert report["observation_reflect"]["task_match_methods"] == {"recent_active_action": 1}
    assert db.memory_observation_source_ids(task_id) == [source_node, new_node]


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
    task_id = db.memory_upsert_observation(
        entity_id=hermes,
        topic_key="memory-system",
        topic_label="memory-system",
        observation_type="task",
        summary="用户正在完善 Hermes Agent 的记忆系统。",
        keywords=["memory"],
        source_node_ids=[source_node],
        metadata={"task_status": "active", "task_source": "inferred_from_observation"},
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

    assert report["observation_reflect"]["task_matched"] == 0
    assert db.memory_observation_source_ids(task_id) == [source_node]
    assert mgr.llm_prompts == []


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
    assert metadata["task_source"] == "inferred_from_observation"
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
    assert "task_status 定义" in OBSERVATION_CONSOLIDATION_PROMPT
    assert "task_status 定义" in OBSERVATION_UPDATE_PROMPT
    assert "首次生成 task 时不要输出 stale" in OBSERVATION_CONSOLIDATION_PROMPT
    assert "stale：" in OBSERVATION_UPDATE_PROMPT
    assert "stale 表示输入 observation" in OBSERVATION_MERGE_PROMPT
    assert "不要输出 stale" in OBSERVATION_CONSOLIDATION_PROMPT
    assert '"task_status": "active | blocked | paused"' in OBSERVATION_CONSOLIDATION_PROMPT
    assert '"task_status": "active | blocked | paused | stale"' in OBSERVATION_UPDATE_PROMPT
    assert '"task_status": "active | blocked | paused | stale"' in OBSERVATION_MERGE_PROMPT
    assert '"task_status": "active | blocked | paused | stale"' not in OBSERVATION_CONSOLIDATION_PROMPT


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
            fact_type="world" if idx < 3 else "experience",
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


def test_observation_consolidation_waits_for_incremental_sources(db):
    alice = db.entity_add_entity("Alice", "PERSON")
    mgr = _NoAsyncMemoryNodeManager(
        db,
        embedding_config={},
        llm_outputs=[
            json.dumps({
                "category": "insight",
                "summary": "Alice's urgent alert workflow is Slack-centered.",
                "keywords": ["Slack", "alerts"],
                "confidence": 0.86,
                "metadata": {"insight_type": "workflow"},
            }),
            json.dumps({
                "category": "insight",
                "summary": "Alice continues to prefer Slack for urgent alert routing.",
                "keywords": ["Slack", "alerts"],
                "confidence": 0.9,
                "metadata": {"insight_type": "workflow"},
            }),
        ],
    )

    def add_source(idx):
        node_id = _add_memory_node(
            db,
            time_key=f"2026-05-0{idx} 10:00:00",
            summary=f"Alice Slack alert fact {idx}.",
            keywords=["Slack alerts"],
        )
        db.entity_link_node(node_id, alice)
        mgr._maybe_consolidate_observations(
            node_id=node_id,
            topics=["slack-alerts"],
            linked_entities=[(alice, "Alice")],
        )
        return node_id

    for idx in range(1, 4):
        add_source(idx)
    first_summary = db._conn.execute("SELECT summary FROM memory_observations").fetchone()["summary"]
    assert first_summary == "Alice's urgent alert workflow is Slack-centered."

    add_source(4)
    second_summary = db._conn.execute("SELECT summary FROM memory_observations").fetchone()["summary"]
    assert second_summary == first_summary

    add_source(5)
    updated_summary = db._conn.execute("SELECT summary FROM memory_observations").fetchone()["summary"]
    assert updated_summary == "Alice continues to prefer Slack for urgent alert routing."
    update_prompt = mgr.llm_prompts[-1]
    assert "已有 observation" in update_prompt
    assert "Alice's urgent alert workflow is Slack-centered." in update_prompt
    assert "Alice Slack alert fact 4." in update_prompt
    assert "Alice Slack alert fact 5." in update_prompt
    assert "Alice Slack alert fact 1." not in update_prompt
    assert "Alice Slack alert fact 2." not in update_prompt
    assert "Alice Slack alert fact 3." not in update_prompt


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
        fact_type="world",
    )
    experience_prior = _add_memory_node(
        db,
        time_key="2026-05-01 11:00:00",
        summary="Hermes recommended Slack alert routing for Alice.",
        keywords=["Alice", "Slack"],
        fact_type="experience",
    )
    current_world = _add_memory_node(
        db,
        time_key="2026-05-01 12:00:00",
        summary="Alice wants urgent alerts in Slack.",
        keywords=["Alice", "Slack"],
        fact_type="world",
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
