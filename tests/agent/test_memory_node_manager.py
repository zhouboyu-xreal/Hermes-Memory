import json

import numpy as np
import pytest

from agent.memory_node_manager import MemoryNodeManager
from agent.memory_node_manager import (
    CAUSAL_RELATION_TYPE_TEXT,
    RELATION_PROMPT_TEMPLATE,
    RETAIN_FACT_EXTRACTION_PROMPT,
)
from hermes_state import SessionDB


class _FakeEmbeddingClient:
    def embed_text(self, text):
        if not text:
            return None
        return np.ones((1, 1536), dtype=np.float32)


class _NoAsyncMemoryNodeManager(MemoryNodeManager):
    def __init__(self, *args, llm_outputs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._embedding_client = _FakeEmbeddingClient()
        self._llm_outputs = list(llm_outputs or [])
        self.async_calls = []

    def _call_llm(self, prompt):
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
        "SELECT id, summary, keywords, tags FROM memory_nodes ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["summary"] == retain_payload["facts"][0]["text"]
    assert rows[1]["summary"] == retain_payload["facts"][1]["text"]
    assert "Alice Slack email" == rows[0]["keywords"]
    assert "fact_type:world" in json.loads(rows[0]["tags"])
    assert "fact_type:experience" in json.loads(rows[1]["tags"])

    detail = db._conn.execute(
        "SELECT original_dialog FROM memory_leaf_details WHERE node_id = ?",
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
    assert mgr.async_calls[0]["run_entity_extraction"] is False
    assert mgr.async_calls[1]["run_entity_extraction"] is False
    assert mgr.async_calls[0]["keywords"] == ["Alice", "Slack", "email"]


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
        "SELECT summary, keywords, tags FROM memory_nodes"
    ).fetchone()
    assert row["summary"] == summary_payload["summary"]
    assert row["keywords"] == "PostgreSQL project"
    assert "fact_kind:conversation_summary" in json.loads(row["tags"])
    assert mgr.async_calls[0]["run_entity_extraction"] is True


def test_retain_and_relation_prompts_share_relation_type_contract():
    assert CAUSAL_RELATION_TYPE_TEXT in RETAIN_FACT_EXTRACTION_PROMPT
    assert CAUSAL_RELATION_TYPE_TEXT in RELATION_PROMPT_TEMPLATE
    assert "Reason/HinderedBy" not in RETAIN_FACT_EXTRACTION_PROMPT
    assert "Reason/HinderedBy" not in RELATION_PROMPT_TEMPLATE


def test_store_turn_filters_plain_time_expressions_from_fact_entities(db):
    retain_payload = {
        "facts": [
            {
                "text": "Alice wants Slack alerts during the Spring Festival launch window.",
                "keywords": ["Alice", "Slack", "春节"],
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


def _add_memory_node(db, *, time_key, summary, keywords):
    return db.memory_add_node(
        time_key=time_key,
        summary=summary,
        keywords=keywords,
        original_dialog="{}",
        query_embedding=np.ones((1, 1536), dtype=np.float32),
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
