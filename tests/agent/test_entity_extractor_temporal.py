import json

from agent.entity_extractor import EntityExtractor


class _FakeDB:
    def __init__(self):
        self.entities = {}
        self.links = []
        self.edges = []

    def entity_add_entity(self, name, entity_type="CONCEPT", **kwargs):
        entity_id = len(self.entities) + 1
        self.entities[name] = {"id": entity_id, "type": entity_type}
        return entity_id

    def entity_link_node(self, node_id, entity_id, mention_count=1):
        self.links.append((node_id, entity_id))

    def entity_add_edge(self, source_entity_id, target_entity_id, relation_type, **kwargs):
        self.edges.append((source_entity_id, target_entity_id, relation_type))
        return len(self.edges)


class _FakeExtractor(EntityExtractor):
    def __init__(self, raw):
        super().__init__(_FakeDB())
        self.raw = raw

    def _call_llm_for_extraction(self, user_message, assistant_response):
        return self.raw


def test_entity_extractor_filters_plain_time_entities():
    raw = json.dumps({
        "entities": [
            {"name": "Alice", "type": "PERSON"},
            {"name": "昨天", "type": "TIME"},
            {"name": "2026-05-07", "type": "CONCEPT"},
            {"name": "Sprint 42", "type": "TOPIC"},
        ],
        "relations": [
            {"source": "Alice", "relation": "mentions", "target": "昨天"},
            {"source": "Alice", "relation": "mentions", "target": "Sprint 42"},
        ],
    })
    extractor = _FakeExtractor(raw)

    assert extractor.extract_from_turn(1, "昨天 Alice 提到 Sprint 42", "ok") is True

    assert set(extractor._db.entities) == {"Alice", "Sprint 42"}
    assert extractor._db.edges == [(1, 2, "mentions")]
