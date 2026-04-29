from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key
from gateway.user_memory import (
    GatewayUserMemoryStore,
    load_builtin_gateway_memory,
)


def _make_source(user_id: str = "ou_user_1", user_id_alt: str = "on_union_1") -> SessionSource:
    return SessionSource(
        platform=Platform.FEISHU,
        user_id=user_id,
        user_id_alt=user_id_alt,
        chat_id=f"oc_chat_{user_id}",
        user_name="Ada",
        chat_type="dm",
    )


def _make_runner(source: SessionSource):
    from gateway.run import GatewayRunner

    session_entry = SessionEntry(
        session_key=build_session_key(source),
        session_id=f"sess-{source.user_id}",
        created_at=datetime(2026, 4, 29, 10, 0),
        updated_at=datetime(2026, 4, 29, 10, 5),
        origin=source,
        platform=Platform.FEISHU,
        chat_type="dm",
    )

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.FEISHU: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {}
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner._session_db = SimpleNamespace(
        search_sessions=MagicMock(return_value=[]),
        message_count=MagicMock(return_value=0),
    )
    return runner


@pytest.mark.asyncio
async def test_memory_add_list_delete_syncs_builtin_user_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    source = _make_source()
    runner = _make_runner(source)

    add_result = await runner._handle_memory_command(
        MessageEvent(text="/memory add 用户喜欢简洁回答。", source=source, message_id="m1")
    )

    assert "Saved memory" in add_result
    store = GatewayUserMemoryStore()
    entries = store.list_entries(source)
    assert len(entries) == 1
    assert entries[0].content == "用户喜欢简洁回答。"
    assert entries[0].kind == "fact"

    builtin_user, builtin_agent = load_builtin_gateway_memory(
        source,
        session_key=build_session_key(source),
    )
    assert builtin_user == ["用户喜欢简洁回答。"]
    assert builtin_agent == []

    list_result = await runner._handle_memory_command(
        MessageEvent(text="/memory list", source=source, message_id="m2")
    )
    assert entries[0].id not in list_result
    assert "用户喜欢简洁回答" in list_result

    delete_result = await runner._handle_memory_command(
        MessageEvent(text="/memory delete 简洁", source=source, message_id="m3")
    )

    assert "Deleted memory" in delete_result
    assert store.list_entries(source) == []
    builtin_user, _ = load_builtin_gateway_memory(
        source,
        session_key=build_session_key(source),
    )
    assert builtin_user == []


@pytest.mark.asyncio
async def test_memory_search_and_delete_are_keyword_based(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    source = _make_source()
    runner = _make_runner(source)

    await runner._handle_memory_command(
        MessageEvent(text="/memory add 用户喜欢先给结论再给依据。", source=source, message_id="m1")
    )

    search_result = await runner._handle_memory_command(
        MessageEvent(text="/memory search 结论", source=source, message_id="m2")
    )
    assert "用户喜欢先给结论" in search_result
    assert GatewayUserMemoryStore().list_entries(source)[0].id not in search_result

    edit_result = await runner._handle_memory_command(
        MessageEvent(text="/memory edit anything replacement", source=source, message_id="m3")
    )
    assert "Editing memories is not available" in edit_result


@pytest.mark.asyncio
async def test_remember_and_memory_delete_by_search(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    source = _make_source()
    runner = _make_runner(source)

    remember_result = await runner._handle_remember_command(
        MessageEvent(text="/remember 下周三提醒 Alex 发方案。", source=source, message_id="m1")
    )
    assert "Saved memory" in remember_result
    assert "Alex" in remember_result
    assert GatewayUserMemoryStore().list_entries(source)[0].id not in remember_result

    delete_result = await runner._handle_memory_command(
        MessageEvent(text="/memory delete Alex", source=source, message_id="m2")
    )

    assert "Deleted memory" in delete_result
    assert GatewayUserMemoryStore().list_entries(source) == []
    builtin_user, _ = load_builtin_gateway_memory(
        source,
        session_key=build_session_key(source),
    )
    assert builtin_user == []


@pytest.mark.asyncio
async def test_memory_command_is_scoped_by_gateway_user(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    source_a = _make_source("ou_user_1", "on_union_1")
    source_b = _make_source("ou_user_2", "on_union_2")

    await _make_runner(source_a)._handle_memory_command(
        MessageEvent(text="/memory add A 用户的记忆。", source=source_a, message_id="m1")
    )

    list_b = await _make_runner(source_b)._handle_memory_command(
        MessageEvent(text="/memory list", source=source_b, message_id="m2")
    )
    assert "No saved memories." in list_b
    assert "A 用户的记忆" not in list_b

    builtin_b, _ = load_builtin_gateway_memory(
        source_b,
        session_key=build_session_key(source_b),
    )
    assert builtin_b == []
