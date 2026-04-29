from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key
from gateway.user_memory import GatewayUserMemoryEntry, GatewayUserMemoryStore


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.FEISHU,
        user_id="ou_user_1",
        user_id_alt="on_union_1",
        chat_id="oc_chat_1",
        user_name="Ada",
        chat_type="dm",
    )


def _make_runner(tmp_path):
    from gateway.run import GatewayRunner

    source = _make_source()
    session_entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-1",
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
        search_sessions=MagicMock(return_value=[
            {
                "id": "sess-1",
                "source": "feishu",
                "user_id": "ou_user_1",
                "title": "current chat",
                "started_at": 1777437600.0,
                "last_active": 1777437900.0,
            },
            {
                "id": "sess-other",
                "source": "feishu",
                "user_id": "ou_user_2",
                "title": "other user",
                "started_at": 1777437600.0,
                "last_active": 1777437900.0,
            },
        ]),
        message_count=MagicMock(return_value=3),
    )
    return runner


@pytest.mark.asyncio
async def test_mydata_command_shows_user_scoped_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    source = _make_source()
    store = GatewayUserMemoryStore()
    store.save_entries(
        source,
        [
            GatewayUserMemoryEntry(
                id="mem-1",
                content="用户希望 Hermes 先给代码证据，再给通俗解释。",
                source="feishu",
            )
        ],
    )

    runner = _make_runner(tmp_path)
    event = MessageEvent(text="/mydata", source=source, message_id="m1")

    result = await runner._handle_mydata_command(event)

    assert "Your Hermes Data" in result
    assert "on_union_1" in result
    assert "sess-1" in result
    assert "mem-1" in result
    assert "代码证据" in result
    assert "sess-other" not in result


@pytest.mark.asyncio
async def test_mydata_command_empty_memory_does_not_expose_global_user_md(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    global_memory = hermes_home / "memories"
    global_memory.mkdir(parents=True)
    (global_memory / "USER.md").write_text("global user profile", encoding="utf-8")

    runner = _make_runner(tmp_path)
    event = MessageEvent(text="/mydata", source=_make_source(), message_id="m1")

    result = await runner._handle_mydata_command(event)

    assert "No user-scoped memories are stored yet." in result
    assert "does not expose profile-global USER.md" in result
    assert "global user profile" not in result
