"""Regression tests for memory provider selection during AIAgent init."""

import hashlib
from types import SimpleNamespace
from unittest.mock import patch


def test_blank_memory_provider_does_not_auto_enable_honcho():
    """Blank memory.provider should remain opt-out even if Honcho fallback looks configured."""
    cfg = {"memory": {"provider": ""}, "agent": {}}
    honcho_cfg = SimpleNamespace(enabled=True, api_key="stale-key", base_url=None)

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.save_config") as save_config,
        patch(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            return_value=honcho_cfg,
        ) as from_global_config,
        patch("plugins.memory.load_memory_provider") as load_memory_provider,
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
        )

    assert agent._memory_manager is None
    from_global_config.assert_not_called()
    load_memory_provider.assert_not_called()
    save_config.assert_not_called()


def test_gateway_builtin_memory_is_scoped_by_feishu_user_identity(tmp_path, monkeypatch):
    """Built-in USER.md/MEMORY.md must not be shared across Feishu users."""
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    cfg = {
        "memory": {
            "memory_enabled": True,
            "user_profile_enabled": True,
            "memory_char_limit": 2200,
            "user_char_limit": 1375,
            "provider": "",
        },
        "agent": {},
    }

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        alice = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            platform="feishu",
            user_id="ou_alice",
            user_id_alt="on_alice",
            chat_id="oc_chat",
            gateway_session_key="agent:main:feishu:group:oc_chat:on_alice",
        )
        bob = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            platform="feishu",
            user_id="ou_bob",
            user_id_alt="on_bob",
            chat_id="oc_chat",
            gateway_session_key="agent:main:feishu:group:oc_chat:on_bob",
        )

    alice._memory_store.add("user", "Name: Alice")
    bob._memory_store.add("user", "Name: Bob")

    alice_key = "agent:main:feishu:group:oc_chat:on_alice"
    bob_key = "agent:main:feishu:group:oc_chat:on_bob"
    alice_digest = hashlib.sha256(alice_key.encode("utf-8")).hexdigest()[:16]
    bob_digest = hashlib.sha256(bob_key.encode("utf-8")).hexdigest()[:16]
    alice_user = hermes_home / "memories" / "gateway" / "feishu" / f"user_{alice_digest}" / "USER.md"
    bob_user = hermes_home / "memories" / "gateway" / "feishu" / f"user_{bob_digest}" / "USER.md"

    assert alice_user.read_text(encoding="utf-8") == "Name: Alice"
    assert bob_user.read_text(encoding="utf-8") == "Name: Bob"
    assert not (hermes_home / "memories" / "USER.md").exists()
