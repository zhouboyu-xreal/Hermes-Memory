from agent.embedding_client import _OpenAIBackend


def test_openai_backend_resolves_configured_environment_reference(monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-key")

    backend = _OpenAIBackend({"api_key": "${EMBEDDING_API_KEY}"})

    assert backend.api_key == "embedding-key"


def test_openai_backend_prefers_embedding_environment_key(monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    backend = _OpenAIBackend({"api_key": ""})

    assert backend.api_key == "embedding-key"


def test_openai_backend_does_not_send_unresolved_environment_reference(
    monkeypatch,
):
    monkeypatch.delenv("MISSING_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    backend = _OpenAIBackend(
        {"api_key": "${MISSING_EMBEDDING_API_KEY}"}
    )

    assert backend.api_key == ""
