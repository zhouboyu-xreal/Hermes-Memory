import logging

import requests

from agent.embedding_client import _OpenAIBackend, _build_openai_embedding_url


def test_openai_backend_resolves_configured_environment_reference(monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-key")

    backend = _OpenAIBackend({"api_key": "${EMBEDDING_API_KEY}"})

    assert backend.api_key == "embedding-key"


def test_openai_backend_uses_embedding_environment_key(monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    backend = _OpenAIBackend({"api_key": ""})

    assert backend.api_key == "embedding-key"


def test_openai_backend_does_not_fall_back_to_model_api_keys(monkeypatch):
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    backend = _OpenAIBackend({"api_key": ""})

    assert backend.api_key == ""


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


def test_openai_backend_skips_request_without_embedding_key(
    monkeypatch, caplog
):
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.setattr(
        "agent.embedding_client.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("request should not be sent")
        ),
    )
    backend = _OpenAIBackend({"api_key": ""})

    assert backend.embed("test") is None
    assert "EMBEDDING_API_KEY is not set" in caplog.text


def test_openai_backend_logs_provider_error_response(monkeypatch, caplog):
    response = requests.Response()
    response.status_code = 401
    response._content = b'{"error":{"message":"User not found.","code":401}}'
    response.url = "https://embedding.example/v1/embeddings"

    monkeypatch.setattr(
        "agent.embedding_client.requests.post",
        lambda *args, **kwargs: response,
    )
    backend = _OpenAIBackend({"api_key": "invalid-key"})

    with caplog.at_level(logging.WARNING):
        assert backend.embed("test") is None

    assert "User not found." in caplog.text


def test_build_openai_embedding_url_preserves_explicit_endpoint():
    assert (
        _build_openai_embedding_url("https://api.z.ai/api/paas/v4/embeddings")
        == "https://api.z.ai/api/paas/v4/embeddings"
    )


def test_build_openai_embedding_url_supports_zai_style_base():
    assert (
        _build_openai_embedding_url("https://api.z.ai/api/paas/v4")
        == "https://api.z.ai/api/paas/v4/embeddings"
    )
