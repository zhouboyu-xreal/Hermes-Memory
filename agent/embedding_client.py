#!/usr/bin/env python3
"""
Embedding Client for Hermes Agent.

Generates text embeddings using configurable backends:
  - ``ollama`` — Ollama's local ``/api/embeddings`` endpoint
  - ``openai`` — Any OpenAI-compatible ``/v1/embeddings`` endpoint

Configuration (config.yaml)::

    embedding:
      provider: "ollama"          # "ollama" | "openai"
      model: "qwen3-embedding:8b" # or "text-embedding-3-small"
      base_url: "http://127.0.0.1:11434"  # Ollama default
      api_key: ""                 # only needed for OpenAI backends
      dimensions: 4096            # vector dimension
      normalize: true             # L2-normalize before returning

Usage::

    from agent.embedding_client import EmbeddingClient

    client = EmbeddingClient()
    vec = client.embed_text("用户喜欢喝美式咖啡")
    # vec.shape == (1, 4096), dtype=float32
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────

DEFAULT_CONFIG: Dict[str, Any] = {
    "provider": "ollama",
    "model": "qwen3-embedding:8b",
    "base_url": "http://127.0.0.1:11434",
    "api_key": "",
    "dimensions": 4096,
    "normalize": True,
}

# ── Embedding dimension auto-detection table ──────────────────────────────

KNOWN_MODEL_DIMS: Dict[str, int] = {
    "qwen3-embedding:8b": 4096,
    "qwen3-embedding": 4096,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "nomic-embed-text": 768,
    "nomic-embed-text-v1": 768,
    "snowflake-arctic-embed": 1024,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "all-MiniLM-L6-v2": 384,
    "bge-small-en-v1.5": 384,
    "bge-base-en-v1.5": 768,
    "bge-large-en-v1.5": 1024,
    "bge-m3": 1024,
    "gte-small": 384,
    "gte-base": 768,
    "gte-large": 1024,
}

# ── Config helpers ────────────────────────────────────────────────────────


def _load_config() -> Dict[str, Any]:
    """Load embedding config from config.yaml, falling back to defaults."""
    try:
        from hermes_cli.config import load_config

        config = load_config()
        emb_cfg = config.get("embedding", {})
    except Exception:
        emb_cfg = {}

    merged = dict(DEFAULT_CONFIG)
    merged.update(emb_cfg)
    return merged


# ── Backend implementations ───────────────────────────────────────────────


class _OllamaBackend:
    """Embedding via Ollama's ``/api/embeddings`` endpoint."""

    def __init__(self, config: Dict[str, Any]) -> None:
        base = config.get("base_url", DEFAULT_CONFIG["base_url"]).rstrip("/")
        self.url = f"{base}/api/embeddings"
        self.model = config.get("model", DEFAULT_CONFIG["model"])
        self.timeout = config.get("timeout", 120)

    def embed(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text string."""
        data = {
            "model": self.model,
            "prompt": text,
        }
        try:
            resp = requests.post(self.url, json=data, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()
            return result.get("embedding")
        except requests.exceptions.RequestException as e:
            logger.warning("Ollama embedding failed: %s", e)
            return None

    def embed_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings for a batch of texts.

        Some Ollama versions support ``input`` as a list; others only accept
        one prompt at a time. Falls back to sequential if batch fails.
        """
        # Try batch path first
        data = {
            "model": self.model,
            "input": texts,
        }
        try:
            resp = requests.post(self.url, json=data, timeout=self.timeout * len(texts))
            resp.raise_for_status()
            result = resp.json()
            embeddings = result.get("embeddings")
            if embeddings and len(embeddings) == len(texts):
                return embeddings
        except Exception:
            pass

        # Sequential fallback
        results: List[List[float]] = []
        for t in texts:
            emb = self.embed(t)
            if emb is not None:
                results.append(emb)
            else:
                # Return None-style marker so caller can detect failure
                results.append([0.0] * DEFAULT_CONFIG["dimensions"])
        return results

    @property
    def dimension(self) -> int:
        """Return the known dimension for this model, or config default."""
        # Check known table
        for key, dim in KNOWN_MODEL_DIMS.items():
            if key in self.model.lower():
                return dim
        return DEFAULT_CONFIG["dimensions"]


class _OpenAIBackend:
    """Embedding via OpenAI-compatible ``/v1/embeddings`` endpoint.

    Supports:
      - OpenAI API (api.openai.com)
      - Local OpenAI-compatible servers (vLLM, llama.cpp, LocalAI)
      - Cloud providers with OpenAI-compatible embedding APIs
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        base = config.get("base_url", "").rstrip("/")
        if not base:
            base = "https://api.openai.com"
        self.url = f"{base}/v1/embeddings"
        self.model = config.get("model", "text-embedding-3-small")
        self.api_key = config.get("api_key", "")
        self.timeout = config.get("timeout", 60)

    def embed(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text string."""
        data = {
            "model": self.model,
            "input": text,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = requests.post(self.url, json=data, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()
            items = result.get("data", [])
            if items:
                return items[0].get("embedding")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning("OpenAI-compatible embedding failed: %s", e)
            return None

    def embed_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings for a batch of texts."""
        data = {
            "model": self.model,
            "input": texts,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = requests.post(self.url, json=data, headers=headers, timeout=self.timeout * 2)
            resp.raise_for_status()
            result = resp.json()
            # Sort by index to preserve order
            items = sorted(result.get("data", []), key=lambda x: x.get("index", 0))
            return [item["embedding"] for item in items]
        except requests.exceptions.RequestException as e:
            logger.warning("OpenAI batch embedding failed, falling back to sequential: %s", e)
            # Sequential fallback
            results: List[List[float]] = []
            for t in texts:
                emb = self.embed(t)
                results.append(emb if emb else [0.0] * DEFAULT_CONFIG["dimensions"])
            return results

    @property
    def dimension(self) -> int:
        for key, dim in KNOWN_MODEL_DIMS.items():
            if key in self.model.lower():
                return dim
        return DEFAULT_CONFIG["dimensions"]


# ── Main client ───────────────────────────────────────────────────────────


class EmbeddingClient:
    """Text embedding generator with configurable backends.

    Typical usage::

        client = EmbeddingClient()
        vector = client.embed_text("用户喜欢喝美式咖啡")
        # → np.ndarray shape (1, 4096), float32, L2-normalized
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or _load_config()
        self._backend = self._build_backend()

        # Re-check config for normalization and dimension override
        self._normalize = bool(self._config.get("normalize", True))
        self._dim = int(self._config.get("dimensions", 0)) or self._backend.dimension

    def _build_backend(self) -> Any:
        provider = self._config.get("provider", "ollama").lower()
        if provider == "ollama":
            return _OllamaBackend(self._config)
        elif provider == "openai":
            return _OpenAIBackend(self._config)
        else:
            logger.warning(
                "Unknown embedding provider '%s'; falling back to 'ollama'",
                provider,
            )
            return _OllamaBackend(self._config)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension for the configured model."""
        return self._dim

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Embed a single text string into a (1, D) float32 numpy array.

        Returns ``None`` if embedding fails.
        """
        if not text or not text.strip():
            logger.debug("Empty text passed to embed_text")
            return None

        raw = self._backend.embed(text.strip())
        if raw is None:
            return None

        vec = np.asarray(raw, dtype=np.float32).reshape(1, -1)

        if self._normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

        return vec

    def embed_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Embed a list of text strings into a (N, D) float32 numpy array.

        Returns ``None`` if all texts fail to embed.
        """
        valid = [t.strip() for t in texts if t and t.strip()]
        if not valid:
            return None

        raw_list = self._backend.embed_batch(valid)
        if raw_list is None:
            return None

        vecs = np.asarray(raw_list, dtype=np.float32)

        if self._normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms

        return vecs

    def normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2-normalize a vector or array of vectors in-place.

        Accepts (D,) or (N, D) arrays. Zeros remain zero (norm clamped to 1).
        """
        vec = np.asarray(vec, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vec / norms

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two (1, D) arrays."""
        v1 = np.asarray(vec1, dtype=np.float32).flatten()
        v2 = np.asarray(vec2, dtype=np.float32).flatten()
        dot = np.dot(v1, v2)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(dot / (n1 * n2))


# ── Quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    client = EmbeddingClient()
    print(f"Backend: {client._config.get('provider')}")
    print(f"Model:   {client._config.get('model')}")
    print(f"Dim:     {client.dimension}")

    test_text = "用户喜欢喝美式咖啡"
    vec = client.embed_text(test_text)
    if vec is not None:
        print(f"✅ Embedding shape: {vec.shape}, dtype: {vec.dtype}")
        print(f"   First 5 values: {vec[0, :5].tolist()}")
        print(f"   Norm: {np.linalg.norm(vec):.6f}")

    # Batch test
    texts = ["今天天气怎么样", "帮我设置闹钟", "给张三打电话"]
    batch = client.embed_batch(texts)
    if batch is not None:
        print(f"✅ Batch shape: {batch.shape}")
        print(f"   Norms: {[f'{np.linalg.norm(batch[i]):.4f}' for i in range(len(texts))]}")

    # Cosine similarity
    if vec is not None and batch is not None:
        sim = EmbeddingClient.cosine_similarity(vec, batch[0])
        print(f"✅ Cosine sim 'coffee' ↔ 'weather': {sim:.4f}")
