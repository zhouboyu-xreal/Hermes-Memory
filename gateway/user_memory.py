"""Per-gateway-user memory storage helpers.

This is intentionally separate from the built-in ``tools.memory_tool``
MEMORY.md / USER.md store.  Gateway users can be distinct people sharing one
Hermes profile, so user-facing management must be scoped by platform identity.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gateway.config import Platform
from gateway.session import SessionSource
from hermes_constants import get_hermes_home
from utils import atomic_replace


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class GatewayUserMemoryEntry:
    id: str
    content: str
    created_at: str = ""
    updated_at: str = ""
    source: str = ""


def gateway_user_identity(source: SessionSource | None) -> tuple[str, str]:
    """Return ``(platform, stable_user_key)`` for a gateway source."""
    if not source:
        return "unknown", "unknown"

    platform = source.platform.value if isinstance(source.platform, Platform) else str(source.platform)
    raw = source.user_id_alt or source.user_id or source.chat_id or "unknown"
    raw = str(raw).strip() or "unknown"

    safe = _SAFE_NAME_RE.sub("_", raw).strip("._-")
    if not safe:
        safe = "unknown"
    if len(safe) > 96:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        safe = f"{safe[:72]}_{digest}"
    return platform, safe


class GatewayUserMemoryStore:
    """Small JSON-file store for memories owned by one gateway user."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or (get_hermes_home() / "memories" / "gateway_users")

    def path_for(self, source: SessionSource | None) -> Path:
        platform, user_key = gateway_user_identity(source)
        return self.base_dir / platform / f"{user_key}.json"

    def list_entries(self, source: SessionSource | None) -> list[GatewayUserMemoryEntry]:
        path = self.path_for(source)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

        raw_entries: Any
        if isinstance(data, dict):
            raw_entries = data.get("memories", [])
        elif isinstance(data, list):
            raw_entries = data
        else:
            raw_entries = []

        entries: list[GatewayUserMemoryEntry] = []
        for idx, item in enumerate(raw_entries, start=1):
            if isinstance(item, str):
                content = item.strip()
                if content:
                    entries.append(GatewayUserMemoryEntry(id=f"legacy-{idx}", content=content))
                continue
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            entries.append(
                GatewayUserMemoryEntry(
                    id=str(item.get("id") or f"mem-{idx}"),
                    content=content,
                    created_at=str(item.get("created_at") or ""),
                    updated_at=str(item.get("updated_at") or ""),
                    source=str(item.get("source") or ""),
                )
            )
        return entries

    def save_entries(self, source: SessionSource | None, entries: list[GatewayUserMemoryEntry]) -> None:
        """Persist entries for future management flows."""
        path = self.path_for(source)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "memories": [
                {
                    "id": entry.id,
                    "content": entry.content,
                    "created_at": entry.created_at,
                    "updated_at": entry.updated_at,
                    "source": entry.source,
                }
                for entry in entries
            ]
        }
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=".user_mem_")
        try:
            with open(fd, "w", encoding="utf-8", closefd=True) as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.write("\n")
            atomic_replace(tmp_path, path)
        except BaseException:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass
            raise


def gateway_builtin_memory_dir(
    source: SessionSource | None,
    *,
    session_key: str | None = None,
) -> Path | None:
    """Return the built-in MemoryStore directory used by gateway AIAgents."""
    if not source:
        return None
    platform = source.platform.value if isinstance(source.platform, Platform) else str(source.platform)
    platform_key = platform.strip().lower()
    if not platform_key or platform_key in {"cli", "local"}:
        return None

    stable_identity = (
        (session_key or "").strip()
        or (source.user_id_alt or "").strip()
        or (source.user_id or "").strip()
        or (source.chat_id or "").strip()
    )
    if not stable_identity:
        return None

    digest = hashlib.sha256(stable_identity.encode("utf-8")).hexdigest()[:16]
    return get_hermes_home() / "memories" / "gateway" / platform_key / f"user_{digest}"


def load_builtin_gateway_memory(
    source: SessionSource | None,
    *,
    session_key: str | None = None,
) -> tuple[list[str], list[str]]:
    """Load built-in ``(USER.md entries, MEMORY.md entries)`` for a gateway user."""
    memory_dir = gateway_builtin_memory_dir(source, session_key=session_key)
    if memory_dir is None:
        return [], []

    from tools.memory_tool import MemoryStore

    store = MemoryStore(memory_dir=memory_dir)
    store.load_from_disk()
    return list(store.user_entries), list(store.memory_entries)
