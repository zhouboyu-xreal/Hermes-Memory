from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict

from hermes_constants import get_hermes_home


SCREEN_MEMORY_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "screenpipe_db": "~/.screenpipe/db.sqlite",
    "openchronicle_db": "~/.openchronicle/index.db",
    "output_db": "",
    "fact_extraction_interval_minutes": 30,
    "fact_clustering_interval_hours": 2,
    "observation_interval_hours": 24,
    "initial_lookback_minutes": 30,
    "openchronicle": {
        "record_link_window_seconds": 6,
        "max_events_per_record": 3,
    },
    "cleaning_policy": {
        "active_interval": 2,
        "bg_interval": 30,
        "ax_trigger_interval": 10,
        "min_quality": 0.18,
        "ignored_apps": [
            "控制中心",
            "通知中心",
            "程序坞",
            "Control Center",
            "Notification Center",
            "Dock",
        ],
        "system_apps_keep_if_focused": ["系统设置", "System Settings"],
        "min_useful_chars": 8,
    },
    "view_generation": {
        "gap_minutes": 8,
        "max_minutes": 30,
    },
    "segment_generation": {
        "enable_LLM_summary": False,
        "gap_minutes": 8,
        "max_minutes": 30,
        "focus_switch_split_minutes": 5,
        "llm_budget": 0,
        "llm_timeout": 120,
    },
    "window_workstream_generation": {
        "enabled": True,
        "enable_LLM_summary": False,
        "min_relevance": 0.35,
        "title_similarity_threshold": 0.82,
        "max_member_views_for_matching": 12,
        "max_views_for_summary": 24,
        "pair_min_score": 0.45,
        "min_support_ratio": 0.5,
        "top_k": 3,
        "min_top_k_avg": 0.55,
        "cluster_seed_min_score": 0.65,
        "cluster_merge_passes": 1,
        "cluster_merge_support_ratio": 0.5,
        "cluster_merge_min_score": 0.35,
        "max_time_gap_days": 30,
        "primary_match_since": "current_week",
        "historical_min_entity_overlap": 2,
        "llm_budget": 0,
        "llm_timeout": 120,
    },
    "task_workstream_generation": {
        "enabled": False,
    },
    "report_block_generation": {
        "enabled": False,
    },
    "screen_memory_generation": {
        "enabled": True,
        "enable_LLM_fact_extraction": True,
        "enable_observation_generation": True,
        "enable_LLM_observation": True,
        "fallback_fact_when_llm_fails": False,
        "fallback_observation_without_llm": True,
        "fact_cluster_min_score": 0.42,
        "observation_merge_fact_min_score": 0.42,
        "observation_merge_min_score": 0.48,
        "observation_merge_support_ratio": 0.5,
        "fact_llm_budget": 400,
        "observation_llm_budget": 300,
        "llm_timeout": 120,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_embedding_config(config: Dict[str, Any]) -> Dict[str, Any]:
    raw = config.get("embedding", {})
    if not isinstance(raw, dict):
        return {"enabled": False, "api_key_env": "EMBEDDING_API_KEY"}

    embedding = copy.deepcopy(raw)
    if "enabled" not in embedding:
        embedding["enabled"] = bool(
            embedding.get("model") and embedding.get("base_url")
        )
    embedding.setdefault("api_key_env", "EMBEDDING_API_KEY")
    return embedding


def load_screen_memory_config(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if config is None:
        from hermes_cli.config import load_config

        config = load_config() or {}
    raw = config.get("screen_memory", {}) if isinstance(config, dict) else {}
    settings = _deep_merge(SCREEN_MEMORY_DEFAULTS, raw)
    output_db = settings.get("output_db")
    if not output_db:
        output_db = get_hermes_home() / "screen_memory" / "memory.db"
    else:
        output_db = Path(output_db).expanduser()
        if not output_db.is_absolute():
            output_db = get_hermes_home() / "screen_memory" / output_db
    settings["screenpipe_db"] = os.path.expanduser(str(settings.get("screenpipe_db") or ""))
    settings["openchronicle_db"] = os.path.expanduser(str(settings.get("openchronicle_db") or ""))
    settings["output_db"] = str(Path(output_db).expanduser())

    # Preserve the PME cleaner's original section names internally.
    return {
        "enabled": bool(settings.get("enabled")),
        "schedule": {
            "fact_extraction_interval_minutes": settings["fact_extraction_interval_minutes"],
            "fact_clustering_interval_hours": settings["fact_clustering_interval_hours"],
            "observation_interval_hours": settings["observation_interval_hours"],
            "initial_lookback_minutes": settings["initial_lookback_minutes"],
        },
        "database": {
            "screenpipe_db": settings["screenpipe_db"],
            "openchronicle_db": settings["openchronicle_db"],
            "cleaned_db": settings["output_db"],
        },
        "openchronicle": settings["openchronicle"],
        "cleaning_policy": settings["cleaning_policy"],
        "view_generation": settings["view_generation"],
        "segment_generation": settings["segment_generation"],
        "window_workstream_generation": settings["window_workstream_generation"],
        "task_workstream_generation": settings["task_workstream_generation"],
        "report_block_generation": settings["report_block_generation"],
        "screen_memory_generation": settings["screen_memory_generation"],
        "embedding": _load_embedding_config(config),
    }
