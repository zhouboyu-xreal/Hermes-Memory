from __future__ import annotations

import concurrent.futures
import contextlib
import io
import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from agent.screen_memory.cleaner import ScreenMemoryCleaner
from agent.screen_memory.config import load_screen_memory_config
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="hermes-screen-memory",
)
_FUTURE_LOCK = threading.Lock()
_ACTIVE_FUTURE: Optional[concurrent.futures.Future] = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_state_time(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _screen_memory_dir() -> Path:
    return get_hermes_home() / "screen_memory"


def _state_path() -> Path:
    return _screen_memory_dir() / "schedule_state.json"


def _lock_path() -> Path:
    return _screen_memory_dir() / ".pipeline.lock"


def _load_state() -> Dict[str, Any]:
    path = _state_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(state: Dict[str, Any]) -> None:
    directory = _screen_memory_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = _state_path()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _is_due(last_run: Any, now: datetime, interval: timedelta) -> bool:
    previous = _parse_state_time(last_run)
    return previous is None or now - previous >= interval


def _configured_model(config: Dict[str, Any]) -> str:
    model = config.get("model")
    if isinstance(model, dict):
        return str(model.get("default") or model.get("model") or "").strip()
    return str(model or "").strip()


def _inject_runtime_config(
    cleaner_config: Dict[str, Any],
    hermes_config: Dict[str, Any],
) -> Any:
    model = _configured_model(hermes_config)
    if not model:
        return None
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(target_model=model)
    except Exception as exc:
        logger.warning("Screen memory could not resolve Hermes LLM runtime: %s", exc)
        return None
    for section_name in (
        "segment_generation",
        "window_workstream_generation",
        "screen_memory_generation",
    ):
        section = cleaner_config.get(section_name)
        if not isinstance(section, dict):
            continue
        section["llm_model"] = model
        section["llm_base_url"] = runtime.get("base_url") or ""
        section["api_key"] = runtime.get("api_key") or ""
    try:
        from agent.auxiliary_client import resolve_provider_client

        client, _resolved_model = resolve_provider_client(
            runtime.get("provider") or "auto",
            model=model,
            explicit_base_url=runtime.get("base_url"),
            explicit_api_key=runtime.get("api_key"),
            api_mode=runtime.get("api_mode"),
            main_runtime=runtime,
        )
        return client
    except Exception as exc:
        logger.warning("Screen memory could not create Hermes LLM client: %s", exc)
        return None


def _validate_sources(config: Dict[str, Any]) -> Optional[str]:
    database = config.get("database", {})
    for name in ("screenpipe_db", "openchronicle_db"):
        path = Path(str(database.get(name) or "")).expanduser()
        if not str(path) or not path.is_file():
            return f"{name}_not_found:{path}"
    return None


def _run_ingest(
    cleaner: ScreenMemoryCleaner,
    state: Dict[str, Any],
    now: datetime,
) -> Optional[Dict[str, Any]]:
    schedule = cleaner.config.get("schedule", {})
    previous = _parse_state_time(state.get("last_ingest_at"))
    if previous is None:
        previous = now - timedelta(
            minutes=max(1, int(schedule.get("initial_lookback_minutes", 30)))
        )
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        stats = cleaner.clean(
            start_time_str=previous.isoformat(),
            end_time_str=now.isoformat(),
            incremental=True,
            generate_screen_facts=True,
            update_window_workstreams=True,
        )
    if output.getvalue().strip():
        logger.debug("Screen memory ingest output:\n%s", output.getvalue().strip())
    if stats is not None:
        state["last_ingest_at"] = now.isoformat()
        state["last_ingest_stats"] = stats
    return stats


def _run_fact_clustering(
    cleaner: ScreenMemoryCleaner,
    state: Dict[str, Any],
    now: datetime,
) -> Dict[str, Any]:
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        connection = cleaner.ensure_cleaned_db()
        try:
            stats = cleaner.update_screen_fact_cluster_tables(connection)
        finally:
            connection.close()
    if output.getvalue().strip():
        logger.debug("Screen memory fact clustering output:\n%s", output.getvalue().strip())
    state["last_fact_clustering_at"] = now.isoformat()
    state["last_fact_clustering_stats"] = stats
    return stats


def _run_observations(
    cleaner: ScreenMemoryCleaner,
    state: Dict[str, Any],
    now: datetime,
) -> Dict[str, Any]:
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        connection = cleaner.ensure_cleaned_db()
        try:
            # A daily observation run first catches any facts left unclustered by
            # a missed six-hour tick.
            cluster_stats = cleaner.update_screen_fact_cluster_tables(connection)
            observation_stats = cleaner.update_screen_observation_tables(connection, None)
        finally:
            connection.close()
    if output.getvalue().strip():
        logger.debug("Screen memory observation output:\n%s", output.getvalue().strip())
    stats = {
        "fact_clustering": cluster_stats,
        "observations": observation_stats,
    }
    state["last_observation_at"] = now.isoformat()
    state["last_observation_stats"] = stats
    return stats


def run_screen_memory_due_work(
    *,
    now: Optional[datetime] = None,
    force_phase: Optional[str] = None,
) -> Dict[str, Any]:
    from hermes_cli.config import load_config

    hermes_config = load_config() or {}
    cleaner_config = load_screen_memory_config(hermes_config)
    if not cleaner_config.get("enabled"):
        return {"status": "disabled"}
    source_error = _validate_sources(cleaner_config)
    if source_error:
        logger.warning("Screen memory skipped: %s", source_error)
        return {"status": "skipped", "reason": source_error}

    directory = _screen_memory_dir()
    directory.mkdir(parents=True, exist_ok=True)
    lock_handle = open(_lock_path(), "a+")
    llm_client = None
    try:
        try:
            import fcntl

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:
            pass
        except OSError:
            return {"status": "busy"}

        state = _load_state()
        current = (now or _utc_now()).astimezone(timezone.utc)
        schedule = cleaner_config.get("schedule", {})

        ingest_due = force_phase == "ingest" or (
            force_phase is None
            and _is_due(
                state.get("last_ingest_at"),
                current,
                timedelta(minutes=max(1, int(schedule["ingest_interval_minutes"]))),
            )
        )
        cluster_due = force_phase == "cluster" or (
            force_phase is None
            and _is_due(
                state.get("last_fact_clustering_at"),
                current,
                timedelta(hours=max(1, int(schedule["fact_clustering_interval_hours"]))),
            )
        )
        observation_due = force_phase == "observation" or (
            force_phase is None
            and _is_due(
                state.get("last_observation_at"),
                current,
                timedelta(hours=max(1, int(schedule["observation_interval_hours"]))),
            )
        )
        if not any((ingest_due, cluster_due, observation_due)):
            return {
                "status": "ok",
                "phases": {},
                "output_db": cleaner_config["database"]["cleaned_db"],
            }

        llm_client = _inject_runtime_config(cleaner_config, hermes_config)
        cleaner = ScreenMemoryCleaner(cleaner_config, llm_client=llm_client)
        phases: Dict[str, Any] = {}
        if ingest_due:
            phases["ingest"] = _run_ingest(cleaner, state, current)
            _save_state(state)
        if cluster_due:
            phases["fact_clustering"] = _run_fact_clustering(
                cleaner,
                state,
                current,
            )
            _save_state(state)
        if observation_due:
            phases["observations"] = _run_observations(cleaner, state, current)
            _save_state(state)

        if phases:
            logger.info(
                "Screen memory phases completed: %s (output=%s)",
                ", ".join(phases),
                cleaner.cleaned_db,
            )
        return {
            "status": "ok",
            "phases": phases,
            "output_db": cleaner.cleaned_db,
        }
    except Exception:
        logger.exception("Screen memory pipeline failed")
        return {"status": "error"}
    finally:
        close_client = getattr(llm_client, "close", None)
        if callable(close_client):
            try:
                close_client()
            except Exception:
                pass
        lock_handle.close()


def schedule_screen_memory_tick() -> Optional[concurrent.futures.Future]:
    """Start due screen-memory work without blocking the gateway cron tick."""
    global _ACTIVE_FUTURE
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
        if not bool((config.get("screen_memory") or {}).get("enabled", False)):
            return None
    except Exception:
        return None
    with _FUTURE_LOCK:
        if _ACTIVE_FUTURE is not None and not _ACTIVE_FUTURE.done():
            return _ACTIVE_FUTURE
        _ACTIVE_FUTURE = _EXECUTOR.submit(run_screen_memory_due_work)
        return _ACTIVE_FUTURE
