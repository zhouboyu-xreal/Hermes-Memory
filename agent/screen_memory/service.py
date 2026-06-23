from __future__ import annotations

import atexit
import concurrent.futures
import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from agent.screen_memory.manager import ScreenMemoryManager
from agent.screen_memory.config import load_screen_memory_config
from agent.screen_memory.screen_db import ScreenMemoryDB
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="hermes-screen-memory",
)
_FUTURE_LOCK = threading.Lock()
_ACTIVE_FUTURE: Optional[concurrent.futures.Future] = None
_TICKER_LOCK = threading.Lock()
_TICKER_STOP_EVENT: Optional[threading.Event] = None
_TICKER_THREAD: Optional[threading.Thread] = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_utc_time(value: datetime) -> str:
    return _as_utc(value).isoformat().replace("+00:00", "Z")


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
    return previous is None or _as_utc(now) - previous >= interval


def _configured_model(config: Dict[str, Any]) -> str:
    model = config.get("model")
    if isinstance(model, dict):
        return str(model.get("default") or model.get("model") or "").strip()
    return str(model or "").strip()


def _inject_runtime_config(
    manager_config: Dict[str, Any],
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
        "screen_fact_generation",
        "screen_observation_generation",
        "task_workstream_generation",
    ):
        section = manager_config.get(section_name)
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


def _run_fact_extraction(
    manager: ScreenMemoryManager,
    state: Dict[str, Any],
    now: datetime,
) -> Optional[Dict[str, Any]]:
    now = _as_utc(now)
    schedule = manager.config.get("schedule", {})
    previous = _parse_state_time(state.get("last_fact_extraction_at"))
    if previous is None:
        previous = now - timedelta(
            minutes=max(1, int(schedule.get("initial_lookback_minutes", 30)))
        )
    stats = manager.update_screen_facts_table(
        start_time_str=_format_utc_time(previous),
        end_time_str=_format_utc_time(now),
        incremental=True,
    )
    if stats is not None:
        state["last_fact_extraction_at"] = _format_utc_time(now)
        state["last_fact_extraction_stats"] = stats
    return stats


def _run_observations(
    manager: ScreenMemoryManager,
    state: Dict[str, Any],
    now: datetime,
) -> Dict[str, Any]:
    now = _as_utc(now)
    stats = manager.update_screen_observation_tables()
    state["last_observation_at"] = _format_utc_time(now)
    state["last_observation_stats"] = stats
    return stats


def _run_tasks(
    manager: ScreenMemoryManager,
    state: Dict[str, Any],
    now: datetime,
) -> Dict[str, Any]:
    now = _as_utc(now)
    stats = manager.update_screen_task_tables()
    state["last_task_at"] = _format_utc_time(now)
    state["last_task_stats"] = stats
    return stats


def run_screen_memory_due_work(
    *,
    now: Optional[datetime] = None,
    force_phase: Optional[str] = None,
) -> Dict[str, Any]:
    from hermes_cli.config import load_config

    hermes_config = load_config() or {}
    manager_config = load_screen_memory_config(hermes_config)
    if not manager_config.get("enabled"):
        return {"status": "disabled"}
    source_error = _validate_sources(manager_config)
    if source_error:
        logger.warning("Screen memory skipped: %s", source_error)
        return {"status": "skipped", "reason": source_error}

    directory = _screen_memory_dir()
    directory.mkdir(parents=True, exist_ok=True)
    lock_handle = open(_lock_path(), "a+")
    llm_client = None
    screen_db = None
    manager = None
    try:
        try:
            import fcntl

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:
            pass
        except OSError:
            return {"status": "busy"}

        state = _load_state()
        current = _as_utc(now or _utc_now())
        schedule = manager_config.get("schedule", {})

        fact_extraction_due = force_phase == "fact_extraction" or (
            force_phase is None
            and _is_due(
                state.get("last_fact_extraction_at"),
                current,
                timedelta(minutes=max(1, int(schedule["fact_extraction_interval_minutes"]))),
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
        task_due = force_phase in {"task", "tasks"} or (
            force_phase is None
            and _is_due(
                state.get("last_task_at"),
                current,
                timedelta(hours=max(1, int(schedule.get("task_interval_hours", 6)))),
            )
        )
        if not any((fact_extraction_due, observation_due, task_due)):
            return {
                "status": "ok",
                "phases": {},
                "output_db": manager_config["database"]["cleaned_db"],
            }

        llm_client = _inject_runtime_config(manager_config, hermes_config)
        screen_db = ScreenMemoryDB(
            manager_config["database"]["cleaned_db"],
            incremental_mode=True,
        )
        manager = ScreenMemoryManager(
            manager_config,
            llm_client=llm_client,
            quiet=True,
            screen_db=screen_db,
        )
        phases: Dict[str, Any] = {}
        if fact_extraction_due:
            phases["fact_extraction"] = _run_fact_extraction(manager, state, current)
            _save_state(state)
        if observation_due:
            phases["observations"] = _run_observations(manager, state, current)
            _save_state(state)
        if task_due:
            phases["tasks"] = _run_tasks(manager, state, current)
            _save_state(state)

        if phases:
            logger.info(
                "Screen memory phases completed: %s (output=%s)",
                ", ".join(phases),
                manager.screen_db.output_path,
            )
        return {
            "status": "ok",
            "phases": phases,
            "output_db": manager.screen_db.output_path,
        }
    except Exception:
        logger.exception("Screen memory pipeline failed")
        return {"status": "error"}
    finally:
        if manager is not None:
            manager.close()
        elif screen_db is not None:
            screen_db.close()
        close_client = getattr(llm_client, "close", None)
        if callable(close_client):
            try:
                close_client()
            except Exception:
                pass
        lock_handle.close()


def schedule_screen_memory_tick() -> Optional[concurrent.futures.Future]:
    """Start due screen-memory work without blocking the caller."""
    global _ACTIVE_FUTURE
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
        if not load_screen_memory_config(config).get("enabled"):
            return None
    except Exception:
        return None
    with _FUTURE_LOCK:
        if _ACTIVE_FUTURE is not None and not _ACTIVE_FUTURE.done():
            return _ACTIVE_FUTURE
        _ACTIVE_FUTURE = _EXECUTOR.submit(run_screen_memory_due_work)
        return _ACTIVE_FUTURE


def _screen_memory_ticker_loop(
    stop_event: threading.Event,
    interval_seconds: float,
) -> None:
    logger.info(
        "Screen memory ticker started (interval=%ss)",
        int(interval_seconds),
    )
    while not stop_event.is_set():
        try:
            schedule_screen_memory_tick()
        except Exception as exc:
            logger.debug("Screen memory ticker error: %s", exc)
        if stop_event.wait(interval_seconds):
            break
    logger.debug("Screen memory ticker stopped")


def start_screen_memory_ticker(
    *,
    interval_seconds: float = 60.0,
) -> Optional[threading.Thread]:
    """Start one process-level ticker for interactive CLI/TUI runtimes."""
    global _TICKER_STOP_EVENT, _TICKER_THREAD
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
        if not load_screen_memory_config(config).get("enabled"):
            return None
    except Exception as exc:
        logger.debug("Screen memory ticker config check failed: %s", exc)
        return None

    clean_interval = max(1.0, float(interval_seconds or 60.0))
    with _TICKER_LOCK:
        if _TICKER_THREAD is not None and _TICKER_THREAD.is_alive():
            return _TICKER_THREAD
        _TICKER_STOP_EVENT = threading.Event()
        _TICKER_THREAD = threading.Thread(
            target=_screen_memory_ticker_loop,
            args=(_TICKER_STOP_EVENT, clean_interval),
            daemon=True,
            name="screen-memory-ticker",
        )
        _TICKER_THREAD.start()
        return _TICKER_THREAD


def stop_screen_memory_ticker(*, timeout: float = 2.0) -> None:
    """Stop the interactive process-level ticker if it is running."""
    global _TICKER_STOP_EVENT, _TICKER_THREAD
    with _TICKER_LOCK:
        stop_event = _TICKER_STOP_EVENT
        thread = _TICKER_THREAD
        _TICKER_STOP_EVENT = None
        _TICKER_THREAD = None
    if stop_event is not None:
        stop_event.set()
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=max(0.0, float(timeout or 0.0)))


atexit.register(stop_screen_memory_ticker)
