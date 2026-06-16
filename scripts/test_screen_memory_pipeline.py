#!/usr/bin/env python3
"""Exercise the screen-memory pipeline against recorded source databases.

Two modes are supported:

* once: fact_extraction the complete source time range, then run fact clustering and
  observation generation once.
* timeline: start at the earliest source timestamp and simulate the production
  schedule in chronological order.

The script uses an isolated output database and in-memory schedule state. It
does not read or update the production screen-memory schedule_state.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.screen_memory.manager import ScreenMemoryManager, parse_timestamp_to_utc
from agent.screen_memory.config import load_screen_memory_config
from agent.screen_memory.screen_db import ScreenMemoryDB
from agent.screen_memory.service import (
    _format_utc_time,
    _inject_runtime_config,
    _run_fact_extraction,
    _run_observations,
)
from hermes_cli.config import load_config
from hermes_constants import get_hermes_home


DEFAULT_OUTPUT_DIR = REPO_ROOT / "tmp" / "screen_memory_pipeline_test"
DEFAULT_LOG_PATH = DEFAULT_OUTPUT_DIR / "screen_memory_pipeline_test.log"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "screen_memory_pipeline_report.json"

DETAIL_TABLES = (
    "window_workstream",
    "screen_facts",
    "screen_fact_clusters",
    "screen_fact_cluster_members",
    "screen_observations",
    "screen_observation_facts",
)
COUNT_TABLES = (
    "records",
    "openchronicle_events",
    "views",
    "window_workstream",
    "window_workstream_members",
    "screen_facts",
    "screen_fact_clusters",
    "screen_fact_cluster_members",
    "screen_observations",
    "screen_observation_facts",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run screen-memory fact_extraction, fact clustering, and observation "
            "generation against Screenpipe and OpenChronicle databases."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("once", "timeline"),
        default="once",
        help="Run one full-range pipeline or simulate the configured schedule.",
    )
    parser.add_argument(
        "--screenpipe-db",
        type=Path,
        required=True,
        help="Path to the Screenpipe SQLite database.",
    )
    parser.add_argument(
        "--openchronicle-db",
        type=Path,
        required=True,
        help="Path to the OpenChronicle SQLite database.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--db-name",
        default="screen_memory_pipeline_test.db",
    )
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--fact-extraction-interval-minutes",
        dest="fact_extraction_interval_minutes",
        type=int,
        help="Override screen_memory.fact_extraction_interval_minutes.",
    )
    parser.add_argument(
        "--fact-clustering-interval-hours",
        type=float,
        help="Override screen_memory.fact_clustering_interval_hours.",
    )
    parser.add_argument(
        "--observation-interval-hours",
        type=float,
        help="Override screen_memory.observation_interval_hours.",
    )
    parser.add_argument(
        "--no-finalize",
        action="store_false",
        dest="finalize",
        help=(
            "In timeline mode, do not run the default completion pass at the "
            "last source timestamp."
        ),
    )
    parser.set_defaults(finalize=True)
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="Disable screen fact and observation LLM calls for a local smoke test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output database, report, and log.",
    )
    parser.add_argument(
        "--detail-row-limit",
        type=int,
        default=0,
        help="Maximum rows logged per detail table; 0 means all rows.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(log_path: Path, log_level: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)


def remove_existing_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    expanded: List[Path] = []
    for path in paths:
        expanded.extend([
            path,
            Path(f"{path}-wal"),
            Path(f"{path}-shm"),
        ])
    existing = [path for path in expanded if path.exists()]
    if existing and not overwrite:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "Output already exists. Pass --overwrite to replace:\n  "
            f"{joined}"
        )
    for path in existing:
        path.unlink()


def _source_edge_timestamp(
    database_path: Path,
    *,
    table: str,
    direction: str,
    immutable: bool = False,
) -> Optional[datetime]:
    order = "ASC" if direction == "first" else "DESC"
    if immutable:
        uri = database_path.resolve().as_uri() + "?immutable=1"
        connection = sqlite3.connect(uri, uri=True)
    else:
        connection = sqlite3.connect(database_path)
    try:
        row = connection.execute(
            f"SELECT timestamp FROM {table} "
            f"WHERE timestamp IS NOT NULL AND timestamp != '' "
            f"ORDER BY julianday(timestamp) {order} LIMIT 1"
        ).fetchone()
    finally:
        connection.close()
    if not row:
        return None
    try:
        return parse_timestamp_to_utc(str(row[0]))
    except (TypeError, ValueError):
        return None


def load_source_time_range(
    screenpipe_db: Path,
    openchronicle_db: Path,
) -> Tuple[datetime, datetime, Dict[str, Optional[str]]]:
    edges = {
        "screenpipe_first": _source_edge_timestamp(
            screenpipe_db,
            table="frames",
            direction="first",
        ),
        "screenpipe_last": _source_edge_timestamp(
            screenpipe_db,
            table="frames",
            direction="last",
        ),
        "openchronicle_first": _source_edge_timestamp(
            openchronicle_db,
            table="captures",
            direction="first",
            immutable=True,
        ),
        "openchronicle_last": _source_edge_timestamp(
            openchronicle_db,
            table="captures",
            direction="last",
            immutable=True,
        ),
    }
    valid = [value for value in edges.values() if value is not None]
    if not valid:
        raise RuntimeError("No valid source timestamps were found")
    serialized = {
        key: _format_utc_time(value) if value is not None else None
        for key, value in edges.items()
    }
    return min(valid), max(valid), serialized


def _phase_result(
    phase: str,
    now: datetime,
    stats: Any,
) -> Dict[str, Any]:
    result = {
        "phase": phase,
        "now": _format_utc_time(now),
        "stats": stats,
    }
    logging.info(
        "phase=%s now=%s stats=%s",
        phase,
        result["now"],
        json.dumps(stats, ensure_ascii=False, default=str, sort_keys=True),
    )
    return result


def run_once(
    manager: ScreenMemoryManager,
    start: datetime,
    end: datetime,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    state: Dict[str, Any] = {
        "last_fact_extraction_at": _format_utc_time(start),
    }
    events = [
        _phase_result("fact_extraction", end, _run_fact_extraction(manager, state, end)),
        _phase_result(
            "observations",
            end,
            _run_observations(manager, state, end),
        ),
    ]
    return state, events


def run_timeline(
    manager: ScreenMemoryManager,
    start: datetime,
    end: datetime,
    *,
    fact_extraction_interval: timedelta,
    observation_interval: timedelta,
    finalize: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    state: Dict[str, Any] = {}
    events: List[Dict[str, Any]] = []
    next_fact_extraction = start
    next_observation = start

    while min(next_fact_extraction, next_observation) <= end:
        now = min(next_fact_extraction, next_observation)
        logging.info("timeline tick=%s", _format_utc_time(now))
        if next_fact_extraction == now:
            events.append(
                _phase_result(
                    "fact_extraction",
                    now,
                    _run_fact_extraction(manager, state, now),
                )
            )
            next_fact_extraction += fact_extraction_interval
        if next_observation == now:
            events.append(
                _phase_result(
                    "observations",
                    now,
                    _run_observations(manager, state, now),
                )
            )
            next_observation += observation_interval

    last_fact_extraction = state.get("last_fact_extraction_at")
    if finalize and last_fact_extraction != _format_utc_time(end):
        logging.info("timeline finalization tick=%s", _format_utc_time(end))
        events.extend([
            _phase_result(
                "fact_extraction",
                end,
                _run_fact_extraction(manager, state, end),
            ),
            _phase_result(
                "observations",
                end,
                _run_observations(manager, state, end),
            ),
        ])
    return state, events


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone() is not None


def collect_database_report(
    database_path: Path,
    *,
    row_limit: int,
) -> Dict[str, Any]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        counts = {
            table: int(
                connection.execute(
                    f"SELECT COUNT(*) AS count FROM {table}"
                ).fetchone()["count"]
            )
            for table in COUNT_TABLES
            if _table_exists(connection, table)
        }
        details: Dict[str, List[Dict[str, Any]]] = {}
        for table in DETAIL_TABLES:
            if not _table_exists(connection, table):
                continue
            query = f"SELECT * FROM {table} ORDER BY rowid"
            parameters: Tuple[Any, ...] = ()
            if row_limit > 0:
                query += " LIMIT ?"
                parameters = (row_limit,)
            details[table] = [
                dict(row)
                for row in connection.execute(query, parameters).fetchall()
            ]
        return {"counts": counts, "details": details}
    finally:
        connection.close()


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(get_hermes_home() / ".env")
    args = parse_args()

    screenpipe_db = args.screenpipe_db.expanduser().resolve()
    openchronicle_db = args.openchronicle_db.expanduser().resolve()
    for label, path in (
        ("Screenpipe", screenpipe_db),
        ("OpenChronicle", openchronicle_db),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} database not found: {path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_db = output_dir / args.db_name
    log_path = args.log_path.expanduser().resolve()
    report_path = args.report_path.expanduser().resolve()
    remove_existing_outputs(
        (output_db, log_path, report_path),
        args.overwrite,
    )
    configure_logging(log_path, args.log_level)

    hermes_config = load_config() or {}
    manager_config = load_screen_memory_config(hermes_config)
    manager_config["enabled"] = True
    manager_config["database"].update({
        "screenpipe_db": str(screenpipe_db),
        "openchronicle_db": str(openchronicle_db),
        "cleaned_db": str(output_db),
    })
    schedule = manager_config["schedule"]
    if args.fact_extraction_interval_minutes is not None:
        schedule["fact_extraction_interval_minutes"] = max(
            1,
            int(args.fact_extraction_interval_minutes),
        )
    if args.fact_clustering_interval_hours is not None:
        schedule["fact_clustering_interval_hours"] = max(
            0.001,
            float(args.fact_clustering_interval_hours),
        )
    if args.observation_interval_hours is not None:
        schedule["observation_interval_hours"] = max(
            0.001,
            float(args.observation_interval_hours),
        )
    if args.disable_llm:
        manager_config["segment_generation"]["enable_LLM_summary"] = False
        manager_config["window_workstream_generation"][
            "enable_LLM_summary"
        ] = False
        screen_generation = manager_config["screen_memory_generation"]
        screen_generation["enable_LLM_fact_extraction"] = False
        screen_generation["enable_LLM_observation"] = False

    start, end, source_edges = load_source_time_range(
        screenpipe_db,
        openchronicle_db,
    )
    logging.info(
        "screen-memory test mode=%s executable=%s source_range=%s..%s "
        "screenpipe=%s openchronicle=%s output=%s schedule=%s",
        args.mode,
        sys.executable,
        _format_utc_time(start),
        _format_utc_time(end),
        screenpipe_db,
        openchronicle_db,
        output_db,
        json.dumps(schedule, ensure_ascii=False, sort_keys=True),
    )

    llm_client = (
        None
        if args.disable_llm
        else _inject_runtime_config(manager_config, hermes_config)
    )
    screen_db = ScreenMemoryDB(
        output_db,
        incremental_mode=True,
    )
    manager = ScreenMemoryManager(
        manager_config,
        llm_client=llm_client,
        quiet=True,
        screen_db=screen_db,
    )
    effective_start = datetime(2026, 5, 9, 0, 0, tzinfo=timezone.utc) if args.mode == "once" else start
    try:
        if args.mode == "once":
            state, events = run_once(manager, effective_start, end)
        else:
            state, events = run_timeline(
                manager,
                effective_start,
                end,
                fact_extraction_interval=timedelta(
                    minutes=max(1, int(schedule["fact_extraction_interval_minutes"]))
                ),
                observation_interval=timedelta(
                    hours=max(
                        0.001,
                        float(schedule["observation_interval_hours"]),
                    )
                ),
                finalize=bool(args.finalize),
            )

        database_report = collect_database_report(
            output_db,
            row_limit=max(0, int(args.detail_row_limit)),
        )
        report = {
            "mode": args.mode,
            "source_databases": {
                "screenpipe": str(screenpipe_db),
                "openchronicle": str(openchronicle_db),
            },
            "source_edges": source_edges,
            "source_range": {
                "start": _format_utc_time(effective_start),
                "end": _format_utc_time(end),
            },
            "schedule": schedule,
            "finalized": bool(args.finalize),
            "llm_disabled": bool(args.disable_llm),
            "state": state,
            "events": events,
            "output_db": str(output_db),
            "database": database_report,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logging.info(
            "database counts=%s",
            json.dumps(
                database_report["counts"],
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        for table, rows in database_report["details"].items():
            logging.info(
                "database table=%s rows=%s\n%s",
                table,
                len(rows),
                json.dumps(rows, ensure_ascii=False, indent=2, default=str),
            )
        logging.info("report=%s log=%s", report_path, log_path)
        print(json.dumps({
            "status": "ok",
            "mode": args.mode,
            "events": len(events),
            "output_db": str(output_db),
            "report": str(report_path),
            "log": str(log_path),
            "counts": database_report["counts"],
        }, ensure_ascii=False, indent=2))
        return 0
    finally:
        manager.close()
        close_client = getattr(llm_client, "close", None)
        if callable(close_client):
            close_client()


if __name__ == "__main__":
    raise SystemExit(main())
