#!/usr/bin/env python3
"""Exercise selected screen-memory pipeline phases against recorded databases.

The script uses an isolated output database and in-memory schedule state. It
does not read or update the production screen-memory schedule_state.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
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
    _run_tasks,
)
from hermes_cli.config import load_config
from hermes_constants import get_hermes_home


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "screen_memory_pipeline_test"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "all"
DEFAULT_LOG_FILENAME = "screen_memory_pipeline_test.log"
DEFAULT_REPORT_FILENAME = "screen_memory_pipeline_report.json"

DETAIL_TABLES = (
    "window_workstream",
    "screen_facts",
    "screen_fact_clusters",
    "screen_fact_cluster_members",
    "screen_observations",
    "screen_observation_facts",
    "task_workstream",
    "task_workstream_observations",
    "task_workstream_members",
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
    "task_workstream",
    "task_workstream_observations",
    "task_workstream_members",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run selected screen-memory fact extraction, observation, and task "
            "generation phases against Screenpipe and OpenChronicle databases."
        )
    )
    parser.add_argument(
        "--phase",
        action="append",
        choices=("all", "fact_extraction", "observations", "tasks"),
        help=(
            "Pipeline phase to run. Pass multiple times to run a subset in "
            "order. Defaults to all."
        ),
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
        help=(
            "Output directory. Defaults to "
            "tmp/screen_memory_pipeline_test/<selected-phases>."
        ),
    )
    parser.add_argument(
        "--db-name",
        default="screen_memory_pipeline_test.db",
    )
    parser.add_argument(
        "--input-db",
        "--seed-db",
        dest="input_db",
        type=Path,
        help=(
            "Existing cleaned screen-memory database to copy into the output "
            "database before running the selected phases."
        ),
    )
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="Disable screen-memory LLM calls for a local smoke test.",
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
    if not overwrite:
        return
    expanded: List[Path] = []
    for path in paths:
        expanded.extend([
            path,
            Path(f"{path}-wal"),
            Path(f"{path}-shm"),
        ])
    for path in (path for path in expanded if path.exists()):
        path.unlink()


def _clear_tables_if_present(
    connection: sqlite3.Connection,
    tables: Iterable[str],
) -> List[str]:
    existing = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    cleared: List[str] = []
    for table in tables:
        if table not in existing:
            continue
        connection.execute(f"DELETE FROM {table}")
        cleared.append(table)
    return cleared


def prune_seed_database_for_phases(
    database_path: Path,
    phases: Iterable[str],
) -> Dict[str, Any]:
    ordered_phases = ["fact_extraction", "observations", "tasks"]
    phase_set = set(phases)
    first_phase = next((phase for phase in ordered_phases if phase in phase_set), "fact_extraction")
    connection = sqlite3.connect(database_path)
    try:
        cleared: List[str] = []
        reset_clusters = False
        if first_phase == "fact_extraction":
            cleared.extend(_clear_tables_if_present(
                connection,
                [
                    "report_blocks",
                    "task_workstream_observations",
                    "task_workstream_members",
                    "task_workstream",
                    "screen_observation_facts",
                    "screen_observations",
                    "screen_fact_cluster_members",
                    "screen_fact_clusters",
                    "screen_facts",
                    "window_workstream_members",
                    "window_workstream",
                    "view_records",
                    "views",
                    "record_ax_events",
                    "openchronicle_events",
                    "records",
                ],
            ))
        elif first_phase == "observations":
            cleared.extend(_clear_tables_if_present(
                connection,
                [
                    "report_blocks",
                    "task_workstream_observations",
                    "task_workstream_members",
                    "task_workstream",
                    "screen_observation_facts",
                    "screen_observations",
                ],
            ))
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'screen_fact_clusters'"
            ).fetchone():
                columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(screen_fact_clusters)"
                    ).fetchall()
                }
                assignments = []
                if "observation_id" in columns:
                    assignments.append("observation_id = NULL")
                if "observed_fact_count" in columns:
                    assignments.append("observed_fact_count = 0")
                if assignments:
                    connection.execute(
                        f"UPDATE screen_fact_clusters SET {', '.join(assignments)}"
                    )
                    reset_clusters = True
        elif first_phase == "tasks":
            cleared.extend(_clear_tables_if_present(
                connection,
                [
                    "report_blocks",
                    "task_workstream_observations",
                    "task_workstream_members",
                    "task_workstream",
                ],
            ))
        connection.commit()
        return {
            "first_phase": first_phase,
            "cleared_tables": cleared,
            "reset_screen_fact_clusters": reset_clusters,
        }
    finally:
        connection.close()


def copy_seed_database(
    seed_db: Path,
    output_db: Path,
    *,
    overwrite: bool,
    phases: Iterable[str],
) -> Dict[str, Any]:
    seed_db = seed_db.expanduser().resolve()
    output_db = output_db.expanduser().resolve()
    if not seed_db.is_file():
        raise FileNotFoundError(f"Input cleaned database not found: {seed_db}")
    if output_db.exists() and not overwrite:
        raise FileExistsError(
            "Output database already exists. Pass --overwrite to replace it "
            f"from --input-db, or use the existing output directly: {output_db}"
        )
    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        remove_existing_outputs((output_db,), overwrite=True)
    source = sqlite3.connect(seed_db)
    try:
        target = sqlite3.connect(output_db)
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()
    prune_report = prune_seed_database_for_phases(output_db, phases)
    return {
        "copied": True,
        "input_db": str(seed_db),
        "output_db": str(output_db),
        **prune_report,
    }


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
    *,
    phases: Iterable[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    state: Dict[str, Any] = {
        "last_fact_extraction_at": _format_utc_time(start),
    }
    events: List[Dict[str, Any]] = []
    for phase in phases:
        if phase == "fact_extraction":
            events.append(
                _phase_result(
                    "fact_extraction",
                    end,
                    _run_fact_extraction(manager, state, end),
                )
            )
        elif phase == "observations":
            events.append(
                _phase_result(
                    "observations",
                    end,
                    _run_observations(manager, state, end),
                )
            )
        elif phase == "tasks":
            events.append(
                _phase_result(
                    "tasks",
                    end,
                    _run_tasks(manager, state, end),
                )
            )
        else:
            raise ValueError(f"Unsupported screen-memory phase: {phase}")
    return state, events


def normalize_phases(raw_phases: Optional[List[str]]) -> List[str]:
    default_phases = ["fact_extraction", "observations", "tasks"]
    if not raw_phases or "all" in raw_phases:
        return default_phases
    phases: List[str] = []
    for phase in raw_phases:
        if phase not in phases:
            phases.append(phase)
    return phases


def phase_output_slug(phases: Iterable[str]) -> str:
    return "__".join(str(phase).strip() for phase in phases if str(phase).strip()) or "all"


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
            if row_limit >= 0:
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
    # load_dotenv(REPO_ROOT / ".env")
    # load_dotenv(get_hermes_home() / ".env")
    args = parse_args()

    screenpipe_db = args.screenpipe_db.expanduser().resolve()
    openchronicle_db = args.openchronicle_db.expanduser().resolve()
    for label, path in (
        ("Screenpipe", screenpipe_db),
        ("OpenChronicle", openchronicle_db),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} database not found: {path}")

    phases = normalize_phases(args.phase)
    selected_output_dir = args.output_dir
    if selected_output_dir == DEFAULT_OUTPUT_DIR:
        selected_output_dir = DEFAULT_OUTPUT_ROOT / phase_output_slug(phases)
    output_dir = selected_output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_db = output_dir / args.db_name
    input_db = args.input_db.expanduser().resolve() if args.input_db else None
    log_path = (
        args.log_path.expanduser().resolve()
        if args.log_path
        else output_dir / DEFAULT_LOG_FILENAME
    )
    report_path = (
        args.report_path.expanduser().resolve()
        if args.report_path
        else output_dir / DEFAULT_REPORT_FILENAME
    )
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
    seed_copy_report = None
    if input_db is not None:
        seed_copy_report = copy_seed_database(
            input_db,
            output_db,
            overwrite=args.overwrite,
            phases=phases,
        )
        logging.info(
            "seeded output database report=%s",
            json.dumps(seed_copy_report, ensure_ascii=False, sort_keys=True),
        )
    if "fact_extraction" not in phases and not output_db.exists():
        raise FileNotFoundError(
            "Selected phases need an existing cleaned database with prior facts. "
            "Run fact_extraction first, reuse an existing --output-dir/--db-name, "
            "or pass --input-db/--seed-db."
        )
    if args.disable_llm:
        manager_config["window_workstream_generation"][
            "enable_LLM_summary"
        ] = False
        manager_config["screen_fact_generation"]["enable_LLM"] = False
        manager_config["screen_observation_generation"][
            "enable_LLM_observation_generation"
        ] = False
        manager_config["task_workstream_generation"][
            "enable_LLM_observation_matching"
        ] = False
        manager_config["task_workstream_generation"][
            "enable_LLM_task_profile_update"
        ] = False

    start, end, source_edges = load_source_time_range(
        screenpipe_db,
        openchronicle_db,
    )
    logging.info(
        "screen-memory test phases=%s executable=%s source_range=%s..%s "
        "screenpipe=%s openchronicle=%s output=%s schedule=%s",
        phases,
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
    effective_start = datetime(2026, 5, 9, 0, 0, tzinfo=timezone.utc)
    try:
        state, events = run_once(
            manager,
            effective_start,
            end,
            phases=phases,
        )

        database_report = collect_database_report(
            output_db,
            row_limit=max(0, int(args.detail_row_limit)),
        )
        report = {
            "mode": "once",
            "phases": phases,
            "source_databases": {
                "screenpipe": str(screenpipe_db),
                "openchronicle": str(openchronicle_db),
                "input_cleaned": str(input_db) if input_db else None,
            },
            "source_edges": source_edges,
            "source_range": {
                "start": _format_utc_time(effective_start),
                "end": _format_utc_time(end),
            },
            "schedule": schedule,
            "llm_disabled": bool(args.disable_llm),
            "seed_copy": seed_copy_report,
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
            "mode": "once",
            "phases": phases,
            "events": len(events),
            "output_db": str(output_db),
            "input_db": str(input_db) if input_db else None,
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
