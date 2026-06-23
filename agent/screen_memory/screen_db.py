import os
import json
import hashlib
from collections import Counter
import sqlite3
import struct
import threading
import time
import unicodedata
from datetime import datetime, timezone, timedelta
from pathlib import Path
import yaml

from agent.screen_memory.utils import *

DATABASE_TIMEZONE = timezone(timedelta(hours=8))


def _parse_json_list(value):
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _format_db_timestamp(value):
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(DATABASE_TIMEZONE).isoformat()


class ScreenMemoryDB:

    def __init__(self, output_path, incremental_mode=False):
        self.incremental_mode = incremental_mode
        self.output_path = str(Path(output_path).expanduser())
        if not self.incremental_mode and os.path.exists(self.output_path):
            os.remove(self.output_path)
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.output_path)
        self._create_schema()

    @classmethod
    def init_cleaned_db(cls, screen_db, output_path):
        """Replace the current database with a freshly initialized one."""
        return cls._replace_screen_db(
            screen_db,
            output_path,
            incremental_mode=False,
        )

    @classmethod
    def ensure_cleaned_db(cls, screen_db, output_path):
        """Return a usable database while preserving existing data."""
        return cls._ensure_screen_db(screen_db, output_path)

    @classmethod
    def _ensure_screen_db(cls, screen_db, output_path):
        if screen_db is not None and screen_db.connection is not None:
            return screen_db
        return cls(output_path, incremental_mode=True)

    @classmethod
    def _replace_screen_db(cls, screen_db, output_path, *, incremental_mode):
        if screen_db is not None:
            screen_db.close()
        return cls(output_path, incremental_mode=incremental_mode)

    @property
    def connection(self):
        return self._conn

    def close(self):
        if self._conn is None:
            return
        self._conn.close()
        self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def encode_embedding_vector(vector):
        return struct.pack(f"<{len(vector)}f", *[float(value) for value in vector])

    @staticmethod
    def decode_embedding_vector(blob, dimensions=None):
        if not blob:
            return None
        size = len(blob) // 4
        if dimensions and size != int(dimensions):
            return None
        try:
            return list(struct.unpack(f"<{size}f", blob))
        except struct.error:
            return None

    @staticmethod
    def build_persisted_screen_fact_cluster_key(window_workstream_id, facts):
        fact_ids = sorted(
            int(fact["id"])
            for fact in facts
            if fact.get("id") is not None
        )
        payload = {
            "window_workstream_id": int(window_workstream_id),
            "fact_ids": fact_ids,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _create_schema(self):
        """Create tables, indexes, FTS and triggers if they don't exist."""
        cursor = self._conn.cursor()

        # Main table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            app_name TEXT,
            window_title TEXT,
            focused INTEGER,
            ocr_text TEXT,
            cleaned_text TEXT,
            ax_window_title TEXT,
            ax_context_json TEXT,
            user_actions_json TEXT,
            text_source TEXT,
            ocr_quality_score REAL,
            content_kind TEXT,
            trigger_reason TEXT,
            raw_frame_id INTEGER
        );
        """)

        existing_columns = {
            row[1] for row in cursor.execute("PRAGMA table_info(records)").fetchall()
        }
        for column_name, column_type in [
            ("cleaned_text", "TEXT"),
            ("ax_window_title", "TEXT"),
            ("ax_context_json", "TEXT"),
            ("user_actions_json", "TEXT"),
            ("text_source", "TEXT"),
            ("ocr_quality_score", "REAL"),
            ("content_kind", "TEXT"),
        ]:
            if column_name not in existing_columns:
                cursor.execute(f"ALTER TABLE records ADD COLUMN {column_name} {column_type}")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_timestamp TEXT NOT NULL,
            end_timestamp TEXT NOT NULL,
            duration_seconds INTEGER,
            activity_type TEXT,
            project_hint TEXT,
            app_names TEXT,
            window_titles TEXT,
            summary TEXT,
            actions_json TEXT,
            artifacts_json TEXT,
            evidence_ids_json TEXT,
            llm_summary_json TEXT,
            llm_summary_text TEXT,
            llm_model TEXT,
            llm_status TEXT,
            llm_error TEXT,
            llm_hash TEXT,
            llm_updated_at TEXT,
            confidence REAL,
            record_count INTEGER
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS views (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            app_name TEXT,
            window_title TEXT,
            content_kind TEXT,
            start_timestamp TEXT NOT NULL,
            end_timestamp TEXT NOT NULL,
            representative_text TEXT,
            topics_json TEXT,
            entities_json TEXT,
            artifacts_json TEXT,
            evidence_ids_json TEXT,
            llm_summary_json TEXT,
            llm_summary_text TEXT,
            llm_model TEXT,
            llm_status TEXT,
            llm_error TEXT,
            llm_hash TEXT,
            llm_updated_at TEXT,
            confidence REAL,
            record_count INTEGER
        );
        """)

        view_records_existed = cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'view_records'"
        ).fetchone() is not None
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS view_records (
            view_id INTEGER NOT NULL,
            record_id INTEGER NOT NULL,
            PRIMARY KEY (view_id, record_id),
            FOREIGN KEY (view_id) REFERENCES views(id) ON DELETE CASCADE,
            FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS view_segments (
            view_id INTEGER NOT NULL,
            segment_id INTEGER NOT NULL,
            record_count INTEGER,
            start_timestamp TEXT,
            end_timestamp TEXT,
            representative_text TEXT,
            evidence_ids_json TEXT,
            PRIMARY KEY (view_id, segment_id),
            FOREIGN KEY (view_id) REFERENCES views(id) ON DELETE CASCADE,
            FOREIGN KEY (segment_id) REFERENCES segments(id) ON DELETE CASCADE
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS openchronicle_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_capture_id TEXT UNIQUE,
            timestamp TEXT NOT NULL,
            timestamp_epoch INTEGER,
            app_name TEXT,
            bundle_id TEXT,
            window_title TEXT,
            event_type TEXT,
            focused_role TEXT,
            focused_value TEXT,
            visible_text TEXT,
            url TEXT,
            normalized_json TEXT,
            app_context_json TEXT,
            feishu_context_json TEXT,
            user_actions_json TEXT,
            raw_json TEXT
        );
        """)

        existing_oc_columns = {
            row[1] for row in cursor.execute("PRAGMA table_info(openchronicle_events)").fetchall()
        }
        for column_name, column_type in [
            ("app_context_json", "TEXT"),
            ("user_actions_json", "TEXT"),
        ]:
            if column_name not in existing_oc_columns:
                cursor.execute(f"ALTER TABLE openchronicle_events ADD COLUMN {column_name} {column_type}")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS record_ax_events (
            record_id INTEGER NOT NULL,
            openchronicle_event_id INTEGER NOT NULL,
            delta_seconds REAL,
            match_reason TEXT,
            PRIMARY KEY (record_id, openchronicle_event_id),
            FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE,
            FOREIGN KEY (openchronicle_event_id) REFERENCES openchronicle_events(id) ON DELETE CASCADE
        );
        """)

        existing_view_columns = {
            row[1] for row in cursor.execute("PRAGMA table_info(views)").fetchall()
        }
        for column_name, column_type in [
            ("llm_summary_json", "TEXT"),
            ("llm_summary_text", "TEXT"),
            ("llm_model", "TEXT"),
            ("llm_status", "TEXT"),
            ("llm_error", "TEXT"),
            ("llm_hash", "TEXT"),
            ("llm_updated_at", "TEXT"),
        ]:
            if column_name not in existing_view_columns:
                cursor.execute(f"ALTER TABLE views ADD COLUMN {column_name} {column_type}")
        if "visible_content_summary" in existing_view_columns:
            cursor.execute("ALTER TABLE views DROP COLUMN visible_content_summary")

        existing_view_segment_columns = {
            row[1] for row in cursor.execute("PRAGMA table_info(view_segments)").fetchall()
        }
        if "visible_content_summary" in existing_view_segment_columns:
            cursor.execute("ALTER TABLE view_segments DROP COLUMN visible_content_summary")

        if not view_records_existed:
            existing_record_ids = {
                row[0] for row in cursor.execute("SELECT id FROM records").fetchall()
            }
            legacy_view_record_links = []
            for view_id, evidence_ids_json in cursor.execute(
                "SELECT id, evidence_ids_json FROM views WHERE evidence_ids_json IS NOT NULL"
            ).fetchall():
                for record_id in _parse_json_list(evidence_ids_json):
                    try:
                        record_id = int(record_id)
                    except (TypeError, ValueError):
                        continue
                    if record_id in existing_record_ids:
                        legacy_view_record_links.append((view_id, record_id))
            if legacy_view_record_links:
                cursor.executemany(
                    "INSERT OR IGNORE INTO view_records (view_id, record_id) VALUES (?, ?)",
                    legacy_view_record_links,
                )

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS window_workstream (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            summary TEXT,
            category TEXT,
            start_timestamp TEXT NOT NULL,
            end_timestamp TEXT NOT NULL,
            topics_json TEXT,
            entities_json TEXT,
            artifacts_json TEXT,
            app_names_json TEXT,
            window_titles_json TEXT,
            view_count INTEGER,
            segment_count INTEGER,
            confidence REAL,
            llm_summary_json TEXT,
            llm_model TEXT,
            llm_status TEXT,
            llm_error TEXT,
            llm_hash TEXT,
            llm_updated_at TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        """)

        existing_workstream_columns = {
            row[1] for row in cursor.execute("PRAGMA table_info(window_workstream)").fetchall()
        }
        for column_name, column_type in [
            ("llm_summary_json", "TEXT"),
            ("llm_model", "TEXT"),
            ("llm_status", "TEXT"),
            ("llm_error", "TEXT"),
            ("llm_hash", "TEXT"),
            ("llm_updated_at", "TEXT"),
        ]:
            if column_name not in existing_workstream_columns:
                cursor.execute(f"ALTER TABLE window_workstream ADD COLUMN {column_name} {column_type}")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS window_workstream_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            window_workstream_id INTEGER NOT NULL,
            view_id INTEGER NOT NULL,
            relevance REAL,
            reason TEXT,
            created_at TEXT,
            FOREIGN KEY (window_workstream_id) REFERENCES window_workstream(id) ON DELETE CASCADE,
            FOREIGN KEY (view_id) REFERENCES views(id) ON DELETE CASCADE
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS task_workstream (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            summary TEXT,
            category TEXT,
            start_timestamp TEXT NOT NULL,
            end_timestamp TEXT NOT NULL,
            topics_json TEXT,
            entities_json TEXT,
            artifacts_json TEXT,
            app_names_json TEXT,
            window_titles_json TEXT,
            window_workstream_count INTEGER,
            view_count INTEGER,
            segment_count INTEGER,
            confidence REAL,
            llm_summary_json TEXT,
            llm_model TEXT,
            llm_status TEXT,
            llm_error TEXT,
            llm_hash TEXT,
            llm_updated_at TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        """)
        existing_task_columns = {
            row[1] for row in cursor.execute("PRAGMA table_info(task_workstream)").fetchall()
        }
        for column_name, column_type in [
            ("task_key", "TEXT"),
            ("status", "TEXT"),
            ("project_key", "TEXT"),
            ("objective_key", "TEXT"),
            ("work_type", "TEXT"),
            ("progress_text", "TEXT"),
            ("blockers_json", "TEXT"),
            ("next_actions_json", "TEXT"),
            ("observation_count", "INTEGER DEFAULT 0"),
            ("last_activity_timestamp", "TEXT"),
            ("metadata_json", "TEXT"),
            ("generation_method", "TEXT"),
        ]:
            if column_name not in existing_task_columns:
                cursor.execute(f"ALTER TABLE task_workstream ADD COLUMN {column_name} {column_type}")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS task_workstream_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_workstream_id INTEGER NOT NULL,
            window_workstream_id INTEGER NOT NULL,
            relevance REAL,
            reason TEXT,
            created_at TEXT,
            FOREIGN KEY (task_workstream_id) REFERENCES task_workstream(id) ON DELETE CASCADE,
            FOREIGN KEY (window_workstream_id) REFERENCES window_workstream(id) ON DELETE CASCADE
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS task_workstream_observations (
            task_workstream_id INTEGER NOT NULL,
            observation_id INTEGER NOT NULL UNIQUE,
            role TEXT NOT NULL DEFAULT 'evidence',
            confidence REAL DEFAULT 1.0,
            reason TEXT,
            created_at TEXT,
            PRIMARY KEY (task_workstream_id, observation_id),
            FOREIGN KEY (task_workstream_id) REFERENCES task_workstream(id) ON DELETE CASCADE,
            FOREIGN KEY (observation_id) REFERENCES screen_observations(id) ON DELETE CASCADE
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS report_blocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_workstream_id INTEGER,
            source_type TEXT,
            source_id INTEGER,
            period_key TEXT NOT NULL,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            title TEXT,
            category TEXT,
            project_key TEXT,
            objective_key TEXT,
            work_type TEXT,
            summary_text TEXT,
            progress_text TEXT,
            key_points_json TEXT,
            decisions_json TEXT,
            blockers_json TEXT,
            next_actions_json TEXT,
            entities_json TEXT,
            artifacts_json TEXT,
            evidence_view_ids_json TEXT,
            evidence_window_workstream_ids_json TEXT,
            evidence_record_ids_json TEXT,
            confidence REAL,
            llm_summary_json TEXT,
            llm_model TEXT,
            llm_status TEXT,
            llm_error TEXT,
            llm_hash TEXT,
            llm_updated_at TEXT,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY (task_workstream_id) REFERENCES task_workstream(id) ON DELETE CASCADE
        );
        """)

        existing_report_block_columns = {
            row[1] for row in cursor.execute("PRAGMA table_info(report_blocks)").fetchall()
        }
        for column_name, column_type in [
            ("source_type", "TEXT"),
            ("source_id", "INTEGER"),
            ("project_key", "TEXT"),
            ("objective_key", "TEXT"),
            ("work_type", "TEXT"),
        ]:
            if column_name not in existing_report_block_columns:
                cursor.execute(f"ALTER TABLE report_blocks ADD COLUMN {column_name} {column_type}")

        report_block_info = cursor.execute("PRAGMA table_info(report_blocks)").fetchall()
        report_block_columns = {row[1] for row in report_block_info}
        task_id_column = next((row for row in report_block_info if row[1] == "task_workstream_id"), None)
        if task_id_column and task_id_column[3]:
            source_type_expr = "'task_workstream'"
            source_id_expr = "task_workstream_id"
            if "source_type" in report_block_columns:
                source_type_expr = "COALESCE(source_type, 'task_workstream')"
            if "source_id" in report_block_columns:
                source_id_expr = "COALESCE(source_id, task_workstream_id)"
            cursor.execute("ALTER TABLE report_blocks RENAME TO report_blocks_old")
            cursor.execute("""
            CREATE TABLE report_blocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_workstream_id INTEGER,
                source_type TEXT,
                source_id INTEGER,
                period_key TEXT NOT NULL,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                title TEXT,
                category TEXT,
                project_key TEXT,
                objective_key TEXT,
                work_type TEXT,
                summary_text TEXT,
                progress_text TEXT,
                key_points_json TEXT,
                decisions_json TEXT,
                blockers_json TEXT,
                next_actions_json TEXT,
                entities_json TEXT,
                artifacts_json TEXT,
                evidence_view_ids_json TEXT,
                evidence_window_workstream_ids_json TEXT,
                evidence_record_ids_json TEXT,
                confidence REAL,
                llm_summary_json TEXT,
                llm_model TEXT,
                llm_status TEXT,
                llm_error TEXT,
                llm_hash TEXT,
                llm_updated_at TEXT,
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY (task_workstream_id) REFERENCES task_workstream(id) ON DELETE CASCADE
            );
            """)
            cursor.execute(f"""
            INSERT INTO report_blocks
            (id, task_workstream_id, source_type, source_id, period_key, period_start, period_end,
             title, category, project_key, objective_key, work_type, summary_text, progress_text,
             key_points_json, decisions_json, blockers_json, next_actions_json, entities_json,
             artifacts_json, evidence_view_ids_json, evidence_window_workstream_ids_json,
             evidence_record_ids_json, confidence, llm_summary_json, llm_model, llm_status,
             llm_error, llm_hash, llm_updated_at, created_at, updated_at)
            SELECT
             id, task_workstream_id, {source_type_expr}, {source_id_expr}, period_key,
             period_start, period_end, title, category, project_key, objective_key, work_type,
             summary_text, progress_text, key_points_json, decisions_json, blockers_json,
             next_actions_json, entities_json, artifacts_json, evidence_view_ids_json,
             evidence_window_workstream_ids_json, evidence_record_ids_json, confidence,
             llm_summary_json, llm_model, llm_status, llm_error, llm_hash, llm_updated_at,
             created_at, updated_at
            FROM report_blocks_old
            """)
            cursor.execute("DROP TABLE report_blocks_old")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS screen_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            view_id INTEGER NOT NULL,
            fact_text TEXT NOT NULL,
            fact_type TEXT NOT NULL DEFAULT 'episodic',
            fact_kind TEXT NOT NULL DEFAULT 'other',
            work_type TEXT NOT NULL DEFAULT 'other',
            project_key TEXT NOT NULL DEFAULT 'unknown',
            objective_key TEXT NOT NULL DEFAULT 'general',
            topics_json TEXT,
            entities_json TEXT,
            artifacts_json TEXT,
            evidence_text TEXT,
            evidence_record_ids_json TEXT,
            app_name TEXT,
            window_title TEXT,
            start_timestamp TEXT,
            end_timestamp TEXT,
            confidence REAL DEFAULT 0.0,
            llm_summary_json TEXT,
            llm_model TEXT,
            llm_status TEXT,
            llm_error TEXT,
            llm_hash TEXT,
            llm_updated_at TEXT,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY (view_id) REFERENCES views(id) ON DELETE CASCADE
        );
        """)

        existing_screen_fact_columns = {
            row[1] for row in cursor.execute("PRAGMA table_info(screen_facts)").fetchall()
        }
        for column_name, column_type in [
            ("embedding_text", "TEXT"),
            ("embedding_provider", "TEXT"),
            ("embedding_model", "TEXT"),
            ("embedding_dimensions", "INTEGER"),
            ("embedding_vector", "BLOB"),
            ("embedding_status", "TEXT"),
            ("embedding_error", "TEXT"),
        ]:
            if column_name not in existing_screen_fact_columns:
                cursor.execute(f"ALTER TABLE screen_facts ADD COLUMN {column_name} {column_type}")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS screen_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            observation_type TEXT NOT NULL DEFAULT 'context',
            scope_type TEXT NOT NULL DEFAULT 'window_workstream',
            scope_id INTEGER,
            cluster_key TEXT,
            period_key TEXT,
            period_start TEXT,
            period_end TEXT,
            title TEXT,
            summary_text TEXT,
            progress_text TEXT,
            project_key TEXT NOT NULL DEFAULT 'unknown',
            objective_key TEXT NOT NULL DEFAULT 'general',
            work_type TEXT NOT NULL DEFAULT 'other',
            category TEXT NOT NULL DEFAULT 'other',
            key_points_json TEXT,
            decisions_json TEXT,
            blockers_json TEXT,
            next_actions_json TEXT,
            entities_json TEXT,
            artifacts_json TEXT,
            evidence_view_ids_json TEXT,
            evidence_record_ids_json TEXT,
            evidence_window_workstream_ids_json TEXT,
            confidence REAL DEFAULT 0.0,
            generation_method TEXT,
            metadata_json TEXT,
            llm_summary_json TEXT,
            llm_model TEXT,
            llm_status TEXT,
            llm_error TEXT,
            llm_hash TEXT,
            llm_updated_at TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        """)
        screen_observation_columns = {
            row[1]
            for row in cursor.execute(
                "PRAGMA table_info(screen_observations)"
            ).fetchall()
        }
        if (
            "observation_kind" in screen_observation_columns
            and "observation_type" not in screen_observation_columns
        ):
            cursor.execute(
                "ALTER TABLE screen_observations "
                "RENAME COLUMN observation_kind TO observation_type"
            )
            cursor.execute(
                """
                UPDATE screen_observations
                SET observation_type = CASE observation_type
                    WHEN 'state_change' THEN 'task_progress'
                    WHEN 'outcome' THEN 'task_progress'
                    WHEN 'conflict' THEN 'problem'
                    WHEN 'task_signal' THEN 'task_state'
                    WHEN 'goal_signal' THEN 'task_state'
                    WHEN 'constraint' THEN 'constraint'
                    ELSE 'context'
                END
                """
            )

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS screen_observation_facts (
            observation_id INTEGER NOT NULL,
            fact_id INTEGER NOT NULL,
            role TEXT NOT NULL DEFAULT 'supporting',
            confidence REAL DEFAULT 1.0,
            PRIMARY KEY (observation_id, fact_id),
            FOREIGN KEY (observation_id) REFERENCES screen_observations(id) ON DELETE CASCADE,
            FOREIGN KEY (fact_id) REFERENCES screen_facts(id) ON DELETE CASCADE
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS screen_fact_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            window_workstream_id INTEGER NOT NULL,
            cluster_key TEXT NOT NULL UNIQUE,
            cluster_score REAL DEFAULT 0.0,
            cluster_reason TEXT,
            observation_id INTEGER,
            observed_fact_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY (window_workstream_id) REFERENCES window_workstream(id) ON DELETE CASCADE,
            FOREIGN KEY (observation_id) REFERENCES screen_observations(id) ON DELETE SET NULL
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS screen_fact_cluster_members (
            cluster_id INTEGER NOT NULL,
            fact_id INTEGER NOT NULL UNIQUE,
            created_at TEXT,
            PRIMARY KEY (cluster_id, fact_id),
            FOREIGN KEY (cluster_id) REFERENCES screen_fact_clusters(id) ON DELETE CASCADE,
            FOREIGN KEY (fact_id) REFERENCES screen_facts(id) ON DELETE CASCADE
        );
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON records(timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_app ON records(app_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_kind ON records(content_kind);")
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_records_source_capture "
            "ON records(raw_frame_id, app_name, window_title);"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_segments_time ON segments(start_timestamp, end_timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_segments_project ON segments(project_hint);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_view_time ON views(start_timestamp, end_timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_view_app ON views(app_name, window_title);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_view_kind ON views(content_kind);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_view_records_view ON view_records(view_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_view_records_record ON view_records(record_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_view_segments_segment ON view_segments(segment_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_openchronicle_events_time ON openchronicle_events(timestamp_epoch);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_openchronicle_events_app ON openchronicle_events(app_name, bundle_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_record_ax_events_record ON record_ax_events(record_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_record_ax_events_event ON record_ax_events(openchronicle_event_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_window_workstream_time ON window_workstream(start_timestamp, end_timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_window_workstream_members_workstream ON window_workstream_members(window_workstream_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_window_workstream_members_view ON window_workstream_members(view_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_task_workstream_time ON task_workstream(start_timestamp, end_timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_task_workstream_members_task ON task_workstream_members(task_workstream_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_task_workstream_members_window ON task_workstream_members(window_workstream_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_task_workstream_observations_task ON task_workstream_observations(task_workstream_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_task_workstream_observations_observation ON task_workstream_observations(observation_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_report_blocks_period ON report_blocks(period_start, period_end);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_report_blocks_task ON report_blocks(task_workstream_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_report_blocks_source ON report_blocks(source_type, source_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_report_blocks_project ON report_blocks(project_key);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_report_blocks_objective ON report_blocks(objective_key);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_screen_facts_view ON screen_facts(view_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_screen_facts_time ON screen_facts(start_timestamp, end_timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_screen_facts_project ON screen_facts(project_key, objective_key);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_screen_observations_scope ON screen_observations(scope_type, scope_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_screen_observations_period ON screen_observations(period_start, period_end);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_screen_observation_facts_fact ON screen_observation_facts(fact_id);")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_screen_fact_clusters_workstream "
            "ON screen_fact_clusters(window_workstream_id, updated_at);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_screen_fact_cluster_members_cluster "
            "ON screen_fact_cluster_members(cluster_id);"
        )
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_report_blocks_task_period "
            "ON report_blocks(task_workstream_id, period_start, period_end);"
        )
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_report_blocks_source_period "
            "ON report_blocks(source_type, source_id, period_start, period_end);"
        )

        fts_row = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='records_fts'"
        ).fetchone()

        fts_needs_rebuild = False
        if fts_row:
            fts_columns = {row[1] for row in cursor.execute("PRAGMA table_info(records_fts)").fetchall()}
            fts_needs_rebuild = "cleaned_text" not in fts_columns
            if fts_needs_rebuild:
                for trigger in ["records_ai", "records_ad", "records_au"]:
                    cursor.execute(f"DROP TRIGGER IF EXISTS {trigger}")
                cursor.execute("DROP TABLE records_fts")

        if not fts_row or fts_needs_rebuild:
            cursor.execute("""
            CREATE VIRTUAL TABLE records_fts USING fts5(
                id UNINDEXED,
                app_name,
                window_title,
                ocr_text,
                cleaned_text,
                tokenize = 'unicode61 remove_diacritics 2'
            );
            """)
            cursor.execute("""
            INSERT INTO records_fts(id, app_name, window_title, ocr_text, cleaned_text)
            SELECT id, app_name, window_title, ocr_text, COALESCE(cleaned_text, ocr_text)
            FROM records;
            """)

        # Triggers — use IF NOT EXISTS workaround (check sqlite_master)
        trigger_exists = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name='records_ai'"
        ).fetchone()
        if not trigger_exists:
            cursor.execute("""
            CREATE TRIGGER records_ai AFTER INSERT ON records BEGIN
                INSERT INTO records_fts(id, app_name, window_title, ocr_text, cleaned_text)
                VALUES (new.id, new.app_name, new.window_title, new.ocr_text, new.cleaned_text);
            END;
            """)
            cursor.execute("""
            CREATE TRIGGER records_ad AFTER DELETE ON records BEGIN
                DELETE FROM records_fts WHERE id = old.id;
            END;
            """)
            cursor.execute("""
            CREATE TRIGGER records_au AFTER UPDATE ON records BEGIN
                DELETE FROM records_fts WHERE id = old.id;
                INSERT INTO records_fts(id, app_name, window_title, ocr_text, cleaned_text)
                VALUES (new.id, new.app_name, new.window_title, new.ocr_text, new.cleaned_text);
            END;
            """)

        self._conn.commit()

    def update_screen_fact_embedding_row(self, fact, vector, embedding_text, config, status="ok", error=None):
        cursor = self._conn.cursor()
        cursor.execute(
            """
            UPDATE screen_facts
            SET embedding_text = ?, embedding_provider = ?,
                embedding_model = ?, embedding_dimensions = ?, embedding_vector = ?,
                embedding_status = ?, embedding_error = ?,
            WHERE id = ?
            """,
            (
                embedding_text,
                config.get("provider") or "openai",
                config.get("model"),
                len(vector or []),
                self.encode_embedding_vector(vector) if vector else None,
                status,
                (error or "")[:1000] if error else None,
                fact.get("id"),
            ),
        )

    def load_window_workstream_ids_with_unclustered_facts(self):

        cursor = self._conn.cursor()
        rows = cursor.execute(
            """
            SELECT DISTINCT wm.window_workstream_id
            FROM screen_facts sf
            JOIN window_workstream_members wm ON wm.view_id = sf.view_id
            LEFT JOIN screen_fact_cluster_members cm ON cm.fact_id = sf.id
            WHERE cm.fact_id IS NULL
            ORDER BY wm.window_workstream_id ASC
            """
        ).fetchall()
        return [int(row[0]) for row in rows]

    def save_screen_fact(self, fact_entry):

        cursor = self._conn.cursor()
        now = now_db_timestamp()
        cursor.execute(
            """
            INSERT OR IGNORE INTO screen_facts
            (view_id, fact_text, fact_type, fact_kind, work_type,
             project_key, objective_key, topics_json, entities_json, artifacts_json,
             evidence_text, evidence_record_ids_json, app_name, window_title,
             start_timestamp, end_timestamp, confidence, llm_summary_json, llm_model,
             llm_status, llm_error, llm_hash, llm_updated_at, embedding_text,
             embedding_provider, embedding_model, embedding_dimensions,
             embedding_vector, embedding_status, embedding_error, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fact_entry["view_id"],
                fact_entry["fact_text"],
                fact_entry["fact_type"],
                fact_entry["fact_kind"],
                fact_entry["work_type"],
                fact_entry["project_key"],
                fact_entry["objective_key"],
                fact_entry["topics_json"],
                fact_entry["entities_json"],
                fact_entry["artifacts_json"],
                fact_entry["evidence_text"],
                fact_entry["evidence_record_ids_json"],
                fact_entry["app_name"],
                fact_entry["window_title"],
                fact_entry["start_timestamp"],
                fact_entry["end_timestamp"],
                fact_entry["confidence"],
                fact_entry.get("llm_summary_json"),
                fact_entry.get("llm_model"),
                fact_entry.get("llm_status"),
                fact_entry.get("llm_error"),
                fact_entry.get("llm_hash"),
                fact_entry.get("llm_updated_at"),
                fact_entry.get("embedding_text"),
                fact_entry.get("embedding_provider"),
                fact_entry.get("embedding_model"),
                fact_entry.get("embedding_dimensions"),
                self.encode_embedding_vector(fact_entry.get("embedding_vector"))
                if fact_entry.get("embedding_vector")
                else None,
                fact_entry.get("embedding_status"),
                fact_entry.get("embedding_error"),
                now,
                now,
            ),
        )
        if cursor.rowcount:
            return cursor.lastrowid
        return None

    def load_screen_facts_by_ids(self, fact_ids):

        cursor = self._conn.cursor()
        fact_ids = [int(fact_id) for fact_id in dict.fromkeys(fact_ids or []) if fact_id is not None]
        if not fact_ids:
            return []
        placeholders = ",".join("?" for _ in fact_ids)
        cursor.execute(
            f"""
            SELECT
                sf.id, sf.view_id, sf.fact_text, sf.fact_type, sf.fact_kind,
                sf.work_type, sf.project_key, sf.objective_key, sf.topics_json,
                sf.entities_json, sf.artifacts_json, sf.evidence_text,
                sf.evidence_record_ids_json, sf.app_name, sf.window_title,
                sf.start_timestamp, sf.end_timestamp, sf.confidence,
                sf.embedding_text, sf.embedding_model,
                sf.embedding_dimensions, sf.embedding_vector
            FROM screen_facts sf
            WHERE sf.id IN ({placeholders})
            ORDER BY sf.start_timestamp ASC, sf.id ASC
            """,
            fact_ids,
        )
        columns = [column[0] for column in cursor.description]
        return [
            self.build_screen_fact_item_from_row(dict(zip(columns, row)))
            for row in cursor.fetchall()
        ]

    def save_persisted_screen_fact_cluster(
        self,
        window_workstream_id,
        cluster,
        existing_cluster=None,
    ):
        cursor = self._conn.cursor()
        now = now_db_timestamp()
        facts = cluster.get("facts") or []
        cluster_key = self.build_persisted_screen_fact_cluster_key(
            window_workstream_id,
            facts,
        )
        cluster_id = (
            existing_cluster.get("fact_cluster_id")
            if existing_cluster
            else None
        )
        if cluster_id:
            cursor.execute(
                """
                UPDATE screen_fact_clusters
                SET cluster_key = ?, cluster_score = ?, cluster_reason = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    cluster_key,
                    cluster.get("cluster_score") or 0.0,
                    cluster.get("cluster_reason") or "",
                    now,
                    cluster_id,
                ),
            )
        else:
            cursor.execute(
                """
                INSERT INTO screen_fact_clusters
                (window_workstream_id, cluster_key, cluster_score, cluster_reason,
                 observed_fact_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    window_workstream_id,
                    cluster_key,
                    cluster.get("cluster_score") or 0.0,
                    cluster.get("cluster_reason") or "",
                    now,
                    now,
                ),
            )
            cluster_id = cursor.lastrowid
        for fact in facts:
            if fact.get("id") is None:
                continue
            cursor.execute(
                """
                INSERT OR IGNORE INTO screen_fact_cluster_members
                (cluster_id, fact_id, created_at)
                VALUES (?, ?, ?)
                """,
                (cluster_id, fact["id"], now),
            )
        return int(cluster_id)

    def save_screen_observation(self, entry):

        cursor = self._conn.cursor()
        now = now_db_timestamp()
        existing = None
        if entry.get("observation_id"):
            existing = cursor.execute(
                "SELECT id FROM screen_observations WHERE id = ? LIMIT 1",
                (entry["observation_id"],),
            ).fetchone()
        if not existing:
            existing = cursor.execute(
                "SELECT id FROM screen_observations WHERE cluster_key = ? LIMIT 1",
                (entry["cluster_key"],),
            ).fetchone()
        if existing:
            observation_id = existing[0]
            cursor.execute(
                """
                UPDATE screen_observations
                SET observation_type = ?, scope_type = ?, scope_id = ?, cluster_key = ?, period_key = ?,
                    period_start = ?, period_end = ?, title = ?, summary_text = ?,
                    progress_text = ?, project_key = ?, objective_key = ?, work_type = ?,
                    category = ?, key_points_json = ?, decisions_json = ?, blockers_json = ?,
                    next_actions_json = ?, entities_json = ?, artifacts_json = ?,
                    evidence_view_ids_json = ?, evidence_record_ids_json = ?,
                    evidence_window_workstream_ids_json = ?, confidence = ?,
                    generation_method = ?, metadata_json = ?, llm_summary_json = ?,
                    llm_model = ?, llm_status = ?, llm_error = ?, llm_hash = ?,
                    llm_updated_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    entry["observation_type"],
                    entry["scope_type"],
                    entry["scope_id"],
                    entry["cluster_key"],
                    entry["period_key"],
                    entry["period_start"],
                    entry["period_end"],
                    entry["title"],
                    entry["summary_text"],
                    entry["progress_text"],
                    entry["project_key"],
                    entry["objective_key"],
                    entry["work_type"],
                    entry["category"],
                    entry["key_points_json"],
                    entry["decisions_json"],
                    entry["blockers_json"],
                    entry["next_actions_json"],
                    entry["entities_json"],
                    entry["artifacts_json"],
                    entry["evidence_view_ids_json"],
                    entry["evidence_record_ids_json"],
                    entry["evidence_window_workstream_ids_json"],
                    entry["confidence"],
                    entry["generation_method"],
                    entry["metadata_json"],
                    entry.get("llm_summary_json"),
                    entry.get("llm_model"),
                    entry.get("llm_status"),
                    entry.get("llm_error"),
                    entry.get("llm_hash"),
                    entry.get("llm_updated_at"),
                    now,
                    observation_id,
                ),
            )
        else:
            cursor.execute(
                """
                INSERT INTO screen_observations
                (observation_type, scope_type, scope_id, cluster_key, period_key,
                 period_start, period_end, title, summary_text, progress_text,
                 project_key, objective_key, work_type, category, key_points_json,
                 decisions_json, blockers_json, next_actions_json, entities_json,
                 artifacts_json, evidence_view_ids_json, evidence_record_ids_json,
                 evidence_window_workstream_ids_json, confidence, generation_method,
                 metadata_json, llm_summary_json, llm_model, llm_status, llm_error,
                 llm_hash, llm_updated_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry["observation_type"],
                    entry["scope_type"],
                    entry["scope_id"],
                    entry["cluster_key"],
                    entry["period_key"],
                    entry["period_start"],
                    entry["period_end"],
                    entry["title"],
                    entry["summary_text"],
                    entry["progress_text"],
                    entry["project_key"],
                    entry["objective_key"],
                    entry["work_type"],
                    entry["category"],
                    entry["key_points_json"],
                    entry["decisions_json"],
                    entry["blockers_json"],
                    entry["next_actions_json"],
                    entry["entities_json"],
                    entry["artifacts_json"],
                    entry["evidence_view_ids_json"],
                    entry["evidence_record_ids_json"],
                    entry["evidence_window_workstream_ids_json"],
                    entry["confidence"],
                    entry["generation_method"],
                    entry["metadata_json"],
                    entry.get("llm_summary_json"),
                    entry.get("llm_model"),
                    entry.get("llm_status"),
                    entry.get("llm_error"),
                    entry.get("llm_hash"),
                    entry.get("llm_updated_at"),
                    now,
                    now,
                ),
            )
            observation_id = cursor.lastrowid
        for fact_id in entry.get("fact_ids") or []:
            cursor.execute(
                """
                INSERT OR IGNORE INTO screen_observation_facts
                (observation_id, fact_id, role, confidence)
                VALUES (?, ?, 'supporting', ?)
                """,
                (observation_id, fact_id, entry["confidence"]),
            )
        return observation_id

    def mark_screen_fact_cluster_observed(
        self,
        fact_cluster_id,
        observation_id,
        observed_fact_count,
    ):
        self._conn.execute(
            """
            UPDATE screen_fact_clusters
            SET observation_id = ?, observed_fact_count = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                observation_id,
                int(observed_fact_count or 0),
                now_db_timestamp(),
                fact_cluster_id,
            ),
        )
        self._conn.commit()

    def get_screen_observation_count(self):

        try:
            return self._conn.cursor().execute("SELECT count(*) FROM screen_observations").fetchone()[0]
        except sqlite3.Error:
            return 0

    def get_task_workstream_count(self):
        try:
            return self._conn.cursor().execute("SELECT count(*) FROM task_workstream").fetchone()[0]
        except sqlite3.Error:
            return 0

    def build_screen_observation_item_from_row(self, item):
        entities = parse_json_list(item.get("entities_json"))
        artifacts = parse_json_list(item.get("artifacts_json"))
        key_points = parse_json_list(item.get("key_points_json"))
        blockers = parse_json_list(item.get("blockers_json"))
        next_actions = parse_json_list(item.get("next_actions_json"))
        evidence_window_workstream_ids = parse_json_list(
            item.get("evidence_window_workstream_ids_json")
        )
        token_text = " ".join([
            item.get("title") or "",
            item.get("summary_text") or "",
            item.get("progress_text") or "",
            " ".join(str(value) for value in entities),
            " ".join(str(value) for value in artifacts),
            " ".join(str(value) for value in key_points),
        ])
        return {
            "id": item.get("id"),
            "observation_type": item.get("observation_type") or "context",
            "scope_type": item.get("scope_type") or "",
            "scope_id": item.get("scope_id"),
            "period_start": item.get("period_start"),
            "period_end": item.get("period_end"),
            "title": item.get("title") or "",
            "summary_text": item.get("summary_text") or "",
            "progress_text": item.get("progress_text") or "",
            "project_key": item.get("project_key") or "unknown",
            "objective_key": item.get("objective_key") or "general",
            "work_type": item.get("work_type") or "other",
            "category": item.get("category") or "other",
            "key_points": key_points,
            "blockers": blockers,
            "next_actions": next_actions,
            "entities": entities,
            "entity_keys": {normalize_signature_text(value) for value in entities if str(value).strip()},
            "artifacts": artifacts,
            "artifact_keys": {normalize_signature_text(value) for value in artifacts if str(value).strip()},
            "evidence_window_workstream_ids": evidence_window_workstream_ids,
            "confidence": item.get("confidence") or 0.0,
            "tokens": tokenize_signature_text(token_text),
        }

    def load_unassigned_screen_observations_for_task_generation(self, limit=None):
        cursor = self._conn.cursor()
        params = []
        limit_clause = ""
        if limit:
            limit_clause = "LIMIT ?"
            params.append(int(limit))
        rows = cursor.execute(
            f"""
            SELECT
                so.id, so.observation_type, so.scope_type, so.scope_id,
                so.period_start, so.period_end, so.title, so.summary_text,
                so.progress_text, so.project_key, so.objective_key,
                so.work_type, so.category, so.key_points_json,
                so.blockers_json, so.next_actions_json, so.entities_json,
                so.artifacts_json, so.evidence_window_workstream_ids_json,
                so.confidence
            FROM screen_observations so
            LEFT JOIN task_workstream_observations two
                ON two.observation_id = so.id
            WHERE two.observation_id IS NULL
            ORDER BY so.period_start ASC, so.id ASC
            {limit_clause}
            """,
            params,
        ).fetchall()
        columns = [column[0] for column in cursor.description]
        return [
            self.build_screen_observation_item_from_row(dict(zip(columns, row)))
            for row in rows
        ]

    def load_task_workstreams_for_observation_generation(self):
        cursor = self._conn.cursor()
        rows = cursor.execute(
            """
            SELECT
                id, title, summary, category, start_timestamp, end_timestamp,
                topics_json, entities_json, artifacts_json, app_names_json,
                window_titles_json, window_workstream_count, view_count,
                segment_count, confidence, task_key, status, project_key,
                objective_key, work_type, progress_text, blockers_json,
                next_actions_json, observation_count, last_activity_timestamp,
                metadata_json, generation_method, created_at, updated_at
            FROM task_workstream
            ORDER BY COALESCE(last_activity_timestamp, end_timestamp, updated_at) DESC, id DESC
            """
        ).fetchall()
        columns = [column[0] for column in cursor.description]
        tasks = []
        for row in rows:
            item = dict(zip(columns, row))
            topics = parse_json_list(item.get("topics_json"))
            entities = parse_json_list(item.get("entities_json"))
            artifacts = parse_json_list(item.get("artifacts_json"))
            app_names = parse_json_list(item.get("app_names_json"))
            window_titles = parse_json_list(item.get("window_titles_json"))
            blockers = parse_json_list(item.get("blockers_json"))
            next_actions = parse_json_list(item.get("next_actions_json"))
            token_text = " ".join([
                item.get("title") or "",
                item.get("summary") or "",
                item.get("progress_text") or "",
                " ".join(str(value) for value in topics),
                " ".join(str(value) for value in entities),
                " ".join(str(value) for value in artifacts),
            ])
            tasks.append({
                "id": item.get("id"),
                "title": item.get("title") or "",
                "summary": item.get("summary") or "",
                "category": item.get("category") or "other",
                "start_timestamp": item.get("start_timestamp"),
                "end_timestamp": item.get("end_timestamp"),
                "topics": topics,
                "entities": entities,
                "entity_keys": {normalize_signature_text(value) for value in entities if str(value).strip()},
                "artifacts": artifacts,
                "artifact_keys": {normalize_signature_text(value) for value in artifacts if str(value).strip()},
                "app_names": app_names,
                "window_titles": window_titles,
                "window_workstream_count": item.get("window_workstream_count") or 0,
                "view_count": item.get("view_count") or 0,
                "segment_count": item.get("segment_count") or 0,
                "confidence": item.get("confidence") or 0.0,
                "task_key": item.get("task_key") or "",
                "status": item.get("status") or "active",
                "project_key": item.get("project_key") or "unknown",
                "objective_key": item.get("objective_key") or "general",
                "work_type": item.get("work_type") or "other",
                "progress_text": item.get("progress_text") or "",
                "blockers": blockers,
                "next_actions": next_actions,
                "observation_count": item.get("observation_count") or 0,
                "last_activity_timestamp": item.get("last_activity_timestamp"),
                "metadata_json": item.get("metadata_json"),
                "generation_method": item.get("generation_method"),
                "tokens": tokenize_signature_text(token_text),
            })
        return tasks

    def save_or_update_observation_task_workstream(self, task_entry):
        cursor = self._conn.cursor()
        now = now_db_timestamp()
        task_workstream_id = task_entry.get("id")
        values = (
            task_entry.get("title") or "",
            task_entry.get("summary") or "",
            task_entry.get("category") or "general_work",
            format_db_timestamp(task_entry.get("start_timestamp")),
            format_db_timestamp(task_entry.get("end_timestamp")),
            task_entry.get("topics_json") or "[]",
            task_entry.get("entities_json") or "[]",
            task_entry.get("artifacts_json") or "[]",
            task_entry.get("app_names_json") or "[]",
            task_entry.get("window_titles_json") or "[]",
            int(task_entry.get("window_workstream_count") or 0),
            int(task_entry.get("view_count") or 0),
            int(task_entry.get("segment_count") or 0),
            float(task_entry.get("confidence") or 0.0),
            task_entry.get("task_key") or "",
            task_entry.get("status") or "active",
            task_entry.get("project_key") or "unknown",
            task_entry.get("objective_key") or "general",
            task_entry.get("work_type") or "other",
            task_entry.get("progress_text") or "",
            task_entry.get("blockers_json") or "[]",
            task_entry.get("next_actions_json") or "[]",
            int(task_entry.get("observation_count") or 0),
            format_db_timestamp(task_entry.get("last_activity_timestamp") or task_entry.get("end_timestamp")),
            task_entry.get("metadata_json") or "{}",
            task_entry.get("generation_method") or "screen_observation_task_clustering",
            task_entry.get("llm_model") or "",
            task_entry.get("llm_status") or "",
            task_entry.get("llm_error") or "",
            format_db_timestamp(task_entry.get("llm_updated_at")) if task_entry.get("llm_updated_at") else None,
        )
        if task_workstream_id:
            cursor.execute(
                """
                UPDATE task_workstream
                SET title = ?, summary = ?, category = ?, start_timestamp = ?,
                    end_timestamp = ?, topics_json = ?, entities_json = ?,
                    artifacts_json = ?, app_names_json = ?, window_titles_json = ?,
                    window_workstream_count = ?, view_count = ?, segment_count = ?,
                    confidence = ?, task_key = ?, status = ?, project_key = ?,
                    objective_key = ?, work_type = ?, progress_text = ?,
                    blockers_json = ?, next_actions_json = ?, observation_count = ?,
                    last_activity_timestamp = ?, metadata_json = ?,
                    generation_method = ?, llm_model = ?, llm_status = ?,
                    llm_error = ?, llm_updated_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (*values, now, task_workstream_id),
            )
        else:
            cursor.execute(
                """
                INSERT INTO task_workstream
                (title, summary, category, start_timestamp, end_timestamp,
                 topics_json, entities_json, artifacts_json, app_names_json,
                 window_titles_json, window_workstream_count, view_count,
                 segment_count, confidence, task_key, status, project_key,
                 objective_key, work_type, progress_text, blockers_json,
                 next_actions_json, observation_count, last_activity_timestamp,
                 metadata_json, generation_method, llm_model, llm_status,
                 llm_error, llm_updated_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (*values, now, now),
            )
            task_workstream_id = cursor.lastrowid
        return int(task_workstream_id)

    def attach_observation_to_task_workstream(
        self,
        task_workstream_id,
        observation_id,
        *,
        role="evidence",
        confidence=1.0,
        reason="",
    ):
        self._conn.execute(
            """
            INSERT OR IGNORE INTO task_workstream_observations
            (task_workstream_id, observation_id, role, confidence, reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                int(task_workstream_id),
                int(observation_id),
                role,
                float(confidence or 0.0),
                reason,
                now_db_timestamp(),
            ),
        )
        self._conn.commit()

    def load_dirty_persisted_screen_fact_clusters(self, window_workstream_ids=None):
        cursor = self._conn.cursor()
        params = []
        where = ""
        if window_workstream_ids:
            placeholders = ",".join("?" for _ in window_workstream_ids)
            where = f"WHERE fc.window_workstream_id IN ({placeholders})"
            params.extend(int(item) for item in window_workstream_ids)
        rows = cursor.execute(
            f"""
            SELECT fc.window_workstream_id, fc.id, fc.observation_id,
                   fc.observed_fact_count, COUNT(cm.fact_id) AS fact_count
            FROM screen_fact_clusters fc
            JOIN screen_fact_cluster_members cm ON cm.cluster_id = fc.id
            {where}
            GROUP BY fc.id
            HAVING fc.observation_id IS NULL
                OR COUNT(cm.fact_id) > fc.observed_fact_count
            ORDER BY fc.window_workstream_id ASC, fc.id ASC
            """,
            params,
        ).fetchall()
        dirty_by_workstream = {}
        for window_workstream_id, cluster_id, _observation_id, _observed, _count in rows:
            dirty_by_workstream.setdefault(int(window_workstream_id), set()).add(int(cluster_id))
        return dirty_by_workstream

    def load_unclustered_screen_facts_for_window_workstream(self, window_workstream_id):
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT
                sf.id, sf.view_id, sf.fact_text, sf.fact_type, sf.fact_kind,
                sf.work_type, sf.project_key, sf.objective_key, sf.topics_json,
                sf.entities_json, sf.artifacts_json, sf.evidence_text,
                sf.evidence_record_ids_json, sf.app_name, sf.window_title,
                sf.start_timestamp, sf.end_timestamp, sf.confidence,
                sf.embedding_text, sf.embedding_model,
                sf.embedding_dimensions, sf.embedding_vector
            FROM screen_facts sf
            JOIN window_workstream_members wm ON wm.view_id = sf.view_id
            LEFT JOIN screen_fact_cluster_members cm ON cm.fact_id = sf.id
            WHERE wm.window_workstream_id = ?
              AND cm.fact_id IS NULL
            ORDER BY sf.start_timestamp ASC, sf.id ASC
            """,
            (window_workstream_id,),
        )
        columns = [column[0] for column in cursor.description]
        return [
            self.build_screen_fact_item_from_row(dict(zip(columns, row)))
            for row in cursor.fetchall()
        ]

    def load_unobserved_screen_facts_for_window_workstream(self, window_workstream_id):
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT
                sf.id, sf.view_id, sf.fact_text, sf.fact_type, sf.fact_kind,
                sf.work_type, sf.project_key, sf.objective_key, sf.topics_json,
                sf.entities_json, sf.artifacts_json, sf.evidence_text,
                sf.evidence_record_ids_json, sf.app_name, sf.window_title,
                sf.start_timestamp, sf.end_timestamp, sf.confidence,
                sf.embedding_text, sf.embedding_model,
                sf.embedding_dimensions, sf.embedding_vector
            FROM screen_facts sf
            JOIN window_workstream_members wm ON wm.view_id = sf.view_id
            LEFT JOIN screen_observation_facts sof ON sof.fact_id = sf.id
            WHERE wm.window_workstream_id = ?
              AND sof.fact_id IS NULL
            ORDER BY sf.start_timestamp ASC, sf.id ASC
            """,
            (window_workstream_id,),
        )
        columns = [column[0] for column in cursor.description]
        facts = []
        for row in cursor.fetchall():
            item = dict(zip(columns, row))
            facts.append(self.build_screen_fact_item_from_row(item))
        return facts

    def load_persisted_screen_fact_clusters_for_window_workstream(self, window_workstream_id):
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT
                fc.id AS fact_cluster_id,
                fc.cluster_key AS fact_cluster_key,
                fc.cluster_score,
                fc.cluster_reason,
                fc.observation_id,
                fc.observed_fact_count,
                sf.id, sf.view_id, sf.fact_text, sf.fact_type, sf.fact_kind,
                sf.work_type, sf.project_key, sf.objective_key, sf.topics_json,
                sf.entities_json, sf.artifacts_json, sf.evidence_text,
                sf.evidence_record_ids_json, sf.app_name, sf.window_title,
                sf.start_timestamp, sf.end_timestamp, sf.confidence,
                sf.embedding_text, sf.embedding_model,
                sf.embedding_dimensions, sf.embedding_vector
            FROM screen_fact_clusters fc
            JOIN screen_fact_cluster_members cm ON cm.cluster_id = fc.id
            JOIN screen_facts sf ON sf.id = cm.fact_id
            WHERE fc.window_workstream_id = ?
            ORDER BY fc.id ASC, sf.start_timestamp ASC, sf.id ASC
            """,
            (window_workstream_id,),
        )
        columns = [column[0] for column in cursor.description]
        clusters = {}
        for row in cursor.fetchall():
            item = dict(zip(columns, row))
            cluster_id = int(item["fact_cluster_id"])
            cluster = clusters.setdefault(cluster_id, {
                "fact_cluster_id": cluster_id,
                "observation_id": item.get("observation_id"),
                "observation_cluster_key": item.get("fact_cluster_key"),
                "cluster_score": item.get("cluster_score") or 0.0,
                "cluster_reason": item.get("cluster_reason") or "",
                "observed_fact_count": int(item.get("observed_fact_count") or 0),
                "facts": [],
            })
            cluster["facts"].append(self.build_screen_fact_item_from_row(item))
        return list(clusters.values())

    def build_screen_fact_item_from_row(self, item):
        topics = parse_json_list(item.get("topics_json"))
        entities = parse_json_list(item.get("entities_json"))
        artifacts = parse_json_list(item.get("artifacts_json"))
        evidence_record_ids = parse_json_list(item.get("evidence_record_ids_json"))
        signature_text = " ".join([
            item.get("fact_text") or "",
            item.get("evidence_text") or "",
            " ".join(topics),
            " ".join(entities),
            " ".join(artifacts),
            item.get("project_key") or "",
            item.get("objective_key") or "",
        ])
        return {
            "id": item["id"],
            "view_id": item["view_id"],
            "fact_text": item.get("fact_text") or "",
            "fact_type": item.get("fact_type") or "episodic",
            "fact_kind": item.get("fact_kind") or "other",
            "work_type": item.get("work_type") or "other",
            "project_key": item.get("project_key") or "unknown",
            "objective_key": item.get("objective_key") or "general",
            "topics": topics,
            "topic_keys": {normalize_signature_text(value) for value in topics if str(value).strip()},
            "entities": entities,
            "entity_keys": {normalize_signature_text(value) for value in entities if str(value).strip()},
            "artifacts": artifacts,
            "artifact_keys": {normalize_signature_text(value) for value in artifacts if str(value).strip()},
            "evidence_text": item.get("evidence_text") or "",
            "evidence_record_ids": evidence_record_ids,
            "app_name": item.get("app_name") or "",
            "window_title": item.get("window_title") or "",
            "start_timestamp": item.get("start_timestamp"),
            "end_timestamp": item.get("end_timestamp"),
            "confidence": item.get("confidence") or 0.0,
            "tokens": tokenize_signature_text(signature_text),
            "embedding_text": item.get("embedding_text") or "",
            "embedding_model": item.get("embedding_model"),
            "embedding_dimensions": item.get("embedding_dimensions"),
            "embedding_vector": self.decode_embedding_vector(
                item.get("embedding_vector"),
                item.get("embedding_dimensions"),
            ),
        }

    def load_existing_window_workstreams(self, cursor):
        cursor.execute("""
            SELECT
                id,
                title,
                summary,
                category,
                start_timestamp,
                end_timestamp,
                topics_json,
                entities_json,
                artifacts_json,
                app_names_json,
                window_titles_json,
                view_count,
                segment_count,
                confidence,
                llm_summary_json,
                llm_model,
                llm_status,
                llm_error,
                llm_hash,
                llm_updated_at,
                created_at,
                updated_at
            FROM window_workstream
            ORDER BY start_timestamp ASC, id ASC
        """)
        columns = [column[0] for column in cursor.description]
        window_workstream = []
        for row in cursor.fetchall():
            item = dict(zip(columns, row))
            topic_list = parse_json_list(item.get("topics_json"))
            entity_list = parse_json_list(item.get("entities_json"))
            artifact_list = parse_json_list(item.get("artifacts_json"))
            app_names = parse_json_list(item.get("app_names_json"))
            window_titles = parse_json_list(item.get("window_titles_json"))
            token_text = " ".join([
                item.get("title") or "",
                item.get("summary") or "",
                " ".join(str(value) for value in topic_list),
                " ".join(str(value) for value in entity_list),
                " ".join(str(value) for value in artifact_list),
                " ".join(str(value) for value in app_names),
                " ".join(str(value) for value in window_titles),
            ])

            segment_rows = cursor.execute(
                """
                SELECT DISTINCT vs.segment_id
                FROM window_workstream_members wm
                JOIN view_segments vs ON vs.view_id = wm.view_id
                WHERE wm.window_workstream_id = ?
                """,
                (item["id"],),
            ).fetchall()
            existing_view_count = item.get("view_count") or 0
            confidence = item.get("confidence") or 0.0
            workstream = {
                "id": item["id"],
                "existing_view_count": existing_view_count,
                "summary": item.get("summary") or "",
                "category": item.get("category") or "",
                "views": [],
                "members": [],
                "start_timestamp": item.get("start_timestamp"),
                "end_timestamp": item.get("end_timestamp"),
                "app_names": app_names,
                "app_keys": {normalize_signature_text(app_name) for app_name in app_names if str(app_name).strip()},
                "window_titles": window_titles,
                "title_keys": {normalize_signature_text(title) for title in window_titles if str(title).strip()},
                "content_kinds": Counter({item.get("category") or "other": max(1, existing_view_count)}),
                "topics": topic_list,
                "topic_keys": {normalize_signature_text(topic_name) for topic_name in topic_list if str(topic_name).strip()},
                "entities": entity_list,
                "entity_keys": {normalize_signature_text(entity) for entity in entity_list if str(entity).strip()},
                "artifacts": artifact_list,
                "artifact_keys": {normalize_signature_text(artifact) for artifact in artifact_list if str(artifact).strip()},
                "tokens": tokenize_signature_text(token_text),
                "segment_ids": {row[0] for row in segment_rows if row[0] is not None},
                "confidence_values": [confidence] * max(1, existing_view_count),
                "relevance_values": [confidence] * max(1, existing_view_count),
                "llm_summary_json": item.get("llm_summary_json"),
                "llm_model": item.get("llm_model"),
                "llm_status": item.get("llm_status"),
                "llm_error": item.get("llm_error"),
                "llm_hash": item.get("llm_hash"),
                "llm_updated_at": item.get("llm_updated_at"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
            }
            window_workstream.append(workstream)
        for workstream in window_workstream:
            workstream["member_views"] = self.load_window_workstream_member_views(cursor, workstream["id"])
        return window_workstream

    def load_window_workstream_signatures_for_task_generation(self, window_workstream_ids=None):

        cursor = self._conn.cursor()
        where_clause = ""
        params = []
        if window_workstream_ids is not None:
            window_workstream_ids = [item for item in window_workstream_ids if item is not None]
            if not window_workstream_ids:
                return []
            placeholders = ",".join("?" for _ in window_workstream_ids)
            where_clause = f"WHERE ww.id IN ({placeholders})"
            params = window_workstream_ids

        cursor.execute(f"""
            SELECT
                ww.id,
                ww.title,
                ww.summary,
                ww.category,
                ww.start_timestamp,
                ww.end_timestamp,
                ww.topics_json,
                ww.entities_json,
                ww.artifacts_json,
                ww.app_names_json,
                ww.window_titles_json,
                ww.view_count,
                ww.segment_count,
                ww.confidence,
                ww.created_at,
                ww.updated_at
            FROM window_workstream ww
            {where_clause}
            ORDER BY ww.start_timestamp ASC, ww.id ASC
        """, params)
        columns = [column[0] for column in cursor.description]
        items = []
        for row in cursor.fetchall():
            item = dict(zip(columns, row))
            topics = parse_json_list(item.get("topics_json"))
            entities = parse_json_list(item.get("entities_json"))
            artifacts = parse_json_list(item.get("artifacts_json"))
            app_names = parse_json_list(item.get("app_names_json"))
            window_titles = parse_json_list(item.get("window_titles_json"))
            signature_text = " ".join([
                item.get("title") or "",
                item.get("summary") or "",
                " ".join(str(value) for value in topics),
                " ".join(str(value) for value in entities),
                " ".join(str(value) for value in artifacts),
                " ".join(str(value) for value in app_names),
                " ".join(str(value) for value in window_titles),
            ])
            try:
                confidence = float(item.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            items.append({
                "id": item["id"],
                "title": item.get("title") or "",
                "summary": item.get("summary") or "",
                "category": item.get("category") or "other",
                "start_timestamp": item.get("start_timestamp"),
                "end_timestamp": item.get("end_timestamp"),
                "topics": topics,
                "topic_keys": {normalize_signature_text(value) for value in topics if str(value).strip()},
                "entities": entities,
                "entity_keys": {normalize_signature_text(value) for value in entities if str(value).strip()},
                "artifacts": artifacts,
                "artifact_keys": {normalize_signature_text(value) for value in artifacts if str(value).strip()},
                "app_names": app_names,
                "window_titles": window_titles,
                "tokens": tokenize_signature_text(signature_text),
                "view_count": item.get("view_count") or 0,
                "segment_count": item.get("segment_count") or 0,
                "confidence": max(0.0, min(1.0, confidence)),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
            })
        return items

    def load_window_workstream_member_views(self, window_workstream_id):

        cursor = self._conn.cursor()
        max_member_views = self.window_workstream_cfg.get("max_member_views_for_matching", 12)
        member_rows = cursor.execute(
            """
            SELECT view_id
            FROM window_workstream_members
            WHERE window_workstream_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (window_workstream_id, max_member_views),
        ).fetchall()
        member_view_ids = [row[0] for row in member_rows]
        return self.load_view_signatures_for_window_workstream_generation(view_ids=member_view_ids)

    def load_existing_window_workstreams(self):

        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT
                id,
                title,
                summary,
                category,
                start_timestamp,
                end_timestamp,
                topics_json,
                entities_json,
                artifacts_json,
                app_names_json,
                window_titles_json,
                view_count,
                segment_count,
                confidence,
                llm_summary_json,
                llm_model,
                llm_status,
                llm_error,
                llm_hash,
                llm_updated_at,
                created_at,
                updated_at
            FROM window_workstream
            ORDER BY start_timestamp ASC, id ASC
        """)
        columns = [column[0] for column in cursor.description]
        window_workstream = []
        for row in cursor.fetchall():
            item = dict(zip(columns, row))
            topic_list = parse_json_list(item.get("topics_json"))
            entity_list = parse_json_list(item.get("entities_json"))
            artifact_list = parse_json_list(item.get("artifacts_json"))
            app_names = parse_json_list(item.get("app_names_json"))
            window_titles = parse_json_list(item.get("window_titles_json"))
            token_text = " ".join([
                item.get("title") or "",
                item.get("summary") or "",
                " ".join(str(value) for value in topic_list),
                " ".join(str(value) for value in entity_list),
                " ".join(str(value) for value in artifact_list),
                " ".join(str(value) for value in app_names),
                " ".join(str(value) for value in window_titles),
            ])

            segment_rows = cursor.execute(
                """
                SELECT DISTINCT vs.segment_id
                FROM window_workstream_members wm
                JOIN view_segments vs ON vs.view_id = wm.view_id
                WHERE wm.window_workstream_id = ?
                """,
                (item["id"],),
            ).fetchall()
            existing_view_count = item.get("view_count") or 0
            confidence = item.get("confidence") or 0.0
            workstream = {
                "id": item["id"],
                "existing_view_count": existing_view_count,
                "summary": item.get("summary") or "",
                "category": item.get("category") or "",
                "views": [],
                "members": [],
                "start_timestamp": item.get("start_timestamp"),
                "end_timestamp": item.get("end_timestamp"),
                "app_names": app_names,
                "app_keys": {normalize_signature_text(app_name) for app_name in app_names if str(app_name).strip()},
                "window_titles": window_titles,
                "title_keys": {normalize_signature_text(title) for title in window_titles if str(title).strip()},
                "content_kinds": Counter({item.get("category") or "other": max(1, existing_view_count)}),
                "topics": topic_list,
                "topic_keys": {normalize_signature_text(topic_name) for topic_name in topic_list if str(topic_name).strip()},
                "entities": entity_list,
                "entity_keys": {normalize_signature_text(entity) for entity in entity_list if str(entity).strip()},
                "artifacts": artifact_list,
                "artifact_keys": {normalize_signature_text(artifact) for artifact in artifact_list if str(artifact).strip()},
                "tokens": tokenize_signature_text(token_text),
                "segment_ids": {row[0] for row in segment_rows if row[0] is not None},
                "confidence_values": [confidence] * max(1, existing_view_count),
                "relevance_values": [confidence] * max(1, existing_view_count),
                "llm_summary_json": item.get("llm_summary_json"),
                "llm_model": item.get("llm_model"),
                "llm_status": item.get("llm_status"),
                "llm_error": item.get("llm_error"),
                "llm_hash": item.get("llm_hash"),
                "llm_updated_at": item.get("llm_updated_at"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
            }
            window_workstream.append(workstream)
        for workstream in window_workstream:
            workstream["member_views"] = self.load_window_workstream_member_views(cursor, workstream["id"])
        return window_workstream

    def load_existing_task_workstreams(self):

        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT
                id,
                title,
                summary,
                category,
                start_timestamp,
                end_timestamp,
                topics_json,
                entities_json,
                artifacts_json,
                app_names_json,
                window_titles_json,
                window_workstream_count,
                view_count,
                segment_count,
                confidence,
                llm_summary_json,
                llm_model,
                llm_status,
                llm_error,
                llm_hash,
                llm_updated_at,
                created_at,
                updated_at
            FROM task_workstream
            ORDER BY start_timestamp ASC, id ASC
        """)
        columns = [column[0] for column in cursor.description]
        tasks = []
        for row in cursor.fetchall():
            item = dict(zip(columns, row))
            topics = parse_json_list(item.get("topics_json"))
            entities = parse_json_list(item.get("entities_json"))
            artifacts = parse_json_list(item.get("artifacts_json"))
            app_names = parse_json_list(item.get("app_names_json"))
            window_titles = parse_json_list(item.get("window_titles_json"))
            token_text = " ".join([
                item.get("title") or "",
                item.get("summary") or "",
                " ".join(str(value) for value in topics),
                " ".join(str(value) for value in entities),
                " ".join(str(value) for value in artifacts),
                " ".join(str(value) for value in app_names),
                " ".join(str(value) for value in window_titles),
            ])
            confidence = item.get("confidence") or 0.0
            task = {
                "id": item["id"],
                "existing_window_workstream_count": item.get("window_workstream_count") or 0,
                "existing_view_count": item.get("view_count") or 0,
                "existing_segment_count": item.get("segment_count") or 0,
                "title": item.get("title") or "",
                "summary": item.get("summary") or "",
                "category": item.get("category") or "",
                "window_workstreams": [],
                "member_window_workstreams": [],
                "members": [],
                "start_timestamp": item.get("start_timestamp"),
                "end_timestamp": item.get("end_timestamp"),
                "topics": topics,
                "topic_keys": {normalize_signature_text(value) for value in topics if str(value).strip()},
                "entities": entities,
                "entity_keys": {normalize_signature_text(value) for value in entities if str(value).strip()},
                "artifacts": artifacts,
                "artifact_keys": {normalize_signature_text(value) for value in artifacts if str(value).strip()},
                "app_names": app_names,
                "window_titles": window_titles,
                "content_kinds": Counter({item.get("category") or "other": max(1, item.get("window_workstream_count") or 0)}),
                "tokens": tokenize_signature_text(token_text),
                "confidence_values": [confidence] * max(1, item.get("window_workstream_count") or 0),
                "relevance_values": [confidence] * max(1, item.get("window_workstream_count") or 0),
                "llm_summary_json": item.get("llm_summary_json"),
                "llm_model": item.get("llm_model"),
                "llm_status": item.get("llm_status"),
                "llm_error": item.get("llm_error"),
                "llm_hash": item.get("llm_hash"),
                "llm_updated_at": item.get("llm_updated_at"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
            }
            tasks.append(task)
        for task in tasks:
            task["member_window_workstreams"] = self.load_task_member_window_workstreams(cursor, task["id"])
        return tasks

    def load_view_signatures_for_window_workstream_generation(self, view_ids=None):

        cursor = self._conn.cursor()
        where_clause = ""
        params = []
        if view_ids is not None:
            view_ids = [view_id for view_id in view_ids if view_id is not None]
            if not view_ids:
                return []
            placeholders = ",".join("?" for _ in view_ids)
            where_clause = f"WHERE v.id IN ({placeholders})"
            params = view_ids

        cursor.execute(f"""
            SELECT
                v.id,
                v.app_name,
                v.window_title,
                v.content_kind,
                v.start_timestamp,
                v.end_timestamp,
                v.representative_text,
                v.topics_json,
                v.entities_json,
                v.artifacts_json,
                v.llm_summary_json,
                v.llm_summary_text,
                v.confidence,
                v.record_count,
                COALESCE(
                    (
                        SELECT json_group_array(vs.segment_id)
                        FROM view_segments vs
                        WHERE vs.view_id = v.id
                    ),
                    '[]'
                ) AS segment_ids_json,
                s.activity_type AS segment_activity_type
            FROM views v
            LEFT JOIN segments s ON s.id = (
                SELECT vs.segment_id
                FROM view_segments vs
                WHERE vs.view_id = v.id
                ORDER BY vs.record_count DESC, vs.segment_id ASC
                LIMIT 1
            )
            {where_clause}
            ORDER BY v.start_timestamp ASC, v.id ASC
        """, params)
        columns = [column[0] for column in cursor.description]
        views = []
        for row in cursor.fetchall():
            item = dict(zip(columns, row))
            topics = parse_json_list(item.get("topics_json"))
            entities = parse_json_list(item.get("entities_json"))
            artifacts = parse_json_list(item.get("artifacts_json"))
            segment_ids = parse_json_list(item.get("segment_ids_json"))

            representative_text = item.get("representative_text") or ""
            signature_text = " ".join([
                item.get("app_name") or "",
                item.get("window_title") or "",
                item.get("content_kind") or "",
                representative_text,
                " ".join(str(topic) for topic in topics),
                " ".join(str(entity) for entity in entities),
                " ".join(str(artifact) for artifact in artifacts),
            ])
            try:
                confidence = float(item.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            views.append({
                "id": item["id"],
                "segment_ids": segment_ids,
                "app_name": item.get("app_name") or "",
                "app_key": normalize_signature_text(item.get("app_name")),
                "window_title": item.get("window_title") or "",
                "title_key": normalize_signature_text(item.get("window_title")),
                "content_kind": item.get("content_kind") or item.get("segment_activity_type") or "other",
                "start_timestamp": item.get("start_timestamp"),
                "end_timestamp": item.get("end_timestamp"),
                "representative_text": representative_text,
                "topics": topics,
                "topic_keys": {normalize_signature_text(topic) for topic in topics if str(topic).strip()},
                "entities": entities,
                "entity_keys": {normalize_signature_text(entity) for entity in entities if str(entity).strip()},
                "artifacts": artifacts,
                "artifact_keys": {normalize_signature_text(artifact) for artifact in artifacts if str(artifact).strip()},
                "tokens": tokenize_signature_text(signature_text),
                "confidence": max(0.0, min(1.0, confidence)),
                "record_count": item.get("record_count") or 0,
            })
        return views

    def get_workstream_stats(self):

        cursor = self._conn.cursor()
        try:
            total_workstream_count = cursor.execute("SELECT count(*) FROM window_workstream").fetchone()[0]
            total_member_count = cursor.execute("SELECT count(*) FROM window_workstream_members").fetchone()[0]
            total_task_count = cursor.execute("SELECT count(*) FROM task_workstream").fetchone()[0]
            total_task_member_count = cursor.execute("SELECT count(*) FROM task_workstream_members").fetchone()[0]
            total_report_block_count = cursor.execute("SELECT count(*) FROM report_blocks").fetchone()[0]
            total_screen_fact_count = cursor.execute("SELECT count(*) FROM screen_facts").fetchone()[0]
            total_screen_observation_count = cursor.execute("SELECT count(*) FROM screen_observations").fetchone()[0]
        except sqlite3.Error:
            return {
                "window_workstream": 0,
                "window_workstream_members": 0,
                "task_workstream": 0,
                "task_workstream_members": 0,
                "report_blocks": 0,
                "screen_facts": 0,
                "screen_observations": 0,
            }
        return {
            "window_workstream": total_workstream_count,
            "window_workstream_members": total_member_count,
            "task_workstream": total_task_count,
            "task_workstream_members": total_task_member_count,
            "report_blocks": total_report_block_count,
            "screen_facts": total_screen_fact_count,
            "screen_observations": total_screen_observation_count,
        }
    def save_window_workstream(self, workstream_entry):

        cursor = self._conn.cursor()
        now = now_db_timestamp()
        cursor.execute(
            """
            INSERT INTO window_workstream
            (title, summary, category, start_timestamp, end_timestamp,
             topics_json, entities_json, artifacts_json, app_names_json, window_titles_json,
             view_count, segment_count, confidence, llm_summary_json, llm_model, llm_status,
             llm_error, llm_hash, llm_updated_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workstream_entry["title"],
                workstream_entry["summary"],
                workstream_entry["category"],
                format_db_timestamp(workstream_entry["start_timestamp"]),
                format_db_timestamp(workstream_entry["end_timestamp"]),
                workstream_entry["topics_json"],
                workstream_entry["entities_json"],
                workstream_entry["artifacts_json"],
                workstream_entry["app_names_json"],
                workstream_entry["window_titles_json"],
                workstream_entry["view_count"],
                workstream_entry["segment_count"],
                workstream_entry["confidence"],
                workstream_entry.get("llm_summary_json"),
                workstream_entry.get("llm_model"),
                workstream_entry.get("llm_status"),
                workstream_entry.get("llm_error"),
                workstream_entry.get("llm_hash"),
                workstream_entry.get("llm_updated_at"),
                now,
                now,
            ),
        )
        window_workstream_id = cursor.lastrowid
        for member in workstream_entry["members"]:
            cursor.execute(
                """
                INSERT INTO window_workstream_members
                (window_workstream_id, view_id, relevance, reason, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    window_workstream_id,
                    member["view_id"],
                    member["relevance"],
                    member["reason"],
                    now,
                ),
            )
        self._conn.commit()
        return window_workstream_id

    def update_window_workstream(self, workstream_entry):
    
        cursor = self._conn.cursor()
        now = now_db_timestamp()
        window_workstream_id = workstream_entry["id"]
        cursor.execute(
            """
            UPDATE window_workstream
            SET title = ?,
                summary = ?,
                category = ?,
                start_timestamp = ?,
                end_timestamp = ?,
                topics_json = ?,
                entities_json = ?,
                artifacts_json = ?,
                app_names_json = ?,
                window_titles_json = ?,
                view_count = ?,
                segment_count = ?,
                confidence = ?,
                llm_summary_json = ?,
                llm_model = ?,
                llm_status = ?,
                llm_error = ?,
                llm_hash = ?,
                llm_updated_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                workstream_entry["title"],
                workstream_entry["summary"],
                workstream_entry["category"],
                format_db_timestamp(workstream_entry["start_timestamp"]),
                format_db_timestamp(workstream_entry["end_timestamp"]),
                workstream_entry["topics_json"],
                workstream_entry["entities_json"],
                workstream_entry["artifacts_json"],
                workstream_entry["app_names_json"],
                workstream_entry["window_titles_json"],
                workstream_entry["view_count"],
                workstream_entry["segment_count"],
                workstream_entry["confidence"],
                workstream_entry.get("llm_summary_json"),
                workstream_entry.get("llm_model"),
                workstream_entry.get("llm_status"),
                workstream_entry.get("llm_error"),
                workstream_entry.get("llm_hash"),
                workstream_entry.get("llm_updated_at"),
                now,
                window_workstream_id,
            ),
        )
        for member in workstream_entry["members"]:
            cursor.execute(
                """
                INSERT INTO window_workstream_members
                (window_workstream_id, view_id, relevance, reason, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    window_workstream_id,
                    member["view_id"],
                    member["relevance"],
                    member["reason"],
                    now,
                ),
            )
        self._conn.commit()

    def save_or_update_window_workstream(self, workstream_entry):
        if workstream_entry.get("id") is None:
            window_workstream_id = self.save_window_workstream(workstream_entry)
        else:
            self.update_window_workstream(workstream_entry)
            window_workstream_id = workstream_entry["id"]
        
        return window_workstream_id
    
    def write_record_table(self, kept_records):
        inserted_records = []

        cursor = self._conn.cursor()
        for record in kept_records:
            cursor.execute(
                """
                INSERT OR IGNORE INTO records
                (timestamp, app_name, window_title, focused, ocr_text, cleaned_text,
                 ax_window_title, ax_context_json, user_actions_json,
                 text_source, ocr_quality_score, content_kind, trigger_reason,
                 raw_frame_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _format_db_timestamp(record["timestamp"]),
                    record["app"],
                    record["window"],
                    record["focused"],
                    record["text"],
                    record["cleaned_text"],
                    record.get("ax_window_title"),
                    record.get("ax_context_json"),
                    record.get("user_actions_json"),
                    record.get("text_source") or "ocr",
                    record["ocr_quality_score"],
                    record["content_kind"],
                    record["trigger"],
                    record["frame_id"],
                )
            )
            if cursor.rowcount == 0:
                continue
            memory_id = cursor.lastrowid
            inserted_records.append({
                "id": memory_id,
                "timestamp": _format_db_timestamp(record["timestamp"]),
                "timestamp_dt": record["timestamp_dt"],
                "app": record["app"],
                "window": record["window"],
                "focused": record["focused"],
                "text": record["text"],
                "cleaned_text": record["cleaned_text"],
                "ax_window_title": record.get("ax_window_title"),
                "ax_context_json": record.get("ax_context_json"),
                "user_actions_json": record.get("user_actions_json"),
                "user_actions": record.get("user_actions") or [],
                "text_source": record.get("text_source") or "ocr",
                "ocr_quality_score": record["ocr_quality_score"],
                "content_kind": record["content_kind"],
                "trigger": record["trigger"],
                "_record_key": record.get("_record_key"),
                "view_window": record.get("view_window"),
                "app_context": record.get("app_context"),
                "edge_context": record.get("edge_context"),
                "ax_events": record.get("ax_events") or [],
            })

        self._conn.commit()

        return inserted_records

    def write_record_ax_event_links(self, events_by_record_id, event_id_by_source):
        if not events_by_record_id or not event_id_by_source:
            return 0
        cursor = self._conn.cursor()
        link_count = 0

        for record_id, events in events_by_record_id.items():
            for event in events:
                event_id = event_id_by_source.get(event.get("source_capture_id"))
                if not event_id:
                    continue
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO record_ax_events
                    (record_id, openchronicle_event_id, delta_seconds, match_reason)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        record_id,
                        event_id,
                        float(event.get("delta_seconds") or 0.0),
                        event.get("match_reason") or "same_app_nearby",
                    ),
                )
                if cursor.rowcount:
                    link_count += 1
                event["id"] = event_id

        self._conn.commit()
        return link_count

    def write_openchronicle_event_table(self, oc_events):
        if not oc_events:
            return {}
        cursor = self._conn.cursor()
        source_ids = []
        for event in oc_events:
            source_id = event.get("source_capture_id")
            if not source_id:
                continue
            source_ids.append(source_id)
            cursor.execute(
                """
                INSERT INTO openchronicle_events
                (source_capture_id, timestamp, timestamp_epoch, app_name, bundle_id, window_title,
                 event_type, focused_role, focused_value, visible_text, url,
                 normalized_json, app_context_json, feishu_context_json, user_actions_json, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_capture_id) DO UPDATE SET
                    timestamp = excluded.timestamp,
                    timestamp_epoch = excluded.timestamp_epoch,
                    app_name = excluded.app_name,
                    bundle_id = excluded.bundle_id,
                    window_title = excluded.window_title,
                    event_type = excluded.event_type,
                    focused_role = excluded.focused_role,
                    focused_value = excluded.focused_value,
                    visible_text = excluded.visible_text,
                    url = excluded.url,
                    normalized_json = excluded.normalized_json,
                    app_context_json = excluded.app_context_json,
                    feishu_context_json = excluded.feishu_context_json,
                    user_actions_json = excluded.user_actions_json,
                    raw_json = excluded.raw_json
                """,
                (
                    source_id,
                    _format_db_timestamp(event["timestamp"]),
                    event["timestamp_epoch"],
                    event.get("app_name"),
                    event.get("bundle_id"),
                    event.get("window_title"),
                    event.get("event_type"),
                    event.get("focused_role"),
                    event.get("focused_value"),
                    event.get("visible_text"),
                    event.get("url"),
                    json.dumps(event.get("normalized") or {}, ensure_ascii=False),
                    json.dumps(event.get("app_context"), ensure_ascii=False) if event.get("app_context") else None,
                    json.dumps(event.get("feishu_context"), ensure_ascii=False) if event.get("feishu_context") else None,
                    json.dumps(event.get("user_actions") or [], ensure_ascii=False) if event.get("user_actions") else None,
                    json.dumps(event.get("raw") or {}, ensure_ascii=False),
                ),
            )

        if not source_ids:
            self._conn.commit()
            return {}
        rows = []
        for start in range(0, len(source_ids), 500):
            chunk = source_ids[start:start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows.extend(cursor.execute(
                f"""
                SELECT id, source_capture_id
                FROM openchronicle_events
                WHERE source_capture_id IN ({placeholders})
                """,
                chunk,
            ).fetchall())
        self._conn.commit()
        return {source_capture_id: event_id for event_id, source_capture_id in rows}

    def write_view_table(self, view_entries, segment_id_by_key):
        cursor = self._conn.cursor()
        for view_entry in view_entries:
            view_info = dict(view_entry["info"])
            view_id = self.save_view(cursor, view_info)
            view_entry["view_id"] = view_id
            self.save_view_record_links(cursor, view_id, view_entry.get("records"))
            segment_entries = {
                segment_id_by_key[segment_key]: slice_info
                for segment_key, slice_info in view_entry.get("segment_slices", {}).items()
                if segment_key in segment_id_by_key
            }
            self.save_view_segment_links(cursor, view_id, segment_entries)
        self._conn.commit()
        return len(view_entries)

    def save_view(self, cursor, view_summary):
        cursor.execute(
            """
            INSERT INTO views
            (app_name, window_title, content_kind, start_timestamp, end_timestamp,
             representative_text, topics_json, entities_json, artifacts_json,
             evidence_ids_json, llm_summary_json, llm_summary_text, llm_model, llm_status,
             llm_error, llm_hash, llm_updated_at, confidence, record_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                view_summary["app_name"],
                view_summary["window_title"],
                view_summary["content_kind"],
                _format_db_timestamp(view_summary["start_timestamp"]),
                _format_db_timestamp(view_summary["end_timestamp"]),
                view_summary["representative_text"],
                view_summary["topics_json"],
                view_summary["entities_json"],
                view_summary["artifacts_json"],
                view_summary["evidence_ids_json"],
                view_summary.get("llm_summary_json"),
                view_summary.get("llm_summary_text"),
                view_summary.get("llm_model"),
                view_summary.get("llm_status"),
                view_summary.get("llm_error"),
                view_summary.get("llm_hash"),
                view_summary.get("llm_updated_at"),
                view_summary["confidence"],
                view_summary["record_count"],
            ),
        )
        return cursor.lastrowid

    @staticmethod
    def save_view_record_links(cursor, view_id, records):
        links = []
        seen_record_ids = set()
        for record in records or []:
            record_id = record.get("id")
            if record_id is None or record_id in seen_record_ids:
                continue
            seen_record_ids.add(record_id)
            links.append((view_id, record_id))
        if links:
            cursor.executemany(
                "INSERT OR IGNORE INTO view_records (view_id, record_id) VALUES (?, ?)",
                links,
            )

    @staticmethod
    def save_view_segment_links(cursor, view_id, segment_entries):
        for segment_id, slice_info in segment_entries.items():
            cursor.execute(
                """
                INSERT OR REPLACE INTO view_segments
                (view_id, segment_id, record_count, start_timestamp, end_timestamp,
                 representative_text, evidence_ids_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    view_id,
                    segment_id,
                    slice_info["record_count"],
                    _format_db_timestamp(slice_info["start_timestamp"]),
                    _format_db_timestamp(slice_info["end_timestamp"]),
                    slice_info["representative_text"],
                    slice_info["evidence_ids_json"],
                ),
            )

    def write_segment_table(
        self,
        segment_entries,
    ):
        cursor = self._conn.cursor()
        segment_id_by_key = {}
        for segment_entry in segment_entries:
            summary = segment_entry["info"]
            segment_id = self.save_segment(cursor, summary)
            segment_id_by_key[segment_entry["segment_key"]] = segment_id
        self._conn.commit()
        return segment_id_by_key

    @staticmethod
    def save_segment(cursor, segment_summary):
        cursor.execute(
            """
            INSERT INTO segments
            (start_timestamp, end_timestamp, duration_seconds, activity_type, project_hint,
             app_names, window_titles, summary, actions_json, artifacts_json,
             evidence_ids_json, llm_summary_json, llm_summary_text, llm_model, llm_status,
             llm_error, llm_hash, llm_updated_at, confidence, record_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _format_db_timestamp(segment_summary["start_timestamp"]),
                _format_db_timestamp(segment_summary["end_timestamp"]),
                segment_summary["duration_seconds"],
                segment_summary["activity_type"],
                segment_summary["project_hint"],
                segment_summary["app_names"],
                segment_summary["window_titles"],
                segment_summary["summary"],
                segment_summary["actions_json"],
                segment_summary["artifacts_json"],
                segment_summary["evidence_ids_json"],
                segment_summary.get("llm_summary_json"),
                segment_summary.get("llm_summary_text"),
                segment_summary.get("llm_model"),
                segment_summary.get("llm_status"),
                segment_summary.get("llm_error"),
                segment_summary.get("llm_hash"),
                segment_summary.get("llm_updated_at"),
                segment_summary["confidence"],
                segment_summary["record_count"],
            ),
        )
        return cursor.lastrowid

    def save_task_workstream(self, task_entry):
        cursor = self._conn.cursor()
        now = now_db_timestamp()
        cursor.execute(
            """
            INSERT INTO task_workstream
            (title, summary, category, start_timestamp, end_timestamp,
             topics_json, entities_json, artifacts_json, app_names_json, window_titles_json,
             window_workstream_count, view_count, segment_count, confidence,
             llm_summary_json, llm_model, llm_status, llm_error, llm_hash, llm_updated_at,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_entry["title"],
                task_entry["summary"],
                task_entry["category"],
                format_db_timestamp(task_entry["start_timestamp"]),
                format_db_timestamp(task_entry["end_timestamp"]),
                task_entry["topics_json"],
                task_entry["entities_json"],
                task_entry["artifacts_json"],
                task_entry["app_names_json"],
                task_entry["window_titles_json"],
                task_entry["window_workstream_count"],
                task_entry["view_count"],
                task_entry["segment_count"],
                task_entry["confidence"],
                task_entry.get("llm_summary_json"),
                task_entry.get("llm_model"),
                task_entry.get("llm_status"),
                task_entry.get("llm_error"),
                task_entry.get("llm_hash"),
                task_entry.get("llm_updated_at"),
                now,
                now,
            ),
        )
        task_workstream_id = cursor.lastrowid
        for member in task_entry["members"]:
            cursor.execute(
                """
                INSERT INTO task_workstream_members
                (task_workstream_id, window_workstream_id, relevance, reason, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    task_workstream_id,
                    member["window_workstream_id"],
                    member["relevance"],
                    member["reason"],
                    now,
                ),
            )
        return task_workstream_id

    def update_task_workstream(self, task_entry):
        cursor = self._conn.cursor()
        now = now_db_timestamp()
        task_workstream_id = task_entry["id"]
        cursor.execute(
            """
            UPDATE task_workstream
            SET title = ?,
                summary = ?,
                category = ?,
                start_timestamp = ?,
                end_timestamp = ?,
                topics_json = ?,
                entities_json = ?,
                artifacts_json = ?,
                app_names_json = ?,
                window_titles_json = ?,
                window_workstream_count = ?,
                view_count = ?,
                segment_count = ?,
                confidence = ?,
                llm_summary_json = ?,
                llm_model = ?,
                llm_status = ?,
                llm_error = ?,
                llm_hash = ?,
                llm_updated_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                task_entry["title"],
                task_entry["summary"],
                task_entry["category"],
                format_db_timestamp(task_entry["start_timestamp"]),
                format_db_timestamp(task_entry["end_timestamp"]),
                task_entry["topics_json"],
                task_entry["entities_json"],
                task_entry["artifacts_json"],
                task_entry["app_names_json"],
                task_entry["window_titles_json"],
                task_entry["window_workstream_count"],
                task_entry["view_count"],
                task_entry["segment_count"],
                task_entry["confidence"],
                task_entry.get("llm_summary_json"),
                task_entry.get("llm_model"),
                task_entry.get("llm_status"),
                task_entry.get("llm_error"),
                task_entry.get("llm_hash"),
                task_entry.get("llm_updated_at"),
                now,
                task_workstream_id,
            ),
        )
        for member in task_entry["members"]:
            existing_member = cursor.execute(
                """
                SELECT 1
                FROM task_workstream_members
                WHERE task_workstream_id = ? AND window_workstream_id = ?
                LIMIT 1
                """,
                (task_workstream_id, member["window_workstream_id"]),
            ).fetchone()
            if existing_member:
                continue
            cursor.execute(
                """
                INSERT INTO task_workstream_members
                (task_workstream_id, window_workstream_id, relevance, reason, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    task_workstream_id,
                    member["window_workstream_id"],
                    member["relevance"],
                    member["reason"],
                    now,
                ),
            )
