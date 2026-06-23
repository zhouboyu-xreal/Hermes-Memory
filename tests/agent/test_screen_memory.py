import json
import sqlite3
import sys
import threading
from datetime import datetime, timedelta, timezone

from agent.screen_memory.manager import (
    ScreenMemoryManager,
    build_view_representative_text,
    infer_openchronicle_event_user_actions,
    select_app_context_event_for_records,
    summarize_view,
    normalize_openchronicle_capture,
)
from agent.screen_memory.config import load_screen_memory_config
from agent.screen_memory.screen_db import ScreenMemoryDB
from agent.screen_memory import service


def _cleaner(tmp_path):
    config = load_screen_memory_config({
        "screen_memory": {
            "enabled": True,
            "screenpipe_db": str(tmp_path / "screenpipe.db"),
            "openchronicle_db": str(tmp_path / "openchronicle.db"),
            "output_db": str(tmp_path / "screen-memory.db"),
            "screen_memory_generation": {
                "enabled": True,
                "enable_LLM_fact_extraction": False,
                "enable_observation_generation": True,
                "enable_LLM_observation": False,
                "fallback_observation_without_llm": True,
            },
        },
        "embedding": {"enabled": False},
    })
    return ScreenMemoryManager(config)


def _seed_workstream(cursor):
    cursor.execute(
        """
        INSERT INTO views
        (app_name, window_title, content_kind, start_timestamp, end_timestamp,
         representative_text, topics_json, entities_json, artifacts_json,
         evidence_ids_json, confidence, record_count)
        VALUES ('Code', 'memory.py', 'coding', '2026-06-01 10:00:00',
                '2026-06-01 10:30:00', 'memory clustering', '["memory"]',
                '["MemoryNodeManager"]', '["memory.py"]', '[]', 0.9, 2)
        """
    )
    view_id = cursor.lastrowid
    cursor.execute(
        """
        INSERT INTO window_workstream
        (title, summary, category, start_timestamp, end_timestamp, topics_json,
         entities_json, artifacts_json, app_names_json, window_titles_json,
         view_count, segment_count, confidence, created_at, updated_at)
        VALUES ('Code - memory.py', 'Memory work', 'coding',
                '2026-06-01 10:00:00', '2026-06-01 10:30:00', '["memory"]',
                '["MemoryNodeManager"]', '["memory.py"]', '["Code"]',
                '["memory.py"]', 1, 0, 0.9, '2026-06-01 10:30:00',
                '2026-06-01 10:30:00')
        """
    )
    workstream_id = cursor.lastrowid
    cursor.execute(
        """
        INSERT INTO window_workstream_members
        (window_workstream_id, view_id, relevance, reason, created_at)
        VALUES (?, ?, 1.0, 'test', '2026-06-01 10:30:00')
        """,
        (workstream_id, view_id),
    )
    return workstream_id, view_id


def _add_fact(cursor, view_id, _fact_key, text, timestamp):
    cursor.execute(
        """
        INSERT INTO screen_facts
        (view_id, fact_text, fact_type, fact_kind, work_type,
         project_key, objective_key, topics_json, entities_json, artifacts_json,
         evidence_text, evidence_record_ids_json, app_name, window_title,
         start_timestamp, end_timestamp, confidence, created_at, updated_at)
        VALUES (?, ?, 'episodic', 'work_event', 'implementation',
                'hermes-agent', 'screen-memory', '["screen memory"]',
                '["ScreenMemoryManager"]', '["memory.py"]', ?, '[]', 'Code',
                'memory.py', ?, ?, 0.9, ?, ?)
        """,
        (
            view_id,
            text,
            text,
            timestamp,
            timestamp,
            timestamp,
            timestamp,
        ),
    )


def test_screen_memory_defaults_use_profile_output_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))

    config = load_screen_memory_config({
        "screen_memory": {
            "enabled": True,
            "screenpipe_db": str(tmp_path / "screenpipe.db"),
            "openchronicle_db": str(tmp_path / "openchronicle.db"),
        }
    })

    assert config["database"]["cleaned_db"] == str(
        tmp_path / "profile" / "screen_memory" / "memory.db"
    )
    assert config["schedule"]["fact_extraction_interval_minutes"] == 30
    assert config["schedule"]["observation_interval_hours"] == 24


def test_screen_memory_reads_embedding_config_from_hermes_config():
    config = load_screen_memory_config({
        "screen_memory": {"enabled": True},
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "base_url": "https://embedding.example/v1",
            "api_key": "embedding-key",
            "dimensions": 1536,
            "normalize": True,
            "batch_size": 16,
        },
    })

    assert config["embedding"] == {
        "enabled": True,
        "provider": "openai",
        "model": "text-embedding-3-small",
        "base_url": "https://embedding.example/v1",
        "api_key": "embedding-key",
        "api_key_env": "EMBEDDING_API_KEY",
        "dimensions": 1536,
        "normalize": True,
        "batch_size": 16,
    }


def test_screen_memory_preserves_explicitly_disabled_embedding():
    config = load_screen_memory_config({
        "screen_memory": {"enabled": True},
        "embedding": {
            "enabled": False,
            "model": "text-embedding-3-small",
            "base_url": "https://embedding.example/v1",
        },
    })

    assert config["embedding"]["enabled"] is False


def test_screen_memory_embedding_resolves_dedicated_env_key(
    tmp_path, monkeypatch
):
    cleaner = _cleaner(tmp_path)
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-key")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"data":[{"index":0,"embedding":[1.0,0.0]}]}'

    captured = {}

    def _urlopen(request, timeout):
        captured["authorization"] = request.headers.get("Authorization")
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(
        "agent.screen_memory.manager.urllib.request.urlopen",
        _urlopen,
    )

    vectors = cleaner.call_embedding_model(
        ["screen fact"],
        {
            "model": "embedding-model",
            "base_url": "https://embedding.example/v1",
            "api_key": "${EMBEDDING_API_KEY}",
            "api_key_env": "EMBEDDING_API_KEY",
        },
    )

    assert captured["authorization"] == "Bearer embedding-key"
    assert vectors == [[1.0, 0.0]]


def test_screen_memory_ticker_is_process_singleton(monkeypatch):
    service.stop_screen_memory_ticker()
    scheduled = threading.Event()
    calls = []

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"screen_memory": {"enabled": True}},
    )
    monkeypatch.setattr(
        service,
        "load_screen_memory_config",
        lambda _cfg: {"enabled": True},
    )

    def fake_schedule():
        calls.append("tick")
        scheduled.set()
        return None

    monkeypatch.setattr(service, "schedule_screen_memory_tick", fake_schedule)
    try:
        first = service.start_screen_memory_ticker(interval_seconds=60)
        second = service.start_screen_memory_ticker(interval_seconds=60)

        assert first is not None
        assert first is second
        assert scheduled.wait(timeout=1.0)
        assert calls == ["tick"]
    finally:
        service.stop_screen_memory_ticker()


def test_screen_memory_ticker_does_not_start_when_disabled(monkeypatch):
    service.stop_screen_memory_ticker()
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"screen_memory": {"enabled": False}},
    )
    monkeypatch.setattr(
        service,
        "load_screen_memory_config",
        lambda _cfg: {"enabled": False},
    )

    assert service.start_screen_memory_ticker(interval_seconds=1) is None


def test_background_ingest_does_not_redirect_process_stdout(tmp_path):
    expected_stdout = sys.stdout

    class _FakeCleaner:
        config = {"schedule": {"initial_lookback_minutes": 30}}

        def update_screen_facts_table(self, **_kwargs):
            assert sys.stdout is expected_stdout
            return {}

    service._run_fact_extraction(
        _FakeCleaner(),
        {},
        datetime(2026, 6, 1, tzinfo=timezone.utc),
    )


def test_run_fact_extraction_normalizes_window_and_state_to_utc():
    captured = {}

    class _FakeCleaner:
        config = {"schedule": {"initial_lookback_minutes": 30}}

        def update_screen_facts_table(self, **kwargs):
            captured.update(kwargs)
            return {"cleaned_records": 1}

    state = {}
    service._run_fact_extraction(
        _FakeCleaner(),
        state,
        datetime(
            2026,
            6,
            1,
            18,
            0,
            tzinfo=timezone(timedelta(hours=8)),
        ),
    )

    assert captured["start_time_str"] == "2026-06-01T09:30:00Z"
    assert captured["end_time_str"] == "2026-06-01T10:00:00Z"
    assert state["last_fact_extraction_at"] == "2026-06-01T10:00:00Z"


def test_screen_memory_state_times_are_normalized_to_utc():
    expected = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)

    assert service._parse_state_time("2026-06-01T10:00:00Z") == expected
    assert service._parse_state_time("2026-06-01T10:00:00+00:00") == expected
    assert service._parse_state_time("2026-06-01T18:00:00+08:00") == expected
    assert service._parse_state_time("2026-06-01T10:00:00") == expected


def test_screenpipe_window_compares_timestamp_values_across_offsets(tmp_path):
    cleaner = _cleaner(tmp_path)
    screenpipe_db = tmp_path / "screenpipe.db"
    cleaner.screenpipe_db = str(screenpipe_db)
    connection = sqlite3.connect(screenpipe_db)
    connection.executescript(
        """
        CREATE TABLE frames (id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL);
        CREATE TABLE ocr_text (
            id INTEGER PRIMARY KEY,
            frame_id INTEGER NOT NULL,
            app_name TEXT,
            window_name TEXT,
            focused INTEGER,
            text TEXT
        );
        INSERT INTO frames VALUES (1, '2026-06-01T09:59:59+00:00');
        INSERT INTO frames VALUES (2, '2026-06-01T10:00:00Z');
        INSERT INTO frames VALUES (3, '2026-06-01T18:00:30+08:00');
        INSERT INTO frames VALUES (4, '2026-06-01T10:01:01.000000+00:00');
        INSERT INTO ocr_text VALUES (1, 1, 'Code', 'before', 1, 'before');
        INSERT INTO ocr_text VALUES (2, 2, 'Code', 'start', 1, 'start');
        INSERT INTO ocr_text VALUES (3, 3, 'Code', 'offset', 1, 'offset');
        INSERT INTO ocr_text VALUES (4, 4, 'Code', 'after', 1, 'after');
        """
    )
    connection.commit()
    connection.close()

    rows = cleaner.load_screenpipe_data(
        datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
        datetime(2026, 6, 1, 10, 1, tzinfo=timezone.utc),
    )

    assert [row[5] for row in rows] == [2, 3]


def test_update_screen_facts_table_invalid_start_uses_ingest_interval(tmp_path, monkeypatch):
    cleaner = _cleaner(tmp_path)
    cleaner.config["schedule"]["fact_extraction_interval_minutes"] = 45
    captured = {}

    def _capture_window(start_time, end_time):
        captured["start"] = start_time
        captured["end"] = end_time
        raise RuntimeError("stop after window parsing")

    monkeypatch.setattr(cleaner, "load_screenpipe_data", _capture_window)

    assert cleaner.update_screen_facts_table(
        start_time_str="not-a-timestamp",
        end_time_str="2026-06-01T10:00:00Z",
        incremental=True,
    ) is None
    assert captured["end"] == datetime(
        2026, 6, 1, 10, 0, tzinfo=timezone.utc
    )
    assert captured["end"] - captured["start"] == timedelta(minutes=45)


def test_update_screen_facts_table_missing_start_uses_ingest_interval(tmp_path, monkeypatch):
    cleaner = _cleaner(tmp_path)
    cleaner.config["schedule"]["fact_extraction_interval_minutes"] = 20
    captured = {}

    def _capture_window(start_time, end_time):
        captured["start"] = start_time
        captured["end"] = end_time
        raise RuntimeError("stop after window parsing")

    monkeypatch.setattr(cleaner, "load_screenpipe_data", _capture_window)

    assert cleaner.update_screen_facts_table(
        end_time_str="2026-06-01T10:00:00Z",
        incremental=True,
    ) is None
    assert captured["end"] - captured["start"] == timedelta(minutes=20)


def test_quiet_cleaner_suppresses_console_output(tmp_path, capsys):
    cleaner = ScreenMemoryManager(_cleaner(tmp_path).config, quiet=True)

    cleaner._print("hidden", "output")

    assert capsys.readouterr().out == ""


def test_screen_db_ensure_reuses_open_database_and_reopens_closed_database(tmp_path):
    output_path = tmp_path / "screen-memory.db"
    screen_db = ScreenMemoryDB.ensure_cleaned_db(None, output_path)

    assert ScreenMemoryDB.ensure_cleaned_db(screen_db, output_path) is screen_db

    screen_db.close()
    reopened = ScreenMemoryDB.ensure_cleaned_db(screen_db, output_path)

    assert reopened is not screen_db
    assert reopened.connection is not None
    reopened.close()


def test_screen_db_init_replaces_database_with_full_reset(tmp_path):
    output_path = tmp_path / "screen-memory.db"
    screen_db = ScreenMemoryDB.ensure_cleaned_db(None, output_path)
    screen_db.connection.execute(
        """
        INSERT INTO records (timestamp, app_name)
        VALUES ('2026-06-01T10:00:00+08:00', 'Code')
        """
    )
    screen_db.connection.commit()

    reset_db = ScreenMemoryDB.init_cleaned_db(screen_db, output_path)

    record_count = reset_db.connection.execute(
        "SELECT COUNT(*) FROM records"
    ).fetchone()[0]
    assert reset_db is not screen_db
    assert screen_db.connection is None
    assert record_count == 0
    reset_db.close()


def test_fact_clusters_persist_and_daily_observation_updates(tmp_path):
    cleaner = _cleaner(tmp_path)
    cleaner.screen_db = ScreenMemoryDB.ensure_cleaned_db(
        cleaner.screen_db,
        cleaner.cleaned_db,
    )
    connection = cleaner.screen_db.connection
    cursor = connection.cursor()
    workstream_id, view_id = _seed_workstream(cursor)
    _add_fact(
        cursor,
        view_id,
        "fact-1",
        "Implemented persistent screen fact clusters.",
        "2026-06-01 10:05:00",
    )
    _add_fact(
        cursor,
        view_id,
        "fact-2",
        "Connected screen fact clusters to daily observations.",
        "2026-06-01 10:15:00",
    )
    _add_fact(
        cursor,
        view_id,
        "fact-3",
        "Prepared screen fact clusters for observation generation.",
        "2026-06-01 10:20:00",
    )
    _add_fact(
        cursor,
        view_id,
        "fact-4",
        "Checked screen fact clustering persistence.",
        "2026-06-01 10:25:00",
    )
    connection.commit()

    observation_stats = cleaner.update_screen_observation_tables()

    assert observation_stats["screen_facts_clustered"] == 0
    assert cursor.execute(
        "SELECT count(*) FROM screen_fact_cluster_members"
    ).fetchone()[0] == 0

    _add_fact(
        cursor,
        view_id,
        "fact-5",
        "Reached the screen fact clustering threshold.",
        "2026-06-01 10:30:00",
    )
    connection.commit()

    observation_stats = cleaner.update_screen_observation_tables()

    assert observation_stats["screen_facts_clustered"] == 5
    cluster_row = cursor.execute(
        """
        SELECT id, observation_id, observed_fact_count
        FROM screen_fact_clusters
        WHERE window_workstream_id = ?
        """,
        (workstream_id,),
    ).fetchone()
    assert cluster_row[1] is not None
    assert cluster_row[2] == 5
    assert cursor.execute(
        "SELECT count(*) FROM screen_fact_cluster_members WHERE cluster_id = ?",
        (cluster_row[0],),
    ).fetchone()[0] == 5

    assert observation_stats["screen_observations_generated"] == 1
    cluster_row = cursor.execute(
        """
        SELECT id, observation_id, observed_fact_count
        FROM screen_fact_clusters
        WHERE id = ?
        """,
        (cluster_row[0],),
    ).fetchone()
    observation_id = cluster_row[1]
    assert observation_id is not None
    assert cluster_row[2] == 5
    observation_columns = {
        row[1]
        for row in cursor.execute(
            "PRAGMA table_info(screen_observations)"
        ).fetchall()
    }
    assert "observation_type" in observation_columns
    assert "observation_kind" not in observation_columns
    assert cursor.execute(
        "SELECT observation_type FROM screen_observations WHERE id = ?",
        (observation_id,),
    ).fetchone()[0] == "context"

    for index in range(6, 11):
        _add_fact(
            cursor,
            view_id,
            f"fact-{index}",
            f"Verified screen memory clustering update {index}.",
            f"2026-06-01 10:{index + 25:02d}:00",
        )
    connection.commit()
    cleaner.update_screen_observation_tables()

    updated_cluster = cursor.execute(
        """
        SELECT observation_id, observed_fact_count
        FROM screen_fact_clusters
        WHERE id = ?
        """,
        (cluster_row[0],),
    ).fetchone()
    assert updated_cluster[0] == observation_id
    assert updated_cluster[1] == 10
    assert cursor.execute(
        "SELECT count(*) FROM screen_observations"
    ).fetchone()[0] == 1
    assert cursor.execute(
        "SELECT count(*) FROM screen_observation_facts WHERE observation_id = ?",
        (observation_id,),
    ).fetchone()[0] == 10
    connection.close()


def test_low_fact_screen_cluster_uses_fallback_observation(tmp_path):
    cleaner = _cleaner(tmp_path)
    cleaner.screen_observation_cfg["observation_min_fact_count"] = 2
    cleaner.screen_db = ScreenMemoryDB.ensure_cleaned_db(
        cleaner.screen_db,
        cleaner.cleaned_db,
    )
    connection = cleaner.screen_db.connection
    cursor = connection.cursor()
    workstream_id, view_id = _seed_workstream(cursor)
    _add_fact(
        cursor,
        view_id,
        "fact-1",
        "Reviewed a single screen event that is still worth preserving.",
        "2026-06-01 10:05:00",
    )
    connection.commit()
    facts = cleaner.screen_db.load_unclustered_screen_facts_for_window_workstream(
        workstream_id,
    )
    cluster_id = cleaner.screen_db.save_persisted_screen_fact_cluster(
        workstream_id,
        {
            "facts": facts,
            "cluster_score": 0.5,
            "cluster_reason": "single_fact_regression_fixture",
        },
    )
    connection.commit()

    observation_stats = cleaner.update_screen_observation_tables(
        window_workstream_ids=[workstream_id],
    )

    cluster_row = cursor.execute(
        """
        SELECT observation_id, observed_fact_count
        FROM screen_fact_clusters
        WHERE id = ?
        """,
        (cluster_id,),
    ).fetchone()
    assert observation_stats["screen_observations_generated"] == 1
    assert observation_stats["screen_observation_low_fact_fallback_count"] == 1
    assert cluster_row[0] is not None
    assert cluster_row[1] == 1
    assert cursor.execute(
        "SELECT summary_text FROM screen_observations WHERE id = ?",
        (cluster_row[0],),
    ).fetchone()[0] == "Reviewed a single screen event that is still worth preserving."
    assert cursor.execute(
        "SELECT count(*) FROM screen_observation_facts WHERE observation_id = ?",
        (cluster_row[0],),
    ).fetchone()[0] == 1
    connection.close()


def test_screen_observations_generate_task_workstreams(tmp_path):
    cleaner = _cleaner(tmp_path)
    cleaner.task_workstream_cfg["enable_LLM_task_profile_update"] = False
    cleaner.task_workstream_cfg["min_observations_to_create_task"] = 1
    cleaner.screen_db = ScreenMemoryDB.ensure_cleaned_db(
        cleaner.screen_db,
        cleaner.cleaned_db,
    )
    connection = cleaner.screen_db.connection
    cursor = connection.cursor()
    _workstream_id, view_id = _seed_workstream(cursor)
    for index in range(1, 6):
        _add_fact(
            cursor,
            view_id,
            f"fact-{index}",
            f"Improved screen task aggregation from observation {index}.",
            f"2026-06-01 10:{index + 5:02d}:00",
        )
    connection.commit()
    observation_stats = cleaner.update_screen_observation_tables()

    task_stats = cleaner.update_screen_task_tables()

    assert observation_stats["screen_observations_generated"] == 1
    assert task_stats["screen_tasks_generated"] == 1
    assert task_stats["screen_observations_assigned_to_tasks"] == 1
    task_row = cursor.execute(
        """
        SELECT id, title, observation_count, project_key, objective_key
        FROM task_workstream
        """
    ).fetchone()
    assert task_row is not None
    assert task_row[2] == 1
    assert task_row[3] == "hermes-agent"
    assert task_row[4] == "screen-memory"
    assert cursor.execute(
        """
        SELECT count(*)
        FROM task_workstream_observations
        WHERE task_workstream_id = ?
        """,
        (task_row[0],),
    ).fetchone()[0] == 1
    connection.close()


def test_screen_task_generation_defers_singleton_observations_by_default(tmp_path):
    cleaner = _cleaner(tmp_path)
    cleaner.task_workstream_cfg["enable_LLM_observation_matching"] = False
    cleaner.task_workstream_cfg["enable_LLM_task_profile_update"] = False
    cleaner.screen_db = ScreenMemoryDB.ensure_cleaned_db(
        cleaner.screen_db,
        cleaner.cleaned_db,
    )
    connection = cleaner.screen_db.connection
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT INTO screen_observations
        (observation_type, scope_type, scope_id, cluster_key, period_key,
         period_start, period_end, title, summary_text, progress_text,
         project_key, objective_key, work_type, category, key_points_json,
         decisions_json, blockers_json, next_actions_json, entities_json,
         artifacts_json, evidence_view_ids_json, evidence_record_ids_json,
         evidence_window_workstream_ids_json, confidence, generation_method,
         metadata_json, created_at, updated_at)
        VALUES
        ('context', 'window_workstream', 1, 'cluster-singleton', 'period-singleton',
         '2026-06-01 12:00:00', '2026-06-01 12:05:00',
         '单条观察不应立即形成 task',
         '用户只产生了一条 task 候选 observation。',
         '等待后续 observation 形成更稳定任务。',
         'unknown', 'general', 'planning',
         'general_work', '[]', '[]', '[]', '[]',
         '["ScreenMemoryManager"]', '["agent/screen_memory/manager.py"]',
         '[]', '[]', '[1]', 0.7, 'test', '{}',
         '2026-06-01 12:05:00', '2026-06-01 12:05:00')
        """
    )
    connection.commit()

    task_stats = cleaner.update_screen_task_tables()

    assert task_stats["screen_tasks_generated"] == 0
    assert task_stats["screen_observations_assigned_to_tasks"] == 0
    assert task_stats["screen_task_deferred_observation_count"] == 1
    assert cursor.execute("SELECT count(*) FROM task_workstream").fetchone()[0] == 0
    assert cursor.execute("SELECT count(*) FROM task_workstream_observations").fetchone()[0] == 0
    connection.close()


def test_screen_task_generation_builds_pending_task_from_all_observations(tmp_path):
    cleaner = _cleaner(tmp_path)
    cleaner.task_workstream_cfg["enable_LLM_observation_matching"] = False
    cleaner.task_workstream_cfg["enable_LLM_task_profile_update"] = False
    cleaner.screen_db = ScreenMemoryDB.ensure_cleaned_db(
        cleaner.screen_db,
        cleaner.cleaned_db,
    )
    connection = cleaner.screen_db.connection
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT INTO screen_observations
        (observation_type, scope_type, scope_id, cluster_key, period_key,
         period_start, period_end, title, summary_text, progress_text,
         project_key, objective_key, work_type, category, key_points_json,
         decisions_json, blockers_json, next_actions_json, entities_json,
         artifacts_json, evidence_view_ids_json, evidence_record_ids_json,
         evidence_window_workstream_ids_json, confidence, generation_method,
         metadata_json, created_at, updated_at)
        VALUES
        ('context', 'window_workstream', 1, 'cluster-a', 'period-a',
         '2026-06-01 09:00:00', '2026-06-01 09:05:00',
         'screen memory task 聚合',
         '用户在设计 pending observation 生成 task 的聚合逻辑。',
         '完成 pending observation 聚合入口。',
         'hermes-agent', 'screen-memory-task', 'implementation',
         'general_work', '["聚合入口"]', '[]', '[]', '["继续验证"]',
         '["ScreenMemoryManager"]', '["agent/screen_memory/manager.py"]',
         '[]', '[]', '[1]', 0.7, 'test', '{}',
         '2026-06-01 09:05:00', '2026-06-01 09:05:00'),
        ('context', 'window_workstream', 1, 'cluster-b', 'period-b',
         '2026-06-01 09:10:00', '2026-06-01 09:15:00',
         'screen memory task 关联',
         '用户发现组内剩余 observation 没有更新 task profile。',
         '修复 task 与全部 observation 的关联和画像更新。',
         'hermes-agent', 'screen-memory-task', 'implementation',
         'general_work', '["画像更新"]', '[]', '[]', '["补充单测"]',
         '["ScreenMemoryManager", "task_workstream_observations"]', '["agent/screen_memory/manager.py", "agent/screen_memory/screen_db.py"]',
         '[]', '[]', '[2]', 0.9, 'test', '{}',
         '2026-06-01 09:15:00', '2026-06-01 09:15:00')
        """
    )
    connection.commit()

    task_stats = cleaner.update_screen_task_tables()

    assert task_stats["screen_tasks_generated"] == 1
    assert task_stats["screen_observations_assigned_to_tasks"] == 2
    row = cursor.execute(
        """
        SELECT observation_count, start_timestamp, end_timestamp, summary,
               entities_json, artifacts_json, metadata_json,
               window_workstream_count
        FROM task_workstream
        LIMIT 1
        """
    ).fetchone()
    assert row[0] == 2
    assert row[1].startswith("2026-06-01T09:00:00")
    assert row[2].startswith("2026-06-01T09:15:00")
    assert "pending observation 生成 task" in row[3]
    assert "剩余 observation 没有更新 task profile" in row[3]
    assert "ScreenMemoryManager" in row[4]
    assert "task_workstream_observations" in row[4]
    assert "manager.py" in row[5]
    assert "screen_db.py" in row[5]
    assert "screen_observation_group" in row[6]
    assert row[7] == 2
    assert cursor.execute(
        "SELECT count(*) FROM task_workstream_observations"
    ).fetchone()[0] == 2
    connection.close()


def test_screen_task_pending_group_ignores_objective_and_primary_artifact(tmp_path):
    cleaner = _cleaner(tmp_path)
    cleaner.task_workstream_cfg["enable_LLM_observation_matching"] = False
    cleaner.task_workstream_cfg["enable_LLM_task_profile_update"] = False
    cleaner.screen_db = ScreenMemoryDB.ensure_cleaned_db(
        cleaner.screen_db,
        cleaner.cleaned_db,
    )
    connection = cleaner.screen_db.connection
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT INTO screen_observations
        (observation_type, scope_type, scope_id, cluster_key, period_key,
         period_start, period_end, title, summary_text, progress_text,
         project_key, objective_key, work_type, category, key_points_json,
         decisions_json, blockers_json, next_actions_json, entities_json,
         artifacts_json, evidence_view_ids_json, evidence_record_ids_json,
         evidence_window_workstream_ids_json, confidence, generation_method,
         metadata_json, created_at, updated_at)
        VALUES
        ('context', 'window_workstream', 1, 'cluster-a', 'period-a',
         '2026-06-01 13:00:00', '2026-06-01 13:05:00',
         'Aura 配件生态调研',
         '用户整理 Aura 配件生态调研材料。',
         '形成配件调研材料。',
         'aura-accessory-ecosystem', 'accessory-survey', 'research',
         'general_work', '[]', '[]', '[]', '[]',
         '["Aura"]', '["Aura 配件专项调研表.md"]',
         '[]', '[]', '[1]', 0.8, 'test', '{}',
         '2026-06-01 13:05:00', '2026-06-01 13:05:00'),
        ('context', 'window_workstream', 1, 'cluster-b', 'period-b',
         '2026-06-01 13:10:00', '2026-06-01 13:15:00',
         'Aura USB 连接调研',
         '用户梳理 Aura 配件生态 USB 连接链路。',
         '形成 USB 连接链路理解。',
         'aura-accessory-ecosystem', 'usb-connection-optimization', 'research',
         'general_work', '[]', '[]', '[]', '[]',
         '["XREAL"]', '["PRD文档"]',
         '[]', '[]', '[2]', 0.8, 'test', '{}',
         '2026-06-01 13:15:00', '2026-06-01 13:15:00')
        """
    )
    connection.commit()

    task_stats = cleaner.update_screen_task_tables()

    assert task_stats["screen_tasks_generated"] == 1
    assert task_stats["screen_observations_assigned_to_tasks"] == 2
    row = cursor.execute(
        """
        SELECT project_key, work_type, observation_count, summary, artifacts_json
        FROM task_workstream
        LIMIT 1
        """
    ).fetchone()
    assert row[0] == "aura-accessory-ecosystem"
    assert row[1] == "research"
    assert row[2] == 2
    assert "配件生态调研材料" in row[3]
    assert "USB 连接链路" in row[3]
    assert "Aura 配件专项调研表.md" in row[4]
    assert "PRD文档" in row[4]
    connection.close()


def test_screen_task_matching_key_ignores_objective_and_primary_artifact(tmp_path):
    cleaner = _cleaner(tmp_path)
    cleaner.task_workstream_cfg["enable_LLM_observation_matching"] = False
    cleaner.task_workstream_cfg["enable_LLM_task_profile_update"] = False
    cleaner.task_workstream_cfg["min_observations_to_create_task"] = 1
    cleaner.screen_db = ScreenMemoryDB.ensure_cleaned_db(
        cleaner.screen_db,
        cleaner.cleaned_db,
    )
    connection = cleaner.screen_db.connection
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT INTO screen_observations
        (observation_type, scope_type, scope_id, cluster_key, period_key,
         period_start, period_end, title, summary_text, progress_text,
         project_key, objective_key, work_type, category, key_points_json,
         decisions_json, blockers_json, next_actions_json, entities_json,
         artifacts_json, evidence_view_ids_json, evidence_record_ids_json,
         evidence_window_workstream_ids_json, confidence, generation_method,
         metadata_json, created_at, updated_at)
        VALUES
        ('context', 'window_workstream', 1, 'cluster-a', 'period-a',
         '2026-06-01 14:00:00', '2026-06-01 14:05:00',
         'Hermes recall ranking 优化',
         '用户优化 Hermes recall ranking 逻辑。',
         '调整 recall ranking。',
         'hermes', 'recall-ranking', 'implementation',
         'general_work', '[]', '[]', '[]', '[]',
         '["Hermes"]', '["memory_node_manager.py"]',
         '[]', '[]', '[1]', 0.8, 'test', '{}',
         '2026-06-01 14:05:00', '2026-06-01 14:05:00'),
        ('context', 'window_workstream', 1, 'cluster-b', 'period-b',
         '2026-06-01 14:10:00', '2026-06-01 14:15:00',
         'Hermes screen task 优化',
         '用户继续优化 Hermes screen task 匹配逻辑。',
         '调整 task matching key。',
         'hermes', 'screen-task-matching', 'implementation',
         'general_work', '[]', '[]', '[]', '[]',
         '["ScreenMemoryManager"]', '["agent/screen_memory/manager.py"]',
         '[]', '[]', '[2]', 0.8, 'test', '{}',
         '2026-06-01 14:15:00', '2026-06-01 14:15:00')
        """
    )
    connection.commit()

    task_stats = cleaner.update_screen_task_tables()

    assert task_stats["screen_tasks_generated"] == 1
    assert task_stats["screen_tasks_updated"] == 1
    assert task_stats["screen_observations_assigned_to_tasks"] == 2
    assert cursor.execute("SELECT count(*) FROM task_workstream").fetchone()[0] == 1
    row = cursor.execute(
        "SELECT task_key, observation_count, artifacts_json FROM task_workstream LIMIT 1"
    ).fetchone()
    assert row[0] == "hermes-implementation"
    assert row[1] == 2
    assert "memory_node_manager.py" in row[2]
    assert "manager.py" in row[2]
    connection.close()


def test_screen_task_unknown_project_key_uses_disambiguation(tmp_path):
    cleaner = _cleaner(tmp_path)
    cleaner.task_workstream_cfg["enable_LLM_observation_matching"] = False
    cleaner.task_workstream_cfg["enable_LLM_task_profile_update"] = False
    cleaner.task_workstream_cfg["min_observations_to_create_task"] = 1
    cleaner.screen_db = ScreenMemoryDB.ensure_cleaned_db(
        cleaner.screen_db,
        cleaner.cleaned_db,
    )
    connection = cleaner.screen_db.connection
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT INTO screen_observations
        (observation_type, scope_type, scope_id, cluster_key, period_key,
         period_start, period_end, title, summary_text, progress_text,
         project_key, objective_key, work_type, category, key_points_json,
         decisions_json, blockers_json, next_actions_json, entities_json,
         artifacts_json, evidence_view_ids_json, evidence_record_ids_json,
         evidence_window_workstream_ids_json, confidence, generation_method,
         metadata_json, created_at, updated_at)
        VALUES
        ('context', 'window_workstream', 1, 'cluster-a', 'period-a',
         '2026-06-01 15:00:00', '2026-06-01 15:05:00',
         'RRA SAR 文案模板查询',
         '用户查询韩国 RRA SAR 文案模板。',
         '获得 SAR 文案模板。',
         'unknown', 'general', 'research',
         'general_work', '[]', '[]', '[]', '[]',
         '["韩国RRA"]', '["SAR文案模板"]',
         '[]', '[]', '[1]', 0.8, 'test', '{}',
         '2026-06-01 15:05:00', '2026-06-01 15:05:00'),
        ('context', 'window_workstream', 1, 'cluster-b', 'period-b',
         '2026-06-01 15:10:00', '2026-06-01 15:15:00',
         'VR180 视场角资料阅读',
         '用户阅读 VR180 模式视场角对比资料。',
         '了解 VR180 视场角差异。',
         'unknown', 'general', 'research',
         'general_work', '[]', '[]', '[]', '[]',
         '["Insta360 EVO"]', '["VR180视场角对比"]',
         '[]', '[]', '[2]', 0.8, 'test', '{}',
         '2026-06-01 15:15:00', '2026-06-01 15:15:00')
        """
    )
    connection.commit()

    task_stats = cleaner.update_screen_task_tables()

    assert task_stats["screen_tasks_generated"] == 2
    assert task_stats["screen_tasks_updated"] == 0
    assert task_stats["screen_observations_assigned_to_tasks"] == 2
    rows = cursor.execute(
        "SELECT task_key, observation_count FROM task_workstream ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0][1] == 1
    assert rows[1][1] == 1
    assert rows[0][0] != rows[1][0]
    connection.close()


def test_screen_task_generation_uses_llm_to_match_candidate_task(tmp_path, monkeypatch):
    cleaner = _cleaner(tmp_path)
    cleaner.task_workstream_cfg["enable_LLM_observation_matching"] = True
    cleaner.task_workstream_cfg["enable_LLM_task_profile_update"] = False
    cleaner.task_workstream_cfg["observation_match_llm_budget"] = 5
    cleaner.task_workstream_cfg["min_observations_to_create_task"] = 1
    cleaner.screen_db = ScreenMemoryDB.ensure_cleaned_db(
        cleaner.screen_db,
        cleaner.cleaned_db,
    )
    connection = cleaner.screen_db.connection
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT INTO screen_observations
        (observation_type, scope_type, scope_id, cluster_key, period_key,
         period_start, period_end, title, summary_text, progress_text,
         project_key, objective_key, work_type, category, key_points_json,
         decisions_json, blockers_json, next_actions_json, entities_json,
         artifacts_json, evidence_view_ids_json, evidence_record_ids_json,
         evidence_window_workstream_ids_json, confidence, generation_method,
         metadata_json, created_at, updated_at)
        VALUES
        ('context', 'window_workstream', 1, 'cluster-a', 'period-a',
         '2026-06-01 10:00:00', '2026-06-01 10:05:00',
         '优化 screen memory task 聚合',
         '用户在优化 screen memory task 聚合逻辑。',
         '推进 observation 到 task 的聚合实现。',
         'hermes-agent', 'screen-memory-task', 'implementation',
         'general_work', '[]', '[]', '[]', '[]',
         '["ScreenMemoryManager"]', '["agent/screen_memory/manager.py"]',
         '[]', '[]', '[1]', 0.8, 'test', '{}',
         '2026-06-01 10:05:00', '2026-06-01 10:05:00'),
        ('context', 'window_workstream', 1, 'cluster-b', 'period-b',
         '2026-06-01 10:10:00', '2026-06-01 10:15:00',
         '继续优化 screen memory task 匹配',
         '用户继续优化 observation 与 task 的匹配逻辑。',
         '使用 LLM 判断 observation 是否属于已有 task。',
         'hermes-agent', 'screen-memory-task', 'implementation',
         'general_work', '[]', '[]', '[]', '[]',
         '["ScreenMemoryManager"]', '["agent/screen_memory/manager.py"]',
         '[]', '[]', '[1]', 0.8, 'test', '{}',
         '2026-06-01 10:15:00', '2026-06-01 10:15:00')
        """
    )
    connection.commit()
    calls = []

    def _match_with_llm(observation, candidates, _config):
        calls.append((observation["id"], [item["task"]["id"] for item in candidates]))
        return {
            "decision": "match_existing",
            "task_workstream_id": candidates[0]["task"]["id"],
            "confidence": 0.91,
            "role": "progress",
            "reason": "同一文件和同一目标的连续优化。",
        }

    monkeypatch.setattr(cleaner, "match_observation_to_task_using_llm", _match_with_llm)

    task_stats = cleaner.update_screen_task_tables()

    assert len(calls) == 1
    assert task_stats["screen_tasks_generated"] == 1
    assert task_stats["screen_tasks_updated"] == 1
    assert task_stats["screen_task_llm_match_count"] == 1
    assert cursor.execute("SELECT count(*) FROM task_workstream").fetchone()[0] == 1
    assert cursor.execute(
        "SELECT count(*) FROM task_workstream_observations"
    ).fetchone()[0] == 2
    role, reason = cursor.execute(
        """
        SELECT role, reason
        FROM task_workstream_observations
        ORDER BY observation_id DESC
        LIMIT 1
        """
    ).fetchone()
    assert role == "progress"
    assert reason.startswith("llm:")
    connection.close()


def test_screen_task_generation_uses_llm_to_update_task_profile(tmp_path, monkeypatch):
    cleaner = _cleaner(tmp_path)
    cleaner.task_workstream_cfg["enable_LLM_observation_matching"] = False
    cleaner.task_workstream_cfg["enable_LLM_task_profile_update"] = True
    cleaner.task_workstream_cfg["task_profile_llm_budget"] = 5
    cleaner.task_workstream_cfg["min_observations_to_create_task"] = 1
    cleaner.screen_db = ScreenMemoryDB.ensure_cleaned_db(
        cleaner.screen_db,
        cleaner.cleaned_db,
    )
    connection = cleaner.screen_db.connection
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT INTO screen_observations
        (observation_type, scope_type, scope_id, cluster_key, period_key,
         period_start, period_end, title, summary_text, progress_text,
         project_key, objective_key, work_type, category, key_points_json,
         decisions_json, blockers_json, next_actions_json, entities_json,
         artifacts_json, evidence_view_ids_json, evidence_record_ids_json,
         evidence_window_workstream_ids_json, confidence, generation_method,
         metadata_json, created_at, updated_at)
        VALUES
        ('context', 'window_workstream', 1, 'cluster-profile', 'period-profile',
         '2026-06-01 11:00:00', '2026-06-01 11:05:00',
         '规则标题',
         '用户实现 observation 到 task 的 profile 更新。',
         'LLM 生成 task profile。',
         'hermes-agent', 'screen-memory-task', 'implementation',
         'general_work', '[]', '[]', '[]', '[]',
         '["ScreenMemoryManager"]', '["agent/screen_memory/manager.py"]',
         '[]', '[]', '[1]', 0.7, 'test', '{}',
         '2026-06-01 11:05:00', '2026-06-01 11:05:00')
        """
    )
    connection.commit()
    calls = []

    def _profile_with_llm(mode, previous_task, observation, local_task, _config):
        calls.append((mode, previous_task, observation["id"], local_task["title"]))
        return {
            "title": "LLM 生成的任务标题",
            "summary": "LLM 归纳后的长期任务摘要。",
            "progress_text": "LLM 归纳后的当前进展。",
            "status": "active",
            "category": "implement_feature",
            "project_key": "hermes-agent",
            "objective_key": "screen-memory-task",
            "work_type": "implementation",
            "entities": ["ScreenMemoryManager", "update_screen_task_tables"],
            "artifacts": ["agent/screen_memory/manager.py"],
            "blockers": [],
            "next_actions": ["继续验证 profile 更新效果"],
            "confidence": 0.92,
        }

    monkeypatch.setattr(cleaner, "update_task_profile_using_llm", _profile_with_llm)

    task_stats = cleaner.update_screen_task_tables()

    assert calls and calls[0][0] == "create"
    assert task_stats["screen_task_profile_llm_count"] == 1
    row = cursor.execute(
        """
        SELECT title, summary, progress_text, confidence, entities_json,
               next_actions_json, metadata_json
        FROM task_workstream
        LIMIT 1
        """
    ).fetchone()
    assert row[0] == "LLM 生成的任务标题"
    assert row[1] == "LLM 归纳后的长期任务摘要。"
    assert row[2] == "LLM 归纳后的当前进展。"
    assert row[3] == 0.92
    assert "update_screen_task_tables" in row[4]
    assert "继续验证 profile 更新效果" in row[5]
    assert "profile_update_method" in row[6]
    connection.close()


def test_screen_memory_service_runs_independent_cadences(tmp_path, monkeypatch):
    config = _cleaner(tmp_path).config
    screenpipe_db = tmp_path / "screenpipe.db"
    openchronicle_db = tmp_path / "openchronicle.db"
    screenpipe_db.touch()
    openchronicle_db.touch()
    config["database"]["screenpipe_db"] = str(screenpipe_db)
    config["database"]["openchronicle_db"] = str(openchronicle_db)

    class _FakeCleaner:
        def __init__(
            self,
            cleaner_config,
            llm_client=None,
            quiet=False,
            screen_db=None,
        ):
            self.config = cleaner_config
            self.screen_db = screen_db

        def close(self):
            return None

    class _FakeScreenDB:
        def __init__(self, output_path, incremental_mode=False):
            self.output_path = str(output_path)

    monkeypatch.setattr(service, "_screen_memory_dir", lambda: tmp_path / "state")
    monkeypatch.setattr(service, "load_screen_memory_config", lambda _cfg: config)
    monkeypatch.setattr(service, "_inject_runtime_config", lambda *_args: None)
    monkeypatch.setattr(service, "ScreenMemoryManager", _FakeCleaner)
    monkeypatch.setattr(service, "ScreenMemoryDB", _FakeScreenDB)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"screen_memory": {"enabled": True}},
    )

    calls = []

    def _ingest(_cleaner, state, now):
        calls.append("ingest")
        state["last_fact_extraction_at"] = now.isoformat()
        return {}

    def _observe(_cleaner, state, now):
        calls.append("observation")
        state["last_observation_at"] = now.isoformat()
        return {}

    def _task(_cleaner, state, now):
        calls.append("task")
        state["last_task_at"] = now.isoformat()
        return {}

    monkeypatch.setattr(service, "_run_fact_extraction", _ingest)
    monkeypatch.setattr(service, "_run_observations", _observe)
    monkeypatch.setattr(service, "_run_tasks", _task)

    start = datetime(2026, 6, 1, tzinfo=timezone.utc)
    service.run_screen_memory_due_work(now=start)
    assert calls == ["ingest", "observation", "task"]

    calls.clear()
    service.run_screen_memory_due_work(
        now=datetime(2026, 6, 1, 1, tzinfo=timezone.utc)
    )
    assert calls == ["ingest"]

    calls.clear()
    service.run_screen_memory_due_work(
        now=datetime(2026, 6, 1, 6, tzinfo=timezone.utc)
    )
    assert calls == ["ingest", "task"]

    calls.clear()
    service.run_screen_memory_due_work(
        now=datetime(2026, 6, 2, tzinfo=timezone.utc)
    )
    assert calls == ["ingest", "observation", "task"]


def test_incremental_update_screen_facts_table_merges_screenpipe_and_openchronicle_into_workstream(tmp_path):
    cleaner = _cleaner(tmp_path)
    screenpipe_db = tmp_path / "screenpipe.db"
    openchronicle_db = tmp_path / "openchronicle.db"
    cleaner.screenpipe_db = str(screenpipe_db)
    cleaner.openchronicle_db = str(openchronicle_db)

    connection = sqlite3.connect(screenpipe_db)
    connection.executescript(
        """
        CREATE TABLE frames (id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL);
        CREATE TABLE ocr_text (
            id INTEGER PRIMARY KEY,
            frame_id INTEGER NOT NULL,
            app_name TEXT,
            window_name TEXT,
            focused INTEGER,
            text TEXT
        );
        INSERT INTO frames VALUES (1, '2026-06-01T10:00:00Z');
        INSERT INTO frames VALUES (2, '2026-06-01T10:00:03Z');
        INSERT INTO ocr_text
        VALUES (1, 1, 'Code', 'memory.py', 1,
                'Implement ScreenMemoryManager records views and workstream pipeline');
        INSERT INTO ocr_text
        VALUES (2, 2, 'Code', 'memory.py', 1,
                'Implement ScreenMemoryManager records views workstream and AXTree pipeline');
        """
    )
    connection.commit()
    connection.close()

    connection = sqlite3.connect(openchronicle_db)
    connection.executescript(
        """
        CREATE TABLE captures (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            app_name TEXT,
            bundle_id TEXT,
            window_title TEXT,
            focused_role TEXT,
            focused_value TEXT,
            visible_text TEXT,
            url TEXT
        );
        INSERT INTO captures
        VALUES ('capture-1', '2026-06-01T10:00:02Z', 'Code',
                'com.microsoft.VSCode', 'memory.py', 'AXTextArea',
                'editor', 'ScreenMemoryManager AXTree structured content', '');
        """
    )
    connection.commit()
    connection.close()

    stats = cleaner.update_screen_facts_table(
        start_time_str="2026-06-01T09:59:00Z",
        end_time_str="2026-06-01T10:01:00Z",
        incremental=True,
    )

    assert stats["cleaned_records"] == 2
    assert stats["openchronicle_events"] == 1
    assert stats["record_ax_event_links"] >= 1
    output = sqlite3.connect(cleaner.cleaned_db)
    assert output.execute("SELECT count(*) FROM views").fetchone()[0] == 1
    assert output.execute("SELECT count(*) FROM window_workstream").fetchone()[0] == 1
    assert output.execute(
        "SELECT count(*) FROM window_workstream_members"
    ).fetchone()[0] == 1
    assert output.execute(
        "SELECT count(*) FROM records WHERE ax_context_json IS NOT NULL"
    ).fetchone()[0] >= 1
    record_timestamp = output.execute(
        "SELECT timestamp FROM records ORDER BY id LIMIT 1"
    ).fetchone()[0]
    event_timestamp = output.execute(
        "SELECT timestamp FROM openchronicle_events ORDER BY id LIMIT 1"
    ).fetchone()[0]
    assert record_timestamp == "2026-06-01T18:00:00+08:00"
    assert event_timestamp == "2026-06-01T18:00:02+08:00"
    output.close()


def test_openchronicle_read_uses_immutable_connection(
    tmp_path,
    monkeypatch,
):
    cleaner = _cleaner(tmp_path)
    openchronicle_db = tmp_path / "openchronicle.db"
    cleaner.openchronicle_db = str(openchronicle_db)
    connection = sqlite3.connect(openchronicle_db)
    connection.executescript(
        """
        CREATE TABLE captures (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            app_name TEXT,
            bundle_id TEXT,
            window_title TEXT,
            focused_role TEXT,
            focused_value TEXT,
            visible_text TEXT,
            url TEXT
        );
        INSERT INTO captures VALUES (
            1, '2026-06-01T10:00:00Z', 'Code', 'com.microsoft.VSCode',
            'memory.py', 'AXTextArea', 'screen memory', 'screen memory', ''
        );
        """
    )
    connection.commit()
    connection.close()

    real_connect = sqlite3.connect
    connection_uris = []

    def guarded_connect(database, *args, **kwargs):
        connection_uris.append(str(database))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", guarded_connect)
    events = cleaner.load_openchronicle_events(
        datetime(2026, 6, 1, 9, tzinfo=timezone.utc),
        datetime(2026, 6, 1, 11, tzinfo=timezone.utc),
    )

    assert len(events) == 1
    assert connection_uris == [
        openchronicle_db.resolve().as_uri() + "?immutable=1"
    ]


def test_select_openchronicle_events_matches_app_aliases(tmp_path):
    cleaner = _cleaner(tmp_path)
    record_time = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)
    records = [
        {
            "id": 1,
            "app": "Safari浏览器",
            "timestamp_dt": record_time,
        },
        {
            "id": 2,
            "app": "终端",
            "timestamp_dt": record_time + timedelta(seconds=2),
        },
    ]
    oc_events = [
        {
            "source_capture_id": "capture-safari",
            "timestamp_dt": record_time + timedelta(seconds=1),
            "timestamp_epoch": int((record_time + timedelta(seconds=1)).timestamp()),
            "app_name": "Safari",
            "bundle_id": "com.apple.Safari",
            "visible_text": "",
            "app_context": None,
        },
        {
            "source_capture_id": "capture-terminal",
            "timestamp_dt": record_time + timedelta(seconds=2),
            "timestamp_epoch": int((record_time + timedelta(seconds=2)).timestamp()),
            "app_name": "Terminal",
            "bundle_id": "com.apple.Terminal",
            "visible_text": "",
            "app_context": None,
        },
    ]

    events_by_record_key, matched = cleaner.select_openchronicle_events_for_records(
        records,
        oc_events,
    )

    assert [event["source_capture_id"] for event in events_by_record_key[1]] == [
        "capture-safari"
    ]
    assert [event["source_capture_id"] for event in events_by_record_key[2]] == [
        "capture-terminal"
    ]
    assert {event["source_capture_id"] for event in matched} == {
        "capture-safari",
        "capture-terminal",
    }


def test_normalize_openchronicle_capture_extracts_safari_context():
    event = normalize_openchronicle_capture(
        {
            "id": "capture-safari-1",
            "timestamp": "2026-05-12T18:09:46+08:00",
            "app_name": "Safari",
            "bundle_id": "com.apple.Safari",
            "window_title": "Agent API Key Validation Issue — Hermes",
            "focused_role": "",
            "focused_value": "",
            "visible_text": "\n".join(
                [
                    "## Safari浏览器 [active]",
                    "_com.apple.Safari_",
                    "### Agent API Key Validation Issue — Hermes",
                    "    - [WebArea] Agent API Key Validation Issue — Hermes",
                    "        - Agent API Key Validation Issue",
                    "        - 23 条消息",
                    "        - [Button] Chat",
                    "        - [Button] Tasks",
                    "        - [Button] Skills",
                    "        - [Button] Memory",
                    "        - [Button] Settings",
                    "        - [Button] New conversation",
                    "        - Hermes 版本查询",
                    "        - PVE 主机死机排查",
                    "        - AI 智能体长效记忆架构与 MemoryLake 演进",
                ]
            ),
            "url": "",
        }
    )

    assert event["normalized"]["has_browser_tab"] is True
    assert event["normalized"]["app_context_route"] == "safari_browser_tab"
    assert event["app_context"]["surface"] == "browser-tab"
    assert event["app_context"]["route"] == "safari_browser_tab"
    assert event["app_context"]["page_title"] == "Agent API Key Validation Issue — Hermes"
    assert "Hermes 版本查询" in event["app_context"]["content_snippets"]
    assert "AI 智能体长效记忆架构与 MemoryLake 演进" in event["app_context"]["content_snippets"]
    assert event["app_context"]["structure_summary"].startswith("Safari标签页")


def test_attach_openchronicle_events_populates_browser_ax_context_json(tmp_path):
    cleaner = _cleaner(tmp_path)
    record_time = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)
    record = {
        "id": 1,
        "app": "Safari浏览器",
        "window": "Hermes 版本查询 — Hermes",
        "text": "Hermes 页面 OCR 文本",
        "cleaned_text": "Hermes 页面 OCR 文本",
        "timestamp_dt": record_time,
    }
    event = normalize_openchronicle_capture(
        {
            "id": "capture-safari-1",
            "timestamp": "2026-05-12T18:09:46+08:00",
            "app_name": "Safari",
            "bundle_id": "com.apple.Safari",
            "window_title": "Hermes 版本查询 — Hermes",
            "focused_role": "",
            "focused_value": "",
            "visible_text": "\n".join(
                [
                    "## Safari浏览器 [active]",
                    "_com.apple.Safari_",
                    "### Hermes 版本查询 — Hermes",
                    "    - [WebArea] Hermes 版本查询 — Hermes",
                    "        - Hermes 版本查询",
                    "        - [Button] Chat",
                    "        - [Button] Tasks",
                    "        - AI 智能体长效记忆架构与 MemoryLake 演进",
                ]
            ),
            "url": "",
        }
    )
    event["delta_seconds"] = 1.0
    event["match_reason"] = "same_app_nearby"

    cleaner.attach_openchronicle_events_to_records(
        [record],
        {1: [event]},
    )

    assert record["ax_window_title"] == "Hermes 版本查询 — Hermes"
    assert record["text_source"] == "ocr+ax_context"
    assert "AI 智能体长效记忆架构与 MemoryLake 演进" in record["ax_context_json"]
    assert "Safari标签页" in record["ax_context_json"]


def test_select_app_context_event_for_records_returns_per_record_primary_context():
    record_time = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)
    safari_event = normalize_openchronicle_capture(
        {
            "id": "capture-safari-1",
            "timestamp": "2026-06-01T10:00:01Z",
            "app_name": "Safari",
            "bundle_id": "com.apple.Safari",
            "window_title": "Hermes 版本查询 — Hermes",
            "focused_role": "",
            "focused_value": "",
            "visible_text": "\n".join(
                [
                    "## Safari浏览器 [active]",
                    "_com.apple.Safari_",
                    "### Hermes 版本查询 — Hermes",
                    "    - [WebArea] Hermes 版本查询 — Hermes",
                    "        - AI 智能体长效记忆架构与 MemoryLake 演进",
                ]
            ),
            "url": "",
        }
    )
    safari_event["delta_seconds"] = 2.0
    closer_safari_event = normalize_openchronicle_capture(
        {
            "id": "capture-safari-2",
            "timestamp": "2026-06-01T10:00:00Z",
            "app_name": "Safari",
            "bundle_id": "com.apple.Safari",
            "window_title": "Agent API Key Validation Issue — Hermes",
            "focused_role": "",
            "focused_value": "",
            "visible_text": "\n".join(
                [
                    "## Safari浏览器 [active]",
                    "_com.apple.Safari_",
                    "### Agent API Key Validation Issue — Hermes",
                    "    - [WebArea] Agent API Key Validation Issue — Hermes",
                ]
            ),
            "url": "",
        }
    )
    closer_safari_event["delta_seconds"] = 0.5
    second_browser_event = normalize_openchronicle_capture(
        {
            "id": "capture-safari-3",
            "timestamp": "2026-06-01T10:00:03Z",
            "app_name": "Safari",
            "bundle_id": "com.apple.Safari",
            "window_title": "AI Response Error — Hermes",
            "focused_role": "",
            "focused_value": "",
            "visible_text": "\n".join(
                [
                    "## Safari浏览器 [active]",
                    "_com.apple.Safari_",
                    "### AI Response Error — Hermes",
                    "    - [WebArea] AI Response Error — Hermes",
                ]
            ),
            "url": "",
        }
    )
    second_browser_event["delta_seconds"] = 1.0

    selected = select_app_context_event_for_records(
        [
            {
                "id": 1,
                "timestamp_dt": record_time,
                "ax_events": [safari_event, closer_safari_event],
            },
            {
                "id": 2,
                "timestamp_dt": record_time + timedelta(seconds=1),
                "ax_events": [second_browser_event],
            },
        ]
    )

    assert selected[1]["event"]["source_capture_id"] == "capture-safari-2"
    assert selected[1]["context"]["route"] == "merged_app_context"
    assert selected[1]["context"]["primary_context"]["page_title"] == "Agent API Key Validation Issue — Hermes"
    assert len(selected[1]["context"]["contexts"]) == 2
    assert selected[2]["event"]["source_capture_id"] == "capture-safari-3"
    assert selected[2]["context"]["route"] == "safari_browser_tab"


def test_infer_openchronicle_browser_actions_distinguishes_feishu_doc_reading_and_editing():
    reading_event = normalize_openchronicle_capture(
        {
            "id": "capture-edge-doc-read",
            "timestamp": "2026-06-01T10:00:00Z",
            "app_name": "Microsoft Edge",
            "bundle_id": "com.microsoft.edgemac",
            "window_title": "Aura 配件深度调研 - 飞书云文档 - Microsoft Edge",
            "focused_role": "AXWebArea",
            "focused_value": "",
            "visible_text": "\n".join(
                [
                    "## Microsoft Edge [active]",
                    "_com.microsoft.edgemac_",
                    "### Aura 配件深度调研 - 飞书云文档 - Microsoft Edge",
                    "    - [WebArea] Aura 配件深度调研",
                    "        - [Heading] 用户场景",
                    "        - 用户正在阅读 XREAL 配件相关的调研内容",
                ]
            ),
            "url": "",
        }
    )
    editing_event = normalize_openchronicle_capture(
        {
            "id": "capture-edge-doc-edit",
            "timestamp": "2026-06-01T10:00:20Z",
            "app_name": "Microsoft Edge",
            "bundle_id": "com.microsoft.edgemac",
            "window_title": "Aura 配件深度调研 - 飞书云文档 - Microsoft Edge",
            "focused_role": "AXTextArea",
            "focused_value": "新增一条关于镜腿配件的需求说明",
            "visible_text": "\n".join(
                [
                    "## Microsoft Edge [active]",
                    "_com.microsoft.edgemac_",
                    "### Aura 配件深度调研 - 飞书云文档 - Microsoft Edge",
                    "    - [WebArea] Aura 配件深度调研",
                    "        - [TextArea] 新增一条关于镜腿配件的需求说明",
                ]
            ),
            "url": "",
        }
    )

    infer_openchronicle_event_user_actions([reading_event, editing_event])

    reading_actions = {action["action_type"] for action in reading_event["user_actions"]}
    editing_actions = {action["action_type"] for action in editing_event["user_actions"]}
    assert "browser_document_reading" in reading_actions
    assert "browser_document_editing" in editing_actions


def test_summarize_view_uses_saved_record_app_context_without_ax_events():
    browser_context = {
        "source_app": "Safari",
        "surface": "browser-tab",
        "route": "safari_browser_tab",
        "page_title": "Hermes 版本查询 — Hermes",
        "url": "",
        "content_snippets": ["AI 智能体长效记忆架构与 MemoryLake 演进"],
        "structure_summary": "Safari标签页「Hermes 版本查询 — Hermes」，可见内容：AI 智能体长效记忆架构与 MemoryLake 演进。",
    }
    summary = summarize_view(
        [
            {
                "id": 1,
                "timestamp_dt": datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
                "timestamp": "2026-06-01T18:00:00+08:00",
                "app": "Safari浏览器",
                "window": "Hermes 版本查询 — Hermes",
                "view_window": "Hermes 版本查询 — Hermes",
                "focused": 1,
                "text": "Hermes OCR",
                "cleaned_text": "Hermes OCR",
                "ax_context_json": json.dumps(browser_context, ensure_ascii=False),
                "app_context": browser_context,
                "ocr_quality_score": 0.9,
                "content_kind": "other",
                "trigger": "test",
                "frame_id": 1,
            }
        ]
    )

    assert summary["window_title"] == "Hermes 版本查询 — Hermes"
    assert summary["content_kind"] == "browsing"


def test_build_view_representative_text_prefers_chat_app_context():
    chat_context = {
        "surface": "wechat-chat",
        "conversation_title": "张三",
        "message_lines": ["这是聊天增强文本"],
        "chat_text": "这是聊天增强文本",
    }
    representative_text = build_view_representative_text(
        [
            {
                "timestamp_dt": datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
                "ax_context_json": json.dumps(chat_context, ensure_ascii=False),
                "cleaned_text": "这是 OCR 文本",
                "text": "原始 OCR 文本",
            }
        ]
    )

    assert "[source=ax_context_json(chat)]" in representative_text
    assert "这是聊天增强文本" in representative_text
    assert "这是 OCR 文本" not in representative_text


def test_build_view_representative_text_combines_ax_context_and_cleaned_text():
    browser_context = {
        "surface": "browser-tab",
        "page_title": "页面标题",
        "content_snippets": ["这是浏览器增强文本"],
        "structure_summary": "Safari标签页「页面标题」，可见内容：这是浏览器增强文本。",
        "url": "",
    }
    representative_text = build_view_representative_text(
        [
            {
                "timestamp_dt": datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
                "ax_context_json": json.dumps(browser_context, ensure_ascii=False),
                "cleaned_text": "这是 OCR 文本",
                "text": "原始 OCR 文本",
            }
        ]
    )

    assert "[source=ax_context_json+cleaned_text]" in representative_text
    assert "这是浏览器增强文本" in representative_text
    assert "这是 OCR 文本" in representative_text


def test_build_view_representative_text_includes_all_record_app_contexts():
    merged_browser_context = {
        "surface": "merged-app-context",
        "route": "merged_app_context",
        "page_title": "Agent API Key Validation Issue — Hermes",
        "primary_context": {
            "surface": "browser-tab",
            "route": "safari_browser_tab",
            "page_title": "Agent API Key Validation Issue — Hermes",
            "content_snippets": ["第一段页面信息"],
            "structure_summary": "Safari标签页「Agent API Key Validation Issue — Hermes」，可见内容：第一段页面信息。",
            "url": "",
        },
        "contexts": [
            {
                "surface": "browser-tab",
                "route": "safari_browser_tab",
                "page_title": "Agent API Key Validation Issue — Hermes",
                "content_snippets": ["第一段页面信息"],
                "structure_summary": "Safari标签页「Agent API Key Validation Issue — Hermes」，可见内容：第一段页面信息。",
                "url": "",
            },
            {
                "surface": "browser-tab",
                "route": "safari_browser_tab",
                "page_title": "Hermes 版本查询 — Hermes",
                "content_snippets": ["第二段页面信息"],
                "structure_summary": "Safari标签页「Hermes 版本查询 — Hermes」，可见内容：第二段页面信息。",
                "url": "",
            },
        ],
        "context_count": 2,
    }
    representative_text = build_view_representative_text(
        [
            {
                "timestamp_dt": datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
                "ax_context_json": json.dumps(merged_browser_context, ensure_ascii=False),
                "cleaned_text": "这是 OCR 文本",
                "text": "原始 OCR 文本",
            }
        ]
    )

    assert "[source=ax_context_json+cleaned_text]" in representative_text
    assert "第一段页面信息" in representative_text
    assert "第二段页面信息" in representative_text
    assert "这是 OCR 文本" in representative_text


def test_build_view_representative_text_includes_user_action_priors():
    representative_text = build_view_representative_text(
        [
            {
                "timestamp_dt": datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
                "cleaned_text": "这是 OCR 文本",
                "text": "原始 OCR 文本",
                "user_actions_json": json.dumps(
                    [
                        {
                            "action_type": "browser_document_reading",
                            "confidence": 0.8,
                            "summary": "浏览器停留在飞书文档页面，推测用户正在阅读文档内容。",
                        }
                    ],
                    ensure_ascii=False,
                ),
            }
        ]
    )

    assert "[source=user_actions+cleaned_text]" in representative_text
    assert "动作先验" in representative_text
    assert "browser_document_reading" in representative_text
    assert "这是 OCR 文本" in representative_text
