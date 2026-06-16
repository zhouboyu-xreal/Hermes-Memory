import sqlite3
import sys
import threading
from datetime import datetime, timedelta, timezone

from agent.screen_memory.manager import ScreenMemoryManager
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
    assert config["schedule"]["fact_clustering_interval_hours"] == 2
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

    cluster_stats = cleaner.update_screen_fact_cluster_tables()

    assert cluster_stats["screen_facts_clustered"] == 0
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

    cluster_stats = cleaner.update_screen_fact_cluster_tables()

    assert cluster_stats["screen_facts_clustered"] == 5
    cluster_row = cursor.execute(
        """
        SELECT id, observation_id, observed_fact_count
        FROM screen_fact_clusters
        WHERE window_workstream_id = ?
        """,
        (workstream_id,),
    ).fetchone()
    assert cluster_row[1] is None
    assert cluster_row[2] == 0
    assert cursor.execute(
        "SELECT count(*) FROM screen_fact_cluster_members WHERE cluster_id = ?",
        (cluster_row[0],),
    ).fetchone()[0] == 5

    observation_stats = cleaner.update_screen_observation_tables()

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
    cleaner.update_screen_fact_cluster_tables()
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

    def _cluster(_cleaner, state, now):
        calls.append("cluster")
        state["last_fact_clustering_at"] = now.isoformat()
        return {}

    def _observe(_cleaner, state, now):
        calls.append("observation")
        state["last_observation_at"] = now.isoformat()
        return {}

    monkeypatch.setattr(service, "_run_fact_extraction", _ingest)
    monkeypatch.setattr(service, "_run_fact_clustering", _cluster)
    monkeypatch.setattr(service, "_run_observations", _observe)

    start = datetime(2026, 6, 1, tzinfo=timezone.utc)
    service.run_screen_memory_due_work(now=start)
    assert calls == ["ingest", "cluster", "observation"]

    calls.clear()
    service.run_screen_memory_due_work(
        now=datetime(2026, 6, 1, 1, tzinfo=timezone.utc)
    )
    assert calls == ["ingest"]

    calls.clear()
    service.run_screen_memory_due_work(
        now=datetime(2026, 6, 1, 6, tzinfo=timezone.utc)
    )
    assert calls == ["ingest", "cluster"]

    calls.clear()
    service.run_screen_memory_due_work(
        now=datetime(2026, 6, 2, tzinfo=timezone.utc)
    )
    assert calls == ["ingest", "cluster", "observation"]


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
