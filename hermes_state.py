#!/usr/bin/env python3
"""
SQLite State Store for Hermes Agent.

Provides persistent session storage with FTS5 full-text search, replacing
the per-session JSONL file approach. Stores session metadata, full message
history, and model configuration for CLI and gateway sessions.

Key design decisions:
- WAL mode for concurrent readers + one writer (gateway multi-platform)
- FTS5 virtual table for fast text search across all session messages
- Compression-triggered session splitting via parent_session_id chains
- Batch runner and RL trajectories are NOT stored here (separate systems)
- Session source tagging ('cli', 'telegram', 'discord', etc.) for filtering
"""

import json
import heapq
import logging
import math
import random
import re
import sqlite3
import threading
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import yaml

from agent.memory_manager import sanitize_context
from hermes_constants import get_hermes_home
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np
from hermes_cli.config import get_memory_config_path

# Optional FAISS — gracefully degrades to keyword-only search when unavailable
try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    faiss = None
    _HAS_FAISS = False

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_DB_PATH = get_hermes_home() / "state.db"

SCHEMA_VERSION = 13
def _get_embedding_dim_from_config() -> int:
    """Read embedding dimension from memory.yaml, falling back to 1536."""
    config_path = get_memory_config_path()
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return int(cfg.get("embedding", {}).get("dimensions", 1536))
    except Exception:
        return 1536

EMBEDDING_DIM = _get_embedding_dim_from_config()

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    api_call_count INTEGER DEFAULT 0,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT,
    codex_message_items TEXT
);

CREATE TABLE IF NOT EXISTS state_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
"""

FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
"""

# Trigram FTS5 table for CJK substring search.  The default unicode61
# tokenizer splits CJK characters into individual tokens, breaking phrase
# matching.  The trigram tokenizer creates overlapping 3-byte sequences so
# substring queries work natively for any script (CJK, Thai, etc.).
FTS_TRIGRAM_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(
    content,
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_update AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
"""

# ── Memory Nodes Schema (summarized event storage with vector search) ──

MEMORY_NODES_SQL = """
CREATE TABLE IF NOT EXISTS memory_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time_key TEXT UNIQUE NOT NULL,
    summary TEXT NOT NULL,
    keywords TEXT NOT NULL,
    topic TEXT NOT NULL,
    primary_entity_id INTEGER REFERENCES entity_nodes(id),
    primary_topic TEXT NOT NULL DEFAULT 'general',
    fact_type TEXT NOT NULL DEFAULT 'semantic',
    fact_subject TEXT NOT NULL DEFAULT 'other',
    fact_kind TEXT NOT NULL DEFAULT 'other',
    task_event_like INTEGER,
    task_event_subject TEXT,
    task_relevance TEXT,
    entity_names TEXT NOT NULL DEFAULT '[]',
    original_dialog TEXT,
    decay_score REAL DEFAULT 1.0,
    decay_updated_at TEXT,
    decay_half_life_days REAL
);
"""

MEMORY_NODES_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memory_nodes_fts USING fts5(
    summary,
    keywords,
    content='memory_nodes',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS memory_nodes_ai AFTER INSERT ON memory_nodes BEGIN
    INSERT INTO memory_nodes_fts(rowid, summary, keywords)
    VALUES (new.id, new.summary, new.keywords);
END;

CREATE TRIGGER IF NOT EXISTS memory_nodes_ad AFTER DELETE ON memory_nodes BEGIN
    DELETE FROM memory_nodes_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS memory_nodes_au AFTER UPDATE ON memory_nodes BEGIN
    UPDATE memory_nodes_fts
    SET summary = new.summary,
        keywords = new.keywords
    WHERE rowid = new.id;
END;
"""

# ── Knowledge Graph Schema (entity extraction + relation graph) ──
# Replicates HindSight's entity knowledge graph with lightweight SQLite storage.

KNOWLEDGE_GRAPH_SQL = """
CREATE TABLE IF NOT EXISTS entity_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL DEFAULT 'CONCEPT',
    embedding BLOB,
    metadata TEXT DEFAULT '{}',
    co_entities TEXT DEFAULT '{}',
    created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS entity_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_entity_id INTEGER NOT NULL REFERENCES entity_nodes(id),
    target_entity_id INTEGER NOT NULL REFERENCES entity_nodes(id),
    relation_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    metadata TEXT DEFAULT '{}',
    created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
    UNIQUE(source_entity_id, target_entity_id, relation_type)
);

CREATE TABLE IF NOT EXISTS memory_node_entities (
    node_id INTEGER NOT NULL REFERENCES memory_nodes(id),
    entity_id INTEGER NOT NULL REFERENCES entity_nodes(id),
    mention_count INTEGER DEFAULT 1,
    PRIMARY KEY (node_id, entity_id)
);

-- Normalized memory node relation table
CREATE TABLE IF NOT EXISTS memory_node_relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_node_id INTEGER NOT NULL REFERENCES memory_nodes(id),
    target_node_id INTEGER NOT NULL REFERENCES memory_nodes(id),
    relation_type TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    semantic_score REAL DEFAULT 0.0,
    causal_score REAL DEFAULT 0.0,
    temporal_score REAL DEFAULT 0.0,
    entity_score REAL DEFAULT 0.0,
    weight REAL DEFAULT 1.0,
    metadata TEXT DEFAULT '{}',
    created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
    UNIQUE(source_node_id, target_node_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_entity_edges_source ON entity_edges(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_edges_target ON entity_edges(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_edges_type ON entity_edges(relation_type);
CREATE INDEX IF NOT EXISTS idx_memory_node_entities_node ON memory_node_entities(node_id);
CREATE INDEX IF NOT EXISTS idx_memory_node_entities_entity ON memory_node_entities(entity_id);
CREATE INDEX IF NOT EXISTS idx_memory_node_relations_source ON memory_node_relations(source_node_id);
CREATE INDEX IF NOT EXISTS idx_memory_node_relations_target ON memory_node_relations(target_node_id);
"""

ENTITY_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS entity_nodes_fts USING fts5(
    name,
    type,
    content='entity_nodes',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS entity_nodes_ai AFTER INSERT ON entity_nodes BEGIN
    INSERT INTO entity_nodes_fts(rowid, name, type)
    VALUES (new.id, new.name, new.type);
END;

CREATE TRIGGER IF NOT EXISTS entity_nodes_ad AFTER DELETE ON entity_nodes BEGIN
    DELETE FROM entity_nodes_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS entity_nodes_au AFTER UPDATE ON entity_nodes BEGIN
    UPDATE entity_nodes_fts
    SET name = new.name,
        type = new.type
    WHERE rowid = new.id;
END;
"""

MEMORY_OBSERVATIONS_SQL = """
CREATE TABLE IF NOT EXISTS memory_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER NOT NULL REFERENCES entity_nodes(id),
    topic_key TEXT NOT NULL,
    topic_label TEXT NOT NULL,
    observation_type TEXT NOT NULL DEFAULT 'context',
    summary TEXT NOT NULL,
    keywords TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_supported_at TEXT,
    source_time_start TEXT,
    source_time_end TEXT,
    embedding BLOB,
    embedding_text TEXT NOT NULL DEFAULT '',
    embedding_updated_at TEXT,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS memory_observation_sources (
    observation_id INTEGER NOT NULL REFERENCES memory_observations(id),
    node_id INTEGER NOT NULL REFERENCES memory_nodes(id),
    role TEXT NOT NULL DEFAULT 'initial',
    confidence REAL DEFAULT 1.0,
    PRIMARY KEY (observation_id, node_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_observations_entity_topic
ON memory_observations(entity_id, topic_key, status);

CREATE INDEX IF NOT EXISTS idx_memory_observations_status
ON memory_observations(status);

CREATE INDEX IF NOT EXISTS idx_memory_observation_sources_node
ON memory_observation_sources(node_id);
"""

MEMORY_INTERPRETATIONS_SQL = """
CREATE TABLE IF NOT EXISTS memory_interpretations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER REFERENCES entity_nodes(id),
    subject_text TEXT NOT NULL DEFAULT '',
    target_text TEXT NOT NULL DEFAULT '',
    scope TEXT NOT NULL DEFAULT 'general',
    interpretation_type TEXT NOT NULL DEFAULT 'behavior_pattern',
    claim TEXT NOT NULL,
    polarity TEXT NOT NULL DEFAULT 'neutral',
    strength REAL DEFAULT 0.5,
    confidence REAL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'current',
    conflict_status TEXT NOT NULL DEFAULT 'none',
    resolution TEXT NOT NULL DEFAULT '',
    action_implication TEXT NOT NULL DEFAULT '',
    evidence_node_ids TEXT DEFAULT '[]',
    evidence_observation_ids TEXT DEFAULT '[]',
    counter_evidence_node_ids TEXT DEFAULT '[]',
    counter_evidence_observation_ids TEXT DEFAULT '[]',
    embedding BLOB,
    embedding_text TEXT NOT NULL DEFAULT '',
    embedding_updated_at TEXT,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_supported_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_memory_interpretations_status
ON memory_interpretations(status);

CREATE INDEX IF NOT EXISTS idx_memory_interpretations_entity
ON memory_interpretations(entity_id, status);

CREATE INDEX IF NOT EXISTS idx_memory_interpretations_type_scope
ON memory_interpretations(interpretation_type, scope, status);
"""


class SessionDB:
    """
    SQLite-backed session storage with FTS5 search.

    Thread-safe for the common gateway pattern (multiple reader threads,
    single writer via WAL mode). Each method opens its own cursor.
    """

    # ── Write-contention tuning ──
    # With multiple hermes processes (gateway + CLI sessions + worktree agents)
    # all sharing one state.db, WAL write-lock contention causes visible TUI
    # freezes.  SQLite's built-in busy handler uses a deterministic sleep
    # schedule that causes convoy effects under high concurrency.
    #
    # Instead, we keep the SQLite timeout short (1s) and handle retries at the
    # application level with random jitter, which naturally staggers competing
    # writers and avoids the convoy.
    _WRITE_MAX_RETRIES = 15
    _WRITE_RETRY_MIN_S = 0.020   # 20ms
    _WRITE_RETRY_MAX_S = 0.150   # 150ms
    # Attempt a PASSIVE WAL checkpoint every N successful writes.
    _CHECKPOINT_EVERY_N_WRITES = 20
    _MEMORY_VECTOR_FILTER_BRUTE_FORCE_LIMIT = 5000
    _MEMORY_OBSERVATION_FACT_HALF_LIFE_DAYS = 365.0
    _MEMORY_OBSERVATION_EXPERIENCE_HALF_LIFE_DAYS = 90.0
    _MEMORY_OBSERVATION_DECAY_THRESHOLD = 0.25
    _MEMORY_RECALL_DECAY_FLOOR = 0.25
    _MEMORY_TASK_PAUSED_IDLE_DAYS = 7.0
    _MEMORY_TASK_STALE_IDLE_DAYS = 30.0

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._write_count = 0
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            # Short timeout — application-level retry with random jitter
            # handles contention instead of sitting in SQLite's internal
            # busy handler for up to 30s.
            timeout=1.0,
            # Autocommit mode: Python's default isolation_level="" auto-starts
            # transactions on DML, which conflicts with our explicit
            # BEGIN IMMEDIATE.  None = we manage transactions ourselves.
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        # ── Memory Nodes: FAISS index for vector search ──
        self._memory_faiss_index = None
        self._memory_faiss_id_map: List[int] = []
        self._memory_faiss_save_path = self.db_path.with_suffix('.faiss')
        self._memory_faiss_ids_path = self.db_path.with_suffix('.faiss_ids.json')
        if _HAS_FAISS:
            try:
                self._memory_faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
                # Try to load persisted FAISS data from disk
                self._memory_load_faiss()
            except Exception as exc:
                logger.warning("Failed to initialize FAISS index: %s", exc)
                self._memory_faiss_index = None

        self._init_schema()

    # ── Core write helper ──

    def _execute_write(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Execute a write transaction with BEGIN IMMEDIATE and jitter retry.

        *fn* receives the connection and should perform INSERT/UPDATE/DELETE
        statements.  The caller must NOT call ``commit()`` — that's handled
        here after *fn* returns.

        BEGIN IMMEDIATE acquires the WAL write lock at transaction start
        (not at commit time), so lock contention surfaces immediately.
        On ``database is locked``, we release the Python lock, sleep a
        random 20-150ms, and retry — breaking the convoy pattern that
        SQLite's built-in deterministic backoff creates.

        Returns whatever *fn* returns.
        """
        last_err: Optional[Exception] = None
        for attempt in range(self._WRITE_MAX_RETRIES):
            try:
                with self._lock:
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        result = fn(self._conn)
                        self._conn.commit()
                    except BaseException:
                        try:
                            self._conn.rollback()
                        except Exception:
                            pass
                        raise
                # Success — periodic best-effort checkpoint.
                self._write_count += 1
                if self._write_count % self._CHECKPOINT_EVERY_N_WRITES == 0:
                    self._try_wal_checkpoint()
                return result
            except sqlite3.OperationalError as exc:
                err_msg = str(exc).lower()
                if "locked" in err_msg or "busy" in err_msg:
                    last_err = exc
                    if attempt < self._WRITE_MAX_RETRIES - 1:
                        jitter = random.uniform(
                            self._WRITE_RETRY_MIN_S,
                            self._WRITE_RETRY_MAX_S,
                        )
                        time.sleep(jitter)
                        continue
                # Non-lock error or retries exhausted — propagate.
                raise
        # Retries exhausted (shouldn't normally reach here).
        raise last_err or sqlite3.OperationalError(
            "database is locked after max retries"
        )

    def _try_wal_checkpoint(self) -> None:
        """Best-effort PASSIVE WAL checkpoint.  Never blocks, never raises.

        Flushes committed WAL frames back into the main DB file for any
        frames that no other connection currently needs.  Keeps the WAL
        from growing unbounded when many processes hold persistent
        connections.
        """
        try:
            with self._lock:
                result = self._conn.execute(
                    "PRAGMA wal_checkpoint(PASSIVE)"
                ).fetchone()
                if result and result[1] > 0:
                    logger.debug(
                        "WAL checkpoint: %d/%d pages checkpointed",
                        result[2], result[1],
                    )
        except Exception:
            pass  # Best effort — never fatal.

    def close(self):
        """Close the database connection and release FAISS resources.

        Attempts a PASSIVE WAL checkpoint first so that exiting processes
        help keep the WAL file from growing unbounded.
        """
        with self._lock:
            # Release FAISS index
            self._memory_faiss_index = None
            self._memory_faiss_id_map = []

            if self._conn:
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except Exception:
                    pass
                self._conn.close()
                self._conn = None

    @staticmethod
    def _parse_schema_columns(schema_sql: str) -> Dict[str, Dict[str, str]]:
        """Extract expected columns per table from SCHEMA_SQL.

        Uses an in-memory SQLite database to parse the SQL — SQLite itself
        handles all syntax (DEFAULT expressions with commas, inline
        REFERENCES, CHECK constraints, etc.) so there are zero regex
        edge cases.  The in-memory DB is opened, the schema DDL is
        executed, and PRAGMA table_info extracts the column metadata.

        Adding a column to SCHEMA_SQL is all that's needed; the
        reconciliation loop picks it up automatically.
        """
        ref = sqlite3.connect(":memory:")
        try:
            ref.executescript(schema_sql)
            table_columns: Dict[str, Dict[str, str]] = {}
            for (tbl,) in ref.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall():
                cols: Dict[str, str] = {}
                for row in ref.execute(
                    f'PRAGMA table_info("{tbl}")'
                ).fetchall():
                    # row: (cid, name, type, notnull, dflt_value, pk)
                    col_name = row[1]
                    col_type = row[2] or ""
                    notnull = row[3]
                    default = row[4]
                    pk = row[5]
                    # Reconstruct the type expression for ALTER TABLE ADD COLUMN
                    parts = [col_type] if col_type else []
                    if notnull and not pk:
                        parts.append("NOT NULL")
                    if default is not None:
                        parts.append(f"DEFAULT {default}")
                    cols[col_name] = " ".join(parts)
                table_columns[tbl] = cols
            return table_columns
        finally:
            ref.close()

    def _reconcile_columns(self, cursor: sqlite3.Cursor) -> None:
        """Ensure live tables have every column declared in SCHEMA_SQL.

        Follows the Beets/sqlite-utils pattern: the CREATE TABLE definition
        in SCHEMA_SQL is the single source of truth for the desired schema.
        On every startup this method diffs the live columns (via PRAGMA
        table_info) against the declared columns, and ADDs any that are
        missing.

        This makes column additions a declarative operation — just add
        the column to SCHEMA_SQL and it appears on the next startup.
        Version-gated migration blocks are no longer needed for ADD COLUMN.
        """
        expected = self._parse_schema_columns(SCHEMA_SQL)
        for table_name, declared_cols in expected.items():
            # Get current columns from the live table
            try:
                rows = cursor.execute(
                    f'PRAGMA table_info("{table_name}")'
                ).fetchall()
            except sqlite3.OperationalError:
                continue  # Table doesn't exist yet (shouldn't happen after executescript)
            live_cols = set()
            for row in rows:
                # PRAGMA table_info returns (cid, name, type, notnull, dflt_value, pk)
                name = row[1] if isinstance(row, (tuple, list)) else row["name"]
                live_cols.add(name)

            for col_name, col_type in declared_cols.items():
                if col_name not in live_cols:
                    safe_name = col_name.replace('"', '""')
                    try:
                        cursor.execute(
                            f'ALTER TABLE "{table_name}" ADD COLUMN "{safe_name}" {col_type}'
                        )
                    except sqlite3.OperationalError as exc:
                        # Expected: "duplicate column name" from a race or
                        # re-run.  Unexpected: "Cannot add a NOT NULL column
                        # with default value NULL" from a schema mistake.
                        # Log at DEBUG so it's visible in agent.log.
                        logger.debug(
                            "reconcile %s.%s: %s", table_name, col_name, exc,
                        )

    def _init_schema(self):
        """Create tables and FTS if they don't exist, reconcile columns.

        Schema management follows the declarative reconciliation pattern
        (Beets, sqlite-utils): SCHEMA_SQL is the single source of truth.
        On existing databases, _reconcile_columns() diffs live columns
        against SCHEMA_SQL and ADDs any missing ones.  This eliminates
        the version-gated migration chain for column additions, making
        it impossible for reordered or inserted migrations to skip columns.

        The schema_version table is retained for future data migrations
        (transforming existing rows) which cannot be handled declaratively.
        """
        cursor = self._conn.cursor()

        cursor.executescript(SCHEMA_SQL)

        # ── Declarative column reconciliation ──────────────────────────
        # Diff live tables against SCHEMA_SQL and ADD any missing columns.
        # This is idempotent and self-healing: even if a version-gated
        # migration was skipped (e.g. due to version renumbering), the
        # column gets created here.
        self._reconcile_columns(cursor)

        # ── Schema version bookkeeping ─────────────────────────────────
        # Bump to current so future data migrations (if any) can gate on
        # version.  No version-gated column additions remain.
        cursor.execute("SELECT version FROM schema_version LIMIT 1")
        row = cursor.fetchone()
        if row is None:
            cursor.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
        else:
            current_version = row["version"] if isinstance(row, sqlite3.Row) else row[0]
            # Data migrations that can't be expressed declaratively (row
            # backfills, index changes tied to a specific version step) stay
            # in a version-gated chain. Column additions are handled by
            # _reconcile_columns() above and no longer need entries here.
            if current_version < 10:
                # v10: trigram FTS5 table for CJK/substring search. The
                # virtual table + triggers are created unconditionally via
                # FTS_TRIGRAM_SQL below, but existing rows need a one-time
                # backfill into the FTS index.
                try:
                    cursor.execute("SELECT * FROM messages_fts_trigram LIMIT 0")
                    _fts_trigram_exists = True
                except sqlite3.OperationalError:
                    _fts_trigram_exists = False
                if not _fts_trigram_exists:
                    cursor.executescript(FTS_TRIGRAM_SQL)
                    cursor.execute(
                        "INSERT INTO messages_fts_trigram(rowid, content) "
                        "SELECT id, content FROM messages WHERE content IS NOT NULL"
                    )
            if current_version < 11:
                # v11: re-index FTS5 tables to cover tool_name + tool_calls and
                # switch from external-content to inline mode. Existing DBs have
                # old-schema FTS tables and triggers that IF NOT EXISTS won't
                # overwrite, so we drop them explicitly and let the post-migration
                # existence checks (below) recreate them from FTS_SQL /
                # FTS_TRIGRAM_SQL, then backfill every message row. Fixes #16751.
                for _trig in (
                    "messages_fts_insert",
                    "messages_fts_delete",
                    "messages_fts_update",
                    "messages_fts_trigram_insert",
                    "messages_fts_trigram_delete",
                    "messages_fts_trigram_update",
                ):
                    try:
                        cursor.execute(f"DROP TRIGGER IF EXISTS {_trig}")
                    except sqlite3.OperationalError:
                        pass
                for _tbl in ("messages_fts", "messages_fts_trigram"):
                    try:
                        cursor.execute(f"DROP TABLE IF EXISTS {_tbl}")
                    except sqlite3.OperationalError:
                        pass
                # Recreate virtual tables + triggers with the new inline-mode
                # schema that indexes content || tool_name || tool_calls.
                cursor.executescript(FTS_SQL)
                cursor.executescript(FTS_TRIGRAM_SQL)
                # Backfill both indexes from every existing messages row.
                cursor.execute(
                    "INSERT INTO messages_fts(rowid, content) "
                    "SELECT id, "
                    "COALESCE(content, '') || ' ' || "
                    "COALESCE(tool_name, '') || ' ' || "
                    "COALESCE(tool_calls, '') "
                    "FROM messages"
                )
                cursor.execute(
                    "INSERT INTO messages_fts_trigram(rowid, content) "
                    "SELECT id, "
                    "COALESCE(content, '') || ' ' || "
                    "COALESCE(tool_name, '') || ' ' || "
                    "COALESCE(tool_calls, '') "
                    "FROM messages"
                )
            if current_version < 12:
                # v12: fix knowledge-graph created_at defaults. The original
                # schema used strftime('%%s','now'), which SQLite stores as the
                # literal string "%s" instead of a Unix timestamp.
                self._repair_graph_created_at_placeholders(cursor)
            if current_version < SCHEMA_VERSION:
                cursor.execute(
                    "UPDATE schema_version SET version = ?",
                    (SCHEMA_VERSION,),
                )

        # Unique title index — always ensure it exists
        try:
            cursor.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique "
                "ON sessions(title) WHERE title IS NOT NULL"
            )
        except sqlite3.OperationalError:
            pass  # Index already exists

        # FTS5 setup (separate because CREATE VIRTUAL TABLE can't be in executescript with IF NOT EXISTS reliably)
        try:
            cursor.execute("SELECT * FROM messages_fts LIMIT 0")
        except sqlite3.OperationalError:
            cursor.executescript(FTS_SQL)

        # Trigram FTS5 for CJK/substring search
        try:
            cursor.execute("SELECT * FROM messages_fts_trigram LIMIT 0")
        except sqlite3.OperationalError:
            cursor.executescript(FTS_TRIGRAM_SQL)

        # ── Memory Nodes tables ──
        cursor.executescript(MEMORY_NODES_SQL)

        # Rebuild memory-node FTS after all memory_nodes migrations below.
        # Existing triggers would fire while we move legacy detail columns onto
        # memory_nodes, so drop them first and recreate/backfill at the end.
        for _trig in ("memory_nodes_ai", "memory_nodes_ad", "memory_nodes_au"):
            try:
                cursor.execute(f"DROP TRIGGER IF EXISTS {_trig}")
            except sqlite3.OperationalError:
                pass
        try:
            cursor.execute("DROP TABLE IF EXISTS memory_nodes_fts")
        except sqlite3.OperationalError:
            pass

        # ── Knowledge Graph tables + FTS5 ──
        cursor.executescript(KNOWLEDGE_GRAPH_SQL)
        for col_name, col_type in {
            "co_entities": "TEXT DEFAULT '{}'",
        }.items():
            try:
                cursor.execute(
                    f"ALTER TABLE entity_nodes ADD COLUMN {col_name} {col_type}"
                )
            except sqlite3.OperationalError:
                pass
        for col_name, col_type in {
            "semantic_score": "REAL DEFAULT 0.0",
            "causal_score": "REAL DEFAULT 0.0",
            "temporal_score": "REAL DEFAULT 0.0",
            "entity_score": "REAL DEFAULT 0.0",
            "weight": "REAL DEFAULT 1.0",
            "metadata": "TEXT DEFAULT '{}'",
        }.items():
            try:
                cursor.execute(
                    f"ALTER TABLE memory_node_relations ADD COLUMN {col_name} {col_type}"
                )
            except sqlite3.OperationalError:
                pass
        try:
            cursor.execute("SELECT * FROM entity_nodes_fts LIMIT 0")
        except sqlite3.OperationalError:
            cursor.executescript(ENTITY_FTS_SQL)
        cursor.executescript(MEMORY_OBSERVATIONS_SQL)
        self._drop_legacy_memory_entity_columns(cursor)
        cursor.executescript(MEMORY_INTERPRETATIONS_SQL)
        for table_name in ("memory_observations", "memory_interpretations"):
            for col_name, col_type in {
                "embedding": "BLOB",
                "embedding_text": "TEXT NOT NULL DEFAULT ''",
                "embedding_updated_at": "TEXT",
            }.items():
                try:
                    cursor.execute(
                        f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"
                    )
                except sqlite3.OperationalError:
                    pass
        self._migrate_memory_opinions_to_interpretations(cursor)
        self._repair_graph_created_at_placeholders(cursor)

        # ── Add tags column to memory_nodes if missing ──
        try:
            cursor.execute("ALTER TABLE memory_nodes ADD COLUMN tags TEXT DEFAULT '[]'")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # ── Add explicit fact_type column to memory_nodes if missing ──
        try:
            cursor.execute("ALTER TABLE memory_nodes ADD COLUMN fact_type TEXT NOT NULL DEFAULT 'semantic'")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # ── Add explicit fact_subject column to memory_nodes if missing ──
        try:
            cursor.execute("ALTER TABLE memory_nodes ADD COLUMN fact_subject TEXT NOT NULL DEFAULT 'other'")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # ── Add explicit fact_kind column to memory_nodes if missing ──
        try:
            cursor.execute("ALTER TABLE memory_nodes ADD COLUMN fact_kind TEXT NOT NULL DEFAULT 'other'")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # ── Memory node detail fields live directly on memory_nodes ──
        try:
            cursor.execute("ALTER TABLE memory_nodes ADD COLUMN original_dialog TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists

        for col_name, col_type in {
            "task_event_like": "INTEGER",
            "task_event_subject": "TEXT",
            "task_relevance": "TEXT",
            "entity_names": "TEXT NOT NULL DEFAULT '[]'",
            "primary_entity_id": "INTEGER REFERENCES entity_nodes(id)",
            "primary_topic": "TEXT NOT NULL DEFAULT 'general'",
            "decay_score": "REAL DEFAULT 1.0",
            "decay_updated_at": "TEXT",
            "decay_half_life_days": "REAL",
        }.items():
            try:
                cursor.execute(
                    f"ALTER TABLE memory_nodes ADD COLUMN {col_name} {col_type}"
                )
            except sqlite3.OperationalError:
                pass
        cursor.execute(
            "UPDATE memory_nodes SET primary_topic = "
            "CASE WHEN instr(trim(topic), ' ') > 0 "
            "THEN substr(trim(topic), 1, instr(trim(topic), ' ') - 1) "
            "ELSE COALESCE(NULLIF(trim(topic), ''), 'general') END "
            "WHERE primary_topic IS NULL OR primary_topic = '' OR primary_topic = 'general'"
        )
        cursor.execute(
            "UPDATE memory_nodes SET primary_entity_id = ("
            "  SELECT MIN(mne.entity_id) FROM memory_node_entities mne "
            "  WHERE mne.node_id = memory_nodes.id"
            ") WHERE primary_entity_id IS NULL"
        )

        # Backfill fact_type from legacy tags for existing retain facts.
        try:
            cursor.execute(
                "UPDATE memory_nodes SET fact_type = 'episodic' "
                "WHERE tags LIKE ?",
                ('%"fact_type:experience"%',),
            )
            cursor.execute(
                "UPDATE memory_nodes SET fact_type = 'episodic' "
                "WHERE fact_type = 'experience'"
            )
            cursor.execute(
                "UPDATE memory_nodes SET fact_type = 'semantic' "
                "WHERE fact_type IS NULL OR fact_type = '' OR fact_type NOT IN ('semantic', 'episodic')"
            )
        except sqlite3.OperationalError:
            pass

        # Backfill fact_subject from tags / legacy fact_type when possible.
        try:
            for fact_subject in ("user", "assistant", "world", "project", "system", "other"):
                cursor.execute(
                    "UPDATE memory_nodes SET fact_subject = ? WHERE tags LIKE ?",
                    (fact_subject, f'%"fact_subject:{fact_subject}"%'),
                )
            cursor.execute(
                "UPDATE memory_nodes SET fact_subject = 'assistant' "
                "WHERE fact_subject = 'other' AND tags LIKE ?",
                ('%"fact_type:experience"%',),
            )
            cursor.execute(
                "UPDATE memory_nodes SET fact_subject = 'other' "
                "WHERE fact_subject IS NULL OR fact_subject = '' "
                "OR fact_subject NOT IN ('user', 'assistant', 'world', 'project', 'system', 'other')"
            )
        except sqlite3.OperationalError:
            pass

        # Backfill fact_kind from legacy tags for existing retain facts.
        try:
            for fact_kind in (
                "preference", "decision", "request", "recommendation",
                "action", "error", "context", "instruction",
                "conversation_summary", "other",
            ):
                cursor.execute(
                    "UPDATE memory_nodes SET fact_kind = ? WHERE tags LIKE ?",
                    (fact_kind, f'%"fact_kind:{fact_kind}"%'),
                )
            cursor.execute(
                "UPDATE memory_nodes SET fact_kind = 'other' "
                "WHERE fact_kind IS NULL OR fact_kind = ''"
            )
        except sqlite3.OperationalError:
            pass

        # ── Index on time_key for time-range search ──
        try:
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_nodes_time "
                "ON memory_nodes(time_key)"
            )
        except sqlite3.OperationalError:
            pass

        # ── Index on fact_type for typed memory recall ──
        try:
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_nodes_fact_type "
                "ON memory_nodes(fact_type)"
            )
        except sqlite3.OperationalError:
            pass

        # ── Index on fact_kind for future kind-aware recall / observation ──
        try:
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_nodes_fact_kind "
                "ON memory_nodes(fact_kind)"
            )
        except sqlite3.OperationalError:
            pass

        # ── Index on fact_subject for actor/source-aware recall ──
        try:
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_nodes_fact_subject "
                "ON memory_nodes(fact_subject)"
            )
        except sqlite3.OperationalError:
            pass

        # ── Memory Nodes FTS5 ──
        try:
            cursor.executescript(MEMORY_NODES_FTS_SQL)
            cursor.execute(
                "INSERT INTO memory_nodes_fts(rowid, summary, keywords) "
                "SELECT id, summary, keywords FROM memory_nodes"
            )
        except sqlite3.OperationalError as _fts_err:
            logger.debug("Memory node FTS setup skipped: %s", _fts_err)

        self._conn.commit()

    def _repair_graph_created_at_placeholders(self, cursor: sqlite3.Cursor) -> None:
        """Repair graph rows that inherited the legacy literal '%s' default."""
        try:
            cursor.execute(
                """UPDATE entity_nodes
                   SET created_at = COALESCE(
                       (
                           SELECT MIN(CAST(strftime('%s', substr(mn.time_key, 1, 19)) AS REAL))
                           FROM memory_node_entities mne
                           JOIN memory_nodes mn ON mn.id = mne.node_id
                           WHERE mne.entity_id = entity_nodes.id
                       ),
                       CAST(strftime('%s','now') AS REAL)
                   )
                   WHERE created_at = '%s'"""
            )
            cursor.execute(
                """UPDATE entity_edges
                   SET created_at = CAST(strftime('%s','now') AS REAL)
                   WHERE created_at = '%s'"""
            )
            cursor.execute(
                """UPDATE memory_node_relations
                   SET created_at = COALESCE(
                       (
                           SELECT CAST(strftime('%s', substr(mn.time_key, 1, 19)) AS REAL)
                           FROM memory_nodes mn
                           WHERE mn.id = memory_node_relations.source_node_id
                       ),
                       CAST(strftime('%s','now') AS REAL)
                   )
                   WHERE created_at = '%s'"""
            )
        except sqlite3.OperationalError:
            pass

    @staticmethod
    def _table_columns(cursor: sqlite3.Cursor, table_name: str) -> List[str]:
        try:
            rows = cursor.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            row["name"] if isinstance(row, sqlite3.Row) else row[1]
            for row in rows
        ]

    @staticmethod
    def _coerce_int_or_none(value: Any) -> Optional[int]:
        try:
            if value is not None and str(value).strip():
                return int(value)
        except (TypeError, ValueError):
            return None
        return None

    def _drop_legacy_memory_entity_columns(self, cursor: sqlite3.Cursor) -> None:
        """Remove old subject/target entity id columns; no legacy data backfill."""
        interpretation_columns = set(self._table_columns(cursor, "memory_interpretations"))
        if interpretation_columns and "entity_id" not in interpretation_columns:
            try:
                cursor.execute("ALTER TABLE memory_interpretations ADD COLUMN entity_id INTEGER REFERENCES entity_nodes(id)")
            except sqlite3.OperationalError as exc:
                logger.debug("add memory_interpretations.entity_id skipped: %s", exc)

        for index_name in (
            "idx_memory_interpretations_subject",
            "idx_memory_interpretations_target",
        ):
            try:
                cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
            except sqlite3.OperationalError:
                pass

        for table_name in ("memory_observations", "memory_interpretations"):
            columns = set(self._table_columns(cursor, table_name))
            for column in ("subject_entity_id", "target_entity_id"):
                if column not in columns:
                    continue
                try:
                    cursor.execute(f"ALTER TABLE {table_name} DROP COLUMN {column}")
                except sqlite3.OperationalError as exc:
                    logger.debug("drop legacy %s.%s skipped: %s", table_name, column, exc)

    def _migrate_memory_opinions_to_interpretations(self, cursor: sqlite3.Cursor) -> None:
        """Copy rows from the short-lived opinion table name to interpretations."""
        try:
            has_opinions = cursor.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_opinions'"
            ).fetchone()
            if not has_opinions:
                return
            opinion_columns = set(self._table_columns(cursor, "memory_opinions"))
            entity_expr = "entity_id" if "entity_id" in opinion_columns else "NULL"
            cursor.execute(
                "INSERT INTO memory_interpretations "
                "(id, entity_id, subject_text, target_text, "
                "scope, interpretation_type, claim, polarity, strength, confidence, "
                "status, conflict_status, resolution, action_implication, "
                "evidence_node_ids, evidence_observation_ids, "
                "counter_evidence_node_ids, counter_evidence_observation_ids, "
                "metadata, created_at, updated_at, last_supported_at) "
                f"SELECT id, {entity_expr}, subject_text, target_text, "
                "scope, interpretation_type, claim, polarity, strength, confidence, "
                "status, conflict_status, resolution, action_implication, "
                "evidence_node_ids, evidence_observation_ids, "
                "counter_evidence_node_ids, counter_evidence_observation_ids, "
                "metadata, created_at, updated_at, last_supported_at "
                "FROM memory_opinions "
                "WHERE id NOT IN (SELECT id FROM memory_interpretations)"
            )
        except sqlite3.OperationalError:
            pass

    # =========================================================================
    # Session lifecycle
    # =========================================================================

    def create_session(
        self,
        session_id: str,
        source: str,
        model: str = None,
        model_config: Dict[str, Any] = None,
        system_prompt: str = None,
        user_id: str = None,
        parent_session_id: str = None,
    ) -> str:
        """Create a new session record. Returns the session_id."""
        def _do(conn):
            conn.execute(
                """INSERT OR IGNORE INTO sessions (id, source, user_id, model, model_config,
                   system_prompt, parent_session_id, started_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    source,
                    user_id,
                    model,
                    json.dumps(model_config) if model_config else None,
                    system_prompt,
                    parent_session_id,
                    time.time(),
                ),
            )
        self._execute_write(_do)
        return session_id

    def end_session(self, session_id: str, end_reason: str) -> None:
        """Mark a session as ended.

        No-ops when the session is already ended. The first end_reason wins:
        compression-split sessions must keep their ``end_reason = 'compression'``
        record even if a later stale ``end_session()`` call (e.g. from a
        desynced CLI session_id after ``/resume`` or ``/branch``) targets them
        with a different reason. Use ``reopen_session()`` first if you
        intentionally need to re-end a closed session with a new reason.
        """
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? "
                "WHERE id = ? AND ended_at IS NULL",
                (time.time(), end_reason, session_id),
            )
        self._execute_write(_do)

    def reopen_session(self, session_id: str) -> None:
        """Clear ended_at/end_reason so a session can be resumed."""
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?",
                (session_id,),
            )
        self._execute_write(_do)

    def update_system_prompt(self, session_id: str, system_prompt: str) -> None:
        """Store the full assembled system prompt snapshot."""
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET system_prompt = ? WHERE id = ?",
                (system_prompt, session_id),
            )
        self._execute_write(_do)

    def update_token_counts(
        self,
        session_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = None,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None,
        actual_cost_usd: Optional[float] = None,
        cost_status: Optional[str] = None,
        cost_source: Optional[str] = None,
        pricing_version: Optional[str] = None,
        billing_provider: Optional[str] = None,
        billing_base_url: Optional[str] = None,
        billing_mode: Optional[str] = None,
        api_call_count: int = 0,
        absolute: bool = False,
    ) -> None:
        """Update token counters and backfill model if not already set.

        When *absolute* is False (default), values are **incremented** — use
        this for per-API-call deltas (CLI path).

        When *absolute* is True, values are **set directly** — use this when
        the caller already holds cumulative totals (gateway path, where the
        cached agent accumulates across messages).
        """
        if absolute:
            sql = """UPDATE sessions SET
                   input_tokens = ?,
                   output_tokens = ?,
                   cache_read_tokens = ?,
                   cache_write_tokens = ?,
                   reasoning_tokens = ?,
                   estimated_cost_usd = COALESCE(?, 0),
                   actual_cost_usd = CASE
                       WHEN ? IS NULL THEN actual_cost_usd
                       ELSE ?
                   END,
                   cost_status = COALESCE(?, cost_status),
                   cost_source = COALESCE(?, cost_source),
                   pricing_version = COALESCE(?, pricing_version),
                   billing_provider = COALESCE(billing_provider, ?),
                   billing_base_url = COALESCE(billing_base_url, ?),
                   billing_mode = COALESCE(billing_mode, ?),
                   model = COALESCE(model, ?),
                   api_call_count = ?
                   WHERE id = ?"""
        else:
            sql = """UPDATE sessions SET
                   input_tokens = input_tokens + ?,
                   output_tokens = output_tokens + ?,
                   cache_read_tokens = cache_read_tokens + ?,
                   cache_write_tokens = cache_write_tokens + ?,
                   reasoning_tokens = reasoning_tokens + ?,
                   estimated_cost_usd = COALESCE(estimated_cost_usd, 0) + COALESCE(?, 0),
                   actual_cost_usd = CASE
                       WHEN ? IS NULL THEN actual_cost_usd
                       ELSE COALESCE(actual_cost_usd, 0) + ?
                   END,
                   cost_status = COALESCE(?, cost_status),
                   cost_source = COALESCE(?, cost_source),
                   pricing_version = COALESCE(?, pricing_version),
                   billing_provider = COALESCE(billing_provider, ?),
                   billing_base_url = COALESCE(billing_base_url, ?),
                   billing_mode = COALESCE(billing_mode, ?),
                   model = COALESCE(model, ?),
                   api_call_count = COALESCE(api_call_count, 0) + ?
                   WHERE id = ?"""
        params = (
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_tokens,
            reasoning_tokens,
            estimated_cost_usd,
            actual_cost_usd,
            actual_cost_usd,
            cost_status,
            cost_source,
            pricing_version,
            billing_provider,
            billing_base_url,
            billing_mode,
            model,
            api_call_count,
            session_id,
        )
        def _do(conn):
            conn.execute(sql, params)
        self._execute_write(_do)

    def ensure_session(
        self,
        session_id: str,
        source: str = "unknown",
        model: str = None,
    ) -> None:
        """Ensure a session row exists, creating it with minimal metadata if absent.

        Used by _flush_messages_to_session_db to recover from a failed
        create_session() call (e.g. transient SQLite lock at agent startup).
        INSERT OR IGNORE is safe to call even when the row already exists.
        """
        def _do(conn):
            conn.execute(
                """INSERT OR IGNORE INTO sessions
                   (id, source, model, started_at)
                   VALUES (?, ?, ?, ?)""",
                (session_id, source, model, time.time()),
            )
        self._execute_write(_do)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            )
            row = cursor.fetchone()
        return dict(row) if row else None

    def resolve_session_id(self, session_id_or_prefix: str) -> Optional[str]:
        """Resolve an exact or uniquely prefixed session ID to the full ID.

        Returns the exact ID when it exists. Otherwise treats the input as a
        prefix and returns the single matching session ID if the prefix is
        unambiguous. Returns None for no matches or ambiguous prefixes.
        """
        exact = self.get_session(session_id_or_prefix)
        if exact:
            return exact["id"]

        escaped = (
            session_id_or_prefix
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id FROM sessions WHERE id LIKE ? ESCAPE '\\' ORDER BY started_at DESC LIMIT 2",
                (f"{escaped}%",),
            )
            matches = [row["id"] for row in cursor.fetchall()]
        if len(matches) == 1:
            return matches[0]
        return None

    # Maximum length for session titles
    MAX_TITLE_LENGTH = 100

    @staticmethod
    def sanitize_title(title: Optional[str]) -> Optional[str]:
        """Validate and sanitize a session title.

        - Strips leading/trailing whitespace
        - Removes ASCII control characters (0x00-0x1F, 0x7F) and problematic
          Unicode control chars (zero-width, RTL/LTR overrides, etc.)
        - Collapses internal whitespace runs to single spaces
        - Normalizes empty/whitespace-only strings to None
        - Enforces MAX_TITLE_LENGTH

        Returns the cleaned title string or None.
        Raises ValueError if the title exceeds MAX_TITLE_LENGTH after cleaning.
        """
        if not title:
            return None

        # Remove ASCII control characters (0x00-0x1F, 0x7F) but keep
        # whitespace chars (\t=0x09, \n=0x0A, \r=0x0D) so they can be
        # normalized to spaces by the whitespace collapsing step below
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', title)

        # Remove problematic Unicode control characters:
        # - Zero-width chars (U+200B-U+200F, U+FEFF)
        # - Directional overrides (U+202A-U+202E, U+2066-U+2069)
        # - Object replacement (U+FFFC), interlinear annotation (U+FFF9-U+FFFB)
        cleaned = re.sub(
            r'[\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff\ufffc\ufff9-\ufffb]',
            '', cleaned,
        )

        # Collapse internal whitespace runs and strip
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        if not cleaned:
            return None

        if len(cleaned) > SessionDB.MAX_TITLE_LENGTH:
            raise ValueError(
                f"Title too long ({len(cleaned)} chars, max {SessionDB.MAX_TITLE_LENGTH})"
            )

        return cleaned

    def set_session_title(self, session_id: str, title: str) -> bool:
        """Set or update a session's title.

        Returns True if session was found and title was set.
        Raises ValueError if title is already in use by another session,
        or if the title fails validation (too long, invalid characters).
        Empty/whitespace-only strings are normalized to None (clearing the title).
        """
        title = self.sanitize_title(title)
        def _do(conn):
            if title:
                # Check uniqueness (allow the same session to keep its own title)
                cursor = conn.execute(
                    "SELECT id FROM sessions WHERE title = ? AND id != ?",
                    (title, session_id),
                )
                conflict = cursor.fetchone()
                if conflict:
                    raise ValueError(
                        f"Title '{title}' is already in use by session {conflict['id']}"
                    )
            cursor = conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (title, session_id),
            )
            return cursor.rowcount
        rowcount = self._execute_write(_do)
        return rowcount > 0

    def get_session_title(self, session_id: str) -> Optional[str]:
        """Get the title for a session, or None."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)
            )
            row = cursor.fetchone()
        return row["title"] if row else None

    def get_session_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Look up a session by exact title. Returns session dict or None."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM sessions WHERE title = ?", (title,)
            )
            row = cursor.fetchone()
        return dict(row) if row else None

    def resolve_session_by_title(self, title: str) -> Optional[str]:
        """Resolve a title to a session ID, preferring the latest in a lineage.

        If the exact title exists, returns that session's ID.
        If not, searches for "title #N" variants and returns the latest one.
        If the exact title exists AND numbered variants exist, returns the
        latest numbered variant (the most recent continuation).
        """
        # First try exact match
        exact = self.get_session_by_title(title)

        # Also search for numbered variants: "title #2", "title #3", etc.
        # Escape SQL LIKE wildcards (%, _) in the title to prevent false matches
        escaped = title.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, title, started_at FROM sessions "
                "WHERE title LIKE ? ESCAPE '\\' ORDER BY started_at DESC",
                (f"{escaped} #%",),
            )
            numbered = cursor.fetchall()

        if numbered:
            # Return the most recent numbered variant
            return numbered[0]["id"]
        elif exact:
            return exact["id"]
        return None

    def get_next_title_in_lineage(self, base_title: str) -> str:
        """Generate the next title in a lineage (e.g., "my session" → "my session #2").

        Strips any existing " #N" suffix to find the base name, then finds
        the highest existing number and increments.
        """
        # Strip existing #N suffix to find the true base
        match = re.match(r'^(.*?) #(\d+)$', base_title)
        if match:
            base = match.group(1)
        else:
            base = base_title

        # Find all existing numbered variants
        # Escape SQL LIKE wildcards (%, _) in the base to prevent false matches
        escaped = base.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._lock:
            cursor = self._conn.execute(
                "SELECT title FROM sessions WHERE title = ? OR title LIKE ? ESCAPE '\\'",
                (base, f"{escaped} #%"),
            )
            existing = [row["title"] for row in cursor.fetchall()]

        if not existing:
            return base  # No conflict, use the base name as-is

        # Find the highest number
        max_num = 1  # The unnumbered original counts as #1
        for t in existing:
            m = re.match(r'^.* #(\d+)$', t)
            if m:
                max_num = max(max_num, int(m.group(1)))

        return f"{base} #{max_num + 1}"

    def get_compression_tip(self, session_id: str) -> Optional[str]:
        """Walk the compression-continuation chain forward and return the tip.

        A compression continuation is a child session where:
        1. The parent's ``end_reason = 'compression'``
        2. The child was created AFTER the parent was ended (started_at >= ended_at)

        The second condition distinguishes compression continuations from
        delegate subagents or branch children, which can also have a
        ``parent_session_id`` but were created while the parent was still live.

        Returns the session_id of the latest continuation in the chain, or the
        input ``session_id`` if it isn't part of a compression chain (or if the
        input itself doesn't exist).
        """
        current = session_id
        # Bound the walk defensively — compression chains this deep are
        # pathological and shouldn't happen in practice. 100 = plenty.
        for _ in range(100):
            with self._lock:
                cursor = self._conn.execute(
                    "SELECT id FROM sessions "
                    "WHERE parent_session_id = ? "
                    "  AND started_at >= ("
                    "      SELECT ended_at FROM sessions "
                    "      WHERE id = ? AND end_reason = 'compression'"
                    "  ) "
                    "ORDER BY started_at DESC LIMIT 1",
                    (current, current),
                )
                row = cursor.fetchone()
            if row is None:
                return current
            current = row["id"]
        return current

    def list_sessions_rich(
        self,
        source: str = None,
        exclude_sources: List[str] = None,
        limit: int = 20,
        offset: int = 0,
        include_children: bool = False,
        project_compression_tips: bool = True,
    ) -> List[Dict[str, Any]]:
        """List sessions with preview (first user message) and last active timestamp.

        Returns dicts with keys: id, source, model, title, started_at, ended_at,
        message_count, preview (first 60 chars of first user message),
        last_active (timestamp of last message).

        Uses a single query with correlated subqueries instead of N+2 queries.

        By default, child sessions (subagent runs, compression continuations)
        are excluded.  Pass ``include_children=True`` to include them.

        With ``project_compression_tips=True`` (default), sessions that are
        roots of compression chains are projected forward to their latest
        continuation — one logical conversation = one list entry, showing the
        live continuation's id/message_count/title/last_active. This prevents
        compressed continuations from being invisible to users while keeping
        delegate subagents and branches hidden. Pass ``False`` to return the
        raw root rows (useful for admin/debug UIs).
        """
        where_clauses = []
        params = []

        if not include_children:
            # Show root sessions and branch sessions (whose parent ended with
            # end_reason='branched' before the child was created), while still
            # hiding sub-agent runs and compression continuations (which also
            # carry a parent_session_id but were spawned while the parent was
            # still live — i.e., started_at < parent.ended_at).
            where_clauses.append(
                "(s.parent_session_id IS NULL"
                " OR EXISTS (SELECT 1 FROM sessions p"
                "            WHERE p.id = s.parent_session_id"
                "            AND p.end_reason = 'branched'"
                "            AND s.started_at >= p.ended_at))"
            )

        if source:
            where_clauses.append("s.source = ?")
            params.append(source)
        if exclude_sources:
            placeholders = ",".join("?" for _ in exclude_sources)
            where_clauses.append(f"s.source NOT IN ({placeholders})")
            params.extend(exclude_sources)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        query = f"""
            SELECT s.*,
                COALESCE(
                    (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)
                     FROM messages m
                     WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                     ORDER BY m.timestamp, m.id LIMIT 1),
                    ''
                ) AS _preview_raw,
                COALESCE(
                    (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                    s.started_at
                ) AS last_active
            FROM sessions s
            {where_sql}
            ORDER BY s.started_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        with self._lock:
            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()
        sessions = []
        for row in rows:
            s = dict(row)
            # Build the preview from the raw substring
            raw = s.pop("_preview_raw", "").strip()
            if raw:
                text = raw[:60]
                s["preview"] = text + ("..." if len(raw) > 60 else "")
            else:
                s["preview"] = ""
            sessions.append(s)

        # Project compression roots forward to their tips. Each row whose
        # end_reason is 'compression' has a continuation child; replace the
        # surfaced fields (id, message_count, title, last_active, ended_at,
        # end_reason, preview) with the tip's values so the list entry acts
        # as the live conversation. Keep the root's started_at to preserve
        # chronological ordering by original conversation start.
        if project_compression_tips and not include_children:
            projected = []
            for s in sessions:
                if s.get("end_reason") != "compression":
                    projected.append(s)
                    continue
                tip_id = self.get_compression_tip(s["id"])
                if tip_id == s["id"]:
                    projected.append(s)
                    continue
                tip_row = self._get_session_rich_row(tip_id)
                if not tip_row:
                    projected.append(s)
                    continue
                # Preserve the root's started_at for stable sort order, but
                # surface the tip's identity and activity data.
                merged = dict(s)
                for key in (
                    "id", "ended_at", "end_reason", "message_count",
                    "tool_call_count", "title", "last_active", "preview",
                    "model", "system_prompt",
                ):
                    if key in tip_row:
                        merged[key] = tip_row[key]
                merged["_lineage_root_id"] = s["id"]
                projected.append(merged)
            sessions = projected

        return sessions

    def _get_session_rich_row(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single session with the same enriched columns as
        ``list_sessions_rich`` (preview + last_active). Returns None if the
        session doesn't exist.
        """
        query = """
            SELECT s.*,
                COALESCE(
                    (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)
                     FROM messages m
                     WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                     ORDER BY m.timestamp, m.id LIMIT 1),
                    ''
                ) AS _preview_raw,
                COALESCE(
                    (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                    s.started_at
                ) AS last_active
            FROM sessions s
            WHERE s.id = ?
        """
        with self._lock:
            cursor = self._conn.execute(query, (session_id,))
            row = cursor.fetchone()
        if not row:
            return None
        s = dict(row)
        raw = s.pop("_preview_raw", "").strip()
        if raw:
            text = raw[:60]
            s["preview"] = text + ("..." if len(raw) > 60 else "")
        else:
            s["preview"] = ""
        return s

    # =========================================================================
    # Message storage
    # =========================================================================

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str = None,
        tool_name: str = None,
        tool_calls: Any = None,
        tool_call_id: str = None,
        token_count: int = None,
        finish_reason: str = None,
        reasoning: str = None,
        reasoning_content: str = None,
        reasoning_details: Any = None,
        codex_reasoning_items: Any = None,
        codex_message_items: Any = None,
    ) -> int:
        """
        Append a message to a session. Returns the message row ID.

        Also increments the session's message_count (and tool_call_count
        if role is 'tool' or tool_calls is present).
        """
        # Serialize structured fields to JSON before entering the write txn
        reasoning_details_json = (
            json.dumps(reasoning_details)
            if reasoning_details else None
        )
        codex_items_json = (
            json.dumps(codex_reasoning_items)
            if codex_reasoning_items else None
        )
        codex_message_items_json = (
            json.dumps(codex_message_items)
            if codex_message_items else None
        )
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None

        # Pre-compute tool call count
        num_tool_calls = 0
        if tool_calls is not None:
            num_tool_calls = len(tool_calls) if isinstance(tool_calls, list) else 1

        def _do(conn):
            cursor = conn.execute(
                """INSERT INTO messages (session_id, role, content, tool_call_id,
                   tool_calls, tool_name, timestamp, token_count, finish_reason,
                   reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                   codex_message_items)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    role,
                    content,
                    tool_call_id,
                    tool_calls_json,
                    tool_name,
                    time.time(),
                    token_count,
                    finish_reason,
                    reasoning,
                    reasoning_content,
                    reasoning_details_json,
                    codex_items_json,
                    codex_message_items_json,
                ),
            )
            msg_id = cursor.lastrowid

            # Update counters
            if num_tool_calls > 0:
                conn.execute(
                    """UPDATE sessions SET message_count = message_count + 1,
                       tool_call_count = tool_call_count + ? WHERE id = ?""",
                    (num_tool_calls, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
                    (session_id,),
                )
            return msg_id

        return self._execute_write(_do)

    def replace_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Atomically replace every message for a session.

        Used by transcript-rewrite flows such as /retry, /undo, and /compress.
        The delete + reinsert sequence must commit as one transaction so a
        mid-rewrite failure does not leave SQLite with a partial transcript.
        """

        def _do(conn):
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            conn.execute(
                "UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = ?",
                (session_id,),
            )

            now_ts = time.time()
            total_messages = 0
            total_tool_calls = 0
            for msg in messages:
                role = msg.get("role", "unknown")
                tool_calls = msg.get("tool_calls")
                reasoning_details = msg.get("reasoning_details") if role == "assistant" else None
                codex_reasoning_items = (
                    msg.get("codex_reasoning_items") if role == "assistant" else None
                )
                codex_message_items = (
                    msg.get("codex_message_items") if role == "assistant" else None
                )

                reasoning_details_json = (
                    json.dumps(reasoning_details) if reasoning_details else None
                )
                codex_items_json = (
                    json.dumps(codex_reasoning_items) if codex_reasoning_items else None
                )
                codex_message_items_json = (
                    json.dumps(codex_message_items) if codex_message_items else None
                )
                tool_calls_json = json.dumps(tool_calls) if tool_calls else None

                conn.execute(
                    """INSERT INTO messages (session_id, role, content, tool_call_id,
                       tool_calls, tool_name, timestamp, token_count, finish_reason,
                       reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                       codex_message_items)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        role,
                        msg.get("content"),
                        msg.get("tool_call_id"),
                        tool_calls_json,
                        msg.get("tool_name"),
                        now_ts,
                        msg.get("token_count"),
                        msg.get("finish_reason"),
                        msg.get("reasoning") if role == "assistant" else None,
                        msg.get("reasoning_content") if role == "assistant" else None,
                        reasoning_details_json,
                        codex_items_json,
                        codex_message_items_json,
                    ),
                )
                total_messages += 1
                if tool_calls is not None:
                    total_tool_calls += (
                        len(tool_calls) if isinstance(tool_calls, list) else 1
                    )
                now_ts += 1e-6

            conn.execute(
                "UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                (total_messages, total_tool_calls, session_id),
            )

        self._execute_write(_do)

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Load all messages for a session, ordered by timestamp."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp, id",
                (session_id,),
            )
            rows = cursor.fetchall()
        result = []
        for row in rows:
            msg = dict(row)
            if msg.get("tool_calls"):
                try:
                    msg["tool_calls"] = json.loads(msg["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Failed to deserialize tool_calls in get_messages, falling back to []")
                    msg["tool_calls"] = []
            result.append(msg)
        return result

    def resolve_resume_session_id(self, session_id: str) -> str:
        """Redirect a resume target to the descendant session that holds the messages.

        Context compression ends the current session and forks a new child session
        (linked via ``parent_session_id``). The flush cursor is reset, so the
        child is where new messages actually land — the parent ends up with
        ``message_count = 0`` rows unless messages had already been flushed to
        it before compression. See #15000.

        This helper walks ``parent_session_id`` forward from ``session_id`` and
        returns the first descendant in the chain that has at least one message
        row. If the original session already has messages, or no descendant
        has any, the original ``session_id`` is returned unchanged.

        The chain is always walked via the child whose ``started_at`` is
        latest; that matches the single-chain shape that compression creates.
        A depth cap (32) guards against accidental loops in malformed data.
        """
        if not session_id:
            return session_id

        with self._lock:
            # If this session already has messages, nothing to redirect.
            try:
                row = self._conn.execute(
                    "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1",
                    (session_id,),
                ).fetchone()
            except Exception:
                return session_id
            if row is not None:
                return session_id

            # Walk descendants: at each step, pick the most-recently-started
                # child session; stop once we find one with messages.
            current = session_id
            seen = {current}
            for _ in range(32):
                try:
                    child_row = self._conn.execute(
                        "SELECT id FROM sessions "
                        "WHERE parent_session_id = ? "
                        "ORDER BY started_at DESC, id DESC LIMIT 1",
                        (current,),
                    ).fetchone()
                except Exception:
                    return session_id
                if child_row is None:
                    return session_id
                child_id = child_row["id"] if hasattr(child_row, "keys") else child_row[0]
                if not child_id or child_id in seen:
                    return session_id
                seen.add(child_id)
                try:
                    msg_row = self._conn.execute(
                        "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1",
                        (child_id,),
                    ).fetchone()
                except Exception:
                    return session_id
                if msg_row is not None:
                    return child_id
                current = child_id
        return session_id

    def get_messages_as_conversation(
        self, session_id: str, include_ancestors: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Load messages in the OpenAI conversation format (role + content dicts).
        Used by the gateway to restore conversation history.
        """
        session_ids = [session_id]
        if include_ancestors:
            session_ids = self._session_lineage_root_to_tip(session_id)

        with self._lock:
            placeholders = ",".join("?" for _ in session_ids)
            rows = self._conn.execute(
                "SELECT role, content, tool_call_id, tool_calls, tool_name, "
                "reasoning, reasoning_content, reasoning_details, codex_reasoning_items, "
                "codex_message_items "
                f"FROM messages WHERE session_id IN ({placeholders}) ORDER BY timestamp, id",
                tuple(session_ids),
            ).fetchall()

        messages = []
        for row in rows:
            content = row["content"]
            if row["role"] in {"user", "assistant"} and isinstance(content, str):
                content = sanitize_context(content).strip()
            msg = {"role": row["role"], "content": content}
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            if row["tool_name"]:
                msg["tool_name"] = row["tool_name"]
            if row["tool_calls"]:
                try:
                    msg["tool_calls"] = json.loads(row["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Failed to deserialize tool_calls in conversation replay, falling back to []")
                    msg["tool_calls"] = []
            # Restore reasoning fields on assistant messages so providers
            # that replay reasoning (OpenRouter, OpenAI, Nous) receive
            # coherent multi-turn reasoning context.
            if row["role"] == "assistant":
                if row["reasoning"]:
                    msg["reasoning"] = row["reasoning"]
                if row["reasoning_content"] is not None:
                    msg["reasoning_content"] = row["reasoning_content"]
                if row["reasoning_details"]:
                    try:
                        msg["reasoning_details"] = json.loads(row["reasoning_details"])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Failed to deserialize reasoning_details, falling back to None")
                        msg["reasoning_details"] = None
                if row["codex_reasoning_items"]:
                    try:
                        msg["codex_reasoning_items"] = json.loads(row["codex_reasoning_items"])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Failed to deserialize codex_reasoning_items, falling back to None")
                        msg["codex_reasoning_items"] = None
                if row["codex_message_items"]:
                    try:
                        msg["codex_message_items"] = json.loads(row["codex_message_items"])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Failed to deserialize codex_message_items, falling back to None")
                        msg["codex_message_items"] = None
            if include_ancestors and self._is_duplicate_replayed_user_message(messages, msg):
                continue
            messages.append(msg)
        return messages

    def _session_lineage_root_to_tip(self, session_id: str) -> List[str]:
        if not session_id:
            return [session_id]

        chain = []
        current = session_id
        seen = set()
        with self._lock:
            for _ in range(100):
                if not current or current in seen:
                    break
                seen.add(current)
                chain.append(current)
                row = self._conn.execute(
                    "SELECT parent_session_id FROM sessions WHERE id = ?",
                    (current,),
                ).fetchone()
                if row is None:
                    break
                current = row["parent_session_id"] if hasattr(row, "keys") else row[0]
        return list(reversed(chain)) or [session_id]

    @staticmethod
    def _is_duplicate_replayed_user_message(messages: List[Dict[str, Any]], msg: Dict[str, Any]) -> bool:
        if msg.get("role") != "user":
            return False
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            return False
        for prev in reversed(messages):
            if prev.get("role") == "user" and prev.get("content") == content:
                return True
            if prev.get("role") == "assistant" and (prev.get("content") or prev.get("tool_calls")):
                return False
        return False

    # =========================================================================
    # Search
    # =========================================================================

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Sanitize user input for safe use in FTS5 MATCH queries.

        FTS5 has its own query syntax where characters like ``"``, ``(``, ``)``,
        ``+``, ``*``, ``{``, ``}`` and bare boolean operators (``AND``, ``OR``,
        ``NOT``) have special meaning.  Passing raw user input directly to
        MATCH can cause ``sqlite3.OperationalError``.

        Strategy:
        - Preserve properly paired quoted phrases (``"exact phrase"``)
        - Strip unmatched FTS5-special characters that would cause errors
        - Wrap unquoted hyphenated and dotted terms in quotes so FTS5
          matches them as exact phrases instead of splitting on the
          hyphen/dot (e.g. ``chat-send``, ``P2.2``, ``my-app.config.ts``)
        """
        # Step 1: Extract balanced double-quoted phrases and protect them
        # from further processing via numbered placeholders.
        _quoted_parts: list = []

        def _preserve_quoted(m: re.Match) -> str:
            _quoted_parts.append(m.group(0))
            return f"\x00Q{len(_quoted_parts) - 1}\x00"

        sanitized = re.sub(r'"[^"]*"', _preserve_quoted, query)

        # Step 2: Strip remaining (unmatched) FTS5-special characters
        sanitized = re.sub(r'[+{}()\"^]', " ", sanitized)

        # Step 3: Collapse repeated * (e.g. "***") into a single one,
        # and remove leading * (prefix-only needs at least one char before *)
        sanitized = re.sub(r"\*+", "*", sanitized)
        sanitized = re.sub(r"(^|\s)\*", r"\1", sanitized)

        # Step 4: Remove dangling boolean operators at start/end that would
        # cause syntax errors (e.g. "hello AND" or "OR world")
        sanitized = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", sanitized.strip())
        sanitized = re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", sanitized.strip())

        # Step 5: Wrap unquoted dotted and/or hyphenated terms in double
        # quotes.  FTS5's tokenizer splits on dots and hyphens, turning
        # ``chat-send`` into ``chat AND send`` and ``P2.2`` into ``p2 AND 2``.
        # Quoting preserves phrase semantics.  A single pass avoids the
        # double-quoting bug that would occur if dotted, hyphenated and underscored
        # patterns were applied sequentially (e.g. ``my-app.config``).
        sanitized = re.sub(r"\b(\w+(?:[._-]\w+)+)\b", r'"\1"', sanitized)

        # Step 6: Restore preserved quoted phrases
        for i, quoted in enumerate(_quoted_parts):
            sanitized = sanitized.replace(f"\x00Q{i}\x00", quoted)

        return sanitized.strip()


    @staticmethod
    def _is_cjk_codepoint(cp: int) -> bool:
        return (0x4E00 <= cp <= 0x9FFF or    # CJK Unified Ideographs
                0x3400 <= cp <= 0x4DBF or    # CJK Extension A
                0x20000 <= cp <= 0x2A6DF or  # CJK Extension B
                0x3000 <= cp <= 0x303F or    # CJK Symbols
                0x3040 <= cp <= 0x309F or    # Hiragana
                0x30A0 <= cp <= 0x30FF or    # Katakana
                0xAC00 <= cp <= 0xD7AF)      # Hangul Syllables

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        """Check if text contains CJK (Chinese, Japanese, Korean) characters."""
        for ch in text:
            cp = ord(ch)
            if (0x4E00 <= cp <= 0x9FFF or    # CJK Unified Ideographs
                0x3400 <= cp <= 0x4DBF or    # CJK Extension A
                0x20000 <= cp <= 0x2A6DF or  # CJK Extension B
                0x3000 <= cp <= 0x303F or    # CJK Symbols
                0x3040 <= cp <= 0x309F or    # Hiragana
                0x30A0 <= cp <= 0x30FF or    # Katakana
                0xAC00 <= cp <= 0xD7AF):     # Hangul Syllables
                return True
        return False

    @classmethod
    def _count_cjk(cls, text: str) -> int:
        """Count CJK characters in text."""
        return sum(1 for ch in text if cls._is_cjk_codepoint(ord(ch)))

    def search_messages(
        self,
        query: str,
        source_filter: List[str] = None,
        exclude_sources: List[str] = None,
        role_filter: List[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Full-text search across session messages using FTS5.

        Supports FTS5 query syntax:
          - Simple keywords: "docker deployment"
          - Phrases: '"exact phrase"'
          - Boolean: "docker OR kubernetes", "python NOT java"
          - Prefix: "deploy*"

        Returns matching messages with session metadata, content snippet,
        and surrounding context (1 message before and after the match).
        """
        if not query or not query.strip():
            return []

        query = self._sanitize_fts5_query(query)
        if not query:
            return []

        # Build WHERE clauses dynamically
        where_clauses = ["messages_fts MATCH ?"]
        params: list = [query]

        if source_filter is not None:
            source_placeholders = ",".join("?" for _ in source_filter)
            where_clauses.append(f"s.source IN ({source_placeholders})")
            params.extend(source_filter)

        if exclude_sources is not None:
            exclude_placeholders = ",".join("?" for _ in exclude_sources)
            where_clauses.append(f"s.source NOT IN ({exclude_placeholders})")
            params.extend(exclude_sources)

        if role_filter:
            role_placeholders = ",".join("?" for _ in role_filter)
            where_clauses.append(f"m.role IN ({role_placeholders})")
            params.extend(role_filter)

        where_sql = " AND ".join(where_clauses)
        params.extend([limit, offset])

        sql = f"""
            SELECT
                m.id,
                m.session_id,
                m.role,
                snippet(messages_fts, 0, '>>>', '<<<', '...', 40) AS snippet,
                m.content,
                m.timestamp,
                m.tool_name,
                s.source,
                s.model,
                s.started_at AS session_started
            FROM messages_fts
            JOIN messages m ON m.id = messages_fts.rowid
            JOIN sessions s ON s.id = m.session_id
            WHERE {where_sql}
            ORDER BY rank
            LIMIT ? OFFSET ?
        """

        # CJK queries bypass the unicode61 FTS5 table.  The default tokenizer
        # splits CJK characters into individual tokens, so "大别山项目" becomes
        # "大 AND 别 AND 山 AND 项 AND 目" — producing false positives and
        # missing exact phrase matches.
        #
        # For queries with 3+ CJK characters, we use the trigram FTS5 table
        # (indexed substring matching with ranking and snippets).  For shorter
        # CJK queries (1-2 chars), trigram can't match (it needs ≥9 UTF-8
        # bytes = 3 CJK chars), so we fall back to LIKE.
        is_cjk = self._contains_cjk(query)
        if is_cjk:
            raw_query = query.strip('"').strip()
            cjk_count = self._count_cjk(raw_query)

            if cjk_count >= 3:
                # Trigram FTS5 path — quote each non-operator token to handle
                # FTS5 special chars (%, *, etc.) while preserving boolean
                # operators (AND, OR, NOT) for multi-term queries.
                tokens = raw_query.split()
                parts = []
                for tok in tokens:
                    if tok.upper() in ("AND", "OR", "NOT"):
                        parts.append(tok)
                    else:
                        parts.append('"' + tok.replace('"', '""') + '"')
                trigram_query = " ".join(parts)
                tri_where = ["messages_fts_trigram MATCH ?"]
                tri_params: list = [trigram_query]
                if source_filter is not None:
                    tri_where.append(f"s.source IN ({','.join('?' for _ in source_filter)})")
                    tri_params.extend(source_filter)
                if exclude_sources is not None:
                    tri_where.append(f"s.source NOT IN ({','.join('?' for _ in exclude_sources)})")
                    tri_params.extend(exclude_sources)
                if role_filter:
                    tri_where.append(f"m.role IN ({','.join('?' for _ in role_filter)})")
                    tri_params.extend(role_filter)
                tri_sql = f"""
                    SELECT
                        m.id,
                        m.session_id,
                        m.role,
                        snippet(messages_fts_trigram, 0, '>>>', '<<<', '...', 40) AS snippet,
                        m.content,
                        m.timestamp,
                        m.tool_name,
                        s.source,
                        s.model,
                        s.started_at AS session_started
                    FROM messages_fts_trigram
                    JOIN messages m ON m.id = messages_fts_trigram.rowid
                    JOIN sessions s ON s.id = m.session_id
                    WHERE {' AND '.join(tri_where)}
                    ORDER BY rank
                    LIMIT ? OFFSET ?
                """
                tri_params.extend([limit, offset])
                with self._lock:
                    try:
                        tri_cursor = self._conn.execute(tri_sql, tri_params)
                    except sqlite3.OperationalError:
                        matches = []
                    else:
                        matches = [dict(row) for row in tri_cursor.fetchall()]
            else:
                # Short CJK query (1-2 chars) — trigram needs ≥3 CJK chars.
                # Fall back to LIKE substring search.
                escaped = raw_query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                like_where = ["(m.content LIKE ? ESCAPE '\\' OR m.tool_name LIKE ? ESCAPE '\\' OR m.tool_calls LIKE ? ESCAPE '\\')"]
                like_params: list = [f"%{escaped}%", f"%{escaped}%", f"%{escaped}%"]
                if source_filter is not None:
                    like_where.append(f"s.source IN ({','.join('?' for _ in source_filter)})")
                    like_params.extend(source_filter)
                if exclude_sources is not None:
                    like_where.append(f"s.source NOT IN ({','.join('?' for _ in exclude_sources)})")
                    like_params.extend(exclude_sources)
                if role_filter:
                    like_where.append(f"m.role IN ({','.join('?' for _ in role_filter)})")
                    like_params.extend(role_filter)
                like_sql = f"""
                    SELECT m.id, m.session_id, m.role,
                           substr(m.content,
                                  max(1, instr(m.content, ?) - 40),
                                  120) AS snippet,
                           m.content, m.timestamp, m.tool_name,
                           s.source, s.model, s.started_at AS session_started
                    FROM messages m
                    JOIN sessions s ON s.id = m.session_id
                    WHERE {' AND '.join(like_where)}
                    ORDER BY m.timestamp DESC
                    LIMIT ? OFFSET ?
                """
                like_params.extend([limit, offset])
                # instr() parameter goes first in the bound list
                like_params = [raw_query] + like_params
                with self._lock:
                    like_cursor = self._conn.execute(like_sql, like_params)
                    matches = [dict(row) for row in like_cursor.fetchall()]
        else:
            with self._lock:
                try:
                    cursor = self._conn.execute(sql, params)
                except sqlite3.OperationalError:
                    # FTS5 query syntax error despite sanitization — return empty
                    return []
                else:
                    matches = [dict(row) for row in cursor.fetchall()]

        # Add surrounding context (1 message before + after each match).
        # Done outside the lock so we don't hold it across N sequential queries.
        for match in matches:
            try:
                with self._lock:
                    ctx_cursor = self._conn.execute(
                        """WITH target AS (
                               SELECT session_id, timestamp, id
                               FROM messages
                               WHERE id = ?
                           )
                           SELECT role, content
                           FROM (
                               SELECT m.id, m.timestamp, m.role, m.content
                               FROM messages m
                               JOIN target t ON t.session_id = m.session_id
                               WHERE (m.timestamp < t.timestamp)
                                  OR (m.timestamp = t.timestamp AND m.id < t.id)
                               ORDER BY m.timestamp DESC, m.id DESC
                               LIMIT 1
                           )
                           UNION ALL
                           SELECT role, content
                           FROM messages
                           WHERE id = ?
                           UNION ALL
                           SELECT role, content
                           FROM (
                               SELECT m.id, m.timestamp, m.role, m.content
                               FROM messages m
                               JOIN target t ON t.session_id = m.session_id
                               WHERE (m.timestamp > t.timestamp)
                                  OR (m.timestamp = t.timestamp AND m.id > t.id)
                               ORDER BY m.timestamp ASC, m.id ASC
                               LIMIT 1
                           )""",
                        (match["id"], match["id"]),
                    )
                    context_msgs = [
                        {"role": r["role"], "content": (r["content"] or "")[:200]}
                        for r in ctx_cursor.fetchall()
                    ]
                match["context"] = context_msgs
            except Exception:
                match["context"] = []

        # Remove full content from result (snippet is enough, saves tokens)
        for match in matches:
            match.pop("content", None)

        return matches

    def search_sessions(
        self,
        source: str = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List sessions, optionally filtered by source.

        Returns rows enriched with a computed ``last_active`` column (latest
        message timestamp for the session, falling back to ``started_at``),
        ordered by most-recently-used first.
        """
        select_with_last_active = (
            "SELECT s.*, COALESCE(m.last_active, s.started_at) AS last_active "
            "FROM sessions s "
            "LEFT JOIN ("
            "SELECT session_id, MAX(timestamp) AS last_active "
            "FROM messages GROUP BY session_id"
            ") m ON m.session_id = s.id "
        )
        with self._lock:
            if source:
                cursor = self._conn.execute(
                    f"{select_with_last_active}"
                    "WHERE s.source = ? "
                    "ORDER BY last_active DESC, s.started_at DESC, s.id DESC LIMIT ? OFFSET ?",
                    (source, limit, offset),
                )
            else:
                cursor = self._conn.execute(
                    f"{select_with_last_active}"
                    "ORDER BY last_active DESC, s.started_at DESC, s.id DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                )
            return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # Memory Nodes (summarized event storage with vector + keyword search)
    # =========================================================================

    MEMORY_TOP_K_CAUSAL = 5
    MEMORY_UPDATE_CAUSAL_THRESHOLD = 0.60
    MEMORY_QUERY_RETRIEVAL_THRESHOLD = 0.3
    MEMORY_QUERY_TOP_K = 8

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _normalize_memory_fact_type(value: Any) -> str:
        """Normalize retained memory fact types to the two recall buckets."""
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if text in {"episodic", "episodic_memory", "experience", "experience_fact"}:
            return "episodic"
        if text in {"semantic", "semantic_memory", "world", "world_fact"}:
            return "semantic"
        return "semantic"

    @staticmethod
    def _normalize_memory_fact_subject(value: Any) -> str:
        """Normalize retained memory fact subjects to source/actor buckets."""
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        allowed = {"user", "assistant", "world", "project", "system", "other"}
        return text if text in allowed else "other"

    @classmethod
    def _memory_fact_type_from_tags(cls, tags: Any) -> str:
        """Infer fact_type from legacy tag arrays."""
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except json.JSONDecodeError:
                tags = [tags]
        if not isinstance(tags, list):
            return "semantic"
        for tag in tags:
            text = str(tag or "").strip().lower()
            if text.startswith("fact_type:"):
                return cls._normalize_memory_fact_type(text.split(":", 1)[1])
        return "semantic"

    @classmethod
    def _memory_fact_subject_from_tags(cls, tags: Any) -> str:
        """Infer fact_subject from legacy tag arrays."""
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except json.JSONDecodeError:
                tags = [tags]
        if not isinstance(tags, list):
            return "other"
        for tag in tags:
            text = str(tag or "").strip().lower()
            if text.startswith("fact_subject:"):
                return cls._normalize_memory_fact_subject(text.split(":", 1)[1])
        for tag in tags:
            text = str(tag or "").strip().lower()
            if text == "fact_type:experience":
                return "assistant"
        return "other"

    @staticmethod
    def _normalize_memory_fact_kind(value: Any) -> str:
        """Normalize retained memory fact kinds to the supported semantic buckets."""
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        allowed = {
            "preference", "decision", "request", "recommendation",
            "action", "error", "context", "instruction",
            "conversation_summary", "other",
        }
        return text if text in allowed else "other"

    @classmethod
    def _memory_fact_kind_from_tags(cls, tags: Any) -> str:
        """Infer fact_kind from legacy tag arrays."""
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except json.JSONDecodeError:
                tags = [tags]
        if not isinstance(tags, list):
            return "other"
        for tag in tags:
            text = str(tag or "").strip().lower()
            if text.startswith("fact_kind:"):
                return cls._normalize_memory_fact_kind(text.split(":", 1)[1])
        return "other"

    @staticmethod
    def _normalize_memory_interpretation_status(value: Any) -> str:
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        allowed = {"current", "superseded", "conflicted", "archived"}
        return text if text in allowed else "current"

    @staticmethod
    def _normalize_memory_interpretation_type(value: Any) -> str:
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        allowed = {
            "insight", "task",
            "explicit_preference", "explicit_instruction", "inferred_preference",
            "behavior_pattern", "project_state", "task_risk", "constraint",
            "conflict_resolution", "strategy", "other",
        }
        return text if text in allowed else "behavior_pattern"

    @staticmethod
    def _normalize_memory_interpretation_conflict_status(value: Any) -> str:
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        allowed = {"none", "resolved", "unresolved"}
        return text if text in allowed else "none"

    @staticmethod
    def _json_int_list(value: Any) -> List[int]:
        if isinstance(value, str):
            try:
                value = json.loads(value or "[]")
            except json.JSONDecodeError:
                value = []
        if not isinstance(value, list):
            return []
        out: List[int] = []
        seen = set()
        for item in value:
            try:
                int_item = int(item)
            except (TypeError, ValueError):
                continue
            if int_item in seen:
                continue
            seen.add(int_item)
            out.append(int_item)
        return out

    def _memory_interpretation_from_row(self, row: Any) -> Dict[str, Any]:
        item = dict(row)
        item["interpretation_type"] = self._normalize_memory_interpretation_type(
            item.get("interpretation_type")
        )
        item["status"] = self._normalize_memory_interpretation_status(item.get("status"))
        item["conflict_status"] = self._normalize_memory_interpretation_conflict_status(
            item.get("conflict_status")
        )
        for key in (
            "evidence_node_ids",
            "evidence_observation_ids",
            "counter_evidence_node_ids",
            "counter_evidence_observation_ids",
        ):
            item[key] = self._json_int_list(item.get(key))
        metadata = item.get("metadata")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata or "{}")
            except json.JSONDecodeError:
                metadata = {}
        item["metadata"] = metadata if isinstance(metadata, dict) else {}
        if self._coerce_int_or_none(item.get("entity_id")) is None:
            item["entity_id"] = self._coerce_int_or_none(item["metadata"].get("entity_id"))
        return item

    def _memory_get_node(self, node_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single memory node with tags and relations."""
        cursor = self._conn.execute(
            """SELECT id, time_key, summary, keywords, topic, original_dialog,
                      tags, fact_type, fact_subject, fact_kind, decay_score, decay_updated_at,
                      decay_half_life_days, task_event_like, task_event_subject,
                      task_relevance, entity_names
               FROM memory_nodes
               WHERE id = ?""",
            (node_id,),
        )
        r = cursor.fetchone()
        if not r:
            return None
        # Fetch relations from normalized table
        rel_cursor = self._conn.execute(
            """SELECT target_node_id, relation_type, confidence
               FROM memory_node_relations
               WHERE source_node_id = ?""",
            (node_id,),
        )
        node_relations = {}
        for rel_row in rel_cursor.fetchall():
            node_relations[str(rel_row[0])] = rel_row[1]
        tags = json.loads(r[6]) if r[6] else []
        fact_type = self._normalize_memory_fact_type(r[7] or self._memory_fact_type_from_tags(tags))
        fact_subject = self._normalize_memory_fact_subject(
            r[8] or self._memory_fact_subject_from_tags(tags)
        )
        fact_kind = self._normalize_memory_fact_kind(r[9] or self._memory_fact_kind_from_tags(tags))
        return {
            "id": r[0],
            "time_key": r[1],
            "summary": r[2],
            "keywords": r[3].split(" "),
            "topic": r[4].split(" "),
            "original_dialog": r[5],
            "tags": tags,
            "fact_type": fact_type,
            "fact_subject": fact_subject,
            "fact_kind": fact_kind,
            "decay_score": float(r[10]) if r[10] is not None else 1.0,
            "decay_updated_at": r[11],
            "decay_half_life_days": r[12],
            "task_event_like": None if r[13] is None else bool(r[13]),
            "task_event_subject": r[14] or "",
            "task_relevance": r[15] or "",
            "entity_names": json.loads(r[16] or "[]"),
            "node_relations": node_relations,
        }

    def memory_nodes_by_ids(self, node_ids: List[int]) -> List[Dict[str, Any]]:
        """Fetch memory fact nodes by id, preserving caller order."""
        clean_ids = self._json_int_list(node_ids)
        if not clean_ids:
            return []
        out: List[Dict[str, Any]] = []
        for node_id in clean_ids:
            node = self._memory_get_node(node_id)
            if node:
                out.append(node)
        return out

    @staticmethod
    def _memory_node_filter_sql(
        *,
        table_alias: str = "",
        allowed_ids: Optional[set] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
    ) -> Tuple[str, List[Any]]:
        """Build SQL predicates for memory-node candidate narrowing."""
        prefix = f"{table_alias}." if table_alias else ""
        clauses: List[str] = []
        params: List[Any] = []
        if time_start is not None:
            clauses.append(f"{prefix}time_key >= ?")
            params.append(time_start)
        if time_end is not None:
            clauses.append(f"{prefix}time_key <= ?")
            params.append(time_end)
        if allowed_ids is not None:
            ids = sorted(int(node_id) for node_id in allowed_ids)
            if not ids:
                clauses.append("0")
            else:
                placeholders = ",".join("?" for _ in ids)
                clauses.append(f"{prefix}id IN ({placeholders})")
                params.extend(ids)
        if not clauses:
            return "", []
        return " AND " + " AND ".join(clauses), params

    def _search_memory_keyword(
        self,
        keyword: str,
        limit: int = 20,
        *,
        allowed_ids: Optional[set] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
    ) -> Dict[int, float]:
        """Keyword search over memory nodes. Returns {rowid: score}.

        Uses FTS5 MATCH for non-CJK queries (returns BM25 scores, lower = better).
        Falls back to complete-term matching on summary/keywords for CJK queries.
        Optional candidate filters are pushed into the SQL query so time-limited
        recall does not first search the full memory table.
        """
        if self._contains_cjk(keyword):
            # Keep CJK words/phrases intact. Splitting Chinese queries into
            # individual characters makes recall noisy ("简洁回答" matching any
            # row that merely contains "答"), while memory keywords are already
            # word-like terms extracted by the memory summarizer. If a complete
            # term has no contiguous match, fall back to requiring all CJK
            # characters in the term so "喜欢结论" can match "喜欢先给结论"
            # without letting partial single-character hits dominate.
            clean = keyword.replace('"', "").replace("*", "").strip()
            raw_terms = re.split(r"\s+OR\s+|\s+", clean, flags=re.IGNORECASE)
            terms = []
            seen_terms = set()
            for raw_term in raw_terms:
                term = raw_term.strip()
                if not term or term.upper() in {"AND", "OR", "NOT"}:
                    continue
                if term not in seen_terms:
                    terms.append(term)
                    seen_terms.add(term)
            if not terms:
                return {}

            filter_sql, filter_params = self._memory_node_filter_sql(
                allowed_ids=allowed_ids,
                time_start=time_start,
                time_end=time_end,
            )

            # Build a UNION ALL query that counts complete-term matches.
            # Each term gets its own SELECT: returns node id if it matches
            # summary OR keywords.  Then GROUP BY + COUNT gives exact match count.
            selects = []
            params = []
            bs = chr(92)  # backslash character for ESCAPE
            for t in terms:
                escaped = t.replace(bs, bs + bs).replace("%", bs + "%").replace("_", bs + "_")
                pattern = f"%{escaped}%"
                selects.append(
                    f"SELECT id FROM memory_nodes WHERE summary LIKE ? ESCAPE '{bs}'{filter_sql}"
                )
                params.extend([pattern] + filter_params)
                selects.append(
                    f"SELECT id FROM memory_nodes WHERE keywords LIKE ? ESCAPE '{bs}'{filter_sql}"
                )
                params.extend([pattern] + filter_params)

            union = " UNION ALL ".join(selects)
            cursor = self._conn.execute(
                f"""SELECT id, CAST(COUNT(*) AS REAL) AS score
                    FROM ({union})
                    GROUP BY id
                    ORDER BY score DESC
                    LIMIT ?""",
                params + [limit],
            )
            results = {row[0]: len(terms) * 2 - row[1] for row in cursor.fetchall()}
            if results:
                return results

            fallback_selects = []
            fallback_params = []
            for term in terms:
                cjk_chars = []
                seen_chars = set()
                for ch in term:
                    if not self._is_cjk_codepoint(ord(ch)) or ch in seen_chars:
                        continue
                    cjk_chars.append(ch)
                    seen_chars.add(ch)
                if len(cjk_chars) < 2:
                    continue
                term_clauses = []
                for ch in cjk_chars:
                    escaped = ch.replace(bs, bs + bs).replace("%", bs + "%").replace("_", bs + "_")
                    pattern = f"%{escaped}%"
                    term_clauses.append(
                        "(summary LIKE ? ESCAPE ? OR keywords LIKE ? ESCAPE ?)"
                    )
                    fallback_params.extend([pattern, bs, pattern, bs])
                fallback_selects.append(
                    f"SELECT id, {len(cjk_chars)} AS score FROM memory_nodes "
                    f"WHERE {' AND '.join(term_clauses)}{filter_sql}"
                )
                fallback_params.extend(filter_params)
            if not fallback_selects:
                return {}
            fallback_union = " UNION ALL ".join(fallback_selects)
            fallback_cursor = self._conn.execute(
                f"""SELECT id, CAST(SUM(score) AS REAL) AS score
                    FROM ({fallback_union})
                    GROUP BY id
                    ORDER BY score DESC
                    LIMIT ?""",
                fallback_params + [limit],
            )
            return {row[0]: max(0.0, len(terms) * 2 - row[1]) for row in fallback_cursor.fetchall()}

        # Non-CJK: use FTS5 BM25
        filter_sql, filter_params = self._memory_node_filter_sql(
            table_alias="mn",
            allowed_ids=allowed_ids,
            time_start=time_start,
            time_end=time_end,
        )
        cursor = self._conn.execute(
            f"""SELECT memory_nodes_fts.rowid, bm25(memory_nodes_fts) AS score
               FROM memory_nodes_fts
               JOIN memory_nodes mn ON mn.id = memory_nodes_fts.rowid
               WHERE memory_nodes_fts MATCH ?
               {filter_sql}
               ORDER BY score
               LIMIT ?""",
            [keyword] + filter_params + [limit],
        )
        return {row[0]: row[1] for row in cursor.fetchall()}

    def _search_memory_vector(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        *,
        allowed_ids: Optional[set] = None,
    ) -> Dict[int, float]:
        """FAISS vector search over memory nodes. Returns {node_id: similarity} (higher = better).

        When a bounded candidate set is supplied, small windows are scored
        directly by reconstructing only those vectors from the flat FAISS index.
        Large windows fall back to overfetching from the global index and then
        filtering, which avoids reconstructing very large historical ranges.
        """
        if not _HAS_FAISS or self._memory_faiss_index is None or self._memory_faiss_index.ntotal == 0:
            return {}
        ntotal = int(self._memory_faiss_index.ntotal)
        if allowed_ids is not None:
            allowed = {int(node_id) for node_id in allowed_ids}
            max_pos = min(ntotal, len(self._memory_faiss_id_map))
            allowed_positions = [
                (idx, self._memory_faiss_id_map[idx])
                for idx in range(max_pos)
                if self._memory_faiss_id_map[idx] in allowed
            ]
            if not allowed_positions:
                return {}
            if len(allowed_positions) <= self._MEMORY_VECTOR_FILTER_BRUTE_FORCE_LIMIT:
                query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
                scored: List[Tuple[int, float]] = []
                for idx, node_id in allowed_positions:
                    try:
                        vector = self._memory_faiss_index.reconstruct(int(idx))
                    except TypeError:
                        vector = np.empty((query_vec.shape[0],), dtype=np.float32)
                        self._memory_faiss_index.reconstruct(int(idx), vector)
                    except Exception:
                        vector = None
                    if vector is None:
                        continue
                    candidate = np.asarray(vector, dtype=np.float32).reshape(-1)
                    if candidate.shape != query_vec.shape:
                        continue
                    scored.append((node_id, float(np.dot(query_vec, candidate))))
                scored.sort(key=lambda item: item[1], reverse=True)
                return dict(scored[:top_k])

            density = len(allowed_positions) / max(ntotal, 1)
            overfetch = int(math.ceil(top_k / max(density, 0.01)) * 2)
            search_k = min(ntotal, max(top_k * 8, overfetch, top_k))
            sims, indices = self._memory_faiss_index.search(query_embedding, search_k)
        else:
            sims, indices = self._memory_faiss_index.search(query_embedding, top_k)
        results: Dict[int, float] = {}
        for sim, idx in zip(sims[0], indices[0]):
            if idx == -1:
                continue
            node_id = self._memory_faiss_id_map[idx]
            if allowed_ids is not None and node_id not in allowed_ids:
                continue
            results[node_id] = sim
            if len(results) >= top_k:
                break
        return results

    def memory_node_embeddings(self, node_ids: List[int]) -> Dict[int, np.ndarray]:
        """Reconstruct stored FAISS vectors for the requested memory nodes."""
        if not _HAS_FAISS or self._memory_faiss_index is None or self._memory_faiss_index.ntotal == 0:
            return {}
        requested = {
            int(node_id)
            for node_id in node_ids
            if node_id is not None
        }
        if not requested:
            return {}
        vectors: Dict[int, np.ndarray] = {}
        max_pos = min(
            int(self._memory_faiss_index.ntotal),
            len(self._memory_faiss_id_map),
        )
        for idx in range(max_pos - 1, -1, -1):
            node_id = int(self._memory_faiss_id_map[idx])
            if node_id not in requested:
                continue
            try:
                vector = self._memory_faiss_index.reconstruct(idx)
            except TypeError:
                vector = np.empty((self._memory_faiss_index.d,), dtype=np.float32)
                self._memory_faiss_index.reconstruct(idx, vector)
            except Exception:
                continue
            candidate = np.asarray(vector, dtype=np.float32).reshape(-1)
            if candidate.size:
                vectors[node_id] = candidate
            if len(vectors) >= len(requested):
                break
        return vectors

    def memory_semantic_neighbors(
        self,
        query_embedding: np.ndarray,
        *,
        exclude_node_id: Optional[int] = None,
        allowed_ids: Optional[set] = None,
        threshold: float = 0.82,
    ) -> Dict[int, float]:
        """Return all FAISS neighbors whose cosine similarity meets threshold.

        Embeddings are L2-normalized by ``EmbeddingClient`` by default and the
        memory FAISS index is ``IndexFlatIP``, so inner product is cosine
        similarity for normal Hermes memory nodes.
        """
        if not _HAS_FAISS or self._memory_faiss_index is None or self._memory_faiss_index.ntotal == 0:
            return {}
        top_k = int(self._memory_faiss_index.ntotal)
        if top_k <= 0:
            return {}
        scores = self._search_memory_vector(query_embedding, top_k=top_k)
        out: Dict[int, float] = {}
        for node_id, score in scores.items():
            if exclude_node_id is not None and node_id == exclude_node_id:
                continue
            if allowed_ids is not None and node_id not in allowed_ids:
                continue
            try:
                similarity = float(score)
            except (TypeError, ValueError):
                continue
            if similarity >= threshold:
                out[node_id] = similarity
        return out

    def _memory_save_faiss(self) -> None:
        """Persist FAISS index and id_map to disk next to the SQLite DB.

        Saves to ``<db_path>.faiss`` (binary FAISS index) and
        ``<db_path>.faiss_ids.json`` (id list).
        Silently skips if FAISS is unavailable or the index is empty.
        """
        if not _HAS_FAISS or self._memory_faiss_index is None or self._memory_faiss_index.ntotal == 0:
            return
        try:
            faiss.write_index(self._memory_faiss_index, str(self._memory_faiss_save_path))
            with open(self._memory_faiss_ids_path, "w", encoding="utf-8") as f:
                json.dump(self._memory_faiss_id_map, f)
        except Exception as exc:
            logger.debug("Failed to save FAISS index to disk: %s", exc)

    def _memory_load_faiss(self) -> None:
        """Load persisted FAISS index and id_map from disk.

        If the saved files don't exist or are corrupt, the index starts
        empty — new nodes will be added incrementally from that point.
        """
        idx_path = self._memory_faiss_save_path
        ids_path = self._memory_faiss_ids_path
        if not idx_path.exists() or not ids_path.exists():
            logger.debug("No persisted FAISS data found at %s — starting fresh", idx_path)
            return
        try:
            loaded_index = faiss.read_index(str(idx_path))
            with open(ids_path, "r", encoding="utf-8") as f:
                loaded_ids = json.load(f)
            if not isinstance(loaded_ids, list) or loaded_index.ntotal != len(loaded_ids):
                logger.debug(
                    "FAISS index (%d vectors) / id_map (%d entries) mismatch — discarding",
                    loaded_index.ntotal, len(loaded_ids),
                )
                return
            self._memory_faiss_index = loaded_index
            self._memory_faiss_id_map = loaded_ids
            logger.debug(
                "Loaded FAISS index with %d vectors and %d id_map entries",
                loaded_index.ntotal, len(loaded_ids),
            )
        except Exception as exc:
            logger.debug("Failed to load FAISS index from disk: %s — starting fresh", exc)

    def _memory_fuse_scores(
        self, keyword_results: Dict[int, float], vec_results: Dict[int, float]
    ) -> List[int]:
        """Fuse BM25 + vector scores with weighted combination. Returns sorted node IDs."""
        all_ids = set(keyword_results.keys()) | set(vec_results.keys())
        fused: Dict[int, float] = {}

        for nid in all_ids:
            bm25 = keyword_results.get(nid)
            sim = vec_results.get(nid)

            # BM25 → positive direction (lower BM25 = better match)
            bm25_score = 1.0 / (1.0 + bm25) if bm25 is not None else 0.0
            vec_score = sim if sim is not None else 0.0

            final = 0.6 * bm25_score + 0.4 * vec_score
            # logger.error(f"bm25_score is {bm25_score}")
            # logger.error(f"vec_score is {vec_score}")
            # logger.error(f"final is {final}")
            
            if final < self.MEMORY_QUERY_RETRIEVAL_THRESHOLD:
                continue
            fused[nid] = final

        return sorted(fused.keys(), key=lambda x: fused[x], reverse=True)

    def _memory_rank_scores(
        self,
        scores: Dict[int, float],
        *,
        higher_is_better: bool,
    ) -> List[int]:
        """Return node IDs sorted by a single retrieval channel."""
        return [
            node_id
            for node_id, _ in sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=higher_is_better,
            )
        ]

    def _memory_rrf(
        self,
        rankings: List[Tuple[List[int], float]],
        *,
        rrf_k: int = 60,
    ) -> List[int]:
        """Reciprocal Rank Fusion over multiple ranked retrieval channels."""
        fused, first_seen = self._memory_rrf_scores(rankings, rrf_k=rrf_k)
        return sorted(
            fused,
            key=lambda node_id: (-fused[node_id], first_seen.get(node_id, 0)),
        )

    def _memory_rrf_scores(
        self,
        rankings: List[Tuple[List[int], float]],
        *,
        rrf_k: int = 60,
    ) -> Tuple[Dict[int, float], Dict[int, int]]:
        """Return raw Reciprocal Rank Fusion scores plus stable tie order."""
        fused: Dict[int, float] = {}
        first_seen: Dict[int, int] = {}
        order = 0
        for ranking, weight in rankings:
            for rank, node_id in enumerate(ranking, 1):
                if node_id not in first_seen:
                    first_seen[node_id] = order
                    order += 1
                fused[node_id] = fused.get(node_id, 0.0) + weight / (rrf_k + rank)
        return fused, first_seen

    def _memory_decay_rerank(
        self,
        ranked_ids: List[int],
        rrf_scores: Dict[int, float],
        first_seen: Dict[int, int],
        *,
        limit: int,
    ) -> List[int]:
        """Apply persisted node recency decay to final recall candidates."""
        candidate_ids = ranked_ids[:max(limit, 1)]
        if not candidate_ids:
            return []
        placeholders = ",".join("?" for _ in candidate_ids)
        rows = self._conn.execute(
            f"SELECT id, decay_score FROM memory_nodes WHERE id IN ({placeholders})",
            candidate_ids,
        ).fetchall()
        decay_by_id: Dict[int, float] = {}
        for row in rows:
            try:
                decay = float(row["decay_score"] if row["decay_score"] is not None else 1.0)
            except (TypeError, ValueError):
                decay = 1.0
            decay_by_id[int(row["id"])] = max(0.0, min(1.0, decay))

        floor = max(0.0, min(1.0, self._MEMORY_RECALL_DECAY_FLOOR))
        scored: List[Tuple[float, int, int]] = []
        for fallback_rank, node_id in enumerate(candidate_ids):
            base_score = rrf_scores.get(node_id, 1.0 / (60 + fallback_rank + 1))
            decay = decay_by_id.get(node_id, 1.0)
            recency_factor = floor + ((1.0 - floor) * decay)
            adjusted = base_score * recency_factor
            scored.append((adjusted, first_seen.get(node_id, fallback_rank), node_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [node_id for _, _, node_id in scored]

    def _memory_filter_ranked_ids(
        self,
        ranked_ids: List[int],
        *,
        allowed_ids: Optional[set] = None,
    ) -> List[int]:
        """Apply an optional allow-list while preserving rank and uniqueness."""
        out: List[int] = []
        seen = set()
        for node_id in ranked_ids:
            if node_id in seen:
                continue
            if allowed_ids is not None and node_id not in allowed_ids:
                continue
            seen.add(node_id)
            out.append(node_id)
        return out

    def _memory_graph_expand_ranked(
        self,
        seed_ids: List[int],
        *,
        depth: int,
        allowed_ids: Optional[set] = None,
        limit: int = 50,
    ) -> List[int]:
        """Rank graph-neighbor memory nodes with priority-guided beam search.

        Graph edges provide reachability.  Edge scores decide expansion order:
        explicit memory-node relations use their semantic / causal / temporal
        components plus the entity overlap score between the two connected
        memory nodes.  This is intentionally best-first rather than plain BFS
        so a strong two-hop path can outrank weak same-depth neighbors.
        """
        if depth <= 0 or not seed_ids:
            return []

        ranked: List[int] = []
        seed_set = set(seed_ids)
        visited = set(seed_ids)
        best_scores: Dict[int, float] = {}
        frontier: List[Tuple[float, int, int, int]] = []
        order = 0

        for rank, seed_id in enumerate(seed_ids, 1):
            seed_score = 1.0 / rank
            for neighbor_id, edge_score in self._memory_graph_neighbors(seed_id, allowed_ids=allowed_ids):
                if neighbor_id in seed_set:
                    continue
                candidate_score = self._memory_graph_path_score(seed_score, edge_score, 1)
                if candidate_score <= best_scores.get(neighbor_id, -1.0):
                    continue
                best_scores[neighbor_id] = candidate_score
                heapq.heappush(frontier, (-candidate_score, 1, order, neighbor_id))
                order += 1

        beam_size = max(limit * 2, len(seed_ids) * 4, 8)
        while frontier and len(ranked) < limit:
            neg_score, node_depth, _, node_id = heapq.heappop(frontier)
            score = -neg_score
            if node_id in visited:
                continue
            visited.add(node_id)
            if allowed_ids is None or node_id in allowed_ids:
                ranked.append(node_id)

            if node_depth >= depth:
                continue
            for neighbor_id, edge_score in self._memory_graph_neighbors(node_id, allowed_ids=allowed_ids):
                if neighbor_id in visited or neighbor_id in seed_set:
                    continue
                next_depth = node_depth + 1
                candidate_score = self._memory_graph_path_score(score, edge_score, next_depth)
                if candidate_score <= best_scores.get(neighbor_id, -1.0):
                    continue
                best_scores[neighbor_id] = candidate_score
                heapq.heappush(frontier, (-candidate_score, next_depth, order, neighbor_id))
                order += 1

            if len(frontier) > beam_size:
                frontier = heapq.nsmallest(beam_size, frontier)
                heapq.heapify(frontier)

        return ranked

    @staticmethod
    def _memory_graph_path_score(current_score: float, edge_score: float, depth: int) -> float:
        """Blend inherited path relevance with the next edge score."""
        return max(0.0, 0.60 * current_score + 0.40 * edge_score - 0.06 * depth)

    @staticmethod
    def _memory_relation_edge_score(
        *,
        confidence: Any = None,
        semantic_score: Any = 0.0,
        causal_score: Any = 0.0,
        temporal_score: Any = 0.0,
        entity_score: Any = 0.0,
        weight: Any = None,
    ) -> float:
        """Normalize stored edge components into one traversal score."""
        def _f(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        stored_weight = _f(weight, -1.0)
        if stored_weight > 0:
            return max(0.0, min(1.0, stored_weight))

        confidence_value = _f(confidence, 1.0)
        semantic_value = _f(semantic_score)
        causal_value = _f(causal_score)
        temporal_value = _f(temporal_score)
        entity_value = _f(entity_score)
        combined = (
            0.30 * semantic_value
            + 0.35 * causal_value
            + 0.20 * temporal_value
            + 0.15 * entity_value
        )
        if combined <= 0:
            combined = confidence_value
        return max(0.0, min(1.0, combined))

    def _memory_graph_neighbors(
        self,
        node_id: int,
        *,
        allowed_ids: Optional[set] = None,
    ) -> List[Tuple[int, float]]:
        """Return graph neighbors ranked by edge strength."""
        scores: Dict[int, float] = {}

        rel_cursor = self._conn.execute(
            "SELECT target_node_id, confidence, semantic_score, causal_score, "
            "temporal_score, entity_score, weight "
            "FROM memory_node_relations WHERE source_node_id = ? "
            "UNION ALL "
            "SELECT source_node_id, confidence, semantic_score, causal_score, "
            "temporal_score, entity_score, weight "
            "FROM memory_node_relations WHERE target_node_id = ?",
            (node_id, node_id),
        )
        for row in rel_cursor.fetchall():
            neighbor_id = row[0]
            if allowed_ids is not None and neighbor_id not in allowed_ids:
                continue
            entity_score = row[5]
            try:
                entity_value = float(entity_score or 0.0)
            except (TypeError, ValueError):
                entity_value = 0.0
            if entity_value <= 0.0:
                entity_score = self._memory_relation_entity_score(node_id, neighbor_id)
            edge_score = self._memory_relation_edge_score(
                confidence=row[1],
                semantic_score=row[2],
                causal_score=row[3],
                temporal_score=row[4],
                entity_score=entity_score,
                weight=row[6],
            )
            scores[neighbor_id] = max(scores.get(neighbor_id, 0.0), edge_score)

        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    def _memory_entity_overlap_ranked(
        self,
        node_id: int,
        *,
        allowed_ids: Optional[set] = None,
        limit: int = 20,
    ) -> List[int]:
        """Rank prior nodes that mention entities from *node_id*."""
        ent_cursor = self._conn.execute(
            "SELECT entity_id FROM memory_node_entities WHERE node_id = ?",
            (node_id,),
        )
        entity_ids = [row[0] for row in ent_cursor.fetchall()]
        if not entity_ids:
            return []
        placeholders = ",".join("?" for _ in entity_ids)
        cursor = self._conn.execute(
            "SELECT node_id, COUNT(*) AS overlap_count "
            "FROM memory_node_entities "
            f"WHERE entity_id IN ({placeholders}) AND node_id != ? "
            "GROUP BY node_id "
            "ORDER BY overlap_count DESC, node_id DESC "
            "LIMIT ?",
            tuple(entity_ids) + (node_id, limit),
        )
        return self._memory_filter_ranked_ids(
            [row[0] for row in cursor.fetchall()],
            allowed_ids=allowed_ids,
        )

    def _memory_temporal_near_ranked(
        self,
        node_id: int,
        *,
        allowed_ids: Optional[set] = None,
        limit: int = 20,
    ) -> List[int]:
        """Rank nearest prior memory nodes by timestamp."""
        row = self._conn.execute(
            "SELECT time_key FROM memory_nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
        if not row:
            return []
        cursor = self._conn.execute(
            "SELECT id FROM memory_nodes "
            "WHERE id != ? AND time_key <= ? "
            "ORDER BY time_key DESC, id DESC "
            "LIMIT ?",
            (node_id, row[0], limit),
        )
        return self._memory_filter_ranked_ids(
            [r[0] for r in cursor.fetchall()],
            allowed_ids=allowed_ids,
        )

    def memory_prior_node_ids(
        self,
        node_id: int,
        *,
        same_day: bool = False,
    ) -> List[int]:
        """Return prior memory node ids for relation graph construction."""
        current = self._memory_get_node(node_id)
        if not current:
            return []
        current_time = str(current.get("time_key", ""))
        params: List[Any] = [node_id, current_time]
        where = "id != ? AND time_key <= ?"
        if same_day and len(current_time) >= 10:
            where += " AND substr(time_key, 1, 10) = ?"
            params.append(current_time[:10])
        cursor = self._conn.execute(
            f"SELECT id FROM memory_nodes WHERE {where} ORDER BY time_key DESC, id DESC",
            params,
        )
        return [row[0] for row in cursor.fetchall()]

    # ── Public API ────────────────────────────────────────────────────────

    def memory_add_node(
        self,
        time_key: str,
        summary: str,
        keywords: List[str],
        topic: List[str],
        original_dialog: str,
        query_embedding: np.ndarray,
        tags: Optional[List[str]] = None,
        fact_type: str = "semantic",
        fact_subject: str = "other",
        fact_kind: str = "other",
        task_event_like: Optional[bool] = None,
        task_event_subject: str = "",
        task_relevance: str = "",
        entity_names: Optional[List[str]] = None,
        primary_entity_id: Optional[int] = None,
        primary_topic: Optional[str] = None,
    ) -> int:
        """Insert a new memory node (SQLite + FAISS). Returns the node ID.

        *time_key* must be a unique timestamp string (e.g. ``"2026-04-20 10:00"``).
        *query_embedding* should be a (1, EMBEDDING_DIM) float32 numpy array.
        """

        def _do(conn):
            keywords_str = " ".join(keywords) if isinstance(keywords, list) else keywords
            topic_str = " ".join(topic) if isinstance(topic, list) else topic
            tags_str = json.dumps(tags or [], ensure_ascii=False)
            entity_names_str = json.dumps(entity_names or [], ensure_ascii=False)
            normalized_fact_type = self._normalize_memory_fact_type(
                fact_type or self._memory_fact_type_from_tags(tags or [])
            )
            normalized_fact_kind = self._normalize_memory_fact_kind(
                fact_kind or self._memory_fact_kind_from_tags(tags or [])
            )
            normalized_fact_subject = self._normalize_memory_fact_subject(
                fact_subject or self._memory_fact_subject_from_tags(tags or [])
            )
            if task_event_like is None:
                task_event_like_value = None
            else:
                task_event_like_value = 1 if bool(task_event_like) else 0
            task_event_subject_value = str(task_event_subject or "").strip().lower()
            if task_event_subject_value not in {"user", "assistant", "both", "other"}:
                task_event_subject_value = ""
            task_relevance_value = str(task_relevance or "").strip().lower()
            if task_relevance_value not in {"none", "weak", "medium", "strong"}:
                task_relevance_value = ""
            if primary_topic is None:
                if isinstance(topic, list):
                    primary_topic_value = str(topic[0] if topic else "").strip()
                else:
                    primary_topic_value = str(topic or "").strip().split(" ", 1)[0]
            else:
                primary_topic_value = str(primary_topic or "").strip()
            primary_topic_value = primary_topic_value or "general"
            cursor = conn.execute(
                """INSERT INTO memory_nodes
                   (time_key, summary, keywords, topic, tags, fact_type, fact_subject, fact_kind,
                    task_event_like, task_event_subject, task_relevance, entity_names,
                    primary_entity_id, primary_topic, original_dialog)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time_key,
                    summary,
                    keywords_str,
                    topic_str,
                    tags_str,
                    normalized_fact_type,
                    normalized_fact_subject,
                    normalized_fact_kind,
                    task_event_like_value,
                    task_event_subject_value,
                    task_relevance_value,
                    entity_names_str,
                    primary_entity_id,
                    primary_topic_value,
                    original_dialog,
                ),
            )
            node_id = cursor.lastrowid
            return node_id

        node_id = self._execute_write(_do)

        # Insert FAISS vector (outside the write transaction)
        if _HAS_FAISS and self._memory_faiss_index is not None:
            with self._lock:
                self._memory_faiss_index.add(query_embedding)
                self._memory_faiss_id_map.append(node_id)
            self._memory_save_faiss()

        return node_id
        
    def memory_relation_candidates(
        self,
        node_id: int,
        query_embedding: np.ndarray,
        keywords: Optional[List[str]] = None,
        top_k: int = None,
        budget: str = "mid",
    ) -> tuple[List[Dict[str, Any]], List[int]]:
        """Find candidate prior nodes for cross-fact causal relation extraction.

        Uses HindSight-style multi-signal candidate generation instead of only
        vector cosine similarity:
        - semantic FAISS neighbors
        - keyword / BM25 matches from fact keywords
        - entity overlap
        - nearby prior memories in time
        - graph neighbors from explicit node relations and entity edges

        The final candidate order is fused with Reciprocal Rank Fusion, then
        the LLM relation classifier decides whether a real relation exists.
        """
        if top_k is None:
            top_k = self.MEMORY_TOP_K_CAUSAL

        current = self._memory_get_node(node_id)
        if not current:
            return [], []

        search_limit = max(top_k * 4, 20)
        current_time = current.get("time_key", "")
        current_fact_type = self._normalize_memory_fact_type(current.get("fact_type", "semantic"))
        allowed_rows = self._conn.execute(
            "SELECT id FROM memory_nodes WHERE id != ? AND time_key <= ? AND fact_type = ?",
            (node_id, current_time, current_fact_type),
        ).fetchall()
        allowed_ids = {row[0] for row in allowed_rows}
        if not allowed_ids:
            return [], []

        keyword_query = " OR ".join(keywords or current.get("keywords", []))
        keyword_scores = (
            self._search_memory_keyword(keyword_query, limit=search_limit, allowed_ids=allowed_ids)
            if keyword_query else {}
        )
        keyword_ranking = self._memory_filter_ranked_ids(
            self._memory_rank_scores(keyword_scores, higher_is_better=False),
            allowed_ids=allowed_ids,
        )

        semantic_scores = self._search_memory_vector(
            query_embedding,
            top_k=search_limit,
            allowed_ids=allowed_ids,
        )
        semantic_ranking = self._memory_filter_ranked_ids(
            self._memory_rank_scores(semantic_scores, higher_is_better=True),
            allowed_ids=allowed_ids,
        )

        entity_ranking = self._memory_entity_overlap_ranked(
            node_id,
            allowed_ids=allowed_ids,
            limit=search_limit,
        )
        temporal_ranking = self._memory_temporal_near_ranked(
            node_id,
            allowed_ids=allowed_ids,
            limit=search_limit,
        )

        seed_ids = self._memory_filter_ranked_ids(
            semantic_ranking[:top_k] + keyword_ranking[:top_k] + entity_ranking[:top_k] + temporal_ranking[:top_k],
            allowed_ids=allowed_ids,
        )
        graph_depth = {"low": 0, "mid": 1, "high": 3}.get(budget, 1)
        graph_ranking = self._memory_graph_expand_ranked(
            seed_ids,
            depth=graph_depth,
            allowed_ids=allowed_ids,
            limit=search_limit,
        )

        rankings: List[Tuple[List[int], float]] = []
        if semantic_ranking:
            rankings.append((semantic_ranking, 1.0))
        if keyword_ranking:
            rankings.append((keyword_ranking, 1.0))
        if entity_ranking:
            rankings.append((entity_ranking, 1.1))
        if temporal_ranking:
            rankings.append((temporal_ranking, 0.8))
        if graph_ranking:
            rankings.append((graph_ranking, 0.7))

        ranked_ids = self._memory_rrf(rankings)[:top_k] if rankings else []
        nodes: List[Dict[str, Any]] = []
        node_ids: List[int] = []
        for candidate_id in ranked_ids:
            node = self._memory_get_node(candidate_id)
            if not node:
                continue
            nodes.append(node)
            node_ids.append(candidate_id)
        return nodes, node_ids

    def search_memory_facts(
        self, keyword: str, query_embedding: np.ndarray, top_k: int = None,
        budget: str = "mid",
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        tags: Optional[List[str]] = None,
        tags_match: str = "any",
        fact_types: Optional[List[str]] = None,
        include_graph: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search raw fact memories with keyword, vector, and temporal signals.

        *keyword* is passed to FTS5 MATCH (use ``" OR "``-joined terms).
        *query_embedding* is a (1, EMBEDDING_DIM) float32 numpy array.
        *top_k* overrides ``MEMORY_QUERY_TOP_K`` (default).
        *budget* controls graph traversal depth only when ``include_graph`` is true.
        *time_start*, *time_end*: optional ISO timestamp strings (``"2026-04-20 10:00:00"``).
            When specified, time range is the PRIMARY filter — all nodes in range
            are candidates, and the pool is padded with newest nodes if semantic
            search returns too few results.
        *tags*: optional list of tags to filter by (matches against JSON array in ``tags`` column).
        *tags_match*: ``"any"`` (default, node has at least one) or ``"all"`` (node has all).
        *fact_types*: optional list of normalized fact buckets (``semantic``/``episodic``).
        *include_graph*: when true, expand from direct matches through memory-node
            relations. Defaults to false so fact recall stays evidence-focused in
            the fact/observation/interpretation memory architecture.
        """
        if top_k is None:
            top_k = self.MEMORY_QUERY_TOP_K

        search_limit = max(top_k * 4, 20)
        keyword_query = " OR ".join(keyword) if isinstance(keyword, list) else str(keyword or "")

        # ── Step 1: Primary filters ──
        _time_ids: Optional[set] = None
        _temporal_ranking: List[int] = []
        if time_start is not None or time_end is not None:
            _time_conditions = []
            _time_params = []
            if time_start is not None:
                _time_conditions.append("time_key >= ?")
                _time_params.append(time_start)
            if time_end is not None:
                _time_conditions.append("time_key <= ?")
                _time_params.append(time_end)
            _time_sql = " AND ".join(_time_conditions)
            _tc = self._conn.execute(
                "SELECT id FROM memory_nodes WHERE {} ORDER BY time_key DESC".format(_time_sql),
                _time_params,
            )
            _temporal_ranking = [r[0] for r in _tc.fetchall()] # latest nodes will have larger weight
            _time_ids = set(_temporal_ranking)
            if not _time_ids:
                return []  # No nodes in the requested time range
            # Actual top-k is at most the number of nodes in time range
            top_k = min(top_k, len(_time_ids))

        _tag_ids: Optional[set] = None
        if tags:
            _tag_conditions = []
            _tag_params = []
            for _t in tags:
                _tag_conditions.append("tags LIKE ?")
                _tag_params.append(f'%"{_t}"%')
            _tag_connector = " OR " if tags_match == "any" else " AND "
            _tag_sql = _tag_connector.join(_tag_conditions)
            _tag_cursor = self._conn.execute(
                "SELECT id FROM memory_nodes WHERE {}".format(_tag_sql),
                _tag_params,
            )
            _tag_ids = {r[0] for r in _tag_cursor.fetchall()}
            if not _tag_ids:
                return []

        _fact_type_ids: Optional[set] = None
        if fact_types:
            normalized_types = sorted({
                self._normalize_memory_fact_type(fact_type)
                for fact_type in fact_types
                if str(fact_type or "").strip()
            })
            if normalized_types:
                placeholders = ",".join("?" for _ in normalized_types)
                _type_cursor = self._conn.execute(
                    f"SELECT id FROM memory_nodes WHERE fact_type IN ({placeholders})",
                    normalized_types,
                )
                _fact_type_ids = {r[0] for r in _type_cursor.fetchall()}
                if not _fact_type_ids:
                    return []

        allowed_ids: Optional[set] = None
        if _time_ids is not None:
            allowed_ids = set(_time_ids)
        if _tag_ids is not None:
            allowed_ids = _tag_ids if allowed_ids is None else allowed_ids & _tag_ids
            if not allowed_ids:
                return []
        if _fact_type_ids is not None:
            allowed_ids = _fact_type_ids if allowed_ids is None else allowed_ids & _fact_type_ids
            if not allowed_ids:
                return []

        # ── Step 2: Independent retrieval channels ──
        keyword_allowed_ids = allowed_ids
        if _time_ids is not None and _tag_ids is None and _fact_type_ids is None:
            # Time-only searches are cheaper as range predicates than as a
            # potentially huge IN-list.  The vector channel still receives
            # allowed_ids because FAISS has no native time predicate.
            keyword_allowed_ids = None
        fts_results = (
            self._search_memory_keyword(
                keyword_query,
                limit=search_limit,
                allowed_ids=keyword_allowed_ids,
                time_start=time_start,
                time_end=time_end,
            )
            if keyword_query else {}
        )
        vec_results = self._search_memory_vector(
            query_embedding,
            top_k=search_limit,
            allowed_ids=allowed_ids,
        )

        keyword_ranking = self._memory_filter_ranked_ids(
            self._memory_rank_scores(fts_results, higher_is_better=False),
            allowed_ids=allowed_ids,
        )
        semantic_ranking = self._memory_filter_ranked_ids(
            self._memory_rank_scores(vec_results, higher_is_better=True),
            allowed_ids=allowed_ids,
        )
        temporal_ranking = self._memory_filter_ranked_ids(
            _temporal_ranking,
            allowed_ids=allowed_ids,
        )

        seed_ids = self._memory_filter_ranked_ids(
            semantic_ranking[:top_k] + keyword_ranking[:top_k] + temporal_ranking[:top_k],
            allowed_ids=allowed_ids,
        )

        graph_ranking: List[int] = []
        if include_graph:
            _graph_depth = {"low": 0, "mid": 1, "high": 3}.get(budget, 1)
            graph_ranking = self._memory_graph_expand_ranked(
                seed_ids,
                depth=_graph_depth,
                allowed_ids=allowed_ids,
                limit=search_limit,
            )

        # ── Step 3: Reciprocal Rank Fusion ──
        rankings: List[Tuple[List[int], float]] = []
        if semantic_ranking:
            rankings.append((semantic_ranking, 1.0))
        if keyword_ranking:
            rankings.append((keyword_ranking, 1.0))
        if temporal_ranking:
            rankings.append((temporal_ranking, 0.9))
        if graph_ranking:
            rankings.append((graph_ranking, 0.45))

        rrf_scores, first_seen = self._memory_rrf_scores(rankings) if rankings else ({}, {})
        ranked_ids = sorted(
            rrf_scores,
            key=lambda node_id: (-rrf_scores[node_id], first_seen.get(node_id, 0)),
        ) if rrf_scores else []

        # Time-filtered recall should still return memories in the requested
        # interval even when semantic/keyword channels are sparse.
        if _time_ids is not None and len(ranked_ids) < top_k:
            existing = set(ranked_ids)
            for node_id in temporal_ranking:
                if node_id in existing:
                    continue
                ranked_ids.append(node_id)
                existing.add(node_id)
                if len(ranked_ids) >= top_k:
                    break

        ranked_ids = self._memory_decay_rerank(
            ranked_ids,
            rrf_scores,
            first_seen,
            limit=max(top_k * 3, top_k),
        )

        nodes: List[Dict[str, Any]] = []
        for node_id in ranked_ids[:top_k]:
            node = self._memory_get_node(node_id)
            if node:
                node["embedding_similarity"] = float(vec_results.get(node_id, 0.0) or 0.0)
                node["keyword_score"] = (
                    None if node_id not in fts_results else float(fts_results[node_id])
                )
                node["temporal_rank"] = (
                    temporal_ranking.index(node_id) + 1
                    if node_id in temporal_ranking else None
                )
                node["retrieval_score"] = float(rrf_scores.get(node_id, 0.0) or 0.0)
                if graph_ranking:
                    node["graph_rank"] = (
                        graph_ranking.index(node_id) + 1
                        if node_id in graph_ranking else None
                    )
                nodes.append(node)
        return nodes

    # ── Entity / Knowledge Graph methods ──────────────────────────────────

    def entity_add_entity(
        self, name: str, entity_type: str = "CONCEPT",
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add or retrieve an entity node. Returns entity ID."""
        def _do(conn):
            emb_blob = embedding.tobytes() if embedding is not None else None
            meta_str = json.dumps(metadata or {}, ensure_ascii=False)
            created_at = time.time()
            cursor = conn.execute(
                "INSERT OR IGNORE INTO entity_nodes (name, type, embedding, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, entity_type, emb_blob, meta_str, created_at),
            )
            if cursor.rowcount:
                return cursor.lastrowid
            # Already exists — fetch existing id
            return conn.execute(
                "SELECT id FROM entity_nodes WHERE name = ?", (name,)
            ).fetchone()[0]
        return self._execute_write(_do)

    def entity_add_edge(
        self, source_entity_id: int, target_entity_id: int,
        relation_type: str, weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add a directed edge between two entities. Returns edge id."""
        meta_str = json.dumps(metadata or {}, ensure_ascii=False)
        def _do(conn):
            created_at = time.time()
            cursor = conn.execute(
                "INSERT OR IGNORE INTO entity_edges "
                "(source_entity_id, target_entity_id, relation_type, weight, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (source_entity_id, target_entity_id, relation_type, weight, meta_str, created_at),
            )
            if cursor.rowcount:
                return cursor.lastrowid
            r = conn.execute(
                "SELECT id FROM entity_edges WHERE source_entity_id=? AND "
                "target_entity_id=? AND relation_type=?",
                (source_entity_id, target_entity_id, relation_type),
            ).fetchone()
            return r[0] if r else -1
        return self._execute_write(_do)

    @staticmethod
    def _entity_load_co_entities(raw_value: Any) -> Dict[str, Dict[str, Any]]:
        try:
            data = json.loads(raw_value or "{}")
        except (TypeError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    @classmethod
    def _entity_record_co_entity(
        cls,
        conn: sqlite3.Connection,
        *,
        entity_id: int,
        other_id: int,
        other_name: str,
        other_type: str,
        count: int = 1,
    ) -> None:
        if entity_id == other_id:
            return
        row = conn.execute(
            "SELECT co_entities FROM entity_nodes WHERE id = ?",
            (entity_id,),
        ).fetchone()
        if not row:
            return
        co_entities = cls._entity_load_co_entities(row["co_entities"])
        key = str(other_id)
        current = co_entities.get(key, {})
        try:
            current_count = int(current.get("count", 0) or 0)
        except (TypeError, ValueError):
            current_count = 0
        co_entities[key] = {
            "name": other_name,
            "type": other_type,
            "count": current_count + max(1, int(count or 1)),
        }
        conn.execute(
            "UPDATE entity_nodes SET co_entities = ? WHERE id = ?",
            (json.dumps(co_entities, ensure_ascii=False, sort_keys=True), entity_id),
        )

    def entity_link_node(self, node_id: int, entity_id: int, mention_count: int = 1) -> None:
        """Link a memory node to an entity (upsert)."""
        def _do(conn):
            cursor = conn.execute(
                "INSERT OR IGNORE INTO memory_node_entities (node_id, entity_id, mention_count) "
                "VALUES (?, ?, ?)",
                (node_id, entity_id, mention_count),
            )
            if not cursor.rowcount:
                return
            rows = conn.execute(
                "SELECT en.id, en.name, en.type "
                "FROM memory_node_entities mne "
                "JOIN entity_nodes en ON en.id = mne.entity_id "
                "WHERE mne.node_id = ?",
                (node_id,),
            ).fetchall()
            by_id = {int(row["id"]): row for row in rows}
            linked = by_id.get(int(entity_id))
            if not linked:
                return
            for other_id, other in by_id.items():
                if other_id == entity_id:
                    continue
                self._entity_record_co_entity(
                    conn,
                    entity_id=entity_id,
                    other_id=other_id,
                    other_name=other["name"],
                    other_type=other["type"],
                    count=mention_count,
                )
                self._entity_record_co_entity(
                    conn,
                    entity_id=other_id,
                    other_id=entity_id,
                    other_name=linked["name"],
                    other_type=linked["type"],
                    count=mention_count,
                )
        self._execute_write(_do)

    @staticmethod
    def _entity_normalized_name(name: str) -> str:
        text = unicodedata.normalize("NFKC", str(name or "")).casefold().strip()
        return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)

    @staticmethod
    def _entity_name_tokens(name: str) -> List[str]:
        return [
            token
            for token in re.split(r"[\s\-_/.,:;(){}\[\]\"']+", unicodedata.normalize("NFKC", str(name or "")).casefold())
            if token
        ]

    @classmethod
    def _entity_name_similarity(cls, left_name: str, right_name: str) -> Tuple[float, str, str]:
        left_norm = cls._entity_normalized_name(left_name)
        right_norm = cls._entity_normalized_name(right_name)
        if not left_norm or not right_norm:
            return 0.0, "empty_name", "high"
        if left_norm == right_norm:
            return 1.0, "normalized_name_match", "low"

        left_tokens = set(cls._entity_name_tokens(left_name))
        right_tokens = set(cls._entity_name_tokens(right_name))
        if left_tokens and right_tokens and (left_tokens <= right_tokens or right_tokens <= left_tokens):
            return 0.82, "token_subset", "medium"

        shorter, longer = sorted((left_norm, right_norm), key=len)
        if len(shorter) >= 2 and shorter in longer:
            return 0.76, "name_substring", "medium"

        return 0.0, "name_mismatch", "high"

    @staticmethod
    def _entity_type_compatible(left_type: str, right_type: str) -> Tuple[bool, float]:
        left = str(left_type or "CONCEPT").upper()
        right = str(right_type or "CONCEPT").upper()
        if left == right:
            return True, 1.0
        weak_pairs = {
            frozenset(("ORG", "PRODUCT")),
            frozenset(("ORGANIZATION", "PRODUCT")),
            frozenset(("CONCEPT", "PRODUCT")),
            frozenset(("CONCEPT", "ORG")),
            frozenset(("CONCEPT", "ORGANIZATION")),
        }
        if frozenset((left, right)) in weak_pairs:
            return True, 0.7
        return False, 0.0

    @classmethod
    def _entity_cooccurrence_similarity(cls, left_raw: Any, right_raw: Any) -> float:
        left = cls._entity_load_co_entities(left_raw)
        right = cls._entity_load_co_entities(right_raw)
        if not left or not right:
            return 0.0
        left_ids = set(left)
        right_ids = set(right)
        union = left_ids | right_ids
        if not union:
            return 0.0
        return len(left_ids & right_ids) / len(union)

    @staticmethod
    def _entity_merging_action(
        *,
        confidence: float,
        reason: str,
        risk: str,
        type_compatible: bool,
    ) -> str:
        if not type_compatible:
            return "skip"
        if reason == "normalized_name_match" and risk == "low":
            return "merge"
        return "candidate"

    def _reflection_entity_merging_candidate_for_pair(
        self,
        left: sqlite3.Row,
        right: sqlite3.Row,
    ) -> Optional[Dict[str, Any]]:
        type_compatible, type_score = self._entity_type_compatible(left["type"], right["type"])
        if type_score <= 0.7:
            return None
        name_score, reason, risk = self._entity_name_similarity(left["name"], right["name"])
        if name_score <= 0:
            return None
        co_score = self._entity_cooccurrence_similarity(left["co_entities"], right["co_entities"])
        confidence = max(0.0, min(1.0, (name_score * 0.72) + (type_score * 0.20) + (co_score * 0.08)))
        action = self._entity_merging_action(
            confidence=confidence,
            reason=reason,
            risk=risk,
            type_compatible=type_compatible,
        )
        if action == "skip":
            return None
        canonical, duplicate = self._entity_choose_canonical(left, right)
        return {
            "canonical_id": int(canonical["id"]),
            "canonical_name": canonical["name"],
            "duplicate_id": int(duplicate["id"]),
            "duplicate_name": duplicate["name"],
            "type_compatible": type_compatible,
            "type_score": round(type_score, 4),
            "name_score": round(name_score, 4),
            "co_entities_score": round(co_score, 4),
            "confidence": round(confidence, 4),
            "reason": reason,
            "risk": risk,
            "action": action,
        }

    def _reflection_entity_merging_candidates(
        self,
        limit: int = 100,
        anchor_entity_ids: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, name, type, co_entities FROM entity_nodes ORDER BY id"
        ).fetchall()
        anchor_ids: Optional[set[int]] = None
        if anchor_entity_ids is not None:
            anchor_ids = {
                int(entity_id)
                for entity_id in anchor_entity_ids
                if str(entity_id or "").strip()
            }
            if not anchor_ids:
                return []

        candidates: List[Dict[str, Any]] = []
        if anchor_ids is None:
            pair_iter = (
                (left, right)
                for i, left in enumerate(rows)
                for right in rows[i + 1:]
            )
        else:
            anchor_rows = [row for row in rows if int(row["id"]) in anchor_ids]
            seen_pairs: set[Tuple[int, int]] = set()

            def _anchored_pairs():
                for left in anchor_rows:
                    left_id = int(left["id"])
                    for right in rows:
                        right_id = int(right["id"])
                        if left_id == right_id:
                            continue
                        pair_key = tuple(sorted((left_id, right_id)))
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)
                        yield left, right

            pair_iter = _anchored_pairs()

        for left, right in pair_iter:
            candidate = self._reflection_entity_merging_candidate_for_pair(left, right)
            if candidate is not None:
                candidates.append(candidate)
        candidates.sort(key=lambda item: (-item["confidence"], item["risk"], item["canonical_id"]))
        return candidates[:limit]

    @staticmethod
    def _entity_choose_canonical(left: sqlite3.Row, right: sqlite3.Row) -> Tuple[sqlite3.Row, sqlite3.Row]:
        left_name = str(left["name"] or "")
        right_name = str(right["name"] or "")
        left_display = left_name.strip()
        right_display = right_name.strip()
        left_norm = SessionDB._entity_normalized_name(left_name)
        right_norm = SessionDB._entity_normalized_name(right_name)
        if left_norm and left_norm == right_norm:
            if len(left_display) < len(right_display):
                return left, right
            if len(right_display) < len(left_display):
                return right, left
            return (left, right) if int(left["id"]) <= int(right["id"]) else (right, left)
        if len(right_name) > len(left_name):
            return right, left
        if len(left_name) > len(right_name):
            return left, right
        return (left, right) if int(left["id"]) <= int(right["id"]) else (right, left)

    def reflect_merging_entities(
        self,
        *,
        limit: int = 100,
        anchor_entity_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Reflect on the entity graph and merge high-confidence duplicate entities.

        Entity merge confidence is driven by normalized name similarity, gated
        by compatible entity types, and lightly adjusted by overlap in
        co-occurring entity profiles. Only low-risk normalized-name matches are
        merged automatically; other similar names are reported as candidates.
        """
        candidates = self._reflection_entity_merging_candidates(
            limit=limit,
            anchor_entity_ids=anchor_entity_ids,
        )
        merged: List[Dict[str, Any]] = []
        for candidate in candidates:
            if candidate["action"] != "merge":
                continue
            self.entity_merge(
                canonical_id=candidate["canonical_id"],
                duplicate_id=candidate["duplicate_id"],
                reason=candidate["reason"],
                confidence=float(candidate["confidence"]),
            )
            merged.append(candidate)
        
        return {
            "candidates": candidates,
            "merged": len(merged),
            "merge_candidates": sum(1 for candidate in candidates if candidate["action"] == "merge"),
            "candidate_count": len(candidates),
            "anchor_entity_count": (
                None
                if anchor_entity_ids is None
                else len({int(entity_id) for entity_id in anchor_entity_ids if str(entity_id or "").strip()})
            ),
            "rules": {
                "auto_merge": "same/compatible type + normalized name match",
                "candidate_only": "token subset or substring names, even with co-entity overlap",
                "score_weights": {"name": 0.72, "type": 0.20, "co_entities": 0.08},
            },
        }

    def entity_merge(
        self,
        *,
        canonical_id: int,
        duplicate_id: int,
        reason: str,
        confidence: float,
    ) -> None:
        """Merge a duplicate entity into a canonical entity."""
        if canonical_id == duplicate_id:
            return

        def _do(conn):
            canonical = conn.execute(
                "SELECT id, name, type, metadata, co_entities FROM entity_nodes WHERE id = ?",
                (canonical_id,),
            ).fetchone()
            duplicate = conn.execute(
                "SELECT id, name, type, metadata, co_entities FROM entity_nodes WHERE id = ?",
                (duplicate_id,),
            ).fetchone()
            if not canonical or not duplicate:
                return

            canonical_meta = json.loads(canonical["metadata"] or "{}")
            aliases = canonical_meta.get("aliases", [])
            if not isinstance(aliases, list):
                aliases = []
            for alias in (duplicate["name"], *(json.loads(duplicate["metadata"] or "{}").get("aliases", []) or [])):
                if alias and alias not in aliases and alias != canonical["name"]:
                    aliases.append(alias)
            canonical_meta["aliases"] = aliases
            merge_history = canonical_meta.get("merge_history", [])
            if not isinstance(merge_history, list):
                merge_history = []
            merge_history.append({
                "merged_entity_id": duplicate_id,
                "merged_entity_name": duplicate["name"],
                "reason": reason,
                "confidence": confidence,
                "merged_at": datetime.now().astimezone().isoformat(),
            })
            canonical_meta["merge_history"] = merge_history

            duplicate_co = self._entity_load_co_entities(duplicate["co_entities"])
            canonical_co = self._entity_load_co_entities(canonical["co_entities"])
            duplicate_key = str(duplicate_id)
            canonical_key = str(canonical_id)
            canonical_co.pop(duplicate_key, None)
            for other_key, other_value in duplicate_co.items():
                if other_key in {duplicate_key, canonical_key}:
                    continue
                current = canonical_co.get(other_key, {})
                current_count = int(current.get("count", 0) or 0) if isinstance(current, dict) else 0
                other_count = int(other_value.get("count", 0) or 0) if isinstance(other_value, dict) else 0
                canonical_co[other_key] = {
                    "name": other_value.get("name", ""),
                    "type": other_value.get("type", "CONCEPT"),
                    "count": current_count + other_count,
                }

            conn.execute(
                "UPDATE entity_nodes SET metadata = ?, co_entities = ? WHERE id = ?",
                (
                    json.dumps(canonical_meta, ensure_ascii=False, sort_keys=True),
                    json.dumps(canonical_co, ensure_ascii=False, sort_keys=True),
                    canonical_id,
                ),
            )
            for row in conn.execute("SELECT id, co_entities FROM entity_nodes").fetchall():
                entity_id = int(row["id"])
                if entity_id == duplicate_id:
                    continue
                co_entities = self._entity_load_co_entities(row["co_entities"])
                duplicate_entry = co_entities.pop(duplicate_key, None)
                if duplicate_entry:
                    existing = co_entities.get(canonical_key, {})
                    existing_count = int(existing.get("count", 0) or 0) if isinstance(existing, dict) else 0
                    duplicate_count = (
                        int(duplicate_entry.get("count", 0) or 0)
                        if isinstance(duplicate_entry, dict)
                        else 0
                    )
                    co_entities[canonical_key] = {
                        "name": canonical["name"],
                        "type": canonical["type"],
                        "count": existing_count + duplicate_count,
                    }
                    conn.execute(
                        "UPDATE entity_nodes SET co_entities = ? WHERE id = ?",
                        (json.dumps(co_entities, ensure_ascii=False, sort_keys=True), entity_id),
                    )
            conn.execute(
                "UPDATE OR IGNORE memory_node_entities SET entity_id = ? WHERE entity_id = ?",
                (canonical_id, duplicate_id),
            )
            conn.execute("DELETE FROM memory_node_entities WHERE entity_id = ?", (duplicate_id,))
            conn.execute(
                "UPDATE memory_nodes SET primary_entity_id = ? WHERE primary_entity_id = ?",
                (canonical_id, duplicate_id),
            )
            conn.execute(
                "UPDATE OR IGNORE entity_edges SET source_entity_id = ? WHERE source_entity_id = ?",
                (canonical_id, duplicate_id),
            )
            conn.execute(
                "UPDATE OR IGNORE entity_edges SET target_entity_id = ? WHERE target_entity_id = ?",
                (canonical_id, duplicate_id),
            )
            conn.execute(
                "DELETE FROM entity_edges WHERE source_entity_id = target_entity_id "
                "OR source_entity_id = ? OR target_entity_id = ?",
                (duplicate_id, duplicate_id),
            )
            conn.execute(
                "UPDATE memory_observations SET entity_id = ? WHERE entity_id = ?",
                (canonical_id, duplicate_id),
            )
            conn.execute(
                "UPDATE memory_interpretations SET entity_id = ? WHERE entity_id = ?",
                (canonical_id, duplicate_id),
            )
            conn.execute("DELETE FROM entity_nodes WHERE id = ?", (duplicate_id,))

        self._execute_write(_do)

    # ── Normalized memory node relations ─────────────────────────────────

    def memory_add_node_relation(
        self, source_node_id: int, target_node_id: int,
        relation_type: str, confidence: float = 1.0,
        semantic_score: Optional[float] = None,
        causal_score: Optional[float] = None,
        temporal_score: Optional[float] = None,
        entity_score: Optional[float] = None,
        weight: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a normalized relation between two memory nodes."""
        relation_text = str(relation_type or "")
        relation_key = relation_text.strip().lower()
        confidence_value = max(0.0, min(1.0, float(confidence or 0.0)))

        if semantic_score is None:
            semantic_score = confidence_value if relation_key == "semantic" else 0.0
        if causal_score is None:
            causal_score = 0.0 if relation_key in {"semantic", "temporal"} else confidence_value
        if temporal_score is None:
            temporal_score = self._memory_relation_temporal_score(source_node_id, target_node_id)
            if relation_key != "temporal":
                temporal_score *= 0.5
        if entity_score is None:
            entity_score = self._memory_relation_entity_score(source_node_id, target_node_id)

        semantic_value = max(0.0, min(1.0, float(semantic_score or 0.0)))
        causal_value = max(0.0, min(1.0, float(causal_score or 0.0)))
        temporal_value = max(0.0, min(1.0, float(temporal_score or 0.0)))
        entity_value = max(0.0, min(1.0, float(entity_score or 0.0)))
        weight_value = (
            max(0.0, min(1.0, float(weight)))
            if weight is not None
            else self._memory_relation_weight_for_type(
                relation_key=relation_key,
                confidence=confidence_value,
                semantic_score=semantic_value,
                causal_score=causal_value,
                temporal_score=temporal_value,
                entity_score=entity_value,
            )
        )
        metadata_str = json.dumps(metadata or {}, ensure_ascii=False)

        def _do(conn):
            created_at = time.time()
            conn.execute(
                "INSERT OR IGNORE INTO memory_node_relations "
                "(source_node_id, target_node_id, relation_type, confidence, "
                "semantic_score, causal_score, temporal_score, entity_score, weight, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    source_node_id,
                    target_node_id,
                    relation_text,
                    confidence_value,
                    semantic_value,
                    causal_value,
                    temporal_value,
                    entity_value,
                    weight_value,
                    metadata_str,
                    created_at,
                ),
            )
        self._execute_write(_do)

    def _memory_relation_temporal_score(self, source_node_id: int, target_node_id: int) -> float:
        """Score two memory nodes by timestamp proximity."""
        rows = self._conn.execute(
            "SELECT id, time_key FROM memory_nodes WHERE id IN (?, ?)",
            (source_node_id, target_node_id),
        ).fetchall()
        by_id = {row[0]: row[1] for row in rows}
        source_time = self._parse_memory_time_key(by_id.get(source_node_id))
        target_time = self._parse_memory_time_key(by_id.get(target_node_id))
        if source_time is None or target_time is None:
            return 0.0
        delta_days = abs((source_time - target_time).total_seconds()) / 86400.0
        return max(0.0, min(1.0, math.exp(-delta_days / 30.0)))

    @staticmethod
    def _memory_relation_weight_for_type(
        *,
        relation_key: str,
        confidence: float,
        semantic_score: float,
        causal_score: float,
        temporal_score: float,
        entity_score: float,
    ) -> float:
        """Precompute traversal strength with relation-specific weights."""
        if relation_key == "semantic":
            value = 0.75 * semantic_score + 0.15 * temporal_score + 0.10 * entity_score
        elif relation_key == "temporal":
            value = 0.80 * temporal_score + 0.20 * entity_score
        else:
            value = (
                0.70 * causal_score
                + 0.15 * semantic_score
                + 0.10 * temporal_score
                + 0.05 * entity_score
            )
        if value <= 0:
            value = confidence
        return max(0.0, min(1.0, value))

    def _memory_relation_entity_score(self, source_node_id: int, target_node_id: int) -> float:
        """Score two memory nodes by normalized entity overlap."""
        cursor = self._conn.execute(
            "SELECT node_id, entity_id FROM memory_node_entities WHERE node_id IN (?, ?)",
            (source_node_id, target_node_id),
        )
        entities: Dict[int, set] = {}
        for row in cursor.fetchall():
            entities.setdefault(row[0], set()).add(row[1])
        source_entities = entities.get(source_node_id, set())
        target_entities = entities.get(target_node_id, set())
        if not source_entities or not target_entities:
            return 0.0
        return len(source_entities & target_entities) / len(source_entities | target_entities)

    @staticmethod
    def _parse_memory_time_key(value: Any) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text:
            return None
        for fmt, length in (
            ("%Y-%m-%d %H:%M:%S", 19),
            ("%Y-%m-%d %H:%M", 16),
            ("%Y-%m-%d", 10),
        ):
            try:
                return datetime.strptime(text[:length], fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    @staticmethod
    def _memory_decay_score(
        memory_time: Optional[datetime],
        *,
        now: datetime,
        half_life_days: float,
    ) -> float:
        """Return a query-time recency score using half-life decay."""
        if memory_time is None:
            return 1.0
        if memory_time.tzinfo is not None and now.tzinfo is None:
            memory_time = memory_time.replace(tzinfo=None)
        elif memory_time.tzinfo is None and now.tzinfo is not None:
            now = now.replace(tzinfo=None)
        age_days = max(0.0, (now - memory_time).total_seconds() / 86400.0)
        half_life = max(1.0, float(half_life_days or 1.0))
        return max(0.0, min(1.0, math.exp(-math.log(2.0) * age_days / half_life)))

    @staticmethod
    def _memory_topic_key(topic: Any) -> str:
        text = str(topic or "").strip().lower()
        text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "-", text).strip("-")
        return text or "general"

    @staticmethod
    def _embedding_to_blob(embedding: Optional[np.ndarray]) -> Optional[bytes]:
        if embedding is None:
            return None
        try:
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        if vector.size == 0:
            return None
        return vector.tobytes()

    @staticmethod
    def _embedding_similarity(
        query_embedding: Optional[np.ndarray],
        stored_embedding: Any,
    ) -> Optional[float]:
        if query_embedding is None or stored_embedding is None:
            return None
        try:
            query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
            if isinstance(stored_embedding, np.ndarray):
                candidate = np.asarray(stored_embedding, dtype=np.float32).reshape(-1)
            else:
                candidate = np.frombuffer(bytes(stored_embedding), dtype=np.float32)
        except (TypeError, ValueError):
            return None
        if query.size == 0 or candidate.size == 0 or query.shape != candidate.shape:
            return None
        denom = float(np.linalg.norm(query) * np.linalg.norm(candidate))
        if denom <= 0.0:
            return None
        score = float(np.dot(query, candidate) / denom)
        if not math.isfinite(score):
            return None
        return max(-1.0, min(1.0, score))

    # ── Memory interpretations ──────────────────────────────────────────

    def memory_upsert_interpretation(
        self,
        *,
        claim: str,
        entity_id: Optional[int] = None,
        subject_text: str = "",
        target_text: str = "",
        scope: str = "general",
        interpretation_type: str = "behavior_pattern",
        polarity: str = "neutral",
        strength: float = 0.5,
        confidence: float = 0.5,
        status: str = "current",
        conflict_status: str = "none",
        resolution: str = "",
        action_implication: str = "",
        evidence_node_ids: Optional[List[int]] = None,
        evidence_observation_ids: Optional[List[int]] = None,
        counter_evidence_node_ids: Optional[List[int]] = None,
        counter_evidence_observation_ids: Optional[List[int]] = None,
        embedding: Optional[np.ndarray] = None,
        embedding_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        interpretation_id: Optional[int] = None,
    ) -> int:
        """Create or update an agent interpretation over memory evidence.

        Interpretations are the mutable current-state layer: they should cite
        immutable fact nodes and/or consolidated observations, but they do not
        replace either evidence layer.
        """
        clean_claim = str(claim or "").strip()
        if not clean_claim:
            raise ValueError("memory interpretation requires a claim")
        now_text = datetime.now().astimezone().isoformat()
        normalized_type = self._normalize_memory_interpretation_type(interpretation_type)
        normalized_status = self._normalize_memory_interpretation_status(status)
        normalized_conflict = self._normalize_memory_interpretation_conflict_status(conflict_status)
        clean_subject = str(subject_text or "").strip()
        clean_target = str(target_text or "").strip()
        clean_scope = str(scope or "general").strip() or "general"
        clean_polarity = str(polarity or "neutral").strip().lower() or "neutral"
        strength_value = max(0.0, min(1.0, float(strength or 0.0)))
        confidence_value = max(0.0, min(1.0, float(confidence or 0.0)))
        evidence_nodes = None if evidence_node_ids is None else self._json_int_list(evidence_node_ids)
        evidence_observations = (
            None
            if evidence_observation_ids is None
            else self._json_int_list(evidence_observation_ids)
        )
        counter_nodes = (
            None
            if counter_evidence_node_ids is None
            else self._json_int_list(counter_evidence_node_ids)
        )
        counter_observations = (
            None
            if counter_evidence_observation_ids is None
            else self._json_int_list(counter_evidence_observation_ids)
        )
        metadata_dict = metadata or {}
        if not isinstance(metadata_dict, dict):
            metadata_dict = {}
        clean_entity_id = entity_id or self._coerce_int_or_none(metadata_dict.get("entity_id"))
        metadata_str = json.dumps(metadata_dict, ensure_ascii=False)
        embedding_blob = self._embedding_to_blob(embedding)
        clean_embedding_text = None if embedding_text is None else str(embedding_text or "").strip()

        def _do(conn):
            existing_id = interpretation_id
            if existing_id is None:
                existing = conn.execute(
                    "SELECT id FROM memory_interpretations "
                    "WHERE entity_id IS ? "
                    "AND subject_text = ? AND target_text = ? "
                    "AND scope = ? AND interpretation_type = ? "
                    "AND status IN ('current', 'conflicted') "
                    "ORDER BY updated_at DESC, id DESC LIMIT 1",
                    (
                        clean_entity_id,
                        clean_subject,
                        clean_target,
                        clean_scope,
                        normalized_type,
                    ),
                ).fetchone()
                if existing:
                    existing_id = existing["id"] if isinstance(existing, sqlite3.Row) else existing[0]
            if existing_id is not None:
                existing_row = conn.execute(
                    "SELECT evidence_node_ids, evidence_observation_ids, "
                    "counter_evidence_node_ids, counter_evidence_observation_ids, "
                    "embedding, embedding_text, embedding_updated_at "
                    "FROM memory_interpretations WHERE id = ?",
                    (existing_id,),
                ).fetchone()
                stored_evidence_nodes = (
                    self._json_int_list(existing_row["evidence_node_ids"])
                    if existing_row
                    else []
                )
                stored_evidence_observations = (
                    self._json_int_list(existing_row["evidence_observation_ids"])
                    if existing_row
                    else []
                )
                stored_counter_nodes = (
                    self._json_int_list(existing_row["counter_evidence_node_ids"])
                    if existing_row
                    else []
                )
                stored_counter_observations = (
                    self._json_int_list(existing_row["counter_evidence_observation_ids"])
                    if existing_row
                    else []
                )
                stored_embedding = existing_row["embedding"] if existing_row else None
                stored_embedding_text = existing_row["embedding_text"] if existing_row else ""
                stored_embedding_updated_at = existing_row["embedding_updated_at"] if existing_row else None
                next_embedding = embedding_blob if embedding_blob is not None else stored_embedding
                next_embedding_text = (
                    clean_embedding_text
                    if clean_embedding_text is not None
                    else (stored_embedding_text or "")
                )
                next_embedding_updated_at = (
                    now_text
                    if embedding_blob is not None
                    else stored_embedding_updated_at
                )
                conn.execute(
                    "UPDATE memory_interpretations SET "
                    "entity_id = ?, subject_text = ?, target_text = ?, "
                    "scope = ?, interpretation_type = ?, claim = ?, "
                    "polarity = ?, strength = ?, confidence = ?, status = ?, "
                    "conflict_status = ?, resolution = ?, action_implication = ?, "
                    "evidence_node_ids = ?, evidence_observation_ids = ?, "
                    "counter_evidence_node_ids = ?, counter_evidence_observation_ids = ?, "
                    "embedding = ?, embedding_text = ?, embedding_updated_at = ?, "
                    "metadata = ?, updated_at = ?, last_supported_at = ? "
                    "WHERE id = ?",
                    (
                        clean_entity_id,
                        clean_subject,
                        clean_target,
                        clean_scope,
                        normalized_type,
                        clean_claim,
                        clean_polarity,
                        strength_value,
                        confidence_value,
                        normalized_status,
                        normalized_conflict,
                        str(resolution or "").strip(),
                        str(action_implication or "").strip(),
                        json.dumps(evidence_nodes if evidence_nodes is not None else stored_evidence_nodes),
                        json.dumps(
                            evidence_observations
                            if evidence_observations is not None
                            else stored_evidence_observations
                        ),
                        json.dumps(counter_nodes if counter_nodes is not None else stored_counter_nodes),
                        json.dumps(
                            counter_observations
                            if counter_observations is not None
                            else stored_counter_observations
                        ),
                        next_embedding,
                        next_embedding_text,
                        next_embedding_updated_at,
                        metadata_str,
                        now_text,
                        now_text,
                        existing_id,
                    ),
                )
                return int(existing_id)
            cursor = conn.execute(
                "INSERT INTO memory_interpretations "
                "(entity_id, subject_text, target_text, "
                "scope, interpretation_type, claim, polarity, strength, confidence, "
                "status, conflict_status, resolution, action_implication, "
                "evidence_node_ids, evidence_observation_ids, "
                "counter_evidence_node_ids, counter_evidence_observation_ids, "
                "embedding, embedding_text, embedding_updated_at, "
                "metadata, created_at, updated_at, last_supported_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    clean_entity_id,
                    clean_subject,
                    clean_target,
                    clean_scope,
                    normalized_type,
                    clean_claim,
                    clean_polarity,
                    strength_value,
                    confidence_value,
                    normalized_status,
                    normalized_conflict,
                    str(resolution or "").strip(),
                    str(action_implication or "").strip(),
                    json.dumps(evidence_nodes or []),
                    json.dumps(evidence_observations or []),
                    json.dumps(counter_nodes or []),
                    json.dumps(counter_observations or []),
                    embedding_blob,
                    clean_embedding_text or "",
                    now_text if embedding_blob is not None else None,
                    metadata_str,
                    now_text,
                    now_text,
                    now_text,
                ),
            )
            return int(cursor.lastrowid)

        return self._execute_write(_do)

    def search_memory_interpretations(
        self,
        keyword: Any,
        *,
        entities: Optional[List[Any]] = None,
        top_k: int = 3,
        statuses: Optional[List[str]] = None,
        min_confidence: float = 0.4,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Search current/conflicted agent interpretations relevant to a query."""
        keyword_query = " ".join(keyword) if isinstance(keyword, list) else str(keyword or "")
        terms = [term.strip().lower() for term in re.split(r"\s+|OR", keyword_query) if term.strip()]
        entity_terms: List[str] = []
        for entity in entities or []:
            if isinstance(entity, dict):
                name = str(entity.get("name", "")).strip()
            else:
                name = str(entity or "").strip()
            if name:
                entity_terms.append(name.lower())
        normalized_statuses = [
            self._normalize_memory_interpretation_status(status)
            for status in (statuses or ["current", "conflicted"])
        ]
        normalized_statuses = list(dict.fromkeys(normalized_statuses))
        placeholders = ",".join("?" for _ in normalized_statuses)
        rows = self._conn.execute(
            "SELECT mo.*, en.name AS entity_name "
            "FROM memory_interpretations mo "
            "LEFT JOIN entity_nodes en ON en.id = mo.entity_id "
            f"WHERE mo.status IN ({placeholders}) AND mo.confidence >= ?",
            [*normalized_statuses, max(0.0, min(1.0, float(min_confidence or 0.0)))],
        ).fetchall()
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for row in rows:
            item = self._memory_interpretation_from_row(row)
            content_haystack = " ".join(
                str(item.get(key) or "")
                for key in (
                    "claim", "action_implication", "subject_text", "target_text",
                    "scope", "interpretation_type", "resolution",
                )
            ).lower()
            entity_haystack = " ".join(
                str(item.get(key) or "")
                for key in (
                    "entity_name",
                )
            ).lower()
            matched_terms = [term for term in terms if term in content_haystack]
            matched_entity_name_terms = [
                term for term in terms
                if term in entity_haystack and term not in matched_terms
            ]
            entity_matches = sum(1 for term in entity_terms if term in entity_haystack)
            embedding_similarity = self._embedding_similarity(query_embedding, item.get("embedding"))
            embedding_match = embedding_similarity is not None and embedding_similarity >= 0.35
            strong_embedding_match = embedding_similarity is not None and embedding_similarity >= 0.55
            if terms or entity_terms:
                if entity_terms:
                    if (
                        entity_matches <= 0
                        and not matched_terms
                        and not matched_entity_name_terms
                        and not strong_embedding_match
                    ):
                        continue
                elif not matched_terms and not matched_entity_name_terms and not embedding_match:
                    continue
            else:
                matched_terms = ["_"]
            keyword_score = (len(matched_terms) * 1.2) + (len(matched_entity_name_terms) * 0.6)
            embedding_score = max(0.0, float(embedding_similarity or 0.0))
            score = (
                keyword_score
                + (entity_matches * 1.5)
                + (embedding_score * 1.4)
                + float(item.get("confidence") or 0.0)
                + (0.5 if item.get("status") == "current" else 0.0)
            )
            if embedding_similarity is not None:
                item["embedding_similarity"] = round(float(embedding_similarity), 4)
            item.pop("embedding", None)
            scored.append((score, item))
        scored.sort(key=lambda pair: (pair[0], pair[1].get("last_supported_at") or ""), reverse=True)
        return [item for _, item in scored[:max(1, int(top_k or 3))]]

    def memory_active_task_interpretations(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        """Return current task interpretations for fact-to-task matching."""
        rows = self._conn.execute(
            "SELECT mi.*, en.name AS entity_name "
            "FROM memory_interpretations mi "
            "LEFT JOIN entity_nodes en ON en.id = mi.entity_id "
            "WHERE mi.status IN ('current', 'conflicted') "
            "AND mi.interpretation_type = 'task' "
            "ORDER BY mi.updated_at DESC, mi.id DESC "
            "LIMIT ?",
            (max(1, int(limit or 50)),),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = self._memory_interpretation_from_row(row)
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            item["entity_id"] = item.get("entity_id") or metadata.get("entity_id")
            item["entity_name"] = item.get("entity_name") or metadata.get("entity_name") or ""
            item["topic_key"] = metadata.get("topic_key") or item.get("scope") or "general"
            item["topic_label"] = metadata.get("topic_label") or item.get("target_text") or item.get("scope") or "general"
            item["summary"] = item.get("claim") or ""
            item["keywords"] = " ".join(
                part
                for part in [
                    str(item.get("target_text") or ""),
                    str(item.get("scope") or ""),
                    str(item.get("claim") or ""),
                ]
                if part.strip()
            )
            out.append(item)
        return out

    def get_interpretations_for_observation(
        self,
        observation_id: int,
        *,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return current/conflicted interpretations that already cite an observation."""
        try:
            clean_id = int(observation_id)
        except (TypeError, ValueError):
            return []
        rows = self._conn.execute(
            "SELECT mi.*, en.name AS entity_name "
            "FROM memory_interpretations mi "
            "LEFT JOIN entity_nodes en ON en.id = mi.entity_id "
            "WHERE mi.status IN ('current', 'conflicted') "
            "AND (mi.evidence_observation_ids LIKE ? OR mi.metadata LIKE ?) "
            "ORDER BY mi.updated_at DESC, mi.id DESC "
            "LIMIT ?",
            (
                f"%{clean_id}%",
                f'%"observation_id": {clean_id}%',
                max(1, int(limit or 20)),
            ),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = self._memory_interpretation_from_row(row)
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            if clean_id in item.get("evidence_observation_ids", []) or metadata.get("observation_id") == clean_id:
                out.append(item)
        return out

    # ── Consolidated observations ───────────────────────────────────────

    def memory_observation_source_nodes(
        self,
        *,
        entity_id: int,
        topic_key: str,
        limit: int = 12,
    ) -> List[Dict[str, Any]]:
        """Return fact nodes for an entity/topic bucket."""
        rows = self._conn.execute(
            "SELECT mn.id, mn.time_key, mn.summary, mn.keywords, mn.topic, mn.fact_type, mn.fact_subject, mn.fact_kind, "
            "mn.task_event_like, mn.task_event_subject, mn.task_relevance "
            "FROM memory_nodes mn "
            "JOIN memory_node_entities mne ON mne.node_id = mn.id "
            "WHERE mne.entity_id = ? "
            "ORDER BY mn.time_key DESC, mn.id DESC "
            "LIMIT ?",
            (entity_id, max(limit * 4, limit)),
        ).fetchall()
        target_key = self._memory_topic_key(topic_key)
        out: List[Dict[str, Any]] = []
        for row in rows:
            topic_text = str(row["topic"] or "")
            stored_keys = {
                self._memory_topic_key(topic)
                for topic in topic_text.split()
                if str(topic or "").strip()
            }
            stored_keys.add(self._memory_topic_key(topic_text))
            if target_key not in stored_keys:
                continue
            item = dict(row)
            item["fact_type"] = self._normalize_memory_fact_type(item.get("fact_type"))
            item["fact_subject"] = self._normalize_memory_fact_subject(item.get("fact_subject"))
            item["task_event_like"] = (
                None
                if item.get("task_event_like") is None
                else bool(item.get("task_event_like"))
            )
            item["task_event_subject"] = item.get("task_event_subject") or ""
            item["task_relevance"] = item.get("task_relevance") or ""
            item["fact_kind"] = self._normalize_memory_fact_kind(item.get("fact_kind"))
            out.append(item)
            if len(out) >= limit:
                break
        return out

    def get_unobserved_nodes_for_observation(
        self,
        *,
        date_key: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return today's memory nodes that have not yet supported an observation."""
        day = str(date_key or datetime.now().astimezone().date().isoformat())[:10]
        rows = self._conn.execute(
            "WITH candidate_nodes AS ("
            "  SELECT mn.id, mn.time_key, mn.summary, mn.keywords, mn.topic, "
            "  mn.primary_entity_id, mn.primary_topic, mn.fact_type, mn.fact_subject, mn.fact_kind, "
            "  mn.task_event_like, mn.task_event_subject, mn.task_relevance "
            "  FROM memory_nodes mn "
            "  WHERE substr(mn.time_key, 1, 10) = ? "
            "  AND mn.fact_type IN ('semantic', 'episodic') "
            "  AND NOT EXISTS ("
            "    SELECT 1 FROM memory_observation_sources mos WHERE mos.node_id = mn.id"
            "  ) "
            "  ORDER BY mn.time_key ASC, mn.id ASC "
            "  LIMIT ?"
            ") "
            "SELECT cn.id AS node_id, cn.time_key, cn.summary, cn.keywords, cn.topic, "
            "cn.primary_entity_id, cn.primary_topic, cn.fact_type, cn.fact_subject, cn.fact_kind, "
            "cn.task_event_like, cn.task_event_subject, cn.task_relevance, "
            "en.id AS entity_id, en.name AS entity_name "
            "FROM candidate_nodes cn "
            "JOIN entity_nodes en ON en.id = COALESCE("
            "  cn.primary_entity_id, "
            "  (SELECT MIN(mne.entity_id) FROM memory_node_entities mne WHERE mne.node_id = cn.id)"
            ") "
            "ORDER BY cn.time_key ASC, cn.id ASC, en.name ASC",
            (day, max(1, int(limit or 100))),
        ).fetchall()

        grouped: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            node_id = int(row["node_id"])
            item = grouped.setdefault(
                node_id,
                {
                    "node_id": node_id,
                    "time_key": row["time_key"],
                    "summary": row["summary"],
                    "keywords": [
                        keyword
                        for keyword in str(row["keywords"] or "").split()
                        if str(keyword or "").strip()
                    ],
                    "fact_type": self._normalize_memory_fact_type(row["fact_type"]),
                    "fact_subject": self._normalize_memory_fact_subject(row["fact_subject"]),
                    "fact_kind": self._normalize_memory_fact_kind(row["fact_kind"]),
                    "task_event_like": (
                        None
                        if row["task_event_like"] is None
                        else bool(row["task_event_like"])
                    ),
                    "task_event_subject": row["task_event_subject"] or "",
                    "task_relevance": row["task_relevance"] or "",
                    "primary_entity_id": int(row["entity_id"]),
                    "primary_entity_name": row["entity_name"],
                    "primary_topic": str(row["primary_topic"] or "").strip()
                    or next(
                        (
                            topic
                            for topic in str(row["topic"] or "").split()
                            if str(topic or "").strip()
                        ),
                        "general",
                    ),
                    "topics": [
                        topic
                        for topic in str(row["topic"] or "").split()
                        if str(topic or "").strip()
                    ] or ["general"],
                    "linked_entities": [(int(row["entity_id"]), row["entity_name"])],
                },
            )

        out = list(grouped.values())[:max(1, int(limit or 100))]
        return out

    def memory_active_task_observations(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        """Return active task observations for fact-to-task matching."""
        rows = self._conn.execute(
            "SELECT mo.*, en.name AS entity_name "
            "FROM memory_observations mo "
            "JOIN entity_nodes en ON en.id = mo.entity_id "
            "WHERE mo.status = 'active' AND mo.observation_type = 'task' "
            "ORDER BY mo.updated_at DESC, mo.id DESC "
            "LIMIT ?",
            (max(1, int(limit or 50)),),
        ).fetchall()
        return [dict(row) for row in rows]

    def memory_observation_source_ids(self, observation_id: int) -> List[int]:
        """Return all source node ids for an observation."""
        rows = self._conn.execute(
            "SELECT node_id FROM memory_observation_sources "
            "WHERE observation_id = ? ORDER BY node_id",
            (int(observation_id),),
        ).fetchall()
        return [int(row["node_id"]) for row in rows]

    def find_observed_source_node_ids(self, node_ids: List[int]) -> List[int]:
        """Return node ids that already support at least one observation."""
        clean_ids = self._json_int_list(node_ids)
        if not clean_ids:
            return []
        placeholders = ",".join("?" for _ in clean_ids)
        rows = self._conn.execute(
            f"SELECT DISTINCT node_id FROM memory_observation_sources "
            f"WHERE node_id IN ({placeholders}) ORDER BY node_id",
            clean_ids,
        ).fetchall()
        return [int(row["node_id"]) for row in rows]

    def get_observations_by_ids(self, observation_ids: List[int]) -> List[Dict[str, Any]]:
        """Return active observations by id, including entity names."""
        clean_ids = self._json_int_list(observation_ids)
        if not clean_ids:
            return []
        placeholders = ",".join("?" for _ in clean_ids)
        rows = self._conn.execute(
            "SELECT mo.*, en.name AS entity_name "
            "FROM memory_observations mo "
            "LEFT JOIN entity_nodes en ON en.id = mo.entity_id "
            f"WHERE mo.id IN ({placeholders}) AND mo.status = 'active'",
            clean_ids,
        ).fetchall()
        by_id = {int(row["id"]): dict(row) for row in rows}
        return [by_id[observation_id] for observation_id in clean_ids if observation_id in by_id]

    def memory_update_observation_metadata(
        self,
        observation_id: int,
        metadata: Dict[str, Any],
    ) -> None:
        """Update observation metadata without changing evidence timestamps."""
        metadata_str = json.dumps(metadata or {}, ensure_ascii=False)

        def _do(conn):
            conn.execute(
                "UPDATE memory_observations SET metadata = ? WHERE id = ?",
                (metadata_str, int(observation_id)),
            )

        self._execute_write(_do)

    def memory_upsert_observation(
        self,
        *,
        entity_id: int,
        topic_key: str,
        topic_label: str,
        observation_type: str,
        summary: str,
        keywords: List[str],
        source_node_ids: List[int],
        confidence: float = 1.0,
        embedding: Optional[np.ndarray] = None,
        embedding_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source_role: str = "initial",
    ) -> int:
        """Create or update the active observation for an entity/topic/type."""
        clean_source_role = str(source_role or "initial").strip().lower()
        if clean_source_role not in {"initial", "matched"}:
            clean_source_role = "initial"
        clean_source_ids = []
        seen = set()
        for node_id in source_node_ids:
            if node_id in seen:
                continue
            seen.add(node_id)
            clean_source_ids.append(node_id)
        if not clean_source_ids:
            raise ValueError("memory observation requires at least one source node")

        placeholders = ",".join("?" for _ in clean_source_ids)
        time_rows = self._conn.execute(
            f"SELECT MIN(time_key) AS start_time, MAX(time_key) AS end_time FROM memory_nodes "
            f"WHERE id IN ({placeholders})",
            clean_source_ids,
        ).fetchone()
        source_time_start = time_rows["start_time"] if time_rows else None
        source_time_end = time_rows["end_time"] if time_rows else None
        now_text = datetime.now().astimezone().isoformat()
        keywords_str = " ".join(keywords) if isinstance(keywords, list) else str(keywords or "")
        metadata_str = json.dumps(metadata or {}, ensure_ascii=False)
        confidence_value = max(0.0, min(1.0, float(confidence or 0.0)))
        embedding_blob = self._embedding_to_blob(embedding)
        clean_embedding_text = None if embedding_text is None else str(embedding_text or "").strip()

        def _do(conn):
            existing = conn.execute(
                "SELECT id, embedding, embedding_text, embedding_updated_at FROM memory_observations "
                "WHERE entity_id = ? AND topic_key = ? AND observation_type = ? AND status = 'active' "
                "ORDER BY updated_at DESC, id DESC LIMIT 1",
                (entity_id, topic_key, observation_type),
            ).fetchone()
            if existing:
                observation_id = existing["id"] if isinstance(existing, sqlite3.Row) else existing[0]
                next_embedding = embedding_blob if embedding_blob is not None else existing["embedding"]
                next_embedding_text = (
                    clean_embedding_text
                    if clean_embedding_text is not None
                    else (existing["embedding_text"] or "")
                )
                next_embedding_updated_at = (
                    now_text
                    if embedding_blob is not None
                    else existing["embedding_updated_at"]
                )
                conn.execute(
                    "UPDATE memory_observations SET "
                    "topic_label = ?, summary = ?, keywords = ?, confidence = ?, "
                    "updated_at = ?, last_supported_at = ?, source_time_start = ?, "
                    "source_time_end = ?, embedding = ?, embedding_text = ?, "
                    "embedding_updated_at = ?, metadata = ? "
                    "WHERE id = ?",
                    (
                        topic_label,
                        summary,
                        keywords_str,
                        confidence_value,
                        now_text,
                        now_text,
                        source_time_start,
                        source_time_end,
                        next_embedding,
                        next_embedding_text,
                        next_embedding_updated_at,
                        metadata_str,
                        observation_id,
                    ),
                )
            else:
                cursor = conn.execute(
                    "INSERT INTO memory_observations "
                    "(entity_id, topic_key, topic_label, observation_type, summary, keywords, "
                    "confidence, status, created_at, updated_at, last_supported_at, "
                    "source_time_start, source_time_end, embedding, embedding_text, "
                    "embedding_updated_at, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        entity_id,
                        topic_key,
                        topic_label,
                        observation_type,
                        summary,
                        keywords_str,
                        confidence_value,
                        now_text,
                        now_text,
                        now_text,
                        source_time_start,
                        source_time_end,
                        embedding_blob,
                        clean_embedding_text or "",
                        now_text if embedding_blob is not None else None,
                        metadata_str,
                    ),
                )
                observation_id = cursor.lastrowid
            for node_id in clean_source_ids:
                conn.execute(
                    "INSERT OR IGNORE INTO memory_observation_sources "
                    "(observation_id, node_id, role, confidence) VALUES (?, ?, ?, ?)",
                    (observation_id, node_id, clean_source_role, confidence_value),
                )
            return observation_id

        return self._execute_write(_do)

    def memory_observation_pending_sources(
        self,
        *,
        entity_id: int,
        topic_key: str,
        observation_type: Optional[str] = None,
        candidate_node_ids: List[int],
    ) -> Tuple[Optional[Dict[str, Any]], List[int]]:
        """Return active observation and candidate source ids not yet attached."""
        if not candidate_node_ids:
            return None, []
        if observation_type:
            observation = self._conn.execute(
                "SELECT * FROM memory_observations "
                "WHERE entity_id = ? AND topic_key = ? AND observation_type = ? AND status = 'active' "
                "ORDER BY updated_at DESC, id DESC LIMIT 1",
                (entity_id, topic_key, observation_type),
            ).fetchone()
        else:
            observation = self._conn.execute(
                "SELECT * FROM memory_observations "
                "WHERE entity_id = ? AND topic_key = ? AND status = 'active' "
                "ORDER BY updated_at DESC, id DESC LIMIT 1",
                (entity_id, topic_key),
            ).fetchone()
        if not observation:
            return None, list(dict.fromkeys(candidate_node_ids))
        observation_dict = dict(observation)
        observation_id = observation_dict["id"]
        placeholders = ",".join("?" for _ in candidate_node_ids)
        rows = self._conn.execute(
            f"SELECT node_id FROM memory_observation_sources "
            f"WHERE observation_id = ? AND node_id IN ({placeholders})",
            [observation_id] + candidate_node_ids,
        ).fetchall()
        attached = {row["node_id"] for row in rows}
        pending = [node_id for node_id in dict.fromkeys(candidate_node_ids) if node_id not in attached]
        return observation_dict, pending

    def memory_observation_pending_source_count(
        self,
        *,
        entity_id: int,
        topic_key: str,
        observation_type: Optional[str] = None,
        candidate_node_ids: List[int],
    ) -> Tuple[Optional[int], int]:
        """Return active observation id and candidate source count not yet attached."""
        observation, pending = self.memory_observation_pending_sources(
            entity_id=entity_id,
            topic_key=topic_key,
            observation_type=observation_type,
            candidate_node_ids=candidate_node_ids,
        )
        return (observation["id"] if observation else None), len(pending)

    def memory_observations_for_entity_topic(
        self,
        *,
        entity_id: int,
        topic_key: str,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Return active observations for one exact entity/topic pair."""
        rows = self._conn.execute(
            "SELECT mo.*, en.name AS entity_name "
            "FROM memory_observations mo "
            "LEFT JOIN entity_nodes en ON en.id = mo.entity_id "
            "WHERE mo.entity_id = ? AND mo.topic_key = ? AND mo.status = 'active' "
            "ORDER BY mo.updated_at DESC, mo.id DESC",
            (int(entity_id), str(topic_key)),
        ).fetchall()
        observations: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            embedding_similarity = self._embedding_similarity(
                query_embedding,
                item.get("embedding"),
            )
            if embedding_similarity is not None:
                item["embedding_similarity"] = round(float(embedding_similarity), 4)
            item.pop("embedding", None)
            observations.append(item)
        return observations

    def search_memory_observations(
        self,
        keyword: Any,
        *,
        entities: Optional[List[Any]] = None,
        top_k: int = 3,
        entity_ids: Optional[List[int]] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Search active observations by keyword/topic/entity."""
        keyword_query = " ".join(keyword) if isinstance(keyword, list) else str(keyword or "")
        terms = [term.strip().lower() for term in re.split(r"\s+|OR", keyword_query) if term.strip()]
        weak_terms = {"父亲", "母亲", "爸爸", "妈妈", "家人", "家庭", "用户", "偏好", "喜欢", "信息", "记录"}
        entity_terms: List[str] = []
        for entity in entities or []:
            if isinstance(entity, dict):
                name = str(entity.get("name", "")).strip()
            else:
                name = str(entity or "").strip()
            if name:
                entity_terms.append(name.lower())
        params: List[Any] = []
        where = ["status = 'active'"]
        if entity_ids:
            placeholders = ",".join("?" for _ in entity_ids)
            where.append(f"entity_id IN ({placeholders})")
            params.extend(entity_ids)
        rows = self._conn.execute(
            "SELECT mo.*, en.name AS entity_name "
            "FROM memory_observations mo "
            "LEFT JOIN entity_nodes en ON en.id = mo.entity_id "
            f"WHERE {' AND '.join(where)}",
            params,
        ).fetchall()
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for row in rows:
            item = dict(row)
            entity_name_text = str(item.get("entity_name", "") or "").lower()
            haystack = f"{item.get('summary', '')} {item.get('keywords', '')} {item.get('topic_label', '')} {entity_name_text}".lower()
            matched_terms = [term for term in terms if term in haystack]
            strong_keyword_matches = sum(1 for term in matched_terms if term not in weak_terms)
            weak_keyword_matches = len(matched_terms) - strong_keyword_matches
            entity_matches = sum(
                1
                for term in entity_terms
                if term in entity_name_text or term in haystack
            )
            embedding_similarity = self._embedding_similarity(query_embedding, item.get("embedding"))
            embedding_match = embedding_similarity is not None and embedding_similarity >= 0.35
            strong_embedding_match = embedding_similarity is not None and embedding_similarity >= 0.55
            if terms or entity_terms:
                if entity_terms:
                    if entity_matches <= 0 and strong_keyword_matches < 2 and not strong_embedding_match:
                        continue
                elif strong_keyword_matches <= 0 and len(matched_terms) < 2 and not embedding_match:
                    continue
            else:
                strong_keyword_matches = 1
            embedding_score = max(0.0, float(embedding_similarity or 0.0))
            score = (
                strong_keyword_matches
                + (weak_keyword_matches * 0.25)
                + (entity_matches * 1.5)
                + (embedding_score * 1.4)
                + float(item.get("confidence") or 0.0)
            )
            if embedding_similarity is not None:
                item["embedding_similarity"] = round(float(embedding_similarity), 4)
            item.pop("embedding", None)
            scored.append((score, item))
        scored.sort(key=lambda pair: (pair[0], pair[1].get("last_supported_at") or ""), reverse=True)
        return [item for _, item in scored[:top_k]]

    def get_observation_supporting_nodes(
        self,
        observation_ids: List[int],
        *,
        per_observation: int = 2,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Fetch supporting fact nodes for observation ids."""
        out: Dict[int, List[Dict[str, Any]]] = {}
        for observation_id in observation_ids:
            rows = self._conn.execute(
                "SELECT mn.id, mn.time_key, mn.summary, mn.keywords, mn.original_dialog, "
                "mn.tags, mn.fact_type, mn.fact_subject, mn.fact_kind, mn.task_event_like, mn.task_event_subject, mn.task_relevance "
                "FROM memory_observation_sources mos "
                "JOIN memory_nodes mn ON mn.id = mos.node_id "
                "WHERE mos.observation_id = ? "
                "ORDER BY mn.time_key DESC, mn.id DESC "
                "LIMIT ?",
                (observation_id, per_observation),
            ).fetchall()
            nodes = []
            for row in rows:
                tags = json.loads(row["tags"]) if row["tags"] else []
                nodes.append({
                    "id": row["id"],
                    "time_key": row["time_key"],
                    "summary": row["summary"],
                    "keywords": row["keywords"].split(" ") if row["keywords"] else [],
                    "original_dialog": row["original_dialog"],
                    "tags": tags,
                    "fact_type": self._normalize_memory_fact_type(row["fact_type"]),
                    "fact_subject": self._normalize_memory_fact_subject(row["fact_subject"]),
                    "fact_kind": self._normalize_memory_fact_kind(row["fact_kind"]),
                    "task_event_like": (
                        None
                        if row["task_event_like"] is None
                        else bool(row["task_event_like"])
                    ),
                    "task_event_subject": row["task_event_subject"] or "",
                    "task_relevance": row["task_relevance"] or "",
                    "node_relations": {},
                })
            out[observation_id] = nodes
        return out

    def find_duplicated_observation_groups(
        self,
        entity_ids: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Return active same-entity/topic/category observation groups that need reflection."""
        params: List[Any] = []
        where = ["mo.status = 'active'"]
        if entity_ids:
            placeholders = ",".join("?" for _ in entity_ids)
            where.append(f"mo.entity_id IN ({placeholders})")
            params.extend(entity_ids)
        rows = self._conn.execute(
            "SELECT mo.*, en.name AS entity_name "
            "FROM memory_observations mo "
            "LEFT JOIN entity_nodes en ON en.id = mo.entity_id "
            f"WHERE {' AND '.join(where)} "
            "ORDER BY mo.entity_id, mo.topic_key, mo.observation_type, mo.updated_at DESC, mo.id DESC",
            params,
        ).fetchall()
        grouped: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = {}
        for row in rows:
            item = dict(row)
            category = str(item.get("observation_type") or "observation")
            grouped.setdefault(
                (int(item["entity_id"]), str(item["topic_key"]), category),
                [],
            ).append(item)

        groups: List[Dict[str, Any]] = []
        for (_entity_id, _topic_key, _category), observations in grouped.items():
            if len(observations) < 2:
                continue
            observation_ids = [int(obs["id"]) for obs in observations]
            source_rows = self._conn.execute(
                "SELECT DISTINCT mn.id, mn.time_key, mn.summary, mn.keywords, mn.fact_type, mn.fact_subject, mn.fact_kind, "
                "mn.task_event_like, mn.task_event_subject, mn.task_relevance "
                "FROM memory_observation_sources mos "
                "JOIN memory_nodes mn ON mn.id = mos.node_id "
                f"WHERE mos.observation_id IN ({','.join('?' for _ in observation_ids)}) "
                "ORDER BY mn.time_key DESC, mn.id DESC",
                observation_ids,
            ).fetchall()
            groups.append({
                "entity_id": int(observations[0]["entity_id"]),
                "entity_name": observations[0].get("entity_name") or "",
                "topic_key": observations[0]["topic_key"],
                "topic_label": observations[0]["topic_label"],
                "observation_type": observations[0]["observation_type"],
                "observations": observations,
                "source_nodes": [
                    {
                        **dict(row),
                        "fact_type": self._normalize_memory_fact_type(row["fact_type"]),
                        "fact_subject": self._normalize_memory_fact_subject(row["fact_subject"]),
                        "fact_kind": self._normalize_memory_fact_kind(row["fact_kind"]),
                    }
                    for row in source_rows
                ],
            })
        return groups

    def memory_reflect_node_decay(
        self,
        *,
        fact_half_life_days: Optional[float] = None,
        experience_half_life_days: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Recompute persisted recency decay scores for all memory nodes."""
        fact_half_life = float(
            fact_half_life_days or self._MEMORY_OBSERVATION_FACT_HALF_LIFE_DAYS
        )
        experience_half_life = float(
            experience_half_life_days or self._MEMORY_OBSERVATION_EXPERIENCE_HALF_LIFE_DAYS
        )
        now_dt = now or datetime.now().astimezone()
        evaluated_at = now_dt.isoformat()
        rows = self._conn.execute(
            "SELECT id, time_key, fact_type FROM memory_nodes "
            "WHERE fact_type IN ('semantic', 'episodic') "
            "ORDER BY time_key DESC, id DESC"
        ).fetchall()

        nodes: List[Dict[str, Any]] = []
        for row in rows:
            fact_type = self._normalize_memory_fact_type(row["fact_type"])
            half_life = experience_half_life if fact_type == "episodic" else fact_half_life
            score = self._memory_decay_score(
                self._parse_memory_time_key(row["time_key"]),
                now=now_dt,
                half_life_days=half_life,
            )
            nodes.append({
                "id": int(row["id"]),
                "fact_type": fact_type,
                "score": score,
                "half_life_days": half_life,
            })

        if nodes:
            def _do(conn):
                for item in nodes:
                    conn.execute(
                        "UPDATE memory_nodes SET decay_score = ?, decay_updated_at = ?, "
                        "decay_half_life_days = ? WHERE id = ?",
                        (
                            item["score"],
                            evaluated_at,
                            item["half_life_days"],
                            item["id"],
                        ),
                    )

            self._execute_write(_do)

        return {
            "evaluated": len(nodes),
            "updated": len(nodes),
            "fact_half_life_days": fact_half_life,
            "experience_half_life_days": experience_half_life,
            "evaluated_at": evaluated_at,
            "nodes": nodes,
        }

    def memory_reflect_observation_decay(
        self,
        *,
        threshold: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Evaluate active observations using persisted source-node decay scores.

        The reflect step should call ``memory_reflect_node_decay`` first so the
        source scores represent the current maintenance run.  The aggregate keeps a
        small "fresh support" component so one recent supporting source can keep
        an observation alive even when it also has many older sources.
        """
        decay_threshold = max(
            0.0,
            min(1.0, float(threshold if threshold is not None else self._MEMORY_OBSERVATION_DECAY_THRESHOLD)),
        )
        now_dt = now or datetime.now().astimezone()
        rows = self._conn.execute(
            "SELECT mo.id AS observation_id, mo.metadata AS observation_metadata, "
            "mos.confidence AS source_confidence, mn.id AS node_id, mn.fact_type, "
            "mn.decay_score, mn.decay_updated_at, mn.decay_half_life_days "
            "FROM memory_observations mo "
            "LEFT JOIN memory_observation_sources mos ON mos.observation_id = mo.id "
            "LEFT JOIN memory_nodes mn ON mn.id = mos.node_id "
            "WHERE mo.status = 'active' "
            "ORDER BY mo.id, mn.time_key DESC, mn.id DESC"
        ).fetchall()

        grouped: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            observation_id = int(row["observation_id"])
            group = grouped.setdefault(
                observation_id,
                {
                    "id": observation_id,
                    "metadata": row["observation_metadata"],
                    "sources": [],
                },
            )
            if row["node_id"] is None:
                continue
            fact_type = self._normalize_memory_fact_type(row["fact_type"])
            try:
                score = float(row["decay_score"] if row["decay_score"] is not None else 1.0)
            except (TypeError, ValueError):
                score = 1.0
            try:
                confidence = float(row["source_confidence"] or 1.0)
            except (TypeError, ValueError):
                confidence = 1.0
            group["sources"].append({
                "node_id": int(row["node_id"]),
                "fact_type": fact_type,
                "score": score,
                "confidence": max(0.0, confidence),
                "decay_updated_at": row["decay_updated_at"],
                "decay_half_life_days": row["decay_half_life_days"],
            })

        evaluated: List[Dict[str, Any]] = []
        for observation_id, group in grouped.items():
            sources = group["sources"]
            if not sources:
                average_score = 0.0
                max_score = 0.0
                combined_score = 0.0
            else:
                weights = [
                    source["confidence"] if source["confidence"] > 0 else 1.0
                    for source in sources
                ]
                total_weight = sum(weights) or float(len(sources))
                average_score = sum(
                    source["score"] * weight
                    for source, weight in zip(sources, weights)
                ) / total_weight
                max_score = max(source["score"] for source in sources)
                combined_score = (0.70 * average_score) + (0.30 * max_score)
            action = "deactivate" if combined_score < decay_threshold else "keep"
            evaluated.append({
                "id": observation_id,
                "action": action,
                "score": combined_score,
                "average_score": average_score,
                "max_score": max_score,
                "source_count": len(sources),
            })

        to_deactivate = [item for item in evaluated if item["action"] == "deactivate"]

        if evaluated:
            evaluated_at = now_dt.isoformat()

            def _do(conn):
                for item in evaluated:
                    observation_id = int(item["id"])
                    metadata_row = grouped[observation_id].get("metadata")
                    try:
                        metadata = json.loads(metadata_row or "{}")
                    except json.JSONDecodeError:
                        metadata = {}
                    metadata["decay"] = {
                        "score": item["score"],
                        "average_score": item["average_score"],
                        "max_score": item["max_score"],
                        "threshold": decay_threshold,
                        "source_count": item["source_count"],
                        "evaluated_at": evaluated_at,
                    }
                    metadata_str = json.dumps(metadata, ensure_ascii=False)
                    if item["action"] == "deactivate":
                        conn.execute(
                            "UPDATE memory_observations SET status = 'inactive', metadata = ?, updated_at = ? "
                            "WHERE id = ?",
                            (metadata_str, evaluated_at, observation_id),
                        )
                    else:
                        conn.execute(
                            "UPDATE memory_observations SET metadata = ? WHERE id = ?",
                            (metadata_str, observation_id),
                        )

            self._execute_write(_do)

        return {
            "evaluated": len(evaluated),
            "inactivated": len(to_deactivate),
            "would_inactivate": len(to_deactivate),
            "threshold": decay_threshold,
            "observations": evaluated,
        }

    def memory_reflect_task_inactivity(
        self,
        *,
        active_to_paused_days: Optional[float] = None,
        stale_days: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Update task interpretation status when active tasks have no recent support."""
        paused_after_days = max(
            1.0,
            float(
                active_to_paused_days
                if active_to_paused_days is not None
                else self._MEMORY_TASK_PAUSED_IDLE_DAYS
            ),
        )
        stale_after_days = max(
            paused_after_days,
            float(stale_days if stale_days is not None else self._MEMORY_TASK_STALE_IDLE_DAYS),
        )
        now_dt = now or datetime.now().astimezone()
        evaluated_at = now_dt.isoformat()
        rows = self._conn.execute(
            "SELECT id, metadata, created_at, updated_at, last_supported_at "
            "FROM memory_interpretations "
            "WHERE status IN ('current', 'conflicted') AND interpretation_type = 'task' "
            "ORDER BY updated_at DESC, id DESC"
        ).fetchall()

        evaluated: List[Dict[str, Any]] = []
        to_update: List[Dict[str, Any]] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            current_status = str(metadata.get("task_status", "active") or "active").strip().lower()
            if current_status not in {"active", "blocked", "paused", "stale"}:
                current_status = "active"
            last_active_raw = row["last_supported_at"] or row["updated_at"] or row["created_at"]
            last_active_at = self._parse_memory_time_key(last_active_raw)
            if last_active_at is None:
                idle_days = 0.0
            else:
                compare_now = now_dt
                compare_last = last_active_at
                if compare_last.tzinfo is not None and compare_now.tzinfo is None:
                    compare_last = compare_last.replace(tzinfo=None)
                elif compare_last.tzinfo is None and compare_now.tzinfo is not None:
                    compare_now = compare_now.replace(tzinfo=None)
                idle_days = max(0.0, (compare_now - compare_last).total_seconds() / 86400.0)

            new_status = current_status
            action = "keep"
            if current_status == "active":
                if idle_days >= stale_after_days:
                    new_status = "stale"
                    action = "stale"
                elif idle_days >= paused_after_days:
                    new_status = "paused"
                    action = "pause"
            elif current_status in {"blocked", "paused"} and idle_days >= stale_after_days:
                new_status = "stale"
                action = "stale"

            item = {
                "id": int(row["id"]),
                "action": action,
                "previous_status": current_status,
                "new_status": new_status,
                "idle_days": idle_days,
                "last_active_at": str(last_active_raw or ""),
            }
            evaluated.append(item)
            if action != "keep":
                updated_metadata = dict(metadata)
                updated_metadata["previous_task_status"] = current_status
                updated_metadata["task_status"] = new_status
                updated_metadata["task_source"] = "inferred_from_interpretation"
                updated_metadata["status_reason"] = "no_recent_support"
                updated_metadata["status_updated_by"] = "reflect_task_inactivity_policy"
                updated_metadata["status_updated_at"] = evaluated_at
                updated_metadata["last_active_at"] = str(last_active_raw or "")
                updated_metadata["task_inactivity"] = {
                    "idle_days": idle_days,
                    "paused_after_days": paused_after_days,
                    "stale_after_days": stale_after_days,
                    "evaluated_at": evaluated_at,
                }
                item["metadata"] = updated_metadata
                to_update.append(item)

        if to_update:
            def _do(conn):
                for item in to_update:
                    conn.execute(
                        "UPDATE memory_interpretations SET metadata = ?, updated_at = ? WHERE id = ?",
                        (
                            json.dumps(item["metadata"], ensure_ascii=False),
                            evaluated_at,
                            int(item["id"]),
                        ),
                    )

            self._execute_write(_do)

        paused = [item for item in to_update if item["new_status"] == "paused"]
        stale = [item for item in to_update if item["new_status"] == "stale"]
        return {
            "checked": len(evaluated),
            "changed": len(to_update),
            "would_change": len(to_update),
            "paused": len(paused),
            "stale": len(stale),
            "would_pause": len(paused),
            "would_stale": len(stale),
            "active_to_paused_days": paused_after_days,
            "stale_days": stale_after_days,
            "tasks": [
                {key: value for key, value in item.items() if key != "metadata"}
                for item in evaluated
            ],
        }

    def memory_replace_observation_group(
        self,
        *,
        keep_observation_id: int,
        remove_observation_ids: List[int],
        observation_type: str,
        summary: str,
        keywords: List[str],
        confidence: float,
        source_node_ids: List[int],
        embedding: Optional[np.ndarray] = None,
        embedding_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source_roles: Optional[Dict[int, str]] = None,
    ) -> None:
        """Replace a duplicate observation group while preserving source roles."""
        clean_remove_ids = [
            int(obs_id)
            for obs_id in dict.fromkeys(remove_observation_ids)
            if int(obs_id) != keep_observation_id
        ]
        clean_source_ids = [int(node_id) for node_id in dict.fromkeys(source_node_ids)]
        if not clean_source_ids:
            raise ValueError("reflected observation requires at least one source node")
        clean_source_roles: Dict[int, str] = {}
        for node_id, role in (source_roles or {}).items():
            try:
                clean_node_id = int(node_id)
            except (TypeError, ValueError):
                continue
            clean_role = str(role or "").strip().lower()
            if clean_role in {"initial", "matched", "supporting"}:
                clean_source_roles[clean_node_id] = clean_role
        placeholders = ",".join("?" for _ in clean_source_ids)
        time_rows = self._conn.execute(
            f"SELECT MIN(time_key) AS start_time, MAX(time_key) AS end_time FROM memory_nodes "
            f"WHERE id IN ({placeholders})",
            clean_source_ids,
        ).fetchone()
        now_text = datetime.now().astimezone().isoformat()
        keywords_str = " ".join(keywords) if isinstance(keywords, list) else str(keywords or "")
        confidence_value = max(0.0, min(1.0, float(confidence or 0.0)))
        metadata_str = json.dumps(metadata or {}, ensure_ascii=False)
        embedding_blob = self._embedding_to_blob(embedding)
        clean_embedding_text = None if embedding_text is None else str(embedding_text or "").strip()
        source_time_start = time_rows["start_time"] if time_rows else None
        source_time_end = time_rows["end_time"] if time_rows else None

        def _do(conn):
            source_observation_ids = [keep_observation_id, *clean_remove_ids]
            source_observation_placeholders = ",".join("?" for _ in source_observation_ids)
            existing_source_rows = conn.execute(
                "SELECT node_id, role FROM memory_observation_sources "
                f"WHERE observation_id IN ({source_observation_placeholders})",
                source_observation_ids,
            ).fetchall()
            role_priority = {"supporting": 0, "matched": 1, "initial": 2}
            existing_source_roles: Dict[int, str] = {}
            for row in existing_source_rows:
                node_id = int(row["node_id"])
                role = str(row["role"] or "supporting").strip().lower()
                if role not in role_priority:
                    role = "supporting"
                previous = existing_source_roles.get(node_id)
                if previous is None or role_priority[role] > role_priority[previous]:
                    existing_source_roles[node_id] = role
            existing = conn.execute(
                "SELECT embedding, embedding_text, embedding_updated_at "
                "FROM memory_observations WHERE id = ?",
                (keep_observation_id,),
            ).fetchone()
            next_embedding = embedding_blob if embedding_blob is not None else (existing["embedding"] if existing else None)
            next_embedding_text = (
                clean_embedding_text
                if clean_embedding_text is not None
                else ((existing["embedding_text"] or "") if existing else "")
            )
            next_embedding_updated_at = (
                now_text
                if embedding_blob is not None
                else (existing["embedding_updated_at"] if existing else None)
            )
            conn.execute(
                "UPDATE memory_observations SET "
                "observation_type = ?, summary = ?, keywords = ?, confidence = ?, "
                "updated_at = ?, last_supported_at = ?, source_time_start = ?, "
                "source_time_end = ?, embedding = ?, embedding_text = ?, "
                "embedding_updated_at = ?, metadata = ? "
                "WHERE id = ?",
                (
                    observation_type,
                    summary,
                    keywords_str,
                    confidence_value,
                    now_text,
                    now_text,
                    source_time_start,
                    source_time_end,
                    next_embedding,
                    next_embedding_text,
                    next_embedding_updated_at,
                    metadata_str,
                    keep_observation_id,
                ),
            )
            if clean_remove_ids:
                remove_placeholders = ",".join("?" for _ in clean_remove_ids)
                conn.execute(
                    f"DELETE FROM memory_observation_sources WHERE observation_id IN ({remove_placeholders})",
                    clean_remove_ids,
                )
                conn.execute(
                    f"DELETE FROM memory_observations WHERE id IN ({remove_placeholders})",
                    clean_remove_ids,
                )
            conn.execute(
                "DELETE FROM memory_observation_sources WHERE observation_id = ?",
                (keep_observation_id,),
            )
            for node_id in clean_source_ids:
                role = clean_source_roles.get(
                    node_id,
                    existing_source_roles.get(node_id, "supporting"),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO memory_observation_sources "
                    "(observation_id, node_id, role, confidence) VALUES (?, ?, ?, ?)",
                    (keep_observation_id, node_id, role, confidence_value),
                )

        self._execute_write(_do)

    # =========================================================================
    # Utility
    # =========================================================================

    def session_count(self, source: str = None) -> int:
        """Count sessions, optionally filtered by source."""
        with self._lock:
            if source:
                cursor = self._conn.execute(
                    "SELECT COUNT(*) FROM sessions WHERE source = ?", (source,)
                )
            else:
                cursor = self._conn.execute("SELECT COUNT(*) FROM sessions")
            return cursor.fetchone()[0]

    def message_count(self, session_id: str = None) -> int:
        """Count messages, optionally for a specific session."""
        with self._lock:
            if session_id:
                cursor = self._conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
                )
            else:
                cursor = self._conn.execute("SELECT COUNT(*) FROM messages")
            return cursor.fetchone()[0]

    # =========================================================================
    # Export and cleanup
    # =========================================================================

    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export a single session with all its messages as a dict."""
        session = self.get_session(session_id)
        if not session:
            return None
        messages = self.get_messages(session_id)
        return {**session, "messages": messages}

    def export_all(self, source: str = None) -> List[Dict[str, Any]]:
        """
        Export all sessions (with messages) as a list of dicts.
        Suitable for writing to a JSONL file for backup/analysis.
        """
        sessions = self.search_sessions(source=source, limit=100000)
        results = []
        for session in sessions:
            messages = self.get_messages(session["id"])
            results.append({**session, "messages": messages})
        return results

    def clear_messages(self, session_id: str) -> None:
        """Delete all messages for a session and reset its counters."""
        def _do(conn):
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            conn.execute(
                "UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = ?",
                (session_id,),
            )
        self._execute_write(_do)

    @staticmethod
    def _remove_session_files(sessions_dir: Optional[Path], session_id: str) -> None:
        """Remove on-disk transcript files for a session.

        Cleans up ``{session_id}.json``, ``{session_id}.jsonl``, and any
        ``request_dump_{session_id}_*.json`` files left by the gateway.
        Silently skips files that don't exist and swallows OSError so a
        filesystem hiccup never blocks a DB operation.
        """
        if sessions_dir is None:
            return
        for suffix in (".json", ".jsonl"):
            p = sessions_dir / f"{session_id}{suffix}"
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        # request_dump files use session_id as a prefix component
        try:
            for p in sessions_dir.glob(f"request_dump_{session_id}_*.json"):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
        except OSError:
            pass

    def delete_session(
        self,
        session_id: str,
        sessions_dir: Optional[Path] = None,
    ) -> bool:
        """Delete a session and all its messages.

        Child sessions are orphaned (parent_session_id set to NULL) rather
        than cascade-deleted, so they remain accessible independently.
        When *sessions_dir* is provided, also removes on-disk transcript
        files (``.json`` / ``.jsonl`` / ``request_dump_*``) for the deleted
        session. Returns True if the session was found and deleted.
        """
        def _do(conn):
            cursor = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE id = ?", (session_id,)
            )
            if cursor.fetchone()[0] == 0:
                return False
            # Orphan child sessions so FK constraint is satisfied
            conn.execute(
                "UPDATE sessions SET parent_session_id = NULL "
                "WHERE parent_session_id = ?",
                (session_id,),
            )
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return True

        deleted = self._execute_write(_do)
        if deleted:
            self._remove_session_files(sessions_dir, session_id)
        return deleted

    def prune_sessions(
        self,
        older_than_days: int = 90,
        source: str = None,
        sessions_dir: Optional[Path] = None,
    ) -> int:
        """Delete sessions older than N days. Returns count of deleted sessions.

        Only prunes ended sessions (not active ones).  Child sessions outside
        the prune window are orphaned (parent_session_id set to NULL) rather
        than cascade-deleted.  When *sessions_dir* is provided, also removes
        on-disk transcript files (``.json`` / ``.jsonl`` /
        ``request_dump_*``) for every pruned session, outside the DB
        transaction.
        """
        cutoff = time.time() - (older_than_days * 86400)
        removed_ids: list[str] = []

        def _do(conn):
            if source:
                cursor = conn.execute(
                    """SELECT id FROM sessions
                       WHERE started_at < ? AND ended_at IS NOT NULL AND source = ?""",
                    (cutoff, source),
                )
            else:
                cursor = conn.execute(
                    "SELECT id FROM sessions WHERE started_at < ? AND ended_at IS NOT NULL",
                    (cutoff,),
                )
            session_ids = set(row["id"] for row in cursor.fetchall())

            if not session_ids:
                return 0

            # Orphan any sessions whose parent is about to be deleted
            placeholders = ",".join("?" * len(session_ids))
            conn.execute(
                f"UPDATE sessions SET parent_session_id = NULL "
                f"WHERE parent_session_id IN ({placeholders})",
                list(session_ids),
            )

            for sid in session_ids:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
                conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
                removed_ids.append(sid)
            return len(session_ids)

        count = self._execute_write(_do)
        # Clean up on-disk files outside the DB transaction
        for sid in removed_ids:
            self._remove_session_files(sessions_dir, sid)
        return count

    # ── Meta key/value (for scheduler bookkeeping) ──

    def get_meta(self, key: str) -> Optional[str]:
        """Read a value from the state_meta key/value store."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM state_meta WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return row["value"] if isinstance(row, sqlite3.Row) else row[0]

    def set_meta(self, key: str, value: str) -> None:
        """Write a value to the state_meta key/value store."""
        def _do(conn):
            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
        self._execute_write(_do)

    # ── Space reclamation ──

    def vacuum(self) -> None:
        """Run VACUUM to reclaim disk space after large deletes.

        SQLite does not shrink the database file when rows are deleted —
        freed pages just get reused on the next insert. After a prune that
        removed hundreds of sessions, the file stays bloated unless we
        explicitly VACUUM.

        VACUUM rewrites the entire DB, so it's expensive (seconds per
        100MB) and cannot run inside a transaction. It also acquires an
        exclusive lock, so callers must ensure no other writers are
        active. Safe to call at startup before the gateway/CLI starts
        serving traffic.
        """
        # VACUUM cannot be executed inside a transaction.
        with self._lock:
            # Best-effort WAL checkpoint first, then VACUUM.
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
            self._conn.execute("VACUUM")

    def maybe_auto_prune_and_vacuum(
        self,
        retention_days: int = 90,
        min_interval_hours: int = 24,
        vacuum: bool = True,
        sessions_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Idempotent auto-maintenance: prune old sessions + optional VACUUM.

        Records the last run timestamp in state_meta so subsequent calls
        within ``min_interval_hours`` no-op. Designed to be called once at
        startup from long-lived entrypoints (CLI, gateway, cron scheduler).

        When *sessions_dir* is provided, on-disk transcript files
        (``.json`` / ``.jsonl`` / ``request_dump_*``) for pruned sessions
        are removed as part of the same sweep (issue #3015).

        Never raises. On any failure, logs a warning and returns a dict
        with ``"error"`` set.

        Returns a dict with keys:
          - ``"skipped"`` (bool) — true if within min_interval_hours of last run
          - ``"pruned"`` (int)   — number of sessions deleted
          - ``"vacuumed"`` (bool) — true if VACUUM ran
          - ``"error"`` (str, optional) — present only on failure
        """
        result: Dict[str, Any] = {"skipped": False, "pruned": 0, "vacuumed": False}
        try:
            # Skip if another process/call did maintenance recently.
            last_raw = self.get_meta("last_auto_prune")
            now = time.time()
            if last_raw:
                try:
                    last_ts = float(last_raw)
                    if now - last_ts < min_interval_hours * 3600:
                        result["skipped"] = True
                        return result
                except (TypeError, ValueError):
                    pass  # corrupt meta; treat as no prior run

            pruned = self.prune_sessions(
                older_than_days=retention_days,
                sessions_dir=sessions_dir,
            )
            result["pruned"] = pruned

            # Only VACUUM if we actually freed rows — VACUUM on a tight DB
            # is wasted I/O. Threshold keeps small DBs from paying the cost.
            if vacuum and pruned > 0:
                try:
                    self.vacuum()
                    result["vacuumed"] = True
                except Exception as exc:
                    logger.warning("state.db VACUUM failed: %s", exc)

            # Record the attempt even if pruned == 0, so we don't retry
            # every startup within the min_interval_hours window.
            self.set_meta("last_auto_prune", str(now))

            if pruned > 0:
                logger.info(
                    "state.db auto-maintenance: pruned %d session(s) older than %d days%s",
                    pruned,
                    retention_days,
                    " + VACUUM" if result["vacuumed"] else "",
                )
        except Exception as exc:
            # Maintenance must never block startup. Log and return error marker.
            logger.warning("state.db auto-maintenance failed: %s", exc)
            result["error"] = str(exc)

        return result
