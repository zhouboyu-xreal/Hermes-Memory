#!/usr/bin/env python3
"""Run recall probes against the DB created by test_memory_store_fact_extraction.

The first version intentionally uses a small hand-written query set. It opens
the fact-extraction test database, runs MemoryNodeManager.recall(), and writes a
full text log with the raw recall context plus section-level interpretation /
observation / fact output.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.memory_node_manager import (  # noqa: E402
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    EXPERIENCE_SECTION_HEADER,
    INTERPRETATION_SECTION_HEADER,
    MEMORY_NODE_HEADER,
    OBSERVATION_SECTION_HEADER,
    OBSERVATION_SUPPORT_SECTION_HEADER,
    WORLD_FACT_SECTION_HEADER,
    MemoryNodeManager,
)
from hermes_constants import get_hermes_home  # noqa: E402
from hermes_state import EMBEDDING_DIM, SessionDB  # noqa: E402


DEFAULT_DB_PATH = REPO_ROOT / "tmp" / "memory_store_fact_test" / "memory_store_fact_test.db"
DEFAULT_LOG_PATH = REPO_ROOT / "tmp" / "memory_store_fact_test" / "memory_store_recall_test.log"
DEFAULT_QUERIES = [
    "用户最近提到过哪些正在推进的任务？",
    "用户有哪些长期偏好或习惯？",
    "关于家庭或亲密关系有什么重要背景？",
    "用户最近遇到的技术问题和解决进展是什么？",
    "有哪些与工作项目相关的记忆？",
    "用户对助手协作方式有什么偏好？",
    "有没有尚未完成或需要跟进的事项？",
    "用户最近的情绪状态或压力来源是什么？",
    "关于健康、作息或生活安排有什么记忆？",
    "有哪些关于工具、代码库或环境配置的事实？",
]


class StableEmbeddingClient:
    """Deterministic embedding stand-in matching test_memory_store_fact_extraction."""

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros((1, EMBEDDING_DIM), dtype=np.float32)
        encoded = str(text or "").encode("utf-8")
        if not encoded:
            vec[0, 0] = 1.0
            return vec
        for idx, byte in enumerate(encoded):
            vec[0, (idx + byte) % EMBEDDING_DIM] += (byte % 17) + 1
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec


class RecallProbeManager(MemoryNodeManager):
    """Use real query summarisation, but deterministic embeddings for DB parity."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._embedding_client = StableEmbeddingClient()

    def _ensure_embedding_client(self, *args: Any, **kwargs: Any) -> bool:
        self._embedding_client = StableEmbeddingClient()
        return True

    def _call_llm(self, prompt: str) -> Optional[str]:
        url = f"{self._llm_base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._llm_api_key:
            headers["Authorization"] = f"Bearer {self._llm_api_key}"
        payload = {
            "model": self._llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 2048,
            "stream": False,
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self._llm_timeout)
            response.raise_for_status()
            choices = response.json().get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
        except requests.exceptions.RequestException as exc:
            logging.error("LLM request failed for %s with model %s: %s", url, self._llm_model, exc)
        except (KeyError, ValueError, TypeError) as exc:
            logging.error("LLM response parse failed for %s with model %s: %s", url, self._llm_model, exc)
        return ""


SECTION_HEADERS = [
    MEMORY_NODE_HEADER,
    INTERPRETATION_SECTION_HEADER,
    OBSERVATION_SECTION_HEADER,
    OBSERVATION_SUPPORT_SECTION_HEADER,
    WORLD_FACT_SECTION_HEADER,
    EXPERIENCE_SECTION_HEADER,
]


def load_hermes_config() -> Dict[str, Any]:
    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise MemoryNodeManager.recall against the fact-extraction test DB."
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--query", action="append", help="Recall query. Can be passed multiple times.")
    parser.add_argument(
        "--queries-file",
        type=Path,
        help="Optional UTF-8 file containing one recall query per line, or a JSON list of strings.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", default="mid", choices=["low", "mid", "high"])
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--llm-timeout", type=int)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--manager-log-level", default="INFO")
    return parser.parse_args()


def configure_logging(log_path: Path, log_level: str, manager_log_level: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    logging.getLogger("agent.memory_node_manager").setLevel(
        getattr(logging, str(manager_log_level).upper(), logging.INFO)
    )


def load_queries(args: argparse.Namespace) -> List[str]:
    queries: List[str] = []
    if args.queries_file:
        text = args.queries_file.read_text(encoding="utf-8").strip()
        if text.startswith("["):
            loaded = json.loads(text)
            if not isinstance(loaded, list):
                raise ValueError("--queries-file JSON must be a list of strings")
            queries.extend(str(item).strip() for item in loaded if str(item).strip())
        else:
            queries.extend(line.strip() for line in text.splitlines() if line.strip())
    if args.query:
        queries.extend(query.strip() for query in args.query if query.strip())
    return queries or list(DEFAULT_QUERIES)


def resolve_llm_args(args: argparse.Namespace) -> None:
    config = load_hermes_config()
    model_config = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    embedding_config = config.get("embedding", {}) if isinstance(config.get("embedding"), dict) else {}
    args.llm_model = (
        args.llm_model
        or os.getenv("HERMES_MEMORY_LLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or str(embedding_config.get("llm_model") or "")
        or str(model_config.get("default") or "")
        or DEFAULT_LLM_MODEL
    )
    args.llm_base_url = (
        args.llm_base_url
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("DEEPSEEK_BASE_URL")
        or str(model_config.get("base_url") or "")
        or DEFAULT_LLM_BASE_URL
    )
    if args.llm_base_url.rstrip("/") == "https://api.deepseek.com":
        args.llm_base_url = "https://api.deepseek.com/v1"
    args.llm_api_key = (
        args.llm_api_key
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("HERMES_LLM_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or str(model_config.get("api_key") or "")
        or ""
    )
    args.llm_timeout = args.llm_timeout or int(os.getenv("HERMES_MEMORY_LLM_TIMEOUT", "120"))


def split_recall_sections(memory_text: str) -> Dict[str, str]:
    if not memory_text.strip():
        return {}
    pattern = "(" + "|".join(re.escape(header) for header in SECTION_HEADERS) + ")"
    parts = re.split(pattern, memory_text)
    sections: Dict[str, str] = {}
    current_header = ""
    for part in parts:
        if not part:
            continue
        if part in SECTION_HEADERS:
            current_header = part
            sections[current_header] = ""
        elif current_header:
            sections[current_header] = (sections[current_header] + part).strip()
    return sections


def db_counts(db: SessionDB) -> Dict[str, int]:
    tables = ["memory_nodes", "memory_observations", "memory_interpretations", "entity_nodes"]
    counts: Dict[str, int] = {}
    for table in tables:
        row = db._conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        counts[table] = int(row["count"] if row else 0)
    return counts


def write_case_log(
    *,
    case_index: int,
    query: str,
    recalled: str,
    started_at: str,
    finished_at: str,
) -> None:
    logging.info("")
    logging.info("=" * 96)
    logging.info("RECALL CASE %02d", case_index)
    logging.info("query: %s", query)
    logging.info("started_at: %s", started_at)
    logging.info("finished_at: %s", finished_at)
    sections = split_recall_sections(recalled)
    if not recalled.strip():
        logging.info("raw_recall_context: <empty>")
        return

    logging.info("raw_recall_context:\n%s", recalled)
    for header in (
        INTERPRETATION_SECTION_HEADER,
        OBSERVATION_SECTION_HEADER,
        OBSERVATION_SUPPORT_SECTION_HEADER,
        WORLD_FACT_SECTION_HEADER,
        EXPERIENCE_SECTION_HEADER,
    ):
        section_text = sections.get(header, "").strip()
        logging.info("")
        logging.info("%s", header)
        logging.info("%s", section_text or "<empty>")


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(get_hermes_home() / ".env")

    args = parse_args()
    resolve_llm_args(args)
    configure_logging(args.log_path, args.log_level, args.manager_log_level)

    if not args.db_path.exists():
        raise FileNotFoundError(
            f"Memory test DB not found: {args.db_path}. "
            "Run scripts/test_memory_store_fact_extraction.py first."
        )
    if not args.llm_api_key and args.llm_base_url.rstrip("/") == DEFAULT_LLM_BASE_URL:
        raise RuntimeError(
            "No OPENAI_API_KEY/HERMES_LLM_API_KEY found. Set one or pass --llm-base-url for a local compatible endpoint."
        )

    queries = load_queries(args)
    logging.info("memory recall probe started")
    logging.info("db_path: %s", args.db_path)
    logging.info("log_path: %s", args.log_path)
    logging.info("query_count: %s", len(queries))
    logging.info("llm_model: %s", args.llm_model)
    logging.info("llm_base_url: %s", args.llm_base_url)

    db = SessionDB(db_path=args.db_path)
    manager = RecallProbeManager(
        db,
        embedding_config={
            "llm_model": args.llm_model,
            "llm_base_url": args.llm_base_url,
            "llm_api_key": args.llm_api_key,
            "llm_timeout": args.llm_timeout,
            "enable_entity_extraction": False,
        },
    )

    non_empty = 0
    try:
        logging.info("db_counts: %s", json.dumps(db_counts(db), ensure_ascii=False, sort_keys=True))
        for index, query in enumerate(queries, 1):
            started_at = datetime.now().astimezone().isoformat()
            logging.info("[%s/%s] recalling: %s", index, len(queries), query)
            recalled = manager.recall(query, top_k=args.top_k, budget=args.budget)
            finished_at = datetime.now().astimezone().isoformat()
            if recalled.strip():
                non_empty += 1
            write_case_log(
                case_index=index,
                query=query,
                recalled=recalled,
                started_at=started_at,
                finished_at=finished_at,
            )
    finally:
        db.close()

    summary = {
        "db_path": str(args.db_path),
        "log_path": str(args.log_path),
        "query_count": len(queries),
        "non_empty_recall_count": non_empty,
        "llm_model": args.llm_model,
        "llm_base_url": args.llm_base_url,
    }
    logging.info("summary: %s", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
