#!/usr/bin/env python3
"""Run memory-node store fact extraction on a dialogue export.

This script is intentionally standalone: it flattens all user/assistant turns
from a history_dialogue.json file, calls MemoryNodeManager.store_turn() for
each turn, and saves an isolated SessionDB under tmp/ by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import requests
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.memory_node_manager import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, MemoryNodeManager
from hermes_constants import get_hermes_home
from hermes_state import EMBEDDING_DIM, SessionDB


DEFAULT_INPUT = Path("/Users/zhouboyu/Downloads/history_dialogue.json")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tmp" / "memory_store_fact_test"
DEFAULT_LOG_PATH = REPO_ROOT / "tmp" / "memory_store_fact_test" / "memory_store_extraction_test.log"

SAMPLE_ID_RE = re.compile(r"(?:^|_)sample(\d+)$")


class StableEmbeddingClient:
    """Small deterministic embedding stand-in for store-path isolation."""

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


class StoreFactExtractionManager(MemoryNodeManager):
    """Use real retain extraction, but skip unrelated async graph work."""

    def __init__(self, *args: Any, report_rows: List[Dict[str, Any]], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._embedding_client = StableEmbeddingClient()
        self.report_rows = report_rows

    def _ensure_embedding_client(self) -> bool:
        self._embedding_client = StableEmbeddingClient()
        return True

    def _call_llm(self, prompt: str) -> str | None:
        url = f"{self._llm_base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._llm_api_key:
            headers["Authorization"] = f"Bearer {self._llm_api_key}"
        payload = {
            "model": self._llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
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

    def _start_async_work(self, **kwargs: Any) -> None:
        return None


def strip_jsonc(text: str) -> str:
    """Remove JSONC comments and trailing commas without touching strings."""
    output: List[str] = []
    in_string = False
    escape = False
    idx = 0
    while idx < len(text):
        char = text[idx]
        next_char = text[idx + 1] if idx + 1 < len(text) else ""
        if in_string:
            output.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            idx += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            idx += 1
            continue
        if char == "/" and next_char == "/":
            idx += 2
            while idx < len(text) and text[idx] not in "\r\n":
                idx += 1
            continue
        if char == "/" and next_char == "*":
            idx += 2
            while idx + 1 < len(text) and not (text[idx] == "*" and text[idx + 1] == "/"):
                idx += 1
            idx += 2
            continue
        output.append(char)
        idx += 1
    return re.sub(r",\s*([}\]])", r"\1", "".join(output))


def sample_hour_offset(sample_id: str, fallback_offset: int) -> int:
    match = SAMPLE_ID_RE.search(sample_id)
    if match:
        return int(match.group(1))
    return fallback_offset


def flatten_dialogue(path: Path) -> List[Tuple[str, int, str, str, int]]:
    data = json.loads(strip_jsonc(path.read_text(encoding="utf-8")))
    dialogue = data.get("dialogue") if isinstance(data, dict) else data
    if not isinstance(dialogue, list):
        raise ValueError("Expected JSON to contain a top-level dialogue list")

    turns: List[Tuple[str, int, str, str, int]] = []
    fallback_sample_offsets: Dict[str, int] = {}
    for group in dialogue:
        if not isinstance(group, dict):
            continue
        for sample_id, sample_turns in group.items():
            if not isinstance(sample_turns, list):
                continue
            sample_key = str(sample_id)
            if sample_key not in fallback_sample_offsets:
                fallback_sample_offsets[sample_key] = len(fallback_sample_offsets)
            hour_offset = sample_hour_offset(sample_key, fallback_sample_offsets[sample_key])
            for turn_index, turn in enumerate(sample_turns):
                if not isinstance(turn, dict):
                    continue
                user = str(turn.get("user") or "").strip()
                assistant = str(turn.get("assistant") or "").strip()
                if turn_index == (len(sample_turns) - 1):
                    is_last_turn = True
                else:
                    is_last_turn = False 
                if user and assistant:
                    turns.append((sample_key, turn_index, user, assistant, hour_offset, is_last_turn))
    return turns


def iter_stored_nodes(db: SessionDB, start_id: int) -> Iterable[Dict[str, Any]]:
    rows = db._conn.execute(
        """SELECT id, time_key, summary, keywords, topic, tags, fact_type, fact_kind, entity_names,
                  task_event_like, task_event_subject, task_relevance, original_dialog
             FROM memory_nodes
            WHERE id > ?
            ORDER BY id""",
        (start_id,),
    ).fetchall()
    for row in rows:
        item = dict(row)
        try:
            item["tags"] = json.loads(item.get("tags") or "[]")
        except json.JSONDecodeError:
            pass
        try:
            item["entity_names"] = json.loads(item.get("entity_names") or "[]")
        except json.JSONDecodeError:
            pass
        try:
            item["original_dialog"] = json.loads(item.get("original_dialog") or "{}")
        except json.JSONDecodeError:
            pass
        yield item


def load_hermes_config() -> Dict[str, Any]:
    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise MemoryNodeManager.store_turn fact extraction against history_dialogue.json."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--db-name", default="memory_store_fact_test.db")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--limit", type=int, default=0, help="Limit turns for smoke testing; 0 means all.")
    parser.add_argument("--start", type=int, default=0, help="Start offset in flattened turns.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output DB/report files.")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--llm-timeout", type=int)
    parser.add_argument(
        "--enable-reflect",
        action="store_true",
        help='Enable reflect',
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--manager-log-level", default="INFO")
    return parser.parse_args()


def remove_existing_outputs(db_path: Path, report_path: Path, overwrite: bool) -> None:
    related = [
        db_path,
        db_path.with_suffix(".faiss"),
        db_path.with_suffix(".faiss_ids.json"),
        report_path,
    ]
    existing = [path for path in related if path.exists()]
    if existing and not overwrite:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists. Pass --overwrite to replace:\n  {joined}")
    for path in existing:
        path.unlink()

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

def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(get_hermes_home() / ".env")

    args = parse_args()
    resolve_llm_args(args)
    configure_logging(args.log_path, args.log_level, args.manager_log_level)
    
    if not args.llm_api_key and args.llm_base_url.rstrip("/") == DEFAULT_LLM_BASE_URL:
        raise RuntimeError(
            "No OPENAI_API_KEY/HERMES_LLM_API_KEY found. Set one or pass --llm-base-url for a local compatible endpoint."
        )

    turns = flatten_dialogue(args.input)
    if args.start:
        turns = turns[args.start :]
    if args.limit:
        turns = turns[: args.limit]
    if not turns:
        raise RuntimeError("No user/assistant turns found in input")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.output_dir / args.db_name
    report_path = args.output_dir / "memory_store_fact_report.jsonl"
    remove_existing_outputs(db_path, report_path, args.overwrite)

    report_rows: List[Dict[str, Any]] = []
    db = SessionDB(db_path=db_path)
    manager = StoreFactExtractionManager(
        db,
        embedding_config={
            "llm_model": args.llm_model,
            "llm_base_url": args.llm_base_url,
            "llm_api_key": args.llm_api_key,
            "llm_timeout": args.llm_timeout,
            "enable_entity_extraction": False,
        },
        report_rows=report_rows,
    )

    stored_turns = 0
    stored_facts = 0
    reflect_runs = 0
    base_turn_timestamp = datetime.now().astimezone()
    try:
        with report_path.open("w", encoding="utf-8") as report:
            for local_index, (sample_id, turn_index, user, assistant, hour_offset, is_last_turn) in enumerate(turns):
                flat_index = args.start + local_index
                # Keep sample-level spacing at one hour for reflect tests, while
                # giving each turn a unique timestamp so memory_nodes.time_key
                # does not collide on "#00" across turns in the same sample.
                turn_timestamp = base_turn_timestamp + timedelta(hours=hour_offset, seconds=turn_index)
                before_id = db._conn.execute("SELECT COALESCE(MAX(id), 0) AS max_id FROM memory_nodes").fetchone()["max_id"]
                ok = manager.store_turn(
                    user,
                    assistant,
                    tags=["store_fact_test", f"sample:{sample_id}", f"turn:{turn_index}"],
                    turn_timestamp=turn_timestamp,
                )
                nodes = list(iter_stored_nodes(db, before_id))
                if ok:
                    stored_turns += 1
                    stored_facts += len(nodes)
                row = {
                    "event": "store_turn",
                    "flat_index": flat_index,
                    "sample_id": sample_id,
                    "turn_index": turn_index,
                    "sample_hour_offset": hour_offset,
                    "turn_second_offset": turn_index,
                    "turn_timestamp": turn_timestamp.isoformat(),
                    "stored": bool(ok),
                    "fact_count": len(nodes),
                    "user": user,
                    "assistant": assistant,
                    "facts": nodes,
                }
                report.write(json.dumps(row, ensure_ascii=False) + "\n")
                report.flush()
                logging.info(
                    "[%s/%s] %s turn=%s stored=%s facts=%s",
                    flat_index - args.start + 1,
                    len(turns),
                    sample_id,
                    turn_index,
                    ok,
                    len(nodes),
                )
                
                if args.enable_reflect and is_last_turn:
                    logging.info(
                        "Running reflect after sample %s",
                        sample_id,
                    )
                    reflect_report = manager.reflect()
                    reflect_runs += 1
                    reflect_row = {
                        "event": "reflect",
                        "after_sample_id": sample_id,
                        "after_flat_index": flat_index,
                        "report": reflect_report,
                    }
                    report.write(json.dumps(reflect_row, ensure_ascii=False, default=str) + "\n")
                    report.flush()
                    logging.info(
                        "Reflect after %s consolidated=%s interpretations=%s",
                        sample_id,
                        reflect_report.get("observations_consolidated"),
                        reflect_report.get("interpretations_generated"),
                    )
    finally:
        db.close()

    summary = {
        "input": str(args.input),
        "db_path": str(db_path),
        "report_path": str(report_path),
        "turns_processed": len(turns),
        "turns_with_facts": stored_turns,
        "facts_stored": stored_facts,
        "reflect_runs": reflect_runs,
        "llm_model": args.llm_model,
        "llm_base_url": args.llm_base_url,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
