#!/usr/bin/env python3
"""Run Hermes memory retrieval on LongMemEval and emit evaluator JSONL.

This script builds an isolated memory DB per benchmark instance, replays the
timestamped history into ``MemoryNodeManager``, runs reflection, recalls memory
for the benchmark question, and uses a reader LLM to produce a final
``hypothesis`` answer.

The output format matches LongMemEval's evaluator expectations:
``{"question_id": "...", "hypothesis": "..."}``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.memory_node_manager import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, MemoryNodeManager
from hermes_cli.config import load_config
from hermes_cli.env_loader import load_hermes_dotenv
from hermes_constants import get_hermes_home
import hermes_state
from hermes_state import SessionDB


_READER_HTTP_SESSION: Optional[requests.Session] = None


DEFAULT_INPUT = Path(
    "/Users/zhouboyu/Documents/xreal/项目/agent_memory/benchmark/LongMemEval/data/longmemeval_oracle.json"
)
DEFAULT_OUTPUT_ROOT_DIR = REPO_ROOT / "tmp" / "longmemeval"


@dataclass(frozen=True)
class SessionReplayStats:
    turn_pairs_total: int
    stored_pairs: int
    skipped_assistant_only: int
    orphan_user_chunks: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Hermes memory benchmark inference on LongMemEval."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root-dir", type=Path, default=DEFAULT_OUTPUT_ROOT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Optional explicit run output directory. By default a run directory "
            "is created under --output-root-dir from the input filename and timestamp."
        ),
    )
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--detail-output", type=Path, help="Optional per-instance detail JSON.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all instances.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of worker processes used to build per-question memory DBs "
            "and recall contexts. Reader answering remains single-process."
        ),
    )
    parser.add_argument(
        "--question-id",
        action="append",
        help="Only run the specified question_id. Can be passed multiple times.",
    )
    parser.add_argument("--max-sessions", type=int, default=0, help="0 means all sessions.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output/state files for this run.",
    )
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--llm-timeout", type=int)
    parser.add_argument(
        "--llm-thinking",
        choices=("disabled", "enabled", "auto"),
        default="disabled",
        help=(
            "Control provider thinking mode for memory LLM calls. Defaults to "
            "disabled to match fact-extraction testing and avoid reasoning latency."
        ),
    )
    parser.add_argument(
        "--no-llm-json-mode",
        action="store_false",
        dest="llm_json_mode",
        help="Do not request provider-enforced JSON output for memory LLM calls.",
    )
    parser.set_defaults(llm_json_mode=True)
    parser.add_argument("--reader-model")
    parser.add_argument("--reader-base-url")
    parser.add_argument("--reader-api-key")
    parser.add_argument("--reader-timeout", type=int)
    parser.add_argument("--reader-max-tokens", type=int, default=4096)
    parser.add_argument("--reader-temperature", type=float, default=0.0)
    parser.add_argument(
        "--reader-max-context-chars",
        type=int,
        default=16000,
        help="Truncate recall context before sending it to the reader model.",
    )
    parser.add_argument(
        "--recall-top-k",
        type=int,
        default=8,
        help="Top-k passed to MemoryNodeManager.recall().",
    )
    parser.add_argument(
        "--recall-budget",
        default="mid",
        choices=["low", "mid", "high"],
        help="Recall traversal budget passed to MemoryNodeManager.recall().",
    )
    parser.add_argument(
        "--recall-gate-mode",
        default="force",
        choices=["auto", "force"],
        help=(
            "Recall gate behavior passed to MemoryNodeManager. "
            "'auto' uses gate analysis; 'force' always attempts recall."
        ),
    )
    parser.add_argument(
        "--enable-reflect",
        action="store_true",
        help='Enable reflect',
    )
    parser.add_argument(
        "--enable-feedback-analysis",
        action="store_true",
        help=(
            "Enable interpretation-feedback analysis and related recall-event "
            "tracking inside MemoryNodeManager during benchmark replay/recall."
        ),
    )
    parser.add_argument(
        "--reflect-every-sessions",
        type=int,
        default=1,
        help="Run reflect after every N replayed sessions. Default: 1.",
    )
    parser.add_argument(
        "--reflect-limit",
        type=int,
        default=100,
        help="Limit passed to MemoryNodeManager.reflect().",
    )
    parser.add_argument(
        "--fact-extraction-interval",
        type=int,
        default=1,
        help="Force fact extraction every N stored pairs. Default: 1 for benchmark fidelity.",
    )
    parser.add_argument(
        "--fact-extraction-max-chars",
        type=int,
        help="Override memory.max_chars_before_store from memory.yaml.",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--manager-log-level", default="INFO")
    return parser.parse_args()


def configure_logging(
    log_path: Path,
    log_level: str,
    manager_log_level: str,
    *,
    stream: bool = True,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    logging.getLogger("agent.memory_node_manager").setLevel(
        getattr(logging, str(manager_log_level).upper(), logging.INFO)
    )


def load_hermes_config() -> Dict[str, Any]:
    loaded = load_config()
    return loaded if isinstance(loaded, dict) else {}


def resolve_llm_args(args: argparse.Namespace) -> None:
    config = load_hermes_config()
    model_config = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    memory_config = config.get("memory", {}) if isinstance(config.get("memory"), dict) else {}

    args.llm_model = (
        args.llm_model
        or os.getenv("HERMES_MEMORY_LLM_MODEL")
        or os.getenv("OPENAI_MODEL")
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
    args.llm_timeout = (
        args.llm_timeout
        or int(
            os.getenv(
                "HERMES_MEMORY_LLM_TIMEOUT",
                str(memory_config.get("llm_timeout", 120)),
            )
        )
    )

    args.reader_model = args.reader_model or args.llm_model
    args.reader_base_url = args.reader_base_url or args.llm_base_url
    args.reader_api_key = args.reader_api_key or args.llm_api_key
    args.reader_timeout = args.reader_timeout or args.llm_timeout

    if args.reader_base_url.rstrip("/") == "https://api.deepseek.com":
        args.reader_base_url = "https://api.deepseek.com/v1"

    if not args.llm_api_key and args.llm_base_url.rstrip("/") == DEFAULT_LLM_BASE_URL:
        raise RuntimeError(
            "No OPENAI_API_KEY/HERMES_LLM_API_KEY found for memory LLM. "
            "Set one or pass --llm-base-url for a local compatible endpoint."
        )
    if not args.reader_api_key and args.reader_base_url.rstrip("/") == DEFAULT_LLM_BASE_URL:
        raise RuntimeError(
            "No OPENAI_API_KEY/HERMES_LLM_API_KEY found for reader LLM. "
            "Set one or pass --reader-base-url for a local compatible endpoint."
        )


def remove_existing_outputs(
    *,
    output_path: Path,
    detail_output_path: Optional[Path],
    state_dir: Path,
    overwrite: bool,
) -> None:
    existing: List[Path] = []
    if output_path.exists():
        existing.append(output_path)
    if detail_output_path and detail_output_path.exists():
        existing.append(detail_output_path)
    if state_dir.exists() and any(state_dir.iterdir()):
        existing.append(state_dir)

    if existing and not overwrite:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "Output already exists. Pass --overwrite to replace:\n  " + joined
        )

    if output_path.exists():
        output_path.unlink()
    if detail_output_path and detail_output_path.exists():
        detail_output_path.unlink()
    if state_dir.exists():
        shutil.rmtree(state_dir)


def _safe_output_name(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(value or "").strip()
    ).strip("._")
    return cleaned or "longmemeval"


def resolve_output_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    input_stem = _safe_output_name(args.input.stem)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_root_dir = Path(args.output_root_dir or DEFAULT_OUTPUT_ROOT_DIR)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else output_root_dir / f"{input_stem}_{timestamp}"
    )
    args.output_root_dir = output_root_dir
    args.output_dir = output_dir

    args.state_dir = Path(args.state_dir) if args.state_dir else output_dir / "state"
    args.log_path = (
        Path(args.log_path)
        if args.log_path
        else output_dir / "run_longmemeval_memory_eval.log"
    )

    brief_output = output_dir / f"{input_stem}_hermes.jsonl"
    if args.detail_output:
        detail_output = Path(args.detail_output)
        if not detail_output.is_absolute():
            detail_output = output_dir / detail_output
    else:
        detail_output = output_dir / f"{input_stem}_hermes.jsonl.details.json"
    args.detail_output = detail_output
    return brief_output, detail_output


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [item for item in data if isinstance(item, dict)]


def filter_instances(
    instances: Sequence[Dict[str, Any]],
    *,
    question_ids: Optional[Sequence[str]],
    start: int,
    limit: int,
) -> List[Dict[str, Any]]:
    filtered = list(instances)
    if question_ids:
        wanted = {str(item).strip() for item in question_ids if str(item).strip()}
        filtered = [
            item
            for item in filtered
            if str(item.get("question_id") or "").strip() in wanted
        ]
    if start:
        filtered = filtered[start:]
    if limit:
        filtered = filtered[:limit]
    return filtered


def parse_longmemeval_datetime(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Empty LongMemEval timestamp")
    if " (" in text and ") " in text:
        left, right = text.split(" (", 1)
        right = right.split(") ", 1)[-1]
        text = f"{left} {right}"
    return datetime.strptime(text, "%Y/%m/%d %H:%M")


def format_memory_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def normalize_unique_timestamp(
    candidate: datetime,
    *,
    seen_keys: set[str],
) -> datetime:
    current = candidate
    while True:
        key = format_memory_time(current)
        if key not in seen_keys:
            seen_keys.add(key)
            return current
        current += timedelta(seconds=1)


def sorted_history_sessions(
    item: Dict[str, Any],
    *,
    max_sessions: int,
) -> List[Tuple[str, datetime, List[Dict[str, Any]], int]]:
    session_ids = list(item.get("haystack_session_ids") or [])
    session_dates = list(item.get("haystack_dates") or [])
    sessions = list(item.get("haystack_sessions") or [])

    packed: List[Tuple[str, datetime, List[Dict[str, Any]], int]] = []
    for idx, session in enumerate(sessions):
        session_id = str(session_ids[idx] if idx < len(session_ids) else f"session_{idx}")
        session_date = parse_longmemeval_datetime(
            session_dates[idx] if idx < len(session_dates) else item.get("question_date")
        )
        packed.append((session_id, session_date, session if isinstance(session, list) else [], idx))

    packed.sort(key=lambda value: (value[1], value[0], value[3]))
    if max_sessions > 0:
        packed = packed[:max_sessions]
    return packed


def effective_question_datetime(
    question_dt: datetime,
    sessions: Sequence[Tuple[str, datetime, List[Dict[str, Any]], int]],
) -> datetime:
    if not sessions:
        return question_dt

    seen_timestamps: set[str] = set()
    latest_fact_dt: Optional[datetime] = None
    latest_session_dt = max(session_dt for _session_id, session_dt, _turns, _idx in sessions)

    for _session_id, session_dt, session_turns, _idx in sessions:
        pairs, _skipped, _orphaned = session_turn_pairs(session_turns)
        for pair_index, _pair in enumerate(pairs):
            fact_dt = normalize_unique_timestamp(
                session_dt + timedelta(seconds=pair_index),
                seen_keys=seen_timestamps,
            )
            latest_fact_dt = fact_dt

    if latest_fact_dt is None:
        return max(question_dt, latest_session_dt)

    # Keep recall's upper bound strictly after the final replayed fact so
    # second-level formatting does not exclude facts that share the same second.
    return max(question_dt, latest_fact_dt + timedelta(seconds=1))


def session_turn_pairs(session: Sequence[Dict[str, Any]]) -> Tuple[List[Tuple[str, str]], int, int]:
    pairs: List[Tuple[str, str]] = []
    pending_user_parts: List[str] = []
    skipped_assistant_only = 0
    orphan_user_chunks = 0

    for turn in session:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "").strip().lower()
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            pending_user_parts.append(content)
            continue
        if role != "assistant":
            continue
        if not pending_user_parts:
            skipped_assistant_only += 1
            continue
        pairs.append(("\n\n".join(pending_user_parts), content))
        pending_user_parts = []

    if pending_user_parts:
        orphan_user_chunks += 1

    return pairs, skipped_assistant_only, orphan_user_chunks


def prepare_runtime_configs(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config = load_hermes_config()
    embedding_config = (
        dict(config.get("embedding") or {})
        if isinstance(config.get("embedding"), dict)
        else {}
    )
    memory_config = (
        dict(config.get("memory") or {})
        if isinstance(config.get("memory"), dict)
        else {}
    )
    memory_config["min_turns_before_store"] = max(1, int(args.fact_extraction_interval))
    if args.fact_extraction_max_chars:
        memory_config["max_chars_before_store"] = max(1, int(args.fact_extraction_max_chars))
    memory_config["llm_timeout"] = int(args.llm_timeout)
    memory_config["llm_thinking"] = str(args.llm_thinking)
    memory_config["llm_json_mode"] = bool(args.llm_json_mode)
    memory_config["enable_interpretation_feedback"] = bool(
        args.enable_feedback_analysis
    )
    memory_config["recall_gate_mode"] = str(args.recall_gate_mode)
    return embedding_config, memory_config


def validate_runtime(manager: MemoryNodeManager) -> None:
    embedding_cfg = getattr(manager, "_embedding_cfg", {}) or {}
    configured_api_key = str(embedding_cfg.get("api_key") or "").strip()
    api_key_env = str(embedding_cfg.get("api_key_env") or "").strip()
    resolved_env_key = ""
    if configured_api_key.startswith("${") and configured_api_key.endswith("}"):
        resolved_env_key = configured_api_key[2:-1].strip()
    api_key_present = bool(
        (api_key_env and os.environ.get(api_key_env))
        or (resolved_env_key and os.environ.get(resolved_env_key))
        or (
            configured_api_key
            and not (
                configured_api_key.startswith("${")
                and configured_api_key.endswith("}")
            )
        )
        or os.environ.get("EMBEDDING_API_KEY")
    )
    logging.info(
        "Embedding runtime: python=%s faiss_available=%s provider=%s model=%s base_url=%s "
        "configured_dimensions=%s api_key_env=%s api_key_present=%s",
        sys.version.split()[0],
        hermes_state._HAS_FAISS,
        embedding_cfg.get("provider"),
        embedding_cfg.get("model"),
        embedding_cfg.get("base_url"),
        embedding_cfg.get("dimensions"),
        api_key_env or resolved_env_key or "EMBEDDING_API_KEY",
        api_key_present,
    )
    if not manager._ensure_embedding_client():
        raise RuntimeError("Failed to initialize the configured embedding client")
    probe = manager._embedding_client.embed_text("LongMemEval embedding probe")
    probe_vector = manager._as_embedding_vector(probe)
    if probe_vector is None:
        raw_shape = getattr(probe, "shape", None)
        raw_size = getattr(probe, "size", None)
        logging.error(
            "Embedding probe failed: provider=%s model=%s base_url=%s raw_type=%s raw_shape=%s "
            "raw_size=%s api_key_present=%s",
            embedding_cfg.get("provider"),
            embedding_cfg.get("model"),
            embedding_cfg.get("base_url"),
            type(probe).__name__,
            raw_shape,
            raw_size,
            api_key_present,
        )
        raise RuntimeError(
            "The configured embedding provider returned an invalid vector "
            f"(provider={embedding_cfg.get('provider')}, model={embedding_cfg.get('model')}, "
            f"base_url={embedding_cfg.get('base_url')}, api_key_present={api_key_present})"
        )
    logging.info(
        "Embedding probe succeeded: dimensions=%s norm=%.6f",
        probe_vector.size,
        float((probe_vector @ probe_vector) ** 0.5),
    )
    if not hermes_state._HAS_FAISS:
        logging.warning(
            "FAISS is unavailable in the active environment. Recall will still run, "
            "but benchmark retrieval quality may be lower than expected."
        )


def db_counts(db: SessionDB) -> Dict[str, int]:
    tables = {
        "facts": "memory_facts",
        "observations": "memory_observations",
        "interpretations": "memory_interpretations",
        "entities": "entity_nodes",
    }
    counts: Dict[str, int] = {}
    for key, table in tables.items():
        row = db._conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        counts[key] = int(row["count"] if row else 0)
    return counts


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max(0, max_chars - 64)
    return text[:keep] + "\n\n[truncated for reader]\n"


def reader_http_session() -> requests.Session:
    global _READER_HTTP_SESSION
    if _READER_HTTP_SESSION is None:
        _READER_HTTP_SESSION = requests.Session()
    return _READER_HTTP_SESSION


def log_snippet(value: Any, *, max_chars: int = 800) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def extract_chat_message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(text, dict) and isinstance(text.get("value"), str):
                    parts.append(text["value"])
        return "".join(parts)
    return str(value)


def extract_reader_final_answer(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""

    candidates = [raw]
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            candidates.append("\n".join(lines[1:-1]).strip())

    start = raw.find("{")
    end = raw.rfind("}")
    if 0 <= start < end:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            answer = parsed.get("final_answer")
            if answer is None:
                answer = parsed.get("answer")
            if answer is not None:
                answer_text = str(answer).strip()
                if answer_text:
                    return answer_text

    logging.warning("Reader response was not valid answer JSON: %s", log_snippet(raw))
    return ""


def call_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    logging.info(
        "Reader LLM request start: model=%s base_url=%s prompt_chars=%s max_tokens=%s "
        "temperature=%s timeout=%s",
        model,
        base_url.rstrip("/"),
        len(prompt),
        max_tokens,
        temperature,
        timeout,
    )
    started = time.perf_counter()
    try:
        response = reader_http_session().post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logging.exception(
            "Reader LLM request failed: model=%s elapsed_ms=%.2f error=%s",
            model,
            elapsed_ms,
            exc,
        )
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000
    logging.info(
        "Reader LLM response received: model=%s status=%s elapsed_ms=%.2f response_chars=%s",
        model,
        response.status_code,
        elapsed_ms,
        len(response.text or ""),
    )
    try:
        response.raise_for_status()
    except requests.HTTPError:
        logging.error(
            "Reader LLM HTTP error: model=%s status=%s body=%s",
            model,
            response.status_code,
            log_snippet(response.text),
        )
        raise

    try:
        response_data = response.json()
    except ValueError:
        logging.error(
            "Reader LLM returned non-JSON response: model=%s body=%s",
            model,
            log_snippet(response.text),
        )
        raise

    choices = response_data.get("choices", [])
    if not isinstance(choices, list) or not choices:
        logging.error(
            "Reader LLM returned no choices: model=%s response_keys=%s usage=%s body=%s",
            model,
            sorted(response_data.keys()) if isinstance(response_data, dict) else [],
            response_data.get("usage") if isinstance(response_data, dict) else None,
            log_snippet(response_data),
        )
        raise RuntimeError(f"Reader LLM returned no choices: {log_snippet(response_data)}")

    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = extract_chat_message_text(message.get("content")).strip()
    fallback_text = extract_chat_message_text(choice.get("text")).strip()
    if not content and fallback_text:
        logging.warning(
            "Reader LLM returned text outside message.content: model=%s finish_reason=%s text_chars=%s",
            model,
            choice.get("finish_reason"),
            len(fallback_text),
        )
        content = fallback_text
    reasoning = extract_chat_message_text(
        message.get("reasoning_content") or message.get("reasoning")
    )
    usage = response_data.get("usage") if isinstance(response_data, dict) else None
    finish_reason = choice.get("finish_reason")
    logging.info(
        "Reader LLM response parsed: model=%s finish_reason=%s content_chars=%s "
        "reasoning_chars=%s message_keys=%s usage=%s",
        model,
        finish_reason,
        len(content),
        len(reasoning),
        sorted(message.keys()),
        usage,
    )
    if not content:
        logging.warning(
            "Reader LLM returned empty message.content: model=%s finish_reason=%s "
            "reasoning_chars=%s message_keys=%s usage=%s",
            model,
            finish_reason,
            len(reasoning),
            sorted(message.keys()),
            usage,
        )
        raise RuntimeError(
            "Reader LLM returned empty message.content "
            f"(model={model}, finish_reason={finish_reason}, usage={usage})"
        )
    return content


def build_reader_prompt(
    *,
    question: str,
    question_type: str,
    question_date: str,
    memory_context: str,
) -> str:
    return (
        "You are answering a LongMemEval benchmark question using retrieved memory only.\n"
        "Follow these rules strictly:\n"
        "1. Use only the memory context below. Do not invent details.\n"
        "2. First identify the 1-5 most relevant memory lines or snippets.\n"
        "3. Ignore irrelevant background memories once you have found the relevant evidence.\n"
        "4. If the answer text itself appears in memory, return it instead of abstaining.\n"
        "5. For counting, totaling, or comparison questions, gather all relevant items before answering.\n"
        "6. For temporal or knowledge-update questions, prefer the most recent applicable evidence before the question date.\n"
        "7. If the memory truly lacks the answer, set final_answer to exactly: "
        "\"I don't know based on the available memory.\"\n"
        "8. Keep the final answer concise and direct.\n\n"
        f"Question type: {question_type}\n"
        f"Question date: {question_date}\n"
        f"Question: {question}\n\n"
        "Memory context:\n"
        f"{memory_context or '[empty]'}\n\n"
        "Return valid JSON only, with no markdown or extra text. Use exactly this schema:\n"
        "{\n"
        "  \"relevant_evidence\": [\"<most relevant memory snippet>\", \"<optional>\"],\n"
        "  \"final_answer\": \"<final answer or exactly I don't know based on the available memory.>\"\n"
        "}\n"
    )


def answer_question_with_reader(
    *,
    args: argparse.Namespace,
    question: str,
    question_type: str,
    question_date: str,
    memory_context: str,
) -> str:
    if not memory_context.strip():
        return "I don't know based on the available memory."

    truncated_context = truncate_text(memory_context, args.reader_max_context_chars)
    prompt = build_reader_prompt(
        question=question,
        question_type=question_type,
        question_date=question_date,
        memory_context=truncated_context,
    )
    response_text = call_chat_completion(
        base_url=args.reader_base_url,
        api_key=args.reader_api_key,
        model=args.reader_model,
        prompt=prompt,
        timeout=int(args.reader_timeout),
        max_tokens=int(args.reader_max_tokens),
        temperature=float(args.reader_temperature),
    )
    answer = extract_reader_final_answer(response_text)
    if answer:
        return answer
    return "I don't know based on the available memory."


def replay_sessions_into_memory(
    *,
    manager: MemoryNodeManager,
    item: Dict[str, Any],
    sessions: Sequence[Tuple[str, datetime, List[Dict[str, Any]], int]],
    enable_reflect: bool,
    reflect_every_sessions: int,
    reflect_limit: int,
) -> Tuple[SessionReplayStats, int]:
    seen_timestamps: set[str] = set()
    turn_pairs_total = 0
    stored_pairs = 0
    skipped_assistant_only = 0
    orphan_user_chunks = 0
    reflect_runs = 0

    for session_position, (session_id, session_dt, session_turns, _original_index) in enumerate(sessions, 1):
        pairs, skipped, orphaned = session_turn_pairs(session_turns)
        skipped_assistant_only += skipped
        orphan_user_chunks += orphaned

        for pair_index, (user_message, assistant_response) in enumerate(pairs):
            turn_pairs_total += 1
            turn_dt = normalize_unique_timestamp(
                session_dt + timedelta(seconds=pair_index),
                seen_keys=seen_timestamps,
            )
            stored = manager.store_turn(
                user_message,
                assistant_response,
                tags=[
                    "longmemeval",
                    f"question_type:{item.get('question_type')}",
                    f"question_id:{item.get('question_id')}",
                    f"session_id:{session_id}",
                    f"session_index:{session_position}",
                ],
                turn_timestamp=turn_dt,
            )
            if stored:
                stored_pairs += 1

        if enable_reflect and session_position % reflect_every_sessions == 0:
            reflect_ts = normalize_unique_timestamp(
                session_dt + timedelta(seconds=max(len(pairs), 1)),
                seen_keys=seen_timestamps,
            )
            manager.reflect(
                limit=reflect_limit,
                reflect_timestamp=reflect_ts,
            )
            reflect_runs += 1
    
    if manager._pending_store_turns:
        flushed = manager.flush_pending_store_turns()
        if flushed:
            stored_pairs += 1
    
    if enable_reflect and sessions:
        if manager._pending_store_turns:
            flushed = manager.flush_pending_store_turns()
            if flushed:
                stored_pairs += 1
        final_ts = normalize_unique_timestamp(
            sessions[-1][1] + timedelta(seconds=3599),
            seen_keys=seen_timestamps,
        )
        manager.reflect(
            limit=reflect_limit,
            reflect_timestamp=final_ts,
        )
        reflect_runs += 1
    
    if manager._pending_store_turns:
        logging.warning(
            "Replay finished with %s pending turns still buffered. "
            "The final batch could not be force-stored.",
            len(manager._pending_store_turns),
        )

    return (
        SessionReplayStats(
            turn_pairs_total=turn_pairs_total,
            stored_pairs=stored_pairs,
            skipped_assistant_only=skipped_assistant_only,
            orphan_user_chunks=orphan_user_chunks,
        ),
        reflect_runs,
    )


def build_instance_memory_context(
    *,
    args: argparse.Namespace,
    item: Dict[str, Any],
    embedding_config: Dict[str, Any],
    memory_config: Dict[str, Any],
) -> Dict[str, Any]:
    question_id = str(item.get("question_id") or "unknown_question")
    question = str(item.get("question") or "").strip()
    question_type = str(item.get("question_type") or "").strip()
    question_date_text = str(item.get("question_date") or "").strip()
    question_dt = parse_longmemeval_datetime(question_date_text)

    question_state_dir = args.state_dir / question_id
    question_state_dir.mkdir(parents=True, exist_ok=True)
    db_path = question_state_dir / "memory.db"

    db = SessionDB(db_path=db_path)
    try:
        manager = MemoryNodeManager(
            db,
            embedding_config=embedding_config,
            memory_config=memory_config,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key,
        )
        validate_runtime(manager)

        sessions = sorted_history_sessions(item, max_sessions=int(args.max_sessions))
        replay_stats, reflect_runs = replay_sessions_into_memory(
            manager=manager,
            item=item,
            sessions=sessions,
            enable_reflect=args.enable_reflect,
            reflect_every_sessions=max(1, int(args.reflect_every_sessions)),
            reflect_limit=int(args.reflect_limit),
        )
        effective_question_dt = effective_question_datetime(question_dt, sessions)
        effective_question_date_text = format_memory_time(effective_question_dt)
        counts = db_counts(db)
        memory_context = manager.recall(
            question,
            top_k=int(args.recall_top_k),
            budget=str(args.recall_budget),
            time_end=effective_question_date_text,
            recall_gate_mode=str(args.recall_gate_mode),
        )

        return {
            "question_id": question_id,
            "question_type": question_type,
            "question": question,
            "question_date": question_date_text,
            "effective_question_date": effective_question_date_text,
            "answer": str(item.get("answer") or ""),
            "history_session_count": len(sessions),
            "replayed_turn_pairs": replay_stats.turn_pairs_total,
            "turn_pairs_with_stored_facts": replay_stats.stored_pairs,
            "skipped_assistant_only_turns": replay_stats.skipped_assistant_only,
            "orphan_user_chunks": replay_stats.orphan_user_chunks,
            "reflect_runs": reflect_runs,
            "db_path": str(db_path),
            "db_counts": counts,
            "recall_context_chars": len(memory_context or ""),
            "recall_context": memory_context,
        }
    finally:
        db.close()


def answer_instance_from_memory_context(
    *,
    args: argparse.Namespace,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    output = dict(result)
    output["hypothesis"] = answer_question_with_reader(
        args=args,
        question=str(result.get("question") or ""),
        question_type=str(result.get("question_type") or ""),
        question_date=str(result.get("effective_question_date") or ""),
        memory_context=str(result.get("recall_context") or ""),
    )
    return output


def run_instance(
    *,
    args: argparse.Namespace,
    item: Dict[str, Any],
    embedding_config: Dict[str, Any],
    memory_config: Dict[str, Any],
) -> Dict[str, Any]:
    memory_result = build_instance_memory_context(
        args=args,
        item=item,
        embedding_config=embedding_config,
        memory_config=memory_config,
    )
    return answer_instance_from_memory_context(args=args, result=memory_result)


def run_instance_memory_context_worker(
    payload: Tuple[int, int, Dict[str, Any], argparse.Namespace, Dict[str, Any], Dict[str, Any]],
) -> Tuple[int, Dict[str, Any]]:
    index, total, item, args, embedding_config, memory_config = payload
    question_id = str(item.get("question_id") or f"item_{index}")
    question_state_dir = args.state_dir / question_id
    question_state_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(
        question_state_dir / "run_longmemeval_memory_eval.log",
        args.log_level,
        args.manager_log_level,
        stream=False,
    )
    logging.info(
        "[%s/%s] Running LongMemEval instance %s (%s)",
        index,
        total,
        question_id,
        item.get("question_type"),
    )
    result = build_instance_memory_context(
        args=args,
        item=item,
        embedding_config=embedding_config,
        memory_config=memory_config,
    )
    logging.info(
        "[%s/%s] Finished %s: sessions=%s facts=%s observations=%s interpretations=%s recall_chars=%s",
        index,
        total,
        question_id,
        result["history_session_count"],
        result["db_counts"]["facts"],
        result["db_counts"]["observations"],
        result["db_counts"]["interpretations"],
        result["recall_context_chars"],
    )
    return index, result


def write_jsonl_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json_output(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(list(rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    loaded_env_paths = load_hermes_dotenv(
        hermes_home=get_hermes_home(),
        project_env=REPO_ROOT / ".env",
    )
    args = parse_args()
    resolve_llm_args(args)
    brief_output, detail_output = resolve_output_paths(args)
    remove_existing_outputs(
        output_path=brief_output,
        detail_output_path=detail_output,
        state_dir=args.state_dir,
        overwrite=bool(args.overwrite),
    )
    brief_output.parent.mkdir(parents=True, exist_ok=True)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(args.log_path, args.log_level, args.manager_log_level)
    if loaded_env_paths:
        logging.info(
            "Loaded environment variables from: %s",
            ", ".join(str(path) for path in loaded_env_paths),
        )
    else:
        logging.info("No Hermes .env file found; using system environment variables")
    instances = load_dataset(args.input)
    selected = filter_instances(
        instances,
        question_ids=args.question_id,
        start=int(args.start),
        limit=int(args.limit),
    )
    if not selected:
        raise RuntimeError("No LongMemEval instances selected")

    embedding_config, memory_config = prepare_runtime_configs(args)
    success_count = 0
    failure_count = 0
    memory_results_by_index: Dict[int, Dict[str, Any]] = {}
    errors_by_index: Dict[int, Dict[str, str]] = {}
    detail_rows: List[Dict[str, Any]] = []

    workers = max(1, int(args.workers or 1))
    if workers > 1:
        logging.info(
            "Building memory contexts with %s worker processes; reader answers run in main process",
            workers,
        )
        payloads = [
            (index, len(selected), item, args, embedding_config, memory_config)
            for index, item in enumerate(selected, 1)
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_instance_memory_context_worker, payload): payload
                for payload in payloads
            }
            for future in as_completed(futures):
                index, _total, item, _args, _embedding_config, _memory_config = futures[future]
                question_id = str(item.get("question_id") or f"item_{index}")
                try:
                    result_index, result = future.result()
                    memory_results_by_index[result_index] = result
                    logging.info(
                        "[%s/%s] Memory context ready for %s: recall_chars=%s",
                        result_index,
                        len(selected),
                        question_id,
                        result.get("recall_context_chars"),
                    )
                except Exception as exc:
                    failure_count += 1
                    errors_by_index[index] = {
                        "question_id": question_id,
                        "status": "error",
                        "stage": "memory_context",
                        "error": str(exc),
                    }
                    logging.exception(
                        "Failed to build memory context for LongMemEval instance %s: %s",
                        question_id,
                        exc,
                    )
    else:
        for index, item in enumerate(selected, 1):
            question_id = str(item.get("question_id") or f"item_{index}")
            logging.info(
                "[%s/%s] Running LongMemEval instance %s (%s)",
                index,
                len(selected),
                question_id,
                item.get("question_type"),
            )
            try:
                memory_results_by_index[index] = build_instance_memory_context(
                    args=args,
                    item=item,
                    embedding_config=embedding_config,
                    memory_config=memory_config,
                )
                result = memory_results_by_index[index]
                logging.info(
                    "[%s/%s] Memory context ready for %s: sessions=%s facts=%s observations=%s interpretations=%s recall_chars=%s",
                    index,
                    len(selected),
                    question_id,
                    result["history_session_count"],
                    result["db_counts"]["facts"],
                    result["db_counts"]["observations"],
                    result["db_counts"]["interpretations"],
                    result["recall_context_chars"],
                )
            except Exception as exc:
                failure_count += 1
                errors_by_index[index] = {
                    "question_id": question_id,
                    "status": "error",
                    "stage": "memory_context",
                    "error": str(exc),
                }
                logging.exception(
                    "Failed to build memory context for LongMemEval instance %s: %s",
                    question_id,
                    exc,
                )

    for index, item in enumerate(selected, 1):
        question_id = str(item.get("question_id") or f"item_{index}")
        if index in errors_by_index:
            detail_rows.append(errors_by_index[index])
            continue
        result = memory_results_by_index.get(index)
        if result is None:
            failure_count += 1
            error_row = {
                "question_id": question_id,
                "status": "error",
                "stage": "memory_context",
                "error": "Memory context worker returned no result",
            }
            detail_rows.append(error_row)
            continue
        try:
            result = answer_instance_from_memory_context(args=args, result=result)
            write_jsonl_row(
                brief_output,
                {
                    "question_id": result["question_id"],
                    "hypothesis": result["hypothesis"],
                },
            )
            detail_rows.append(result)
            success_count += 1
            logging.info(
                "[%s/%s] Answered %s: recall_chars=%s hypothesis_chars=%s",
                index,
                len(selected),
                question_id,
                result["recall_context_chars"],
                len(result.get("hypothesis") or ""),
            )
        except Exception as exc:
            failure_count += 1
            logging.exception("Failed to answer LongMemEval instance %s: %s", question_id, exc)
            detail_rows.append(
                {
                    "question_id": question_id,
                    "status": "error",
                    "stage": "reader",
                    "error": str(exc),
                    "recall_context": result.get("recall_context", ""),
                    "recall_context_chars": result.get("recall_context_chars", 0),
                }
            )

    write_json_output(detail_output, detail_rows)

    summary = {
        "input": str(args.input),
        "output_root_dir": str(args.output_root_dir),
        "output_dir": str(args.output_dir),
        "output": str(brief_output),
        "detail_output": str(detail_output),
        "state_dir": str(args.state_dir),
        "log_path": str(args.log_path),
        "instances_requested": len(selected),
        "instances_succeeded": success_count,
        "instances_failed": failure_count,
        "llm_model": args.llm_model,
        "reader_model": args.reader_model,
        "llm_thinking": args.llm_thinking,
        "llm_json_mode": args.llm_json_mode,
        "workers": workers,
        "recall_top_k": args.recall_top_k,
        "recall_budget": args.recall_budget,
        "recall_gate_mode": args.recall_gate_mode,
        "feedback_analysis_enabled": args.enable_feedback_analysis,
        "reflect_every_sessions": args.reflect_every_sessions,
        "fact_extraction_interval": args.fact_extraction_interval,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
