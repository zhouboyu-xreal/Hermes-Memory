#!/usr/bin/env python3
"""Run Hermes memory retrieval on LoCoMo and emit LoCoMo-style context JSON.

This script mirrors ``run_longmemeval_memory_eval.py`` for the LoCoMo data
shape. It builds an isolated memory DB per LoCoMo conversation, replays the
conversation into ``MemoryNodeManager``, and recalls memory for each annotated
QA question.

The main output is a JSON list shaped like LoCoMo's evaluator output:
``[{"sample_id": "...", "qa": [{"question": "...", "hermes-memory_context_text": "..."}]}]``.
The detail output is written once by the main process as a JSON list.
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


DEFAULT_INPUT = Path(
    "/Users/zhouboyu/Documents/xreal/项目/agent_memory/benchmark/locomo/data/locomo10.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "tmp" / "locomo" / "locomo10_hermes_memory.json"
DEFAULT_STATE_DIR = REPO_ROOT / "tmp" / "locomo" / "state"
DEFAULT_LOG_PATH = REPO_ROOT / "tmp" / "locomo" / "run_locomo_memory_eval.log"
DEFAULT_CONTEXT_KEY = "hermes-memory_context_text"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_READER_BASE_URL = "https://api.openai.com/v1"
DEFAULT_PREDICTION_KEY = "hermes-memory_prediction"
NO_INFORMATION_ANSWER = "No information available."
_READER_HTTP_SESSION: Optional[requests.Session] = None


@dataclass(frozen=True)
class LocomoReplayStats:
    dialog_turns_total: int
    turn_pairs_total: int
    stored_pairs: int
    orphan_dialog_turns: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Hermes memory benchmark inference on LoCoMo."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--detail-output", type=Path, help="Optional per-QA detail JSON.")
    parser.add_argument("--context-key", default=DEFAULT_CONTEXT_KEY)
    parser.add_argument("--prediction-key", default=DEFAULT_PREDICTION_KEY)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all samples.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes used to build per-sample memory DBs and recall contexts.",
    )
    parser.add_argument("--sample-id", action="append", help="Only run the given sample_id.")
    parser.add_argument("--qa-start", type=int, default=0)
    parser.add_argument("--qa-limit", type=int, default=0, help="0 means all QA rows per sample.")
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
    parser.add_argument("--reader-max-tokens", type=int, default=1024)
    parser.add_argument("--reader-temperature", type=float, default=0.0)
    parser.add_argument(
        "--reader-max-context-chars",
        type=int,
        default=16000,
        help="Truncate recall context before sending it to the reader model.",
    )
    parser.add_argument("--recall-top-k", type=int, default=8)
    parser.add_argument("--recall-budget", default="mid", choices=["low", "mid", "high"])
    parser.add_argument("--enable-reflect", action="store_true")
    parser.add_argument(
        "--enable-feedback-analysis",
        action="store_true",
        help="Enable interpretation-feedback analysis inside MemoryNodeManager.",
    )
    parser.add_argument("--reflect-every-sessions", type=int, default=1)
    parser.add_argument("--reflect-limit", type=int, default=100)
    parser.add_argument(
        "--fact-extraction-interval",
        type=int,
        default=1,
        help="Force fact extraction every N replayed dialog pairs.",
    )
    parser.add_argument("--fact-extraction-max-chars", type=int)
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
    if args.llm_base_url.rstrip("/") in [
        DEEPSEEK_BASE_URL,
        DEEPSEEK_BASE_URL + "/v1",
    ]:
        args.llm_base_url = DEEPSEEK_BASE_URL
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

    if args.reader_base_url.rstrip("/") in [
        DEEPSEEK_BASE_URL,
        DEEPSEEK_BASE_URL + "/v1",
    ]:
        args.reader_base_url = DEEPSEEK_BASE_URL + "/v1"

    if not args.llm_api_key and args.llm_base_url.rstrip("/") == DEFAULT_LLM_BASE_URL:
        raise RuntimeError(
            "No OPENAI_API_KEY/HERMES_LLM_API_KEY found for memory LLM. "
            "Set one or pass --llm-base-url for a local compatible endpoint."
        )
    if not args.reader_api_key and args.reader_base_url.rstrip("/") == DEFAULT_READER_BASE_URL:
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


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [item for item in data if isinstance(item, dict)]


def filter_samples(
    samples: Sequence[Dict[str, Any]],
    *,
    sample_ids: Optional[Sequence[str]],
    start: int,
    limit: int,
) -> List[Dict[str, Any]]:
    filtered = list(samples)
    if sample_ids:
        wanted = {str(item).strip() for item in sample_ids if str(item).strip()}
        filtered = [
            item for item in filtered if str(item.get("sample_id") or "").strip() in wanted
        ]
    if start:
        filtered = filtered[start:]
    if limit:
        filtered = filtered[:limit]
    return filtered


def parse_locomo_datetime(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Empty LoCoMo timestamp")
    return datetime.strptime(text, "%I:%M %p on %d %B, %Y")


def format_memory_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def normalize_unique_timestamp(candidate: datetime, *, seen_keys: set[str]) -> datetime:
    current = candidate
    while True:
        key = format_memory_time(current)
        if key not in seen_keys:
            seen_keys.add(key)
            return current
        current += timedelta(seconds=1)


def sorted_locomo_sessions(
    sample: Dict[str, Any],
    *,
    max_sessions: int,
) -> List[Tuple[int, datetime, str, List[Dict[str, Any]]]]:
    conversation = sample.get("conversation") if isinstance(sample.get("conversation"), dict) else {}
    session_nums = sorted(
        int(key.split("_")[-1])
        for key in conversation
        if key.startswith("session_") and not key.endswith("_date_time")
    )
    packed: List[Tuple[int, datetime, str, List[Dict[str, Any]]]] = []
    for session_num in session_nums:
        session_key = f"session_{session_num}"
        date_key = f"{session_key}_date_time"
        date_text = str(conversation.get(date_key) or "")
        session_dt = parse_locomo_datetime(date_text)
        turns = conversation.get(session_key)
        packed.append((session_num, session_dt, date_text, turns if isinstance(turns, list) else []))
    if max_sessions > 0:
        packed = packed[:max_sessions]
    return packed


def paired_dialog_turns(
    turns: Sequence[Dict[str, Any]],
    *,
    date_text: str,
) -> Tuple[List[Tuple[str, str, List[str]]], int]:
    pairs: List[Tuple[str, str, List[str]]] = []
    orphan_dialog_turns = 0
    index = 0
    while index < len(turns):
        first = turns[index] if isinstance(turns[index], dict) else {}
        second = turns[index + 1] if index + 1 < len(turns) and isinstance(turns[index + 1], dict) else None
        if second is None:
            orphan_dialog_turns += 1
            break
        first_text = dialog_turn_text(first)
        second_text = dialog_turn_text(second)
        if not first_text or not second_text:
            orphan_dialog_turns += 1
            index += 2
            continue
        user_message = f"LoCoMo conversation date: {date_text}\n{first_text}"
        assistant_response = f"LoCoMo conversation date: {date_text}\n{second_text}"
        dia_ids = [
            str(turn.get("dia_id") or "").strip()
            for turn in (first, second)
            if str(turn.get("dia_id") or "").strip()
        ]
        pairs.append((user_message, assistant_response, dia_ids))
        index += 2
    return pairs, orphan_dialog_turns


def dialog_turn_text(turn: Dict[str, Any]) -> str:
    speaker = str(turn.get("speaker") or "Unknown speaker").strip()
    text = str(turn.get("text") or "").strip()
    if not text:
        return ""
    output = f'{speaker} said: "{text}"'
    caption = str(turn.get("blip_caption") or "").strip()
    if caption:
        output += f" The shared image was captioned: {caption}"
    dia_id = str(turn.get("dia_id") or "").strip()
    if dia_id:
        output = f"[{dia_id}] " + output
    return output


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
    memory_config["enable_interpretation_feedback"] = bool(args.enable_feedback_analysis)
    return embedding_config, memory_config


def validate_runtime(manager: MemoryNodeManager) -> None:
    logging.info(
        "Embedding runtime: python=%s faiss_available=%s",
        sys.version.split()[0],
        hermes_state._HAS_FAISS,
    )
    if not manager._ensure_embedding_client():
        raise RuntimeError("Failed to initialize the configured embedding client")
    probe = manager._embedding_client.embed_text("LoCoMo embedding probe")
    probe_vector = manager._as_embedding_vector(probe)
    if probe_vector is None:
        raise RuntimeError("The configured embedding provider returned an invalid vector")
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
    category: Any,
    memory_context: str,
) -> str:
    return (
        "You are answering a LoCoMo long-term conversational memory benchmark question.\n"
        "Use only the retrieved Hermes memory context below.\n"
        "Rules:\n"
        "1. Answer with a short phrase, not a full explanation.\n"
        "2. Use exact names, dates, places, and words from memory whenever possible.\n"
        "3. If the question asks when something happened, answer with the best date or approximate date from memory.\n"
        "4. If memory does not contain enough information, answer exactly: No information available.\n"
        "5. Do not mention that you used memory.\n\n"
        f"LoCoMo category: {category}\n"
        f"Question: {question}\n\n"
        "Retrieved Hermes memory:\n"
        f"{memory_context or '[empty]'}\n\n"
        "Short answer:"
    )


def answer_question_with_reader(
    *,
    args: argparse.Namespace,
    question: str,
    category: Any,
    memory_context: str,
) -> str:
    if not str(memory_context or "").strip():
        return NO_INFORMATION_ANSWER
    prompt = build_reader_prompt(
        question=question,
        category=category,
        memory_context=truncate_text(
            str(memory_context),
            int(args.reader_max_context_chars),
        ),
    )
    return call_chat_completion(
        base_url=args.reader_base_url,
        api_key=args.reader_api_key,
        model=args.reader_model,
        prompt=prompt,
        timeout=int(args.reader_timeout),
        max_tokens=int(args.reader_max_tokens),
        temperature=float(args.reader_temperature),
    )


def replay_sample_into_memory(
    *,
    manager: MemoryNodeManager,
    sample: Dict[str, Any],
    sessions: Sequence[Tuple[int, datetime, str, List[Dict[str, Any]]]],
    enable_reflect: bool,
    reflect_every_sessions: int,
    reflect_limit: int,
) -> Tuple[LocomoReplayStats, int]:
    seen_timestamps: set[str] = set()
    dialog_turns_total = 0
    turn_pairs_total = 0
    stored_pairs = 0
    orphan_dialog_turns = 0
    reflect_runs = 0
    sample_id = str(sample.get("sample_id") or "unknown_sample")

    for session_position, (session_num, session_dt, date_text, turns) in enumerate(sessions, 1):
        dialog_turns_total += len(turns)
        pairs, orphaned = paired_dialog_turns(turns, date_text=date_text)
        orphan_dialog_turns += orphaned
        for pair_index, (user_message, assistant_response, dia_ids) in enumerate(pairs):
            turn_pairs_total += 1
            turn_dt = normalize_unique_timestamp(
                session_dt + timedelta(seconds=pair_index),
                seen_keys=seen_timestamps,
            )
            stored = manager.store_turn(
                user_message,
                assistant_response,
                tags=[
                    "locomo",
                    f"sample_id:{sample_id}",
                    f"session_id:S{session_num}",
                    *[f"dia_id:{dia_id}" for dia_id in dia_ids],
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
            manager.reflect(limit=reflect_limit, reflect_timestamp=reflect_ts)
            reflect_runs += 1

    if manager._pending_store_turns:
        flushed = manager.flush_pending_store_turns()
        if flushed:
            stored_pairs += 1

    if enable_reflect and sessions:
        final_ts = normalize_unique_timestamp(
            sessions[-1][1] + timedelta(seconds=3599),
            seen_keys=seen_timestamps,
        )
        manager.reflect(limit=reflect_limit, reflect_timestamp=final_ts)
        reflect_runs += 1

    if manager._pending_store_turns:
        logging.warning(
            "Replay finished with %s pending turns still buffered for %s.",
            len(manager._pending_store_turns),
            sample_id,
        )

    return (
        LocomoReplayStats(
            dialog_turns_total=dialog_turns_total,
            turn_pairs_total=turn_pairs_total,
            stored_pairs=stored_pairs,
            orphan_dialog_turns=orphan_dialog_turns,
        ),
        reflect_runs,
    )


def selected_qa_rows(
    sample: Dict[str, Any],
    *,
    qa_start: int,
    qa_limit: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    qas = sample.get("qa") if isinstance(sample.get("qa"), list) else []
    indexed = list(enumerate(qas))
    if qa_start:
        indexed = indexed[qa_start:]
    if qa_limit:
        indexed = indexed[:qa_limit]
    return [(idx, qa) for idx, qa in indexed if isinstance(qa, dict)]


def normalize_locomo_qa_for_eval(qa: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(qa)
    if "answer" not in normalized and "adversarial_answer" in normalized:
        normalized["answer"] = normalized["adversarial_answer"]
    return normalized


def run_sample(
    *,
    args: argparse.Namespace,
    sample: Dict[str, Any],
    embedding_config: Dict[str, Any],
    memory_config: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    sample_id = str(sample.get("sample_id") or "unknown_sample")
    sample_state_dir = args.state_dir / sample_id
    sample_state_dir.mkdir(parents=True, exist_ok=True)
    db_path = sample_state_dir / "memory.db"

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

        sessions = sorted_locomo_sessions(sample, max_sessions=int(args.max_sessions))
        replay_stats, reflect_runs = replay_sample_into_memory(
            manager=manager,
            sample=sample,
            sessions=sessions,
            enable_reflect=args.enable_reflect,
            reflect_every_sessions=max(1, int(args.reflect_every_sessions)),
            reflect_limit=int(args.reflect_limit),
        )
        counts = db_counts(db)

        out_sample = {
            "sample_id": sample_id,
            "qa": [
                normalize_locomo_qa_for_eval(qa) if isinstance(qa, dict) else qa
                for qa in json.loads(json.dumps(sample.get("qa", [])))
            ],
        }
        detail_rows: List[Dict[str, Any]] = []
        for qa_index, qa in selected_qa_rows(
            sample,
            qa_start=int(args.qa_start),
            qa_limit=int(args.qa_limit),
        ):
            question = str(qa.get("question") or "").strip()
            category = qa.get("category")
            memory_context = manager.recall(
                question,
                top_k=int(args.recall_top_k),
                budget=str(args.recall_budget),
                tags=["locomo", f"sample_id:{sample_id}"],
            )
            out_sample["qa"][qa_index][args.context_key] = memory_context
            detail_rows.append(
                {
                    "sample_id": sample_id,
                    "qa_index": qa_index,
                    "question": question,
                    "answer": qa.get("answer"),
                    "category": category,
                    "evidence": qa.get("evidence", []),
                    "context_key": args.context_key,
                    "recall_context_chars": len(memory_context or ""),
                    "recall_context": memory_context,
                    "db_path": str(db_path),
                    "db_counts": counts,
                }
            )

        out_sample["_hermes_memory_eval"] = {
            "db_path": str(db_path),
            "history_session_count": len(sessions),
            "replayed_dialog_turns": replay_stats.dialog_turns_total,
            "replayed_turn_pairs": replay_stats.turn_pairs_total,
            "turn_pairs_with_stored_facts": replay_stats.stored_pairs,
            "orphan_dialog_turns": replay_stats.orphan_dialog_turns,
            "reflect_runs": reflect_runs,
            "db_counts": counts,
        }
        return out_sample, detail_rows
    finally:
        db.close()


def run_sample_worker(
    payload: Tuple[int, int, Dict[str, Any], argparse.Namespace, Dict[str, Any], Dict[str, Any]],
) -> Tuple[int, Dict[str, Any], List[Dict[str, Any]]]:
    index, total, sample, args, embedding_config, memory_config = payload
    sample_id = str(sample.get("sample_id") or f"sample_{index}")
    sample_state_dir = args.state_dir / sample_id
    sample_state_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(
        sample_state_dir / "run_locomo_memory_eval.log",
        args.log_level,
        args.manager_log_level,
        stream=False,
    )
    logging.info("[%s/%s] Running LoCoMo sample %s", index, total, sample_id)
    result, detail_rows = run_sample(
        args=args,
        sample=sample,
        embedding_config=embedding_config,
        memory_config=memory_config,
    )
    stats = result["_hermes_memory_eval"]
    logging.info(
        "[%s/%s] Finished %s: sessions=%s facts=%s observations=%s interpretations=%s detail_rows=%s",
        index,
        total,
        sample_id,
        stats["history_session_count"],
        stats["db_counts"]["facts"],
        stats["db_counts"]["observations"],
        stats["db_counts"]["interpretations"],
        len(detail_rows),
    )
    return index, result, detail_rows


def generate_reader_answers(
    *,
    args: argparse.Namespace,
    outputs: List[Dict[str, Any]],
    detail_rows: List[Dict[str, Any]],
) -> Tuple[int, int]:
    detail_by_sample_index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in detail_rows:
        if not isinstance(row, dict):
            continue
        if "qa_index" not in row:
            continue
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            continue
        try:
            qa_index = int(row["qa_index"])
        except (TypeError, ValueError):
            continue
        detail_by_sample_index[(sample_id, qa_index)] = row

    answer_success_count = 0
    answer_failure_count = 0
    for sample in outputs:
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("sample_id") or "").strip()
        if not sample_id:
            continue
        if isinstance(sample.get("_hermes_memory_eval"), dict) and sample["_hermes_memory_eval"].get("status") == "error":
            continue
        qas = sample.get("qa") if isinstance(sample.get("qa"), list) else []
        for qa_index, qa in enumerate(qas):
            if not isinstance(qa, dict):
                continue
            if "answer" not in qa and "adversarial_answer" in qa:
                qa["answer"] = qa["adversarial_answer"]
            detail_row = detail_by_sample_index.get((sample_id, qa_index))
            memory_context = str(
                qa.get(args.context_key)
                or (detail_row or {}).get("recall_context")
                or ""
            )
            question = str(qa.get("question") or "").strip()
            category = qa.get("category")
            try:
                prediction = answer_question_with_reader(
                    args=args,
                    question=question,
                    category=category,
                    memory_context=memory_context,
                )
                answer_success_count += 1
                logging.info(
                    "Answered LoCoMo QA: sample_id=%s qa_index=%s context_chars=%s prediction_chars=%s",
                    sample_id,
                    qa_index,
                    len(memory_context),
                    len(prediction or ""),
                )
            except Exception as exc:
                answer_failure_count += 1
                prediction = NO_INFORMATION_ANSWER
                logging.exception(
                    "Failed to answer LoCoMo QA sample_id=%s qa_index=%s: %s",
                    sample_id,
                    qa_index,
                    exc,
                )
                if detail_row is not None:
                    detail_row["reader_status"] = "error"
                    detail_row["reader_error"] = str(exc)

            qa[args.prediction_key] = prediction
            qa["hypothesis"] = prediction
            if memory_context:
                qa[f"{args.prediction_key}_context_text"] = memory_context
            if detail_row is not None:
                detail_row[args.prediction_key] = prediction
                detail_row["hypothesis"] = prediction
                detail_row["reader_model"] = args.reader_model
                detail_row["reader_base_url"] = args.reader_base_url
    return answer_success_count, answer_failure_count


def write_output(path: Path, samples: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(list(samples), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    loaded_env_paths = load_hermes_dotenv(
        hermes_home=get_hermes_home(),
        project_env=REPO_ROOT / ".env",
    )
    args = parse_args()
    resolve_llm_args(args)
    configure_logging(args.log_path, args.log_level, args.manager_log_level)
    if loaded_env_paths:
        logging.info(
            "Loaded environment variables from: %s",
            ", ".join(str(path) for path in loaded_env_paths),
        )
    else:
        logging.info("No Hermes .env file found; using system environment variables")

    detail_output = args.detail_output
    if detail_output is None:
        detail_output = args.output.with_suffix(args.output.suffix + ".details.json")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    remove_existing_outputs(
        output_path=args.output,
        detail_output_path=detail_output,
        state_dir=args.state_dir,
        overwrite=bool(args.overwrite),
    )
    args.state_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(args.input)
    selected = filter_samples(
        samples,
        sample_ids=args.sample_id,
        start=int(args.start),
        limit=int(args.limit),
    )
    if not selected:
        raise RuntimeError("No LoCoMo samples selected")

    embedding_config, memory_config = prepare_runtime_configs(args)
    success_count = 0
    failure_count = 0
    outputs_by_index: Dict[int, Dict[str, Any]] = {}
    detail_rows_by_index: Dict[int, List[Dict[str, Any]]] = {}
    errors_by_index: Dict[int, Dict[str, str]] = {}

    workers = max(1, int(args.workers or 1))
    if workers > 1:
        logging.info("Running LoCoMo memory evaluation with %s worker processes", workers)
        payloads = [
            (index, len(selected), sample, args, embedding_config, memory_config)
            for index, sample in enumerate(selected, 1)
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_sample_worker, payload): payload
                for payload in payloads
            }
            for future in as_completed(futures):
                index, _total, sample, _args, _embedding_config, _memory_config = futures[future]
                sample_id = str(sample.get("sample_id") or f"sample_{index}")
                try:
                    result_index, result, detail_rows = future.result()
                    outputs_by_index[result_index] = result
                    detail_rows_by_index[result_index] = detail_rows
                    logging.info(
                        "[%s/%s] LoCoMo sample ready %s: detail_rows=%s",
                        result_index,
                        len(selected),
                        sample_id,
                        len(detail_rows),
                    )
                except Exception as exc:
                    failure_count += 1
                    errors_by_index[index] = {
                        "sample_id": sample_id,
                        "status": "error",
                        "stage": "sample",
                        "error": str(exc),
                    }
                    logging.exception("Failed LoCoMo sample %s: %s", sample_id, exc)
    else:
        for index, sample in enumerate(selected, 1):
            sample_id = str(sample.get("sample_id") or f"sample_{index}")
            logging.info("[%s/%s] Running LoCoMo sample %s", index, len(selected), sample_id)
            try:
                result, detail_rows = run_sample(
                    args=args,
                    sample=sample,
                    embedding_config=embedding_config,
                    memory_config=memory_config,
                )
                outputs_by_index[index] = result
                detail_rows_by_index[index] = detail_rows
                stats = result["_hermes_memory_eval"]
                logging.info(
                    "[%s/%s] Finished %s: sessions=%s facts=%s observations=%s interpretations=%s detail_rows=%s",
                    index,
                    len(selected),
                    sample_id,
                    stats["history_session_count"],
                    stats["db_counts"]["facts"],
                    stats["db_counts"]["observations"],
                    stats["db_counts"]["interpretations"],
                    len(detail_rows),
                )
            except Exception as exc:
                failure_count += 1
                errors_by_index[index] = {
                    "sample_id": sample_id,
                    "status": "error",
                    "stage": "sample",
                    "error": str(exc),
                }
                logging.exception("Failed LoCoMo sample %s: %s", sample_id, exc)

    outputs: List[Dict[str, Any]] = []
    detail_rows_output: List[Dict[str, Any]] = []
    for index, sample in enumerate(selected, 1):
        sample_id = str(sample.get("sample_id") or f"sample_{index}")
        if index in errors_by_index:
            outputs.append(
                {
                    "sample_id": sample_id,
                    "qa": sample.get("qa", []),
                    "_hermes_memory_eval": errors_by_index[index],
                }
            )
            detail_rows_output.append(errors_by_index[index])
            continue
        result = outputs_by_index.get(index)
        if result is None:
            failure_count += 1
            error_row = {
                "sample_id": sample_id,
                "status": "error",
                "stage": "sample",
                "error": "Sample worker returned no result",
            }
            outputs.append(
                {
                    "sample_id": sample_id,
                    "qa": sample.get("qa", []),
                    "_hermes_memory_eval": error_row,
                }
            )
            detail_rows_output.append(error_row)
            continue
        outputs.append(result)
        detail_rows_output.extend(detail_rows_by_index.get(index, []))
        success_count += 1

    answer_success_count, answer_failure_count = generate_reader_answers(
        args=args,
        outputs=outputs,
        detail_rows=detail_rows_output,
    )

    write_output(args.output, outputs)
    write_output(detail_output, detail_rows_output)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "detail_output": str(detail_output),
        "state_dir": str(args.state_dir),
        "samples_requested": len(selected),
        "samples_succeeded": success_count,
        "samples_failed": failure_count,
        "context_key": args.context_key,
        "prediction_key": args.prediction_key,
        "llm_model": args.llm_model,
        "reader_model": args.reader_model,
        "llm_thinking": args.llm_thinking,
        "llm_json_mode": args.llm_json_mode,
        "workers": workers,
        "reader_answers_succeeded": answer_success_count,
        "reader_answers_failed": answer_failure_count,
        "recall_top_k": args.recall_top_k,
        "recall_budget": args.recall_budget,
        "feedback_analysis_enabled": args.enable_feedback_analysis,
        "reflect_every_sessions": args.reflect_every_sessions,
        "fact_extraction_interval": args.fact_extraction_interval,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failure_count == 0 and answer_failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
