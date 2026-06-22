#!/usr/bin/env python3
"""Inspect Feishu messenger chat extraction from an OpenChronicle capture."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.screen_memory.manager import extract_feishu_messenger_chat_context


DEFAULT_OPENCHRONICLE_DB = Path(
    "/Users/zhouboyu/Documents/xreal/pme/openchronicle.db"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tmp" / "screen_memory_feishu_chat_text"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract speaker-attributed Feishu chat_text from one "
            "OpenChronicle captures row."
        )
    )
    parser.add_argument(
        "--openchronicle-db",
        type=Path,
        default=DEFAULT_OPENCHRONICLE_DB,
        help="Path to OpenChronicle SQLite database.",
    )
    parser.add_argument(
        "--rowid",
        type=int,
        default=7812,
        help="SQLite rowid in captures table.",
    )
    parser.add_argument(
        "--capture-id",
        default=None,
        help="Optional captures.id value. When set, it takes precedence over --rowid.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Extract every Feishu captures row whose AXTree contains "
            "messenger-chat."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum rows to process in --all mode; 0 means no limit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full extracted context as JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Write capture metadata and extracted chat context to this JSON file. "
            "Defaults to tmp/screen_memory_feishu_chat_text/feishu_chat_text_<target>.json."
        ),
    )
    return parser.parse_args()


def load_capture(database_path: Path, rowid: int, capture_id: Optional[str]) -> Dict[str, Any]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        if capture_id:
            row = connection.execute(
                """
                SELECT rowid, id, timestamp, app_name, window_title, visible_text
                FROM captures
                WHERE id = ?
                LIMIT 1
                """,
                (capture_id,),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT rowid, id, timestamp, app_name, window_title, visible_text
                FROM captures
                WHERE rowid = ?
                LIMIT 1
                """,
                (rowid,),
            ).fetchone()
    finally:
        connection.close()
    if not row:
        target = f"id={capture_id}" if capture_id else f"rowid={rowid}"
        raise SystemExit(f"No captures row found for {target}")
    return dict(row)


def load_feishu_chat_captures(database_path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        sql = """
            SELECT rowid, id, timestamp, app_name, window_title, visible_text
            FROM captures
            WHERE visible_text LIKE '%messenger-chat%'
              AND (
                  app_name LIKE '%Feishu%'
                  OR app_name LIKE '%飞书%'
                  OR visible_text LIKE '%## 飞书%'
                  OR visible_text LIKE '%_com.electron.lark_%'
              )
            ORDER BY timestamp ASC, rowid ASC
        """
        params: tuple[Any, ...] = ()
        if limit and limit > 0:
            sql += " LIMIT ?"
            params = (int(limit),)
        rows = connection.execute(sql, params).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


def default_output_json_path(args: argparse.Namespace, capture: Dict[str, Any]) -> Path:
    if args.output_json:
        return args.output_json
    if args.all:
        return DEFAULT_OUTPUT_DIR / "feishu_chat_text_all.json"
    if args.capture_id:
        target = str(args.capture_id).replace("/", "_").replace(":", "-")
    else:
        target = f"rowid_{capture.get('rowid') or args.rowid}"
    return DEFAULT_OUTPUT_DIR / f"feishu_chat_text_{target}.json"


def write_result_json(path: Path, capture: Dict[str, Any], context: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "capture": {
            "rowid": capture.get("rowid"),
            "id": capture.get("id"),
            "timestamp": capture.get("timestamp"),
            "app_name": capture.get("app_name"),
            "window_title": capture.get("window_title"),
        },
        "context": context,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_all_results_json(path: Path, results: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extracted_results = [
        result for result in results if result.get("context")
    ]
    payload = {
        "summary": {
            "captures_scanned": len(results),
            "captures_extracted": len(extracted_results),
            "message_count": sum(
                len((result.get("context") or {}).get("messages") or [])
                for result in extracted_results
            ),
        },
        "results": results,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_capture_payload(capture: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rowid": capture.get("rowid"),
        "id": capture.get("id"),
        "timestamp": capture.get("timestamp"),
        "app_name": capture.get("app_name"),
        "window_title": capture.get("window_title"),
    }


def run_all(args: argparse.Namespace) -> int:
    captures = load_feishu_chat_captures(args.openchronicle_db, args.limit)
    results = []
    for capture in captures:
        context = extract_feishu_messenger_chat_context(capture.get("visible_text") or "")
        results.append({
            "capture": build_capture_payload(capture),
            "context": context,
            "message_count": len((context or {}).get("messages") or []),
            "chat_text": (context or {}).get("chat_text") or "",
        })

    output_json_path = default_output_json_path(args, {})
    write_all_results_json(output_json_path, results)
    extracted_count = sum(1 for result in results if result.get("context"))
    message_count = sum(int(result.get("message_count") or 0) for result in results)
    print(
        "Processed Feishu messenger-chat captures:",
        f"scanned={len(results)}",
        f"extracted={extracted_count}",
        f"messages={message_count}",
    )
    print(f"Wrote JSON: {output_json_path}")
    if args.json:
        print(json.dumps({
            "summary": {
                "captures_scanned": len(results),
                "captures_extracted": extracted_count,
                "message_count": message_count,
            },
            "results": results,
        }, ensure_ascii=False, indent=2))
    return 0 if extracted_count else 1


def main() -> int:
    args = parse_args()
    if args.all:
        return run_all(args)

    capture = load_capture(args.openchronicle_db, args.rowid, args.capture_id)
    context = extract_feishu_messenger_chat_context(capture.get("visible_text") or "")
    if not context:
        print("No Feishu messenger-chat context extracted.")
        return 1

    print(
        "Capture:",
        f"rowid={capture.get('rowid')}",
        f"id={capture.get('id')}",
        f"timestamp={capture.get('timestamp')}",
        f"app={capture.get('app_name')}",
        f"window={capture.get('window_title')}",
    )
    print()
    print("Conversation:", context.get("conversation_title") or "")
    print()
    print("chat_text:")
    print(context.get("chat_text") or "")
    print()
    print("messages:")
    for index, message in enumerate(context.get("messages") or [], start=1):
        speaker = message.get("speaker") or "未知"
        text = message.get("text") or ""
        reply_to = message.get("reply_to")
        suffix = f" (回复 {reply_to})" if reply_to else ""
        print(f"{index:02d}. {speaker}{suffix}: {text}")
    print()
    print("structure_summary:")
    print(context.get("structure_summary") or "")

    output_json_path = default_output_json_path(args, capture)
    write_result_json(output_json_path, capture, context)
    print()
    print(f"Wrote JSON: {output_json_path}")

    if args.json:
        print()
        print("context_json:")
        print(json.dumps(context, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
