#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.screen_memory.service import run_screen_memory_due_work


def main():
    parser = argparse.ArgumentParser(description="Run Hermes screen-memory pipeline phases.")
    parser.add_argument(
        "--phase",
        choices=("ingest", "cluster", "observation"),
        help="Force one phase. Omit to run only phases currently due.",
    )
    args = parser.parse_args()
    result = run_screen_memory_due_work(force_phase=args.phase)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
