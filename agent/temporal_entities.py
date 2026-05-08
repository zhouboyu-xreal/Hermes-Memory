"""Helpers for keeping plain time expressions out of the entity graph."""

from __future__ import annotations

import re

_TEMPORAL_TYPE_NAMES = {
    "TIME",
    "DATE",
    "DATETIME",
    "DURATION",
    "PERIOD",
    "TEMPORAL",
}

_TEMPORAL_WORD_RE = re.compile(
    r"(今天|今日|昨天|昨日|前天|明天|后天|本周|这周|上周|下周|本月|这个月|上个月|下个月|"
    r"今年|去年|明年|最近|过去|刚才|稍后|早上|上午|中午|下午|晚上|凌晨|周末|工作日|"
    r"星期[一二三四五六日天]|周[一二三四五六日天])"
)
_DATE_SHAPE_RE = re.compile(
    r"^\s*(?:\d{4}[-/.年]\d{1,2}(?:[-/.月]\d{1,2})?(?:日|号)?|"
    r"\d{1,2}[-/.月]\d{1,2}(?:日|号)?|"
    r"\d{1,2}:\d{2}(?::\d{2})?|"
    r"\d+\s*(?:秒|分钟|小时|天|周|星期|个月|月|年))\s*$"
)


def is_temporal_entity(name: str, entity_type: str = "") -> bool:
    """Return True for ordinary time expressions that should stay metadata.

    Named calendar concepts with domain meaning, such as "春节", "Q3 财报季",
    or "Sprint 42", are intentionally not matched here; they may still be
    useful graph entities. This filter targets plain dates, durations, and
    relative time words that otherwise create noisy high-degree graph nodes.
    """
    clean_name = str(name or "").strip()
    clean_type = str(entity_type or "").strip().upper()
    if not clean_name:
        return False
    if clean_type in _TEMPORAL_TYPE_NAMES:
        return True
    if _DATE_SHAPE_RE.match(clean_name):
        return True
    return bool(_TEMPORAL_WORD_RE.search(clean_name))
