import re
import json
from datetime import datetime, timezone, timedelta

DATABASE_TIMEZONE = timezone(timedelta(hours=8))

def normalize_signature_text(value):
    value = (value or "").lower()
    value = re.sub(r"https?://[^\s]+", " URL ", value)
    value = re.sub(r"[\s\-_–—|/\\:：]+", " ", value)
    value = re.sub(r"[^\w\u4e00-\u9fff.]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()

def tokenize_signature_text(value):
    normalized = normalize_signature_text(value)
    tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9][a-z0-9_.-]{1,}", normalized))
    return {token for token in tokens if len(token) >= 2}

def parse_json_list(value):
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []

def dump_json_list(values):
    return json.dumps(values or [], ensure_ascii=False)

def parse_iso_datetime(value):
    if not value:
        return None
    try:
        return parse_user_time_to_utc(str(value))
    except ValueError:
        return None

def datetime_to_epoch_second(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def parse_timestamp_to_utc(ts_str):
    """Parse a source timestamp; naive source values are treated as UTC."""
    dt = datetime.fromisoformat((ts_str or "").replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_user_time_to_utc(ts_str):
    """Parse a user-entered time; naive values are interpreted as UTC+8."""
    dt = datetime.fromisoformat((ts_str or "").replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=DATABASE_TIMEZONE)
    return dt.astimezone(timezone.utc)


def to_db_timezone(dt):
    if isinstance(dt, str):
        dt = parse_user_time_to_utc(dt)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(DATABASE_TIMEZONE)


def format_db_timestamp(dt):
    """Format timestamps for the cleaned database in UTC+8."""
    return to_db_timezone(dt).isoformat()


def now_db_timestamp():
    return datetime.now(DATABASE_TIMEZONE).isoformat()


def format_utc_timestamp(dt):
    """Format a datetime as an explicit UTC ISO-8601 timestamp."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
