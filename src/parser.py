"""Find topics in a PX4 .ulg flight log and convert them to pandas DataFrames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyulog import ULog

from src.px4_topics import category


def topic_key(name: str, instance: int = 0) -> str:
    """Dict key of a topic instance. Extra instances (e.g. 3 estimators) are 'name_<id>'."""
    return name if instance == 0 else f"{name}_{instance}"


def _open(path: str | Path, topics: list[str] | None) -> ULog:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Log file not found: {path}")
    if path.suffix.lower() != ".ulg":
        raise ValueError(f"Expected a .ulg file, got: {path.name}")
    return ULog(str(path), message_name_filter_list=topics)


def _frames(ulog: ULog) -> dict[str, pd.DataFrame]:
    start_us = ulog.start_timestamp
    frames: dict[str, pd.DataFrame] = {}
    for dataset in ulog.data_list:
        df = pd.DataFrame(dataset.data)
        # timestamp is uint64; cast first so samples logged just before start_us don't wrap around.
        df.insert(0, "time_s", (df["timestamp"].astype("int64") - start_us) / 1e6)
        df.attrs["topic"] = dataset.name
        df.attrs["instance"] = dataset.multi_id
        df.attrs["log_start_us"] = start_us
        frames[topic_key(dataset.name, dataset.multi_id)] = df
    return frames


def load_topics(path: str | Path, topics: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Parse a .ulg file and return {topic_key: DataFrame} for every topic instance.

    Each DataFrame starts with `time_s` (seconds since log start), followed by the
    raw `timestamp` (µs) and the topic's fields. The topic name, instance and log
    start (µs) are kept in `df.attrs["topic"]`, `["instance"]` and `["log_start_us"]`.

    Args:
        path: path to the .ulg file.
        topics: optional list of topic names to load; loading only what you
            need is much faster for large logs. None loads every topic.
    """
    return _frames(_open(path, topics))


def _firmware_version(info: dict) -> str:
    # ver_sw_release packs the version as 0xMMmmppTT (TT = release type).
    release = info.get("ver_sw_release")
    if not release:
        return str(info.get("ver_sw", "unknown"))
    release = int(release)
    major, minor, patch, kind = (release >> 24) & 0xFF, (release >> 16) & 0xFF, (release >> 8) & 0xFF, release & 0xFF
    suffix = {0: "-dev", 64: "-alpha", 128: "-beta", 192: "-rc", 255: ""}.get(kind, f"-{kind}")
    return f"v{major}.{minor}.{patch}{suffix}"


def _dropouts(ulog: ULog) -> list[dict[str, float]]:
    # pyulog can report several dropouts at the same timestamp; they are one interruption.
    merged: dict[int, float] = {}
    for dropout in ulog.dropouts:
        merged[dropout.timestamp] = merged.get(dropout.timestamp, 0.0) + dropout.duration / 1e3
    return [
        {"time_s": (ts - ulog.start_timestamp) / 1e6, "duration_s": duration}
        for ts, duration in sorted(merged.items())
    ]


def load_log(path: str | Path, topics: list[str] | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    """Like load_topics, plus log metadata, all on the same time axis (seconds since log start).

    Metadata: file, start_us, duration_s, hardware, firmware, system, utc_start,
    dropouts [{time_s, duration_s}] and messages [{time_s, level, text}] printed by PX4.
    """
    ulog = _open(path, topics)
    info = ulog.msg_info_dict
    utc_us = info.get("time_ref_utc")
    metadata = {
        "file": Path(path).name,
        "start_us": int(ulog.start_timestamp),
        "duration_s": (ulog.last_timestamp - ulog.start_timestamp) / 1e6,
        "hardware": str(info.get("ver_hw", "unknown")),
        "firmware": _firmware_version(info),
        "system": str(info.get("sys_name", "unknown")),
        "utc_start": (
            pd.Timestamp(int(utc_us), unit="us", tz="UTC").isoformat() if utc_us else None
        ),
        "dropouts": _dropouts(ulog),
        "messages": [
            {"time_s": (m.timestamp - ulog.start_timestamp) / 1e6, "level": m.log_level_str(), "text": m.message.strip()}
            for m in ulog.logged_messages
        ],
    }
    return _frames(ulog), metadata


def get_topic(topics: dict[str, pd.DataFrame], name: str, instance: int = 0) -> pd.DataFrame:
    """Return one topic instance as a DataFrame."""
    key = topic_key(name, instance)
    if key in topics:
        return topics[key]
    available = sorted(df.attrs["instance"] for df in topics.values() if df.attrs.get("topic") == name)
    if available:
        raise KeyError(f"Topic '{name}' has no instance {instance}; available instances: {available}")
    raise KeyError(f"Topic '{name}' not found in log")


def list_topics(topics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per topic instance: category (UNKNOWN if not a standard PX4 topic),
    sample count and when the topic was recorded, in seconds since log start."""
    rows = []
    for df in topics.values():
        n = len(df)
        start = float(df["time_s"].iloc[0]) if n else float("nan")
        end = float(df["time_s"].iloc[-1]) if n else float("nan")
        duration = end - start if n else 0.0
        rows.append({
            "topic": df.attrs["topic"],
            "instance": df.attrs["instance"],
            "category": category(df.attrs["topic"]),
            "samples": n,
            "fields": len(df.columns) - 2,  # without time_s and timestamp
            "start_s": start,
            "end_s": end,
            "duration_s": duration,
            "rate_hz": (n - 1) / duration if duration > 0 else float("nan"),
        })
    columns = ["topic", "instance", "category", "samples", "fields", "start_s", "end_s", "duration_s", "rate_hz"]
    return pd.DataFrame(rows, columns=columns).sort_values(["topic", "instance"], ignore_index=True)
