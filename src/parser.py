"""Find topics in a PX4 .ulg flight log and convert them to pandas DataFrames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyulog import ULog

from src.px4_topics import category


def topic_key(name: str, instance: int = 0) -> str:
    """Dict key of a topic instance. Extra instances (e.g. 3 estimators) are 'name_<id>'."""
    return name if instance == 0 else f"{name}_{instance}"


def load_topics(path: str | Path, topics: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Parse a .ulg file and return {topic_key: DataFrame} for every topic instance.

    Each DataFrame starts with `time_s` (seconds since log start), followed by the
    raw `timestamp` (µs) and the topic's fields. The topic name and instance are
    kept in `df.attrs["topic"]` and `df.attrs["instance"]`.

    Args:
        path: path to the .ulg file.
        topics: optional list of topic names to load; loading only what you
            need is much faster for large logs. None loads every topic.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Log file not found: {path}")
    if path.suffix.lower() != ".ulg":
        raise ValueError(f"Expected a .ulg file, got: {path.name}")

    ulog = ULog(str(path), message_name_filter_list=topics)
    start_us = ulog.start_timestamp

    frames: dict[str, pd.DataFrame] = {}
    for dataset in ulog.data_list:
        df = pd.DataFrame(dataset.data)
        # timestamp is uint64; cast first so samples logged just before start_us don't wrap around.
        df.insert(0, "time_s", (df["timestamp"].astype("int64") - start_us) / 1e6)
        df.attrs["topic"] = dataset.name
        df.attrs["instance"] = dataset.multi_id
        frames[topic_key(dataset.name, dataset.multi_id)] = df
    return frames


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
