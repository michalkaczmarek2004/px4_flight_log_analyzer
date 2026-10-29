"""Load a PX4 .ulg flight log and expose its contents as plain Python / pandas objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pyulog import ULog


@dataclass
class LogMessage:
    """A text message printed by the flight stack (e.g. 'Takeoff detected')."""

    timestamp_s: float  # seconds since log start
    level: str
    text: str


@dataclass
class FlightLog:
    """Parsed contents of a single .ulg file."""

    path: Path
    start_timestamp_us: int
    last_timestamp_us: int
    info: dict[str, object]
    parameters: dict[str, float]
    messages: list[LogMessage]
    topics: dict[str, pd.DataFrame] = field(repr=False)
    dropouts_s: float = 0.0  # total time of data lost due to logging dropouts

    @property
    def duration_s(self) -> float:
        return (self.last_timestamp_us - self.start_timestamp_us) / 1e6

    @property
    def utc_start(self) -> datetime | None:
        """Wall-clock start time, if the log contains a GPS time reference."""
        ref_us = self.info.get("time_ref_utc")
        if not ref_us:
            return None
        return datetime.fromtimestamp(int(ref_us), tz=timezone.utc)

    @property
    def firmware_version(self) -> str:
        """PX4 version, e.g. 'v1.15.0'. ver_sw_release packs it as 0xMMmmppTT."""
        release = self.info.get("ver_sw_release")
        if not release:
            return str(self.info.get("ver_sw", "unknown"))
        release = int(release)
        major, minor, patch, kind = (release >> 24) & 0xFF, (release >> 16) & 0xFF, (release >> 8) & 0xFF, release & 0xFF
        suffix = {0: "-dev", 64: "-alpha", 128: "-beta", 192: "-rc", 255: ""}.get(kind, f"-{kind}")
        return f"v{major}.{minor}.{patch}{suffix}"

    def topic(self, name: str, multi_id: int = 0) -> pd.DataFrame:
        """Return a topic's data. Instances other than 0 are stored as 'name_<id>'."""
        key = name if multi_id == 0 else f"{name}_{multi_id}"
        try:
            return self.topics[key]
        except KeyError:
            raise KeyError(f"Topic '{key}' not found in {self.path.name}") from None

    def has_topic(self, name: str, multi_id: int = 0) -> bool:
        return (name if multi_id == 0 else f"{name}_{multi_id}") in self.topics


def _topic_to_dataframe(data: dict, start_us: int) -> pd.DataFrame:
    df = pd.DataFrame(data)
    # Relative time in seconds is far easier to plot and reason about than raw µs.
    df.insert(0, "time_s", (df["timestamp"] - start_us) / 1e6)
    return df


def load_ulog(path: str | Path, topics: list[str] | None = None) -> FlightLog:
    """Parse a .ulg file.

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

    topic_frames: dict[str, pd.DataFrame] = {}
    for dataset in ulog.data_list:
        key = dataset.name if dataset.multi_id == 0 else f"{dataset.name}_{dataset.multi_id}"
        topic_frames[key] = _topic_to_dataframe(dataset.data, start_us)

    messages = [
        LogMessage(
            timestamp_s=(m.timestamp - start_us) / 1e6,
            level=m.log_level_str(),
            text=m.message.strip(),
        )
        for m in ulog.logged_messages
    ]

    return FlightLog(
        path=path,
        start_timestamp_us=start_us,
        last_timestamp_us=ulog.last_timestamp,
        info=dict(ulog.msg_info_dict),
        parameters=dict(ulog.initial_parameters),
        messages=messages,
        topics=topic_frames,
        dropouts_s=sum(d.duration for d in ulog.dropouts) / 1e3,  # duration is in ms
    )
