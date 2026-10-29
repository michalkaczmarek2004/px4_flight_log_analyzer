"""Prepare topic DataFrames for analysis: common time axis, derived signals,
requested/measured alignment and masking of invalid samples.

Every function returns new data and leaves its input unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation import FIELD_SPECS, wrong_value_mask

# PX4 validity flags and the fields they vouch for.
VALIDITY_FLAGS: dict[str, dict[str, list[str]]] = {
    "vehicle_local_position": {
        "xy_valid": ["x", "y"],
        "z_valid": ["z"],
        "v_xy_valid": ["vx", "vy"],
        "v_z_valid": ["vz"],
    },
}

# Limit fields where PX4 uses inf to mean "no limit"; inf there is valid data.
INF_MEANS_NO_LIMIT = {"vxy_max", "vz_max", "hagl_min", "hagl_max", "hagl_max_z", "hagl_max_xy"}

# A held setpoint older than this many median setpoint intervals is treated as missing.
MAX_AGE_FACTOR = 3


def calculate_relative_time(topics: dict[str, pd.DataFrame], t0_us: int | None = None) -> dict[str, pd.DataFrame]:
    """Put all topics on one time axis: time_s = (timestamp - t0) / 1e6.

    Args:
        t0_us: reference time in µs (e.g. the arming timestamp). Defaults to the
            earliest sample of all topics, so the common axis starts at 0 s.
    """
    if t0_us is None:
        firsts = [int(df["timestamp"][df["timestamp"] > 0].min()) for df in topics.values() if (df["timestamp"] > 0).any()]
        if not firsts:
            return {name: df.copy() for name, df in topics.items()}
        t0_us = min(firsts)

    result = {}
    for name, df in topics.items():
        df = df.copy()
        # timestamp is uint64; cast so samples before t0 become negative instead of wrapping.
        time_s = (df["timestamp"].astype("int64") - int(t0_us)) / 1e6
        if "time_s" in df.columns:
            df["time_s"] = time_s
        else:
            df.insert(0, "time_s", time_s)
        result[name] = df
    return result


def calculate_altitude(local_pos: pd.DataFrame) -> pd.Series:
    """Local height, positive up: H_local = -z (PX4 local frame is NED, z points down)."""
    return (-local_pos["z"]).rename("altitude_m")


def calculate_horizontal_speed(local_pos: pd.DataFrame) -> pd.Series:
    """Horizontal speed: V_xy = sqrt(vx^2 + vy^2)."""
    return np.sqrt(local_pos["vx"] ** 2 + local_pos["vy"] ** 2).rename("horizontal_speed_ms")


def _sorted_unique_time(df: pd.DataFrame) -> pd.DataFrame:
    # Repeated timestamps: the later sample is the newest value for that instant.
    return df.sort_values("time_s", kind="stable").drop_duplicates("time_s", keep="last")


def align_signals(
    requested: pd.DataFrame,
    measured: pd.DataFrame,
    signals: list[str] | dict[str, tuple[str, str]],
    method: str = "hold",
    max_age_s: float | None = None,
) -> pd.DataFrame:
    """Synchronize a requested (setpoint) signal with its measured counterpart.

    The measured time axis is kept and the requested signal is evaluated at each
    measured timestamp, inside the time range both topics cover.

    Args:
        requested: setpoint topic, e.g. vehicle_local_position_setpoint.
        measured: measured topic, e.g. vehicle_local_position.
        signals: column names present in both topics, or
            {output_name: (requested_column, measured_column)}.
        method: "hold" keeps the last setpoint (how PX4 controllers use them);
            "linear" interpolates between two consecutive setpoints.
        max_age_s: a setpoint older than this ("hold") or a setpoint gap longer
            than this ("linear") gives NaN instead of a stale value. Defaults to
            MAX_AGE_FACTOR x the median setpoint interval.

    Returns:
        DataFrame with time_s, requested_age_s and, for each signal,
        <name>_requested, <name>_measured and <name>_error (measured - requested).
        NaN setpoints (PX4: axis not controlled) stay NaN.
    """
    if method not in ("hold", "linear"):
        raise ValueError(f"method must be 'hold' or 'linear', got {method!r}")
    if isinstance(signals, dict):
        pairs = signals
    else:
        pairs = {name: (name, name) for name in signals}
    for name, (req_col, meas_col) in pairs.items():
        if req_col not in requested.columns:
            raise KeyError(f"requested topic has no column '{req_col}' (signal '{name}')")
        if meas_col not in measured.columns:
            raise KeyError(f"measured topic has no column '{meas_col}' (signal '{name}')")

    req = _sorted_unique_time(requested)
    meas = measured.sort_values("time_s", kind="stable")
    t_req = req["time_s"].to_numpy()
    if len(t_req) < 2:
        raise ValueError("requested topic needs at least 2 samples to align")

    if max_age_s is None:
        max_age_s = MAX_AGE_FACTOR * float(np.median(np.diff(t_req)))

    overlap = (meas["time_s"] >= t_req[0]) & (meas["time_s"] <= t_req[-1])
    meas = meas[overlap]
    t = meas["time_s"].to_numpy()

    # i = last setpoint at or before each measurement, i + 1 = the next one
    i = np.searchsorted(t_req, t, side="right") - 1
    nxt = np.minimum(i + 1, len(t_req) - 1)
    age = t - t_req[i]

    if method == "hold":
        usable = age <= max_age_s
    else:
        gap = t_req[nxt] - t_req[i]
        usable = (gap <= max_age_s) | (age == 0)
        frac = np.divide(age, gap, out=np.zeros_like(age), where=gap > 0)

    out = pd.DataFrame({"time_s": t, "requested_age_s": age})
    for name, (req_col, meas_col) in pairs.items():
        v = req[req_col].to_numpy(dtype=float)
        if method == "hold":
            value = v[i]
        else:
            value = v[i] + (v[nxt] - v[i]) * frac
        value = np.where(usable, value, np.nan)
        measured_values = meas[meas_col].to_numpy(dtype=float)
        out[f"{name}_requested"] = value
        out[f"{name}_measured"] = measured_values
        out[f"{name}_error"] = measured_values - value
    return out


def filter_invalid_samples(df: pd.DataFrame, topic: str | None = None) -> pd.DataFrame:
    """Mask invalid values as NaN without dropping or hiding anything.

    Masks inf, PX4 'unknown' sentinels, values outside the field's allowed
    domain (validation.FIELD_SPECS) and fields PX4 flagged as invalid
    (e.g. x/y when xy_valid is false). Rows are never removed, so time gaps
    stay visible.

    The returned copy has:
        - a `masked` column, True for rows where any value was masked
        - df.attrs["masked"]: list of (field, reason, count) describing what was masked
    """
    topic = topic or df.attrs.get("topic")
    spec = FIELD_SPECS.get(topic, {})
    flags = VALIDITY_FLAGS.get(topic, {})
    out = df.copy()
    row_masked = pd.Series(False, index=df.index)
    report: list[tuple[str, str, int]] = []

    def mask(column: str, where: pd.Series, reason: str) -> None:
        nonlocal row_masked
        # Skip values that are already NaN so each problem is counted once.
        where = where & out[column].notna()
        count = int(where.sum())
        if not count:
            return
        if not pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].astype(float)
        out.loc[where, column] = np.nan
        row_masked |= where
        report.append((column, reason, count))

    skip = {"time_s", "timestamp"}
    for column in out.select_dtypes(include="number").columns:
        if column in skip:
            continue
        values = out[column].astype(float)
        if column not in INF_MEANS_NO_LIMIT:
            mask(column, pd.Series(np.isinf(values), index=out.index), "inf")

        field = spec.get(column)
        if field is None:
            continue
        if field.unknown is not None:
            mask(column, values == field.unknown, f"PX4 unknown ({field.unknown:g})")
        wrong = wrong_value_mask(df[column], field).reindex(out.index, fill_value=False)
        mask(column, wrong, "out of range" if field.kind != "enum" else "invalid enum")

    for flag, columns in flags.items():
        if flag not in df.columns:
            continue
        invalid = df[flag] == 0
        for column in columns:
            if column in out.columns:
                mask(column, invalid, f"{flag}=0")

    out["masked"] = row_masked
    out.attrs["masked"] = report
    return out


def preprocess(topics: dict[str, pd.DataFrame], t0_us: int | None = None) -> dict[str, pd.DataFrame]:
    """Standard pipeline: common time axis, invalid samples masked, derived signals added."""
    topics = calculate_relative_time(topics, t0_us)
    topics = {name: filter_invalid_samples(df) for name, df in topics.items()}
    if "vehicle_local_position" in topics:
        local_pos = topics["vehicle_local_position"]
        if "z" in local_pos.columns:
            local_pos["altitude_m"] = calculate_altitude(local_pos)
        if {"vx", "vy"} <= set(local_pos.columns):
            local_pos["horizontal_speed_ms"] = calculate_horizontal_speed(local_pos)
    return topics
