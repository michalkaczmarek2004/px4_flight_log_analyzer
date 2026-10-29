"""Compute flight metrics from topic DataFrames (ideally after src.preprocessing.preprocess)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import align_signals, calculate_altitude, calculate_horizontal_speed

# Topics the metrics below need; pass to parser.load_topics to skip the rest.
REQUIRED_TOPICS = ["vehicle_local_position", "vehicle_local_position_setpoint", "battery_status", "vehicle_status"]

ARMING_STATE_ARMED = 2  # vehicle_status.arming_state

# An interval counts as regular when within this fraction of the median interval.
REGULAR_INTERVAL_TOLERANCE = 0.5


def _first(series: pd.Series) -> float:
    # Masked (NaN) samples are skipped rather than reported as the start value.
    valid = series.dropna()
    return float(valid.iloc[0]) if len(valid) else float("nan")


def _last(series: pd.Series) -> float:
    valid = series.dropna()
    return float(valid.iloc[-1]) if len(valid) else float("nan")


def _rmse(error: pd.Series | np.ndarray) -> float:
    error = np.asarray(error, dtype=float)
    error = error[~np.isnan(error)]
    return float(np.sqrt(np.mean(error ** 2))) if len(error) else float("nan")


def calculate_duration(topics: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Recording time: first to last sample over all topics, and time spent armed."""
    starts = [df["time_s"].min() for df in topics.values() if not df.empty]
    ends = [df["time_s"].max() for df in topics.values() if not df.empty]
    result = {"log_duration_s": float(max(ends) - min(starts)) if starts else 0.0}

    status = topics.get("vehicle_status")
    if status is not None and len(status) >= 2 and "arming_state" in status:
        t = status["time_s"].to_numpy()
        armed = status["arming_state"].to_numpy() == ARMING_STATE_ARMED
        # Each sample's state holds until the next sample.
        result["armed_time_s"] = float(np.sum(np.diff(t)[armed[:-1]]))
    return result


def calculate_altitude_stats(local_pos: pd.DataFrame) -> dict[str, float]:
    """Min/max local height H_local = -z, and the range between them.

    H_local is relative to the EKF origin, which isn't always the takeoff point,
    so the range (max - min) is the most robust number.
    """
    alt = calculate_altitude(local_pos)
    return {
        "max_altitude_m": float(alt.max()),
        "min_altitude_m": float(alt.min()),
        "altitude_range_m": float(alt.max() - alt.min()),
    }


def calculate_speed_stats(local_pos: pd.DataFrame) -> dict[str, float]:
    """Maximum and average horizontal speed V_xy = sqrt(vx^2 + vy^2)."""
    speed = calculate_horizontal_speed(local_pos)
    return {
        "max_horizontal_speed_ms": float(speed.max()),
        "mean_horizontal_speed_ms": float(speed.mean()),
    }


def calculate_vertical_speed(local_pos: pd.DataFrame) -> dict[str, float]:
    """Vertical speed, positive up (V_up = -vz): max climb, max descent and mean absolute."""
    v_up = -local_pos["vz"]
    return {
        "max_climb_rate_ms": float(v_up.max()),
        "max_descent_rate_ms": float(-v_up.min()),
        "mean_vertical_speed_ms": float(v_up.abs().mean()),
    }


def calculate_battery_stats(battery: pd.DataFrame) -> dict[str, float]:
    """Minimum voltage and how much battery was used."""
    # Drop samples before the battery driver reports a valid voltage.
    battery = battery[battery["voltage_v"] > 0]
    if battery.empty:
        return {}
    voltage = battery["voltage_v"]
    result = {
        "voltage_start_v": _first(voltage),
        "voltage_end_v": _last(voltage),
        "voltage_min_v": float(voltage.min()),
        "voltage_drop_v": _first(voltage) - float(voltage.min()),
    }
    # remaining / discharged_mah are -1 (or NaN after preprocessing) when unknown.
    remaining = battery["remaining"][battery["remaining"] >= 0]
    if len(remaining):
        result["remaining_start_pct"] = _first(remaining) * 100
        result["remaining_end_pct"] = _last(remaining) * 100
        result["battery_used_pct"] = (_first(remaining) - _last(remaining)) * 100
    discharged = battery["discharged_mah"][battery["discharged_mah"] >= 0]
    if len(discharged):
        result["discharged_mah"] = _last(discharged) - _first(discharged)
    return result


def calculate_position_error(setpoint: pd.DataFrame, local_pos: pd.DataFrame) -> dict[str, float]:
    """RMSE and max error between position setpoint and measured position.

    The setpoint is held at each measured sample (preprocessing.align_signals).
    Samples where PX4 doesn't control an axis (NaN setpoint) or the position is
    invalid (masked) are left out, separately for horizontal and vertical.
    """
    aligned = align_signals(setpoint, local_pos, ["x", "y", "z"])
    horizontal = np.hypot(aligned["x_error"], aligned["y_error"])
    vertical = aligned["z_error"].abs()
    total = np.sqrt(aligned["x_error"] ** 2 + aligned["y_error"] ** 2 + aligned["z_error"] ** 2)
    return {
        "position_rmse_m": _rmse(total),
        "position_max_error_m": float(total.max()),
        "horizontal_rmse_m": _rmse(horizontal),
        "horizontal_max_error_m": float(horizontal.max()),
        "vertical_rmse_m": _rmse(vertical),
        "vertical_max_error_m": float(vertical.max()),
        "position_error_samples": float(total.notna().sum()),
        "position_error_coverage_pct": 100 * float(total.notna().sum()) / len(local_pos) if len(local_pos) else 0.0,
    }


def calculate_sample_rate(df: pd.DataFrame) -> dict[str, float]:
    """Logging frequency and how regular the sample intervals are.

    regularity_pct: share of intervals within ±50% of the median interval.
    jitter_s: standard deviation of the intervals.
    """
    t = df["time_s"].dropna().to_numpy()
    if len(t) < 2:
        return {"samples": float(len(t))}
    dt = np.diff(t)
    duration = t[-1] - t[0]
    median_dt = float(np.median(dt))
    regular = np.abs(dt - median_dt) <= REGULAR_INTERVAL_TOLERANCE * median_dt
    return {
        "samples": float(len(t)),
        "mean_rate_hz": (len(t) - 1) / duration if duration > 0 else float("nan"),
        "nominal_rate_hz": 1 / median_dt if median_dt > 0 else float("nan"),
        "median_interval_s": median_dt,
        "max_interval_s": float(dt.max()),
        "jitter_s": float(dt.std()),
        "regularity_pct": 100 * float(regular.mean()),
    }


def compute_sample_rates(topics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """calculate_sample_rate for every topic, one row per topic."""
    rows = {name: calculate_sample_rate(df) for name, df in sorted(topics.items())}
    return pd.DataFrame.from_dict(rows, orient="index")


def compute_metrics(topics: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Compute every metric whose source topic is present in the log."""
    metrics = calculate_duration(topics)
    local_pos = topics.get("vehicle_local_position")
    if local_pos is not None and not local_pos.empty:
        metrics |= calculate_altitude_stats(local_pos)
        metrics |= calculate_speed_stats(local_pos)
        metrics |= calculate_vertical_speed(local_pos)
        setpoint = topics.get("vehicle_local_position_setpoint")
        if setpoint is not None and len(setpoint) >= 2:
            metrics |= calculate_position_error(setpoint, local_pos)
    if "battery_status" in topics:
        metrics |= calculate_battery_stats(topics["battery_status"])
    return metrics
