"""Compute flight metrics from topic DataFrames produced by src.parser."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Topics the metrics below need; pass to parser.load_topics to skip the rest.
REQUIRED_TOPICS = ["vehicle_local_position", "battery_status", "vehicle_status"]

ARMING_STATE_ARMED = 2  # vehicle_status.arming_state


def log_duration_s(topics: dict[str, pd.DataFrame]) -> float:
    return max((df["time_s"].iloc[-1] for df in topics.values() if not df.empty), default=0.0)


def armed_time_s(vehicle_status: pd.DataFrame) -> float:
    """Total time the vehicle was armed."""
    t = vehicle_status["time_s"].to_numpy()
    armed = vehicle_status["arming_state"].to_numpy() == ARMING_STATE_ARMED
    # Each sample's state holds until the next sample.
    return float(np.sum(np.diff(t)[armed[:-1]]))


def altitude_metrics(local_pos: pd.DataFrame) -> dict[str, float]:
    # PX4 uses NED, so up is -z. The EKF origin isn't necessarily the takeoff
    # point, so report height relative to the first sample instead.
    alt = -local_pos["z"]
    alt = alt - alt.iloc[0]
    return {"max_altitude_m": float(alt.max()), "min_altitude_m": float(alt.min())}


def speed_metrics(local_pos: pd.DataFrame) -> dict[str, float]:
    h_speed = np.hypot(local_pos["vx"], local_pos["vy"])
    return {
        "max_horizontal_speed_ms": float(h_speed.max()),
        "max_climb_rate_ms": float((-local_pos["vz"]).max()),
        "max_descent_rate_ms": float(local_pos["vz"].max()),
    }


def battery_metrics(battery: pd.DataFrame) -> dict[str, float]:
    # Drop samples before the battery driver reports a valid voltage.
    battery = battery[battery["voltage_v"] > 0]
    if battery.empty:
        return {}
    result = {
        "voltage_start_v": float(battery["voltage_v"].iloc[0]),
        "voltage_end_v": float(battery["voltage_v"].iloc[-1]),
        "voltage_min_v": float(battery["voltage_v"].min()),
        "remaining_start_pct": float(battery["remaining"].iloc[0] * 100),
        "remaining_end_pct": float(battery["remaining"].iloc[-1] * 100),
    }
    discharged = float(battery["discharged_mah"].iloc[-1])
    if discharged >= 0:  # -1 means no current sensor
        result["discharged_mah"] = discharged
    return result


def compute_metrics(topics: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Compute every metric whose source topic is present in the log."""
    metrics: dict[str, float] = {"log_duration_s": log_duration_s(topics)}
    if "vehicle_status" in topics:
        metrics["armed_time_s"] = armed_time_s(topics["vehicle_status"])
    if "vehicle_local_position" in topics:
        metrics |= altitude_metrics(topics["vehicle_local_position"])
        metrics |= speed_metrics(topics["vehicle_local_position"])
    if "battery_status" in topics:
        metrics |= battery_metrics(topics["battery_status"])
    return metrics
