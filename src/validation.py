"""Validate topic DataFrames produced by src.parser.

validate_timestamps   - missing data (gaps/coverage), repeated and wrong timestamps
validate_fields       - topics and fields exist and have the expected data type
check_missing_values  - NaN, inf and values outside a field's allowed domain
validate_signal       - signal behaviour against PX4 rules (validity flags, resets, limits)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ERROR = "ERROR"      # data is unusable for the metrics that depend on it
WARNING = "WARNING"  # data is usable but results may be inaccurate
INFO = "INFO"        # worth knowing, not a problem by itself


@dataclass
class Issue:
    severity: str
    topic: str
    check: str
    message: str


@dataclass(frozen=True)
class Field:
    """Expected type and allowed values of one topic field."""

    kind: str  # "float", "uint", "bool" or "enum"
    min: float = -np.inf
    max: float = np.inf
    values: frozenset[int] = frozenset()  # allowed values for "enum"
    unknown: float | None = None  # PX4 sentinel meaning "not available"
    required: bool = False  # needed by src.metrics


def _float(min=-np.inf, max=np.inf, unknown=None, required=False) -> Field:
    return Field("float", min, max, unknown=unknown, required=required)


def _bool(required=False) -> Field:
    return Field("bool", 0, 1, required=required)


def _uint(max=np.inf) -> Field:
    return Field("uint", 0, max)


def _enum(*values: int, required=False) -> Field:
    return Field("enum", values=frozenset(values), required=required)


# Field definitions follow the PX4 uORB messages (msg/*.msg).
FIELD_SPECS: dict[str, dict[str, Field]] = {
    "vehicle_local_position": {
        "x": _float(-1e6, 1e6),
        "y": _float(-1e6, 1e6),
        "z": _float(-1e5, 1e5, required=True),
        "vx": _float(-1000, 1000, required=True),
        "vy": _float(-1000, 1000, required=True),
        "vz": _float(-1000, 1000, required=True),
        "heading": _float(-np.pi - 1e-3, np.pi + 1e-3),
        "eph": _float(0),
        "epv": _float(0),
        "xy_valid": _bool(required=True),
        "z_valid": _bool(required=True),
        "v_xy_valid": _bool(required=True),
        "v_z_valid": _bool(required=True),
        "xy_reset_counter": _uint(255),
        "z_reset_counter": _uint(255),
    },
    "battery_status": {
        "connected": _bool(),
        "voltage_v": _float(0, 100, required=True),
        "current_a": _float(-50, 1000, unknown=-1),
        "remaining": _float(0, 1, unknown=-1, required=True),
        "discharged_mah": _float(0, 1e6, unknown=-1, required=True),
        "cell_count": _uint(32),
        # NONE, LOW, CRITICAL, EMERGENCY, FAILED, (5 unused), UNHEALTHY, CHARGING
        "warning": _enum(0, 1, 2, 3, 4, 5, 6, 7),
    },
    "vehicle_status": {
        # 1.14+: DISARMED=1, ARMED=2; older firmware also used 0, 3, 4, 5.
        "arming_state": _enum(0, 1, 2, 3, 4, 5, required=True),
        "nav_state": _uint(31),
        "failsafe": _bool(),
    },
}

# --- timestamp thresholds ---
GAP_FACTOR = 10              # a gap is this many times the median sample interval...
MIN_GAP_S = 0.5              # ...and at least this long, so jitter on fast topics isn't reported
COVERAGE_TOLERANCE_S = 5.0   # topic starts/ends this far from the log edges -> data missing
MAX_TIME_BEFORE_START_S = 1.0  # samples logged slightly before start are normal

# --- PX4 signal rules ---
MAX_SPEED_MS = 100.0             # faster than any multicopter / small fixed-wing
MAX_POSITION_JUMP_M = 50.0       # between consecutive samples without an EKF reset
MAX_EPH_M = 10.0                 # horizontal position accuracy considered poor
CELL_VOLTAGE_RANGE_V = (2.5, 4.5)  # LiPo/Li-ion cell, outside this the reading is implausible
REMAINING_RISE_TOLERANCE = 0.05  # battery % shouldn't go up during a flight
BATTERY_WARNING_NAMES = {1: "LOW", 2: "CRITICAL", 3: "EMERGENCY", 4: "FAILED", 6: "UNHEALTHY"}
ARMING_STATE_ARMED = 2


def _present(topics: dict[str, pd.DataFrame], topic: str) -> pd.DataFrame | None:
    df = topics.get(topic)
    return None if df is None or df.empty else df


def _has(df: pd.DataFrame, *columns: str) -> bool:
    return all(c in df.columns for c in columns)


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def validate_timestamps(topics: dict[str, pd.DataFrame]) -> list[Issue]:
    """Missing data (gaps, partial coverage), repeated timestamps and wrong timestamps."""
    issues = []
    sampled = {name: df for name, df in topics.items() if len(df) >= 2}
    if not sampled:
        return issues
    log_start = min(df["time_s"].iloc[0] for df in sampled.values())
    log_end = max(df["time_s"].iloc[-1] for df in sampled.values())

    for topic, df in topics.items():
        if len(df) < 2:
            issues.append(Issue(WARNING, topic, "time", f"only {len(df)} sample(s), timing can't be checked"))
            continue
        t = df["time_s"].to_numpy()
        dt = np.diff(t)

        # Wrong time
        zero = int((df["timestamp"] == 0).sum())
        if zero:
            issues.append(Issue(ERROR, topic, "time", f"{zero} sample(s) with timestamp 0 (not set)"))
        too_early = int((t < -MAX_TIME_BEFORE_START_S).sum()) - zero
        if too_early > 0:
            issues.append(Issue(
                ERROR, topic, "time",
                f"{too_early} sample(s) more than {MAX_TIME_BEFORE_START_S:.0f} s before log start",
            ))
        backwards = int((dt < 0).sum())
        if backwards:
            worst = int(np.argmin(dt))
            issues.append(Issue(
                ERROR, topic, "time",
                f"timestamps go backwards {backwards} time(s); worst {dt[worst]:.3f} s at t={t[worst]:.2f} s",
            ))

        # Repeated time
        repeated = int((dt == 0).sum())
        if repeated:
            issues.append(Issue(WARNING, topic, "time", f"{repeated} repeated timestamp(s)"))

        # Missing data: gaps inside the topic
        forward = dt[dt > 0]
        median_dt = float(np.median(forward)) if len(forward) else 0.0
        gap_threshold = max(GAP_FACTOR * median_dt, MIN_GAP_S)
        gap_idx = np.flatnonzero(dt > gap_threshold)
        if len(gap_idx):
            worst = gap_idx[np.argmax(dt[gap_idx])]
            missing_s = float(dt[gap_idx].sum() - len(gap_idx) * median_dt)
            issues.append(Issue(
                WARNING, topic, "time",
                f"{len(gap_idx)} gap(s) > {gap_threshold:.2f} s (median interval {median_dt:.3f} s), "
                f"~{missing_s:.1f} s of data missing; largest {dt[worst]:.2f} s at t={t[worst]:.2f} s",
            ))

        # Missing data: topic doesn't cover the whole log
        late = t[0] - log_start
        early = log_end - t[-1]
        if late > COVERAGE_TOLERANCE_S:
            issues.append(Issue(WARNING, topic, "time", f"first sample {late:.1f} s after log start"))
        if early > COVERAGE_TOLERANCE_S:
            issues.append(Issue(WARNING, topic, "time", f"last sample {early:.1f} s before log end"))
    return issues


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------

def _dtype_matches(series: pd.Series, field: Field) -> bool:
    if field.kind == "float":
        return pd.api.types.is_float_dtype(series)
    return pd.api.types.is_integer_dtype(series) or pd.api.types.is_bool_dtype(series)


def validate_fields(topics: dict[str, pd.DataFrame]) -> list[Issue]:
    """Each known topic exists, has samples, has its fields, and each field has the right type."""
    issues = []
    for topic, spec in FIELD_SPECS.items():
        needed_by_metrics = any(f.required for f in spec.values())
        if topic not in topics:
            issues.append(Issue(ERROR if needed_by_metrics else INFO, topic, "fields", "topic not found in log"))
            continue
        df = topics[topic]
        if df.empty:
            issues.append(Issue(ERROR if needed_by_metrics else INFO, topic, "fields", "topic has no samples"))
            continue

        missing_required = [name for name, f in spec.items() if f.required and name not in df.columns]
        missing_optional = [name for name, f in spec.items() if not f.required and name not in df.columns]
        if missing_required:
            issues.append(Issue(ERROR, topic, "fields", f"missing required field(s): {', '.join(missing_required)}"))
        if missing_optional:
            # Fields get added/renamed between PX4 versions.
            issues.append(Issue(INFO, topic, "fields", f"missing optional field(s): {', '.join(missing_optional)}"))

        for name, field in spec.items():
            if name in df.columns and not _dtype_matches(df[name], field):
                issues.append(Issue(
                    ERROR if field.required else WARNING, topic, "fields",
                    f"{name}: expected {field.kind}, got {df[name].dtype}",
                ))
    return issues


# ---------------------------------------------------------------------------
# NaN / inf / wrong values
# ---------------------------------------------------------------------------

def _wrong_values(series: pd.Series, field: Field) -> pd.Series:
    """Mask of finite values outside the field's allowed domain (unknown sentinel excluded)."""
    finite = series[np.isfinite(series.astype(float))]
    if field.kind == "enum":
        wrong = ~finite.isin(field.values)
    else:
        wrong = (finite < field.min) | (finite > field.max)
    if field.unknown is not None:
        wrong &= finite != field.unknown
    return wrong


def check_missing_values(topics: dict[str, pd.DataFrame]) -> list[Issue]:
    """Count NaN, inf, 'unknown' sentinels and values outside each field's allowed domain.

    PX4 logs NaN to mean 'not available', so NaN in fields the metrics don't use is INFO.
    """
    issues = []
    for topic, df in topics.items():
        if df.empty:
            continue
        spec = FIELD_SPECS.get(topic, {})
        numeric = df.select_dtypes(include="number").drop(columns=["timestamp", "time_s"], errors="ignore")
        for column in numeric.columns:
            values = numeric[column].astype(float)
            n = len(values)
            field = spec.get(column)
            required = field is not None and field.required

            nan = int(values.isna().sum())
            inf = int(np.isinf(values).sum())
            if nan or inf:
                severity = (ERROR if nan + inf == n else WARNING) if required else INFO
                parts = [f"{nan} NaN"] if nan else []
                parts += [f"{inf} inf"] if inf else []
                issues.append(Issue(
                    severity, topic, "values",
                    f"{column}: {' + '.join(parts)} of {n} samples ({100 * (nan + inf) / n:.0f}%)",
                ))

            if field is None:
                continue
            if field.unknown is not None:
                unknown = int((values == field.unknown).sum())
                if unknown:
                    severity = WARNING if required and unknown == n else INFO
                    issues.append(Issue(
                        severity, topic, "values",
                        f"{column}: {unknown}/{n} samples are {field.unknown:g} (PX4 'unknown')",
                    ))
            wrong = int(_wrong_values(numeric[column], field).sum())
            if wrong:
                allowed = sorted(field.values) if field.kind == "enum" else f"{field.min:g}..{field.max:g}"
                issues.append(Issue(
                    ERROR if required else WARNING, topic, "values",
                    f"{column}: {wrong}/{n} wrong value(s), allowed {allowed}",
                ))
    return issues


# ---------------------------------------------------------------------------
# Signal correctness (PX4 rules)
# ---------------------------------------------------------------------------

def _signal_local_position(df: pd.DataFrame) -> list[Issue]:
    topic = "vehicle_local_position"
    issues = []

    # PX4 marks estimates it doesn't trust; metrics computed over those samples are unreliable.
    for flag in ("xy_valid", "z_valid", "v_xy_valid", "v_z_valid"):
        if _has(df, flag):
            invalid = int((df[flag] == 0).sum())
            if invalid:
                issues.append(Issue(
                    WARNING, topic, "signal",
                    f"{flag} false in {invalid}/{len(df)} samples ({100 * invalid / len(df):.1f}%)",
                ))

    if _has(df, "vx", "vy", "vz"):
        speed = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2 + df["vz"] ** 2)
        too_fast = int((speed > MAX_SPEED_MS).sum())
        if too_fast:
            issues.append(Issue(ERROR, topic, "signal", f"{too_fast} sample(s) with speed > {MAX_SPEED_MS:.0f} m/s"))

    # The EKF can shift its position estimate on purpose; it then increments the reset
    # counter. A jump without a counter change is a real discontinuity in the signal.
    if _has(df, "x", "y", "z"):
        jump = np.sqrt(df["x"].diff() ** 2 + df["y"].diff() ** 2 + df["z"].diff() ** 2)
        big = jump > MAX_POSITION_JUMP_M
        if big.any():
            reset = pd.Series(False, index=df.index)
            for counter in ("xy_reset_counter", "z_reset_counter"):
                if _has(df, counter):
                    reset |= df[counter].diff().fillna(0) != 0
            explained = int((big & reset).sum())
            unexplained = int((big & ~reset).sum())
            if explained:
                issues.append(Issue(
                    INFO, topic, "signal",
                    f"{explained} position jump(s) > {MAX_POSITION_JUMP_M:.0f} m caused by EKF reset",
                ))
            if unexplained:
                at = df["time_s"][big & ~reset].iloc[0]
                issues.append(Issue(
                    WARNING, topic, "signal",
                    f"{unexplained} position jump(s) > {MAX_POSITION_JUMP_M:.0f} m without EKF reset "
                    f"(first at t={at:.2f} s)",
                ))

    if _has(df, "eph"):
        poor = int((df["eph"] > MAX_EPH_M).sum())
        if poor:
            issues.append(Issue(
                WARNING, topic, "signal",
                f"horizontal accuracy (eph) worse than {MAX_EPH_M:.0f} m in {poor}/{len(df)} samples",
            ))
    return issues


def _signal_battery(df: pd.DataFrame) -> list[Issue]:
    topic = "battery_status"
    issues = []

    if _has(df, "connected") and (df["connected"] == 0).all():
        issues.append(Issue(WARNING, topic, "signal", "battery never reported as connected"))

    if _has(df, "voltage_v"):
        voltage = df["voltage_v"]
        measured = voltage[voltage > 0]
        if measured.empty:
            issues.append(Issue(
                WARNING, topic, "signal",
                "voltage is 0 V in every sample (no battery on this instance; check battery_status_1)",
            ))
        else:
            zero = int((voltage == 0).sum())
            if zero:
                issues.append(Issue(INFO, topic, "signal", f"{zero} sample(s) with 0 V before the battery was measured"))
            if measured.nunique() == 1 and len(measured) > 1:
                issues.append(Issue(WARNING, topic, "signal", "voltage is constant for the whole log (simulated battery?)"))
            if _has(df, "cell_count"):
                cells = df.loc[measured.index, "cell_count"]
                per_cell = (measured / cells)[cells > 0]
                low, high = CELL_VOLTAGE_RANGE_V
                implausible = int(((per_cell < low) | (per_cell > high)).sum())
                if implausible:
                    issues.append(Issue(
                        WARNING, topic, "signal",
                        f"{implausible} sample(s) with cell voltage outside {low}-{high} V "
                        f"(wrong BAT_N_CELLS or voltage divider?)",
                    ))

    if _has(df, "remaining"):
        remaining = df["remaining"][df["remaining"] >= 0]
        if len(remaining):
            rise = float(remaining.max() - remaining.iloc[0])
            if rise > REMAINING_RISE_TOLERANCE:
                issues.append(Issue(
                    WARNING, topic, "signal",
                    f"battery remaining rose by {100 * rise:.0f} points during the log (estimate unreliable)",
                ))

    if _has(df, "discharged_mah"):
        discharged = df["discharged_mah"][df["discharged_mah"] >= 0]
        # Consumed charge is integrated by PX4 and must never decrease.
        drops = int((discharged.diff() < -1e-3).sum())
        if drops:
            issues.append(Issue(WARNING, topic, "signal", f"discharged_mah decreased {drops} time(s)"))

    if _has(df, "warning"):
        # Wrong enum values are reported by check_missing_values; ignore them here.
        known = df["warning"][df["warning"].isin(BATTERY_WARNING_NAMES)]
        if len(known):
            worst = int(known.max())
            at = df["time_s"][df["warning"] == worst].iloc[0]
            severity = INFO if worst == 1 else WARNING
            issues.append(Issue(
                severity, topic, "signal",
                f"battery warning {BATTERY_WARNING_NAMES[worst]} reached at t={at:.2f} s",
            ))
    return issues


def _signal_vehicle_status(df: pd.DataFrame) -> list[Issue]:
    topic = "vehicle_status"
    issues = []
    if _has(df, "arming_state"):
        armed = df["arming_state"] == ARMING_STATE_ARMED
        if not armed.any():
            issues.append(Issue(INFO, topic, "signal", "vehicle never armed during the log"))
    if _has(df, "failsafe") and (df["failsafe"] != 0).any():
        at = df["time_s"][df["failsafe"] != 0].iloc[0]
        issues.append(Issue(WARNING, topic, "signal", f"failsafe active (first at t={at:.2f} s)"))
    return issues


_SIGNAL_CHECKS = {
    "vehicle_local_position": _signal_local_position,
    "battery_status": _signal_battery,
    "vehicle_status": _signal_vehicle_status,
}


def validate_signal(topics: dict[str, pd.DataFrame]) -> list[Issue]:
    """Check signal behaviour against PX4 rules for each known topic type."""
    issues = []
    for topic, check in _SIGNAL_CHECKS.items():
        df = _present(topics, topic)
        if df is not None:  # missing topics are reported by validate_fields
            issues.extend(check(df))
    return issues


# ---------------------------------------------------------------------------

def validate(topics: dict[str, pd.DataFrame]) -> list[Issue]:
    """Run every check and return the issues found, most severe first."""
    issues = (
        validate_fields(topics)
        + validate_timestamps(topics)
        + check_missing_values(topics)
        + validate_signal(topics)
    )
    order = {ERROR: 0, WARNING: 1, INFO: 2}
    return sorted(issues, key=lambda i: order[i.severity])


def has_errors(issues: list[Issue]) -> bool:
    return any(i.severity == ERROR for i in issues)
