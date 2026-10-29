"""Interactive flight plots (plotly) built from preprocessed topic DataFrames.

Conventions:
- measured signals are solid blue, requested (setpoint) signals dashed orange
- quantities with different units go in stacked panels, never on a second y-axis
- masked samples (NaN) and logging gaps break the line instead of being bridged
- anomaly intervals are shaded in status colors with a text label (highlight_anomalies)
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.preprocessing import align_signals, calculate_altitude, calculate_horizontal_speed, quaternion_to_euler

# Reference palette (validated: CVD-safe, all pairs)
MEASURED = "#2a78d6"
REQUESTED = "#eb6834"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Traces longer than this are reduced with min/max bucketing so spikes survive.
MAX_POINTS = 5000
# A time step this many times the median interval is drawn as a gap.
GAP_FACTOR = 5


def _layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    # Draw setpoints underneath so they never hide the measured signal; legend still lists Measured first.
    fig.data = sorted(fig.data, key=lambda trace: trace.name != "Setpoint")
    fig.for_each_trace(lambda trace: trace.update(legendrank=2 if trace.name == "Setpoint" else 1))
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 16, "color": TEXT_PRIMARY}},
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif", "size": 12, "color": TEXT_SECONDARY},
        margin={"l": 64, "r": 24, "t": 64, "b": 48},
        legend={"orientation": "h", "x": 0, "y": 1.02, "xanchor": "left", "yanchor": "bottom"},
        hovermode="x unified",
        hoverlabel={"bgcolor": SURFACE, "font": {"color": TEXT_PRIMARY}},
    )
    fig.update_xaxes(
        gridcolor=GRID, gridwidth=1, linecolor=BASELINE, zeroline=False, tickfont={"color": TEXT_MUTED},
        showspikes=True, spikemode="across", spikethickness=1, spikecolor=TEXT_MUTED, spikedash="solid",
    )
    fig.update_yaxes(
        gridcolor=GRID, gridwidth=1, linecolor=BASELINE, zeroline=False, tickfont={"color": TEXT_MUTED},
    )
    return fig


def _break_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Insert a NaN row inside every logging gap so lines don't bridge missing data."""
    t = df["time_s"].to_numpy()
    if len(t) < 3:
        return df
    dt = np.diff(t)
    positive = dt[dt > 0]
    if not len(positive):
        return df
    gaps = np.flatnonzero(dt > GAP_FACTOR * np.median(positive))
    if not len(gaps):
        return df
    filler = pd.DataFrame({"time_s": (t[gaps] + t[gaps + 1]) / 2})
    return pd.concat([df, filler], ignore_index=True).sort_values("time_s", kind="stable", ignore_index=True)


def _downsample(df: pd.DataFrame, columns: list[str], max_points: int = MAX_POINTS) -> pd.DataFrame:
    """Keep each bucket's min and max per column (and NaN gap markers) so peaks stay visible."""
    if len(df) <= max_points:
        return df
    df = df.reset_index(drop=True)
    bucket = np.arange(len(df)) * (max_points // 2) // len(df)
    keep = {0, len(df) - 1}
    for column in columns:
        values = df[column].astype(float)
        # NaN-safe: all-NaN buckets just return an arbitrary row of that bucket.
        keep.update(values.fillna(np.inf).groupby(bucket).idxmin())
        keep.update(values.fillna(-np.inf).groupby(bucket).idxmax())
        gap_start = values.isna() & values.shift(fill_value=0).notna()
        keep.update(np.flatnonzero(gap_start.to_numpy()))
    return df.loc[sorted(keep)]


def _prepare(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return _downsample(_break_gaps(df[["time_s", *columns]]), columns)


def _line(x, y, name: str, requested: bool = False, unit: str = "", **kwargs) -> go.Scattergl:
    return go.Scattergl(
        x=x, y=y, name=name, mode="lines", connectgaps=False,
        line={"color": REQUESTED if requested else MEASURED, "width": 2, "dash": "dash" if requested else "solid"},
        hovertemplate=f"%{{y:.2f}} {unit}<extra>{name}</extra>",
        **kwargs,
    )


# ---------------------------------------------------------------------------

def plot_altitude(local_pos: pd.DataFrame, setpoint: pd.DataFrame | None = None) -> go.Figure:
    """Local height H = -z over time, with the height setpoint if available."""
    data = local_pos.assign(altitude_m=calculate_altitude(local_pos))
    data = _prepare(data, ["altitude_m"])
    fig = go.Figure(_line(data["time_s"], data["altitude_m"], "Measured", unit="m"))
    if setpoint is not None and "z" in setpoint and setpoint["z"].notna().any():
        sp = _prepare(setpoint.assign(altitude_m=-setpoint["z"]), ["altitude_m"])
        fig.add_trace(_line(sp["time_s"], sp["altitude_m"], "Setpoint", requested=True, unit="m", line_shape="hv"))
    fig.update_xaxes(title="Time (s)")
    fig.update_yaxes(title="Local height (m)")
    return _layout(fig, "Local height")


def plot_velocity(local_pos: pd.DataFrame) -> go.Figure:
    """Horizontal speed and vertical speed (positive up) in two panels."""
    data = local_pos.assign(
        horizontal_speed_ms=calculate_horizontal_speed(local_pos),
        vertical_speed_ms=-local_pos["vz"],
    )
    data = _prepare(data, ["horizontal_speed_ms", "vertical_speed_ms"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=("Horizontal speed", "Vertical speed (up +)"))
    fig.add_trace(_line(data["time_s"], data["horizontal_speed_ms"], "Horizontal", unit="m/s"), row=1, col=1)
    fig.add_trace(_line(data["time_s"], data["vertical_speed_ms"], "Vertical", unit="m/s", showlegend=False), row=2, col=1)
    fig.update_yaxes(title="m/s")
    fig.update_xaxes(title="Time (s)", row=2, col=1)
    fig.update_layout(showlegend=False)
    return _layout(fig, "Velocity", height=560)


def plot_position(local_pos: pd.DataFrame, setpoint: pd.DataFrame | None = None) -> go.Figure:
    """Local XY trajectory, top-down: east (y) on the horizontal axis, north (x) vertical."""
    data = _prepare(local_pos, ["x", "y"])
    fig = go.Figure(go.Scattergl(
        x=data["y"], y=data["x"], name="Measured", mode="lines", connectgaps=False,
        line={"color": MEASURED, "width": 2},
        customdata=data["time_s"],
        hovertemplate="t=%{customdata:.1f} s<br>E %{x:.2f} m, N %{y:.2f} m<extra>Measured</extra>",
    ))
    if setpoint is not None and {"x", "y"} <= set(setpoint.columns) and setpoint["x"].notna().any():
        sp = _prepare(setpoint, ["x", "y"])
        fig.add_trace(go.Scattergl(
            x=sp["y"], y=sp["x"], name="Setpoint", mode="lines", connectgaps=False,
            line={"color": REQUESTED, "width": 2, "dash": "dash"},
            customdata=sp["time_s"],
            hovertemplate="t=%{customdata:.1f} s<br>E %{x:.2f} m, N %{y:.2f} m<extra>Setpoint</extra>",
        ))
    valid = local_pos[["x", "y"]].dropna()
    if len(valid):
        ends = valid.iloc[[0, -1]]
        fig.add_trace(go.Scatter(
            x=ends["y"], y=ends["x"], mode="markers+text", text=["start", "end"], textposition="top center",
            marker={"size": 9, "color": MEASURED, "symbol": ["circle", "square"], "line": {"color": SURFACE, "width": 2}},
            textfont={"color": TEXT_SECONDARY}, showlegend=False, hoverinfo="skip",
        ))
    fig.update_xaxes(title="East (m)", showspikes=False)
    fig.update_yaxes(title="North (m)", scaleanchor="x", scaleratio=1)
    _layout(fig, "Local trajectory (top-down)", height=560)
    fig.update_layout(hovermode="closest")
    return fig


def plot_battery(battery: pd.DataFrame) -> go.Figure:
    """Battery voltage and remaining charge in two panels."""
    battery = battery[battery["voltage_v"] > 0]  # before the first measurement
    data = battery.assign(remaining_pct=battery["remaining"].where(battery["remaining"] >= 0) * 100)
    data = _prepare(data, ["voltage_v", "remaining_pct"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=("Voltage", "Remaining charge"))
    fig.add_trace(_line(data["time_s"], data["voltage_v"], "Voltage", unit="V"), row=1, col=1)
    fig.add_trace(_line(data["time_s"], data["remaining_pct"], "Remaining", unit="%"), row=2, col=1)
    fig.update_yaxes(title="V", row=1, col=1)
    fig.update_yaxes(title="%", range=[0, 100], row=2, col=1)
    fig.update_xaxes(title="Time (s)", row=2, col=1)
    fig.update_layout(showlegend=False)
    return _layout(fig, "Battery", height=560)


def _attitude_setpoint_euler(setpoint: pd.DataFrame) -> pd.DataFrame | None:
    # PX4 >= 1.14 only logs the quaternion q_d; older logs also have roll/pitch/yaw_body.
    if all(f"q_d[{i}]" in setpoint for i in range(4)) and setpoint["q_d[0]"].notna().any():
        return quaternion_to_euler(setpoint, "q_d")
    if {"roll_body", "pitch_body", "yaw_body"} <= set(setpoint.columns):
        return pd.DataFrame({
            "time_s": setpoint["time_s"],
            "roll_deg": np.degrees(setpoint["roll_body"]),
            "pitch_deg": np.degrees(setpoint["pitch_body"]),
            "yaw_deg": np.degrees(setpoint["yaw_body"]),
        })
    return None


def plot_attitude(attitude: pd.DataFrame, setpoint: pd.DataFrame | None = None) -> go.Figure:
    """Roll, pitch and yaw over time (from the attitude quaternion), with setpoints if available."""
    angles = ["roll_deg", "pitch_deg", "yaw_deg"]
    measured = _prepare(quaternion_to_euler(attitude), angles)
    requested = _attitude_setpoint_euler(setpoint) if setpoint is not None else None
    if requested is not None:
        requested = _prepare(requested, angles)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Roll", "Pitch", "Yaw"))
    for row, angle in enumerate(angles, start=1):
        first = row == 1
        fig.add_trace(_line(measured["time_s"], measured[angle], "Measured", unit="°", showlegend=first,
                            legendgroup="measured"), row=row, col=1)
        if requested is not None:
            fig.add_trace(_line(requested["time_s"], requested[angle], "Setpoint", requested=True, unit="°",
                                showlegend=first, legendgroup="setpoint"), row=row, col=1)
        fig.update_yaxes(title="deg", row=row, col=1)
    fig.update_yaxes(range=[-180, 180], tickvals=[-180, -90, 0, 90, 180], row=3, col=1)
    fig.update_xaxes(title="Time (s)", row=3, col=1)
    return _layout(fig, "Attitude", height=760)


def plot_tracking_error(setpoint: pd.DataFrame, local_pos: pd.DataFrame) -> go.Figure:
    """Horizontal and vertical position error (measured - setpoint) over time.

    Gaps mean PX4 wasn't controlling that axis (NaN setpoint) or the position was invalid.
    """
    aligned = align_signals(setpoint, local_pos, ["x", "y", "z"])
    aligned["horizontal_error_m"] = np.hypot(aligned["x_error"], aligned["y_error"])
    aligned["vertical_error_m"] = -aligned["z_error"]  # positive = above the setpoint

    def rmse(s: pd.Series) -> str:
        s = s.dropna()
        return f"{np.sqrt(np.mean(s ** 2)):.2f} m" if len(s) else "n/a"

    data = _prepare(aligned, ["horizontal_error_m", "vertical_error_m"])
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
        subplot_titles=(
            f"Horizontal error · RMSE {rmse(aligned['horizontal_error_m'])}",
            f"Vertical error (above setpoint +) · RMSE {rmse(aligned['vertical_error_m'])}",
        ),
    )
    fig.add_trace(_line(data["time_s"], data["horizontal_error_m"], "Horizontal", unit="m"), row=1, col=1)
    fig.add_trace(_line(data["time_s"], data["vertical_error_m"], "Vertical", unit="m"), row=2, col=1)
    fig.add_hline(y=0, line={"color": BASELINE, "width": 1}, row=2, col=1)
    fig.update_yaxes(title="m")
    fig.update_xaxes(title="Time (s)", row=2, col=1)
    fig.update_layout(showlegend=False)
    return _layout(fig, "Position tracking error", height=560)


# ---------------------------------------------------------------------------

# Topics the plots use; pass to parser.load_topics together with metrics.REQUIRED_TOPICS.
PLOT_TOPICS = [
    "vehicle_local_position", "vehicle_local_position_setpoint", "battery_status",
    "vehicle_attitude", "vehicle_attitude_setpoint",
]


def create_figures(topics: dict[str, pd.DataFrame]) -> dict[str, go.Figure]:
    """Every plot whose source topics are in the log; missing ones are skipped."""

    def get(name: str) -> pd.DataFrame | None:
        df = topics.get(name)
        return df if df is not None and len(df) >= 2 else None

    local_pos = get("vehicle_local_position")
    setpoint = get("vehicle_local_position_setpoint")
    battery = get("battery_status")
    attitude = get("vehicle_attitude")

    figures: dict[str, go.Figure] = {}
    if local_pos is not None:
        figures["altitude"] = plot_altitude(local_pos, setpoint)
        figures["velocity"] = plot_velocity(local_pos)
        figures["position"] = plot_position(local_pos, setpoint)
        if setpoint is not None and setpoint[["x", "y", "z"]].notna().any().any():
            figures["tracking_error"] = plot_tracking_error(setpoint, local_pos)
    if battery is not None and (battery["voltage_v"] > 0).any():
        figures["battery"] = plot_battery(battery)
    if attitude is not None:
        figures["attitude"] = plot_attitude(attitude, get("vehicle_attitude_setpoint"))
    return figures


# Status colors (reserved for anomaly severity, always shown with a text label)
SEVERITY_COLORS = {"critical": "#d03b3b", "warning": "#fab219"}

# Which charts show each detector's events. Sampling gaps / invalid data map by topic.
_CHARTS_BY_DETECTOR = {
    "logging_dropouts": ["altitude", "velocity", "battery", "attitude", "tracking_error"],
    "battery": ["battery"],
    "altitude_tracking": ["altitude", "tracking_error"],
    "position_tracking": ["tracking_error"],
    "estimator_resets": ["altitude", "tracking_error"],
    "estimator_quality": ["altitude", "tracking_error"],
    "flight_events": ["altitude", "velocity", "attitude"],
}
_CHARTS_BY_TOPIC = {
    "vehicle_local_position": ["altitude", "velocity", "tracking_error"],
    "battery_status": ["battery"],
    "vehicle_attitude": ["attitude"],
}


def _charts_for(anomaly) -> list[str]:
    if anomaly.detector == "sampling_gaps":
        topics = anomaly.evidence.get("topics", [anomaly.topic])
        return sorted({chart for topic in topics for chart in _CHARTS_BY_TOPIC.get(topic, [])})
    return _CHARTS_BY_DETECTOR.get(anomaly.detector, [])


def highlight_anomalies(figures: dict[str, go.Figure], anomalies: list) -> dict[str, go.Figure]:
    """Shade warning/critical anomaly intervals (and mark instants) on the time charts they concern.

    Informational events, summary events and PX4 text messages are not drawn; the report lists them.
    Each chart labels the first event of each check, so color is never the only cue.
    """
    labelled: set[tuple[str, str]] = set()
    label_count: dict[str, int] = {}
    for anomaly in anomalies:
        color = SEVERITY_COLORS.get(anomaly.severity)
        if color is None or anomaly.evidence.get("summary") or anomaly.start_s is None:
            continue
        if anomaly.topic == "logged_messages":  # text messages are listed in the report, not drawn
            continue
        for chart in _charts_for(anomaly):
            fig = figures.get(chart)
            if fig is None:
                continue
            rows = {"row": "all", "col": 1} if fig._has_subplots() else {}
            if anomaly.end_s is not None and anomaly.end_s > anomaly.start_s:
                fig.add_vrect(x0=anomaly.start_s, x1=anomaly.end_s, fillcolor=color, opacity=0.18,
                              line_width=0, layer="below", **rows)
            else:
                fig.add_vline(x=anomaly.start_s, line={"color": color, "width": 1.5, "dash": "dot"}, **rows)
            # One label per check per chart; the report's table has every event.
            if (chart, anomaly.detector) not in labelled:
                labelled.add((chart, anomaly.detector))
                # Stagger labels in three rows so nearby events don't overlap.
                n = label_count.get(chart, 0)
                label_count[chart] = n + 1
                fig.add_annotation(
                    x=anomaly.start_s, y=1 - 0.09 * (n % 3), xref="x", yref="y domain", xanchor="left", yanchor="top",
                    text=f"{'▲' if anomaly.severity == 'critical' else '●'} {anomaly.title}",
                    showarrow=False, font={"size": 10, "color": TEXT_SECONDARY},
                    bgcolor="rgba(252,252,251,0.85)", bordercolor=color, borderwidth=1,
                )
    return figures
