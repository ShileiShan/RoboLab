#!/usr/bin/env python3
"""Build a 4-way Piper joint tracking report with shared real-state reference."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np


BASELINE_COLOR = "#dc742d"
CANDIDATE_COLOR = "#2c82f6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--four-param-baseline-summary", type=Path, required=True)
    parser.add_argument("--four-param-baseline-timeseries", type=Path, required=True)
    parser.add_argument("--four-param-candidate-summary", type=Path, required=True)
    parser.add_argument("--four-param-candidate-timeseries", type=Path, required=True)
    parser.add_argument("--two-param-baseline-summary", type=Path, required=True)
    parser.add_argument("--two-param-baseline-timeseries", type=Path, required=True)
    parser.add_argument("--two-param-candidate-summary", type=Path, required=True)
    parser.add_argument("--two-param-candidate-timeseries", type=Path, required=True)
    parser.add_argument("--four-param-baseline-video", required=True)
    parser.add_argument("--four-param-candidate-video", required=True)
    parser.add_argument("--two-param-baseline-video", required=True)
    parser.add_argument("--two-param-candidate-video", required=True)
    parser.add_argument("--four-param-baseline-sheet", required=True)
    parser.add_argument("--four-param-candidate-sheet", required=True)
    parser.add_argument("--two-param-baseline-sheet", required=True)
    parser.add_argument("--two-param-candidate-sheet", required=True)
    parser.add_argument("--four-param-left-plot", required=True)
    parser.add_argument("--four-param-right-plot", required=True)
    parser.add_argument("--four-param-gripper-plot", required=True)
    parser.add_argument("--two-param-left-plot", required=True)
    parser.add_argument("--two-param-right-plot", required=True)
    parser.add_argument("--two-param-gripper-plot", required=True)
    parser.add_argument("--lead-sim-steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: np.asarray(handle[key]) for key in handle.files}


def _lead_steps(values: np.ndarray, steps: int) -> np.ndarray:
    values = np.asarray(values)
    if steps <= 0 or values.shape[0] <= 1:
        return values.copy()
    steps = min(int(steps), values.shape[0] - 1)
    return np.concatenate((values[steps:], np.repeat(values[-1:], steps, axis=0)), axis=0)


def _compute_display(timeseries: dict[str, np.ndarray], lead_steps: int) -> dict:
    time_s = np.asarray(timeseries["time_s"], dtype=np.float64)
    reference_state = np.asarray(timeseries["reference_state"], dtype=np.float64)
    actual_state = _lead_steps(np.asarray(timeseries["actual_state"], dtype=np.float64), lead_steps)

    arm_reference = reference_state[:, 0:12]
    gripper_reference = reference_state[:, 12:14]
    arm_actual = actual_state[:, 0:12]
    gripper_actual = actual_state[:, 12:14]

    arm_error = arm_actual - arm_reference
    gripper_error = gripper_actual - gripper_reference
    arm_abs = np.abs(arm_error)
    gripper_abs = np.abs(gripper_error)

    return {
        "time_s": time_s,
        "arm_mean_abs_error_curve": np.mean(arm_abs, axis=1),
        "arm_max_abs_error_curve": np.max(arm_abs, axis=1),
        "left_arm_mean_abs_error_curve": np.mean(arm_abs[:, 0:6], axis=1),
        "right_arm_mean_abs_error_curve": np.mean(arm_abs[:, 6:12], axis=1),
        "left_gripper_abs_error_curve": gripper_abs[:, 0],
        "right_gripper_abs_error_curve": gripper_abs[:, 1],
        "arm_mae": float(np.mean(arm_abs)),
        "arm_rmse": float(np.sqrt(np.mean(np.square(arm_error)))),
        "arm_max_abs": float(np.max(arm_abs)),
        "left_arm_rmse": float(np.sqrt(np.mean(np.square(arm_error[:, 0:6])))),
        "right_arm_rmse": float(np.sqrt(np.mean(np.square(arm_error[:, 6:12])))),
        "gripper_mae": float(np.mean(gripper_abs)),
        "gripper_rmse": float(np.sqrt(np.mean(np.square(gripper_error)))),
        "left_gripper_rmse": float(np.sqrt(np.mean(np.square(gripper_error[:, 0])))),
        "right_gripper_rmse": float(np.sqrt(np.mean(np.square(gripper_error[:, 1])))),
    }


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _compare_rows(
    left_metrics: dict[str, float],
    right_metrics: dict[str, float],
    *,
    left_name: str,
    right_name: str,
) -> str:
    keys = [
        ("Arm MAE", "arm_mae"),
        ("Arm RMSE", "arm_rmse"),
        ("Arm Max Abs", "arm_max_abs"),
        ("Left Arm RMSE", "left_arm_rmse"),
        ("Right Arm RMSE", "right_arm_rmse"),
        ("Gripper MAE", "gripper_mae"),
        ("Gripper RMSE", "gripper_rmse"),
        ("Left Gripper RMSE", "left_gripper_rmse"),
        ("Right Gripper RMSE", "right_gripper_rmse"),
    ]
    rows = []
    for label, key in keys:
        left_value = float(left_metrics[key])
        right_value = float(right_metrics[key])
        winner = left_name if left_value < right_value else right_name if right_value < left_value else "Tie"
        rows.append(
            f"<tr><th>{html.escape(label)}</th>"
            f"<td>{_fmt(left_value)}</td>"
            f"<td>{_fmt(right_value)}</td>"
            f"<td>{right_value - left_value:+.6f}</td>"
            f"<td>{html.escape(winner)}</td></tr>"
        )
    return "".join(rows)


def _metadata_table(summary: dict, title: str) -> str:
    params = summary.get("actuator_params", {})
    rows = [
        ("Source HDF5", summary.get("source_hdf5")),
        ("Control Hz", summary.get("control_hz")),
        ("Reference States", summary.get("num_reference_states")),
    ]
    if "stiffness" in params:
        rows.append(("Stiffness", params["stiffness"]))
    if "damping" in params:
        rows.append(("Damping", params["damping"]))
    if "armature" in params:
        rows.append(("Armature", params["armature"]))
    if "friction" in params:
        rows.append(("Friction", params["friction"]))
    body = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in rows
    )
    return f"<table class=\"meta-table\"><caption>{html.escape(title)}</caption>{body}</table>"


def _render_chart(title: str, x_values: np.ndarray, series: list[dict], *, y_label: str) -> str:
    payload = {
        "title": title,
        "y_label": y_label,
        "x_values": np.asarray(x_values, dtype=np.float64).tolist(),
        "series": [
            {
                "name": item["name"],
                "color": item["color"],
                "dash": item.get("dash"),
                "values": np.asarray(item["values"], dtype=np.float64).tolist(),
            }
            for item in series
        ],
    }
    payload_json = html.escape(json.dumps(payload, separators=(",", ":")))
    legend = "".join(
        "<span class=\"legend-item\">"
        f"<span class=\"legend-swatch\" style=\"background:{item['color']}\"></span>"
        f"{html.escape(item['name'])}</span>"
        for item in series
    )
    return f"""
    <div class="chart-card interactive-chart" data-chart="{payload_json}">
      <div class="chart-header">
        <h3>{html.escape(title)}</h3>
        <div class="chart-header-right">
          <span class="unit-badge">Unit: {html.escape(y_label)}</span>
          <div class="legend">{legend}</div>
        </div>
      </div>
      <div class="chart-stage">
        <svg class="chart-svg" viewBox="0 0 920 280" role="img" aria-label="{html.escape(title)}"></svg>
        <div class="chart-tooltip" hidden></div>
      </div>
    </div>
    """


def _error_curve_section(
    *,
    title: str,
    baseline: dict,
    candidate: dict,
    baseline_label: str,
    candidate_label: str,
) -> str:
    time_s = np.asarray(baseline["time_s"], dtype=np.float64)
    charts = [
        ("Arm Mean Absolute Error", "rad", "arm_mean_abs_error_curve"),
        ("Arm Max Absolute Error", "rad", "arm_max_abs_error_curve"),
        ("Left Arm Mean Absolute Error", "rad", "left_arm_mean_abs_error_curve"),
        ("Right Arm Mean Absolute Error", "rad", "right_arm_mean_abs_error_curve"),
        ("Left Gripper Absolute Error", "m", "left_gripper_abs_error_curve"),
        ("Right Gripper Absolute Error", "m", "right_gripper_abs_error_curve"),
    ]
    rendered = []
    for chart_title, unit, key in charts:
        rendered.append(
            _render_chart(
                chart_title,
                time_s,
                [
                    {"name": baseline_label, "color": BASELINE_COLOR, "values": baseline[key]},
                    {"name": candidate_label, "color": CANDIDATE_COLOR, "values": candidate[key]},
                ],
                y_label=unit,
            )
        )
    return f"""
      <details open>
        <summary>{html.escape(title)}</summary>
        <div class="chart-stack">
          {''.join(rendered)}
        </div>
      </details>
    """


def main() -> None:
    args = parse_args()

    fpb_summary = _load_json(args.four_param_baseline_summary)
    fpc_summary = _load_json(args.four_param_candidate_summary)
    tpb_summary = _load_json(args.two_param_baseline_summary)
    tpc_summary = _load_json(args.two_param_candidate_summary)

    fpb_display = _compute_display(_load_npz(args.four_param_baseline_timeseries), args.lead_sim_steps)
    fpc_display = _compute_display(_load_npz(args.four_param_candidate_timeseries), args.lead_sim_steps)
    tpb_display = _compute_display(_load_npz(args.two_param_baseline_timeseries), args.lead_sim_steps)
    tpc_display = _compute_display(_load_npz(args.two_param_candidate_timeseries), args.lead_sim_steps)

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Piper Joint Tracking 4-Way Comparison</title>
  <style>
    :root {{
      --bg: #f4f1eb;
      --panel: #fffdf8;
      --border: #d9d3c7;
      --text: #171717;
      --muted: #5f5a52;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--text);
      background: radial-gradient(circle at top, #fff9ef 0%, var(--bg) 55%);
    }}
    h1, h2, h3, p {{ margin: 0; }}
    .shell {{ max-width: 1600px; margin: 0 auto; display: grid; gap: 18px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 12px 30px rgba(24, 18, 8, 0.07);
    }}
    .subtle {{ color: var(--muted); }}
    .toolbar {{
      display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px;
      margin-bottom: 16px; padding: 12px 14px;
      border: 1px solid var(--border); border-radius: 14px;
      background: rgba(255,255,255,0.72);
    }}
    .toolbar button {{
      appearance: none; border: 0; border-radius: 999px; padding: 9px 16px;
      font: inherit; font-weight: 600; color: #fffdf8;
      background: linear-gradient(135deg, #1f5ea8 0%, #15395f 100%); cursor: pointer;
    }}
    .toolbar button.secondary {{
      background: linear-gradient(135deg, #8c5d19 0%, #5d3a08 100%);
    }}
    .toolbar label {{
      display: inline-flex; align-items: center; gap: 8px; color: var(--muted); font-size: 14px;
    }}
    .video-grid, .sheet-grid, .plot-grid, .meta-grid {{
      display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    }}
    .chart-stack {{ display: grid; gap: 16px; margin-top: 12px; }}
    figure {{ margin: 0; display: grid; gap: 10px; }}
    figcaption {{
      font-weight: 700; display: flex; justify-content: space-between; gap: 12px;
    }}
    video, img {{
      width: 100%; border-radius: 14px; border: 1px solid #d8d0c1; background: #0a0a0a; display: block;
    }}
    details {{
      border: 1px solid var(--border); border-radius: 14px; padding: 12px 14px;
      background: rgba(255,255,255,0.7);
    }}
    details + details {{ margin-top: 12px; }}
    summary {{ cursor: pointer; font-weight: 700; list-style: none; }}
    summary::-webkit-details-marker {{ display: none; }}
    .metric-table, .meta-table {{
      width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px;
    }}
    .metric-table th, .metric-table td, .meta-table th, .meta-table td {{
      border-bottom: 1px solid var(--border); text-align: left; padding: 10px 8px; vertical-align: top;
    }}
    .sheet-link {{ text-decoration: none; color: inherit; }}
    .sheet-link:hover figcaption {{ color: #234d7f; }}
    .meta-table caption {{
      text-align: left; font-weight: 700; margin-bottom: 8px;
    }}
    .chart-card {{
      border: 1px solid var(--border); border-radius: 14px; padding: 12px; background: #fff;
    }}
    .chart-header {{
      display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px; margin-bottom: 8px;
    }}
    .chart-header-right {{
      display: flex; flex-wrap: wrap; justify-content: flex-end; align-items: center; gap: 8px 12px;
    }}
    .unit-badge {{
      display: inline-flex; align-items: center; padding: 5px 10px; border-radius: 999px;
      background: #f2ece1; color: #655b4b; font-size: 12px; font-weight: 700; letter-spacing: 0.02em;
    }}
    .legend {{
      display: flex; flex-wrap: wrap; gap: 10px; font-size: 13px; color: var(--muted);
    }}
    .legend-item {{
      display: inline-flex; align-items: center; gap: 6px;
    }}
    .legend-swatch {{
      width: 12px; height: 12px; border-radius: 999px; display: inline-block;
    }}
    .grid {{ stroke: #e9e2d6; stroke-width: 1; }}
    .axis {{ stroke: #948a78; stroke-width: 1.3; }}
    .axis-label {{
      fill: #5c574e; font-size: 12px; font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    }}
    .chart-stage {{ position: relative; }}
    .chart-svg {{ width: 100%; height: auto; display: block; }}
    .chart-overlay {{ cursor: crosshair; }}
    .hover-line {{
      stroke: #413b32; stroke-width: 1.25; stroke-dasharray: 5 4;
    }}
    .chart-tooltip {{
      position: absolute; min-width: 190px; max-width: min(260px, 82%);
      padding: 10px 12px; border-radius: 12px; border: 1px solid #cfc5b4;
      background: rgba(255,253,248,0.96); box-shadow: 0 10px 26px rgba(26,20,10,0.16);
      font-size: 13px; line-height: 1.45; pointer-events: none;
    }}
    .tooltip-row {{
      display: inline-flex; align-items: center; gap: 6px;
    }}
    .tooltip-swatch {{
      width: 10px; height: 10px; border-radius: 999px; display: inline-block; flex: 0 0 auto;
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="panel">
      <h1>Piper Joint Tracking 4-Way Comparison</h1>
      <p class="subtle">
        All four overlays below now use the same real-state proxy reference source. Top row is the original
        four-parameter comparison. Bottom row removes friction and armature from PiperCfg and compares only kp/kd.
      </p>
      <p class="subtle">Displayed qpos/absolute-error curves shift simulated trajectories {args.lead_sim_steps} control steps earlier to compensate for the observed lag.</p>
    </section>

    <section class="panel">
      <details open>
        <summary>Cross-Configuration Metric Comparison</summary>
        <h3>Baseline: 2-Param vs 4-Param</h3>
        <table class="metric-table">
          <tr><th>Metric</th><th>2-Param (kp=400, kd=80)</th><th>4-Param (f=0.10, a=0.05, kp=400, kd=80)</th><th>4-Param - 2-Param</th><th>Winner</th></tr>
          {_compare_rows(tpb_display, fpb_display, left_name="2-Param", right_name="4-Param")}
        </table>
        <h3 style="margin-top:16px;">Candidate: 2-Param vs 4-Param</h3>
        <table class="metric-table">
          <tr><th>Metric</th><th>2-Param (kp=867.0147, kd=127.7443)</th><th>4-Param (f=0.3516, a=0.4855, kp=867.0147, kd=127.7443)</th><th>4-Param - 2-Param</th><th>Winner</th></tr>
          {_compare_rows(tpc_display, fpc_display, left_name="2-Param", right_name="4-Param")}
        </table>
      </details>

      <details>
        <summary>Within-Setting Metric Comparison</summary>
        <h3>Four-Parameter Comparison</h3>
        <table class="metric-table">
          <tr><th>Metric</th><th>Baseline (f=0.10, a=0.05, kp=400, kd=80)</th><th>Candidate (f=0.3516, a=0.4855, kp=867.0147, kd=127.7443)</th><th>Candidate - Baseline</th><th>Winner</th></tr>
          {_compare_rows(fpb_display, fpc_display, left_name="Baseline", right_name="Candidate")}
        </table>
        <h3 style="margin-top:16px;">Two-Parameter Comparison</h3>
        <table class="metric-table">
          <tr><th>Metric</th><th>Baseline (kp=400, kd=80)</th><th>Candidate (kp=867.0147, kd=127.7443)</th><th>Candidate - Baseline</th><th>Winner</th></tr>
          {_compare_rows(tpb_display, tpc_display, left_name="Baseline", right_name="Candidate")}
        </table>
      </details>
    </section>

    <section class="panel">
      <div class="toolbar">
        <button id="play-all" type="button">Play All Videos</button>
        <button id="pause-all" type="button" class="secondary">Pause All Videos</button>
        <label><input id="sync-seek" type="checkbox" checked /> Keep timelines synchronized</label>
        <span id="sync-status" class="subtle">Ready</span>
      </div>
      <div class="video-grid">
        <figure>
          <figcaption><span>4-Param Baseline vs Real</span><span class="subtle">blue = real-state, orange = baseline</span></figcaption>
          <video class="sync-video" controls preload="metadata" src="{html.escape(args.four_param_baseline_video)}"></video>
        </figure>
        <figure>
          <figcaption><span>4-Param Candidate vs Real</span><span class="subtle">blue = real-state, orange = candidate</span></figcaption>
          <video class="sync-video" controls preload="metadata" src="{html.escape(args.four_param_candidate_video)}"></video>
        </figure>
        <figure>
          <figcaption><span>2-Param Baseline vs Real</span><span class="subtle">blue = real-state, orange = baseline</span></figcaption>
          <video class="sync-video" controls preload="metadata" src="{html.escape(args.two_param_baseline_video)}"></video>
        </figure>
        <figure>
          <figcaption><span>2-Param Candidate vs Real</span><span class="subtle">blue = real-state, orange = candidate</span></figcaption>
          <video class="sync-video" controls preload="metadata" src="{html.escape(args.two_param_candidate_video)}"></video>
        </figure>
      </div>

      {_error_curve_section(
        title="Four-Parameter Error Curves",
        baseline=fpb_display,
        candidate=fpc_display,
        baseline_label="Baseline (f=0.10, a=0.05, kp=400, kd=80)",
        candidate_label="Candidate (f=0.3516, a=0.4855, kp=867.0147, kd=127.7443)",
      )}

      {_error_curve_section(
        title="Two-Parameter Error Curves",
        baseline=tpb_display,
        candidate=tpc_display,
        baseline_label="Baseline (kp=400, kd=80)",
        candidate_label="Candidate (kp=867.0147, kd=127.7443)",
      )}

      <details>
        <summary>Contact Sheets</summary>
        <div class="sheet-grid" style="margin-top:12px;">
          <a class="sheet-link" href="{html.escape(args.four_param_baseline_sheet)}" target="_blank" rel="noopener noreferrer"><figure><figcaption><span>4-Param Baseline vs Real</span><span class="subtle">Open full image</span></figcaption><img src="{html.escape(args.four_param_baseline_sheet)}" alt="4-param baseline contact sheet" /></figure></a>
          <a class="sheet-link" href="{html.escape(args.four_param_candidate_sheet)}" target="_blank" rel="noopener noreferrer"><figure><figcaption><span>4-Param Candidate vs Real</span><span class="subtle">Open full image</span></figcaption><img src="{html.escape(args.four_param_candidate_sheet)}" alt="4-param candidate contact sheet" /></figure></a>
          <a class="sheet-link" href="{html.escape(args.two_param_baseline_sheet)}" target="_blank" rel="noopener noreferrer"><figure><figcaption><span>2-Param Baseline vs Real</span><span class="subtle">Open full image</span></figcaption><img src="{html.escape(args.two_param_baseline_sheet)}" alt="2-param baseline contact sheet" /></figure></a>
          <a class="sheet-link" href="{html.escape(args.two_param_candidate_sheet)}" target="_blank" rel="noopener noreferrer"><figure><figcaption><span>2-Param Candidate vs Real</span><span class="subtle">Open full image</span></figcaption><img src="{html.escape(args.two_param_candidate_sheet)}" alt="2-param candidate contact sheet" /></figure></a>
        </div>
      </details>

      <details>
        <summary>Per-Joint Qpos Plots</summary>
        <h3 style="margin-top:12px;">Four-Parameter</h3>
        <div class="plot-grid" style="margin-top:12px;">
          <figure><figcaption>Left Arm</figcaption><img src="{html.escape(args.four_param_left_plot)}" alt="Four-param left arm plot" /></figure>
          <figure><figcaption>Right Arm</figcaption><img src="{html.escape(args.four_param_right_plot)}" alt="Four-param right arm plot" /></figure>
          <figure><figcaption>Gripper</figcaption><img src="{html.escape(args.four_param_gripper_plot)}" alt="Four-param gripper plot" /></figure>
        </div>
        <h3 style="margin-top:16px;">Two-Parameter</h3>
        <div class="plot-grid" style="margin-top:12px;">
          <figure><figcaption>Left Arm</figcaption><img src="{html.escape(args.two_param_left_plot)}" alt="Two-param left arm plot" /></figure>
          <figure><figcaption>Right Arm</figcaption><img src="{html.escape(args.two_param_right_plot)}" alt="Two-param right arm plot" /></figure>
          <figure><figcaption>Gripper</figcaption><img src="{html.escape(args.two_param_gripper_plot)}" alt="Two-param gripper plot" /></figure>
        </div>
      </details>

      <details>
        <summary>Run Metadata</summary>
        <div class="meta-grid" style="margin-top:12px;">
          {_metadata_table(fpb_summary, "4-Param Baseline")}
          {_metadata_table(fpc_summary, "4-Param Candidate")}
          {_metadata_table(tpb_summary, "2-Param Baseline")}
          {_metadata_table(tpc_summary, "2-Param Candidate")}
        </div>
      </details>
    </section>
  </main>

  <script>
    const svgNs = "http://www.w3.org/2000/svg";

    function createSvg(tag, attrs = {{}}) {{
      const node = document.createElementNS(svgNs, tag);
      for (const [key, value] of Object.entries(attrs)) {{
        node.setAttribute(key, String(value));
      }}
      return node;
    }}

    function renderInteractiveChart(card) {{
      const payload = JSON.parse(card.dataset.chart);
      const svg = card.querySelector(".chart-svg");
      const tooltip = card.querySelector(".chart-tooltip");
      const width = 920;
      const height = 280;
      const margin = {{ top: 18, right: 22, bottom: 42, left: 76 }};
      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;
      const xValues = payload.x_values.slice();
      const ySeries = payload.series.map((item) => item.values.slice());
      const flat = ySeries.flat();
      let yMin = Math.min(...flat);
      let yMax = Math.max(...flat);
      if (!(yMax > yMin)) yMax = yMin + 1;
      const yPad = Math.max((yMax - yMin) * 0.08, 1e-6);
      yMin -= yPad;
      yMax += yPad;
      const xMin = xValues[0];
      const xMax = xValues[xValues.length - 1] > xMin ? xValues[xValues.length - 1] : xMin + 1;
      const mapX = (value) => margin.left + ((value - xMin) / (xMax - xMin)) * innerWidth;
      const mapY = (value) => height - margin.bottom - ((value - yMin) / (yMax - yMin)) * innerHeight;

      svg.innerHTML = "";
      svg.appendChild(createSvg("rect", {{ x: 0, y: 0, width, height, fill: "#ffffff", rx: 12, ry: 12 }}));

      for (let idx = 0; idx < 5; idx += 1) {{
        const fraction = idx / 4;
        const y = margin.top + fraction * innerHeight;
        const value = yMax - fraction * (yMax - yMin);
        svg.appendChild(createSvg("line", {{ x1: margin.left, y1: y, x2: width - margin.right, y2: y, class: "grid" }}));
        const label = createSvg("text", {{ x: margin.left - 10, y: y + 4, class: "axis-label", "text-anchor": "end" }});
        label.textContent = value.toFixed(4);
        svg.appendChild(label);
      }}

      svg.appendChild(createSvg("line", {{ x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, class: "axis" }}));
      svg.appendChild(createSvg("line", {{ x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, class: "axis" }}));

      const xTicks = [
        {{ x: margin.left, value: xMin }},
        {{ x: margin.left + innerWidth / 2, value: (xMin + xMax) / 2 }},
        {{ x: width - margin.right, value: xMax }},
      ];
      for (const tick of xTicks) {{
        const label = createSvg("text", {{ x: tick.x, y: height - 12, class: "axis-label", "text-anchor": "middle" }});
        label.textContent = `${{tick.value.toFixed(2)}}s`;
        svg.appendChild(label);
      }}
      const xLabel = createSvg("text", {{ x: margin.left + innerWidth / 2, y: height - 24, class: "axis-label", "text-anchor": "middle" }});
      xLabel.textContent = "Time";
      svg.appendChild(xLabel);

      const hoverLine = createSvg("line", {{ y1: margin.top, y2: height - margin.bottom, class: "hover-line", visibility: "hidden" }});
      svg.appendChild(hoverLine);
      const circles = payload.series.map((item) => {{
        const circle = createSvg("circle", {{ r: 4.5, fill: item.color, stroke: "#ffffff", "stroke-width": 1.5, visibility: "hidden" }});
        svg.appendChild(circle);
        return circle;
      }});

      for (const item of payload.series) {{
        const points = item.values.map((value, index) => `${{mapX(xValues[index]).toFixed(2)}},${{mapY(value).toFixed(2)}}`).join(" ");
        const polyline = createSvg("polyline", {{ fill: "none", stroke: item.color, "stroke-width": 2.5, points }});
        if (item.dash) polyline.setAttribute("stroke-dasharray", item.dash);
        svg.appendChild(polyline);
      }}

      const overlay = createSvg("rect", {{ x: margin.left, y: margin.top, width: innerWidth, height: innerHeight, fill: "transparent", class: "chart-overlay" }});
      svg.appendChild(overlay);

      function updateHover(clientX) {{
        const rect = svg.getBoundingClientRect();
        const x = ((clientX - rect.left) / rect.width) * width;
        const clamped = Math.min(width - margin.right, Math.max(margin.left, x));
        const index = Math.min(xValues.length - 1, Math.max(0, Math.round(((clamped - margin.left) / innerWidth) * (xValues.length - 1))));
        const currentX = mapX(xValues[index]);
        hoverLine.setAttribute("x1", currentX);
        hoverLine.setAttribute("x2", currentX);
        hoverLine.setAttribute("visibility", "visible");
        const lines = [`<strong>${{payload.title}}</strong>`, `t = ${{xValues[index].toFixed(3)}} s`];
        payload.series.forEach((item, seriesIndex) => {{
          const value = item.values[index];
          circles[seriesIndex].setAttribute("cx", currentX);
          circles[seriesIndex].setAttribute("cy", mapY(value));
          circles[seriesIndex].setAttribute("visibility", "visible");
          lines.push(`<span class="tooltip-row"><span class="tooltip-swatch" style="background:${{item.color}}"></span>${{item.name}}: ${{value.toFixed(6)}}</span>`);
        }});
        tooltip.innerHTML = lines.join("<br />");
        tooltip.hidden = false;
        const stageRect = card.querySelector(".chart-stage").getBoundingClientRect();
        const left = ((clientX - stageRect.left) / stageRect.width) * 100;
        tooltip.style.left = `${{Math.min(84, Math.max(4, left + 1.5))}}%`;
        tooltip.style.top = "10px";
      }}

      overlay.addEventListener("mousemove", (event) => updateHover(event.clientX));
      overlay.addEventListener("mouseenter", (event) => updateHover(event.clientX));
      overlay.addEventListener("mouseleave", () => {{
        tooltip.hidden = true;
        hoverLine.setAttribute("visibility", "hidden");
        circles.forEach((circle) => circle.setAttribute("visibility", "hidden"));
      }});
    }}

    function setupSynchronizedVideos() {{
      const videos = Array.from(document.querySelectorAll(".sync-video"));
      const playButton = document.getElementById("play-all");
      const pauseButton = document.getElementById("pause-all");
      const syncSeek = document.getElementById("sync-seek");
      const status = document.getElementById("sync-status");
      let syncing = false;

      function runSync(work) {{
        if (syncing) return;
        syncing = true;
        try {{ work(); }}
        finally {{ window.requestAnimationFrame(() => {{ syncing = false; }}); }}
      }}

      function alignTo(source) {{
        if (!syncSeek.checked) return;
        videos.forEach((video) => {{
          if (video === source) return;
          if (Math.abs(video.currentTime - source.currentTime) > 0.03) {{
            video.currentTime = source.currentTime;
          }}
        }});
      }}

      playButton.addEventListener("click", async () => {{
        const anchor = videos[0].currentTime;
        runSync(() => {{
          if (syncSeek.checked) videos.forEach((video) => {{ video.currentTime = anchor; }});
        }});
        const results = await Promise.allSettled(videos.map((video) => video.play()));
        status.textContent = results.some((item) => item.status === "rejected") ? "Browser blocked one video; click again." : "All videos playing";
      }});

      pauseButton.addEventListener("click", () => {{
        runSync(() => videos.forEach((video) => video.pause()));
        status.textContent = "All videos paused";
      }});

      videos.forEach((video) => {{
        video.addEventListener("play", () => {{
          runSync(() => videos.forEach((other) => {{ if (other !== video && other.paused) other.play().catch(() => {{}}); }}));
          status.textContent = "Synchronized playback";
        }});
        video.addEventListener("pause", () => {{
          runSync(() => videos.forEach((other) => {{ if (other !== video && !other.paused) other.pause(); }}));
          status.textContent = "Paused";
        }});
        video.addEventListener("seeked", () => {{
          runSync(() => alignTo(video));
          status.textContent = `Aligned at ${{video.currentTime.toFixed(2)}} s`;
        }});
        video.addEventListener("timeupdate", () => {{
          if (!syncSeek.checked || syncing) return;
          runSync(() => alignTo(video));
        }});
      }});
    }}

    document.querySelectorAll(".interactive-chart").forEach(renderInteractiveChart);
    setupSynchronizedVideos();
  </script>
</body>
</html>
"""

    args.output.write_text(html_text, encoding="utf-8")
    print(f"Wrote HTML report: {args.output}")


if __name__ == "__main__":
    main()
