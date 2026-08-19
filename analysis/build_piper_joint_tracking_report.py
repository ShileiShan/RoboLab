#!/usr/bin/env python3
"""Build a static HTML report for two Piper joint-tracking runs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np


BASELINE_COLOR = "#dc742d"
CANDIDATE_COLOR = "#2c82f6"
BASELINE_ALT = "#f1b37f"
CANDIDATE_ALT = "#89b9ff"
REFERENCE_COLOR = "#5b5b5b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--baseline-timeseries", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--candidate-timeseries", type=Path, required=True)
    parser.add_argument("--baseline-video", required=True,
                        help="Video path to embed in the report, typically relative to the HTML file.")
    parser.add_argument("--candidate-video", required=True,
                        help="Video path to embed in the report, typically relative to the HTML file.")
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--candidate-label", default="Candidate")
    parser.add_argument("--baseline-video-label",
                        help="Optional label for the left video card; defaults to --baseline-label.")
    parser.add_argument("--candidate-video-label",
                        help="Optional label for the right video card; defaults to --candidate-label.")
    parser.add_argument("--baseline-video-note",
                        help="Optional note shown on the left video card.")
    parser.add_argument("--candidate-video-note",
                        help="Optional note shown on the right video card.")
    parser.add_argument("--baseline-contact-sheet",
                        help="Optional left contact-sheet image path, typically relative to the HTML file.")
    parser.add_argument("--candidate-contact-sheet",
                        help="Optional right contact-sheet image path, typically relative to the HTML file.")
    parser.add_argument("--lead-sim-steps", type=int, default=0,
                        help="Shift simulated states earlier by this many timesteps for displayed metrics/charts.")
    parser.add_argument("--lead-sim-one-step", action="store_true",
                        help="Shift simulated states one timestep earlier for chart and displayed-metric alignment.")
    parser.add_argument("--left-arm-plot",
                        help="Optional left-arm qpos comparison image path, typically relative to the HTML file.")
    parser.add_argument("--right-arm-plot",
                        help="Optional right-arm qpos comparison image path, typically relative to the HTML file.")
    parser.add_argument("--gripper-plot",
                        help="Optional gripper qpos comparison image path, typically relative to the HTML file.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.lead_sim_one_step and args.lead_sim_steps == 0:
        args.lead_sim_steps = 1
    if args.lead_sim_steps < 0:
        parser.error("--lead-sim-steps must be non-negative.")
    return args


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_timeseries(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: np.asarray(handle[key]) for key in handle.files}


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _winner_text(baseline_value: float, candidate_value: float) -> str:
    if abs(baseline_value - candidate_value) <= 1.0e-12:
        return "Tie"
    return "Baseline" if baseline_value < candidate_value else "Candidate"


def _metric_row(label: str, key: str, baseline: dict, candidate: dict, digits: int = 6) -> str:
    b_value = float(baseline[key])
    c_value = float(candidate[key])
    delta = c_value - b_value
    delta_text = f"{delta:+.{digits}f}"
    return (
        f"<tr><th>{html.escape(label)}</th>"
        f"<td>{_fmt(b_value, digits)}</td>"
        f"<td>{_fmt(c_value, digits)}</td>"
        f"<td>{delta_text}</td>"
        f"<td>{_winner_text(b_value, c_value)}</td></tr>"
    )


def _render_chart(title: str, x_values: np.ndarray, series: list[dict], *, y_label: str) -> str:
    legend = []
    series_payload = []
    for item in series:
        legend.append(
            "<span class=\"legend-item\">"
            f"<span class=\"legend-swatch\" style=\"background:{item['color']}\"></span>"
            f"{html.escape(item['name'])}</span>"
        )
        series_payload.append({
            "name": item["name"],
            "color": item["color"],
            "dash": item.get("dash"),
            "values": np.asarray(item["values"], dtype=np.float64).tolist(),
        })

    payload = {
        "title": title,
        "y_label": y_label,
        "x_values": np.asarray(x_values, dtype=np.float64).tolist(),
        "series": series_payload,
    }
    payload_json = html.escape(json.dumps(payload, separators=(",", ":")))
    return f"""
    <div class="chart-card interactive-chart" data-chart="{payload_json}">
      <div class="chart-header">
        <h3>{html.escape(title)}</h3>
        <div class="chart-header-right">
          <span class="unit-badge">Unit: {html.escape(y_label)}</span>
          <div class="legend">{''.join(legend)}</div>
        </div>
      </div>
      <div class="chart-stage">
        <svg class="chart-svg" viewBox="0 0 920 280" role="img" aria-label="{html.escape(title)}"></svg>
        <div class="chart-tooltip" hidden></div>
      </div>
    </div>
    """


def _render_metadata(summary: dict, label: str) -> str:
    params = summary.get("actuator_params", {})
    gripper_min = summary.get("gripper_real_min")
    gripper_max = summary.get("gripper_real_max")
    mapping = "n/a"
    if gripper_min is not None and gripper_max is not None:
        mapping = (
            f"left [{gripper_min[0]:.6g}, {gripper_max[0]:.6g}] -> [0, 0.035] m; "
            f"right [{gripper_min[1]:.6g}, {gripper_max[1]:.6g}] -> [0, 0.035] m"
        )
    return f"""
    <table class="meta-table">
      <caption>{html.escape(label)}</caption>
      <tr><th>Trace</th><td>{html.escape(str(summary.get('trace_path')))}</td></tr>
      <tr><th>Source HDF5</th><td>{html.escape(str(summary.get('source_hdf5')))}</td></tr>
      <tr><th>Control Hz</th><td>{summary.get('control_hz')}</td></tr>
      <tr><th>Reference States</th><td>{summary.get('num_reference_states')}</td></tr>
      <tr><th>Stiffness</th><td>{params.get('stiffness')}</td></tr>
      <tr><th>Damping</th><td>{params.get('damping')}</td></tr>
      <tr><th>Armature</th><td>{params.get('armature')}</td></tr>
      <tr><th>Friction</th><td>{params.get('friction')}</td></tr>
      <tr><th>Gripper Mapping</th><td>{html.escape(mapping)}</td></tr>
    </table>
    """


def _lead_steps(values: np.ndarray, steps: int) -> np.ndarray:
    values = np.asarray(values)
    if steps <= 0 or values.shape[0] <= 1:
        return values.copy()
    steps = min(int(steps), values.shape[0] - 1)
    return np.concatenate((values[steps:], np.repeat(values[-1:], steps, axis=0)), axis=0)


def _compute_tracking_metrics(timeseries: dict[str, np.ndarray], *, lead_sim_steps: int) -> dict:
    reference_state = np.asarray(timeseries["reference_state"], dtype=np.float64)
    actual_state = np.asarray(timeseries["actual_state"], dtype=np.float64)
    if lead_sim_steps > 0:
        actual_state = _lead_steps(actual_state, lead_sim_steps)

    arm_reference = reference_state[:, 0:12]
    gripper_reference = reference_state[:, 12:14]
    arm_actual = actual_state[:, 0:12]
    gripper_actual = actual_state[:, 12:14]

    arm_error = arm_actual - arm_reference
    gripper_error = gripper_actual - gripper_reference
    arm_abs = np.abs(arm_error)
    gripper_abs = np.abs(gripper_error)

    return {
        "reference_state": reference_state,
        "actual_state": actual_state,
        "reference_gripper": gripper_reference,
        "actual_gripper": gripper_actual,
        "arm_error": arm_error,
        "gripper_error": gripper_error,
        "arm_mean_abs_error": np.mean(arm_abs, axis=1),
        "arm_max_abs_error": np.max(arm_abs, axis=1),
        "left_arm_mean_abs_error": np.mean(arm_abs[:, 0:6], axis=1),
        "right_arm_mean_abs_error": np.mean(arm_abs[:, 6:12], axis=1),
        "left_gripper_abs_error": gripper_abs[:, 0],
        "right_gripper_abs_error": gripper_abs[:, 1],
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


def _render_contact_sheets(args: argparse.Namespace) -> str:
    if not args.baseline_contact_sheet and not args.candidate_contact_sheet:
        return ""

    video_label_left = args.baseline_video_label or args.baseline_label
    video_label_right = args.candidate_video_label or args.candidate_label
    cards = []
    if args.baseline_contact_sheet:
        cards.append(
            f"""
            <a class="sheet-link" href="{html.escape(args.baseline_contact_sheet)}" target="_blank" rel="noopener noreferrer">
              <figure class="image-card">
                <figcaption>
                  <span>{html.escape(video_label_left)}</span>
                  <span class="subtle">Open full image</span>
                </figcaption>
                <img src="{html.escape(args.baseline_contact_sheet)}" alt="{html.escape(video_label_left)} contact sheet" loading="lazy" />
              </figure>
            </a>
            """
        )
    if args.candidate_contact_sheet:
        cards.append(
            f"""
            <a class="sheet-link" href="{html.escape(args.candidate_contact_sheet)}" target="_blank" rel="noopener noreferrer">
              <figure class="image-card">
                <figcaption>
                  <span>{html.escape(video_label_right)}</span>
                  <span class="subtle">Open full image</span>
                </figcaption>
                <img src="{html.escape(args.candidate_contact_sheet)}" alt="{html.escape(video_label_right)} contact sheet" loading="lazy" />
              </figure>
            </a>
            """
        )

    return f"""
      <details open>
        <summary>Overlay Contact Sheets</summary>
        <div class="sheet-grid">
          {''.join(cards)}
        </div>
      </details>
    """


def _render_image_gallery(args: argparse.Namespace) -> str:
    plots = [
        ("Left Arm Qpos Tracking", args.left_arm_plot),
        ("Right Arm Qpos Tracking", args.right_arm_plot),
        ("Gripper Qpos Tracking", args.gripper_plot),
    ]
    available = [(title, path) for title, path in plots if path]
    if not available:
        return ""

    figures = []
    for title, path in available:
        figures.append(
            f"""
            <figure class="image-card">
              <figcaption>{html.escape(title)}</figcaption>
              <img src="{html.escape(path)}" alt="{html.escape(title)}" loading="lazy" />
            </figure>
            """
        )

    return f"""
      <details open>
        <summary>Per-Joint Qpos Tracking</summary>
        <div class="image-grid">
          {''.join(figures)}
        </div>
      </details>
    """


def main() -> None:
    args = parse_args()
    baseline_summary = _load_summary(args.baseline_summary)
    candidate_summary = _load_summary(args.candidate_summary)
    baseline_ts = _load_timeseries(args.baseline_timeseries)
    candidate_ts = _load_timeseries(args.candidate_timeseries)
    baseline_display = _compute_tracking_metrics(baseline_ts, lead_sim_steps=args.lead_sim_steps)
    candidate_display = _compute_tracking_metrics(candidate_ts, lead_sim_steps=args.lead_sim_steps)
    shifted_note_html = ""
    if args.lead_sim_steps > 0:
        noun = "step" if args.lead_sim_steps == 1 else "steps"
        shifted_note_html = (
            '<p class="subtle">Displayed qpos/absolute-error curves shift simulated trajectories '
            f'{args.lead_sim_steps} control {noun} earlier to compensate for the observed lag.</p>'
        )

    common_steps = min(len(baseline_ts["time_s"]), len(candidate_ts["time_s"]))
    time_s = np.asarray(baseline_ts["time_s"][:common_steps], dtype=np.float64)
    script_block = """
  <script>
    const svgNs = "http://www.w3.org/2000/svg";

    function createSvg(tag, attrs = {}) {
      const node = document.createElementNS(svgNs, tag);
      for (const [key, value] of Object.entries(attrs)) {
        node.setAttribute(key, String(value));
      }
      return node;
    }

    function renderInteractiveChart(card) {
      const payload = JSON.parse(card.dataset.chart);
      const svg = card.querySelector(".chart-svg");
      const tooltip = card.querySelector(".chart-tooltip");
      const width = 920;
      const height = 280;
      const margin = { top: 18, right: 22, bottom: 42, left: 76 };
      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;
      const xValues = payload.x_values.slice();
      const ySeries = payload.series.map((item) => item.values.slice());
      const flat = ySeries.flat();
      let yMin = Math.min(...flat);
      let yMax = Math.max(...flat);
      if (!(yMax > yMin)) {
        yMax = yMin + 1;
      }
      const yPad = Math.max((yMax - yMin) * 0.08, 1e-6);
      yMin -= yPad;
      yMax += yPad;
      const xMin = xValues[0];
      const xMax = xValues[xValues.length - 1] > xMin ? xValues[xValues.length - 1] : xMin + 1;

      const mapX = (value) => margin.left + ((value - xMin) / (xMax - xMin)) * innerWidth;
      const mapY = (value) => height - margin.bottom - ((value - yMin) / (yMax - yMin)) * innerHeight;

      svg.innerHTML = "";
      svg.appendChild(createSvg("rect", {
        x: 0, y: 0, width, height, fill: "#ffffff", rx: 12, ry: 12,
      }));

      for (let idx = 0; idx < 5; idx += 1) {
        const fraction = idx / 4;
        const y = margin.top + fraction * innerHeight;
        const value = yMax - fraction * (yMax - yMin);
        svg.appendChild(createSvg("line", {
          x1: margin.left,
          y1: y,
          x2: width - margin.right,
          y2: y,
          class: "grid",
        }));
        const label = createSvg("text", {
          x: margin.left - 10,
          y: y + 4,
          class: "axis-label",
          "text-anchor": "end",
        });
        label.textContent = value.toFixed(4);
        svg.appendChild(label);
      }

      const xTicks = [
        { x: margin.left, value: xMin },
        { x: margin.left + innerWidth / 2, value: (xMin + xMax) / 2 },
        { x: width - margin.right, value: xMax },
      ];

      svg.appendChild(createSvg("line", {
        x1: margin.left,
        y1: height - margin.bottom,
        x2: width - margin.right,
        y2: height - margin.bottom,
        class: "axis",
      }));
      svg.appendChild(createSvg("line", {
        x1: margin.left,
        y1: margin.top,
        x2: margin.left,
        y2: height - margin.bottom,
        class: "axis",
      }));

      for (const tick of xTicks) {
        const label = createSvg("text", {
          x: tick.x,
          y: height - 12,
          class: "axis-label",
          "text-anchor": "middle",
        });
        label.textContent = `${tick.value.toFixed(2)}s`;
        svg.appendChild(label);
      }

      const xLabel = createSvg("text", {
        x: margin.left + innerWidth / 2,
        y: height - 24,
        class: "axis-label",
        "text-anchor": "middle",
      });
      xLabel.textContent = "Time";
      svg.appendChild(xLabel);

      const hoverLine = createSvg("line", {
        y1: margin.top,
        y2: height - margin.bottom,
        class: "hover-line",
        visibility: "hidden",
      });
      svg.appendChild(hoverLine);

      const circles = payload.series.map((item) => {
        const circle = createSvg("circle", {
          r: 4.5,
          fill: item.color,
          stroke: "#ffffff",
          "stroke-width": 1.5,
          visibility: "hidden",
        });
        svg.appendChild(circle);
        return circle;
      });

      for (const item of payload.series) {
        const points = item.values.map((value, index) => `${mapX(xValues[index]).toFixed(2)},${mapY(value).toFixed(2)}`).join(" ");
        const polyline = createSvg("polyline", {
          fill: "none",
          stroke: item.color,
          "stroke-width": 2.5,
          points,
        });
        if (item.dash) {
          polyline.setAttribute("stroke-dasharray", item.dash);
        }
        svg.appendChild(polyline);
      }

      const overlay = createSvg("rect", {
        x: margin.left,
        y: margin.top,
        width: innerWidth,
        height: innerHeight,
        fill: "transparent",
        class: "chart-overlay",
      });
      svg.appendChild(overlay);

      function updateHover(clientX) {
        const rect = svg.getBoundingClientRect();
        const x = ((clientX - rect.left) / rect.width) * width;
        const clamped = Math.min(width - margin.right, Math.max(margin.left, x));
        const index = Math.min(
          xValues.length - 1,
          Math.max(0, Math.round(((clamped - margin.left) / innerWidth) * (xValues.length - 1))),
        );
        const currentX = mapX(xValues[index]);
        hoverLine.setAttribute("x1", currentX);
        hoverLine.setAttribute("x2", currentX);
        hoverLine.setAttribute("visibility", "visible");

        const lines = [`<strong>${payload.title}</strong>`, `t = ${xValues[index].toFixed(3)} s`];
        payload.series.forEach((item, seriesIndex) => {
          const value = item.values[index];
          circles[seriesIndex].setAttribute("cx", currentX);
          circles[seriesIndex].setAttribute("cy", mapY(value));
          circles[seriesIndex].setAttribute("visibility", "visible");
          lines.push(
            `<span class="tooltip-row"><span class="tooltip-swatch" style="background:${item.color}"></span>${item.name}: ${value.toFixed(6)}</span>`,
          );
        });
        tooltip.innerHTML = lines.join("<br />");
        tooltip.hidden = false;

        const stageRect = card.querySelector(".chart-stage").getBoundingClientRect();
        const left = ((clientX - stageRect.left) / stageRect.width) * 100;
        tooltip.style.left = `${Math.min(84, Math.max(4, left + 1.5))}%`;
        tooltip.style.top = "10px";
      }

      overlay.addEventListener("mousemove", (event) => updateHover(event.clientX));
      overlay.addEventListener("mouseenter", (event) => updateHover(event.clientX));
      overlay.addEventListener("mouseleave", () => {
        tooltip.hidden = true;
        hoverLine.setAttribute("visibility", "hidden");
        circles.forEach((circle) => circle.setAttribute("visibility", "hidden"));
      });
    }

    function setupSynchronizedVideos() {
      const baseline = document.getElementById("baseline-video");
      const candidate = document.getElementById("candidate-video");
      const playButton = document.getElementById("play-both");
      const pauseButton = document.getElementById("pause-both");
      const syncSeek = document.getElementById("sync-seek");
      const status = document.getElementById("sync-status");
      const videos = [baseline, candidate];
      let syncing = false;

      function runSync(work) {
        if (syncing) return;
        syncing = true;
        try {
          work();
        } finally {
          window.requestAnimationFrame(() => { syncing = false; });
        }
      }

      function alignTo(source) {
        if (!syncSeek.checked) return;
        videos.forEach((video) => {
          if (video === source) return;
          if (Math.abs(video.currentTime - source.currentTime) > 0.03) {
            video.currentTime = source.currentTime;
          }
        });
      }

      playButton.addEventListener("click", async () => {
        const anchor = baseline.currentTime;
        runSync(() => {
          if (syncSeek.checked) {
            videos.forEach((video) => { video.currentTime = anchor; });
          }
        });
        const results = await Promise.allSettled(videos.map((video) => video.play()));
        status.textContent = results.some((item) => item.status === "rejected")
          ? "Browser blocked one video; click the button again."
          : "Both videos playing";
      });

      pauseButton.addEventListener("click", () => {
        runSync(() => {
          videos.forEach((video) => video.pause());
        });
        status.textContent = "Both videos paused";
      });

      videos.forEach((video, index) => {
        const label = index === 0 ? "Baseline" : "Candidate";
        video.addEventListener("play", () => {
          runSync(() => {
            alignTo(video);
            videos.forEach((other) => {
              if (other !== video && other.paused) {
                other.play().catch(() => {});
              }
            });
          });
          status.textContent = `${label} started; synchronized playback enabled`;
        });
        video.addEventListener("pause", () => {
          if (video.ended) return;
          runSync(() => {
            videos.forEach((other) => {
              if (other !== video && !other.paused) {
                other.pause();
              }
            });
          });
          status.textContent = "Both videos paused";
        });
        ["seeking", "timeupdate", "ratechange"].forEach((eventName) => {
          video.addEventListener(eventName, () => {
            runSync(() => {
              alignTo(video);
              videos.forEach((other) => {
                if (other !== video && other.playbackRate !== video.playbackRate) {
                  other.playbackRate = video.playbackRate;
                }
              });
            });
          });
        });
      });
    }

    document.addEventListener("DOMContentLoaded", () => {
      document.querySelectorAll(".interactive-chart").forEach(renderInteractiveChart);
      setupSynchronizedVideos();
    });
  </script>
"""

    report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Piper Joint Tracking Comparison</title>
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
    .shell {{
      max-width: 1500px;
      margin: 0 auto;
      display: grid;
      gap: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 12px 30px rgba(24, 18, 8, 0.07);
    }}
    .hero {{
      display: grid;
      gap: 8px;
    }}
    .subtle {{ color: var(--muted); }}
    .video-grid {{
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    }}
    .video-toolbar {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px 14px;
      margin-bottom: 16px;
      padding: 12px 14px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.72);
    }}
    .video-toolbar button {{
      appearance: none;
      border: 0;
      border-radius: 999px;
      padding: 9px 16px;
      font: inherit;
      font-weight: 600;
      color: #fffdf8;
      background: linear-gradient(135deg, #1f5ea8 0%, #15395f 100%);
      cursor: pointer;
    }}
    .video-toolbar button.secondary {{
      background: linear-gradient(135deg, #8c5d19 0%, #5d3a08 100%);
    }}
    .video-toolbar label {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
    }}
    figure {{ margin: 0; display: grid; gap: 10px; }}
    figcaption {{
      font-weight: 600;
      display: flex;
      justify-content: space-between;
      gap: 12px;
    }}
    video {{
      width: 100%;
      border-radius: 14px;
      background: #0a0a0a;
      border: 1px solid #cfc7b6;
    }}
    details {{
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px 14px;
      background: rgba(255, 255, 255, 0.7);
    }}
    details + details {{ margin-top: 12px; }}
    summary {{
      cursor: pointer;
      font-weight: 600;
      list-style: none;
    }}
    summary::-webkit-details-marker {{ display: none; }}
    .metric-table, .meta-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 14px;
    }}
    .metric-table th, .metric-table td, .meta-table th, .meta-table td {{
      border-bottom: 1px solid var(--border);
      text-align: left;
      padding: 10px 8px;
      vertical-align: top;
    }}
    .metric-table th {{ width: 28%; }}
    .meta-grid {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      margin-top: 12px;
    }}
    .meta-table caption {{
      text-align: left;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .chart-stack {{
      display: grid;
      gap: 16px;
      margin-top: 12px;
    }}
    .image-grid {{
      display: grid;
      gap: 16px;
      grid-template-columns: 1fr;
      margin-top: 12px;
    }}
    .sheet-grid {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      margin-top: 12px;
    }}
    .image-card {{
      margin: 0;
      display: grid;
      gap: 10px;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
      background: #fff;
    }}
    .image-card img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 12px;
      border: 1px solid #d8d0c1;
      background: #fbf8f2;
    }}
    .sheet-link {{
      text-decoration: none;
      color: inherit;
    }}
    .sheet-link:hover figcaption {{
      color: #234d7f;
    }}
    .chart-card {{
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
      background: #fff;
    }}
    .chart-header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
    }}
    .chart-header-right {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      align-items: center;
      gap: 8px 12px;
    }}
    .unit-badge {{
      display: inline-flex;
      align-items: center;
      padding: 5px 10px;
      border-radius: 999px;
      background: #f2ece1;
      color: #655b4b;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      font-size: 13px;
      color: var(--muted);
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .legend-swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
    }}
    .grid {{
      stroke: #e9e2d6;
      stroke-width: 1;
    }}
    .axis {{
      stroke: #948a78;
      stroke-width: 1.3;
    }}
    .axis-label {{
      fill: #5c574e;
      font-size: 12px;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    }}
    .chart-stage {{
      position: relative;
    }}
    .chart-svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .chart-overlay {{
      cursor: crosshair;
    }}
    .hover-line {{
      stroke: #413b32;
      stroke-width: 1.25;
      stroke-dasharray: 5 4;
    }}
    .chart-tooltip {{
      position: absolute;
      min-width: 190px;
      max-width: min(260px, 82%);
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid #cfc5b4;
      background: rgba(255, 253, 248, 0.96);
      box-shadow: 0 10px 26px rgba(26, 20, 10, 0.16);
      font-size: 13px;
      line-height: 1.45;
      pointer-events: none;
    }}
    .tooltip-row {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .tooltip-swatch {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
      flex: 0 0 auto;
    }}
    @media (max-width: 900px) {{
      body {{ padding: 14px; }}
      .panel {{ padding: 14px; }}
      .video-toolbar {{
        align-items: stretch;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="panel hero">
      <h1>Piper Joint Tracking Comparison</h1>
      <p class="subtle">
        Same real-robot joint reference, two actuator parameter sets. Metrics are lower-is-better because
        they measure sim-vs-reference tracking error.
      </p>
      {shifted_note_html}
    </section>

    <section class="panel">
      <div class="video-toolbar">
        <button id="play-both" type="button">Play Both Videos</button>
        <button id="pause-both" type="button" class="secondary">Pause Both Videos</button>
        <label><input id="sync-seek" type="checkbox" checked /> Keep timelines synchronized</label>
        <span id="sync-status" class="subtle">Ready</span>
      </div>
      <div class="video-grid">
        <figure>
          <figcaption>
            <span>{html.escape(args.baseline_video_label or args.baseline_label)}</span>
            <span class="subtle">{html.escape(args.baseline_video_note or f"arm RMSE {_fmt(float(baseline_display['arm_rmse']))}")}</span>
          </figcaption>
          <video id="baseline-video" controls preload="metadata" src="{html.escape(args.baseline_video)}"></video>
        </figure>
        <figure>
          <figcaption>
            <span>{html.escape(args.candidate_video_label or args.candidate_label)}</span>
            <span class="subtle">{html.escape(args.candidate_video_note or f"arm RMSE {_fmt(float(candidate_display['arm_rmse']))}")}</span>
          </figcaption>
          <video id="candidate-video" controls preload="metadata" src="{html.escape(args.candidate_video)}"></video>
        </figure>
      </div>

      <details open>
        <summary>Summary Metrics</summary>
        <table class="metric-table">
          <tr><th>Metric</th><th>{html.escape(args.baseline_label)}</th><th>{html.escape(args.candidate_label)}</th><th>Candidate - Baseline</th><th>Winner</th></tr>
          {_metric_row("Arm MAE", "arm_mae", baseline_display, candidate_display)}
          {_metric_row("Arm RMSE", "arm_rmse", baseline_display, candidate_display)}
          {_metric_row("Arm Max Abs", "arm_max_abs", baseline_display, candidate_display)}
          {_metric_row("Left Arm RMSE", "left_arm_rmse", baseline_display, candidate_display)}
          {_metric_row("Right Arm RMSE", "right_arm_rmse", baseline_display, candidate_display)}
          {_metric_row("Gripper MAE", "gripper_mae", baseline_display, candidate_display)}
          {_metric_row("Gripper RMSE", "gripper_rmse", baseline_display, candidate_display)}
          {_metric_row("Left Gripper RMSE", "left_gripper_rmse", baseline_display, candidate_display)}
          {_metric_row("Right Gripper RMSE", "right_gripper_rmse", baseline_display, candidate_display)}
        </table>
      </details>

      {_render_contact_sheets(args)}

      {_render_image_gallery(args)}

      <details>
        <summary>Error Curves</summary>
        <div class="chart-stack">
          {_render_chart(
            "Arm Mean Absolute Error",
            time_s,
            [
              {"name": args.baseline_label, "color": BASELINE_COLOR, "values": baseline_display["arm_mean_abs_error"][:common_steps]},
              {"name": args.candidate_label, "color": CANDIDATE_COLOR, "values": candidate_display["arm_mean_abs_error"][:common_steps]},
            ],
            y_label="rad",
          )}
          {_render_chart(
            "Left / Right Arm Mean Absolute Error",
            time_s,
            [
              {"name": f"{args.baseline_label} left", "color": BASELINE_COLOR, "values": baseline_display["left_arm_mean_abs_error"][:common_steps]},
              {"name": f"{args.baseline_label} right", "color": BASELINE_ALT, "values": baseline_display["right_arm_mean_abs_error"][:common_steps]},
              {"name": f"{args.candidate_label} left", "color": CANDIDATE_COLOR, "values": candidate_display["left_arm_mean_abs_error"][:common_steps]},
              {"name": f"{args.candidate_label} right", "color": CANDIDATE_ALT, "values": candidate_display["right_arm_mean_abs_error"][:common_steps]},
            ],
            y_label="rad",
          )}
          {_render_chart(
            "Left Gripper Absolute Error",
            time_s,
            [
              {"name": args.baseline_label, "color": BASELINE_COLOR, "values": baseline_display["left_gripper_abs_error"][:common_steps]},
              {"name": args.candidate_label, "color": CANDIDATE_COLOR, "values": candidate_display["left_gripper_abs_error"][:common_steps]},
            ],
            y_label="m",
          )}
          {_render_chart(
            "Right Gripper Absolute Error",
            time_s,
            [
              {"name": args.baseline_label, "color": BASELINE_COLOR, "values": baseline_display["right_gripper_abs_error"][:common_steps]},
              {"name": args.candidate_label, "color": CANDIDATE_COLOR, "values": candidate_display["right_gripper_abs_error"][:common_steps]},
            ],
            y_label="m",
          )}
          {_render_chart(
            "Left Gripper Tracking",
            time_s,
            [
              {"name": "Reference", "color": REFERENCE_COLOR, "values": baseline_display["reference_gripper"][:common_steps, 0], "dash": "6 4"},
              {"name": args.baseline_label, "color": BASELINE_COLOR, "values": baseline_display["actual_gripper"][:common_steps, 0]},
              {"name": args.candidate_label, "color": CANDIDATE_COLOR, "values": candidate_display["actual_gripper"][:common_steps, 0]},
            ],
            y_label="m",
          )}
          {_render_chart(
            "Right Gripper Tracking",
            time_s,
            [
              {"name": "Reference", "color": REFERENCE_COLOR, "values": baseline_display["reference_gripper"][:common_steps, 1], "dash": "6 4"},
              {"name": args.baseline_label, "color": BASELINE_COLOR, "values": baseline_display["actual_gripper"][:common_steps, 1]},
              {"name": args.candidate_label, "color": CANDIDATE_COLOR, "values": candidate_display["actual_gripper"][:common_steps, 1]},
            ],
            y_label="m",
          )}
        </div>
      </details>

      <details>
        <summary>Run Metadata</summary>
        <div class="meta-grid">
          {_render_metadata(baseline_summary, args.baseline_label)}
          {_render_metadata(candidate_summary, args.candidate_label)}
        </div>
      </details>
    </section>
  </main>
{script_block}
</body>
</html>
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_html, encoding="utf-8")
    print(f"Wrote HTML report: {args.output}")


if __name__ == "__main__":
    main()
