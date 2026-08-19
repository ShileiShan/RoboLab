#!/usr/bin/env python3
"""Build a static HTML report for real-state vs parameter comparison videos."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--baseline-overlay-video", type=Path, required=True)
    parser.add_argument("--candidate-overlay-video", type=Path, required=True)
    parser.add_argument("--baseline-contact-sheet", type=Path, required=True)
    parser.add_argument("--candidate-contact-sheet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-title", default="Real-State Proxy vs Baseline")
    parser.add_argument("--candidate-title", default="Real-State Proxy vs Candidate")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path, output: Path) -> str:
    return os.path.relpath(path.resolve(), output.parent.resolve())


def _fmt_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _metric_table(manifest: dict, title: str) -> str:
    rows = [
        ("Duration (s)", _fmt_float(float(manifest["common_duration_s"]))),
        ("FPS", _fmt_float(float(manifest["fps"]))),
        ("Frames", str(int(manifest["common_frames"]))),
        ("Initial RGB MAE", _fmt_float(float(manifest["initial_frame_rgb_mae"]))),
        ("Initial Robot Mask IoU", _fmt_float(float(manifest["initial_robot_mask_iou"]))),
        ("Initial Object Mask IoU", _fmt_float(float(manifest["initial_object_mask_iou"]))),
    ]
    body = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
        for label, value in rows
    )
    return f"""
    <section class="panel metric-panel">
      <h3>{html.escape(title)}</h3>
      <table class="metric-table">{body}</table>
    </section>
    """


def main() -> None:
    args = parse_args()
    baseline_manifest = _load_json(args.baseline_manifest)
    candidate_manifest = _load_json(args.candidate_manifest)

    baseline_overlay = _rel(args.baseline_overlay_video, args.output)
    candidate_overlay = _rel(args.candidate_overlay_video, args.output)
    baseline_sheet = _rel(args.baseline_contact_sheet, args.output)
    candidate_sheet = _rel(args.candidate_contact_sheet, args.output)

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Piper Real-State vs Params</title>
  <style>
    :root {{
      --bg: #f4f1eb;
      --panel: #fffdf8;
      --border: #d9d3c7;
      --text: #171717;
      --muted: #5f5a52;
      --blue: #2c82f6;
      --orange: #dc742d;
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
      max-width: 1600px;
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
    .subtle {{ color: var(--muted); }}
    .toolbar {{
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
    .toolbar button {{
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
    .toolbar button.secondary {{
      background: linear-gradient(135deg, #8c5d19 0%, #5d3a08 100%);
    }}
    .toolbar label {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
    }}
    .video-grid, .sheet-grid, .metric-grid {{
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    }}
    figure {{
      margin: 0;
      display: grid;
      gap: 12px;
    }}
    figcaption {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-weight: 700;
    }}
    video, img {{
      width: 100%;
      border-radius: 14px;
      border: 1px solid #d8d0c1;
      background: #0a0a0a;
      display: block;
    }}
    .sheet-link {{
      text-decoration: none;
      color: inherit;
    }}
    .sheet-link:hover figcaption {{
      color: #234d7f;
    }}
    .metric-panel h3 {{
      margin-bottom: 10px;
    }}
    .metric-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .metric-table th, .metric-table td {{
      border-bottom: 1px solid var(--border);
      text-align: left;
      padding: 10px 8px;
      vertical-align: top;
    }}
    .metric-table th {{
      width: 55%;
      color: var(--muted);
      font-weight: 600;
    }}
    @media (max-width: 900px) {{
      body {{ padding: 14px; }}
      .panel {{ padding: 14px; }}
      .toolbar {{ align-items: stretch; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="panel">
      <h1>Piper Real-State vs Parameter Comparison</h1>
      <p class="subtle">
        Blue/Orange overlay videos compare the strict joint-state replay against the baseline and candidate
        joint-target replays. Contact sheets sample the full sequence at matched timestamps.
      </p>
    </section>

    <section class="panel">
      <div class="toolbar">
        <button id="play-both" type="button">Play Both Overlays</button>
        <button id="pause-both" type="button" class="secondary">Pause Both Overlays</button>
        <label><input id="sync-seek" type="checkbox" checked /> Keep timelines synchronized</label>
        <span id="sync-status" class="subtle">Ready</span>
      </div>
      <div class="video-grid">
        <figure>
          <figcaption>
            <span>{html.escape(args.baseline_title)}</span>
            <span class="subtle">blue = real-state, orange = baseline</span>
          </figcaption>
          <video id="baseline-video" controls preload="metadata" src="{html.escape(baseline_overlay)}"></video>
        </figure>
        <figure>
          <figcaption>
            <span>{html.escape(args.candidate_title)}</span>
            <span class="subtle">blue = real-state, orange = candidate</span>
          </figcaption>
          <video id="candidate-video" controls preload="metadata" src="{html.escape(candidate_overlay)}"></video>
        </figure>
      </div>
    </section>

    <section class="panel">
      <h2>Contact Sheets</h2>
      <div class="sheet-grid">
        <a class="sheet-link" href="{html.escape(baseline_sheet)}" target="_blank" rel="noopener noreferrer">
          <figure>
            <figcaption>
              <span>{html.escape(args.baseline_title)}</span>
              <span class="subtle">Open full image</span>
            </figcaption>
            <img src="{html.escape(baseline_sheet)}" alt="{html.escape(args.baseline_title)} contact sheet" />
          </figure>
        </a>
        <a class="sheet-link" href="{html.escape(candidate_sheet)}" target="_blank" rel="noopener noreferrer">
          <figure>
            <figcaption>
              <span>{html.escape(args.candidate_title)}</span>
              <span class="subtle">Open full image</span>
            </figcaption>
            <img src="{html.escape(candidate_sheet)}" alt="{html.escape(args.candidate_title)} contact sheet" />
          </figure>
        </a>
      </div>
    </section>

    <section class="panel">
      <h2>Comparison Metadata</h2>
      <div class="metric-grid">
        {_metric_table(baseline_manifest, args.baseline_title)}
        {_metric_table(candidate_manifest, args.candidate_title)}
      </div>
    </section>
  </main>

  <script>
    const baseline = document.getElementById("baseline-video");
    const candidate = document.getElementById("candidate-video");
    const playButton = document.getElementById("play-both");
    const pauseButton = document.getElementById("pause-both");
    const syncSeek = document.getElementById("sync-seek");
    const status = document.getElementById("sync-status");
    const videos = [baseline, candidate];
    let syncing = false;

    function runSync(work) {{
      if (syncing) return;
      syncing = true;
      try {{
        work();
      }} finally {{
        window.requestAnimationFrame(() => {{ syncing = false; }});
      }}
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
      const anchor = baseline.currentTime;
      runSync(() => {{
        if (syncSeek.checked) {{
          videos.forEach((video) => {{ video.currentTime = anchor; }});
        }}
      }});
      const results = await Promise.allSettled(videos.map((video) => video.play()));
      status.textContent = results.some((item) => item.status === "rejected")
        ? "Browser blocked one video; click the button again."
        : "Both overlay videos playing";
    }});

    pauseButton.addEventListener("click", () => {{
      runSync(() => {{
        videos.forEach((video) => video.pause());
      }});
      status.textContent = "Both overlay videos paused";
    }});

    videos.forEach((video, index) => {{
      const label = index === 0 ? "Baseline comparison" : "Candidate comparison";
      video.addEventListener("play", () => {{
        runSync(() => {{
          alignTo(video);
          videos.forEach((other) => {{
            if (other !== video && other.paused) {{
              other.play().catch(() => {{}});
            }}
          }});
        }});
        status.textContent = `${{label}} started; synchronized playback enabled`;
      }});
      video.addEventListener("pause", () => {{
        if (video.ended) return;
        runSync(() => {{
          videos.forEach((other) => {{
            if (other !== video && !other.paused) {{
              other.pause();
            }}
          }});
        }});
        status.textContent = "Both overlay videos paused";
      }});
      ["seeking", "timeupdate", "ratechange"].forEach((eventName) => {{
        video.addEventListener(eventName, () => {{
          runSync(() => {{
            alignTo(video);
            videos.forEach((other) => {{
              if (other !== video && other.playbackRate !== video.playbackRate) {{
                other.playbackRate = video.playbackRate;
              }}
            }});
          }});
        }});
      }});
    }});
  </script>
</body>
</html>
"""
    args.output.write_text(html_text, encoding="utf-8")
    print(f"Wrote HTML report: {args.output}")


if __name__ == "__main__":
    main()
