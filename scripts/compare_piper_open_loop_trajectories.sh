#!/usr/bin/env bash
# Replay one fixed Piper action trace under two actuator configurations and
# create a synchronized viewport contact sheet.  No policy server is used.

set -euo pipefail

readonly DEFAULT_CONTAINER="robolab_lh"
readonly DEFAULT_TASK="MakeBreakfastTask"
readonly DEFAULT_DEVICE="cuda:0"
readonly DEFAULT_FRAMES=8

baseline_friction=0
baseline_armature=0
baseline_kp=400
baseline_kd=80
candidate_friction=0.3516
candidate_armature=0.4855
candidate_kp=867.0147
candidate_kd=127.7443

container="$DEFAULT_CONTAINER"
task="$DEFAULT_TASK"
device="$DEFAULT_DEVICE"
frames="$DEFAULT_FRAMES"
layout="overlay"
action_trace=""
include_breakfast_objects=false
publish_dir=""
dry_run=false

usage() {
  cat <<'EOF'
Usage: scripts/compare_piper_open_loop_trajectories.sh --action-trace TRACE.npz [options]

Replay exactly the same 14-D Piper action trace for a baseline and candidate
actuator configuration.  This is an open-loop physical comparison: it never
contacts a policy/RTC server.

Required:
  --action-trace TRACE.npz       Trace created by record_piper_action_trace.sh.

Actuator options:
  --baseline-friction VALUE      Baseline joint friction (default: 0)
  --baseline-armature VALUE      Baseline armature (default: 0)
  --baseline-kp VALUE            Baseline stiffness / Kp (default: 400)
  --baseline-kd VALUE            Baseline damping / Kd (default: 80)
  --candidate-friction VALUE     Candidate joint friction (default: 0.3516)
  --candidate-armature VALUE     Candidate armature (default: 0.4855)
  --candidate-kp VALUE           Candidate stiffness / Kp (default: 867.0147)
  --candidate-kd VALUE           Candidate damping / Kd (default: 127.7443)

Other options:
  --frames N                     Number of synchronized contact-sheet times (default: 8)
  --task NAME                    Piper-compatible task class (default: MakeBreakfastTask)
  --device DEVICE                Physical Isaac GPU, e.g. cuda:1 (default: cuda:0)
  --container NAME               Docker container (default: robolab_lh)
  --layout MODE                  overlay (default) or side-by-side
  --include-breakfast-objects    Include the two bread slices and toaster in
                                 the blue/orange overlay (MakeBreakfastTask only)
  --publish-dir DIR              Write only the final overlay MP4, contact sheet,
                                 background, manifest, and individual replay videos
                                 to DIR.  Raw replay files remain under output/
                                 (default: this run's directory).
  --dry-run                      Validate and print configuration only
  -h, --help                     Show this help text
EOF
}

need_value() { (($# >= 2)) || { echo "Missing value for $1." >&2; exit 2; }; }

while (($#)); do
  case "$1" in
    --action-trace) need_value "$@"; action_trace="$2"; shift 2 ;;
    --baseline-friction) need_value "$@"; baseline_friction="$2"; shift 2 ;;
    --baseline-armature) need_value "$@"; baseline_armature="$2"; shift 2 ;;
    --baseline-kp) need_value "$@"; baseline_kp="$2"; shift 2 ;;
    --baseline-kd) need_value "$@"; baseline_kd="$2"; shift 2 ;;
    --candidate-friction) need_value "$@"; candidate_friction="$2"; shift 2 ;;
    --candidate-armature) need_value "$@"; candidate_armature="$2"; shift 2 ;;
    --candidate-kp) need_value "$@"; candidate_kp="$2"; shift 2 ;;
    --candidate-kd) need_value "$@"; candidate_kd="$2"; shift 2 ;;
    --frames) need_value "$@"; frames="$2"; shift 2 ;;
    --task) need_value "$@"; task="$2"; shift 2 ;;
    --device) need_value "$@"; device="$2"; shift 2 ;;
    --container) need_value "$@"; container="$2"; shift 2 ;;
    --layout) need_value "$@"; layout="$2"; shift 2 ;;
    --include-breakfast-objects) include_breakfast_objects=true; shift ;;
    --publish-dir) need_value "$@"; publish_dir="$2"; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

is_nonnegative_number() {
  [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] && awk -v value="$1" 'BEGIN { exit !(value >= 0) }'
}

[[ -n "$action_trace" ]] || { echo "--action-trace is required." >&2; exit 2; }
[[ "$frames" =~ ^[1-9][0-9]*$ ]] || { echo "--frames must be a positive integer." >&2; exit 2; }
[[ "$device" =~ ^cuda:([0-9]+)$ ]] || { echo "--device must be like cuda:0." >&2; exit 2; }
[[ "$layout" == "side-by-side" || "$layout" == "overlay" ]] || {
  echo "--layout must be side-by-side or overlay." >&2; exit 2;
}
comparison_breakfast_env=0
if "$include_breakfast_objects"; then
  [[ "$task" == "MakeBreakfastTask" ]] || {
    echo "--include-breakfast-objects is supported only with --task MakeBreakfastTask." >&2; exit 2;
  }
  comparison_breakfast_env=1
fi
for value in "$baseline_friction" "$baseline_armature" "$baseline_kp" "$baseline_kd" \
             "$candidate_friction" "$candidate_armature" "$candidate_kp" "$candidate_kd"; do
  is_nonnegative_number "$value" || { echo "Actuator values must be non-negative numbers: $value" >&2; exit 2; }
done

repo_root="$(pwd -P)"
trace_host_path="$(realpath -m "$action_trace")"
case "$trace_host_path" in
  "$repo_root"/*) trace_container_path="/workspace/robolab/${trace_host_path#"$repo_root"/}" ;;
  *) echo "--action-trace must be inside this workspace: $repo_root" >&2; exit 2 ;;
esac
[[ -f "$trace_host_path" ]] || { echo "Action trace does not exist: $trace_host_path" >&2; exit 2; }

gpu_index="${device#cuda:}"
timestamp="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
comparison_id="piper_open_loop_${task}_${timestamp}"
output_root="${repo_root}/output"
comparison_dir="${output_root}/${comparison_id}"
baseline_run_id="${comparison_id}_baseline"
candidate_run_id="${comparison_id}_candidate"
presentation_dir="${comparison_dir}"
if [[ -n "$publish_dir" ]]; then
  presentation_dir="$(realpath -m "$publish_dir")"
fi

echo "[RoboLab] Piper open-loop actuator comparison"
echo "[RoboLab] trace=${trace_host_path}; task=${task}; device=${device}; frames=${frames}"
echo "[RoboLab] baseline:  friction=${baseline_friction}, armature=${baseline_armature}, kp=${baseline_kp}, kd=${baseline_kd}"
echo "[RoboLab] candidate: friction=${candidate_friction}, armature=${candidate_armature}, kp=${candidate_kp}, kd=${candidate_kd}"
echo "[RoboLab] output: ${comparison_dir}"
echo "[RoboLab] presentation output: ${presentation_dir}"
echo "[RoboLab] include breakfast objects: ${include_breakfast_objects}"
if "$dry_run"; then
  echo "[RoboLab] Dry run complete; no replay was started."
  exit 0
fi

docker inspect --format '{{.State.Running}}' "$container" 2>/dev/null | grep -qx true || {
  echo "Docker container '$container' is not running." >&2; exit 1;
}
mkdir -p "$comparison_dir"
mkdir -p "$presentation_dir"

launch_replay() {
  local label="$1" run_id="$2" friction="$3" armature="$4" kp="$5" kd="$6"
  local log_file="${comparison_dir}/${label}.log"
  echo "[RoboLab] Starting ${label} replay; log: ${log_file}"
  docker exec -i \
    -e "ROBOLAB_PIPER_ARM_FRICTION=${friction}" \
    -e "ROBOLAB_PIPER_ARM_ARMATURE=${armature}" \
    -e "ROBOLAB_PIPER_ARM_KP=${kp}" \
    -e "ROBOLAB_PIPER_ARM_KD=${kd}" \
    -e "ROBOLAB_WRITE_INITIAL_VIDEO_FRAME=1" \
    -e "ROBOLAB_COMPARISON_INCLUDE_BREAKFAST_OBJECTS=${comparison_breakfast_env}" \
    "$container" \
    bash -lc '
      set -euo pipefail
      cd /workspace/robolab
      /workspace/isaaclab/isaaclab.sh -p policies/pi0_family/run_piper_action_replay.py \
        --action-trace "$1" --task "$2" --num-envs 1 --num-runs 1 --headless --device "$3" \
        --kit-args "--/renderer/activeGpu=$5 --/physics/cudaDevice=$5 --/renderer/multiGpu/enabled=false" \
        --video-mode viewport --output-folder-name "$4"
    ' bash "$trace_container_path" "$task" "$device" "$run_id" "$gpu_index" \
    >"$log_file" 2>&1
}

launch_replay baseline "$baseline_run_id" "$baseline_friction" "$baseline_armature" "$baseline_kp" "$baseline_kd"
launch_replay candidate "$candidate_run_id" "$candidate_friction" "$candidate_armature" "$candidate_kp" "$candidate_kd"

find_single_viewport_video() {
  local run_id="$1"
  mapfile -t videos < <(find "${output_root}/${run_id}" -type f -name '*_viewport.mp4' | sort)
  if ((${#videos[@]} != 1)); then
    echo "Expected exactly one viewport video for ${run_id}, found ${#videos[@]}." >&2
    printf '  %s\n' "${videos[@]:-}" >&2
    exit 1
  fi
  printf '%s' "${videos[0]}"
}

baseline_video="$(find_single_viewport_video "$baseline_run_id")"
candidate_video="$(find_single_viewport_video "$candidate_run_id")"
comparison_png="${presentation_dir}/trajectory_comparison.png"
comparison_video="${presentation_dir}/trajectory_overlay.mp4"
static_background="${presentation_dir}/static_background.png"
baseline_presentation_video="${presentation_dir}/baseline_replay.mp4"
candidate_presentation_video="${presentation_dir}/candidate_replay.mp4"

find_single_viewport_mask() {
  local run_id="$1"
  mapfile -t masks < <(find "${output_root}/${run_id}" -type f -name '*_viewport_robot_mask.mp4' | sort)
  if ((${#masks[@]} != 1)); then
    echo "Expected exactly one robot mask video for ${run_id}, found ${#masks[@]}." >&2
    printf '  %s\n' "${masks[@]:-}" >&2
    exit 1
  fi
  printf '%s' "${masks[0]}"
}

baseline_mask="$(find_single_viewport_mask "$baseline_run_id")"
candidate_mask="$(find_single_viewport_mask "$candidate_run_id")"

breakfast_mask_args=()
if "$include_breakfast_objects"; then
  find_single_breakfast_objects_mask() {
    local run_id="$1"
    mapfile -t masks < <(find "${output_root}/${run_id}" -type f -name '*_viewport_breakfast_objects_mask.mp4' | sort)
    if ((${#masks[@]} != 1)); then
      echo "Expected exactly one breakfast-object mask video for ${run_id}, found ${#masks[@]}." >&2
      printf '  %s\n' "${masks[@]:-}" >&2
      exit 1
    fi
    printf '%s' "${masks[0]}"
  }
  baseline_breakfast_mask="$(find_single_breakfast_objects_mask "$baseline_run_id")"
  candidate_breakfast_mask="$(find_single_breakfast_objects_mask "$candidate_run_id")"
  breakfast_mask_args=(
    --baseline-breakfast-objects-mask-video "$baseline_breakfast_mask"
    --candidate-breakfast-objects-mask-video "$candidate_breakfast_mask"
  )
fi

python3 analysis/visualize_piper_trajectory_comparison.py \
  --baseline-video "$baseline_video" --candidate-video "$candidate_video" \
  --baseline-mask-video "$baseline_mask" --candidate-mask-video "$candidate_mask" \
  "${breakfast_mask_args[@]}" \
  --frames "$frames" --layout "$layout" \
  --overlay-video "$comparison_video" --background-output "$static_background" \
  --baseline-label "Baseline (f=${baseline_friction}, a=${baseline_armature}, kp=${baseline_kp}, kd=${baseline_kd})" \
  --candidate-label "Candidate (f=${candidate_friction}, a=${candidate_armature}, kp=${candidate_kp}, kd=${candidate_kd})" \
  --output "$comparison_png"

publish_browser_video() {
  local source="$1" destination="$2" label="$3"
  if command -v ffmpeg >/dev/null 2>&1; then
    # Source viewport MP4s use OpenCV's mp4v codec, which many browsers report
    # as a zero-second video.  Publish H.264/yuv420p copies beside the overlay.
    ffmpeg -y -v error -i "$source" \
      -map 0:v:0 -an -c:v libx264 -preset medium -crf 20 \
      -pix_fmt yuv420p -movflags +faststart "$destination"
  else
    cp "$source" "$destination"
    echo "[RoboLab] Warning: ffmpeg unavailable; ${label} may not play in a browser." >&2
  fi
}

publish_browser_video "$baseline_video" "$baseline_presentation_video" "baseline replay"
publish_browser_video "$candidate_video" "$candidate_presentation_video" "candidate replay"

echo "[RoboLab] Baseline video: ${baseline_video}"
echo "[RoboLab] Candidate video: ${candidate_video}"
echo "[RoboLab] Baseline robot mask: ${baseline_mask}"
echo "[RoboLab] Candidate robot mask: ${candidate_mask}"
if "$include_breakfast_objects"; then
  echo "[RoboLab] Baseline breakfast-object mask: ${baseline_breakfast_mask}"
  echo "[RoboLab] Candidate breakfast-object mask: ${candidate_breakfast_mask}"
fi
echo "[RoboLab] Overlay video: ${comparison_video}"
echo "[RoboLab] Baseline replay video: ${baseline_presentation_video}"
echo "[RoboLab] Candidate replay video: ${candidate_presentation_video}"
echo "[RoboLab] Static background: ${static_background}"
echo "[RoboLab] Contact sheet: ${comparison_png}"
