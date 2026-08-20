#!/usr/bin/env bash
# Run the same Piper task twice with different arm-actuator values, then create
# a synchronized multi-frame trajectory comparison from the viewport videos.

set -euo pipefail

readonly DEFAULT_CONTAINER="robolab_lh"
readonly DEFAULT_TASK="PiperSingleBananaPickPlaceTask"
readonly DEFAULT_REMOTE_HOST="127.0.0.1"
readonly DEFAULT_REMOTE_PORT=18002
readonly DEFAULT_SECONDS=20
readonly DEFAULT_FRAMES=8
readonly DEFAULT_POLICY="pi05"
readonly DEFAULT_DEVICE="cuda:0"

# Values before the current Piper tuning landed.
baseline_kp=400
baseline_kd=80

# Checked-in Piper tuning at the time this launcher was added.
candidate_kp=867.0147
candidate_kd=127.7443

container="$DEFAULT_CONTAINER"
task="$DEFAULT_TASK"
remote_host="$DEFAULT_REMOTE_HOST"
remote_port="$DEFAULT_REMOTE_PORT"
seconds="$DEFAULT_SECONDS"
frames="$DEFAULT_FRAMES"
policy="$DEFAULT_POLICY"
device="$DEFAULT_DEVICE"
layout="side-by-side"
dry_run=false

usage() {
  cat <<'EOF'
Usage: scripts/compare_piper_actuator_trajectories.sh [options]

Run an identical Piper policy evaluation for a baseline and a candidate actuator
configuration, then write a synchronized viewport contact sheet under output/.

Actuator options (applied identically to the left and right arm):
  --baseline-kp VALUE           Baseline stiffness / Kp (default: 400)
  --baseline-kd VALUE           Baseline damping / Kd (default: 80)
  --candidate-kp VALUE          Candidate stiffness / Kp (default: 867.0147)
  --candidate-kd VALUE          Candidate damping / Kd (default: 127.7443)

Evaluation and output options:
  --frames N                    Number of equally spaced timestamps (default: 8)
  --seconds S                   Episode duration in simulated seconds (default: 20)
  --task NAME                   Piper task class (default: PiperSingleBananaPickPlaceTask)
  --container NAME              Running Docker container (default: robolab_lh)
  --remote-host HOST            RTC policy-server host (default: 127.0.0.1)
  --remote-port PORT            RTC policy-server port (default: 18002)
  --policy NAME                 Pi0-family policy variant (default: pi05)
  --device DEVICE               Physical Isaac GPU, e.g. cuda:1 (default: cuda:0).
                                Passed to both physics and the Kit renderer.
  --layout MODE                 side-by-side (default) or overlay
  --dry-run                     Validate and print the comparison configuration only
  -h, --help                    Show this help text

The tool writes t=0 before any control command.  The first tile is therefore
the initial-pose check.  In --layout overlay mode, use robot-mask videos with
the Python visualizer for true blue/orange robot-only overlay; this launcher
uses the robust side-by-side view by default.
EOF
}

need_value() {
  (($# >= 2)) || { echo "Missing value for $1." >&2; exit 2; }
}

while (($#)); do
  case "$1" in
    --baseline-kp) need_value "$@"; baseline_kp="$2"; shift 2 ;;
    --baseline-kd) need_value "$@"; baseline_kd="$2"; shift 2 ;;
    --candidate-kp) need_value "$@"; candidate_kp="$2"; shift 2 ;;
    --candidate-kd) need_value "$@"; candidate_kd="$2"; shift 2 ;;
    --frames) need_value "$@"; frames="$2"; shift 2 ;;
    --seconds) need_value "$@"; seconds="$2"; shift 2 ;;
    --task) need_value "$@"; task="$2"; shift 2 ;;
    --container) need_value "$@"; container="$2"; shift 2 ;;
    --remote-host) need_value "$@"; remote_host="$2"; shift 2 ;;
    --remote-port) need_value "$@"; remote_port="$2"; shift 2 ;;
    --policy) need_value "$@"; policy="$2"; shift 2 ;;
    --device) need_value "$@"; device="$2"; shift 2 ;;
    --layout) need_value "$@"; layout="$2"; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

is_nonnegative_number() {
  [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] && awk -v value="$1" 'BEGIN { exit !(value >= 0) }'
}

for value in "$baseline_kp" "$baseline_kd" "$candidate_kp" "$candidate_kd"; do
  is_nonnegative_number "$value" || { echo "Actuator values must be non-negative numbers: $value" >&2; exit 2; }
done
[[ "$frames" =~ ^[1-9][0-9]*$ ]] || { echo "--frames must be a positive integer." >&2; exit 2; }
is_nonnegative_number "$seconds" && awk -v value="$seconds" 'BEGIN { exit !(value > 0) }' || {
  echo "--seconds must be a positive number." >&2; exit 2;
}
[[ "$remote_port" =~ ^[1-9][0-9]*$ ]] && ((remote_port <= 65535)) || {
  echo "--remote-port must be an integer in 1..65535." >&2; exit 2;
}
[[ "$device" =~ ^cuda:([0-9]+)$ ]] || {
  echo "--device must name a physical CUDA device such as cuda:0." >&2; exit 2;
}
gpu_index="${device#cuda:}"
[[ "$layout" == "side-by-side" || "$layout" == "overlay" ]] || {
  echo "--layout must be side-by-side or overlay." >&2; exit 2;
}

timestamp="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
comparison_id="piper_actuator_comparison_${task}_${timestamp}"
output_root="$(pwd)/output"
comparison_dir="${output_root}/${comparison_id}"
baseline_run_id="${comparison_id}_baseline"
candidate_run_id="${comparison_id}_candidate"

print_config() {
  echo "[RoboLab] Piper actuator trajectory comparison"
  echo "[RoboLab] task=${task}, seconds=${seconds}, frames=${frames}, layout=${layout}"
  echo "[RoboLab] Isaac physics and renderer device=${device}"
  echo "[RoboLab] baseline:  kp=${baseline_kp}, kd=${baseline_kd}"
  echo "[RoboLab] candidate: kp=${candidate_kp}, kd=${candidate_kd}"
  echo "[RoboLab] output: ${comparison_dir}"
}

print_config
if "$dry_run"; then
  echo "[RoboLab] Dry run complete; no evaluation was started."
  exit 0
fi

docker inspect --format '{{.State.Running}}' "$container" 2>/dev/null | grep -qx true || {
  echo "Docker container '$container' is not running." >&2
  exit 1
}

mkdir -p "$comparison_dir"

launch_run() {
  local label="$1"
  local run_id="$2"
  local kp="$3"
  local kd="$4"
  local log_file="${comparison_dir}/${label}.log"

  echo "[RoboLab] Starting ${label} evaluation; log: ${log_file}"
  docker exec -i \
    -e "ROBOLAB_PIPER_ARM_KP=${kp}" \
    -e "ROBOLAB_PIPER_ARM_KD=${kd}" \
    -e "ROBOLAB_EPISODE_LENGTH_S=${seconds}" \
    -e "ROBOLAB_WRITE_INITIAL_VIDEO_FRAME=1" \
    "$container" \
    bash -lc '
      set -euo pipefail
      cd /workspace/robolab
      /workspace/isaaclab/isaaclab.sh -p policies/pi0_family/run_piper.py \
        --rtc --remote-host "$1" --remote-port "$2" --policy "$3" \
        --task "$4" --num-envs 1 --num-runs 1 --headless --device "$5" \
        --kit-args "--/renderer/activeGpu=$7 --/physics/cudaDevice=$7 --/renderer/multiGpu/enabled=false" \
        --video-mode viewport --output-folder-name "$6"
    ' bash "$remote_host" "$remote_port" "$policy" "$task" "$device" "$run_id" "$gpu_index" \
    >"$log_file" 2>&1
}

launch_run baseline "$baseline_run_id" "$baseline_kp" "$baseline_kd"
launch_run candidate "$candidate_run_id" "$candidate_kp" "$candidate_kd"

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
comparison_png="${comparison_dir}/trajectory_comparison.png"

python3 analysis/visualize_piper_trajectory_comparison.py \
  --baseline-video "$baseline_video" \
  --candidate-video "$candidate_video" \
  --frames "$frames" \
  --layout "$layout" \
  --baseline-label "Baseline (kp=${baseline_kp}, kd=${baseline_kd})" \
  --candidate-label "Candidate (kp=${candidate_kp}, kd=${candidate_kd})" \
  --output "$comparison_png"

echo "[RoboLab] Baseline video: ${baseline_video}"
echo "[RoboLab] Candidate video: ${candidate_video}"
echo "[RoboLab] Contact sheet: ${comparison_png}"
