#!/usr/bin/env bash
# Compare two Piper actuator parameter sets on the same real-robot joint trace.

set -euo pipefail

readonly DEFAULT_CONTAINER="robolab_lh"
readonly DEFAULT_TASK="MakeBreakfastTask"
readonly DEFAULT_DEVICE="cuda:0"

baseline_friction=0.10
baseline_armature=0.05
baseline_kp=400
baseline_kd=80
candidate_friction=0.3516
candidate_armature=0.4855
candidate_kp=867.0147
candidate_kd=127.7443

container="$DEFAULT_CONTAINER"
task="$DEFAULT_TASK"
device="$DEFAULT_DEVICE"
publish_dir=""
dry_run=false
real_hdf5=""
lead_sim_steps=2

usage() {
  cat <<'EOF'
Usage: scripts/compare_piper_joint_tracking_params.sh --real-hdf5 EPISODE.hdf5 [options]

Replay one real-robot Piper joint trajectory twice in RoboLab, once with the
baseline actuator parameters and once with the candidate parameters. The
resulting report contains side-by-side videos plus collapsible tracking metrics
and error curves.

Required:
  --real-hdf5 FILE              Raw real-robot HDF5 episode with observations/qpos.

Actuator options:
  --baseline-friction VALUE     Baseline joint friction (default: 0.10)
  --baseline-armature VALUE     Baseline armature (default: 0.05)
  --baseline-kp VALUE           Baseline stiffness / Kp (default: 400)
  --baseline-kd VALUE           Baseline damping / Kd (default: 80)
  --candidate-friction VALUE    Candidate joint friction (default: 0.3516)
  --candidate-armature VALUE    Candidate armature (default: 0.4855)
  --candidate-kp VALUE          Candidate stiffness / Kp (default: 867.0147)
  --candidate-kd VALUE          Candidate damping / Kd (default: 127.7443)

Other options:
  --task NAME                   Piper-compatible task class (default: MakeBreakfastTask)
  --device DEVICE               Physical Isaac GPU, e.g. cuda:1 (default: cuda:0)
  --container NAME              Docker container (default: robolab_lh)
  --publish-dir DIR             Copy final report artifacts to DIR (default: comparison run dir)
  --lead-sim-steps N            Shift displayed sim curves earlier by N timesteps (default: 2)
  --dry-run                     Validate and print configuration only
  -h, --help                    Show this help text
EOF
}

need_value() { (($# >= 2)) || { echo "Missing value for $1." >&2; exit 2; }; }

while (($#)); do
  case "$1" in
    --real-hdf5) need_value "$@"; real_hdf5="$2"; shift 2 ;;
    --baseline-friction) need_value "$@"; baseline_friction="$2"; shift 2 ;;
    --baseline-armature) need_value "$@"; baseline_armature="$2"; shift 2 ;;
    --baseline-kp) need_value "$@"; baseline_kp="$2"; shift 2 ;;
    --baseline-kd) need_value "$@"; baseline_kd="$2"; shift 2 ;;
    --candidate-friction) need_value "$@"; candidate_friction="$2"; shift 2 ;;
    --candidate-armature) need_value "$@"; candidate_armature="$2"; shift 2 ;;
    --candidate-kp) need_value "$@"; candidate_kp="$2"; shift 2 ;;
    --candidate-kd) need_value "$@"; candidate_kd="$2"; shift 2 ;;
    --task) need_value "$@"; task="$2"; shift 2 ;;
    --device) need_value "$@"; device="$2"; shift 2 ;;
    --container) need_value "$@"; container="$2"; shift 2 ;;
    --publish-dir) need_value "$@"; publish_dir="$2"; shift 2 ;;
    --lead-sim-steps) need_value "$@"; lead_sim_steps="$2"; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

is_nonnegative_number() {
  [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] && awk -v value="$1" 'BEGIN { exit !(value >= 0) }'
}

[[ -n "$real_hdf5" ]] || { echo "--real-hdf5 is required." >&2; exit 2; }
[[ "$device" =~ ^cuda:([0-9]+)$ ]] || { echo "--device must be like cuda:0." >&2; exit 2; }
for value in "$baseline_friction" "$baseline_armature" "$baseline_kp" "$baseline_kd" \
             "$candidate_friction" "$candidate_armature" "$candidate_kp" "$candidate_kd"; do
  is_nonnegative_number "$value" || { echo "Actuator values must be non-negative numbers: $value" >&2; exit 2; }
done
[[ "$lead_sim_steps" =~ ^[0-9]+$ ]] || { echo "--lead-sim-steps must be a non-negative integer." >&2; exit 2; }

repo_root="$(pwd -P)"
real_hdf5_host="$(realpath -m "$real_hdf5")"
[[ -f "$real_hdf5_host" ]] || { echo "Real HDF5 does not exist: $real_hdf5_host" >&2; exit 2; }

gpu_index="${device#cuda:}"
timestamp="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
comparison_id="piper_joint_tracking_${task}_${timestamp}"
output_root="${repo_root}/output"
comparison_dir="${output_root}/${comparison_id}"
baseline_run_id="${comparison_id}_baseline"
candidate_run_id="${comparison_id}_candidate"
presentation_dir="${comparison_dir}"
if [[ -n "$publish_dir" ]]; then
  presentation_dir="$(realpath -m "$publish_dir")"
fi

trace_host_path="${comparison_dir}/reference_joint_trace.npz"

echo "[RoboLab] Piper joint-tracking comparison"
echo "[RoboLab] real_hdf5=${real_hdf5_host}; task=${task}; device=${device}"
echo "[RoboLab] baseline:  friction=${baseline_friction}, armature=${baseline_armature}, kp=${baseline_kp}, kd=${baseline_kd}"
echo "[RoboLab] candidate: friction=${candidate_friction}, armature=${candidate_armature}, kp=${candidate_kp}, kd=${candidate_kd}"
echo "[RoboLab] output: ${comparison_dir}"
echo "[RoboLab] presentation output: ${presentation_dir}"
if "$dry_run"; then
  echo "[RoboLab] Dry run complete; no replay was started."
  exit 0
fi

docker ps --format '{{.Names}}' | grep -qx "$container" || {
  echo "Docker container '$container' is not running." >&2; exit 1;
}

mkdir -p "$comparison_dir"
mkdir -p "$presentation_dir"

python3 analysis/extract_piper_joint_target_trace.py \
  --hdf5 "$real_hdf5_host" \
  --output "$trace_host_path"

case "$trace_host_path" in
  "$repo_root"/*) trace_container_path="/workspace/robolab/${trace_host_path#"$repo_root"/}" ;;
  *) echo "Trace path must stay inside this workspace: $trace_host_path" >&2; exit 1 ;;
esac

launch_replay() {
  local label="$1" run_id="$2" friction="$3" armature="$4" kp="$5" kd="$6"
  local log_file="${comparison_dir}/${label}.log"
  echo "[RoboLab] Starting ${label} replay; log: ${log_file}"
  docker exec -i \
    -e "ROBOLAB_PIPER_ARM_FRICTION=${friction}" \
    -e "ROBOLAB_PIPER_ARM_ARMATURE=${armature}" \
    -e "ROBOLAB_PIPER_ARM_KP=${kp}" \
    -e "ROBOLAB_PIPER_ARM_KD=${kd}" \
    "$container" \
    bash -lc '
      set -euo pipefail
      cd /workspace/robolab
      /workspace/isaaclab/isaaclab.sh -p policies/pi0_family/run_piper_joint_target_replay.py \
        --joint-trace "$1" --task "$2" --num-envs 1 --headless --device "$3" \
        --video-mode viewport --output-folder-name "$4" \
        --renderer realtime --rendering-type performance \
        --kit-args "--/renderer/activeGpu=$5 --/physics/cudaDevice=$5 --/renderer/multiGpu/enabled=false"
    ' bash "$trace_container_path" "$task" "$device" "$run_id" "$gpu_index" \
    >"$log_file" 2>&1
}

launch_replay baseline "$baseline_run_id" "$baseline_friction" "$baseline_armature" "$baseline_kp" "$baseline_kd"
launch_replay candidate "$candidate_run_id" "$candidate_friction" "$candidate_armature" "$candidate_kp" "$candidate_kd"

find_single_file() {
  local dir="$1" pattern="$2" description="$3"
  mapfile -t matches < <(find "$dir" -type f -name "$pattern" | sort)
  if ((${#matches[@]} != 1)); then
    echo "Expected exactly one ${description} in ${dir}, found ${#matches[@]}." >&2
    printf '  %s\n' "${matches[@]:-}" >&2
    exit 1
  fi
  printf '%s' "${matches[0]}"
}

baseline_dir="${output_root}/${baseline_run_id}/${task}"
candidate_dir="${output_root}/${candidate_run_id}/${task}"
baseline_video="$(find_single_file "$baseline_dir" '*joint_target_replay_viewport.mp4' 'baseline viewport video')"
candidate_video="$(find_single_file "$candidate_dir" '*joint_target_replay_viewport.mp4' 'candidate viewport video')"
baseline_summary="$(find_single_file "$baseline_dir" 'joint_tracking_summary.json' 'baseline summary JSON')"
candidate_summary="$(find_single_file "$candidate_dir" 'joint_tracking_summary.json' 'candidate summary JSON')"
baseline_timeseries="$(find_single_file "$baseline_dir" 'joint_tracking_timeseries.npz' 'baseline timeseries NPZ')"
candidate_timeseries="$(find_single_file "$candidate_dir" 'joint_tracking_timeseries.npz' 'candidate timeseries NPZ')"

baseline_presentation_video="${presentation_dir}/baseline_replay.mp4"
candidate_presentation_video="${presentation_dir}/candidate_replay.mp4"
baseline_presentation_summary="${presentation_dir}/baseline_summary.json"
candidate_presentation_summary="${presentation_dir}/candidate_summary.json"
baseline_presentation_timeseries="${presentation_dir}/baseline_timeseries.npz"
candidate_presentation_timeseries="${presentation_dir}/candidate_timeseries.npz"
left_arm_plot="${presentation_dir}/qpos_tracking_left_arm.png"
right_arm_plot="${presentation_dir}/qpos_tracking_right_arm.png"
gripper_plot="${presentation_dir}/qpos_tracking_gripper.png"
report_html="${presentation_dir}/joint_tracking_report.html"

publish_browser_video() {
  local source="$1" destination="$2" label="$3"
  if command -v ffmpeg >/dev/null 2>&1; then
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
cp "$baseline_summary" "$baseline_presentation_summary"
cp "$candidate_summary" "$candidate_presentation_summary"
cp "$baseline_timeseries" "$baseline_presentation_timeseries"
cp "$candidate_timeseries" "$candidate_presentation_timeseries"

python3 analysis/plot_piper_qpos_tracking.py \
  --baseline-timeseries "$baseline_presentation_timeseries" \
  --candidate-timeseries "$candidate_presentation_timeseries" \
  --output-dir "$presentation_dir" \
  --lead-sim-steps "$lead_sim_steps"

python3 analysis/build_piper_joint_tracking_report.py \
  --baseline-summary "$baseline_presentation_summary" \
  --baseline-timeseries "$baseline_presentation_timeseries" \
  --candidate-summary "$candidate_presentation_summary" \
  --candidate-timeseries "$candidate_presentation_timeseries" \
  --baseline-video "$(basename "$baseline_presentation_video")" \
  --candidate-video "$(basename "$candidate_presentation_video")" \
  --baseline-label "Baseline (f=${baseline_friction}, a=${baseline_armature}, kp=${baseline_kp}, kd=${baseline_kd})" \
  --candidate-label "Candidate (f=${candidate_friction}, a=${candidate_armature}, kp=${candidate_kp}, kd=${candidate_kd})" \
  --left-arm-plot "$(basename "$left_arm_plot")" \
  --right-arm-plot "$(basename "$right_arm_plot")" \
  --gripper-plot "$(basename "$gripper_plot")" \
  --lead-sim-steps "$lead_sim_steps" \
  --output "$report_html"

echo "[RoboLab] Baseline video: ${baseline_video}"
echo "[RoboLab] Candidate video: ${candidate_video}"
echo "[RoboLab] Baseline summary: ${baseline_summary}"
echo "[RoboLab] Candidate summary: ${candidate_summary}"
echo "[RoboLab] HTML report: ${report_html}"
