#!/usr/bin/env bash
# Compare strict real-state visualization against two Piper actuator parameter sets.

set -euo pipefail

readonly DEFAULT_CONTAINER="robolab_lh"
readonly DEFAULT_TASK="MakeBreakfastTask"
readonly DEFAULT_DEVICE="cuda:0"
readonly DEFAULT_FRAMES=8

baseline_kp=400
baseline_kd=80
baseline_armature=0.05
baseline_friction=0.10
candidate_kp=867.0147
candidate_kd=127.7443
candidate_armature=0.4855
candidate_friction=0.3516

container="$DEFAULT_CONTAINER"
task="$DEFAULT_TASK"
device="$DEFAULT_DEVICE"
frames="$DEFAULT_FRAMES"
publish_dir=""
dry_run=false
real_hdf5=""

usage() {
  cat <<'EOF'
Usage: scripts/compare_piper_real_state_vs_params.sh --real-hdf5 EPISODE.hdf5 [options]

Generate three RoboLab videos from the same real-robot qpos trace:
1. strict joint-state replay ("real-state proxy")
2. baseline joint-target replay
3. candidate joint-target replay

Then build two blue/orange overlay comparisons:
- real-state proxy vs baseline
- real-state proxy vs candidate

Required:
  --real-hdf5 FILE              Raw real-robot HDF5 episode with observations/qpos.

Actuator options:
  --baseline-kp VALUE           Baseline stiffness / Kp (default: 400)
  --baseline-kd VALUE           Baseline damping / Kd (default: 80)
  --baseline-armature VALUE     Baseline armature (default: 0.05)
  --baseline-friction VALUE     Baseline friction (default: 0.10)
  --candidate-kp VALUE          Candidate stiffness / Kp (default: 867.0147)
  --candidate-kd VALUE          Candidate damping / Kd (default: 127.7443)
  --candidate-armature VALUE    Candidate armature (default: 0.4855)
  --candidate-friction VALUE    Candidate friction (default: 0.3516)

Other options:
  --task NAME                   Piper-compatible task class (default: MakeBreakfastTask)
  --device DEVICE               Physical Isaac GPU, e.g. cuda:1 (default: cuda:0)
  --container NAME              Docker container (default: robolab_lh)
  --frames N                    Number of equally spaced contact-sheet frames (default: 8)
  --publish-dir DIR             Copy final artifacts to DIR (default: comparison run dir)
  --dry-run                     Validate and print configuration only
  -h, --help                    Show this help text
EOF
}

need_value() { (($# >= 2)) || { echo "Missing value for $1." >&2; exit 2; }; }

while (($#)); do
  case "$1" in
    --real-hdf5) need_value "$@"; real_hdf5="$2"; shift 2 ;;
    --baseline-kp) need_value "$@"; baseline_kp="$2"; shift 2 ;;
    --baseline-kd) need_value "$@"; baseline_kd="$2"; shift 2 ;;
    --baseline-armature) need_value "$@"; baseline_armature="$2"; shift 2 ;;
    --baseline-friction) need_value "$@"; baseline_friction="$2"; shift 2 ;;
    --candidate-kp) need_value "$@"; candidate_kp="$2"; shift 2 ;;
    --candidate-kd) need_value "$@"; candidate_kd="$2"; shift 2 ;;
    --candidate-armature) need_value "$@"; candidate_armature="$2"; shift 2 ;;
    --candidate-friction) need_value "$@"; candidate_friction="$2"; shift 2 ;;
    --task) need_value "$@"; task="$2"; shift 2 ;;
    --device) need_value "$@"; device="$2"; shift 2 ;;
    --container) need_value "$@"; container="$2"; shift 2 ;;
    --frames) need_value "$@"; frames="$2"; shift 2 ;;
    --publish-dir) need_value "$@"; publish_dir="$2"; shift 2 ;;
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
[[ "$frames" =~ ^[1-9][0-9]*$ ]] || { echo "--frames must be a positive integer." >&2; exit 2; }
for value in \
  "$baseline_kp" "$baseline_kd" "$baseline_armature" "$baseline_friction" \
  "$candidate_kp" "$candidate_kd" "$candidate_armature" "$candidate_friction"; do
  is_nonnegative_number "$value" || { echo "Actuator values must be non-negative numbers: $value" >&2; exit 2; }
done

repo_root="$(pwd -P)"
real_hdf5_host="$(realpath -m "$real_hdf5")"
[[ -f "$real_hdf5_host" ]] || { echo "Real HDF5 does not exist: $real_hdf5_host" >&2; exit 2; }

gpu_index="${device#cuda:}"
timestamp="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
comparison_id="piper_real_state_vs_params_${task}_${timestamp}"
output_root="${repo_root}/output"
comparison_dir="${output_root}/${comparison_id}"
presentation_dir="${comparison_dir}"
if [[ -n "$publish_dir" ]]; then
  presentation_dir="$(realpath -m "$publish_dir")"
fi

trace_host_path="${comparison_dir}/reference_joint_trace.npz"
real_run_id="${comparison_id}_real_state"
baseline_run_id="${comparison_id}_baseline"
candidate_run_id="${comparison_id}_candidate"

echo "[RoboLab] Piper real-state vs parameter comparison"
echo "[RoboLab] real_hdf5=${real_hdf5_host}; task=${task}; device=${device}; frames=${frames}"
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

launch_state_replay() {
  local run_id="$1"
  local log_file="${comparison_dir}/real_state.log"
  echo "[RoboLab] Starting real-state replay; log: ${log_file}"
  docker exec -i "$container" bash -lc '
      set -euo pipefail
      cd /workspace/robolab
      /workspace/isaaclab/isaaclab.sh -p policies/pi0_family/run_piper_joint_state_replay.py \
        --joint-trace "$1" --task "$2" --num-envs 1 --headless --device "$3" \
        --video-mode viewport --comparison-object-set pour_water --output-folder-name "$4" \
        --renderer realtime --rendering-type performance \
        --kit-args "--/renderer/activeGpu=$5 --/physics/cudaDevice=$5 --/renderer/multiGpu/enabled=false"
    ' bash "$trace_container_path" "$task" "$device" "$run_id" "$gpu_index" \
    >"$log_file" 2>&1
}

launch_target_replay() {
  local label="$1" run_id="$2" kp="$3" kd="$4" armature="$5" friction="$6"
  local log_file="${comparison_dir}/${label}.log"
  echo "[RoboLab] Starting ${label} replay; log: ${log_file}"
  docker exec -i \
    -e "ROBOLAB_PIPER_ARM_KP=${kp}" \
    -e "ROBOLAB_PIPER_ARM_KD=${kd}" \
    -e "ROBOLAB_PIPER_ARM_ARMATURE=${armature}" \
    -e "ROBOLAB_PIPER_ARM_FRICTION=${friction}" \
    "$container" \
    bash -lc '
      set -euo pipefail
      cd /workspace/robolab
      /workspace/isaaclab/isaaclab.sh -p policies/pi0_family/run_piper_joint_target_replay.py \
        --joint-trace "$1" --task "$2" --num-envs 1 --headless --device "$3" \
        --video-mode viewport --comparison-object-set pour_water --output-folder-name "$4" \
        --renderer realtime --rendering-type performance \
        --kit-args "--/renderer/activeGpu=$5 --/physics/cudaDevice=$5 --/renderer/multiGpu/enabled=false"
    ' bash "$trace_container_path" "$task" "$device" "$run_id" "$gpu_index" \
    >"$log_file" 2>&1
}

launch_state_replay "$real_run_id"
launch_target_replay baseline "$baseline_run_id" "$baseline_kp" "$baseline_kd" "$baseline_armature" "$baseline_friction"
launch_target_replay candidate "$candidate_run_id" "$candidate_kp" "$candidate_kd" "$candidate_armature" "$candidate_friction"

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

real_dir="${output_root}/${real_run_id}/${task}"
baseline_dir="${output_root}/${baseline_run_id}/${task}"
candidate_dir="${output_root}/${candidate_run_id}/${task}"

real_video="$(find_single_file "$real_dir" 'joint_state_replay_viewport.mp4' 'real-state viewport video')"
real_robot_mask="$(find_single_file "$real_dir" 'joint_state_replay_viewport_robot_mask.mp4' 'real-state robot mask video')"
real_object_mask="$(find_single_file "$real_dir" 'joint_state_replay_viewport_tracked_objects_mask.mp4' 'real-state object mask video')"
baseline_video="$(find_single_file "$baseline_dir" 'joint_target_replay_viewport.mp4' 'baseline viewport video')"
baseline_robot_mask="$(find_single_file "$baseline_dir" 'joint_target_replay_viewport_robot_mask.mp4' 'baseline robot mask video')"
baseline_object_mask="$(find_single_file "$baseline_dir" 'joint_target_replay_viewport_tracked_objects_mask.mp4' 'baseline object mask video')"
candidate_video="$(find_single_file "$candidate_dir" 'joint_target_replay_viewport.mp4' 'candidate viewport video')"
candidate_robot_mask="$(find_single_file "$candidate_dir" 'joint_target_replay_viewport_robot_mask.mp4' 'candidate robot mask video')"
candidate_object_mask="$(find_single_file "$candidate_dir" 'joint_target_replay_viewport_tracked_objects_mask.mp4' 'candidate object mask video')"

publish_browser_video() {
  local source="$1" destination="$2"
  if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -y -v error -i "$source" \
      -map 0:v:0 -an -c:v libx264 -preset medium -crf 20 \
      -pix_fmt yuv420p -movflags +faststart "$destination"
  else
    cp "$source" "$destination"
  fi
}

real_video_out="${presentation_dir}/real_state_replay.mp4"
real_robot_mask_out="${presentation_dir}/real_state_robot_mask.mp4"
real_object_mask_out="${presentation_dir}/real_state_tracked_objects_mask.mp4"
baseline_video_out="${presentation_dir}/baseline_replay.mp4"
baseline_robot_mask_out="${presentation_dir}/baseline_robot_mask.mp4"
baseline_object_mask_out="${presentation_dir}/baseline_tracked_objects_mask.mp4"
candidate_video_out="${presentation_dir}/candidate_replay.mp4"
candidate_robot_mask_out="${presentation_dir}/candidate_robot_mask.mp4"
candidate_object_mask_out="${presentation_dir}/candidate_tracked_objects_mask.mp4"

publish_browser_video "$real_video" "$real_video_out"
publish_browser_video "$real_robot_mask" "$real_robot_mask_out"
publish_browser_video "$real_object_mask" "$real_object_mask_out"
publish_browser_video "$baseline_video" "$baseline_video_out"
publish_browser_video "$baseline_robot_mask" "$baseline_robot_mask_out"
publish_browser_video "$baseline_object_mask" "$baseline_object_mask_out"
publish_browser_video "$candidate_video" "$candidate_video_out"
publish_browser_video "$candidate_robot_mask" "$candidate_robot_mask_out"
publish_browser_video "$candidate_object_mask" "$candidate_object_mask_out"

python3 analysis/visualize_piper_trajectory_comparison.py \
  --baseline-video "$real_video_out" \
  --candidate-video "$baseline_video_out" \
  --baseline-mask-video "$real_robot_mask_out" \
  --candidate-mask-video "$baseline_robot_mask_out" \
  --baseline-object-mask-video "$real_object_mask_out" \
  --candidate-object-mask-video "$baseline_object_mask_out" \
  --frames "$frames" \
  --layout overlay \
  --baseline-label "Real-State Proxy" \
  --candidate-label "Baseline (friction=${baseline_friction}, armature=${baseline_armature}, kp=${baseline_kp}, kd=${baseline_kd})" \
  --overlay-video "${presentation_dir}/real_vs_baseline_overlay.mp4" \
  --output "${presentation_dir}/real_vs_baseline_contact_sheet.png"

python3 analysis/visualize_piper_trajectory_comparison.py \
  --baseline-video "$real_video_out" \
  --candidate-video "$candidate_video_out" \
  --baseline-mask-video "$real_robot_mask_out" \
  --candidate-mask-video "$candidate_robot_mask_out" \
  --baseline-object-mask-video "$real_object_mask_out" \
  --candidate-object-mask-video "$candidate_object_mask_out" \
  --frames "$frames" \
  --layout overlay \
  --baseline-label "Real-State Proxy" \
  --candidate-label "Candidate (friction=${candidate_friction}, armature=${candidate_armature}, kp=${candidate_kp}, kd=${candidate_kd})" \
  --overlay-video "${presentation_dir}/real_vs_candidate_overlay.mp4" \
  --output "${presentation_dir}/real_vs_candidate_contact_sheet.png"

echo "[RoboLab] Real-state video: ${real_video_out}"
echo "[RoboLab] Baseline video: ${baseline_video_out}"
echo "[RoboLab] Candidate video: ${candidate_video_out}"
echo "[RoboLab] Real vs baseline overlay: ${presentation_dir}/real_vs_baseline_overlay.mp4"
echo "[RoboLab] Real vs candidate overlay: ${presentation_dir}/real_vs_candidate_overlay.mp4"
echo "[RoboLab] Real vs baseline contact sheet: ${presentation_dir}/real_vs_baseline_contact_sheet.png"
echo "[RoboLab] Real vs candidate contact sheet: ${presentation_dir}/real_vs_candidate_contact_sheet.png"
