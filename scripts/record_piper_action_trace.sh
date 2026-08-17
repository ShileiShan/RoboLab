#!/usr/bin/env bash
# Generate one Piper policy trajectory, then extract its environment-space
# actions into a portable .npz file for open-loop replay.

set -euo pipefail

readonly DEFAULT_CONTAINER="robolab_lh"
readonly DEFAULT_TASK="MakeBreakfastTask"
readonly DEFAULT_POLICY="pi05"
readonly DEFAULT_SECONDS=20
readonly DEFAULT_DEVICE="cuda:0"

container="$DEFAULT_CONTAINER"
task="$DEFAULT_TASK"
policy="$DEFAULT_POLICY"
seconds="$DEFAULT_SECONDS"
device="$DEFAULT_DEVICE"
remote_host=""
remote_port=""
trace_output=""
dry_run=false

usage() {
  cat <<'EOF'
Usage: scripts/record_piper_action_trace.sh --remote-host HOST --remote-port PORT --output TRACE.npz [options]

Run one closed-loop Piper policy episode and save its already post-processed
14-D environment actions as an open-loop replay trace.

Required:
  --remote-host HOST            Pi0 RTC policy-server host.
  --remote-port PORT            Pi0 RTC policy-server port.
  --output TRACE.npz            Workspace path for the extracted action trace.

Options:
  --task NAME                   Piper-compatible task class (default: MakeBreakfastTask)
  --policy NAME                 Pi0-family policy variant (default: pi05)
  --seconds S                   Policy episode duration in simulated seconds (default: 20)
  --device DEVICE               Physical Isaac GPU, e.g. cuda:1 (default: cuda:0)
  --container NAME              Docker container (default: robolab_lh)
  --dry-run                     Print configuration without starting an evaluation.
  -h, --help                    Show this help text.
EOF
}

need_value() { (($# >= 2)) || { echo "Missing value for $1." >&2; exit 2; }; }

while (($#)); do
  case "$1" in
    --remote-host) need_value "$@"; remote_host="$2"; shift 2 ;;
    --remote-port) need_value "$@"; remote_port="$2"; shift 2 ;;
    --output) need_value "$@"; trace_output="$2"; shift 2 ;;
    --task) need_value "$@"; task="$2"; shift 2 ;;
    --policy) need_value "$@"; policy="$2"; shift 2 ;;
    --seconds) need_value "$@"; seconds="$2"; shift 2 ;;
    --device) need_value "$@"; device="$2"; shift 2 ;;
    --container) need_value "$@"; container="$2"; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

is_positive_number() {
  [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] && awk -v value="$1" 'BEGIN { exit !(value > 0) }'
}

[[ -n "$remote_host" ]] || { echo "--remote-host is required." >&2; exit 2; }
[[ "$remote_port" =~ ^[1-9][0-9]*$ ]] && ((remote_port <= 65535)) || {
  echo "--remote-port must be an integer in 1..65535." >&2; exit 2;
}
[[ -n "$trace_output" ]] || { echo "--output is required." >&2; exit 2; }
is_positive_number "$seconds" || { echo "--seconds must be positive." >&2; exit 2; }
[[ "$device" =~ ^cuda:([0-9]+)$ ]] || { echo "--device must be like cuda:0." >&2; exit 2; }

repo_root="$(pwd -P)"
trace_host_path="$(realpath -m "$trace_output")"
case "$trace_host_path" in
  "$repo_root"/*) trace_container_path="/workspace/robolab/${trace_host_path#"$repo_root"/}" ;;
  *) echo "--output must be inside this workspace: $repo_root" >&2; exit 2 ;;
esac

gpu_index="${device#cuda:}"
timestamp="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
run_id="piper_action_trace_${task}_${timestamp}"
output_root="${repo_root}/output"
log_dir="${output_root}/${run_id}"

echo "[RoboLab] Recording Piper action trace"
echo "[RoboLab] task=${task}, seconds=${seconds}, device=${device}"
echo "[RoboLab] policy endpoint=${remote_host}:${remote_port}"
echo "[RoboLab] trace=${trace_host_path}"
if "$dry_run"; then
  echo "[RoboLab] Dry run complete; no evaluation was started."
  exit 0
fi

docker inspect --format '{{.State.Running}}' "$container" 2>/dev/null | grep -qx true || {
  echo "Docker container '$container' is not running." >&2; exit 1;
}
mkdir -p "$log_dir"

docker exec -i \
  -e "ROBOLAB_EPISODE_LENGTH_S=${seconds}" \
  -e "ROBOLAB_ENABLE_COMPARISON_CAMERA=0" \
  "$container" \
  bash -lc '
    set -euo pipefail
    cd /workspace/robolab
    /workspace/isaaclab/isaaclab.sh -p policies/pi0_family/run_piper.py \
      --rtc --remote-host "$1" --remote-port "$2" --policy "$3" \
      --task "$4" --num-envs 1 --num-runs 1 --headless --device "$5" \
      --kit-args "--/renderer/activeGpu=$7 --/physics/cudaDevice=$7 --/renderer/multiGpu/enabled=false" \
      --video-mode none --output-folder-name "$6"
  ' bash "$remote_host" "$remote_port" "$policy" "$task" "$device" "$run_id" "$gpu_index" \
  >"${log_dir}/record.log" 2>&1

mapfile -t hdf5_files < <(find "${output_root}/${run_id}" -type f -name 'run_0.hdf5' | sort)
if ((${#hdf5_files[@]} != 1)); then
  echo "Expected exactly one run_0.hdf5, found ${#hdf5_files[@]}." >&2
  printf '  %s\n' "${hdf5_files[@]:-}" >&2
  echo "[RoboLab] The evaluation log is ${log_dir}/record.log" >&2
  echo "[RoboLab] Last 80 log lines:" >&2
  tail -n 80 "${log_dir}/record.log" >&2 || true
  exit 1
fi

python3 analysis/extract_piper_action_trace.py \
  --hdf5 "${hdf5_files[0]}" \
  --output "$trace_host_path" \
  --control-hz 30

echo "[RoboLab] Source HDF5: ${hdf5_files[0]}"
echo "[RoboLab] Replay trace: ${trace_host_path%.npz}.npz"
