# Pi0 Family (OpenPI)

The Pi0 family — `pi0`, `pi0_fast`, `pi05`, `paligemma`, and `paligemma_fast` — is served by a single client, `Pi0DroidJointposClient`, over a WebSocket-based OpenPI policy server. The variant is selected at runtime via `--policy`; each variant supplies its own per-variant defaults inside the client.

See the [policies README](../README.md) for the shared client architecture and common CLI options.
For pi0-family variants, pass `--policy {pi0,pi0_fast,pi05,paligemma,paligemma_fast}`.

## Install the server

1. Clone [`git@github.com:xuningy/openpi.git`](https://github.com/xuningy/openpi) and follow install instructions there. **Do not** install OpenPI in the same virtual environment as RoboLab — it runs separately.

2. Install the OpenPI **client** in the RoboLab environment:
   ```bash
   cd robolab
   uv pip install -e ../openpi/packages/openpi-client
   ```

## Start the policy server

Open a separate terminal and launch the server. We set `XLA_PYTHON_CLIENT_MEM_FRACTION` to 50% to avoid JAX consuming all GPU memory.

**Pi05:**
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid_jointpos \
    --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos
```

**Pi0-fast:**
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_droid_jointpos \
    --policy.dir=gs://openpi-assets-simeval/pi0_fast_droid_jointpos
```

**Pi0:**
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_droid_jointpos \
    --policy.dir=gs://openpi-assets-simeval/pi0_droid_jointpos
```

**PaliGemma Binning:**
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=paligemma_binning_droid_jointpos \
    --policy.dir=gs://openpi-assets-simeval/paligemma_binning_droid_jointpos
```

## Run evaluation

In the RoboLab terminal

```bash
cd robolab
uv run python policies/pi0_family/run.py --policy pi05 --headless
```

The default connection is `localhost:8000`. To change:
```bash
uv run python policies/pi0_family/run.py --policy pi05 --remote-host <HOST> --remote-port <PORT>
```

A full WebSocket URI (e.g. a hosted endpoint) can be passed with `--remote-uri`, which overrides `--remote-host` / `--remote-port`.

## Dual-arm Piper RTC mode

The dual-arm Piper runner supports the training-time action-conditioning RTC
server. It executes a full server chunk locally at 30 Hz, then launches a
background replan after ten actions. The replan sends the latest three camera
images, 12 arm joints, two normalized gripper states, the instruction, and the
unexecuted **32-D model-space** portion of the old chunk. The server uses a
five-action hard-conditioned prefix and returns paired `(50, 14)` robot-space
and `(50, 32)` model-space chunks. The robot action order is
`[left_arm(6), left_gripper, right_arm(6), right_gripper]`.

```bash
uv run python policies/pi0_family/run_piper.py \
  --rtc --task PiperSingleObjectPickPlaceTask --headless --video-mode none
```

RTC defaults to `localhost:8001`, controls at 30 Hz, replans every ten actions,
and conditions on five committed actions. Use `--remote-host`, `--remote-port`,
`--remote-uri`, `--control-hz`, `--rtc-execution-horizon`, or
`--rtc-inference-delay-steps` to override those settings. Inference is
asynchronous after the first chunk, so it does not block control ticks. This
protocol uses one ordered WebSocket stream; run it with `--num-envs 1`.

## Variation scripts

The pi0_family folder also ships controlled-variation runners that wrap the same client to sweep a single axis per registered env:

- `run_lighting.py` — lighting variations
- `run_camera_pose_variation.py` — camera pose variations
- `run_background_variation.py` — background variations
- `run_table_variation.py` — table variations

Each takes the same connection and common eval flags as `run.py`.
