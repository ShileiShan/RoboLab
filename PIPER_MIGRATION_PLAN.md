# Double Piper Migration Plan

Porting the "Double Piper" dual-arm single-object pick-and-place eval from
`/home/disk/ssl/IsaacLab-Arena` into RoboLab's own conventions (server-client
pi0 eval, USD scenes, `Task` dataclasses) — **not** porting Arena's
`Job`/`JobManager` batch-sweep machinery.

## Scope decisions (confirmed)

- Priority: get the environment running first. Full multi-object /
  grid-sampling batch eval is explicitly deferred.
- Dual-arm pi0 client (`Pi0PiperDualArmClient`) is being built concurrently,
  not deferred.
- Reuse RoboLab's own object/fixture assets (`rubiks_cube.usd`, `grey_bin.usd`,
  `table_maple.usd`) instead of porting Arena's asset files byte-for-byte.
- Dual-gripper handling: Piper tasks use a list of per-finger contact sensor
  names. RoboLab's gripper contact helper ORs across that list, so either arm
  and either finger can satisfy grab/detach checks.

## Progress so far

Implemented, not yet run/verified (no `pxr`/USD tooling or `uv` venv was
available in the implementing shell — done per explicit instruction to
implement first, skip the verification loop):

| File | Status |
|---|---|
| `robolab/robots/piper.py` | written — dual-arm robot def, four per-finger `contact_gripper` sensors; right-arm paths confirmed, left-arm paths inferred |
| `assets/robots/double_piper.usd` | added |
| `assets/scenes/piper_single_object_pick_place.usda` | written — object placement is **hand-authored, not physics-settled** |
| `robolab/tasks/piper/piper_single_object_pick_place_task.py` | written |
| `robolab/registrations/piper/` (`auto_env_registrations_jointpos.py`, `camera_presets.py`, `__init__.py`) | written |
| `examples/run_piper_gripper_toggle.py` | written |
| `policies/pi0_family/piper_client.py` (`Pi0PiperDualArmClient`) | written — wire format (`_pack_request` keys, `openpi_embodiment_adapter="piper"`, gripper rescale `[0,1]→[0,0.035]`) is **inferred from Arena, not verified against a live pi0/openpi server** |
| `policies/pi0_family/run_piper.py` | written |
| `robolab/core/task/conditionals.py`, `robolab/core/task/predicate_logic.py` | modified in-place (diff not yet reviewed) |

## Open risks / TODO before trusting this pipeline

1. **Left-arm finger prim paths unconfirmed** — `piper_L/left_finger_link`
   and `piper_L/right_finger_link` in `robolab/robots/piper.py` are inferred
   from the right-arm naming pattern and need checking against the real
   `double_piper.usd` prim tree. Right-arm paths are confirmed as
   `piper_R/left_finger_link` and `piper_R/right_finger_link`. Wrong paths fail
   silently (no contact detected), doesn't error.
2. **Scene not physics-settled** — run:
   ```
   assets/scenes/_utils/settle_scenes.py --scene piper_single_object_pick_place.usda --replace --screenshot
   ```
3. **pi0 wire format unverified** — needs a real running pi0/openpi server for
   the `piper` embodiment to confirm request/response shape and gripper
   rescaling.
4. **Not yet run at all**:
   ```
   uv run python examples/run_piper_gripper_toggle.py --headless
   uv run pytest tests/
   ```
5. `robolab/core/task/conditionals.py` / `predicate_logic.py` changes have not
   been reviewed against how the rest of the task library uses them (other
   tasks besides Piper depend on these files).

## Next steps

- [ ] Run `run_piper_gripper_toggle.py --headless` in a real IsaacSim venv,
      fix whatever the finger-contact / scene-settle issues surface.
- [ ] Settle the scene and re-check object placement visually.
- [ ] Stand up a pi0/openpi server for `piper` embodiment and validate
      `piper_client.py` wire format end-to-end.
- [ ] Review `conditionals.py`/`predicate_logic.py` diff for regressions in
      non-Piper tasks.
- [ ] Verify all four Piper finger contact sensors report contact as expected
      in Isaac Sim.
