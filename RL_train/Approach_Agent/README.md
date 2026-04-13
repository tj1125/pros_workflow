# Approach_Agent

## Phase 0: PyBullet smoke test

This directory starts with a minimal PyBullet validation step before adding the
full `depth -> voxel -> IK -> OMPL` evaluator.

The Phase 0 goal is deliberately small:

- confirm `pybullet` imports correctly in the intended runtime
- load the real arm URDF already used by the project
- verify the expected controllable joint count
- reset the arm to a known pose
- solve one IK target and measure end-effector error
- run obstacle distance / collision queries
- validate whether a straight-line joint interpolation path remains obstacle-free

This is an environment check, not the final evaluator.

## Phase 1: PyBullet + OMPL box planning

After Phase 0 passes, the next isolated step is a real joint-space planner with
manual box obstacles:

- use the same project arm URDF and reset pose
- solve one IK target
- run `OMPL + RRTConnect` from reset state to the IK goal state
- reject states that collide with the manual box obstacles
- replay the planned path in GUI mode or export path frames in headless mode

This is the first true obstacle-avoidance stage. It still uses manual obstacles,
not depth/voxel reconstruction yet.

For this stage, joint planning bounds are defined explicitly in
`configs/pybullet_ompl.yaml` via `joint_bounds_deg`. This keeps the planner
independent from unrelated shared control configs whose limits may not match the
current reset pose or PyBullet test scene.

## Runtime expectation

The current shell in this workspace does not have `numpy` / `pybullet`
available, but `VLM_RL/Dockerfile` already installs the expected dependencies.
The OMPL stage also requires Python OMPL bindings. Please run this inside the
project Docker environment, or rebuild the image if the container is missing
packages.

## How to run

Run from this directory:

```bash
cd /home/scream/TJ/VLM_RL/RL_train/Approach_Agent
python -m scripts.test_pybullet_smoke --config configs/pybullet_smoke.yaml
```

To open a PyBullet window for inspection:

```bash
python -m scripts.test_pybullet_smoke --config configs/pybullet_smoke.yaml --gui --hold-seconds 30
```

When `--gui` is enabled, headless `.ppm` and frame exports are skipped automatically.

If you are inside a headless Docker container and cannot connect to X11, save an
off-screen debug render instead:

```bash
python -m scripts.test_pybullet_smoke \
  --config configs/pybullet_smoke.yaml \
  --save-debug-ppm outputs/pybullet_smoke.ppm
```

The `.ppm` image is rendered with PyBullet TinyRenderer in `DIRECT` mode, so it
does not require a display server. The same run also writes
`outputs/pybullet_smoke_topdown.ppm` and `outputs/pybullet_smoke_side.ppm` for
top-down and side views.

To export a headless animation sequence of the arm moving from reset pose to the
IK target:

```bash
python -m scripts.test_pybullet_smoke \
  --config configs/pybullet_smoke.yaml \
  --save-animation-dir outputs/pybullet_smoke_frames \
  --save-debug-ppm outputs/pybullet_smoke.ppm
```

This writes `frame_0000.ppm`, `frame_0001.ppm`, ... into the output directory.
If GUI is available, `--gui` will also show the same motion live instead of
jumping instantly to the goal.

The script prints a JSON report to stdout and exits with code `0` only when the
smoke test passes.

The current implementation is collision-aware in a limited sense:

- it checks whether the reset pose is already in collision
- it checks whether the IK goal state is in collision
- it checks each interpolated frame against the test obstacle

It still does not use OMPL or perform true obstacle avoidance. The current path
is only a straight interpolation in joint space.

## Phase 1 usage

Run the OMPL planning stage from this directory:

```bash
python -m scripts.test_pybullet_ompl --config configs/pybullet_ompl.yaml
```

To inspect the planner path in the PyBullet GUI:

```bash
python -m scripts.test_pybullet_ompl --config configs/pybullet_ompl.yaml --gui --hold-seconds 30
```

When `--gui` is enabled, headless `.ppm` and frame exports are skipped automatically.

To export headless renders plus the planned path frames:

```bash
python -m scripts.test_pybullet_ompl \
  --config configs/pybullet_ompl.yaml \
  --save-animation-dir outputs/pybullet_ompl_frames \
  --save-debug-ppm outputs/pybullet_ompl.ppm
```

This writes:

- `outputs/pybullet_ompl.ppm`
- `outputs/pybullet_ompl_topdown.ppm`
- `outputs/pybullet_ompl_side.ppm`
- `outputs/pybullet_ompl_frames/frame_0000.ppm` and onward

The OMPL JSON output reports whether:

- the reset pose is collision-free
- the IK goal state is collision-free
- `RRTConnect` found a path within the configured timeout
- the returned path stayed collision-free when replayed back through PyBullet

## Phase 2: Camera_Car depth -> voxel -> OMPL

The next isolated stage captures real `Camera_Car` RGBD from ROS, decodes the
depth PNG, backprojects it into a camera-frame point cloud, remaps that cloud
into the verified PyBullet basis, aligns the gripper midpoint to the PyBullet
EE anchor, voxelizes occupied points at 5 cm, and feeds those voxel boxes into
the same PyBullet + OMPL planner. For offline testing, the same stage can also
load a precomputed camera-frame point cloud from
`3090server/VLM_RL/grasp_agent/data/debug_outputs/latest_grasp_debug.npz`.

Run it from this directory:

```bash
python -m scripts.run_camera_car_voxel_ompl --config configs/camera_car_voxel_ompl.yaml
```

For GUI playback:

```bash
python -m scripts.run_camera_car_voxel_ompl --config configs/camera_car_voxel_ompl.yaml --gui --hold-seconds 30
```

The pipeline writes a timestamped output folder under `outputs/camera_car_voxel_ompl/`
containing:

- the captured RGB image bytes
- the captured depth PNG bytes
- `voxel_snapshot.npz`
- `camera_car_voxel_ompl_report.json`
- planner renders/frames when not using GUI

When the input debug NPZ includes a full `best_grasp_camera` pose, the planner
now also reports pose-quality metrics for the chosen end-effector result:

- `ee_position_error_m`
- `ee_orientation_error_deg`
- `approach_axis_offset_m`
- `lateral_offset_m`

These are measured against the requested target grasp pose after it is remapped
into the verified PyBullet basis. The current implementation treats the target
pose local `+X` axis as the grasp approach direction when splitting positional
error into approach-axis and lateral components.

For PyBullet-side inspection of a saved voxel snapshot plus the successful
planner path, use:

```bash
python -m scripts.visualize_voxel_snapshot \
  outputs/camera_car_voxel_ompl/<timestamp>/voxel_snapshot.npz \
  --report-json outputs/camera_car_voxel_ompl/<timestamp>/camera_car_voxel_ompl_report.json \
  --play-path
```

This viewer now draws:

- the target grasp pose frame when `target_rotation_pybullet_matrix` is present
- the reset EE frame and snapshot anchor frame
- the successful OMPL joint path when `planned_path_joint_states_deg` exists in
  the report JSON

Important:

- `scene_pc_camera` and the backprojected depth cloud are interpreted in camera
  coordinates: `x` right, `y` down, `z` forward
- before inserting obstacles into PyBullet, the current verified basis mapping
  is applied: `camera [x, y, z] -> pybullet [z, x, -y]`
- `gripper_midpoint_camera_xyz` is the single source of truth for the gripper
  midpoint in camera coordinates and is aligned to the PyBullet EE anchor
- `/amcl_pose` is kept as metadata only; without a hand-eye extrinsic, it is not
  used to pretend we know `Camera_Car -> arm base`
- if you want offline testing, set `debug_npz_path` in
  `configs/camera_car_voxel_ompl.yaml` and the pipeline will load
  `scene_pc_camera` directly instead of calling ROS capture
- example offline debug file:
  `../../../3090server/VLM_RL/grasp_agent/data/debug_outputs/latest_grasp_debug.npz`

## Files

- `configs/pybullet_smoke.yaml`: smoke test parameters
- `configs/pybullet_ompl.yaml`: OMPL box-planning parameters
- `configs/camera_car_voxel_ompl.yaml`: Camera_Car depth-to-voxel planning parameters
- `scripts/test_pybullet_smoke.py`: CLI entrypoint
- `scripts/test_pybullet_ompl.py`: OMPL planning CLI entrypoint
- `scripts/run_camera_car_voxel_ompl.py`: Camera_Car voxel-planning CLI entrypoint
- `src/pybullet_smoke.py`: minimal PyBullet validation logic
- `src/pybullet_ompl.py`: manual-obstacle OMPL planning stage
- `src/camera_car_voxel_ompl.py`: Camera_Car capture + voxel + OMPL stage

## Notes

- The smoke test mirrors the existing PyBullet setup in
  `tools/car_control/src/arm_control_pkg/arm_control_pkg/pybullet_ik_gripper.py`
  without depending on its ROS-coupled controller class.
- If IK fails because the chosen target is not reachable in your environment,
  adjust `test_target_position_base` in the config and rerun.
- Useful collision-related config fields:
  - `collision_query_distance_m`
  - `path_collision_threshold_m`
  - `stop_on_path_collision`
