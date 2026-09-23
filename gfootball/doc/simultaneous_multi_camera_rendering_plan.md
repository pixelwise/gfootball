# Simultaneous multi-camera rendering

## Goal

Generate synchronized output for `STATIC_SIDE_0` through `STATIC_SIDE_3`
from one simulated episode. Physics and AI must advance once per environment
step. After that advance, the unchanged world state is rendered sequentially
through all four calibrated cameras, so every set of views has the same
`engine_step`.

This is simultaneous in simulation time, not parallel GPU rendering. The four
views still require four complete render passes through one graphics context.
The implementation keeps `GameConfig.camera` as the primary compatibility
view and exposes the complete synchronized capture as ordered camera mappings.

## Configuration and compatibility

Use the optional YAML field:

```yaml
cameras:
  - "static-side-0"
  - "static-side-1"
  - "static-side-2"
  - "static-side-3"
```

- Keep `camera` as the existing single-view interface. When `cameras` is
  absent, effective cameras are `[camera]` and all current behavior,
  observation fields, filenames, and `dump_info` keys remain unchanged.
- When `cameras` is present, require a non-empty list with no duplicates. Its
  first entry is the primary camera used for legacy scalar observation fields.
  `cameras` takes precedence over `camera`; the resolved configuration saved
  in the episode directory sets `camera` to the first list entry.
- All views initially share `render_resolution_x`, `render_resolution_y`,
  FPS, `video_format`, and quality. Per-camera resolutions and codecs are out
  of scope for the first implementation.
- The existing `write_video`, `write_segmentation_video`,
  `write_instance_segmentation_video`, and `write_ball_coordinates` switches
  apply to every selected camera.
- Add the four-view configuration as a checked-in YAML preset rather than
  changing the default single-camera configuration.

- The checked-in `gfootball/configs/static-side-multi.yaml` preset selects
  all four calibrated cameras without changing the default configuration.

Run it with:

```sh
./start_game_docker.sh gfootball/configs/static-side-multi.yaml
```

At native `2160 x 3840` resolution, one RGB bundle is roughly 95 MiB before
segmentation and encoder buffers. Rendering work is approximately four times a
single view because the graphics passes are sequential; physics and AI still
advance only once.

The Python `GameConfig` model in
[`play_game.py`](../play_game.py) should expose `cameras` as an optional list
of `CameraType` values and perform the validation above. Environment creation
in [`football_env_core.py`](../env/football_env_core.py) should resolve the
list once and pass the native camera enum sequence to the engine.

## Native rendering architecture

### Capture one world state from several cameras

Extend `GameEnv` with a configured vector of render cameras and per-camera
capture storage. `GameEnv::step()` should retain its existing physics loop, but
replace the final single `render()` call with a multi-view capture operation:

1. Advance physics and AI exactly as today.
2. Freeze that completed simulation state and record its `engine_step`.
3. Render the secondary cameras first and the primary camera last, while
   retaining the configured order in the returned mapping. For each camera:
   - select the camera type;
   - run the existing `Match::UpdateCamera()` pose/FOV setup for that type;
   - render RGB, including the camera's calibrated postprocess distortion;
   - render/capture segmentation when requested;
   - copy both framebuffer results before the next camera overwrites them;
   - compute distorted ball position and visibility using that active camera;
     and
   - store the capture under the camera enum and the same `engine_step`.
4. Swap/present only the final primary view when an onscreen buffer swap is
   needed.

Do not advance `ProcessPhase()`, animation time, randomness, controllers, or
the scenario clock between camera renders. Camera selection must be a scoped
render operation, not another environment step.

Refactor camera pose selection in
[`match.cpp`](../../third_party/gfootball_engine/src/onthepitch/match.cpp) so
it can be invoked deliberately for each view. During each render, the active
`GameConfig.camera` value must match the selected view because the RGB and
segmentation postprocess obtains lens calibration from that value. The four
existing pose matrices, FOVs, and `LensCalibration` entries remain the source
of truth.

### Native capture API

Add methods exposed through the Boost.Python wrapper for:

```text
set_render_cameras(sequence<CameraType>)
get_frame_for_camera(CameraType) -> bytes
get_segmentation_frame_for_camera(CameraType) -> bytes
get_ball_screen_position_for_camera(CameraType) -> (x, y)
get_ball_screen_visible_for_camera(CameraType) -> bool
get_capture_engine_step_for_camera(CameraType) -> int
```

Keep `get_frame()`, `get_segmentation_frame()`, and the scalar ball fields in
`SharedInfo`; they return the primary camera capture for backward
compatibility. Reject requests for a camera not configured in the current
capture rather than silently returning the last framebuffer.

Store copied byte buffers per view in the native engine. The renderer currently
owns only a last-screen and last-segmentation buffer, so retaining references
would make every camera appear to contain the final view.

## Python observations and recording

For multi-camera mode, add these observation fields:

```python
camera_frames: dict[str, np.ndarray]
camera_segmentation_frames: dict[str, np.ndarray]
camera_ball_screen_position: dict[str, np.ndarray]
camera_ball_screen_visible: dict[str, bool]
camera_engine_step: dict[str, int]
```

Keys are serialized camera names such as `static-side-0`. Preserve the
existing `frame`, `segmentation_frame`, `ball_screen_position`,
`ball_screen_visible`, and `engine_step` fields as aliases of the primary
camera. Validate that every camera in a bundle reports the same engine step
before accepting it.

Avoid deep-copying or pickling four native-resolution images on every step.
Keep the multi-view pixel bundle transient, pass it directly to the observation
processor, and remove it before constructing the durable trace observation.
The `.dump` continues to store gameplay state/actions and configuration, not
raw RGB or segmentation arrays. This is important because four `2160 x 3840`
RGB frames occupy about 95 MiB before segmentation and temporary copies.

Refactor `ObservationProcessor`/`ActiveDump` into one incremental writer set
per camera. Each set owns RGB, semantic segmentation, instance segmentation,
ball coordinate, and engine-step state. All writers receive one item from the
same validated capture bundle before processing the next environment step.

### Artifact naming

In multi-camera mode, use deterministic camera-qualified names:

```text
<episode>_static-side-0.<format>
<episode>_static-side-1.<format>
<episode>_static-side-2.<format>
<episode>_static-side-3.<format>
```

Associated artifacts use the same camera qualifier:

```text
<episode>_static-side-0_segmentation.<aux-format>
<episode>_static-side-0_instances.<aux-format>
<episode>_static-side-0_instances.npz
<episode>_static-side-0_ball.npz
```

For M3U8 output, each RGB view retains the existing exact pair convention:

```text
<episode>_static-side-0.m3u8
<episode>_static-side-0_segments.ts
```

The auxiliary format matches `video_format` for AVI, WebM, and MP4. Semantic
and instance videos continue to fall back to MP4 when RGB uses M3U8.
Publish each TS file before its playlist and finalize all writers even when one
writer reports an error.

Add plural mappings to `dump_info`:

```python
videos: dict[str, str]
segmentation_videos: dict[str, str]
instance_segmentation_videos: dict[str, str]
instance_segmentation_metadata_by_camera: dict[str, str]
ball_coordinates_by_camera: dict[str, str]
```

The existing singular keys remain aliases for the primary camera. Single-view
mode continues to use its current filenames and dictionary shape.

## Implemented architecture

1. Add and validate the `cameras` configuration while normalizing single-view
   and multi-view requests to one ordered internal camera list.
2. Refactor ball projection into a reusable per-camera operation, without
   changing the current calibrated transform.
3. Add native camera-list configuration, sequential same-state rendering,
   copied capture storage, accessors, and Boost.Python bindings.
4. Retrieve and validate the capture bundle in `FootballEnvCore`, while
   maintaining primary-camera observation aliases.
5. Refactor trace/video processing into per-camera writer sets and implement
   camera-qualified artifact and `dump_info` naming.
6. Add the four-camera YAML preset, document the public fields and expected
   cost, then rebuild the native extension and Docker image.

## Validation checklist

- Native tests: one physics advance produces four captures with identical
  engine steps; camera rendering does not change positions, score, RNG/state,
  or the next-step result; missing/unconfigured capture access fails clearly.
- Camera equivalence: from the same saved engine state, each multi-camera frame
  matches a corresponding single-camera render. Compare RGB exactly where
  deterministic, otherwise with a documented pixel tolerance.
- Calibration: compare all four views with the existing `cam7_0.png` through
  `cam7_3.png` references, including center, sidelines, and corners.
- Segmentation: every view matches RGB dimensions and distortion, contains only
  valid instance IDs, and has identical label channels. Semantic and instance
  writers must consume the same per-view label frame.
- Ball coordinates: overlay each reported point on its corresponding RGB
  frame, verify visibility bounds, and confirm all coordinate metadata engine
  steps match video/instance frame steps.
- Recording: verify four complete RGB outputs, four auxiliary output sets when
  enabled, correct `dump_info` mappings, synchronized frame counts/FPS, and no
  filename collisions. For M3U8, validate every playlist/TS pair with
  `ffprobe` and check byte ranges plus `#EXT-X-ENDLIST`.
- Compatibility: run existing single-camera environment, replay, MP4, M3U8,
  segmentation, state snapshot, and `play_game.py` tests unchanged.
- Resource behavior: measure peak memory and render throughput at
  `1080 x 1920` and `2160 x 3840`; confirm frames are written incrementally
  and are not retained in `.dump` files or multiplied by trace deep copies.
- Docker: rebuild the image and run a deterministic four-camera episode under
  Xvfb/off-screen rendering, then verify that all views share the same ordered
  engine-step sequence.

### Recorded validation results

The Docker/Xvfb validation at `2026-09-23` measured one four-view step with RGB
and segmentation enabled. At `1080 x 1920`, the step took about `0.42 s` and
process RSS was about `749 MiB`; the four RGB arrays and four label arrays were
`23.7 MiB` each. At `2160 x 3840`, the step took about `1.51 s` and process RSS
was about `1.63 GiB`; RGB and labels were `94.9 MiB` each. These are software
Mesa/container measurements and are intended as capacity guidance, not a GPU
performance benchmark.

A deterministic comparison confirmed that every multi-camera RGB and label
frame is byte-identical to its corresponding isolated single-camera render.
A short recorded episode produced 14 synchronized frames for every RGB,
semantic, and instance stream; all four ball metadata files contained the same
ordered engine-step sequence.

## Acceptance criteria

- One call to the environment step advances one simulation and yields four
  camera captures for the same engine state.
- Each camera's RGB, segmentation, and ball coordinates use its own calibrated
  pose and lens distortion.
- All four videos have identical frame counts, FPS, dimensions, and engine-step
  sequences.
- Existing single-camera configurations and artifact consumers continue to
  work without changes.
- The implementation uses one graphics context and bounded per-step memory; it
  does not create four environment instances or replay the episode four times.
