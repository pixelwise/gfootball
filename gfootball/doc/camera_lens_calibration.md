# Real-lens camera calibration and distortion

This document describes how the four `STATIC_SIDE_*` cameras reproduce the
Wolfsburg `cam7` lens views. The implementation combines calibrated camera
poses with a post-process radial lens warp. The same warp is applied to RGB,
player segmentation, and screen-space ball coordinates.

## Calibration source

The calibration came from recording
`4fe34067-e277-447a-af7e-9e565efad535` on `WolfsburgServer`, view
`15e6e2e9-2ccc-4b99-f28b-1b80c60d66d7` (`AOK - Pano Mitte`). The external
source files used during calibration were:

```text
/data/non-www/permanent-data/recordings/WolfsburgServer/4fe34067-e277-447a-af7e-9e565efad535/recording.json
```

The streams are `cam7_0` through `cam7_3`. Their native frames are
`2160 x 3840` pixels (width by height), so the normalized image aspect is

```text
a = width / height = 2160 / 3840 = 0.5625
```

All four lenses share the implemented world position
`(0.13412166, -39.63250069, 13.84474775)`. This is the decomposed calibrated
position corresponding approximately to the surveyed position
`(0, -39.6, 13.8)`.

## Camera pose and pinhole render

Each camera first renders an ideal pinhole image. The matrices below are
camera-to-world rotations passed to `Matrix3`; the camera's local `-Z` axis is
the optical axis expected by the OpenGL renderer.

`STATIC_SIDE_0` / `cam7_0`:

```text
 0.43947573  -0.01746386   0.89808468
 0.72682648   0.59439877  -0.34411243
-0.52781090   0.80398079   0.27391703
```

`STATIC_SIDE_1` / `cam7_1`:

```text
 0.93237555  -0.02762154   0.36043430
 0.30722095   0.58599676  -0.74981536
-0.19050227   0.80984248   0.55485497
```

`STATIC_SIDE_2` / `cam7_2`:

```text
 0.92894101  -0.02717736  -0.36922891
-0.28441997   0.58606085  -0.75870809
 0.23701029   0.80981113   0.53668617
```

`STATIC_SIDE_3` / `cam7_3`:

```text
 0.43615526  -0.02779244  -0.89944215
-0.72026378   0.58838843  -0.36744952
 0.53943367   0.80810064   0.23661082
```

The pinhole render uses a wider vertical field of view than the physical lens.
Without this extra source area, the post-process warp would sample outside the
rendered image and lose pixels near the sensor corners. The scale relating the
physical and source projections is

```text
s = tan(physical_vertical_fov / 2) / tan(render_vertical_fov / 2)
```

The pose and source FOVs live in
[`match.cpp`](../../third_party/gfootball_engine/src/onthepitch/match.cpp).

## Lens parameters

The radial coefficients and aspect are shared by all four lenses:

```text
k1 = -0.31888890
k2 =  0.09000000
a  =  0.5625
```

Optical centers below use the calibration convention: normalized coordinates
with a top-left image origin.

| Camera | Stream | Optical center `(cx, cy)` | Physical vertical FOV | Render vertical FOV | `s` |
| --- | --- | --- | ---: | ---: | ---: |
| `STATIC_SIDE_0` | `cam7_0` | `(0.49440938, 0.50254971)` | `79.847 deg` | `99.40683 deg` | `0.70959521` |
| `STATIC_SIDE_1` | `cam7_1` | `(0.49967515, 0.50241035)` | `79.567 deg` | `98.76896 deg` | `0.71408503` |
| `STATIC_SIDE_2` | `cam7_2` | `(0.50138474, 0.50205743)` | `79.936 deg` | `99.10011 deg` | `0.71458701` |
| `STATIC_SIDE_3` | `cam7_3` | `(0.50848764, 0.50017411)` | `80.369 deg` | `100.25191 deg` | `0.70555054` |

The shared `LensCalibration` source of truth is in
[`utils.hpp`](../../third_party/gfootball_engine/src/utils.hpp) and
[`utils.cpp`](../../third_party/gfootball_engine/src/utils.cpp). Cameras without
an entry receive a disabled calibration, making the transform an identity.

## GPU inverse warp

The post-process shader performs inverse mapping: for every pixel in the final
distorted image, it finds the corresponding coordinate in the wider pinhole
render. Inverse mapping fills every destination pixel and avoids the holes that
a forward image warp would create.

Let `d` be a normalized destination coordinate, `c` the optical center, and
`delta = d - c`. The aspect-corrected destination radius is

```text
normalization = 4 / (a^2 + 1)
rd = sqrt((delta.x^2 * a^2 + delta.y^2) * normalization)
```

The calibrated radial model is

```text
rd = r * (1 + k1 * r^2 + k2 * r^4)
```

The shader inverts it with twelve fixed-point iterations:

```text
r0 = rd
r(n+1) = rd / (1 + k1 * r(n)^2 + k2 * r(n)^4)
```

It then samples the pinhole render at

```text
u = (0.5, 0.5) + delta * (r / rd) * s
```

At the optical center (`rd < 1e-6`), `u` is defined as `(0.5, 0.5)`.

Calibration centers use a top-left origin, while OpenGL texture coordinates use
a bottom-left origin. Before setting the shader uniform, C++ therefore converts
the center to `(cx, 1 - cy)`. The implementation is
[`postprocess.frag`](../../third_party/gfootball_engine/data/media/shaders/postprocess.frag),
and the uniforms are populated in
[`graphics_camera.cpp`](../../third_party/gfootball_engine/src/systems/graphics/objects/graphics_camera.cpp).

## Segmentation and ball-coordinate alignment

Player segmentation is rendered into the existing off-screen g-buffer before
the normal RGB geometry pass. It is then drawn through the same post-process
shader and lens uniforms as RGB. In segmentation mode the shader:

- skips lighting, fog, contrast, and color-space processing;
- uses `texelFetch` with an integer source pixel;
- clamps the lookup to the source texture; and
- uses nearest-neighbor semantics so background `0` and player instance IDs
  remain categorical values.

The regular RGB pass clears and reuses the g-buffer afterward, so this does not
require another full-resolution render target.

`ball_screen_position` starts as a normalized pinhole coordinate with a
top-left origin. Its CPU transform is the analytical forward counterpart of the
shader lookup. For source offset `q = u - (0.5, 0.5)`, compute

```text
source_radius = sqrt((q.x^2 * a^2 + q.y^2) * normalization)
r = source_radius / s
radial_scale = 1 + k1 * r^2 + k2 * r^4
d = c + q * radial_scale / s
```

Visibility remains true only when the original projection is visible and the
distorted result is inside `[0, 1)` on both axes. This keeps dumped ball
coordinates aligned with the final RGB and segmentation frames.

## Adding another calibrated camera

1. Obtain the native frame dimensions, radial coefficients, optical center,
   physical vertical FOV, camera position, and camera-to-world rotation.
2. Add the camera type and its Python mapping using the existing
   `STATIC_SIDE_*` cameras as the compatibility pattern.
3. Configure the static position, rotation, near/far caps, and a wider source
   FOV in `Match::UpdateIngameCamera()`.
4. Compute `s` from the physical and render vertical FOVs using the tangent
   ratio above, then add one `LensCalibration` entry.
5. Keep calibration centers in top-left coordinates. Let the renderer perform
   the OpenGL Y conversion; do not pre-invert the stored value.
6. Add a portrait configuration under [`gfootball/configs`](../configs) with
   `width / height` equal to the calibrated sensor aspect. Native `cam7` output
   is `2160 x 3840`; `1080 x 1920` and `540 x 960` preserve the same geometry.
7. Set `video_quality_level` to `1` or `2` when MP4 output must retain the
   configured resolution. Level `0` resizes recordings to `800 x 450`.
8. Render a deterministic scene and compare it with the real reference at the
   center, sidelines, and corners. Check field lines and stadium structure, not
   only the players.
9. Overlay the instance mask on RGB and verify silhouettes at the image edges.
   Confirm all label channels are equal and contain only valid IDs.
10. Check `ball_screen_position` against the rendered ball and verify that
    visibility becomes false for coordinates outside the distorted viewport.

The calibrated-camera regression is
`FootballEnvTest.test_static_side_segmentation_matches_render_output` in
[`football_env_test.py`](../env/football_env_test.py).
