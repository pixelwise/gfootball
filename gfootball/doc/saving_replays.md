# Saving replays, logs, traces #

GFootball environment supports recording of scenarios for later watching or
analysis. Each trace dump consists of a pickled episode trace (observations,
reward, additional debug info) and optionally a video with the rendered episode.
Pickled episode trace can be played back later on using `replay.py` script.
By default trace dumps are disabled to not occupy disk space. They
are controlled by the following set of flags:

-  `dump_full_episodes` - should trace for each entire episode be recorded.
-  `dump_scores` - should sample traces for scores be recorded.
-  `tracesdir` - directory in which trace dumps are saved.
-  `write_video` - should a video be recorded together with the trace.
    If rendering is disabled (`render` config flag), the video contains a simple
    episode animation.
-  `video_format` - container and codec used for the rendered episode. Supported
   values are `avi` (Xvid, Motion JPEG, or lossless PNG depending on quality),
   `webm` (VP8), `mp4` (MPEG-4 Part 2), and `m3u8` (H.264 HLS VOD). HLS creates
   `<dump-name>.m3u8` and one byte-range `<dump-name>_segments.ts` file.
   Semantic and instance segmentation videos use MP4 when HLS is selected.
   A four-camera `static-side-0` through `static-side-3` HLS dump also creates
   `<episode-directory>/recording.json`, which maps the generated streams to the
   Wolfsburg `cam7` calibration for `genptzvideo`. If several dumps share a
   directory, the latest successfully finalized compatible dump replaces this
   manifest atomically.
-  `video_quality_level` - video quality from `0` (low) through `2` (high). Low
   quality limits output to 800x450. Medium and high retain the configured render
   resolution. AVI also selects a progressively higher-quality codec; WebM and
   MP4 use one codec at every level, so levels `1` and `2` have the same encoding
   settings for those formats.
-  `write_segmentation_video` - should a companion video be recorded with player
   pixels in white and all other pixels in black. The file is named
   `<dump-name>_segmentation.<video_format>` and requires rendering to be enabled.
   With `video_format: m3u8`, the file is instead named
   `<dump-name>_segmentation.mp4`.
   Segmentation videos are lossless when AVI is selected. MP4 and WebM use lossy
   codecs and should be treated as visualizations rather than exact label data.
-  `write_instance_segmentation_video` - should a companion video be recorded
   containing player instance labels. It is named
   `<dump-name>_instances.<video_format>`. With `video_format: m3u8`, the file is
   instead named `<dump-name>_instances.mp4`. AVI preserves the labels losslessly;
   MP4 and WebM do not guarantee exact label values after decoding.

There are following scripts provided to operate on trace dumps:

-  `dump_to_txt.py` - converts trace dump to human-readable form.
-  `dump_to_video.py` - converts trace dump to a 2D representation video.
-  `replay.py` - replays a given trace dump using environment.

## Stitching calibrated camera dumps

Four-camera static-side M3U8 dumps can be rendered as a stitched overview with
`genptzvideo`. The command discovers `recording.json` in `--data-dir`; its
`cam7_sub` preview stream groups the four synchronized physical streams for
`--use-preview-input`:

```sh
/opt/gameon_core/bin/gameon_env genptzvideo \
  --data-dir ./dumps/episode-20260928-163051 \
  --view-id 15e6e2e9-2ccc-4b99-f28b-1b80c60d66d7 \
  --out ./dumps/episode-20260928-163051/overview.mp4 \
  --num-hardware-decoder 4 \
  --use-hardware-encoder \
  --fx 1.0 \
  --fy 1.0 \
  -w 1824 \
  -h 608 \
  --overview-margin 1.0 \
  --overview-headspace 2.0 \
  --pan-radians 0.008000001311302185 \
  --tilt-radians 0.40999913215637207 \
  --scale 0.4041883647441864 \
  --colorspace bt709 \
  --colorrange full \
  --vprofile high \
  --quality high \
  --vbr \
  --no-scoreboard \
  --with-lens true \
  --smooth false \
  --motion-blur false \
  --fps 25 \
  --gop-length 50 \
  --logo-opacity 0 \
  --logo-size 0 \
  --use-preview-input
```

The physical stream metadata retains the generated input frame rate (`100 /
physics_steps_per_frame`). The command's `--fps 25` independently selects the
stitched output frame rate. `--with-lens true` intentionally enables distortion
for the final virtual-camera output.

## Environment logs
Environment uses `absl.logging` module for logging.
You can change logging level by setting --verbosity flag to one of the following values:

-  `-1` - warning, only warnings and above are logged when problems are encountered,
-  `0` - info (the default), some per-episode statistics and similar are logged as well,
-  `1` - debug, additional debugging messages are included.
