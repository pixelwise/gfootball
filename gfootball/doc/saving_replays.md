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
   `webm` (VP8), and `mp4` (MPEG-4 Part 2). The selected format also applies to
   semantic and instance segmentation videos.
-  `video_quality_level` - video quality from `0` (low) through `2` (high). Low
   quality limits output to 800x450. Medium and high retain the configured render
   resolution. AVI also selects a progressively higher-quality codec; WebM and
   MP4 use one codec at every level, so levels `1` and `2` have the same encoding
   settings for those formats.
-  `write_segmentation_video` - should a companion video be recorded with player
   pixels in white and all other pixels in black. The file is named
   `<dump-name>_segmentation.<video_format>` and requires rendering to be enabled.
   Segmentation videos are lossless when AVI is selected. MP4 and WebM use lossy
   codecs and should be treated as visualizations rather than exact label data.
-  `write_instance_segmentation_video` - should a companion video be recorded
   containing player instance labels. It is named
   `<dump-name>_instances.<video_format>`. AVI preserves the labels losslessly;
   MP4 and WebM do not guarantee exact label values after decoding.

There are following scripts provided to operate on trace dumps:

-  `dump_to_txt.py` - converts trace dump to human-readable form.
-  `dump_to_video.py` - converts trace dump to a 2D representation video.
-  `replay.py` - replays a given trace dump using environment.

## Environment logs
Environment uses `absl.logging` module for logging.
You can change logging level by setting --verbosity flag to one of the following values:

-  `-1` - warning, only warnings and above are logged when problems are encountered,
-  `0` - info (the default), some per-episode statistics and similar are logged as well,
-  `1` - debug, additional debugging messages are included.
