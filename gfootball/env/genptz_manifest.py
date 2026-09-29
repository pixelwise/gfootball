# coding=utf-8
"""Builds local genptzvideo manifests for calibrated camera dumps."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from decimal import Decimal
import json
import os
from pathlib import Path
import tempfile


CALIBRATED_CAMERAS = (
    'static-side-0',
    'static-side-1',
    'static-side-2',
    'static-side-3',
)
_SOURCE_STREAMS = {
    'static-side-0': 'cam7_0',
    'static-side-1': 'cam7_1',
    'static-side-2': 'cam7_2',
    'static-side-3': 'cam7_3',
}
_PREVIEW_STREAM = 'cam7_sub'
VIEW_ID = '15e6e2e9-2ccc-4b99-f28b-1b80c60d66d7'
_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent / 'data' /
    'wolfsburg_cam7_recording.json')


def _playlist_duration(playlist_path):
  """Returns a finalized HLS duration after checking local media references."""
  playlist_path = Path(playlist_path)
  if not playlist_path.is_file():
    raise RuntimeError('Missing HLS playlist: %s' % playlist_path)
  duration = Decimal(0)
  media_paths = []
  finalized = False
  with playlist_path.open(encoding='utf-8') as playlist:
    for raw_line in playlist:
      line = raw_line.strip()
      if line.startswith('#EXTINF:'):
        duration += Decimal(line[len('#EXTINF:'):].split(',', 1)[0])
      elif line == '#EXT-X-ENDLIST':
        finalized = True
      elif line and not line.startswith('#'):
        media_paths.append(playlist_path.parent / line)
  if not finalized:
    raise RuntimeError('HLS playlist is not finalized: %s' % playlist_path)
  if duration <= 0:
    raise RuntimeError('HLS playlist contains no video duration: %s' %
                       playlist_path)
  if not media_paths:
    raise RuntimeError('HLS playlist contains no media references: %s' %
                       playlist_path)
  missing = sorted({str(path) for path in media_paths if not path.is_file()})
  if missing:
    raise RuntimeError('Missing HLS media file(s): %s' % ', '.join(missing))
  return duration


def _load_template():
  with _TEMPLATE_PATH.open(encoding='utf-8') as template_file:
    return json.load(template_file)


def _write_json_atomically(path, value):
  path = Path(path)
  fd, temporary_path = tempfile.mkstemp(
      prefix='.%s.' % path.name, suffix='.tmp', dir=str(path.parent))
  try:
    with os.fdopen(fd, 'w', encoding='utf-8') as output:
      json.dump(value, output, indent=2)
      output.write('\n')
      output.flush()
      os.fsync(output.fileno())
    os.replace(temporary_path, path)
  except Exception:
    if os.path.exists(temporary_path):
      os.remove(temporary_path)
    raise


def write_genptz_manifest(dump_name, videos, frame_dim, fps):
  """Writes a calibrated manifest for four finalized static-side HLS videos."""
  if set(videos) != set(CALIBRATED_CAMERAS):
    return None
  ordered_videos = {
      camera: Path(videos[camera]) for camera in CALIBRATED_CAMERAS
  }
  if any(video.suffix != '.m3u8' for video in ordered_videos.values()):
    return None

  durations = {
      camera: _playlist_duration(video)
      for camera, video in ordered_videos.items()
  }
  max_duration = max(durations.values())
  min_duration = min(durations.values())
  frame_duration = Decimal(1) / Decimal(str(fps))
  if max_duration - min_duration > frame_duration:
    raise RuntimeError(
        'Calibrated camera HLS durations differ by more than one frame: %s' %
        ', '.join('%s=%s' % item for item in durations.items()))

  manifest = _load_template()
  manifest['name'] = 'Synthetic GFootball episode %s' % Path(dump_name).name
  manifest['duration_seconds'] = float(min_duration)
  manifest['is_live'] = False
  manifest['tracking_streams'] = []
  manifest['audio_streams'] = []
  manifest['tags'] = []

  width, height = frame_dim
  streams_by_name = {stream['name']: stream for stream in manifest['streams']}
  cameras_by_name = {
      camera['name']: camera for camera in manifest['views'][0]['cameras']
  }
  output_streams = []
  output_cameras = []
  output_names = []
  for camera_name in CALIBRATED_CAMERAS:
    source_name = _SOURCE_STREAMS[camera_name]
    output_name = ordered_videos[camera_name].stem
    output_names.append(output_name)

    stream = streams_by_name[source_name]
    stream['name'] = output_name
    stream['friendly_name'] = 'Synthetic %s' % camera_name.upper()
    stream['fps'] = fps
    stream['width'] = width
    stream['height'] = height
    stream['offset_seconds'] = 0
    stream['path'] = None
    stream['bitrate'] = None
    output_streams.append(stream)

    camera = cameras_by_name[source_name]
    camera['name'] = output_name
    camera['width'] = width
    camera['height'] = height
    output_cameras.append(camera)

  preview_stream = streams_by_name[_PREVIEW_STREAM]
  preview_stream['concatenated_streams'] = output_names
  manifest['streams'] = output_streams + [preview_stream]
  view = manifest['views'][0]
  view['streams'] = output_names
  view['cameras'] = output_cameras
  view['preview_stream'] = _PREVIEW_STREAM
  view['has_color_correction'] = False

  manifest_path = Path(dump_name).parent / 'recording.json'
  _write_json_atomically(manifest_path, manifest)
  return str(manifest_path)
