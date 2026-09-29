# coding=utf-8
"""Tests for calibrated genptzvideo recording manifests."""

import json
from pathlib import Path
import tempfile
from unittest import mock

from absl.testing import absltest

from gfootball.env import genptz_manifest


class GenptzManifestTest(absltest.TestCase):

  def _write_playlist(self, directory, camera, durations=(2.0, 0.1),
                      dump_prefix='episode_done_test'):
    stem = '%s_%s' % (dump_prefix, camera)
    playlist = Path(directory, stem + '.m3u8')
    segment = Path(directory, stem + '_segments.ts')
    segment.write_bytes(b'transport-stream')
    lines = [
        '#EXTM3U',
        '#EXT-X-PLAYLIST-TYPE:VOD',
    ]
    for duration in durations:
      lines.extend([
          '#EXTINF:%.6f,' % duration,
          segment.name,
      ])
    lines.append('#EXT-X-ENDLIST')
    playlist.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return str(playlist)

  def _videos(self, directory, durations=(2.0, 0.1),
              dump_prefix='episode_done_test'):
    # Deliberately reverse insertion order: calibration is keyed by identity.
    return {
        camera: self._write_playlist(
            directory, camera, durations, dump_prefix)
        for camera in reversed(genptz_manifest.CALIBRATED_CAMERAS)
    }

  def test_writes_manifest_with_calibration_and_render_metadata(self):
    with tempfile.TemporaryDirectory() as directory:
      dump_name = str(Path(directory, 'episode_done_test'))
      result = genptz_manifest.write_genptz_manifest(
          dump_name, self._videos(directory), (540, 960), 10.0)

      self.assertEqual(str(Path(directory, 'recording.json')), result)
      with open(result, encoding='utf-8') as manifest_file:
        manifest = json.load(manifest_file)

      expected_names = [
          'episode_done_test_%s' % camera
          for camera in genptz_manifest.CALIBRATED_CAMERAS
      ]
      self.assertEqual(2.1, manifest['duration_seconds'])
      self.assertEqual([], manifest['tracking_streams'])
      self.assertEqual([], manifest['audio_streams'])
      self.assertEqual([], manifest['tags'])
      physical_streams = manifest['streams'][:4]
      self.assertEqual(expected_names,
                       [stream['name'] for stream in physical_streams])
      for stream in physical_streams:
        self.assertEqual(10.0, stream['fps'])
        self.assertEqual(540, stream['width'])
        self.assertEqual(960, stream['height'])
        self.assertEqual(0, stream['offset_seconds'])
      preview_stream = manifest['streams'][4]
      self.assertEqual('cam7_sub', preview_stream['name'])
      self.assertEqual(expected_names, preview_stream['concatenated_streams'])
      self.assertIsNone(preview_stream['fps'])
      self.assertIsNone(preview_stream['width'])
      self.assertIsNone(preview_stream['height'])
      self.assertIsNone(preview_stream['offset_seconds'])

      view = manifest['views'][0]
      self.assertEqual(genptz_manifest.VIEW_ID, view['id'])
      self.assertEqual(expected_names, view['streams'])
      self.assertFalse(view['has_color_correction'])
      self.assertEqual('cam7_sub', view['preview_stream'])
      self.assertEqual(expected_names,
                       [camera['name'] for camera in view['cameras']])
      self.assertEqual(
          [0.49440938234329224, 0.4996751546859741,
           0.5013847351074219, 0.5084876418113708],
          [camera['lens']['cx'] for camera in view['cameras']])
      self.assertEqual(
          2.1297607421875, view['cameras'][0]['projection'][0])
      for camera in view['cameras']:
        self.assertEqual(540, camera['width'])
        self.assertEqual(960, camera['height'])

  def test_rejects_camera_durations_more_than_one_frame_apart(self):
    with tempfile.TemporaryDirectory() as directory:
      dump_name = str(Path(directory, 'episode_done_test'))
      videos = self._videos(directory)
      videos['static-side-3'] = self._write_playlist(
          directory, 'static-side-3', (2.0, 0.3))

      with self.assertRaisesRegex(RuntimeError, 'more than one frame'):
        genptz_manifest.write_genptz_manifest(
            dump_name, videos, (540, 960), 10.0)

      self.assertFalse(Path(directory, 'recording.json').exists())

  def test_requires_exact_camera_set_and_m3u8_outputs(self):
    with tempfile.TemporaryDirectory() as directory:
      dump_name = str(Path(directory, 'episode_done_test'))
      videos = self._videos(directory)
      videos.pop('static-side-3')
      self.assertIsNone(genptz_manifest.write_genptz_manifest(
          dump_name, videos, (540, 960), 10.0))

      videos = self._videos(directory)
      videos['static-side-3'] = str(Path(directory, 'camera.mp4'))
      self.assertIsNone(genptz_manifest.write_genptz_manifest(
          dump_name, videos, (540, 960), 10.0))

  def test_missing_segment_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      dump_name = str(Path(directory, 'episode_done_test'))
      videos = self._videos(directory)
      missing_segment = Path(directory,
                             'episode_done_test_static-side-2_segments.ts')
      missing_segment.unlink()

      with self.assertRaisesRegex(RuntimeError, 'Missing HLS media'):
        genptz_manifest.write_genptz_manifest(
            dump_name, videos, (540, 960), 10.0)

  def test_atomic_publish_failure_leaves_no_manifest_or_temporary_file(self):
    with tempfile.TemporaryDirectory() as directory:
      dump_name = str(Path(directory, 'episode_done_test'))
      with mock.patch.object(
          genptz_manifest.os, 'replace', side_effect=OSError('publish failed')):
        with self.assertRaisesRegex(OSError, 'publish failed'):
          genptz_manifest.write_genptz_manifest(
              dump_name, self._videos(directory), (540, 960), 10.0)

      self.assertFalse(Path(directory, 'recording.json').exists())
      self.assertFalse(list(Path(directory).glob('.*recording.json.*.tmp')))

  def test_later_dump_replaces_directory_manifest(self):
    with tempfile.TemporaryDirectory() as directory:
      first_dump = str(Path(directory, 'episode_done_first'))
      first_result = genptz_manifest.write_genptz_manifest(
          first_dump, self._videos(directory), (540, 960), 10.0)
      second_dump = str(Path(directory, 'episode_done_second'))
      second_result = genptz_manifest.write_genptz_manifest(
          second_dump,
          self._videos(directory, dump_prefix='episode_done_second'),
          (720, 1280), 20.0)

      expected_path = str(Path(directory, 'recording.json'))
      self.assertEqual(expected_path, first_result)
      self.assertEqual(expected_path, second_result)
      with open(second_result, encoding='utf-8') as manifest_file:
        manifest = json.load(manifest_file)
      expected_names = [
          'episode_done_second_static-side-%d' % index
          for index in range(4)
      ]
      self.assertEqual(expected_names, manifest['views'][0]['streams'])
      self.assertEqual(expected_names,
                       manifest['streams'][4]['concatenated_streams'])
      self.assertEqual(720, manifest['streams'][0]['width'])
      self.assertEqual(20.0, manifest['streams'][0]['fps'])
      self.assertFalse(Path(first_dump + '_recording.json').exists())
      self.assertFalse(Path(second_dump + '_recording.json').exists())

  def test_calibration_template_is_packaged_source_data(self):
    self.assertTrue(genptz_manifest._TEMPLATE_PATH.is_file())


if __name__ == '__main__':
  absltest.main()
