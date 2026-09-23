# coding=utf-8
"""Tests for observation dump processing."""

from absl.testing import absltest
import cv2
import numpy as np
import os
import pickle
from pathlib import Path
import tempfile
from unittest import mock, skipUnless

from gfootball.env import config
from gfootball.env import observation_processor


class ObservationProcessorTest(absltest.TestCase):

  def test_resize_segmentation_frame_stays_binary(self):
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0] = 255

    resized = observation_processor.resize_segmentation_frame(frame, (7, 5))

    self.assertEqual((5, 7, 3), resized.shape)
    self.assertEqual({0, 255}, set(np.unique(resized)))
    self.assertTrue(np.any(np.all(resized == 255, axis=2)))
    self.assertTrue(np.any(np.all(resized == 0, axis=2)))

  def test_segmentation_is_disabled_by_default(self):
    self.assertFalse(config.Config()['write_segmentation_video'])

  def test_instance_segmentation_is_disabled_by_default(self):
    self.assertFalse(config.Config()['write_instance_segmentation_video'])

  def test_resize_instance_segmentation_preserves_labels(self):
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0] = 3
    frame[0, 1] = 17
    frame[1, 0] = 129

    resized = observation_processor.resize_instance_segmentation_frame(
        frame, (8, 6))

    self.assertEqual((6, 8, 3), resized.shape)
    self.assertEqual({0, 3, 17, 129}, set(np.unique(resized)))

  def test_ball_coordinates_are_disabled_by_default(self):
    self.assertFalse(config.Config()['write_ball_coordinates'])

  def test_active_dump_writes_synchronized_ball_coordinates(self):
    with tempfile.TemporaryDirectory() as directory:
      dump_config = config.Config({
          'display_game_stats': False,
          'render_resolution_x': 8,
          'render_resolution_y': 6,
          'video_quality_level': 2,
          'write_video': True,
          'write_ball_coordinates': True,
      })
      name = os.path.join(directory, 'episode_done_test')
      active_dump = observation_processor.ActiveDump(name, 2, dump_config)
      frame = np.zeros((6, 8, 3), dtype=np.uint8)

      active_dump.add_frame(
          frame, ball_screen_position=(0.25, 0.5),
          ball_screen_visible=True, engine_step=7)
      active_dump.add_frame(
          frame, ball_screen_position=(0.75, 0.25),
          ball_screen_visible=False, engine_step=8)
      dump_info = active_dump.finalize()

      self.assertEqual(name + '_ball.npz', dump_info['ball_coordinates'])
      video = cv2.VideoCapture(dump_info['video'])
      video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
      video.release()
      with np.load(dump_info['ball_coordinates']) as ball_data:
        np.testing.assert_allclose(ball_data['xy'], [[2, 3], [6, 1.5]])
        np.testing.assert_array_equal(ball_data['visible'], [True, False])
        np.testing.assert_array_equal(ball_data['engine_step'], [7, 8])
        np.testing.assert_array_equal(ball_data['frame_size'], [8, 6])
        self.assertEqual(video_frames, len(ball_data['xy']))

  def test_multi_camera_dump_writes_qualified_artifacts(self):
    with tempfile.TemporaryDirectory() as directory:
      cameras = ['static-side-0', 'static-side-1']
      dump_config = config.Config({
          'cameras': cameras,
          'display_game_stats': False,
          'render_resolution_x': 8,
          'render_resolution_y': 6,
          'video_quality_level': 2,
          'write_video': True,
          'write_ball_coordinates': True,
      })
      name = os.path.join(directory, 'episode_done_test')
      active_dump = observation_processor.MultiCameraActiveDump(
          name, 2, dump_config)
      frames = {
          camera: np.full((6, 8, 3), index * 50, dtype=np.uint8)
          for index, camera in enumerate(cameras)
      }
      state = observation_processor.ObservationState({
          'debug': {},
          'observation': {
              'camera_frames': frames,
              'camera_ball_screen_position': {
                  camera: np.array([0.25 + 0.5 * index, 0.5])
                  for index, camera in enumerate(cameras)
              },
              'camera_ball_screen_visible': {
                  camera: True for camera in cameras
              },
              'camera_engine_step': {
                  camera: 17 for camera in cameras
              },
          },
      })

      active_dump.add_step(state)
      dump_info = active_dump.finalize()

      expected_videos = {
          camera: name + '_' + camera + '.avi' for camera in cameras
      }
      self.assertEqual(expected_videos, dump_info['videos'])
      self.assertEqual(expected_videos[cameras[0]], dump_info['video'])
      for video_name in expected_videos.values():
        video = cv2.VideoCapture(video_name)
        self.assertTrue(video.isOpened())
        self.assertEqual(1, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
        video.release()

      expected_ball_files = {
          camera: name + '_' + camera + '_ball.npz' for camera in cameras
      }
      self.assertEqual(
          expected_ball_files, dump_info['ball_coordinates_by_camera'])
      self.assertEqual(
          expected_ball_files[cameras[0]], dump_info['ball_coordinates'])
      with open(dump_info['dump'], 'rb') as dump_file:
        stored_trace = pickle.load(dump_file)
      forbidden = {
          'frame', 'segmentation_frame', 'camera_frames',
          'camera_segmentation_frames', 'camera_ball_screen_position',
          'camera_ball_screen_visible', 'camera_engine_step',
      }
      self.assertFalse(
          forbidden.intersection(stored_trace['observation']))

  def test_multi_camera_pixels_are_transient_and_not_pickled(self):
    dump_config = config.Config({
        'cameras': ['static-side-0', 'static-side-1'],
    })
    processor = observation_processor.ObservationProcessor(dump_config)
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    trace = {
        'debug': {},
        'observation': {
            'frame': frame,
            'segmentation_frame': frame,
            'camera_frames': {
                'static-side-0': frame,
                'static-side-1': frame,
            },
            'camera_segmentation_frames': {
                'static-side-0': frame,
                'static-side-1': frame,
            },
            'camera_ball_screen_position': {
                'static-side-0': np.zeros(2),
                'static-side-1': np.zeros(2),
            },
            'camera_ball_screen_visible': {
                'static-side-0': False,
                'static-side-1': False,
            },
            'camera_engine_step': {
                'static-side-0': 3,
                'static-side-1': 3,
            },
        },
    }

    processor.update(trace)

    durable_observation = processor[-1]._trace['observation']
    forbidden = {
        'frame', 'segmentation_frame', 'camera_frames',
        'camera_segmentation_frames', 'camera_ball_screen_position',
        'camera_ball_screen_visible', 'camera_engine_step',
    }
    self.assertFalse(forbidden.intersection(durable_observation))
    self.assertIsNotNone(processor._transient_camera_state)

  def test_active_dump_writes_mp4_video(self):
    with tempfile.TemporaryDirectory() as directory:
      dump_config = config.Config({
          'render_resolution_x': 32,
          'render_resolution_y': 24,
          'video_format': 'mp4',
          'video_quality_level': 2,
          'write_video': True,
      })
      name = os.path.join(directory, 'episode_done_test')
      active_dump = observation_processor.ActiveDump(name, 2, dump_config)
      frame = np.zeros((24, 32, 3), dtype=np.uint8)

      active_dump.add_frame(frame)
      active_dump.add_frame(frame)
      dump_info = active_dump.finalize()

      self.assertEqual(name + '.mp4', dump_info['video'])
      video = cv2.VideoCapture(dump_info['video'])
      self.assertTrue(video.isOpened())
      self.assertEqual(2, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
      ok, decoded_frame = video.read()
      video.release()
      self.assertTrue(ok)
      self.assertEqual((24, 32, 3), decoded_frame.shape)

  def test_multi_camera_m3u8_writes_one_pair_per_camera(self):
    if not observation_processor.shutil.which('ffmpeg'):
      self.skipTest('ffmpeg is required for HLS integration tests')
    with tempfile.TemporaryDirectory() as directory:
      cameras = ['static-side-0', 'static-side-1']
      dump_config = config.Config({
          'cameras': cameras,
          'display_game_stats': False,
          'render_resolution_x': 32,
          'render_resolution_y': 24,
          'physics_steps_per_frame': 10,
          'video_format': 'm3u8',
          'video_quality_level': 2,
          'write_video': True,
      })
      name = os.path.join(directory, 'episode_done_test')
      active_dump = observation_processor.MultiCameraActiveDump(
          name, 25, dump_config)
      for frame_index in range(25):
        frames = {
            camera: np.full(
                (24, 32, 3), frame_index * 5 + camera_index,
                dtype=np.uint8)
            for camera_index, camera in enumerate(cameras)
        }
        active_dump.add_frame(frames, engine_step={
            camera: frame_index for camera in cameras
        })

      dump_info = active_dump.finalize()

      for camera in cameras:
        playlist = name + '_' + camera + '.m3u8'
        segment = name + '_' + camera + '_segments.ts'
        self.assertEqual(playlist, dump_info['videos'][camera])
        self.assertTrue(os.path.isfile(playlist))
        self.assertTrue(os.path.isfile(segment))
        with open(playlist) as playlist_file:
          playlist_text = playlist_file.read()
        self.assertIn(os.path.basename(segment), playlist_text)
        self.assertIn('#EXT-X-BYTERANGE:', playlist_text)
        self.assertIn('#EXT-X-ENDLIST', playlist_text)
      self.assertEqual(dump_info['videos'][cameras[0]], dump_info['video'])
      self.assertEqual(2, len([
          path for path in os.listdir(directory) if path.endswith('.m3u8')
      ]))
      self.assertEqual(2, len([
          path for path in os.listdir(directory) if path.endswith('.ts')
      ]))

  @skipUnless(observation_processor.shutil.which('ffmpeg'),
              'ffmpeg is required for HLS integration tests')
  def test_active_dump_writes_single_file_hls_video(self):
    with tempfile.TemporaryDirectory() as directory:
      dump_config = config.Config({
          'render_resolution_x': 32,
          'render_resolution_y': 24,
          'physics_steps_per_frame': 10,
          'video_format': 'm3u8',
          'video_quality_level': 2,
          'write_video': True,
          'write_ball_coordinates': True,
      })
      name = os.path.join(directory, 'episode_done_test')
      active_dump = observation_processor.ActiveDump(name, 25, dump_config)
      for frame_index in range(25):
        frame = np.full((24, 32, 3), frame_index * 5, dtype=np.uint8)
        active_dump.add_frame(
            frame, ball_screen_position=(0.25, 0.5),
            ball_screen_visible=True, engine_step=frame_index)

      dump_info = active_dump.finalize()
      playlist = Path(name + '.m3u8')
      segment = Path(name + '_segments.ts')

      self.assertEqual(str(playlist), dump_info['video'])
      self.assertTrue(playlist.is_file())
      self.assertTrue(segment.is_file())
      self.assertEqual([playlist], list(Path(directory).glob('*.m3u8')))
      self.assertEqual([segment], list(Path(directory).glob('*.ts')))
      playlist_text = playlist.read_text()
      self.assertIn('#EXT-X-PLAYLIST-TYPE:VOD', playlist_text)
      self.assertIn('#EXT-X-BYTERANGE:', playlist_text)
      self.assertIn('#EXTINF:2.000000,', playlist_text)
      self.assertIn('episode_done_test_segments.ts', playlist_text)
      self.assertIn('#EXT-X-ENDLIST', playlist_text)
      self.assertGreaterEqual(playlist_text.count('#EXTINF:'), 2)

      video = cv2.VideoCapture(str(playlist))
      self.assertTrue(video.isOpened())
      self.assertEqual(25, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
      self.assertEqual(10, int(round(video.get(cv2.CAP_PROP_FPS))))
      ok, decoded_frame = video.read()
      video.release()
      self.assertTrue(ok)
      self.assertEqual((24, 32, 3), decoded_frame.shape)

      with np.load(dump_info['ball_coordinates']) as ball_data:
        self.assertEqual(25, len(ball_data['xy']))
        np.testing.assert_array_equal(ball_data['engine_step'], range(25))

  def test_m3u8_masks_fall_back_to_mp4_without_ffmpeg(self):
    with tempfile.TemporaryDirectory() as directory:
      dump_config = config.Config({
          'render_resolution_x': 32,
          'render_resolution_y': 24,
          'video_format': 'm3u8',
          'video_quality_level': 2,
          'write_video': False,
          'write_segmentation_video': True,
          'write_instance_segmentation_video': True,
      })
      name = os.path.join(directory, 'episode_done_test')
      with mock.patch.object(observation_processor.shutil, 'which',
                             return_value=None):
        active_dump = observation_processor.ActiveDump(name, 2, dump_config)
      labels = np.zeros((24, 32, 3), dtype=np.uint8)
      labels[:, :16] = 17
      active_dump.add_frame(np.zeros_like(labels), labels, engine_step=10)
      active_dump.add_frame(np.zeros_like(labels), labels, engine_step=20)

      dump_info = active_dump.finalize()

      expected_videos = {
          'segmentation_video': name + '_segmentation.mp4',
          'instance_segmentation_video': name + '_instances.mp4',
      }
      for key, expected_name in expected_videos.items():
        self.assertEqual(expected_name, dump_info[key])
        video = cv2.VideoCapture(expected_name)
        self.assertTrue(video.isOpened())
        self.assertEqual(2, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
        video.release()
      self.assertFalse(list(Path(directory).glob('*.m3u8')))
      self.assertFalse(list(Path(directory).glob('*.ts')))

  def test_unknown_video_format_is_rejected_at_runtime(self):
    with tempfile.TemporaryDirectory() as directory:
      dump_config = config.Config({
          'render_resolution_x': 32,
          'render_resolution_y': 24,
          'video_format': 'mov',
          'write_video': True,
      })
      with self.assertRaisesRegex(ValueError, 'Unsupported video format: mov'):
        observation_processor.ActiveDump(
            os.path.join(directory, 'episode_done_test'), 1, dump_config)

  def test_m3u8_requires_ffmpeg_for_rgb_video(self):
    with tempfile.TemporaryDirectory() as directory:
      dump_config = config.Config({
          'render_resolution_x': 32,
          'render_resolution_y': 24,
          'video_format': 'm3u8',
          'video_quality_level': 2,
          'write_video': True,
      })
      with mock.patch.object(observation_processor.shutil, 'which',
                             return_value=None):
        with self.assertRaisesRegex(RuntimeError, 'ffmpeg was not found'):
          observation_processor.ActiveDump(
              os.path.join(directory, 'episode_done_test'), 1, dump_config)

  def test_m3u8_rejects_odd_rgb_dimensions(self):
    with tempfile.TemporaryDirectory() as directory:
      dump_config = config.Config({
          'render_resolution_x': 31,
          'render_resolution_y': 24,
          'video_format': 'm3u8',
          'video_quality_level': 2,
          'write_video': True,
      })
      with self.assertRaisesRegex(ValueError, 'even render dimensions'):
        observation_processor.ActiveDump(
            os.path.join(directory, 'episode_done_test'), 1, dump_config)

  def test_m3u8_encoder_failure_is_reported_and_staging_is_removed(self):
    with tempfile.TemporaryDirectory() as directory:
      playlist = os.path.join(directory, 'episode_done_test.m3u8')
      process = mock.Mock()
      process.wait.return_value = 1
      with mock.patch.object(observation_processor.shutil, 'which',
                             return_value='/usr/bin/ffmpeg'):
        with mock.patch.object(observation_processor.subprocess, 'Popen',
                               return_value=process):
          writer = observation_processor._HlsVideoWriter(
              playlist, (32, 24), 10, 2)
      writer.write(np.zeros((24, 32, 3), dtype=np.uint8))

      with self.assertRaisesRegex(
          RuntimeError, 'FFmpeg failed to finalize.*exit code 1'):
        writer.release()

      process.stdin.close.assert_called_once_with()
      self.assertFalse(Path(playlist).exists())
      self.assertFalse(
          Path(directory, 'episode_done_test_segments.ts').exists())
      self.assertFalse(list(Path(directory).glob('.gfootball-hls-*')))

  @skipUnless(observation_processor.shutil.which('ffmpeg'),
              'ffmpeg is required for HLS integration tests')
  def test_empty_m3u8_dump_cleans_up_and_finalize_is_idempotent(self):
    with tempfile.TemporaryDirectory() as directory:
      dump_config = config.Config({
          'render_resolution_x': 32,
          'render_resolution_y': 24,
          'video_format': 'm3u8',
          'video_quality_level': 2,
          'write_video': True,
      })
      name = os.path.join(directory, 'episode_done_test')
      active_dump = observation_processor.ActiveDump(name, 0, dump_config)

      self.assertNotIn('video', active_dump.finalize())
      self.assertEqual({}, active_dump.finalize())
      self.assertFalse(Path(name + '.m3u8').exists())
      self.assertFalse(Path(name + '_segments.ts').exists())
      self.assertFalse(list(Path(directory).glob('.gfootball-hls-*')))

  def test_mp4_format_applies_to_segmentation_videos(self):
    with tempfile.TemporaryDirectory() as directory:
      dump_config = config.Config({
          'render_resolution_x': 32,
          'render_resolution_y': 24,
          'video_format': 'mp4',
          'video_quality_level': 2,
          'write_segmentation_video': True,
          'write_instance_segmentation_video': True,
      })
      name = os.path.join(directory, 'episode_done_test')
      active_dump = observation_processor.ActiveDump(name, 2, dump_config)
      labels = np.zeros((24, 32, 3), dtype=np.uint8)
      labels[:, :16] = 17

      active_dump.add_frame(np.zeros_like(labels), labels, engine_step=10)
      active_dump.add_frame(np.zeros_like(labels), labels, engine_step=20)
      dump_info = active_dump.finalize()

      expected_videos = {
          'segmentation_video': name + '_segmentation.mp4',
          'instance_segmentation_video': name + '_instances.mp4',
      }
      for key, expected_name in expected_videos.items():
        self.assertEqual(expected_name, dump_info[key])
        video = cv2.VideoCapture(expected_name)
        self.assertTrue(video.isOpened())
        self.assertEqual(2, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
        ok, decoded_frame = video.read()
        video.release()
        self.assertTrue(ok)
        self.assertEqual((24, 32, 3), decoded_frame.shape)

  def test_active_dump_writes_lossless_segmentation_video(self):
    with tempfile.TemporaryDirectory() as directory:
      dump_config = config.Config({
          'render_resolution_x': 8,
          'render_resolution_y': 6,
          'video_quality_level': 2,
          'write_segmentation_video': True,
      })
      name = os.path.join(directory, 'episode_done_test')
      active_dump = observation_processor.ActiveDump(name, 2, dump_config)
      mask = np.zeros((6, 8, 3), dtype=np.uint8)
      # Instance labels are converted to a binary semantic mask.
      mask[:, :4] = 7

      active_dump.add_frame(np.zeros_like(mask), mask)
      active_dump.add_frame(np.zeros_like(mask), mask)
      dump_info = active_dump.finalize()

      self.assertEqual(name + '_segmentation.avi',
                       dump_info['segmentation_video'])
      video = cv2.VideoCapture(dump_info['segmentation_video'])
      self.assertEqual(2, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
      ok, decoded_mask = video.read()
      video.release()
      self.assertTrue(ok)
      self.assertEqual({0, 255}, set(np.unique(decoded_mask)))

  def test_active_dump_writes_lossless_instance_video_and_metadata(self):
    with tempfile.TemporaryDirectory() as directory:
      dump_config = config.Config({
          'render_resolution_x': 8,
          'render_resolution_y': 6,
          'video_quality_level': 2,
          'write_instance_segmentation_video': True,
      })
      name = os.path.join(directory, 'episode_done_test')
      active_dump = observation_processor.ActiveDump(name, 2, dump_config)
      active_dump._capture_instance_metadata({
          'left_team_instance_id': np.array([1, 3], dtype=np.uint8),
          'right_team_instance_id': np.array([17], dtype=np.uint8),
      })
      # IDs first observed later (for example, after a substitution) are added.
      active_dump._capture_instance_metadata({
          'left_team_instance_id': np.array([1, 3], dtype=np.uint8),
          'right_team_instance_id': np.array([17, 18], dtype=np.uint8),
      })
      labels = np.zeros((6, 8, 3), dtype=np.uint8)
      labels[:, :3] = 1
      labels[:, 3:6] = 3
      labels[:, 6:] = 17

      active_dump.add_frame(
          np.zeros_like(labels), labels, engine_step=10)
      active_dump.add_frame(
          np.zeros_like(labels), labels, engine_step=20)
      dump_info = active_dump.finalize()

      self.assertEqual(name + '_instances.avi',
                       dump_info['instance_segmentation_video'])
      self.assertEqual(name + '_instances.npz',
                       dump_info['instance_segmentation_metadata'])
      video = cv2.VideoCapture(dump_info['instance_segmentation_video'])
      self.assertEqual(2, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
      ok, decoded_labels = video.read()
      video.release()
      self.assertTrue(ok)
      self.assertEqual({1, 3, 17}, set(np.unique(decoded_labels)))
      np.testing.assert_array_equal(
          decoded_labels[:, :, 0], decoded_labels[:, :, 1])
      np.testing.assert_array_equal(
          decoded_labels[:, :, 1], decoded_labels[:, :, 2])
      with np.load(dump_info['instance_segmentation_metadata']) as metadata:
        np.testing.assert_array_equal(metadata['engine_step'], [10, 20])
        np.testing.assert_array_equal(
            metadata['instance_id'], [1, 3, 17, 18])
        np.testing.assert_array_equal(metadata['team'], [0, 0, 1, 1])
        np.testing.assert_array_equal(
            metadata['team_player_index'], [0, 1, 0, 1])
        np.testing.assert_array_equal(metadata['frame_size'], [8, 6])


if __name__ == '__main__':
  absltest.main()
