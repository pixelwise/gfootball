# coding=utf-8
"""Tests for observation dump processing."""

from absl.testing import absltest
import cv2
import numpy as np
import os
import tempfile

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
