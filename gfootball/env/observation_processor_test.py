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
      mask[:, :4] = 255

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


if __name__ == '__main__':
  absltest.main()
