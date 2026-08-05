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
