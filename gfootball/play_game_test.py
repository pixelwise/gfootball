"""Tests for the interactive game runner."""

import unittest
import cv2
import numpy as np
import os
from pathlib import Path
import tempfile
import yaml

from pydantic import ValidationError

from gfootball import play_game


class FakeEnv(object):

  def __init__(self, results):
    self._results = iter(results)
    self.reset_calls = 0
    self.step_calls = 0

  def reset(self):
    self.reset_calls += 1

  def step(self, actions):
    self.assert_actions(actions)
    self.step_calls += 1
    return None, None, next(self._results), {}

  def assert_actions(self, actions):
    if actions != []:
      raise AssertionError('Expected no player actions.')


class FakeRenderEnv(object):

  def __init__(self):
    self.reset_calls = 0
    self.render_calls = []

  def reset(self):
    self.reset_calls += 1

  def render(self, mode):
    self.render_calls.append(mode)
    return np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8)


class PlayGameTest(unittest.TestCase):

  def test_one_episode_stops_without_another_reset(self):
    env = FakeEnv([False, True])

    play_game.run_episodes(env, episodes=1)

    self.assertEqual(1, env.reset_calls)
    self.assertEqual(2, env.step_calls)

  def test_multiple_episodes_reset_between_episodes(self):
    env = FakeEnv([True, True])

    play_game.run_episodes(env, episodes=2)

    self.assertEqual(2, env.reset_calls)
    self.assertEqual(2, env.step_calls)

  def test_episodes_must_not_be_negative(self):
    with self.assertRaises(ValidationError):
      play_game.GameConfig(episodes=-1)

  def test_mp4_is_a_valid_video_format(self):
    game_config = play_game.GameConfig(video_format='mp4')

    self.assertEqual('mp4', game_config.video_format)

  def test_unknown_video_format_is_rejected(self):
    with self.assertRaises(ValidationError):
      play_game.GameConfig(video_format='mov')

  def test_write_single_frame_is_loaded_from_yaml(self):
    with tempfile.NamedTemporaryFile(mode='w') as config_file:
      yaml.safe_dump({'write_single_frame': True}, config_file)
      config_file.flush()
      config = play_game.GameConfig.from_yaml(config_file.name)

    self.assertTrue(config.write_single_frame)

  def test_write_single_frame_writes_initial_render(self):
    env = FakeRenderEnv()
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'debug_frame.png'
      play_game.write_single_frame(env, output)

      frame = cv2.imread(str(output), cv2.IMREAD_COLOR)

    self.assertEqual(1, env.reset_calls)
    self.assertEqual(['rgb_array'], env.render_calls)
    np.testing.assert_array_equal(
        np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8), frame)


if __name__ == '__main__':
  unittest.main()
