"""Tests for the interactive game runner."""

import unittest

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


if __name__ == '__main__':
  unittest.main()
