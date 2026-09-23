# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Script allowing to play the game by multiple players."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from enum import Enum
from pathlib import Path
import sys
import time
from typing import Literal
from typing import Optional
from typing import TextIO

import tempfile
import yaml
import cv2

from absl import app
from absl import flags
from absl import logging
from pydantic import BaseModel
from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from gfootball.env import config
from gfootball.env import football_env
from gfootball.env.football_env_core import CameraType


class ActionSet(str, Enum):
    DEFAULT = "default"
    FULL = "full"


class Scenario(str, Enum):
  STANDARD = "11_vs_11_stochastic"
  EASY = "11_vs_11_easy_stochastic"
  HARD = "11_vs_11_hard_stochastic"
  ACADEMY_3_VS_1 = "academy_3_vs_1_with_keeper"
  ACADEMY_CORNER = "academy_corner"


class DisplaySettings(BaseModel):
  radar: bool = True
  scoreboard: bool = True
  player_names: bool = False


class GameConfig(BaseSettings):
  model_config = SettingsConfigDict(extra="ignore")

  action_set: ActionSet = ActionSet.DEFAULT
  players: str = ""
  level: Scenario = Scenario.STANDARD
  camera: CameraType = CameraType.WIDE
  custom_display_stats: Optional[str] = None
  display_game_stats: bool = True
  show_progress: bool = True
  episodes: int = Field(default=1, ge=0)
  dump_full_episodes: bool = True
  dump_scores: bool = False
  physics_steps_per_frame: int = 10
  render_resolution_x: int = 1280
  render_resolution_y: int = 720
  real_time: bool = False
  tracesdir: Path = Path(tempfile.gettempdir()) / "dumps"
  video_format: Literal["avi", "webm", "mp4", "m3u8"] = "avi"
  video_quality_level: int = 0
  write_video: bool = True
  write_segmentation_video: bool = True
  write_instance_segmentation_video: bool = False
  write_ball_coordinates: bool = False
  write_single_frame: bool = False
  game_engine_random_seed: int = 48
  display_settings: DisplaySettings = Field(default_factory=DisplaySettings)

  @classmethod
  def from_yaml(cls, path: str):
    with open(path) as f:
      data = yaml.safe_load(f)
    return cls(**data)

  def to_yaml(self, path: Path) -> None:
    data = self.model_dump(mode="json")
    with open(path, "w") as f:
      yaml.safe_dump(data, f, sort_keys=False)

  def with_dynamic_tracesdir(self) -> "GameConfig":
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dynamic_path = self.tracesdir / f"episode-{timestamp}"
    return self.model_copy(update={"tracesdir": dynamic_path})


FLAGS = flags.FLAGS

flags.DEFINE_string('config_file', None, 'Path to YAML configuration file')
flags.DEFINE_bool('render', True, 'Whether to do game rendering.')


def _format_duration(seconds: Optional[float]) -> str:
  """Formats a duration for compact progress output."""
  if seconds is None:
    return '--:--'
  seconds = max(0, int(seconds))
  minutes, seconds = divmod(seconds, 60)
  hours, minutes = divmod(minutes, 60)
  if hours:
    return f'{hours:d}:{minutes:02d}:{seconds:02d}'
  return f'{minutes:02d}:{seconds:02d}'


class _EpisodeProgress(object):
  """Dependency-free terminal progress reporting for episode generation."""

  _TTY_UPDATE_INTERVAL_SECONDS = 0.1
  _NON_TTY_UPDATE_INTERVAL_SECONDS = 5.0
  _BAR_WIDTH = 20

  def __init__(self, episodes: int, enabled: bool,
               stream: TextIO, clock=time.monotonic):
    self._episodes = episodes
    self._enabled = enabled
    self._stream = stream
    self._clock = clock
    self._is_tty = bool(getattr(stream, 'isatty', lambda: False)())
    self._episode = 0
    self._started_at = 0.0
    self._last_output_at = 0.0
    self._total_steps = None
    self._last_line_width = 0
    self._line_open = False

  def start_episode(self, episode: int, observation) -> None:
    if not self._enabled:
      return
    self._episode = episode
    self._started_at = self._clock()
    self._last_output_at = self._started_at
    self._total_steps = self._initial_total_steps(observation)
    self._write(observation, done=False, now=self._started_at)

  def update(self, observation, done: bool = False) -> None:
    if not self._enabled:
      return
    now = self._clock()
    interval = (self._TTY_UPDATE_INTERVAL_SECONDS if self._is_tty else
                self._NON_TTY_UPDATE_INTERVAL_SECONDS)
    if not done and now - self._last_output_at < interval:
      return
    self._write(observation, done=done, now=now)

  def close(self) -> None:
    """Terminates an active in-place progress line."""
    if self._enabled and self._is_tty and self._line_open:
      self._stream.write('\n')
      self._stream.flush()
      self._line_open = False

  def _initial_total_steps(self, observation) -> Optional[int]:
    observation = self._progress_observation(observation)
    if observation is None or 'steps_left' not in observation:
      return None
    try:
      steps_left = max(0, int(observation['steps_left']))
      engine_step = max(0, int(observation.get('engine_step', 0)))
    except (TypeError, ValueError):
      return None
    return steps_left + engine_step

  @staticmethod
  def _progress_observation(observation):
    # Player-controlled environments return one observation per player.
    if isinstance(observation, (list, tuple)) and observation:
      observation = observation[0]
    return observation if isinstance(observation, dict) else None

  def _completed_steps(self, observation) -> Optional[int]:
    observation = self._progress_observation(observation)
    if (self._total_steps is None or observation is None or
        'steps_left' not in observation):
      return None
    try:
      steps_left = max(0, int(observation['steps_left']))
    except (TypeError, ValueError):
      return None
    return min(self._total_steps, max(0, self._total_steps - steps_left))

  def _format_line(self, observation, done: bool, now: float) -> str:
    episode = (f'Episode {self._episode}/{self._episodes}' if self._episodes
               else f'Episode {self._episode}')
    elapsed = max(0.0, now - self._started_at)
    completed = self._completed_steps(observation)
    if self._total_steps is None or completed is None:
      fraction = 1.0 if done else 0.0
      step_text = '?/?'
      rate_text = '--.- steps/s'
      eta = 0.0 if done else None
    else:
      displayed_completed = self._total_steps if done else completed
      fraction = (displayed_completed / self._total_steps
                  if self._total_steps else float(done))
      step_text = f'{displayed_completed}/{self._total_steps}'
      rate = completed / elapsed if elapsed else 0.0
      rate_text = f'{rate:.1f} steps/s'
      eta = (0.0 if done else
             (self._total_steps - completed) / rate if rate else None)
    filled = min(self._BAR_WIDTH, int(fraction * self._BAR_WIDTH))
    bar = '#' * filled + '-' * (self._BAR_WIDTH - filled)
    return (f'{episode} [{bar}] {fraction * 100:5.1f}% | {step_text} | '
            f'elapsed {_format_duration(elapsed)} | {rate_text} | '
            f'ETA {_format_duration(eta)}')

  def _write(self, observation, done: bool, now: float) -> None:
    line = self._format_line(observation, done, now)
    if self._is_tty:
      padded_line = line.ljust(self._last_line_width)
      self._stream.write('\r' + padded_line)
      self._last_line_width = len(line)
      self._line_open = True
      if done:
        self._stream.write('\n')
        self._line_open = False
    else:
      self._stream.write(line + '\n')
    self._stream.flush()
    self._last_output_at = now


def run_episodes(env, episodes: int, show_progress: bool = True,
                 progress_stream: Optional[TextIO] = None,
                 clock=time.monotonic) -> None:
  """Run completed episodes, or continuously when episodes is zero."""
  progress = _EpisodeProgress(
      episodes, show_progress, progress_stream or sys.stderr, clock)
  completed_episodes = 0
  try:
    observation = env.reset()
    progress.start_episode(1, observation)
    while True:
      observation, _, done, _ = env.step([])
      progress.update(observation, done=done)
      if done:
        completed_episodes += 1
        if episodes and completed_episodes >= episodes:
          return
        observation = env.reset()
        progress.start_episode(completed_episodes + 1, observation)
  finally:
    progress.close()


def write_single_frame(env, output: Path) -> None:
  """Renders the initial state and writes it as a PNG."""
  env.reset()
  frame = env.render(mode='rgb_array')
  if not cv2.imwrite(str(output), frame):
    raise RuntimeError(f'Could not write frame to {format(output)}')


def main(_):

  if FLAGS.config_file:
      cfg = GameConfig.from_yaml(FLAGS.config_file)
  else:
      cfg = GameConfig()

  cfg = cfg.with_dynamic_tracesdir()
  cfg.tracesdir.mkdir(parents=True, exist_ok=True)

  cfg.to_yaml(cfg.tracesdir / "config.yaml")

  cfg_values = cfg.model_dump()
  env_cfg = config.Config(cfg_values)
  env = football_env.FootballEnv(env_cfg)

  try:
    if cfg.write_single_frame:
      write_single_frame(env, cfg.tracesdir / 'debug_frame.png')
      return
    if FLAGS.render:
      env.render()
    run_episodes(env, cfg.episodes, show_progress=cfg.show_progress)
  except KeyboardInterrupt:
    logging.warning('Game stopped, writing dump...')
    env.write_dump('shutdown')
    exit(1)
  finally:
    env.close()


if __name__ == '__main__':
  app.run(main)
