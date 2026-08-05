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
from pathlib import Path
from typing import Literal
from typing import Optional
from enum import Enum

import tempfile
import yaml

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
  dump_full_episodes: bool = True
  dump_scores: bool = False
  physics_steps_per_frame: int = 10
  render_resolution_x: int = 1280
  render_resolution_y: int = 720
  real_time: bool = False
  tracesdir: Path = Path(tempfile.gettempdir()) / "dumps"
  video_format: Literal["avi", "webm"] = "avi"
  video_quality_level: int = 0
  write_video: bool = True
  write_segmentation_video: bool = True
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

  if FLAGS.render:
    env.render()
  env.reset()
  try:
    while True:
      _, _, done, _ = env.step([])
      if done:
        env.reset()
  except KeyboardInterrupt:
    logging.warning('Game stopped, writing dump...')
    env.write_dump('shutdown')
    exit(1)


if __name__ == '__main__':
  app.run(main)
