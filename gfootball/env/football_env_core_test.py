# coding=utf-8
# Copyright 2026 Google LLC
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

"""Tests for the core football environment."""

from absl.testing import absltest

import gfootball_engine as libgame

from gfootball.env import football_env_core


class FootballEnvCoreTest(absltest.TestCase):

  def test_pano_camera_mapping(self):
    self.assertEqual(football_env_core.CameraType.PANO.value, 'pano')
    self.assertEqual(football_env_core.CAMERA_MAP['pano'], libgame.CameraType.PANO)


if __name__ == '__main__':
  absltest.main()
