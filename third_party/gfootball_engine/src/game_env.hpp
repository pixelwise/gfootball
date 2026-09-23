// Copyright 2019 Google LLC & Bastiaan Konings
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _GAME_ENV
#define _GAME_ENV

#include "onthepitch/match.hpp"
#include "gamedefines.hpp"
#include "gfootball_actions.h"
#include <map>
#include <vector>
#include "main.hpp"

class AIControlledKeyboard;
class GameTask;

typedef std::vector<std::string> StringVector;
struct CameraCapture {
  screenshoot frame;
  screenshoot segmentation_frame;
  std::vector<float> ball_screen_position;
  bool ball_screen_visible = false;
  int engine_step = -1;
};

typedef std::vector<CameraType> CameraTypeVector;


class ContextHolder {
 public:
  ContextHolder(GameEnv* game) : game(game) {
     SetGame(game);
     GetGraphicsSystem()->SetContext();
  }
  ~ContextHolder() {
    if (GetGame() != game) {
      Log(e_FatalError, "football", "main", "game state was corrupted");
    }
    GetGraphicsSystem()->DisableContext();
  }
 private:
  const GameEnv* game;
};

// Game environment. This is the class that can be used directly from Python.
struct GameEnv {
  GameEnv() { DO_VALIDATION;}
  // Start the game (in separate process).
  void start_game();

  // Get the current state of the game (observation).
  SharedInfo get_info();

  // Get the current rendered frame.
  screenshoot get_frame();
  screenshoot get_segmentation_frame();
  void set_render_cameras(const CameraTypeVector& cameras);
  screenshoot get_frame_for_camera(CameraType camera);
  screenshoot get_segmentation_frame_for_camera(CameraType camera);
  std::vector<float> get_ball_screen_position_for_camera(CameraType camera);
  bool get_ball_screen_visible_for_camera(CameraType camera);
  int get_capture_engine_step_for_camera(CameraType camera);


  // Executes the action inside the game.
  bool sticky_action_state(int action, bool left_team, int player);
  void action(int action, bool left_team, int player);
  void reset(ScenarioConfig& game_config, bool init_animation);
  void render(bool swap_buffer = true);
  std::string get_state(const std::string& pickle);
  std::string set_state(const std::string& state);
  void tracker_setup(long start, long end) { GetTracker()->setup(start, end); }
  void step();
  void ProcessState(EnvState* state);
  ScenarioConfig& config();

 private:
  void setConfig(ScenarioConfig& scenario_config);
  void do_step(int count);
  void getObservations();
  const CameraCapture& get_camera_capture(CameraType camera) const;
  void update_capture_engine_steps();
  AIControlledKeyboard* keyboard_ = nullptr;
  bool disable_graphics_ = false;
  int last_step_rendered_frames_ = 1;
 public:
  ScenarioConfig scenario_config;
  GameConfig game_config;
  CameraTypeVector render_cameras_;
  std::map<CameraType, CameraCapture> camera_captures_;
  GameContext* context = nullptr;
  GameState state = game_created;
  int waiting_for_game_count = 0;
};

#endif
