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


"""Observation processor, providing multiple support methods for analyzing observations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import datetime
import os
import shutil
import subprocess
import tempfile
import timeit
import traceback

from absl import logging
from gfootball.env import constants as const
from gfootball.env import football_action_set
from gfootball.scenarios import e_PlayerRole_GK
import numpy as np
from six.moves import range
from six.moves import zip
import six.moves.cPickle

# How many past frames are kept around for the dumps to make use of them.
PAST_STEPS_TRACE_SIZE = 100

WRITE_FILES = True

try:
  import cv2
except ImportError:
  import cv2


def resize_segmentation_frame(frame, frame_dim):
  """Resizes a segmentation frame while preserving its binary labels."""
  frame = cv2.resize(frame, frame_dim, interpolation=cv2.INTER_NEAREST)
  return np.where(frame != 0, 255, 0).astype(np.uint8)


def resize_instance_segmentation_frame(frame, frame_dim):
  """Resizes an uint8 instance-label frame without interpolating labels."""
  return cv2.resize(frame, frame_dim,
                    interpolation=cv2.INTER_NEAREST).astype(np.uint8)


def _video_fourcc(video_format, video_quality_level, lossless=False):
  """Returns the OpenCV codec identifier for a configured video format."""
  if video_format == 'avi':
    if lossless or video_quality_level == 2:
      return cv2.VideoWriter_fourcc('p', 'n', 'g', ' ')
    if video_quality_level == 1:
      return cv2.VideoWriter_fourcc(*'MJPG')
    return cv2.VideoWriter_fourcc(*'XVID')
  if video_format == 'webm':
    return cv2.VideoWriter_fourcc(*'vp80')
  return cv2.VideoWriter_fourcc(*'mp4v')


class _HlsVideoWriter(object):
  """Streams RGB frames to a single-file, byte-range HLS package."""

  def __init__(self, playlist_path, frame_dim, fps, quality_level):
    width, height = frame_dim
    if width % 2 or height % 2:
      raise ValueError(
          'M3U8 video requires even render dimensions for yuv420p; got '
          '%dx%d.' % (width, height))

    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
      raise RuntimeError(
          'Unable to create M3U8 video because ffmpeg was not found. '
          'Install ffmpeg or use avi, webm, or mp4.')

    self._playlist_path = playlist_path
    root, extension = os.path.splitext(playlist_path)
    assert extension == '.m3u8'
    self._segment_path = root + '_segments.ts'
    self._frame_dim = frame_dim
    output_directory = os.path.dirname(os.path.abspath(playlist_path))
    self._staging_directory = tempfile.mkdtemp(
        prefix='.gfootball-hls-', dir=output_directory)
    self._staged_playlist = os.path.join(
        self._staging_directory, os.path.basename(self._playlist_path))
    self._staged_segment = os.path.join(
        self._staging_directory, os.path.basename(self._segment_path))
    self._stderr = tempfile.TemporaryFile(mode='w+b')
    self._frame_count = 0
    self._released = False

    crf = {1: 23, 2: 18}.get(quality_level, 28)
    gop_size = max(1, int(round(fps * 2.0)))
    command = [
        ffmpeg,
        '-hide_banner',
        '-loglevel', 'error',
        '-y',
        '-f', 'rawvideo',
        '-pixel_format', 'bgr24',
        '-video_size', '%dx%d' % (width, height),
        '-framerate', str(fps),
        '-i', 'pipe:0',
        '-an',
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        '-flags', '+cgop',
        '-g', str(gop_size),
        '-keyint_min', str(gop_size),
        '-sc_threshold', '0',
        '-f', 'hls',
        '-hls_time', '2',
        '-hls_playlist_type', 'vod',
        '-hls_segment_type', 'mpegts',
        '-hls_flags', 'single_file',
        '-hls_segment_filename', self._staged_segment,
        self._staged_playlist,
    ]
    try:
      self._process = subprocess.Popen(
          command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
          stderr=self._stderr)
    except Exception:
      self._cleanup_staging()
      self._stderr.close()
      raise

  def write(self, frame):
    if self._released:
      raise RuntimeError('Cannot write to a finalized M3U8 video.')
    expected_shape = (self._frame_dim[1], self._frame_dim[0], 3)
    if frame.shape != expected_shape or frame.dtype != np.uint8:
      raise ValueError(
          'M3U8 frame must be uint8 with shape %s; got %s %s.' %
          (expected_shape, frame.shape, frame.dtype))
    try:
      self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
    except BrokenPipeError:
      raise RuntimeError(self._ffmpeg_error('FFmpeg stopped accepting frames'))
    self._frame_count += 1

  def _ffmpeg_error(self, prefix):
    self._stderr.flush()
    self._stderr.seek(0)
    details = self._stderr.read().decode('utf-8', errors='replace').strip()
    return '%s%s' % (prefix, ': ' + details if details else '')

  def _cleanup_staging(self):
    shutil.rmtree(self._staging_directory, ignore_errors=True)

  def release(self, write_files=True):
    """Finalizes and optionally publishes the HLS package."""
    if self._released:
      return False
    self._released = True
    try:
      self._process.stdin.close()
    except BrokenPipeError:
      pass
    return_code = self._process.wait()

    if self._frame_count == 0:
      self._cleanup_staging()
      self._stderr.close()
      return False
    if return_code != 0:
      error = self._ffmpeg_error(
          'FFmpeg failed to finalize M3U8 video with exit code %d' %
          return_code)
      self._cleanup_staging()
      self._stderr.close()
      raise RuntimeError(error)

    try:
      if write_files:
        os.replace(self._staged_segment, self._segment_path)
        try:
          os.replace(self._staged_playlist, self._playlist_path)
        except Exception:
          if os.path.exists(self._segment_path):
            os.remove(self._segment_path)
          raise
    finally:
      self._cleanup_staging()
      self._stderr.close()
    return True


class DumpConfig(object):

  def __init__(self,
               max_count=1,
               steps_before=PAST_STEPS_TRACE_SIZE,
               steps_after=0,
               min_frequency=10):
    self._steps_before = steps_before
    self._steps_after = steps_after
    self._max_count = max_count
    # Make sure self._last_dump_time < timeit.default_timer() - min_frequency
    # holds upon startup.
    self._last_dump_time = timeit.default_timer() - 2 * min_frequency
    self._active_dump = None
    self._min_frequency = min_frequency


class TextWriter(object):

  def __init__(self, frame, x, y=0, field_coords=False, color=(255, 255, 255)):
    self._frame = frame
    if field_coords:
      x = 400 * (x + 1) - 10
      y = 695 * (y + 0.43)
    self._pos_x = int(x)
    self._pos_y = int(y) + 20
    self._color = color
    self._font = cv2.FONT_HERSHEY_SIMPLEX
    self._lineType = 2
    self._arrow_types = ('top', 'top_right', 'right', 'bottom_right', 'bottom',
                         'bottom_left', 'left', 'top_left')

  def write(self, text, scale_factor=1, color=None):
    textPos = (self._pos_x, self._pos_y)
    fontScale = 0.8 * scale_factor
    cv2.putText(self._frame, text, textPos, self._font, fontScale, color or self._color,
                self._lineType)
    self._pos_y += int(25 * scale_factor)

  def write_table(self, data, widths, scale_factor=1, offset=0):
    # data is a list of rows. Each row is a list of strings.
    fontScale = 0.5 * scale_factor
    init_x = self._pos_x
    for row in data:
      assert (len(row) == len(widths))
      self._pos_x += offset
      for col, cell in enumerate(row):
        color = self._color
        if isinstance(cell, tuple):
          assert (len(cell) == 2)
          (text, color) = cell
        else:
          assert (isinstance(cell, str))
          text = cell

        if text in self._arrow_types:
          self.write_arrow(text, scale_factor=scale_factor)
        else:
          textPos = (self._pos_x, self._pos_y)
          cv2.putText(self._frame, text, textPos, self._font, fontScale, color,
                      self._lineType)
        self._pos_x += widths[col]
      self._pos_x = init_x
      self._pos_y += int(20 * scale_factor)
    self._pos_x = init_x

  def write_arrow(self, arrow_type, scale_factor=1):
    assert (arrow_type in self._arrow_types)
    thickness = 1
    arrow_offsets = {
        'top': (12, 0, 12, -16),
        'top_right': (4, -4, 16, -16),
        'right': (0, -10, 20, -10),
        'bottom_right': (4, -16, 16, -4),
        'bottom': (10, -16, 10, 0),
        'bottom_left': (12, -12, 0, 0),
        'left': (20, -10, 0, -10),
        'top_left': (16, -4, 4, -16)
    }
    (s_x, s_y, e_x,
     e_y) = tuple(int(v * scale_factor) for v in arrow_offsets[arrow_type])
    start_point = (self._pos_x + s_x, self._pos_y + s_y)
    end_point = (self._pos_x + e_x, self._pos_y + e_y)
    image = cv2.arrowedLine(self._frame, start_point, end_point, self._color,
                            thickness)


def write_players_state(writer, players_info):
  table_text = [["PLAYER", "SPRINT", "DRIBBLE", "DIRECTION", "ACTION"]]
  widths = [65, 65, 70, 85, 85]

  # Sort the players according to the order they appear in observations
  for _, player_info in sorted(players_info.items()):
    table_text.append([
      (player_info['id'], player_info['color']),
      str(player_info.get("sprint", "-")),
      str(player_info.get("dribble", "-")),
      player_info.get("DIRECTION", "O"),
      player_info.get("ACTION", "-")])
  writer.write_table(table_text, widths, scale_factor=1.0, offset=10)


def get_frame(trace):
  if 'frame' in trace._trace['observation']:
    return trace._trace['observation']['frame']
  frame = np.uint8(np.zeros((600, 800, 3)))
  corner1 = (0, 0)
  corner2 = (799, 0)
  corner3 = (799, 599)
  corner4 = (0, 599)
  line_color = (0, 255, 255)
  cv2.line(frame, corner1, corner2, line_color)
  cv2.line(frame, corner2, corner3, line_color)
  cv2.line(frame, corner3, corner4, line_color)
  cv2.line(frame, corner4, corner1, line_color)
  cv2.line(frame, (399, 0), (399, 799), line_color)
  writer = TextWriter(
      frame,
      trace['ball'][0],
      trace['ball'][1],
      field_coords=True,
      color=(248, 244, 236))
  writer.write('B')
  for player_idx, player_coord in enumerate(trace['left_team']):
    writer = TextWriter(
        frame,
        player_coord[0],
        player_coord[1],
        field_coords=True,
        color=(238, 68, 47))
    letter = str(player_idx)
    if trace['left_team_roles'][player_idx] == e_PlayerRole_GK:
      letter = 'G'
    writer.write(letter)
  for player_idx, player_coord in enumerate(trace['right_team']):
    writer = TextWriter(
        frame,
        player_coord[0],
        player_coord[1],
        field_coords=True,
        color=(99, 172, 190))
    letter = str(player_idx)
    if trace['right_team_roles'][player_idx] == e_PlayerRole_GK:
      letter = 'G'
    writer.write(letter)
  return frame


def softmax(x):
  return np.exp(x) / np.sum(np.exp(x), axis=0)


class ActiveDump(object):

  def __init__(self,
               name,
               finish_step,
               config):
    self._name = name
    self._finish_step = finish_step
    self._config = config
    self._initializing = True
    self._video_fd = None
    self._video_tmp = None
    self._video_writer = None
    self._segmentation_video_fd = None
    self._segmentation_video_tmp = None
    self._segmentation_video_writer = None
    self._instance_video_fd = None
    self._instance_video_tmp = None
    self._instance_video_writer = None
    self._instance_engine_step = []
    self._instance_metadata = {}
    self._ball_xy = [] if (config['write_ball_coordinates'] and
                           config['write_video']) else None
    self._ball_visible = [] if self._ball_xy is not None else None
    self._ball_engine_step = [] if self._ball_xy is not None else None
    self._frame_dim = None
    self._step_cnt = 0
    self._dump_file = None
    self._video_format = None
    self._video_suffix = None
    self._segmentation_video_suffix = None
    self._instance_video_suffix = None
    if (config['write_video'] or config['write_segmentation_video'] or
        config['write_instance_segmentation_video']):
      video_format = config['video_format']
      if video_format not in ['avi', 'webm', 'mp4', 'm3u8']:
        raise ValueError(
            'Unsupported video format: %s' % video_format)
      self._video_format = video_format
      self._video_suffix = '.%s' % video_format
      mask_video_format = 'mp4' if video_format == 'm3u8' else video_format
      self._segmentation_video_suffix = '.%s' % mask_video_format
      self._instance_video_suffix = '.%s' % mask_video_format
      self._frame_dim = (
          config['render_resolution_x'], config['render_resolution_y'])
      if config['video_quality_level'] not in [1, 2]:
        # Reduce resolution to (800, 450).
        self._frame_dim = min(self._frame_dim, (800, 450))
      fps = (const.PHYSICS_STEPS_PER_SECOND /
             config['physics_steps_per_frame'])
    if config['write_video']:
      if video_format == 'm3u8':
        self._video_writer = _HlsVideoWriter(
            self._name + self._video_suffix, self._frame_dim, fps,
            config['video_quality_level'])
      else:
        self._video_fd, self._video_tmp = tempfile.mkstemp(
            suffix=self._video_suffix)
        fcc = _video_fourcc(video_format, config['video_quality_level'])
        self._video_writer = cv2.VideoWriter(
            self._video_tmp, fcc, fps, self._frame_dim)
        self._ensure_video_writer_open(
            '_video_writer', '_video_fd', '_video_tmp', video_format)
    if config['write_segmentation_video']:
      self._segmentation_video_fd, self._segmentation_video_tmp = \
          tempfile.mkstemp(suffix=self._segmentation_video_suffix)
      mask_fcc = _video_fourcc(
          mask_video_format, config['video_quality_level'], lossless=True)
      self._segmentation_video_writer = cv2.VideoWriter(
          self._segmentation_video_tmp, mask_fcc, fps, self._frame_dim)
      self._ensure_video_writer_open(
          '_segmentation_video_writer', '_segmentation_video_fd',
          '_segmentation_video_tmp', mask_video_format)
    if config['write_instance_segmentation_video']:
      self._instance_video_fd, self._instance_video_tmp = \
          tempfile.mkstemp(suffix=self._instance_video_suffix)
      instance_fcc = _video_fourcc(
          mask_video_format, config['video_quality_level'], lossless=True)
      self._instance_video_writer = cv2.VideoWriter(
          self._instance_video_tmp, instance_fcc, fps, self._frame_dim)
      self._ensure_video_writer_open(
          '_instance_video_writer', '_instance_video_fd',
          '_instance_video_tmp', mask_video_format)
    if WRITE_FILES:
      self._dump_file = open(name + '.dump', 'wb')
    self._initializing = False

  def __del__(self):
    if not self._initializing:
      self.finalize()

  def _ensure_video_writer_open(
      self, writer_attribute, fd_attribute, tmp_attribute, video_format):
    """Cleans up and raises when OpenCV cannot create a video writer."""
    writer = getattr(self, writer_attribute)
    if writer.isOpened():
      return
    writer.release()
    setattr(self, writer_attribute, None)
    os.close(getattr(self, fd_attribute))
    setattr(self, fd_attribute, None)
    os.remove(getattr(self, tmp_attribute))
    setattr(self, tmp_attribute, None)
    raise RuntimeError(
        'Unable to open %s video writer. Check that the OpenCV video backend '
        'supports this format.' % video_format)

  def _add_ball_coordinate(self, ball_screen_position, visible, engine_step):
    """Adds one ball record in final-video pixel coordinates."""
    if self._ball_xy is None:
      return
    if ball_screen_position is None or len(ball_screen_position) < 2:
      self._ball_xy.append((np.nan, np.nan))
      self._ball_visible.append(False)
    else:
      self._ball_xy.append((
          float(ball_screen_position[0]) * self._frame_dim[0],
          float(ball_screen_position[1]) * self._frame_dim[1]))
      self._ball_visible.append(bool(visible))
    self._ball_engine_step.append(int(engine_step))

  def _write_segmentation_frames(self, segmentation_frame, engine_step):
    """Writes binary and instance views derived from one renderer label map."""
    if segmentation_frame is None:
      return
    if self._segmentation_video_writer:
      binary_frame = resize_segmentation_frame(
          segmentation_frame, self._frame_dim)
      self._segmentation_video_writer.write(binary_frame)
    if self._instance_video_writer:
      instance_frame = resize_instance_segmentation_frame(
          segmentation_frame, self._frame_dim)
      self._instance_video_writer.write(instance_frame)
      self._instance_engine_step.append(int(engine_step))

  def _capture_instance_metadata(self, observation):
    if not self._instance_video_writer:
      return
    for team, key in enumerate(
        ['left_team_instance_id', 'right_team_instance_id']):
      if key not in observation:
        continue
      for player_index, instance_id in enumerate(observation[key]):
        self._instance_metadata.setdefault(
            int(instance_id), (team, player_index))

  def add_frame(self, frame, segmentation_frame=None,
                ball_screen_position=None, ball_screen_visible=False,
                engine_step=-1):
    if self._video_writer:
      frame = frame[..., ::-1]
      frame = cv2.resize(frame, self._frame_dim, interpolation=cv2.INTER_AREA)
      self._video_writer.write(frame)
      self._add_ball_coordinate(ball_screen_position, ball_screen_visible,
                                engine_step)
    self._write_segmentation_frames(segmentation_frame, engine_step)

  def add_step(self, o):
    # Write video if requested.
    if self._video_writer:
      frame = get_frame(o)
      frame = frame[..., ::-1]
      frame = cv2.resize(frame, self._frame_dim, interpolation=cv2.INTER_AREA)
      writer = TextWriter(frame, self._frame_dim[0] - 300)
      if self._config['custom_display_stats']:
        for line in self._config['custom_display_stats']:
          writer.write(line)
      if self._config['display_game_stats']:
        writer.write('SCORE: %d - %d' % (o['score'][0], o['score'][1]))
        if o['ball_owned_team'] == 0:
          player = 'G' if o['left_team_roles'][
              o['ball_owned_player']] == e_PlayerRole_GK else o[
                  'ball_owned_player']
          writer.write('BALL OWNED: %s' % player, color=(47, 68, 238))
        elif o['ball_owned_team'] == 1:
          player = 'G' if o['right_team_roles'][
              o['ball_owned_player']] == e_PlayerRole_GK else o[
                  'ball_owned_player']
          writer.write('BALL OWNED: %s' % player, color=(190, 172, 99))
        else:
          writer.write('BALL OWNED: ---')
        writer = TextWriter(frame, 0)
        writer.write('STEP: %d' % self._step_cnt)
        sticky_actions = football_action_set.get_sticky_actions(self._config)

        players_info = {}
        for team in ['left', 'right']:
          player_info = {}
          sticky_actions_field = '%s_agent_sticky_actions' % team
          for player in range(len(o[sticky_actions_field])):
            assert len(sticky_actions) == len(o[sticky_actions_field][player])
            player_idx = o['%s_agent_controlled_player' % team][player]
            player_info = {}
            player_info['color'] = (
                47, 68, 238) if team == 'left' else (190, 172, 99)
            player_info['id'] = 'G' if o[
                '%s_team_roles' %
                team][player_idx] == e_PlayerRole_GK else str(player_idx)
            active_direction = None
            for i in range(len(sticky_actions)):
              if sticky_actions[i]._directional:
                if o[sticky_actions_field][player][i]:
                  active_direction = sticky_actions[i]
              else:
                player_info[sticky_actions[i]._name] = \
                    o[sticky_actions_field][player][i]

            # Info about direction
            player_info['DIRECTION'] = \
                'O' if active_direction is None else active_direction._name
            if 'action' in o._trace['debug']:
              # Info about action
              player_info['ACTION'] = \
                  o['action'][len(players_info)]._name
            players_info[(team, player_idx)] = player_info

        write_players_state(writer, players_info)

        if 'baseline' in o._trace['debug']:
          writer.write('BASELINE: %.5f' % o._trace['debug']['baseline'])
        if 'logits' in o._trace['debug']:
          probs = softmax(o._trace['debug']['logits'])
          action_set = football_action_set.get_action_set(self._config)
          for action, prob in zip(action_set, probs):
            writer.write('%s: %.5f' % (action.name, prob), scale_factor=0.5)
        for d in o._debugs:
          writer.write(d)
      self._video_writer.write(frame)
      self._add_ball_coordinate(
          o['ball_screen_position'] if 'ball_screen_position' in o else None,
          o['ball_screen_visible'] if 'ball_screen_visible' in o else False,
          o['engine_step'] if 'engine_step' in o else -1)
    self._capture_instance_metadata(o)
    if 'segmentation_frame' in o:
      self._write_segmentation_frames(
          o['segmentation_frame'],
          o['engine_step'] if 'engine_step' in o else -1)
    # Write the dump.
    temp_frame = None
    if 'frame' in o._trace['observation']:
      temp_frame = o._trace['observation']['frame']
      del o._trace['observation']['frame']
    temp_segmentation_frame = None
    if 'segmentation_frame' in o._trace['observation']:
      temp_segmentation_frame = o._trace['observation']['segmentation_frame']
      del o._trace['observation']['segmentation_frame']

    # Add config to the first frame for our replay tools to use.
    if self._step_cnt == 0:
      o['debug']['config'] = self._config.get_dictionary()

    if self._dump_file:
      six.moves.cPickle.dump(o._trace, self._dump_file)
    if temp_frame is not None:
      o._trace['observation']['frame'] = temp_frame
    if temp_segmentation_frame is not None:
      o._trace['observation']['segmentation_frame'] = temp_segmentation_frame
    self._step_cnt += 1

  def finalize(self):
    dump_info = {}
    if self._video_writer:
      if self._video_format == 'm3u8':
        video_writer = self._video_writer
        self._video_writer = None
        if video_writer.release(write_files=WRITE_FILES):
          video = self._name + self._video_suffix
          dump_info['video'] = video
          logging.info('Video written to %s', video)
        else:
          logging.warning('No frames written to M3U8 video.')
      else:
        self._video_writer.release()
        self._video_writer = None
        os.close(self._video_fd)
        try:
          # For some reason sometimes the file is missing, so the code fails.
          if WRITE_FILES:
            shutil.move(self._video_tmp, self._name + self._video_suffix)
          dump_info['video'] = '%s%s' % (self._name, self._video_suffix)
          logging.info('Video written to %s%s', self._name,
                       self._video_suffix)
        except:
          logging.error(traceback.format_exc())
    if self._segmentation_video_writer:
      self._segmentation_video_writer.release()
      self._segmentation_video_writer = None
      os.close(self._segmentation_video_fd)
      try:
        segmentation_video = (
            self._name + '_segmentation' + self._segmentation_video_suffix)
        if WRITE_FILES:
          shutil.move(self._segmentation_video_tmp, segmentation_video)
        dump_info['segmentation_video'] = segmentation_video
        logging.info('Segmentation video written to %s', segmentation_video)
      except:
        logging.error(traceback.format_exc())
    if self._instance_video_writer:
      self._instance_video_writer.release()
      self._instance_video_writer = None
      os.close(self._instance_video_fd)
      try:
        instance_video = (
            self._name + '_instances' + self._instance_video_suffix)
        if WRITE_FILES:
          shutil.move(self._instance_video_tmp, instance_video)
        dump_info['instance_segmentation_video'] = instance_video
        logging.info('Instance segmentation video written to %s',
                     instance_video)
      except:
        logging.error(traceback.format_exc())

      instance_metadata = self._name + '_instances.npz'
      instance_ids = np.asarray(
          sorted(self._instance_metadata), dtype=np.uint8)
      teams = np.asarray([
          self._instance_metadata[int(instance_id)][0]
          for instance_id in instance_ids], dtype=np.int8)
      team_player_index = np.asarray([
          self._instance_metadata[int(instance_id)][1]
          for instance_id in instance_ids], dtype=np.int16)
      if WRITE_FILES:
        np.savez(
            instance_metadata,
            engine_step=np.asarray(
                self._instance_engine_step, dtype=np.int32),
            instance_id=instance_ids,
            team=teams,
            team_player_index=team_player_index,
            frame_size=np.asarray(self._frame_dim, dtype=np.int32),
            fps=np.asarray(
                const.PHYSICS_STEPS_PER_SECOND /
                self._config['physics_steps_per_frame'], dtype=np.float32))
      dump_info['instance_segmentation_metadata'] = instance_metadata
      logging.info('Instance segmentation metadata written to %s',
                   instance_metadata)
      self._instance_engine_step = None
      self._instance_metadata = None
    if self._ball_xy is not None:
      ball_coordinates = self._name + '_ball.npz'
      if WRITE_FILES:
        np.savez(
            ball_coordinates,
            xy=np.asarray(self._ball_xy, dtype=np.float32).reshape((-1, 2)),
            visible=np.asarray(self._ball_visible, dtype=np.bool_),
            engine_step=np.asarray(self._ball_engine_step, dtype=np.int32),
            frame_size=np.asarray(self._frame_dim, dtype=np.int32),
            fps=np.asarray(
                const.PHYSICS_STEPS_PER_SECOND /
                self._config['physics_steps_per_frame'], dtype=np.float32))
      dump_info['ball_coordinates'] = ball_coordinates
      logging.info('Ball coordinates written to %s', ball_coordinates)
      self._ball_xy = None
      self._ball_visible = None
      self._ball_engine_step = None
    if self._dump_file:
      self._dump_file.close()
      self._dump_file = None
      if self._step_cnt == 0:
        logging.warning('No data to write to the dump.')
      else:
        dump_info['dump'] = '%s.dump' % self._name
        logging.info('Dump written to %s.dump', self._name)
    return dump_info


class MultiCameraActiveDump(object):
  """Coordinates one pixel writer per camera and one shared state dump."""

  _CAMERA_FIELDS = {
      'camera_frames': 'frame',
      'camera_segmentation_frames': 'segmentation_frame',
      'camera_ball_screen_position': 'ball_screen_position',
      'camera_ball_screen_visible': 'ball_screen_visible',
      'camera_engine_step': 'engine_step',
  }

  def __init__(self, name, finish_step, config):
    self._name = name
    self._finish_step = finish_step
    self._config = config
    self._initializing = True
    self._step_cnt = 0
    self._finalized = False
    self._camera_dumps = collections.OrderedDict()
    self._dump_file = None
    cameras = config['cameras']
    for camera in cameras:
      camera_name = camera.value if hasattr(camera, 'value') else str(camera)
      camera_config = config.__class__(config.get_dictionary())
      camera_config['camera'] = camera_name
      camera_config['cameras'] = None
      camera_dump = ActiveDump(name + '_' + camera_name, finish_step,
                               camera_config)
      if camera_dump._dump_file:
        camera_dump._dump_file.close()
        camera_dump._dump_file = None
        os.remove(camera_dump._name + '.dump')
      self._camera_dumps[camera_name] = camera_dump
    self._dump_file = open(name + '.dump', 'wb') if WRITE_FILES else None
    self._initializing = False

  def __del__(self):
    if not self._initializing:
      self.finalize()

  def _state_for_camera(self, state, camera):
    trace = state._trace.copy()
    observation = state._trace['observation'].copy()
    for map_name, scalar_name in self._CAMERA_FIELDS.items():
      values = observation.pop(map_name, None)
      if values is not None:
        observation[scalar_name] = values[camera]
    trace['observation'] = observation
    trace['debug'] = state._trace['debug'].copy()
    camera_state = ObservationState(trace)
    camera_state._debugs = list(state._debugs)
    return camera_state

  def _write_dump_step(self, state):
    if not self._dump_file:
      return
    trace = state._trace.copy()
    trace['observation'] = state._trace['observation'].copy()
    for field in [
        'frame', 'segmentation_frame', 'camera_frames',
        'camera_segmentation_frames', 'camera_ball_screen_position',
        'camera_ball_screen_visible', 'camera_engine_step'
    ]:
      trace['observation'].pop(field, None)
    trace['debug'] = state._trace['debug'].copy()
    if self._step_cnt == 0:
      trace['debug']['config'] = self._config.get_dictionary()
    six.moves.cPickle.dump(trace, self._dump_file)

  def add_step(self, state):
    for camera, camera_dump in self._camera_dumps.items():
      camera_dump.add_step(self._state_for_camera(state, camera))
    self._write_dump_step(state)
    self._step_cnt += 1

  def add_frame(self, frame, segmentation_frame=None,
                ball_screen_position=None, ball_screen_visible=False,
                engine_step=-1):
    if not isinstance(frame, dict):
      raise ValueError('Multi-camera frame bundle must be a dictionary')
    segmentation_frames = segmentation_frame or {}
    ball_positions = ball_screen_position or {}
    ball_visibility = ball_screen_visible or {}
    engine_steps = engine_step or {}
    for camera, camera_dump in self._camera_dumps.items():
      camera_dump.add_frame(
          frame[camera], segmentation_frames.get(camera),
          ball_positions.get(camera), ball_visibility.get(camera, False),
          engine_steps.get(camera, -1))

  def finalize(self):
    if self._finalized:
      return {}
    self._finalized = True
    dump_info = {}
    primary_camera = next(iter(self._camera_dumps))
    camera_info = collections.OrderedDict()
    errors = []
    for camera, camera_dump in self._camera_dumps.items():
      try:
        camera_info[camera] = camera_dump.finalize()
      except Exception as error:
        logging.error(
            'Failed to finalize camera %s: %s', camera, error)
        camera_info[camera] = {}
        errors.append((camera, error))
    plural_keys = {
        'video': 'videos',
        'segmentation_video': 'segmentation_videos',
        'instance_segmentation_video': 'instance_segmentation_videos',
        'instance_segmentation_metadata':
            'instance_segmentation_metadata_by_camera',
        'ball_coordinates': 'ball_coordinates_by_camera',
    }
    for singular, plural in plural_keys.items():
      values = collections.OrderedDict(
          (camera, info[singular])
          for camera, info in camera_info.items() if singular in info)
      if values:
        dump_info[plural] = values
        if primary_camera in values:
          dump_info[singular] = values[primary_camera]
    self._camera_dumps = collections.OrderedDict()
    if self._dump_file:
      self._dump_file.close()
      self._dump_file = None
      if self._step_cnt == 0:
        logging.warning('No data to write to the dump.')
      else:
        dump_info['dump'] = '%s.dump' % self._name
        logging.info('Dump written to %s.dump', self._name)
    if errors:
      raise RuntimeError(
          'Failed to finalize camera writers: %s' %
          ', '.join(camera for camera, _ in errors))
    return dump_info


def _create_active_dump(name, finish_step, config):
  cameras = config['cameras'] if 'cameras' in config else None
  if cameras and len(cameras) > 1:
    return MultiCameraActiveDump(name, finish_step, config)
  return ActiveDump(name, finish_step, config)


class ObservationState(object):

  def __init__(self, trace):
    # Observations
    self._trace = trace
    self._additional_frames = []
    self._debugs = []

  def __getitem__(self, key):
    if key in self._trace:
      return self._trace[key]
    if key in self._trace['observation']:
      return self._trace['observation'][key]
    return self._trace['debug'][key]

  def __contains__(self, key):
    if key in self._trace:
      return True
    if key in self._trace['observation']:
      return True
    return key in self._trace['debug']

  def _distance(self, o1, o2):
    # We add 'z' dimension if not present, as ball has 3 dimensions, while
    # players have only 2.
    if len(o1) == 2:
      o1 = np.array([o1[0], o1[1], 0])
    if len(o2) == 2:
      o2 = np.array([o2[0], o2[1], 0])
    return np.linalg.norm(o1 - o2)

  def add_debug(self, text):
    self._debugs.append(text)

  def add_frame(self, frame, segmentation_frame=None,
                ball_screen_position=None, ball_screen_visible=False,
                engine_step=-1):
    self._additional_frames.append((
        frame, segmentation_frame, ball_screen_position,
        ball_screen_visible, engine_step))


class ObservationProcessor(object):

  def __init__(self, config):
    # Const. configuration
    self._ball_takeover_epsilon = 0.03
    self._ball_lost_epsilon = 0.05
    self._frame = 0
    self._dump_config = {}
    self._dump_config['score'] = DumpConfig(
        steps_before=PAST_STEPS_TRACE_SIZE,
        max_count=(100000 if config['dump_scores'] else 0),
        min_frequency=600,
        steps_after=1)
    self._dump_config['lost_score'] = DumpConfig(
        steps_before=PAST_STEPS_TRACE_SIZE,
        max_count=(100000 if config['dump_scores'] else 0),
        min_frequency=600,
        steps_after=1)
    self._dump_config['episode_done'] = DumpConfig(
        steps_before=0,
        # Record entire episode.
        steps_after=10000,
        max_count=(100000 if config['dump_full_episodes'] else 0))
    self._dump_config['shutdown'] = DumpConfig(steps_before=PAST_STEPS_TRACE_SIZE)
    self._dump_directory = None
    self._config = config
    self.clear_state()

  def clear_state(self):
    self._frame = 0
    self._state = None
    self._trace = collections.deque([], PAST_STEPS_TRACE_SIZE)
    self._transient_camera_state = None

  def reset(self):
    self.clear_state()

  def len(self):
    return len(self._trace)

  def __getitem__(self, key):
    return self._trace[key]

  def add_frame(self, frame, segmentation_frame=None,
                ball_screen_position=None, ball_screen_visible=False,
                engine_step=-1):
    if len(self._trace) > 0 and (
        self._config['write_video'] or
        self._config['write_segmentation_video'] or
        self._config['write_instance_segmentation_video']):
      dumps = self.pending_dumps()
      # Multi-camera frames are too large to retain in the rolling trace.
      # Once an incremental dump exists, stream them directly to its writers.
      if not isinstance(frame, dict) or not dumps:
        self._trace[-1].add_frame(
            frame, segmentation_frame, ball_screen_position,
            ball_screen_visible, engine_step)
      for dump in dumps:
        dump.add_frame(
            frame, segmentation_frame, ball_screen_position,
            ball_screen_visible, engine_step)

  def update(self, trace):
    self._frame += 1
    self._transient_camera_state = None
    if 'camera_frames' in trace['observation']:
      # Feed this full state to writers, but retain only non-pixel gameplay
      # data in the rolling history and serialized dump.
      transient_state = ObservationState(trace)
      durable_trace = trace.copy()
      durable_trace['observation'] = trace['observation'].copy()
      for field in [
          'frame', 'segmentation_frame', 'camera_frames',
          'camera_segmentation_frames', 'camera_ball_screen_position',
          'camera_ball_screen_visible', 'camera_engine_step'
      ]:
        durable_trace['observation'].pop(field, None)
      self._state = ObservationState(durable_trace)
      self._trace.append(self._state)
      dumps = self.pending_dumps()
      for dump in dumps:
        dump.add_step(transient_state)
      if not dumps:
        # The full-episode dump is requested immediately after the first
        # update. Keep this one bundle just long enough for write_dump().
        self._transient_camera_state = (self._state, transient_state)
      return

    remove_frame = (not self._config['write_video'] and
                    'frame' in trace['observation'])
    remove_segmentation = (
        not (self._config['write_segmentation_video'] or
             self._config['write_instance_segmentation_video']) and
        'segmentation_frame' in trace['observation'])
    if remove_frame or remove_segmentation:
      # Keep only pixel observations used by the requested video writers.
      video_trace = trace.copy()
      video_trace['observation'] = trace['observation'].copy()
      if remove_frame:
        del video_trace['observation']['frame']
      if remove_segmentation:
        del video_trace['observation']['segmentation_frame']
      self._state = ObservationState(video_trace)
    else:
      self._state = ObservationState(trace)
    self._trace.append(self._state)
    for dump in self.pending_dumps():
      dump.add_step(self._state)

  def get_last_frame(self):
    if not self._state:
      return []
    return get_frame(self._state)

  def write_dump(self, name):
    if not name in self._dump_config:
      self._dump_config[name] = DumpConfig()
    config = self._dump_config[name]
    if config._active_dump:
      logging.debug('Dump "%s": already pending', name)
      return
    if config._max_count <= 0:
      logging.debug('Dump "%s": count limit reached / disabled', name)
      return
    if config._last_dump_time > timeit.default_timer() - config._min_frequency:
      logging.debug('Dump "%s": too frequent', name)
      return
    config._max_count -= 1
    config._last_dump_time = timeit.default_timer()
    if self._dump_directory is None:
      self._dump_directory = self._config['tracesdir']
      if WRITE_FILES:
        if not os.path.exists(self._dump_directory):
          os.makedirs(self._dump_directory)
    dump_name = '{2}{3}{0}_{1}'.format(name,
        datetime.datetime.now().strftime('%Y%m%d-%H%M%S%f'),
        self._dump_directory, os.sep)
    config._active_dump = _create_active_dump(dump_name,
        self._frame + config._steps_after, self._config)
    for step in list(self._trace)[-config._steps_before:]:
      dump_step = step
      if (self._transient_camera_state and
          self._transient_camera_state[0] is step):
        dump_step = self._transient_camera_state[1]
      config._active_dump.add_step(dump_step)
      for (frame, segmentation_frame, ball_screen_position,
           ball_screen_visible, engine_step) in step._additional_frames:
        config._active_dump.add_frame(
            frame, segmentation_frame, ball_screen_position,
            ball_screen_visible, engine_step)
    self._transient_camera_state = None
    if config._steps_after == 0:
      # Synchronously finalize dump, so that crash dump is recorded.
      config._active_dump.finalize()
      config._active_dump = None
    return dump_name

  def pending_dumps(self):
    dumps = []
    for config in self._dump_config.values():
      if config._active_dump:
        dumps.append(config._active_dump)
    return dumps

  def process_pending_dumps(self, episode_done=False):
    dumps = []
    for name in self._dump_config:
      config = self._dump_config[name]
      if config._active_dump and (
          episode_done or config._active_dump._finish_step <= self._frame):
        dump_info = config._active_dump.finalize()
        dump_info['name'] = name
        dumps.append(dump_info)
        config._active_dump = None
    return dumps
