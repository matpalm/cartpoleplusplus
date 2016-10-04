#!/usr/bin/env python
import event_pb2
import gzip
import matplotlib.pyplot as plt
import numpy as np
import StringIO
import struct

def rgb_to_png(rgb):
  """convert RGB data from render to png"""
  sio = StringIO.StringIO()
  plt.imsave(sio, rgb)
  return sio.getvalue()

def png_to_rgb(png_bytes):
  """convert png (from rgb_to_png) to RGB"""
  # note PNG is always RGBA so we need to slice off A
  rgba = plt.imread(StringIO.StringIO(png_bytes))
  return rgba[:,:,:3]

def read_state_from_event(event):
  """unpack state from event (i.e. inverse of add_state_to_event)"""
  if event.state[0].HasField("render"):
    num_repeats = len(event.state)
    eg_render = event.state[0].render
    state = np.empty((eg_render.height, eg_render.width, num_repeats*3))
    for i, s in enumerate(event.state):
      state[:,:,3*i:3*(i+1)] = png_to_rgb(s.render.png_bytes)
  else:
    state = np.empty((len(event.state), 2, 7))
    for i, s in enumerate(event.state):
      state[i][0] = s.cart_pose
      state[i][1] = s.pole_pose
  return state

class EventLog(object):

  def __init__(self, path, use_raw_pixels):
    self.log_file = open(path, "ab")
    self.episode_entry = None
    self.use_raw_pixels = use_raw_pixels

  def reset(self):
    if self.episode_entry is not None:
      # *sigh* have to frame these ourselves :/
      # (a long as a header-len will do...)
      buff = self.episode_entry.SerializeToString()
      if len(buff) > 0:
        buff_len = struct.pack('=l', len(buff))
        self.log_file.write(buff_len)
        self.log_file.write(buff)
        self.log_file.flush()
    self.episode_entry = event_pb2.Episode()

  def add_state_to_event(self, state, event):
    """pack state into event"""
    if self.use_raw_pixels:
      # TODO: be nice to have pose info here too in the pixel case...
      for r in range(state.shape[2] / 3):  # 3 channels per action repeat
        s = event.state.add()
        s.render.width = state.shape[1]
        s.render.height = state.shape[0]
        s.render.png_bytes = rgb_to_png(state[:,:,3*r:3*(r+1)])
    else:
      for r in range(state.shape[0]):
        s = event.state.add()
        s.cart_pose.extend(map(float, state[r][0]))
        s.pole_pose.extend(map(float, state[r][1]))

  def add(self, state, action, reward):
    event = self.episode_entry.event.add()
    self.add_state_to_event(state, event)
    if isinstance(action, int):
      event.action.append(action)  # single action
    else:
      assert action.shape[0] == 1  # never log batch operations
      event.action.extend(map(float, action[0]))
    event.reward = reward

  def add_just_state(self, state):
    event = self.episode_entry.event.add()
    self.add_state_to_event(state, event)


class EventLogReader(object):

  def __init__(self, path):
    if path.endswith(".gz"):
      self.log_file = gzip.open(path, "rb")
    else:
      self.log_file = open(path, "rb")

  def entries(self):
    episode = event_pb2.Episode()
    while True:
      buff_len_bytes = self.log_file.read(4)
      if len(buff_len_bytes) == 0: return
      buff_len = struct.unpack('=l', buff_len_bytes)[0]
      buff = self.log_file.read(buff_len)
      episode.ParseFromString(buff)
      yield episode

def make_dir(d):
  if not os.path.exists(d):
    os.makedirs(d)

if __name__ == "__main__":
  import argparse, os, sys, Image, ImageDraw
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--log-file', type=str, default=None)
  parser.add_argument('--echo', action='store_true', help="write event to stdout")
  parser.add_argument('--episodes', type=str, default=None,
                      help="if set only process these specific episodes (comma separated list)")
  parser.add_argument('--img-output-dir', type=str, default=None,
                      help="if set output all renders to this DIR/e_NUM/s_NUM.png")
  parser.add_argument('--img-debug-overlay', action='store_true',
                      help="if set overlay image with debug info")
  # TODO args for episode range
  opts = parser.parse_args()

  episode_whitelist = None
  if opts.episodes is not None:
    episode_whitelist = set(map(int, opts.episodes.split(",")))

  if opts.img_output_dir is not None:
    make_dir(opts.img_output_dir)

  total_num_read_episodes = 0
  total_num_read_events = 0

  elr = EventLogReader(opts.log_file)
  for episode_id, episode in enumerate(elr.entries()):
    if episode_whitelist is not None and episode_id not in episode_whitelist:
      continue
    if opts.echo:
      print "-----", episode_id
      print episode
    total_num_read_episodes += 1
    total_num_read_events += len(episode.event)
    if opts.img_output_dir is not None:
      dir = "%s/ep_%05d" % (opts.img_output_dir, episode_id)
      make_dir(dir)
      for event_id, event in enumerate(episode.event):
        for state_id, state in enumerate(event.state):
          # open RGB png in an image canvas
          img = Image.open(StringIO.StringIO(state.render.png_bytes))
          if opts.img_debug_overlay:
            canvas = ImageDraw.Draw(img)
            # draw episode and event number in top left
            canvas.text((0, 0), "%d %d" % (episode_id, event_id), fill="black")
            # draw simple fx/fy representation in bottom right...
            # a bounding box
            bx, by, bw = 40, 40, 10
            canvas.line((bx-bw,by-bw, bx+bw,by-bw, bx+bw,by+bw, bx-bw,by+bw, bx-bw,by-bw), fill="black")
            # then a simple fx/fy line
            fx, fy = event.action[0], event.action[1]
            canvas.line((bx,by, bx+(fx*bw), by+(fy*bw)), fill="black")
          # write it out
          img = img.resize((200, 200))
          filename = "%s/ev_%05d_r%d.png" % (dir, event_id, state_id)
          img.save(filename)
  print >>sys.stderr, "read", total_num_read_episodes, "episodes for a total of", total_num_read_events, "events"



