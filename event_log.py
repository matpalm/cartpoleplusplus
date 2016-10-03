#!/usr/bin/env python
import event_pb2
import gzip
import matplotlib.pyplot as plt
import numpy as np
import StringIO
import struct

def rgb_to_png(rgb):
  sio = StringIO.StringIO()
  plt.imsave(sio, rgb)
  return sio.getvalue()

def png_to_rgb(png_bytes):
  # note PNG is always RGBA so we need to slice off A
  rgba = plt.imread(StringIO.StringIO(png_bytes))
  return rgba[:,:,:3]


def add_state_to_event(state, event, use_raw_pixels):
  """ serialises state and adds to event."""
  if use_raw_pixels:
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

def read_state_from_event(event, use_raw_pixels):
  """ reads serialised state from event and returns in form equiv to add_state_to_event"""
  if use_raw_pixels:
    raise Exception("todo")
  else:
    state = np.empty((len(event.state), 2, 7))
    for i, s in enumerate(event.state):
      state[i][0] = s.cart_pose
      state[i][1] = s.pole_pose
    return state
  

class EventLog(object):

  def __init__(self, path):
    self.log_file = open(path, "ab")
    self.episode_entry = None

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

  def add(self, state, use_raw_pixels, done, action, reward):
    event = self.episode_entry.event.add()
    add_state_to_event(state, event, use_raw_pixels)
    event.is_terminal = done
    if isinstance(action, int):
      event.action.append(action)  # single action
    else:
      assert action.shape[0] == 1  # never log batch operations
      event.action.extend(map(float, action[0]))
    event.reward = reward

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
      # > DEBUG
      for event in episode.event:
        print read_state_from_event(event, False)
      # < DEBUG
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



