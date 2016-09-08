#!/usr/bin/env python
import event_pb2
import gzip
import matplotlib.pyplot as plt
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
  import argparse, os, sys
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--log-file', type=str, default=None)
  parser.add_argument('--echo', action='store_true', help="write event to stdout")
  parser.add_argument('--max-process', type=int, help="if set only process this many")
  parser.add_argument('--img-output-dir', type=str, default=None,
                      help="if set output all renders to this DIR/e_NUM/s_NUM.png")
  # TODO args for episode range
  opts = parser.parse_args()

  if opts.img_output_dir is not None:
    make_dir(opts.img_output_dir)

  total_num_events = 0
  elr = EventLogReader(opts.log_file)
  for episode_id, episode in enumerate(elr.entries()):
    total_num_events += len(episode.event)
    if opts.echo:
      print "-----", episode_id
      print episode
    if opts.img_output_dir is not None:
      dir = "%s/ep_%05d" % (opts.img_output_dir, episode_id)
      make_dir(dir)
      for event_id, event in enumerate(episode.event):
        for state_id, state in enumerate(event.state):
          assert state.render.png_bytes
          filename = "%s/ev_%05d_r%d.png" % (dir, event_id, state_id)
          with open(filename, "w") as f:
            f.write(state.render.png_bytes)
    if opts.max_process is not None and e_id+1 >= opts.max_process:
      break
  print >>sys.stderr, "read", episode_id+1, "episodes for a total of", total_num_events, "events"



