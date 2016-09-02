#!/usr/bin/env python
import event_pb2
import gzip
import struct

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

  def add(self, state, done, action, reward, render=None):
    event = self.episode_entry.event.add()
    assert len(state) == 14
    event.state.pole_pose.extend(state[:7])
    event.state.cart_pose.extend(state[7:])
    event.is_terminal = done
    assert action.shape[0] == 1  # never log batch operations
    event.action.extend(map(float, action[0]))
    event.reward = reward
    if render is not None:
      event.state.render.width = render[0]
      event.state.render.height = render[1]
      event.state.render.rgba = render[2]  # png encoded width x height image

# retreive using
# plt.imread(StringIO.StringIO(event.state.render.rgba))

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

  elr = EventLogReader(opts.log_file)
  for e_id, episode in enumerate(elr.entries()):
    if opts.echo:
      print "-----", e_id
      print episode
    if opts.img_output_dir is not None:
      make_dir("%s/e_%05d" % (opts.img_output_dir, e_id))
      for s_id, event in enumerate(episode.event):
        with open("%s/e_%05d/s_%05d.png" % (opts.img_output_dir, e_id, s_id), "w") as f:
          f.write(event.state.render.rgba)
    if opts.max_process is not None and e_id+1 >= opts.max_process:
      break
  print >>sys.stderr, "read", e_id+1, "episodes"



