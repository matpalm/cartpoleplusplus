#!/usr/bin/env python

import event_pb2
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

  def add(self, state, done, action, reward):
    event = self.episode_entry.event.add()
    assert len(state) == 14
    event.state.pole_pose.extend(state[:7])
    event.state.cart_pose.extend(state[7:])
    event.is_terminal = done
    assert action.shape[0] == 1  # never log batch operations
    event.action.extend(map(float, action[0]))
    event.reward = reward

class EventLogReader(object):
  
  def __init__(self, path):
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

if __name__ == "__main__":
  import sys
  for episode in EventLogReader(sys.argv[1]).entries():
    print "-----"
    print episode
    
    
    
