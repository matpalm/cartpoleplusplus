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
    for row in state:
      s = event.state.add()
      s.pole_pose.extend(map(float, row[:7]))
      s.cart_pose.extend(map(float, row[7:]))
    event.is_terminal = done
    if isinstance(action, int):
      event.action.append(action)  # single action
    else:
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
  elr = EventLogReader(sys.argv[1])
  for i, episode in enumerate(elr.entries()):
    print "-----", i
    print episode
    
    
    
