import collections
import event_log
import numpy as np
import sys
import tensorflow as tf
import time
import util

Batch = collections.namedtuple("Batch", "state_1 action reward terminal_mask state_2")

class ReplayMemory(object):
  def __init__(self, buffer_size, state_shape, action_dim, load_factor=1.5):
    assert load_factor >= 1.5, "load_factor has to be at least 1.5"
    self.buffer_size = buffer_size
    self.state_shape = state_shape
    self.insert = 0
    self.full = False

    # the elements of the replay memory. each event represents a row in the following
    # five matrices.
    self.state_1_idx = np.empty(buffer_size, dtype=np.int32)
    self.action = np.empty((buffer_size, action_dim), dtype=np.float32)
    self.reward = np.empty((buffer_size, 1), dtype=np.float32)
    self.terminal_mask = np.empty((buffer_size, 1), dtype=np.float32)
    self.state_2_idx = np.empty(buffer_size, dtype=np.int32)

    # states themselves, since they can either be state_1 or state_2 in an event
    # are stored in a separate matrix. it is sized fractionally larger than the replay
    # memory since a rollout of length n contains n+1 states.
    self.state_buffer_size = int(buffer_size*load_factor)
    shape = [self.state_buffer_size] + list(state_shape)
    self.state = np.empty(shape, dtype=np.float16)

    # keep track of free slots in state buffer
    self.state_free_slots = list(range(self.state_buffer_size))

    # some stats
    self.stats = collections.Counter()

  def reset_from_event_log(self, log_file):
    elr = event_log.EventLogReader(log_file)
    num_episodes = 0
    num_events = 0
    start = time.time()
    for episode in elr.entries():
      initial_state = None
      action_reward_state_sequence = []
      for event_id, event in enumerate(episode.event):
        if event_id == 0:
          assert len(event.action) == 0
          assert not event.HasField("reward")
          initial_state = event_log.read_state_from_event(event)
        else:
          action_reward_state_sequence.append((event.action, event.reward,
                                               event_log.read_state_from_event(event)))
        num_events += 1
      num_episodes += 1
      self.add_episode(initial_state, action_reward_state_sequence)
      if self.full:
        break
    print >>sys.stderr, "reset_from_event_log \"%s\" num_episodes=%d num_events=%d took %s sec"  % (log_file, num_episodes, num_events, time.time()-start)

  def add_episode(self, initial_state, action_reward_state_sequence):
    self.stats['>add_episode'] += 1
    assert len(action_reward_state_sequence) > 0
    state_1_idx = self.state_free_slots.pop(0)
    self.state[state_1_idx] = initial_state
    for n, (action, reward, state_2) in enumerate(action_reward_state_sequence):
      terminal = n == len(action_reward_state_sequence)-1
      state_2_idx = self._add(state_1_idx, action, reward, terminal, state_2)
      state_1_idx = state_2_idx

  def _add(self, s1_idx, a, r, t, s2):
#    print ">add s1_idx=%s, a=%s, r=%s, t=%s" % (s1_idx, a, r, t)

    self.stats['>add'] += 1
    assert s1_idx >= 0, s1_idx
    assert s1_idx < self.state_buffer_size, s1_idx
    assert s1_idx not in self.state_free_slots, s1_idx

    if self.full:
      # are are about to overwrite an existing entry.
      # we always free the state_1 slot we are about to clobber...
      self.state_free_slots.append(self.state_1_idx[self.insert])
#      print "full; so free slot", self.state_1_idx[self.insert]
      # and we free the state_2 slot also if the slot is a terminal event
      # (since that implies no other event uses this state_2 as a state_1)
#      self.stats['cache_evicted_s1'] += 1
      if self.terminal_mask[self.insert] == 0:
        self.state_free_slots.append(self.state_2_idx[self.insert])
#        print "also, since terminal, free", self.state_2_idx[self.insert]
        self.stats['cache_evicted_s2'] += 1

    # add s1, a, r
    self.state_1_idx[self.insert] = s1_idx
    self.action[self.insert] = a
    self.reward[self.insert] = r

    # if terminal we set terminal mask to 0.0 representing the masking of the righthand
    # side of the bellman equation
    self.terminal_mask[self.insert] = 0.0 if t else 1.0

    # state_2 is fully provided so we need to prepare a new slot for it
    s2_idx = self.state_free_slots.pop(0)
    self.state_2_idx[self.insert] = s2_idx
    self.state[s2_idx] = s2

    # move insert ptr forward
    self.insert += 1
    if self.insert >= self.buffer_size:
      self.insert = 0
      self.full = True

#    print "<add s1_idx=%s, a=%s, r=%s, t=%s s2_idx=%s (free %s)" \
#      % (s1_idx, a, r, t, s2_idx, 
#         util.collapsed_successive_ranges(self.state_free_slots))

    return s2_idx

  def size(self):
    return self.buffer_size if self.full else self.insert

  def random_indexes(self, n=1):
    if self.full:
      return np.random.randint(0, self.buffer_size, n)
    elif self.insert == 0:  # empty
      return []
    else:
      return np.random.randint(0, self.insert, n)

  def batch(self, batch_size=None):
    self.stats['>batch'] += 1
    idxs = self.random_indexes(batch_size)
    return Batch(np.copy(self.state[self.state_1_idx[idxs]]),
                 np.copy(self.action[idxs]),
                 np.copy(self.reward[idxs]),
                 np.copy(self.terminal_mask[idxs]),
                 np.copy(self.state[self.state_2_idx[idxs]]))

  def dump(self):
    print ">>>> dump"
    print "insert", self.insert
    print "full?", self.full
    print "state free slots", util.collapsed_successive_ranges(self.state_free_slots)
    if self.insert==0 and not self.full:
      print "EMPTY!"
    else:
      idxs = range(self.buffer_size if self.full else self.insert)
      for idx in idxs:
        print "idx", idx,
        print "state_1_idx", self.state_1_idx[idx],
        print "state_1", self.state[self.state_1_idx[idx]]
        print "action", self.action[idx],
        print "reward", self.reward[idx],
        print "terminal_mask", self.terminal_mask[idx],
        print "state_2_idx", self.state_2_idx[idx]
        print "state_2", self.state[self.state_2_idx[idx]]
    print "<<<< dump"

  def current_stats(self):
    current_stats = dict(self.stats)
    current_stats["free_slots"] = len(self.state_free_slots)
    return current_stats


if __name__ == "__main__":
  # LATE NIGHT SUPER HACK SOAK TEST. I WILL PAY FOR THIS HACK LATER !!!!
  rm = ReplayMemory(buffer_size=43, state_shape=(2,3), action_dim=2)
  def s(i):  # state for insert i
    i = (i * 10) % 199
    return [[i+1,0,0],[0,0,0]]
  def ars(i):  # action, reward, state_2 for insert i
    return ((i,0), i, s(i))
  def FAILDOG(b, i, d):  # dump batch and rm in case of assertion
    print "FAILDOG", i, d
    print b
    rm.dump()
    assert False   
  def check_batch_valid(b):  # check batch is valid by consistency of how we build elements
    for i in range(3):
      r = int(b.reward[i][0])
      if b.state_1[i][0][0] != (((r-1)*10)%199)+1: FAILDOG(b, i, "s1")
      if b.action[i][0] != r: FAILDOG(b, i, "r")
      if b.terminal_mask[i] != (0 if r in terminals else 1): FAILDOG(b, i, "r")
      if b.state_2[i][0][0] != ((r*10)%199)+1: FAILDOG(b, i, "s2")
  terminals = set()
  i = 0
  import random
  while True:
    initial_state = s(i)
    action_reward_state_sequence = []
    episode_len = int(3 + (random.random() * 5))
    for _ in range(episode_len):
      i += 1
      action_reward_state_sequence.append(ars(i))
    rm.add_episode(initial_state, action_reward_state_sequence)
    terminals.add(i)
    print rm.stats
    for _ in range(7): check_batch_valid(rm.batch(13))
    i += 1
