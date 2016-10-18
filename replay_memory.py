import collections
import event_log
import numpy as np
import sys
import tensorflow as tf
import time
import util

Batch = collections.namedtuple("Batch", "state_1_idx action reward terminal_mask state_2_idx")

class ReplayMemory(object):
  def __init__(self, sess, buffer_size, state_shape, action_dim, load_factor=1.5):
    assert load_factor >= 1.5, "load_factor has to be at least 1.5"
    self.sess = sess
    self.buffer_size = buffer_size
    self.state_shape = state_shape
    self.insert = 0
    self.full = False

    # the elements of the replay memory. in main memory we only keep idxs to states
    # stored in tf variables (which gives the option of being placed in GPU memory)
    self.state_1_idx = np.empty(buffer_size, dtype=np.int32)
    self.action = np.empty((buffer_size, action_dim), dtype=np.float32)
    self.reward = np.empty((buffer_size, 1), dtype=np.float32)
    self.terminal_mask = np.empty((buffer_size, 1), dtype=np.float32)
    self.state_2_idx = np.empty(buffer_size, dtype=np.int32)
    # TODO: write to disk as part of ckpting

    # init variable in tf graph representing storage of states. this needs to be
    # fractionally larger than the replay memory since a rollout of length n contains
    # n+1 states.
    self.state_buffer_size = int(buffer_size*load_factor)
    with tf.variable_scope("replay_memory"):
      shape = [self.state_buffer_size] + list(state_shape)
      self.state = tf.Variable(tf.zeros(shape), trainable=False, name="state_buffer")

    # create an op, and corresponding placeholders, used for inserting a single entry into
    # state cache.
    self.state_idxs = idx = tf.placeholder(tf.int32, shape=[None])
    self.state_updates = tf.placeholder(tf.float32, shape=[None]+list(state_shape))
    self.update_op = tf.scatter_update(self.state,
                                       indices=self.state_idxs,
                                       updates=self.state_updates)

    # keep track of free slots in state buffer
    self.state_free_slots = list(range(self.state_buffer_size))

    # placeholders and indexing into state. we will only index into state based on
    # a 1d batch of indexes so we are explicit about shape here so gather can
    # infer correct shape for downstream consumers.
    self.batch_state_1_idx = tf.placeholder(shape=(None,), dtype=tf.int32)
    self.batch_state_1 = tf.gather(self.state, self.batch_state_1_idx)
    self.batch_state_2_idx = tf.placeholder(shape=(None,), dtype=tf.int32)
    self.batch_state_2 = tf.gather(self.state, self.batch_state_2_idx)

    # some stats
    self.stats = collections.Counter()

  def reset_from_event_log(self, event_log_file):
    """ resets contents from event_log.
        assumes event_log will fit in main memory.
        only uses first self.buffer_size entries of log"""
    start_time = time.time()
    elr = event_log.EventLogReader(event_log_file)
    new_state = np.empty(self.state.get_shape())  # will clobber self.state
    # see _add method for more info
    new_state_idx = 0
    self.insert = 0
    for episode in elr.entries():
      for i, event in enumerate(episode.event):
#        print ">event", i
        # add state to tmp buffer
        state = event_log.read_state_from_event(event)
        new_state[new_state_idx] = state
        if i == 0:
          # first event is just for state; i.e. should be no action or reward
          assert len(event.action) == 0
          assert not event.HasField("reward")
        else:
          # add event to replay memory
#          print "s1_idx", new_state_idx-1
          self.state_1_idx[self.insert] = new_state_idx - 1
          self.action[self.insert] = event.action
          self.reward[self.insert] = event.reward
          is_last_event = i == len(episode.event)-1
          self.terminal_mask[self.insert] = 0 if is_last_event else 1
          self.state_2_idx[self.insert] = new_state_idx
#          print "inserted s1=%s a=%s r=%s t=%s s2=%s" % (new_state_idx-1, event.action, event.reward, is_last_event, new_state_idx)
          self.insert += 1
          if self.insert == self.buffer_size:
#            print "insert buffer full!"
            break
        # roll over event
        new_state_idx += 1
        if new_state_idx == self.state_buffer_size:
#          print "state_buffer is full!"
          break
#        print "<event"
      if self.insert == self.buffer_size:
#        print "insert buffer full!!!!"
        break
#      print "<episode"

    # assign variable in one hit.
#    print "self.state.get_shape()", self.state.get_shape()
#    print "new_state.shape", new_state.shape
#    print ">sess.run assign state"
    self.sess.run(self.state.assign(new_state))
#    print "<sess.run assign state"
    self.full = self.insert == self.buffer_size
    if self.full: self.insert = 0
    self.state_free_slots = list(range(new_state_idx, self.state_buffer_size))
    self.stats = collections.Counter()
    end_time = time.time()

    print "prepopulated event_log from [%s] in %f sec." \
          " #states_loaded=%d replay_buffer.full=%s" % (event_log_file, (end_time - start_time), new_state_idx, self.full)

  def batch_ops(self):
    """ returns placeholders for idxs (and corresponding opts gathering over those idxs) for state1 and state2"""
    return (self.batch_state_1, self.batch_state_1_idx,
            self.batch_state_2, self.batch_state_2_idx)

  def add_episode(self, initial_state, action_reward_state_sequence):
    self.stats['>add_episode'] += 1
    assert len(action_reward_state_sequence) > 0

    state_1_idx = self.state_free_slots.pop(0)
    update_slots = [state_1_idx]
    update_states = [initial_state]

    for n, (action, reward, state_2) in enumerate(action_reward_state_sequence):
      terminal = n == len(action_reward_state_sequence)-1
      state_2_idx = self._add(state_1_idx, action, reward, terminal, state_2)
      update_slots.append(state_2_idx)
      update_states.append(state_2)
      state_1_idx = state_2_idx  # be explicit about rolling

    # update variable in one call for entire episode
    self.sess.run(self.update_op,
                  feed_dict={self.state_idxs: update_slots,
                             self.state_updates: update_states})

  def _add(self, s1_idx, a, r, t, s2):
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
      self.stats['cache_evicted_s1'] += 1
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

    # move insert ptr forward
    self.insert += 1
    if self.insert >= self.buffer_size:
      self.insert = 0
      self.full = True

#    print ">add s1_idx=%s, a=%s, r=%s, t=%s s2_idx=%s (free %s)" % (s1_idx, a, r, t, s2_idx, self.state_free_slots)

    # return s2 idx
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

  def batch(self, batch_size=None, idxs=None):
    self.stats['>batch'] += 1
    assert (batch_size is None) ^ (idxs is None)
    if batch_size: idxs = self.random_indexes(batch_size)
    return Batch(self.state_1_idx[idxs],
                 self.action[idxs],
                 self.reward[idxs],
                 self.terminal_mask[idxs],
                 self.state_2_idx[idxs])

  def dump(self):
    print ">>>> dump"
    print "insert", self.insert
    print "full?", self.full
    print "state free slots", util.collapsed_successive_ranges(self.state_free_slots)
    if self.insert==0 and not self.full:
      print "EMPTY!"
      return
    entire_state = self.sess.run(self.state)
    idxs = range(self.buffer_size if self.full else self.insert)
    for idx in idxs:
      print "idx", idx,
      print "state_1_idx", self.state_1_idx[idx],
#      print "state_1", entire_state[self.state_1_idx[idx]]
      print "action", self.action[idx],
      print "reward", self.reward[idx],
      print "terminal_mask", self.terminal_mask[idx],
      print "state_2_idx", self.state_2_idx[idx]
#      print "state_2", entire_state[self.state_2_idx[idx]]
    print "<<<< dump"

  def current_stats(self):
    current_stats = dict(self.stats)
    current_stats["free_slots"] = len(self.state_free_slots)
    return current_stats


if __name__ == "__main__":
  with tf.Session() as sess:
    rm = ReplayMemory(sess, buffer_size=10, state_shape=(2,2,7),
                      action_dim=2)
    sess.run(tf.initialize_all_variables())
    rm.dump()
    rm.reset_from_event_log("events")
    rm.dump()
