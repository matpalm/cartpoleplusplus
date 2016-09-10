import collections
import numpy as np
import sys
import tensorflow as tf

Batch = collections.namedtuple("Batch", "s1_idx action reward terminal_mask s2_idx")

class ReplayMemory(object):
  def __init__(self, sess, buffer_size, state_shape, action_dim, load_factor=1.5):
    assert load_factor >= 1.5, "load_factor has to be at least 1.5"
    self.sess = sess
    self.buffer_size = buffer_size
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
    self.state_idx = idx = tf.placeholder(tf.int32)
    self.state_update = tf.placeholder(tf.float32, shape=state_shape)
    self.update_op = tf.scatter_update(self.state, indices=[self.state_idx],
                                       updates=[self.state_update])

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

  def batch_ops(self):
    """ returns placeholders for idxs (and corresponding opts gathering over those idxs) for state1 and state2"""
    return (self.batch_state_1, self.batch_state_1_idx,
            self.batch_state_2, self.batch_state_2_idx)

  def cache_state(self, state):
    """ add a single state to memory & return index. used for first state in episode """
    self.stats['>cache_state'] += 1
    assert state is not None
    if len(self.state_free_slots) == 0:
      raise Exception("out of memory for state buffer; decrease buffer_size (or include load_factor)")
    slot = self.state_free_slots.pop(0)
    self.sess.run(self.update_op,
                  feed_dict={self.state_idx: slot,
                             self.state_update: state})
    return slot

  def add_episode(self, initial_state, action_reward_state_sequence):
    self.stats['>add_episode'] += 1
    assert len(action_reward_state_sequence) > 0
    state_1_idx = self.cache_state(initial_state)
    for n, (action, reward, state_2) in enumerate(action_reward_state_sequence):
      terminal = n == len(action_reward_state_sequence)-1
      state_2_idx = self.add(state_1_idx, action, reward, terminal, state_2)
      state_1_idx = state_2_idx  # be explicit about rolling

  def add(self, s1_idx, a, r, t, s2):
    self.stats['>add'] += 1
    assert s1_idx >= 0, s1_idx
    assert s1_idx < self.state_buffer_size, s1_idx
    assert s1_idx not in self.state_free_slots, s1_idx

    if self.full:
      # are are about to overwrite an existing entry.
      # we always free the state_1 slot we are about to clobber...
      self.state_free_slots.append(self.state_1_idx[self.insert])
      # and we free the state_2 slot also if the slot is a terminal event
      # (since that implies no other event uses this state_2 as a state_1)
      self.stats['cache_evicted_s1'] += 1
      if self.terminal_mask[self.insert] == 0:
        self.state_free_slots.append(self.state_2_idx[self.insert])
        self.stats['cache_evicted_s2'] += 1

    # add s1, a, r
    self.state_1_idx[self.insert] = s1_idx
    self.action[self.insert] = a
    self.reward[self.insert] = r

    # if terminal we set terminal mask to 0.0 representing the masking of the righthand
    # side of the bellman equation
    self.terminal_mask[self.insert] = 0.0 if t else 1.0

    # state_2 is fully provided so we need to cache it and store the reference index
    s2_idx = self.cache_state(s2)
    self.state_2_idx[self.insert] = s2_idx

    # move insert ptr forward
    self.insert += 1
    if self.insert >= self.buffer_size:
      self.insert = 0
      self.full = True

    # return s2 ptr
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
    print ">dump"
    print "insert", self.insert
    print "full?", self.full
    print "state free slots", self.state_free_slots
    if self.insert==0 and not self.full:
      print "EMPTY!"
      return
    idxs = range(self.buffer_size if self.full else self.insert)
    for idx in idxs:
      print "- idx", idx,
      print "-- state_1_idx", self.state_1_idx[idx],
      print self.state.eval(session=self.sess)[self.state_1_idx[idx]]
      print "-- action", self.action[idx]
      print "-- reward", self.reward[idx]
      print "-- state_2_idx", self.state_2_idx[idx],
      print "-- terminal_mask", self.terminal_mask[idx]
      print self.state.eval(session=self.sess)[self.state_2_idx[idx]]

  def current_stats(self):
    current_stats = dict(self.stats)
    current_stats["free_slots"] = len(self.state_free_slots)
    return current_stats


