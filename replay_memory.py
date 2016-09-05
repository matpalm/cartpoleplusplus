import numpy as np
import sys

class RingBuffer(object):
  def __init__(self, buffer_size, shape=1):
    self.buffer_size = buffer_size
    if isinstance(shape, int):
      # shape is just outer depth
      memory_shape = (buffer_size, shape)
    else:
      # shape is explicitly a shape, add leading buffer size dimension
      memory_shape = tuple([buffer_size]+list(shape))
    self.memory = np.empty(memory_shape)
    self.insert = 0
    self.full = False

  def add(self, entry):
    self.memory[self.insert] = entry
    self.insert += 1
    if self.insert >= self.buffer_size:
      self.insert = 0
      self.full = True

  def random_indexes(self, n=1):
    if self.full:
      return np.random.randint(0, self.buffer_size, n)
    elif self.insert == 0:  # empty
      return []
    else:
      return np.random.randint(0, self.insert, n)

  def size(self):
    return self.buffer_size if self.full else self.insert

  def debug_dump(self):
    for r in xrange(self.buffer_size):
      print "   ", r, self.memory[r]
    print "insert=%s full=%s" % (self.insert, self.full)

class ReplayMemory(object):
  def __init__(self, buffer_size, state_shape, action_dim):
    self.state_1 = RingBuffer(buffer_size, state_shape)
    self.action = RingBuffer(buffer_size, action_dim)
    self.reward = RingBuffer(buffer_size, 1)
    self.terminal_mask = RingBuffer(buffer_size, 1)
    self.state_2 = RingBuffer(buffer_size, state_shape)
    # TODO: write to disk as part of ckpting

  def add(self, s1, a, r, t, s2):
    self.state_1.add(s1)
    self.action.add(a)
    self.reward.add(r)
    # note: if state is terminal (i.e. True) we record a 0 to
    # be used as mask during training
    self.terminal_mask.add(0 if t else 1)
    self.state_2.add(s2)

  def size(self):
    return self.state_1.size()

  def dump(self):
    print "---- State1"
    self.state_1.debug_dump()
    print "---- action"
    self.action.debug_dump()
    print "---- reward"
    self.reward.debug_dump()
    print "---- terminal"
    self.terminal_mask.debug_dump()
    print "---- state2"
    self.state_2.debug_dump()

  def random_batch(self, batch_size):
    idxs = self.state_1.random_indexes(batch_size)
    return (self.state_1.memory[idxs],
            self.action.memory[idxs],
            self.reward.memory[idxs],
            self.terminal_mask.memory[idxs],
            self.state_2.memory[idxs])

