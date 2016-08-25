import numpy as np

class RingBuffer(object):
  # 2d fill only ring buffer over 2d ndarray
  def __init__(self, buffer_size, depth=1):
    self.buffer_size = buffer_size
    self.depth = depth
    self.memory = np.empty((buffer_size, depth))
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

  def size(self):  # number of entries
    return self.buffer_size if self.full else self.insert

  def debug_dump(self):
    for r in xrange(len(self.memory)):
      print "   ", r, self.memory[r]
    print "insert=%s full=%s" % (self.insert, self.full)

class ReplayMemory(object):
  def __init__(self, buffer_size, state_dim, action_dim):
    self.state_1 = RingBuffer(buffer_size, state_dim)
    self.action = RingBuffer(buffer_size, action_dim)
    self.reward = RingBuffer(buffer_size, 1)
    self.state_2 = RingBuffer(buffer_size, state_dim)

  def add(self, s1, a, r, s2):
    self.state_1.add(s1)
    self.action.add(a)
    self.reward.add(r)
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
    print "---- state2"
    self.state_2.debug_dump()

  def batch(self, batch_size):
    idxs = self.state_1.random_indexes(batch_size)
    return (self.state_1.memory[idxs],
            self.action.memory[idxs],
            self.reward.memory[idxs],
            self.state_2.memory[idxs])

