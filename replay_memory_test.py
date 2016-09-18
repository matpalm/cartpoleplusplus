#!/usr/bin/env python
import numpy as np
import random
import tensorflow as tf
import time
import unittest
from util import StopWatch
from replay_memory import ReplayMemory

class TestReplayMemory(unittest.TestCase):
  def setUp(self):
    self.sess = tf.Session()
    self.rm = ReplayMemory(self.sess, buffer_size=3, state_shape=(2, 3), action_dim=2, load_factor=2)
    self.sess.run(tf.initialize_all_variables())

  def assert_np_eq(self, a, b):
    self.assertTrue(np.all(np.equal(a, b)))

  def test_empty_memory(self):
    # api
    self.assertEqual(self.rm.size(), 0)
    self.assertEqual(self.rm.random_indexes(), [])
    b = self.rm.batch(4)
    self.assertEqual(len(b), 5)
    for i in range(5):
      self.assertEqual(len(b[i]), 0)

    # internals
    self.assertEqual(self.rm.insert, 0)
    self.assertEqual(self.rm.full, False)

  def test_adds_to_full(self):
    # add entries to full
    initial_state = [[11,12,13], [14,15,16]]
    action_reward_state = [(17, 18, [[21,22,23],[24,25,26]]),
                           (27, 28, [[31,32,33],[34,35,36]]),
                           (37, 38, [[41,42,43],[44,45,46]])]
    self.rm.add_episode(initial_state, action_reward_state)

    # api
    self.assertEqual(self.rm.size(), 3)
    # random_idxs are valid
    idxs = self.rm.random_indexes(n=100)
    self.assertEqual(len(idxs), 100)
    self.assertEquals(sorted(set(idxs)), [0,1,2])
    # batch returns values

    # internals
    self.assertEqual(self.rm.insert, 0)
    self.assertEqual(self.rm.full, True)
    # check state contains these entries
    state = self.rm.sess.run(self.rm.state)
    self.assertEqual(state[0][0][0], 11)
    self.assertEqual(state[1][0][0], 21)
    self.assertEqual(state[2][0][0], 31)
    self.assertEqual(state[3][0][0], 41)

  def test_adds_over_full(self):
    def s_for(i):
      return (np.array(range(1,7))+(10*i)).reshape(2, 3)

    # add one episode of 5 states; 0X -> 4X
    initial_state = s_for(0)
    action_reward_state = []
    for i in range(1, 5):
      a, r, s2 = (i*10)+7, (i*10)+8, s_for(i)
      action_reward_state.append((a, r, s2))
    self.rm.add_episode(initial_state, action_reward_state)
    # add another episode of 4 states; 5X -> 8X
    initial_state = s_for(5)
    action_reward_state = []
    for i in range(6, 9):
      a, r, s2 = (i*10)+7, (i*10)+8, s_for(i)
      action_reward_state.append((a, r, s2))
    self.rm.add_episode(initial_state, action_reward_state)

    # api
    self.assertEqual(self.rm.size(), 3)
    # random_idxs are valid
    idxs = self.rm.random_indexes(n=100)
    self.assertEqual(len(idxs), 100)
    self.assertEquals(sorted(set(idxs)), [0,1,2])
    # fetch a batch, of all items
    batch = self.rm.batch(idxs=[0,1,2])
    self.assert_np_eq(batch.reward, [[88], [68], [78]])
    self.assert_np_eq(batch.terminal_mask, [[0], [1], [1]])

  def test_large_var(self):
    ### python replay_memory_test.py TestReplayMemory.test_large_var

    s = StopWatch()

    state_shape = (50, 50, 6)
    s.reset()
    rm = ReplayMemory(self.sess, buffer_size=10000, state_shape=state_shape, action_dim=2, load_factor=1.5)
    self.sess.run(tf.initialize_all_variables())
    print "cstr_and_init", s.time()

    bs1, bs1i, bs2, bs2i = rm.batch_ops()

    # build a simple, useless, net that uses state_1 & state_2 idxs
    # we want this to reduce to a single value to minimise data coming
    # back from GPU
    added_states = bs1 + bs2
    total_value = tf.reduce_sum(added_states)

    def random_s():
      return np.random.random(state_shape)

    for i in xrange(10):
      # add an episode to rm
      episode_len = random.choice([5,7,9,10,15])
      initial_state = random_s()
      action_reward_state = []
      for i in range(i+1, i+episode_len+1):
        a, r, s2 = (i*10)+7, (i*10)+8, random_s()
        action_reward_state.append((a, r, s2))
      start = time.time()
      s.reset()
      rm.add_episode(initial_state, action_reward_state)
      t = s.time()
      num_states = len(action_reward_state)+1
      print "add_episode_time", t, "#states=", num_states, "=> s/state", t/num_states
      i += episode_len + 1

      # get a random batch state
      b = rm.batch(batch_size=128)
      s.reset()
      x = self.sess.run(total_value, feed_dict={bs1i: b.state_1_idx, 
                                                bs2i: b.state_2_idx})
      print "fetch_and_run", x, s.time()


  def test_soak(self):
    state_shape = (50,50,6)
    rm = ReplayMemory(self.sess, buffer_size=10000, 
                      state_shape=state_shape, action_dim=2, load_factor=1.5)
    self.sess.run(tf.initialize_all_variables())
    def s_for(i):
      return np.random.random(state_shape)
    import random
    i = 0
    for e in xrange(10000):
      # add an episode to rm
      episode_len = random.choice([5,7,9,10,15])
      initial_state = s_for(i)
      action_reward_state = []
      for i in range(i+1, i+episode_len+1):
        a, r, s2 = (i*10)+7, (i*10)+8, s_for(i)
        action_reward_state.append((a, r, s2))
      rm.add_episode(initial_state, action_reward_state)
      i += episode_len + 1
      # dump
      print rm.current_stats()
      # fetch a batch, of all items, but do nothing with it.
      _ = rm.batch(idxs=range(10))



if __name__ == '__main__':
  unittest.main()






















