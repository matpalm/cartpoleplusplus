#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import unittest
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

  def test_caching_states(self):
    # insert some
    s0 = [[1,2,3], [4,5,6]]
    idx0 = self.rm.cache_state(s0)
    self.assertEqual(idx0, 0)
    s1 = [[7,8,9],[10,11,12]]
    idx1 = self.rm.cache_state(s1)
    self.assertEqual(idx1, 1)
    # get them back out
    state = self.rm.sess.run(self.rm.state)
    self.assert_np_eq(state[0], s0)
    self.assert_np_eq(state[1], s1)

  def test_single_add(self):
    # add one entry
    s0 = [[1,2,3],[4,5,6]]
    s1_idx = self.rm.cache_state(s0)
    s2_idx = self.rm.add(s1_idx, 11, 12, False, [[4,5,6],[7,8,9]])
    self.assertEqual(s2_idx, 1)

    # api
    self.assertEqual(self.rm.size(), 1)
    self.assertEqual(self.rm.random_indexes(), [0])
    # check batch
    b = self.rm.batch(2)
    self.assert_np_eq(b.s1_idx, [0, 0])
    self.assert_np_eq(b.action, [11, 11])
    self.assert_np_eq(b.reward, [12, 12])
    self.assert_np_eq(b.terminal_mask, [1, 1])
    self.assert_np_eq(b.s2_idx, [1, 1])

    # internals
    self.assertEqual(self.rm.insert, 1)
    self.assertEqual(self.rm.full, False)

  def test_adds_to_full(self):
    # add entries to full
    s0i = self.rm.cache_state([[11,12,13],[14,15,16]])
    s1i = self.rm.add(s0i, 17, 18, False, [[21,22,23],[24,25,26]])
    s2i = self.rm.add(s1i, 27, 28, False, [[31,32,33],[34,35,36]])
    s3i = self.rm.add(s2i, 37, 38, True, [[41,42,43],[44,45,46]])
    # check add gives distinct idxs
    self.assertEqual(len(set([s0i, s1i, s2i, s3i])), 4)

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
    self.assertEqual(state[s0i][0][0], 11)
    self.assertEqual(state[s1i][0][0], 21)
    self.assertEqual(state[s2i][0][0], 31)
    self.assertEqual(state[s3i][0][0], 41)

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

  def __test_soak(self):
    rm = ReplayMemory(self.sess, buffer_size=100, state_shape=(2, 3), action_dim=2, load_factor=1.5)
    self.sess.run(tf.initialize_all_variables())
    def s_for(i):
      return (np.array(range(1,7))+(10*i)).reshape(2, 3)
    import random
    i = 0
    for e in xrange(10000):
      episode_len = random.choice([5,7,9,10,15])
      # add episode
      initial_state = s_for(i)
      action_reward_state = []
      for i in range(i+1, i+episode_len+1):
        a, r, s2 = (i*10)+7, (i*10)+8, s_for(i)
        action_reward_state.append((a, r, s2))
      rm.add_episode(initial_state, action_reward_state)
      i += episode_len + 1
      # dump
      rm.dump_stats()
      # fetch a batch, of all items, but do nothing with it.
      _ = rm.batch(idxs=range(10))



if __name__ == '__main__':
  unittest.main()






















