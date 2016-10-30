#!/usr/bin/env python
import argparse
import bullet_cartpole
import collections
import datetime
import gym
import json
import numpy as np
import replay_memory
import signal
import sys
import tensorflow as tf
import time
import util

np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-eval', type=int, default=0,
                    help="if >0 just run this many episodes with no training")
parser.add_argument('--max-num-actions', type=int, default=0,
                    help="train for (at least) this number of actions (always finish"
                         " current episode) ignore if <=0")
parser.add_argument('--max-run-time', type=int, default=0,
                    help="train for (at least) this number of seconds (always finish"
                         " current episode) ignore if <=0")
parser.add_argument('--ckpt-dir', type=str, default=None,
                    help="if set save ckpts to this dir")
parser.add_argument('--ckpt-freq', type=int, default=3600, help="freq (sec) to save ckpts")
parser.add_argument('--batch-size', type=int, default=128, help="training batch size")
parser.add_argument('--batches-per-step', type=int, default=5,
                    help="number of batches to train per step")
parser.add_argument('--dont-do-rollouts', action="store_true",
                    help="by dft we do rollouts to generate data then train after each rollout. if this flag is set we"
                         " dont do any rollouts. this only makes sense to do if --event-log-in set.")
parser.add_argument('--target-update-rate', type=float, default=0.0001,
                    help="affine combo for updating target networks each time we run a"
                         " training batch")
# TODO params per value, P, output_action networks?
parser.add_argument('--share-input-state-representation', action='store_true',
                    help="if set we have one network for processing input state that is"
                         " shared between value, l_value and output_action networks. if"
                         " not set each net has it's own network.")
parser.add_argument('--hidden-layers', type=str, default="100,50",
                    help="hidden layer sizes")
parser.add_argument('--use-batch-norm', action='store_true',
                    help="whether to use batch norm on conv layers")
parser.add_argument('--discount', type=float, default=0.99,
                    help="discount for RHS of bellman equation update")
parser.add_argument('--event-log-in', type=str, default=None,
                    help="prepopulate replay memory with entries from this event log")
parser.add_argument('--replay-memory-size', type=int, default=22000,
                    help="max size of replay memory")
parser.add_argument('--replay-memory-burn-in', type=int, default=1000,
                    help="dont train from replay memory until it reaches this size")
parser.add_argument('--eval-action-noise', action='store_true',
                    help="whether to use noise during eval")
parser.add_argument('--action-noise-theta', type=float, default=0.01,
                    help="OrnsteinUhlenbeckNoise theta (rate of change) param for action"
                         " exploration")
parser.add_argument('--action-noise-sigma', type=float, default=0.05,
                    help="OrnsteinUhlenbeckNoise sigma (magnitude) param for action"
                         " exploration")
parser.add_argument('--gpu-mem-fraction', type=float, default=None,
                    help="if not none use only this fraction of gpu memory")

util.add_opts(parser)

bullet_cartpole.add_opts(parser)
opts = parser.parse_args()
sys.stderr.write("%s\n" % opts)

# TODO: check that if --dont-do-rollouts set then --event-log-in also set

# TODO: if we import slim before cartpole env we can't start bullet withGL gui o_O
env = bullet_cartpole.BulletCartpole(opts=opts, discrete_actions=False)
import base_network
import tensorflow.contrib.slim as slim

VERBOSE_DEBUG = False
def toggle_verbose_debug(signal, frame):
  global VERBOSE_DEBUG
  VERBOSE_DEBUG = not VERBOSE_DEBUG
signal.signal(signal.SIGUSR1, toggle_verbose_debug)

DUMP_WEIGHTS = False
def set_dump_weights(signal, frame):
  global DUMP_WEIGHTS
  DUMP_WEIGHTS = True
signal.signal(signal.SIGUSR2, set_dump_weights)


class ValueNetwork(base_network.Network):
  """ Value network component of a NAF network. Created as seperate net because it has a target network."""

  def __init__(self, namespace, input_state, hidden_layer_config):
    super(ValueNetwork, self).__init__(namespace)

    self.input_state = input_state

    with tf.variable_scope(namespace):
      # expose self.input_state_representation since it will be the network "shared"
      # by l_value & output_action network when running --share-input-state-representation
      self.input_state_representation = self.input_state_network(input_state, opts)
      self.value = slim.fully_connected(scope='fc',
                                        inputs=self.input_state_representation,
                                        num_outputs=1,
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                        activation_fn=None)  # (batch, 1)

  def value_given(self, state):
    return tf.get_default_session().run(self.value,
                                        feed_dict={self.input_state: state,
                                                   base_network.IS_TRAINING: False})


class NafNetwork(base_network.Network):

  def __init__(self, namespace,
               input_state, input_state_2,
               value_net, target_value_net,
               action_dim):
    super(NafNetwork, self).__init__(namespace)

    # noise to apply to actions during rollouts
    self.exploration_noise = util.OrnsteinUhlenbeckNoise(action_dim,
                                                         opts.action_noise_theta,
                                                         opts.action_noise_sigma)

    # we already have the V networks, created independently because it also
    # has a target network.
    self.value_net = value_net
    self.target_value_net = target_value_net

    # keep placeholders provided and build any others required
    self.input_state = input_state
    self.input_state_2 = input_state_2
    self.input_action = tf.placeholder(shape=[None, action_dim],
                                       dtype=tf.float32, name="input_action")
    self.reward =  tf.placeholder(shape=[None, 1],
                                  dtype=tf.float32, name="reward")
    self.terminal_mask = tf.placeholder(shape=[None, 1],
                                        dtype=tf.float32, name="terminal_mask")

    # TODO: dont actually use terminal mask?

    with tf.variable_scope(namespace):
      # mu (output_action) is also a simple NN mapping input state -> action
      # this is our target op for inference (i.e. value that maximises Q given input_state)
      with tf.variable_scope("output_action"):
        if opts.share_input_state_representation:
          input_representation = value_net.input_state_representation
        else:
          input_representation = self.input_state_network(self.input_state, opts)
        weights_initializer = tf.random_uniform_initializer(-0.001, 0.001)
        self.output_action = slim.fully_connected(scope='fc',
                                                  inputs=input_representation,
                                                  num_outputs=action_dim,
                                                  weights_initializer=weights_initializer,
                                                  weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                  activation_fn=tf.nn.tanh)  # (batch, action_dim)

      # A (advantage) is a bit more work and has three components...
      # first the u / mu difference. note: to use in a matmul we need
      # to convert this vector into a matrix by adding an "unused"
      # trailing dimension
      u_mu_diff = self.input_action - self.output_action  # (batch, action_dim)
      u_mu_diff = tf.expand_dims(u_mu_diff, -1)           # (batch, action_dim, 1)

      # next we have P = L(x).L(x)_T  where L is the values of lower triangular
      # matrix with diagonals exp'd. yikes!

      # first the L lower triangular values; a network on top of the input state
      num_l_values = (action_dim*(action_dim+1))/2
      with tf.variable_scope("l_values"):
        if opts.share_input_state_representation:
          input_representation = value_net.input_state_representation
        else:
          input_representation = self.input_state_network(self.input_state, opts)
        l_values = slim.fully_connected(scope='fc',
                                        inputs=input_representation,
                                        num_outputs=num_l_values,
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                        activation_fn=None)

      # we will convert these l_values into a matrix one row at a time.
      rows = []

      self._l_values = l_values

      # each row is made of three components;
      # 1) the lower part of the matrix, i.e. elements to the left of diagonal
      # 2) the single diagonal element (that we exponentiate)
      # 3) the upper part of the matrix; all zeros
      batch_size = tf.shape(l_values)[0]
      row_idx = 0
      for row_idx in xrange(action_dim):
        row_offset_in_l = (row_idx*(row_idx+1))/2
        lower = tf.slice(l_values, begin=(0, row_offset_in_l), size=(-1, row_idx))
        diag  = tf.exp(tf.slice(l_values, begin=(0, row_offset_in_l+row_idx), size=(-1, 1)))
        upper = tf.zeros((batch_size, action_dim - tf.shape(lower)[1] - 1)) # -1 for diag
        rows.append(tf.concat(1, [lower, diag, upper]))
      # full L matrix is these rows packed.
      L = tf.pack(rows, 0)
      # and since leading axis in l was always the batch
      # we need to transpose it back to axis0 again
      L = tf.transpose(L, (1, 0, 2))  # (batch_size, action_dim, action_dim)
      self.check_L = tf.check_numerics(L, "L")

      # P is L.L_T
      L_T = tf.transpose(L, (0, 2, 1))  # TODO: update tf & use batch_matrix_transpose
      P = tf.batch_matmul(L, L_T)  # (batch_size, action_dim, action_dim)

      # can now calculate advantage
      u_mu_diff_T = tf.transpose(u_mu_diff, (0, 2, 1))
      advantage = -0.5 * tf.batch_matmul(u_mu_diff_T, tf.batch_matmul(P, u_mu_diff))  # (batch, 1, 1)
      # and finally we need to reshape off the axis we added to be able to matmul
      self.advantage = tf.reshape(advantage, [-1, 1])  # (batch, 1)

      # Q is value + advantage
      self.q_value = value_net.value + self.advantage

      # target y is reward + discounted target value
      # TODO: pull discount out
      self.target_y = self.reward + (self.terminal_mask * opts.discount * \
                                     target_value_net.value)
      self.target_y = tf.stop_gradient(self.target_y)

      # loss is squared difference that we want to minimise.
      self.loss = tf.reduce_mean(tf.pow(self.q_value - self.target_y, 2))
      with tf.variable_scope("optimiser"):
        # dynamically create optimiser based on opts
        optimiser = util.construct_optimiser(opts)
        # calc gradients
        gradients = optimiser.compute_gradients(self.loss)
        # potentially clip and wrap with debugging tf.Print
        gradients = util.clip_and_debug_gradients(gradients, opts)
        # apply
        self.train_op = optimiser.apply_gradients(gradients)

      # sanity checks (in the dependent order)
      checks = []
      for op, name in [(l_values, 'l_values'), (L,'L'), (self.loss, 'loss')]:
        checks.append(tf.check_numerics(op, name))
      self.check_numerics = tf.group(*checks)

  def action_given(self, state, add_noise):
    # NOTE: noise is added _outside_ tf graph. we do this simply because the noisy output
    # is never used for any part of computation graph required for online training. it's
    # only used during training after being the replay buffer.
    actions = tf.get_default_session().run(self.output_action,
                                           feed_dict={self.input_state: [state],
                                                      base_network.IS_TRAINING: False})

    if add_noise:
      if VERBOSE_DEBUG:
        pre_noise = str(actions)
      actions[0] += self.exploration_noise.sample()
      actions = np.clip(1, -1, actions)  # action output is _always_ (-1, 1)
      if VERBOSE_DEBUG:
        print "TRAIN action_given pre_noise %s post_noise %s" % (pre_noise, actions)
    return actions

  def train(self, batch):
    _, _, l = tf.get_default_session().run([self.check_numerics, self.train_op, self.loss],
                                 feed_dict={self.input_state: batch.state_1,
                                            self.input_action: batch.action,
                                            self.reward: batch.reward,
                                            self.terminal_mask: batch.terminal_mask,
                                            self.input_state_2: batch.state_2,
                                            base_network.IS_TRAINING: True})
    return l

  def debug_values(self, batch):
    values = tf.get_default_session().run([self._l_values, self.loss, self.value_net.value,
                                           self.advantage, self.target_value_net.value],
                                   feed_dict={self.input_state: batch.state_1,
                                              self.input_action: batch.action,
                                              self.reward: batch.reward,
                                              self.terminal_mask: batch.terminal_mask,
                                              self.input_state_2: batch.state_2,
                                              base_network.IS_TRAINING: False})
    values = [np.squeeze(v) for v in values]
    return values


class NormalizedAdvantageFunctionAgent(object):
  def __init__(self, env):
    self.env = env
    state_shape = self.env.observation_space.shape
    action_dim = self.env.action_space.shape[1]

    # for now, with single machine synchronous training, use a replay memory for training.
    # TODO: switch back to async training with multiple replicas (as in drivebot project)
    self.replay_memory = replay_memory.ReplayMemory(opts.replay_memory_size,
                                                    state_shape, action_dim)

    # s1 and s2 placeholders
    batched_state_shape = [None] + list(state_shape)
    s1 = tf.placeholder(shape=batched_state_shape, dtype=tf.float32)
    s2 = tf.placeholder(shape=batched_state_shape, dtype=tf.float32)

    # initialise base models for value & naf networks. value subportion of net is
    # explicitly created seperate because it has a target network note: in the case of
    # --share-input-state-representation the input state network of the value_net will
    # be reused by the naf.l_value and naf.output_actions net
    self.value_net = ValueNetwork("value", s1, opts.hidden_layers)
    self.target_value_net = ValueNetwork("target_value", s2, opts.hidden_layers)
    self.naf = NafNetwork("naf", s1, s2,
                          self.value_net, self.target_value_net,
                          action_dim)

  def post_var_init_setup(self):
    # prepopulate replay memory (if configured to do so)
    # TODO: rewrite!!!
    if opts.event_log_in:
      self.replay_memory.reset_from_event_log(opts.event_log_in)
    # hook networks up to their targets
    # ( does one off clobber to init all vars in target network )
    self.target_value_net.set_as_target_network_for(self.value_net,
                                                    opts.target_update_rate)

  def run_training(self, max_num_actions, max_run_time, batch_size, batches_per_step,
                   saver_util):
    # log start time, in case we are limiting by time...
    start_time = time.time()

    # run for some max number of actions
    num_actions_taken = 0
    n = 0
    while True:
      rewards = []
      losses = []

      # run an episode
      if opts.dont_do_rollouts:
        # _not_ gathering experience online
        pass
      else:
        # start a new episode
        state_1 = self.env.reset()
        # prepare data for updating replay memory at end of episode
        initial_state = np.copy(state_1)
        action_reward_state_sequence = []
        episode_start = time.time()
        done = False
        while not done:
          # choose action
          action = self.naf.action_given(state_1, add_noise=True)
          # take action step in env
          state_2, reward, done, _ = self.env.step(action)
          rewards.append(reward)
          # cache for adding to replay memory
          action_reward_state_sequence.append((action, reward, np.copy(state_2)))
          # roll state for next step.
          state_1 = state_2
        # at end of episode update replay memory
        print "episode_took", time.time() - episode_start, len(rewards)

        replay_add_start = time.time()
        self.replay_memory.add_episode(initial_state, action_reward_state_sequence)
        print "replay_took", time.time() - replay_add_start

      # do a training step (after waiting for buffer to fill a bit...)
      if self.replay_memory.size() > opts.replay_memory_burn_in:
        # run a set of batches
        for _ in xrange(batches_per_step):
          batch_start = time.time()
          batch = self.replay_memory.batch(batch_size)
          losses.append(self.naf.train(batch))
          print "batch_took", time.time() - batch_start
        # update target nets
        self.target_value_net.update_weights()
        # do debug (if requested) on last batch
        if VERBOSE_DEBUG:
          print "-----"
          print "> BATCH"
          print "state_1", batch.state_1.T
          print "action\n", batch.action.T
          print "reward        ", batch.reward.T
          print "terminal_mask ", batch.terminal_mask.T
          print "state_2", batch.state_2.T
          print "< BATCH"
          l_values, l, v, a, vp = self.naf.debug_values(batch)
          print "> BATCH DEBUG VALUES"
          print "l_values\n", l_values.T
          print "loss\t", l
          print "val\t" , np.mean(v), "\t", v.T
          print "adv\t", np.mean(a), "\t", a.T
          print "val'\t", np.mean(vp), "\t", vp.T
          print "< BATCH DEBUG VALUES"

      # dump some stats and progress info
      stats = collections.OrderedDict()
      stats["time"] = time.time()
      stats["n"] = n
      stats["mean_losses"] = float(np.mean(losses))
      stats["total_reward"] = np.sum(rewards)
      stats["episode_len"] = len(rewards)
      stats["replay_memory_stats"] = self.replay_memory.current_stats()
      print "STATS %s\t%s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                              json.dumps(stats))
      sys.stdout.flush()
      n += 1

      # save if required
      if saver_util is not None:
        saver_util.save_if_required()

      # emit occasional eval
      if VERBOSE_DEBUG or n % 10 == 0:
        self.run_eval(1)

      # dump weights once if requested
      global DUMP_WEIGHTS
      if DUMP_WEIGHTS:
        self.debug_dump_network_weights()
        DUMP_WEIGHTS = False

      # exit when finished
      num_actions_taken += len(rewards)
      if max_num_actions > 0 and num_actions_taken > max_num_actions:
        break
      if max_run_time > 0 and time.time() > start_time + max_run_time:
        break

  def run_eval(self, num_episodes, add_noise=False):
    """ run num_episodes of eval and output episode length and rewards """
    for i in xrange(num_episodes):
      state = self.env.reset()
      total_reward = 0
      steps = 0
      done = False
      while not done:
        action = self.naf.action_given(state, add_noise)
        state, reward, done, _ = self.env.step(action)
        print "EVALSTEP e%d s%d action=%s (l2=%s) => reward %s" % (i, steps, action,
                                                                   np.linalg.norm(action), reward)
        total_reward += reward
        steps += 1
        if False:  # RENDER ALL STATES / ACTIVATIONS to /tmp
          self.naf.render_all_convnet_activations(steps, self.naf.input_state, state)
          util.render_state_to_png(steps, state)
          util.render_action_to_png(steps, action)
      print "EVAL", i, steps, total_reward
    sys.stdout.flush()

  def debug_dump_network_weights(self):
    fn = "/tmp/weights.%s" % time.time()
    with open(fn, "w") as f:
      f.write("DUMP time %s\n" % time.time())
      for var in tf.all_variables():
        f.write("VAR %s %s\n" % (var.name, var.get_shape()))
        f.write("%s\n" % var.eval())
    print "weights written to", fn


def main():
  config = tf.ConfigProto()
#  config.gpu_options.allow_growth = True
#  config.log_device_placement = True
  if opts.gpu_mem_fraction is not None:
    config.gpu_options.per_process_gpu_memory_fraction = opts.gpu_mem_fraction
  with tf.Session(config=config) as sess:
    agent = NormalizedAdvantageFunctionAgent(env=env)

    # setup saver util and either load latest ckpt or init variables
    saver_util = None
    if opts.ckpt_dir is not None:
      saver_util = util.SaverUtil(sess, opts.ckpt_dir, opts.ckpt_freq)
    else:
      sess.run(tf.initialize_all_variables())

    for v in tf.all_variables():
      print >>sys.stderr, v.name, util.shape_and_product_of(v)

    # now that we've either init'd from scratch, or loaded up a checkpoint,
    # we can do any required post init work.
    agent.post_var_init_setup()

    # run either eval or training
    if opts.num_eval > 0:
      agent.run_eval(opts.num_eval, opts.eval_action_noise)
    else:
      agent.run_training(opts.max_num_actions, opts.max_run_time,
                         opts.batch_size, opts.batches_per_step,
                         saver_util)
      if saver_util is not None:
        saver_util.force_save()

    env.reset()  # just to flush logging, clumsy :/

if __name__ == "__main__":
  main()
