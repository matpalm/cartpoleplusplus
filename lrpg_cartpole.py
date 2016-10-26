#!/usr/bin/env python
import argparse
import bullet_cartpole
import collections
import datetime
import gym
import json
import numpy as np
import signal
import sys
import tensorflow as tf
from tensorflow.python.ops import init_ops
import time
import util

np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-eval', type=int, default=0,
                    help="if >0 just run this many episodes with no training")
parser.add_argument('--max-num-actions', type=int, default=0,
                    help="train for (at least) this number of actions (always finish current episode)"
                         " ignore if <=0")
parser.add_argument('--max-run-time', type=int, default=0,
                    help="train for (at least) this number of seconds (always finish current episode)"
                         " ignore if <=0")
parser.add_argument('--ckpt-dir', type=str, default=None, help="if set save ckpts to this dir")
parser.add_argument('--ckpt-freq', type=int, default=3600, help="freq (sec) to save ckpts")
parser.add_argument('--hidden-layers', type=str, default="100,50", help="hidden layer sizes")
parser.add_argument('--learning-rate', type=float, default=0.0001, help="learning rate")
parser.add_argument('--num-train-batches', type=int, default=10,
                    help="number of training batches to run")
parser.add_argument('--rollouts-per-batch', type=int, default=10,
                    help="number of rollouts to run for each training batch")
parser.add_argument('--eval-action-noise', action='store_true', help="whether to use noise during eval")

util.add_opts(parser)

bullet_cartpole.add_opts(parser)
opts = parser.parse_args()
sys.stderr.write("%s\n" % opts)
assert not opts.use_raw_pixels, "TODO: add convnet from ddpg here"

# TODO: if we import slim _before_ building cartpole env we can't start bullet with GL gui o_O
env = bullet_cartpole.BulletCartpole(opts=opts, discrete_actions=True)
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


class LikelihoodRatioPolicyGradientAgent(base_network.Network):

  def __init__(self, env):
    self.env = env

    num_actions = self.env.action_space.n

    # we have three place holders we'll use...
    # observations; used either during rollout to sample some actions, or
    # during training when combined with actions_taken and advantages.
    shape_with_batch = [None] + list(self.env.observation_space.shape)
    self.observations = tf.placeholder(shape=shape_with_batch,
                                       dtype=tf.float32)
    # the actions we took during rollout
    self.actions = tf.placeholder(tf.int32, name='actions')
    # the advantages we got from taken 'action_taken' in 'observation'
    self.advantages = tf.placeholder(tf.float32, name='advantages')

    # our model is a very simple MLP
    with tf.variable_scope("model"):
      # stack of hidden layers on flattened input; (batch,2,2,7) -> (batch,28)
      flat_input_state = slim.flatten(self.observations, scope='flat')
      final_hidden = self.hidden_layers_starting_at(flat_input_state,
                                                    opts.hidden_layers)
      logits = slim.fully_connected(inputs=final_hidden,
                                    num_outputs=num_actions,
                                    activation_fn=None)

    # in the eval case just pick arg max
    self.action_argmax = tf.argmax(logits, 1)

    # for rollouts we need an op that samples actions from this
    # model to give a stochastic action.
    sample_action = tf.multinomial(logits, num_samples=1)
    self.sampled_action_op = tf.reshape(sample_action, shape=[])

    # we are trying to maximise the product of two components...
    # 1) the log_p of "good" actions.
    # 2) the advantage term based on the rewards from actions.

    # first we need the log_p values for each observation for the actions we specifically
    # took by sampling... we first run a log_softmax over the action logits to get
    # probabilities.
    log_softmax = tf.nn.log_softmax(logits)
    self.debug_softmax = tf.exp(log_softmax)

    # we then use a mask to only select the elements of the softmaxs that correspond
    # to the actions we actually took. we could also do this by complex indexing and a
    # gather but i always think this is more natural. the "cost" of dealing with the
    # mostly zero one hot, as opposed to doing a gather on sparse indexes, isn't a big
    # deal when the number of observations is >> number of actions.
    action_mask = tf.one_hot(indices=self.actions, depth=num_actions)
    action_log_prob = tf.reduce_sum(log_softmax * action_mask, reduction_indices=1)

    # the (element wise) product of these action log_p's with the total reward of the
    # episode represents the quantity we want to maximise. we standardise the advantage
    # values so roughly 1/2 +ve / -ve as a variance control.
    action_mul_advantages = tf.mul(action_log_prob,
                                   util.standardise(self.advantages))
    self.loss = -tf.reduce_sum(action_mul_advantages)  # recall: we are maximising.
    with tf.variable_scope("optimiser"):
      # dynamically create optimiser based on opts
      optimiser = util.construct_optimiser(opts)
      # calc gradients
      gradients = optimiser.compute_gradients(self.loss)
      # potentially clip and wrap with debugging tf.Print
      gradients = util.clip_and_debug_gradients(gradients, opts)
      # apply
      self.train_op = optimiser.apply_gradients(gradients)

  def sample_action_given(self, observation, doing_eval=False):
    """ sample one action given observation"""
    if doing_eval:
      sao, sm = tf.get_default_session().run([self.sampled_action_op, self.debug_softmax],
                                             feed_dict={self.observations: [observation]})
      print "EVAL sm ", sm, "action", sao
      return sao

    # epilson greedy "noise" will do for this simple case..
    if np.random.random() < 0.1:
      return self.env.action_space.sample()

    # sample from logits
    return tf.get_default_session().run(self.sampled_action_op,
                                        feed_dict={self.observations: [observation]})


  def rollout(self, doing_eval=False):
    """ run one episode collecting observations, actions and advantages"""
    observations, actions, rewards = [], [], []
    observation = self.env.reset()
    done = False
    while not done:
      observations.append(observation)
      action = self.sample_action_given(observation, doing_eval)
      assert action != 5, "FAIL! (multinomial logits sampling bug?"
      observation, reward, done, _ = self.env.step(action)
      actions.append(action)
      rewards.append(reward)
    if VERBOSE_DEBUG:
      print "rollout: actions=%s" % (actions)
    return observations, actions, rewards

  def train(self, observations, actions, advantages):
    """ take one training step given observations, actions and subsequent advantages"""
    if VERBOSE_DEBUG:
      print "TRAIN"
      print "observations", np.stack(observations)
      print "actions", actions
      print "advantages", advantages
      _, loss = tf.get_default_session().run([self.train_op, self.loss],
                                             feed_dict={self.observations: observations,
                                                        self.actions: actions,
                                                        self.advantages: advantages})

    else:
      _, loss = tf.get_default_session().run([self.train_op, self.loss],
                                             feed_dict={self.observations: observations,
                                                        self.actions: actions,
                                                        self.advantages: advantages})
    return float(loss)

  def post_var_init_setup(self):
    pass

  def run_training(self, max_num_actions, max_run_time, rollouts_per_batch,
                   saver_util):
    # log start time, in case we are limiting by time...
    start_time = time.time()

    # run for some max number of actions
    num_actions_taken = 0
    n = 0
    while True:
      total_rewards = []
      losses = []

      # perform a number of rollouts
      batch_observations, batch_actions, batch_advantages = [], [], []

      for _ in xrange(rollouts_per_batch):
        observations, actions, rewards = self.rollout()
        batch_observations += observations
        batch_actions += actions
        # train with advantages, not per observation/action rewards.
        # _every_ observation/action in this rollout gets assigned
        # the _total_ reward of the episode. (crazy that this works!)
        batch_advantages += [sum(rewards)] * len(rewards)
        # keep total rewards just for debugging / stats
        total_rewards.append(sum(rewards))

      if min(total_rewards) == max(total_rewards):
        # converged ??
        sys.stderr.write("converged? standardisation of advantaged will barf here....\n")
        loss = 0
      else:
        loss = self.train(batch_observations, batch_actions, batch_advantages)
        losses.append(loss)

      # dump some stats and progress info
      stats = collections.OrderedDict()
      stats["time"] = time.time()
      stats["n"] = n
      stats["mean_losses"] = float(np.mean(losses))
      stats["total_reward"] = np.sum(total_rewards)
      stats["episode_len"] = len(rewards)

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
    for _ in xrange(num_episodes):
      _, _, rewards = self.rollout(doing_eval=True)
      print sum(rewards)

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
  with tf.Session(config=config) as sess:
    agent = LikelihoodRatioPolicyGradientAgent(env)

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
                         opts.rollouts_per_batch,
                         saver_util)
      if saver_util is not None:
        saver_util.force_save()

    env.reset()  # just to flush logging, clumsy :/

if __name__ == "__main__":
  main()
