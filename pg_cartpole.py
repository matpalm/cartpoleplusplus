#!/usr/bin/env python
import argparse
import bullet_cartpole
import datetime
import gym
import json
import numpy as np
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import util

np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)

parser = argparse.ArgumentParser()
parser.add_argument('--num-hidden', type=int, default=32)
parser.add_argument('--num-eval', type=int, default=0,
                    help="if >0 just run this many episodes with no training")
parser.add_argument('--num-train-batches', type=int, default=10,
                    help="number of training batches to run")
parser.add_argument('--rollouts-per-batch', type=int, default=10,
                    help="number of rollouts to run for each training batch")
parser.add_argument('--run-id', type=str, default=None,
                    help='if set use --ckpt-dir=ckpts/<run-id> and write stats to <run-id>.stats')
parser.add_argument('--ckpt-dir', type=str, default=None,
                    help="if set save ckpts to this dir")
parser.add_argument('--ckpt-freq', type=int, default=60,
                    help="freq (sec) to save ckpts")
# bullet cartpole specific ...
parser.add_argument('--gui', action='store_true',
                    help="whether to call env.render()")
parser.add_argument('--delay', type=float, default=0.0,
                    help="gui per step delay")
parser.add_argument('--max-episode-len', type=int, default=200,
                    help="maximum episode len for cartpole")
parser.add_argument('--initial-force', type=float, default=55.0,
                    help="magnitude of initial push, in random direction")
parser.add_argument('--action-force', type=float, default=50.0,
                    help="magnitude of action push")
parser.add_argument('--calc-explicit-delta', action='store_true',
                    help="if true state is (current,current-last) else state is (current,last)")
opts = parser.parse_args()
sys.stderr.write("%s\n" % opts)

EPSILON = 1e-3

class PolicyGradientAgent(object):
  def __init__(self, env, hidden_dim, optimiser, gui=False):
    self.env = env
    self.gui = gui

    # base model mapping from observation to actions through single hidden layer.
    observation_dim = self.env.observation_space.shape[0]
    num_actions = self.env.action_space.n

    # we have three place holders we'll use...
    # observations; used either during rollout to sample some actions, or
    # during training when combined with actions_taken and advantages.
    self.observations = tf.placeholder(shape=[None, observation_dim],
                                       dtype=tf.float32)
    # the actions we took during rollout
    self.actions = tf.placeholder(tf.int32, name='actions')
    # the advantages we got from taken 'action_taken' in 'observation'
    self.advantages = tf.placeholder(tf.float32, name='advantages')

    # our model is a very simple MLP
    hidden = slim.fully_connected(inputs=self.observations,
                                       num_outputs=hidden_dim,
                                       activation_fn=tf.nn.tanh)
    logits = slim.fully_connected(inputs=hidden,
                                       num_outputs=num_actions)

    # for rollouts we need an op that samples actions from this
    # model to give a stochastic action.
    sample_action = tf.multinomial(logits, num_samples=1),
    self.sampled_action_op = tf.reshape(sample_action, shape=[])

    # we are trying to maximise the product of two components...
    # 1) the log_p of "good" actions.
    # 2) the advantage term based on the rewards from actions.

    # first we need the log_p values for each observation for the actions we specifically
    # took by sampling...
    # first we run a softmax over the action logits to get probabilities.
    # ( +epsilon to avoid near zero instabilities )
    softmax = tf.nn.softmax(logits + 1e-20)
    softmax = tf.verify_tensor_all_finite(softmax, msg="softmax")

    # we then use a mask to only select the elements of the softmaxs that correspond
    # to the actions we actually took. we could also do this by complex indexing and a
    # gather but i always think this is more natural. the "cost" of dealing with the
    # mostly zero one hot, as opposed to doing a gather on sparse indexes, isn't a big
    # deal when the number of observations is >> number of actions.
    action_mask = tf.one_hot(indices=self.actions, depth=num_actions)
    action_prob = tf.reduce_sum(softmax * action_mask, reduction_indices=1)
    action_log_prob = tf.log(action_prob)

    # the (element wise) product of these action log_p's with the total reward of the
    # episode represents the quantity we want to maximise. we standardise the advantage
    # values so roughly 1/2 +ve / -ve as a variance control.
    action_mul_advantages = tf.mul(action_log_prob,
                                   util.standardise(self.advantages))
    self.loss = -tf.reduce_sum(action_mul_advantages)  # recall: we are maximising.
    self.train_op = optimiser.minimize(self.loss)

  def sample_action_given(self, observation):
    """ sample one action given observation"""
    return tf.get_default_session().run(self.sampled_action_op,
                                        feed_dict={self.observations: [observation]})

  def rollout(self):
    """ run one episode collecting observations, actions and advantages"""
    observations, actions, rewards = [], [], []
    observation = self.env.reset()
    done = False
    while not done:
      observations.append(observation)
      action = self.sample_action_given(observation)
      observation, reward, done, _ = self.env.step(action)
      actions.append(action)
      rewards.append(reward)
      if self.gui:
        self.env.render()
    return observations, actions, rewards

  def train(self, observations, actions, advantages):
    """ take one training step given observations, actions and subsequent advantages"""
    _, loss = tf.get_default_session().run([self.train_op, self.loss],
                                           feed_dict={self.observations: observations,
                                                      self.actions: actions,
                                                      self.advantages: advantages })
    return float(loss)

  def run_training(self, num_batches, rollouts_per_batch, saver_util):
    stats_f = None
    if opts.run_id is not None:
      stats_f = open("%s.stats" % opts.run_id, "a")

    for batch_id in xrange(num_batches):
      # perform a number of rollouts
      batch_observations, batch_actions, batch_advantages = [], [], []
      total_rewards = []
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

      # dump some stats
      stats = {"time": int(time.time()),
               "batch": batch_id,
               "rewards": total_rewards,
               "loss": loss}
      stream = stats_f if stats_f is not None else sys.stdout
      stream.write("STATS %s\t%s\n" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                       json.dumps(stats)))

      # dump progress to stderr, assuming stats going to file...
      sys.stderr.write("\rbatch %s/%s             " % (batch_id, num_batches))
      sys.stderr.flush()

      # save if required
      if saver_util is not None:
        saver_util.save_if_required()

    # close stats_f if required
    if stats_f is not None:
      stats_f.close()

  def run_eval(self, num_eval):
    for _ in xrange(num_eval):
      _, _, rewards = self.rollout()
      print sum(rewards)


def main():
  env = bullet_cartpole.BulletCartpole(gui=opts.gui, action_force=opts.action_force,
                                       max_episode_len=opts.max_episode_len,
                                       initial_force=opts.initial_force, delay=opts.delay,
                                       calc_explicit_delta=opts.calc_explicit_delta,
                                       discrete_actions=True)

  with tf.Session() as sess:
    agent = PolicyGradientAgent(env=env, gui=opts.gui,
                                hidden_dim=opts.num_hidden,
                                optimiser=tf.train.AdamOptimizer())

    # setup saver util; will load latest ckpt, or init if none...
    saver_util = None
    ckpt_dir = None
    if opts.run_id is not None:
      ckpt_dir = "ckpts/%s" % opts.run_id
    elif opts.ckpt_dir is not None:
      ckpt_dir = opts.ckpt_dir
    if ckpt_dir is not None:
      saver_util = util.SaverUtil(sess, ckpt_dir, opts.ckpt_freq)
    else:
      sess.run(tf.initialize_all_variables())

    # run either eval or training
    if opts.num_eval > 0:
      agent.run_eval(opts.num_eval)
    else:
      agent.run_training(opts.num_train_batches, opts.rollouts_per_batch,
                         saver_util)
      if saver_util is not None:
        saver_util.force_save()

if __name__ == "__main__":
  main()
