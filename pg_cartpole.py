#!/usr/bin/env python
import argparse
import bullet_cartpole
import gym
import json
import numpy as np
import sys
import tensorflow as tf
import tflearn as tfl
import util

np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)

parser = argparse.ArgumentParser()
parser.add_argument('--num-hidden', type=int, default=32)
parser.add_argument('--num-eval', type=int, default=0,
                    help="if >0 just run eval and no training")
parser.add_argument('--num-train-batches', type=int, default=10,
                    help="number of training batches to run")
parser.add_argument('--rollouts-per-batch', type=int, default=10,
                    help="number of rollouts to run for each training batch")
parser.add_argument('--ckpt-dir', type=str, default=None,
                    help="if set save ckpts to this dir")
parser.add_argument('--ckpt-freq', type=int, default=60,
                    help="freq (sec) to save ckpts")
# bullet cartpole specific ...
parser.add_argument('--gui', type=bool, default=False,
                    help="whether to call env.render()")
parser.add_argument('--delay', type=float, default=0.0,
                    help="gui per step delay")
parser.add_argument('--initial-force', type=float, default=55.0,
                    help="magnitude of initial push, in random direction")
parser.add_argument('--action-force', type=float, default=50.0,
                    help="magnitude of action push")
opts = parser.parse_args()
print opts

EPSILON = 1e-3

class PolicyGradientAgent(object):
  def __init__(self, sess, env, hidden_dim, optimiser, gui=False):
    self.sess = sess
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
    self.actions = tf.placeholder(tf.int32)
    # the advantages we got from taken 'action_taken' in 'observation'
    self.advantages = tf.placeholder(tf.float32)

    # our model is a very simple MLP
    hidden = tfl.fully_connected(self.observations,
                                 n_units=hidden_dim,
                                 activation='tanh')
    self.logits = tfl.fully_connected(hidden,
                                      n_units=num_actions)

    # for rollouts we need an op that samples actions from this
    # model to give a stochastic action.
    sample_action = tf.multinomial(self.logits, num_samples=1),
    self.sampled_action_op = tf.reshape(sample_action, shape=[])

    # there are two components of what we are trying to maximise...
    # 1) the log_p of "good" actions
    # 2) the advantage term based on the rewards from actions.

    # first the (|obs|, |action|) log probabilties of actions
    self.log_p = tf.log(tf.nn.softmax(self.logits))

    # the log_p matrix gives us log probabilities for all possible actions in this
    # set of observations but we want to consider only the ones for the _actual_ actions
    # that were taken, these will contribute to the loss. there are two ways to do this;
    # 1) by masking the entire log_p matrix with one hot row vectors and then summing
    # over axis=1 or 2) by calculating the values to pick, per row, using indexing
    # assuming the log_p matrix is reshaped to a vector (row major ordered). in this
    # code we do the second.
    num_rows_in_log_p = tf.shape(self.observations)[0]
    # 0 -> num rows sequence.
    log_p_indices = tf.range(0, num_rows_in_log_p)
    # 0, a, 2a, ...   ie 0th column in log_p matrix (for 'a' actions).
    log_p_indices = log_p_indices * num_actions
    # indices in flattened form of log_p that correspond to actions taken.
    log_p_indices = log_p_indices + self.actions
    # finally uses these to fetch the action log probs taken.
    action_log_p = tf.gather(tf.reshape(self.log_p, [-1]),
                             log_p_indices)

    # the (element wise) product of these action log_p's with the total reward of the
    # episode represents the quantity we want to maximise. we standardise the advantage
    # values so roughly 1/2 +ve / -ve as a variance control.
    action_mul_advantages = tf.mul(action_log_p,
                                   util.standardise(self.advantages))
    self.loss = -tf.reduce_sum(action_mul_advantages)  # recall: we are maximising.
    self.train_op = optimiser.minimize(self.loss)

  def sample_action_given(self, observation):
    """ sample one action given observation"""
    return self.sess.run(self.sampled_action_op,
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
    _, loss = self.sess.run([self.train_op, self.loss],
                            feed_dict={self.observations: observations,
                                       self.actions: actions,
                                       self.advantages: advantages })
    return float(loss)

  def run_training(self, num_batches, rollouts_per_batch, saver_util):
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

      # train
      loss = self.train(batch_observations, batch_actions, batch_advantages)

      # dump some stats
      stats = {"batch": batch_id,
               "rewards": total_rewards,
               "mean_reward": np.mean(total_rewards),
               "loss": loss}
      print "STATS\t%s" % json.dumps(stats)

      # save if required
      if saver_util is not None:
        saver_util.save_if_required()

  def run_eval(self, num_eval):
    for _ in xrange(num_eval):
      _, _, rewards = self.rollout()
      total_rewards = sum(rewards)
      print total_rewards

    
def main():
#  env = gym.make('CartPole-v0')
  env = bullet_cartpole.BulletCartpole(gui=opts.gui, action_force=opts.action_force,
                                       initial_force=opts.initial_force, delay=opts.delay)

  with tf.Session() as sess:
    agent = PolicyGradientAgent(sess=sess, env=env, gui=opts.gui,
                                hidden_dim=opts.num_hidden,
                                optimiser=tf.train.AdamOptimizer())

    # setup saver util; will load latest ckpt, or init if none...
    saver_util = None
    if opts.ckpt_dir is not None:
      saver_util = util.SaverUtil(sess, opts.ckpt_dir, opts.ckpt_freq)
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
