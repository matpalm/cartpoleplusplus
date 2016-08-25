#!/usr/bin/env python
import argparse
import bullet_cartpole
import datetime
import gym
import json
import numpy as np
import replay_memory
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import util

np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('--num-hidden', type=int, default=32)  TODO USE!
parser.add_argument('--num-eval', type=int, default=0,
                    help="if >0 just run this many episodes with no training")
parser.add_argument('--max-num-actions', type=int, default=1000,
                    help="when training run complete episodes until the total number of"
                         " actions taken exceeds this number")
parser.add_argument('--run-id', type=str, default=None,
                    help="if set use --ckpt-dir=ckpts/<run-id> and write stats to <run-id>.stats")
parser.add_argument('--ckpt-dir', type=str, default=None, help="if set save ckpts to this dir")
parser.add_argument('--ckpt-freq', type=int, default=60, help="freq (sec) to save ckpts")
parser.add_argument('--batch-size', type=int, default=16, help="training batch size")
parser.add_argument('--training-action-freq', type=int, default=1,
                    help="run a training batch every --training-action-freq actions")
parser.add_argument('--target-update-rate', type=float, default=0.001,
                    help="affine combo for updating target networks each time we run a training batch")
parser.add_argument('--replay-memory-size', type=int, default=100000, help="max size of replay memory")
parser.add_argument('--action-noise-theta', type=float, default=0.01,
                    help="OrnsteinUhlenbeckNoise theta (rate of change) param for action exploration")
parser.add_argument('--action-noise-sigma', type=float, default=0.2,
                    help="OrnsteinUhlenbeckNoise sigma (magnitude) param for action exploration")
# bullet cartpole specific ...
parser.add_argument('--gui', action='store_true', help="whether to call env.render()")
parser.add_argument('--delay', type=float, default=0.0, help="gui per step delay")
parser.add_argument('--max-episode-len', type=int, default=200, help="maximum episode len for cartpole")
parser.add_argument('--initial-force', type=float, default=55.0,
                    help="magnitude of initial push, in random direction")
parser.add_argument('--action-force', type=float, default=50.0,
                    help="magnitude of action push")
parser.add_argument('--calc-explicit-delta', action='store_true',
                    help="if true state is (current,current-last) else state is (current,last)")
opts = parser.parse_args()
sys.stderr.write("%s\n" % opts)

EPSILON = 1e-3


class OrnsteinUhlenbeckNoise(object):
  """Time correlated noise for action exploration."""

  def __init__(self, action_dim, theta=0.01, sigma=0.2, max_magnitude=1.5):
    # theta: how quickly the value moves; near zero => slow, near one => fast
    #        0.01 gives very roughly 2/3 peaks troughs over ~1000 samples
    # sigma: maximum range of values; 0.2 gives approximately the range (-1.5, 1.5)
    #        which is useful for shifting the output of a tanh which is (-1, 1)
    # max_magnitude: max +ve / -ve value to clip at. dft clip at 1.5 (again for 
    #                adding to output from tanh
    self.action_dim = action_dim
    self.theta = theta
    self.sigma = sigma
    self.max_magnitude = max_magnitude
    self.state = np.zeros(self.action_dim)

  def sample(self):
    self.state += self.theta * -self.state
    self.state += self.sigma * np.random.randn(self.action_dim)
    self.state = np.clip(self.max_magnitude, -self.max_magnitude, self.state)
    return np.copy(self.state)


class Network(object):
  """Common base class for actor & critic handling operations relating to making / updating target networks."""

  def __init__(self, namespace):
    self.namespace = namespace
    self.target_update_op = None

  def _create_variables_copy_op(self, source_namespace, affine_combo_coeff):
    """create a single op that does an affine combo of all vars in source_namespace to target_namespace"""
    assert affine_combo_coeff >= 0.0 and affine_combo_coeff <= 1.0
    assign_ops = []
    with tf.variable_scope("", reuse=True):  # for grabbing the targets by full namespace
      for src_var in tf.all_variables():
        if not src_var.name.startswith(source_namespace):
          continue
        target_var_name = src_var.name.replace(source_namespace, self.namespace).replace(":0", "")
        target_var = tf.get_variable(target_var_name)
        assert src_var.get_shape() == target_var.get_shape()
        assign_ops.append(target_var.assign_sub(affine_combo_coeff * (target_var - src_var)))
    single_assign_op = tf.group(*assign_ops)
    return single_assign_op

  def set_as_target_network_for(self, source_network, target_update_rate):
    """Create an op that will update this networks weights based on a source_network"""
    # first, as a one off, copy _all_ variables across.
    # i.e. initial target network will be a copy of source network.
    op = self._create_variables_copy_op(source_network.namespace, affine_combo_coeff=1.0)
    tf.get_default_session().run(op)
    # next build target update op for running later during training
    self.update_weights_op = self._create_variables_copy_op(source_network.namespace,
                                                           target_update_rate)

  def update_weights(self):
    """called during training to update target network. requires prior call to set_source_network"""
    if self.update_weights_op is None:
      raise Exception("not a target network? or set_source_network not yet called")
    return tf.get_default_session().run(self.update_weights_op)

  def trainable_model_vars(self):
    v = []
    for var in tf.all_variables():
      if var.name.startswith(self.namespace):
        v.append(var)
    return v

  def dump_vars(self):
    for v in self.trainable_model_vars():
      print v.name, v.eval()


class ActorNetwork(Network):
  def __init__(self, namespace, state_dim, action_dim, explore_theta=0.0, explore_sigma=0.0):
    super(ActorNetwork, self).__init__(namespace)
    self.state_dim = state_dim
    self.action_dim = action_dim

    self.input_state = tf.placeholder(shape=[None, state_dim],
                                      dtype=tf.float32, name="actor_input_state")

    self.exploration_noise = OrnsteinUhlenbeckNoise(action_dim, explore_theta, explore_sigma)

    with tf.variable_scope(namespace):
      hidden = slim.fully_connected(inputs=self.input_state,
                                    num_outputs=32,  # TODO config
                                    activation_fn=tf.nn.tanh)
      hidden = slim.fully_connected(inputs=hidden,
                                    num_outputs=32,  # TODO config
                                    activation_fn=tf.nn.tanh)
      self.output_action = slim.fully_connected(inputs=hidden,
                                                num_outputs=action_dim,
                                                activation_fn=tf.nn.tanh)
      # for exploration during rollout we additionally provide an op
      # that provides output with noise. not used during training.
      # TODO: ACTUALLY add noise !!!
      self.output_action_with_noise = self.output_action

  def init_ops_for_training(self, critic, learning_rate):
    # actors gradients are the gradients for it's output w.r.t actor vars using initial
    # gradients provided by critic). this requires that critic was init'd with an
    # input_action = self.output_action. wrap in namespace since we don't want this
    # as part of copy to target networks. we negate the gradients from critic since we
    # are trying to maximise the q values (not minimise like a loss)
    with tf.variable_scope("optimiser"):
      self.actor_gradients = tf.gradients(self.output_action,
                                          self.trainable_model_vars(),
                                          tf.neg(critic.q_gradients_wrt_actions()))
      optimiser = tf.train.AdamOptimizer(learning_rate)
      self.train_op = optimiser.apply_gradients(zip(self.actor_gradients,
                                                    self.trainable_model_vars()))

  def action_given(self, state, add_noise):
    # NOTE: noise added _outside_ tf graph. we do this simply because the noisy output
    # is never used for any part of computation graph. it's only used during training
    # when feed through a placeholder; i.e. we never backprop through a noisy action.
    actions = tf.get_default_session().run(self.output_action,
                                           feed_dict={self.input_state: state})
    print "actions (pre noise)", actions
    if add_noise:
      assert actions.shape[0] == 1  # Clumsy; we expect to only add noise during online rollout
      actions[0] += self.exploration_noise.sample()
      actions = np.clip(1, -1, actions)  # action output is _always_ (-1, 1)
      print "actions (post noise)", actions
    return actions    

  def train(self, state):
    # training actor only requires state since we are trying to maximise the
    # q_value according to the critic.
    return tf.get_default_session().run(self.train_op,
                                        feed_dict={self.input_state: state})


class CriticNetwork(Network):
  def __init__(self, namespace, actor, discount=0.9):
    super(CriticNetwork, self).__init__(namespace)
    self.discount = discount  # bellman update discount

    self.input_state = actor.input_state
    self.input_action = actor.output_action

    with tf.variable_scope(namespace):
      concat_inputs = tf.concat(1, [self.input_state, self.input_action])
      hidden = slim.fully_connected(inputs=concat_inputs,
                                    num_outputs=32,  # TODO config
                                    activation_fn=tf.nn.tanh)
      hidden = slim.fully_connected(inputs=hidden,
                                    num_outputs=32,  # TODO config
                                    activation_fn=tf.nn.tanh)
      self.q_value = slim.fully_connected(inputs=hidden,
                                          num_outputs=1,
                                          activation_fn=None)

  def init_ops_for_training(self, target_critic, learning_rate):
    # update critic using bellman equation; Q(s1, a) = reward + discount * Q(s2, A(s2))
    # left hand side of bellman is just q_value
    #  requires { critic.input_state: state_1, critic.input_action: action }
    # right hand side is reward + discounted q value from target actor & critic
    #  requires { reward: reward, target_critic.input_state: state_2 }
    self.reward = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="critic_reward")
    self.input_state_2 = target_critic.input_state
    bellman_rhs = self.reward + (self.discount * target_critic.q_value)
    # since we are NOT training target networks we stop gradients flowing to them
    bellman_rhs = tf.stop_gradient(bellman_rhs)
    # what we are trying to mimimise is the difference between these two. used a
    # squared loss for optimisation and, as for actor, wrap optimiser in a namespace
    # so it's not picked up by target network variable handling.
    temporal_difference = self.q_value - bellman_rhs
    self.temporal_difference_loss = tf.reduce_mean(tf.pow(temporal_difference, 2))
    with tf.variable_scope("optimiser"):
      self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.temporal_difference_loss)

  def q_gradients_wrt_actions(self):
    return tf.gradients(self.q_value, self.input_action)[0]

  def debug_q_value_for(self, state, action=None):
    feed_dict = {self.input_state: state}
    if action is not None:
      feed_dict[self.input_action] = action
    return tf.get_default_session().run(self.q_value, feed_dict=feed_dict)

  def train(self, state_1, action, reward, state_2):
    l, t = tf.get_default_session().run([self.temporal_difference_loss, self.train_op],
                                        feed_dict={self.input_state: state_1,
                                                   self.input_action: action,
                                                   self.reward: reward,
                                                   self.input_state_2: state_2})
    print "Q LOSS", l
    return t


class DeepDeterministicPolicyGradientAgent(object):
  def __init__(self, env, training_action_freq, replay_memory_size):
    self.env = env
    self.training_action_freq = training_action_freq
    state_dim = self.env.observation_space.shape[0]
    action_dim = self.env.action_space.shape[1]
    print "state_dim", state_dim, "action_dim", action_dim
    self.replay_memory = replay_memory.ReplayMemory(replay_memory_size, state_dim, action_dim)

    # initialise base models for actor / critic and their corresponding target networks
    # target_actor is never used for online sampling so doesn't need explore noise.
    self.actor = ActorNetwork("actor", state_dim, action_dim,
                              explore_theta=opts.action_noise_theta,
                              explore_sigma=opts.action_noise_sigma)
    self.critic = CriticNetwork("critic", self.actor)
    self.target_actor = ActorNetwork("target_actor", state_dim, action_dim)
    self.target_critic = CriticNetwork("target_critic", self.target_actor)

    # setup training ops;
    # training actor requires the critic (for gradients)
    # training critic requires target_critic (for RHS of bellman update)
    self.actor.init_ops_for_training(self.critic, learning_rate=0.01)
    self.critic.init_ops_for_training(self.target_critic, learning_rate=0.01)

  def hook_up_target_networks(self, target_update_rate):
    # hook networks up to their targets
    # ( does one off clobber to init all vars in target network )
    self.target_actor.set_as_target_network_for(self.actor, target_update_rate)
    self.target_critic.set_as_target_network_for(self.critic, target_update_rate)

  def run_training(self, num_episodes, batch_size, saver_util):
    # setup a stream to write stats to
    stats_stream = sys.stdout
    if opts.run_id is not None:
      stats_stream = open("%s.stats" % opts.run_id, "a")

    # run for some max number of actions
    num_actions_taken = 0
    episode_num = 0
    while True:
      state_1 = self.env.reset()
      rewards_for_episode = []
      done = False
      while not done:
        # take a step in env
        action = self.actor.action_given([state_1], add_noise=True)
        print "EXPECTED Q", self.critic.debug_q_value_for([state_1])[0][0], "given state", state_1, "action", action
        state_2, reward, done, _ = self.env.step(action)
        print "STEP! s1", state_1, "action", action, "reward", reward, "state2", state_2
        # keep some info
        rewards_for_episode.append(reward)
        self.replay_memory.add(state_1, action, reward, state_2)
        # do a training step sometimes
        if self.replay_memory.size() > batch_size and num_actions_taken % self.training_action_freq == 0:
          state_1_b, reward_b, action_b, state_2_b = self.replay_memory.batch(batch_size)
#          print "TRAIN!", self.replay_memory.size(), state_1_b, reward_b, action_b, state_2_b
          self.actor.train(state_1_b)
          self.critic.train(state_1_b, reward_b, action_b, state_2_b)
          self.target_actor.update_weights()
          self.target_critic.update_weights()
        # roll state for next step.
        state_1 = state_2
        num_actions_taken += 1

      # dump some stats and progress info
      stats = {"time": int(time.time()), "episode": episode_num, 
               "rewards": rewards_for_episode, "episode_len": len(rewards_for_episode)}
      stats_stream.write("STATS %s\t%s\n" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        json.dumps(stats)))
      if stats_stream is not sys.stdout:
        sys.stderr.write("\rbatch %s/%s             " % (batch_id, num_batches))
        sys.stderr.flush()

      # save if required
      if saver_util is not None:
        saver_util.save_if_required()

      # exit when finished
      if num_actions_taken > opts.max_num_actions:
        break
      episode_num += 1

    # close stats_f if required
    if stats_stream is not sys.stdout:
      stats_stream.close()

  def run_eval(self, num_episodes):
    for _ in xrange(num_episodes):
      state = self.env.reset()
      total_reward = 0
      done = False
      while not done:
        action = self.actor.action_given(state, add_noise=False)
        state, reward, done, _ = self.env.step(action)
        total_reward + reward
      print total_rewards


def main():
  env = bullet_cartpole.BulletCartpole(gui=opts.gui, action_force=opts.action_force,
                                       max_episode_len=opts.max_episode_len,
                                       initial_force=opts.initial_force, delay=opts.delay,
                                       calc_explicit_delta=opts.calc_explicit_delta,
                                       discrete_actions=False)
  with tf.Session() as sess:
    agent = DeepDeterministicPolicyGradientAgent(env=env,
                                                 training_action_freq=opts.training_action_freq,
                                                 replay_memory_size=opts.replay_memory_size)

    # setup saver util and either load latest ckpt, or init if none...
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

    # now that we've either init'd from scratch, or loaded up a checkpoint,
    # we can hook together target networks
    agent.hook_up_target_networks(opts.target_update_rate)

    # run either eval or training
    if opts.num_eval > 0:
      agent.run_eval(opts.num_eval)
    else:
      agent.run_training(opts.max_num_actions, opts.batch_size, saver_util)
      if saver_util is not None:
        saver_util.force_save()

if __name__ == "__main__":
  main()
