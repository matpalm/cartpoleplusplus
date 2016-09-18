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
                    help="train for (at least) this number of actions (always finish current episode)"
                         " ignore if <=0")
parser.add_argument('--max-run-time', type=int, default=0,
                    help="train for (at least) this number of seconds (always finish current episode)"
                         " ignore if <=0")
parser.add_argument('--ckpt-dir', type=str, default=None, help="if set save ckpts to this dir")
parser.add_argument('--ckpt-freq', type=int, default=300, help="freq (sec) to save ckpts")
parser.add_argument('--batch-size', type=int, default=128, help="training batch size")
parser.add_argument('--batches-per-step', type=int, default=5,
                    help="number of batches to train per step")
parser.add_argument('--target-update-rate', type=float, default=0.0001,
                    help="affine combo for updating target networks each time we run a training batch")
parser.add_argument('--actor-hidden-layers', type=str, default="100,100,50", help="actor hidden layer sizes")
parser.add_argument('--critic-hidden-layers', type=str, default="100,100,50", help="actor hidden layer sizes")
parser.add_argument('--actor-learning-rate', type=float, default=0.001, help="learning rate for actor")
parser.add_argument('--critic-learning-rate', type=float, default=0.01, help="learning rate for critic")
parser.add_argument('--actor-gradient-clip', type=float, default=None, help="clip actor gradients at this l2 norm")
parser.add_argument('--critic-gradient-clip', type=float, default=None, help="clip critic gradients at this l2 norm")
parser.add_argument('--critic-bellman-discount', type=float, default=0.99, help="discount for RHS of critic bellman equation update")
parser.add_argument('--replay-memory-size', type=int, default=22000, help="max size of replay memory")
parser.add_argument('--replay-memory-burn-in', type=int, default=1000, help="dont train from replay memory until it reaches this size")
parser.add_argument('--eval-action-noise', action='store_true', help="whether to use noise during eval")
parser.add_argument('--action-noise-theta', type=float, default=0.01,
                    help="OrnsteinUhlenbeckNoise theta (rate of change) param for action exploration")
parser.add_argument('--action-noise-sigma', type=float, default=0.2,
                    help="OrnsteinUhlenbeckNoise sigma (magnitude) param for action exploration")
bullet_cartpole.add_opts(parser)
opts = parser.parse_args()
sys.stderr.write("%s\n" % opts)

# TODO: if we import slim _before_ building cartpole env we can't start bullet with GL gui o_O
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


class ActorNetwork(base_network.Network):
  """ the actor represents the learnt policy mapping states to actions"""

  def __init__(self, namespace, input_state, input_state_idx, action_dim,
               hidden_layer_config,
               explore_theta=0.0, explore_sigma=0.0,
               use_raw_pixels=False):
    super(ActorNetwork, self).__init__(namespace)

    # since state is keep in a tf variable we keep track of the variable itself
    # as well as an indexing placeholder
    self.input_state = input_state
    self.input_state_idx = input_state_idx

    self.exploration_noise = util.OrnsteinUhlenbeckNoise(action_dim, explore_theta, explore_sigma)

    with tf.variable_scope(namespace):
      if use_raw_pixels:
        # simple conv net with one hidden layer on top
        conv_net = self.simple_conv_net_on(self.input_state)
        hidden1 = slim.fully_connected(conv_net, 400, scope='hidden1')
        final_hidden = slim.fully_connected(hidden1, 50, scope='hidden2')
      else:
        # stack of hidden layers on flattened input; (batch,2,2,7) -> (batch,28)
        flat_input_state = slim.flatten(input_state, scope='flat')
        final_hidden = self.hidden_layers_starting_at(flat_input_state,
                                                      hidden_layer_config)

      # TODO: add dropout for both nets!

      self.output_action = slim.fully_connected(scope='output_action',
                                                inputs=final_hidden,
                                                num_outputs=action_dim,
                                                weights_regularizer=self.l2(),
                                                activation_fn=tf.nn.tanh)

  def init_ops_for_training(self, critic, learning_rate, gradient_clip_norm):
    # actors gradients are the gradients for it's output w.r.t it's vars using initial
    # gradients provided by critic. this requires that critic was init'd with an
    # input_action = actor.output_action (which is natural anyway)
    # we wrap the optimiser in namespace since we don't want this as part of copy to
    # target networks.
    # note that we negate the gradients from critic since we are trying to maximise
    # the q values (not minimise like a loss)
    with tf.variable_scope("optimiser"):
      actor_gradients = tf.gradients(self.output_action,
                                     self.trainable_model_vars(),
                                     tf.neg(critic.q_gradients_wrt_actions()))
      for i, gradient in enumerate(actor_gradients):
        if gradient_clip_norm is not None:
          gradient = tf.clip_by_norm(gradient, gradient_clip_norm)
#        gradient = tf.Print(gradient, [util.l2_norm(gradient)], "actor gradient %d l2_norm pre " % i)
        actor_gradients[i] = gradient
#      optimiser = tf.train.AdamOptimizer(learning_rate)
      optimiser = tf.train.GradientDescentOptimizer(learning_rate)
      self.train_op = optimiser.apply_gradients(zip(actor_gradients,
                                                    self.trainable_model_vars()))

  def action_given(self, state_idx=None, explicit_state=None, add_noise=False):
    # TODO: drop state_idx, never use it...
    assert (state_idx is None) ^ (explicit_state is None)
    if state_idx is not None:
      # feed state_idx; state has to have been cached previously during training rollout.
      actions = tf.get_default_session().run(self.output_action,
                                             feed_dict={self.input_state_idx: [state_idx]})
    else:
      # feed explicitly provided state
      actions = tf.get_default_session().run(self.output_action,
                                             feed_dict={self.input_state: [explicit_state]})

    # NOTE: noise is added _outside_ tf graph. we do this simply because the noisy output
    # is never used for any part of computation graph required for online training. it's
    # only used during training after being the replay buffer.
    if add_noise:
      actions[0] += self.exploration_noise.sample()
      actions = np.clip(1, -1, actions)  # action output is _always_ (-1, 1)

    return actions

  def train(self, state_idx):
    # training actor only requires state since we are trying to maximise the
    # q_value according to the critic.
    tf.get_default_session().run(self.train_op,
                                 feed_dict={self.input_state_idx: state_idx})


class CriticNetwork(base_network.Network):
  """ the critic represents a mapping from state & actors action to a quality score."""

  def __init__(self, namespace, actor, hidden_layer_config,
               discount, use_raw_pixels=False):
    super(CriticNetwork, self).__init__(namespace)
    self.discount = discount  # bellman update discount  TODO: config!

    # input state to the critic is the _same_ state given to the actor.
    # input action to the critic is simply the output action of the actor.
    # even though when training we explicitly provide a new value for the
    # input action (via the input_action placeholder) we need to be stop the gradient
    # flowing to the actor since there is a path through the actor to the input_state
    # too, hence we need to be explicit about cutting it (otherwise training the
    # critic will attempt to train the actor too.
    self.input_state = actor.input_state
    self.input_state_idx = actor.input_state_idx
    self.input_action = tf.stop_gradient(actor.output_action)

    with tf.variable_scope(namespace):
      if use_raw_pixels:
        conv_net = self.simple_conv_net_on(self.input_state)
        hidden1 = slim.fully_connected(conv_net, 200, scope='hidden1')
        hidden2 = slim.fully_connected(hidden1, 50, scope='hidden2')
        concat_inputs = tf.concat(1, [hidden2, self.input_action])
        final_hidden = slim.fully_connected(concat_inputs, 50, scope="hidden3")
      else:
        # stack of hidden layers on flattened input; (batch,2,2,7) -> (batch,28)
        flat_input_state = slim.flatten(self.input_state, scope='flat')
        concat_inputs = tf.concat(1, [flat_input_state, self.input_action])
        final_hidden = self.hidden_layers_starting_at(concat_inputs,
                                                      hidden_layer_config)

      # output from critic is a single q-value
      self.q_value = slim.fully_connected(scope='q_value',
                                          inputs=final_hidden,
                                          num_outputs=1,
                                          weights_regularizer=self.l2(),
                                          activation_fn=None)

  def init_ops_for_training(self, target_critic, learning_rate, gradient_clip_norm):
    # update critic using bellman equation; Q(s1, a) = reward + discount * Q(s2, A(s2))

    # left hand side of bellman is just q_value, but let's be explicit about it...
    bellman_lhs = self.q_value

    # right hand side is ...
    #  = reward + discounted q value from target actor & critic in the non terminal case
    #  = reward  # in the terminal case
    self.reward = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="critic_reward")
    self.terminal_mask = tf.placeholder(shape=[None, 1], dtype=tf.float32,
                                        name="critic_terminal_mask")
    self.input_state_2 = target_critic.input_state
    self.input_state_2_idx = target_critic.input_state_idx
    bellman_rhs = self.reward + (self.terminal_mask * self.discount * target_critic.q_value)

    # note: since we are NOT training target networks we stop gradients flowing to them
    bellman_rhs = tf.stop_gradient(bellman_rhs)

    # the value we are trying to mimimise is the difference between these two; the
    # temporal difference we use a squared loss for optimisation and, as for actor, we
    # wrap optimiser in a namespace so it's not picked up by target network variable
    # handling.
    temporal_difference = bellman_lhs - bellman_rhs
    self.temporal_difference_loss = tf.reduce_mean(tf.pow(temporal_difference, 2))
#    self.temporal_difference_loss = tf.Print(self.temporal_difference_loss, [self.temporal_difference_loss], 'temporal_difference_loss')
    with tf.variable_scope("optimiser"):
      #optimizer = tf.train.AdamOptimizer(learning_rate)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      gradients = optimizer.compute_gradients(self.temporal_difference_loss)
      for i, (gradient, variable) in enumerate(gradients):
        if gradient is None:  # these are the stop gradient cases; ignore them
          continue
        if gradient_clip_norm is not None:
          gradient = tf.clip_by_norm(gradient, gradient_clip_norm)
#        gradient = tf.Print(gradient, [util.l2_norm(gradient)], "critic gradient %d l2_norm " % i)
        gradients[i] = (gradient, variable)
      self.train_op = optimizer.apply_gradients(gradients)

  def q_gradients_wrt_actions(self):
    """ gradients for the q.value w.r.t just input_action; used for actor training"""
    return tf.gradients(self.q_value, self.input_action)[0]

  def debug_q_value_for(self, input_state, action=None):
    feed_dict = {self.input_state: input_state}
    if action is not None:
      feed_dict[self.input_action] = action
    return np.squeeze(tf.get_default_session().run(self.q_value, feed_dict=feed_dict))

  def train(self, batch):
    tf.get_default_session().run(self.train_op,
                                 feed_dict={self.input_state_idx: batch.state_1_idx,
                                            self.input_action: batch.action,
                                            self.reward: batch.reward,
                                            self.terminal_mask: batch.terminal_mask,
                                            self.input_state_2_idx: batch.state_2_idx})

  def check_loss(self, state_1_idx, action, reward, terminal_mask, state_2_idx):
    return tf.get_default_session().run(self.temporal_difference_loss,
                                        feed_dict={self.input_state_idx: state_1_idx,
                                                   self.input_action: action,
                                                   self.reward: reward,
                                                   self.terminal_mask: terminal_mask,
                                                   self.input_state_2_idx: state_2_idx})


class DeepDeterministicPolicyGradientAgent(object):
  def __init__(self, env, agent_opts):
    self.env = env
    state_shape = self.env.observation_space.shape
    action_dim = self.env.action_space.shape[1]

    # for now, with single machine synchronous training, use a replay memory for training.
    # this replay memory stores states in a Variable (ie potentially in gpu memory)
    # TODO: switch back to async training with multiple replicas (as in drivebot project)
    self.replay_memory = replay_memory.ReplayMemory(tf.get_default_session(),
                                                    agent_opts.replay_memory_size,
                                                    state_shape, action_dim)

    # in the --use-raw-pixels case states are very large so we want them stored in tf variable
    # (i.e. on the gpu). we do batch training by only passing indexs to this memory when
    # feeding batchs. specifically 's1' is an op that emits state when 's1_idx' placeholder is fed
    s1, s1_idx, s2, s2_idx = self.replay_memory.batch_ops()

    # initialise base models for actor / critic and their corresponding target networks
    # target_actor is never used for online sampling so doesn't need explore noise.
    self.actor = ActorNetwork("actor", s1, s1_idx, action_dim,
                              agent_opts.actor_hidden_layers,
                              agent_opts.action_noise_theta,
                              agent_opts.action_noise_sigma,
                              agent_opts.use_raw_pixels)
    self.critic = CriticNetwork("critic", self.actor,
                                agent_opts.critic_hidden_layers,
                                agent_opts.critic_bellman_discount,
                                use_raw_pixels=agent_opts.use_raw_pixels)
    self.target_actor = ActorNetwork("target_actor", s2, s2_idx, action_dim,
                                     agent_opts.actor_hidden_layers,
                                     use_raw_pixels=agent_opts.use_raw_pixels)
    self.target_critic = CriticNetwork("target_critic", self.target_actor,
                                       agent_opts.critic_hidden_layers,
                                       agent_opts.critic_bellman_discount,
                                       use_raw_pixels=agent_opts.use_raw_pixels)

    # setup training ops;
    # training actor requires the critic (for getting gradients)
    # training critic requires target_critic (for RHS of bellman update)
    self.actor.init_ops_for_training(self.critic,
                                     agent_opts.actor_learning_rate,
                                     agent_opts.actor_gradient_clip)
    self.critic.init_ops_for_training(self.target_critic,
                                      agent_opts.critic_learning_rate,
                                      agent_opts.critic_gradient_clip)

  def hook_up_target_networks(self, target_update_rate):
    # hook networks up to their targets
    # ( does one off clobber to init all vars in target network )
    self.target_actor.set_as_target_network_for(self.actor, target_update_rate)
    self.target_critic.set_as_target_network_for(self.critic, target_update_rate)


  def run_training(self, max_num_actions, max_run_time, batch_size, batches_per_step,
                   saver_util):
    # log start time, in case we are limiting by time...
    start_time = time.time()

    # run for some max number of actions
    num_actions_taken = 0
    n = 0
    while True:      
      rewards = []
      # start a new episode
      state_1 = self.env.reset()
      # prepare data for updating replay memory at end of episode
      initial_state = np.copy(state_1)
      action_reward_state_sequence = []

      done = False
      while not done:
        # choose action
        action = self.actor.action_given(explicit_state=state_1, add_noise=True)
        # take action step in env
        state_2, reward, done, _ = self.env.step(action)
        rewards.append(reward)
        # cache for adding to replay memory
        action_reward_state_sequence.append((action, reward, np.copy(state_2)))
        # do a training step (after waiting for buffer to fill a bit...)
        if self.replay_memory.size() > opts.replay_memory_burn_in:
          # run a set of batches
          for _ in xrange(batches_per_step):
            batch = self.replay_memory.batch(batch_size)
            self.actor.train(batch.state_1_idx)
            self.critic.train(batch)
          # update target nets
          self.target_actor.update_weights()
          self.target_critic.update_weights()
          # do debug (if requested) on last batch
          if VERBOSE_DEBUG:
            print "-----"
            #print "state_1", state_1
            print "action\n", batch.action.T
            print "reward        ", batch.reward.T
            print "terminal_mask ", batch.terminal_mask.T
            #print "state_2", state_2
            expected_q = self.critic.debug_q_value_for(batch.state_1_idx)
            expected_target_q = self.target_critic.debug_q_value_for(batch.state_1_idx)
            print "EXPECTED_Q_VALUES", expected_q, expected_target_q
        # roll state for next step.
        state_1 = state_2

      # at end of episode update replay memory
      self.replay_memory.add_episode(initial_state, action_reward_state_sequence)

      # dump some stats and progress info
      stats = collections.OrderedDict()
      stats["time"] = time.time()
      stats["n"] = n
      stats["total_reward"] = np.sum(rewards)
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
        action = self.actor.action_given(explicit_state=state,
                                         add_noise=add_noise)
        state, reward, done, _ = self.env.step(action)
        print "EVALSTEP r%s %s %s %s %s" % (i, steps, np.squeeze(action), np.linalg.norm(action), reward)
        total_reward += reward
        steps += 1
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
  with tf.Session(config=config) as sess:
    agent = DeepDeterministicPolicyGradientAgent(env=env, agent_opts=opts)

    # setup saver util and either load latest ckpt, or init if none...
    saver_util = None
    if opts.ckpt_dir is not None:
      saver_util = util.SaverUtil(sess, opts.ckpt_dir, opts.ckpt_freq)
    else:
      sess.run(tf.initialize_all_variables())

    # now that we've either init'd from scratch, or loaded up a checkpoint,
    # we can hook together target networks
    agent.hook_up_target_networks(opts.target_update_rate)

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
