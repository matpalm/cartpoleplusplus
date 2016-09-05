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
parser.add_argument('--target-update-rate', type=float, default=0.001,
                    help="affine combo for updating target networks each time we run a training batch")
parser.add_argument('--actor-hidden-layers', type=str, default="100,100,50", help="actor hidden layer sizes")
parser.add_argument('--critic-hidden-layers', type=str, default="100,100,50", help="actor hidden layer sizes")
parser.add_argument('--actor-learning-rate', type=float, default=0.001, help="learning rate for actor")
parser.add_argument('--critic-learning-rate', type=float, default=0.01, help="learning rate for critic")
parser.add_argument('--actor-gradient-clip', type=float, default=None, help="clip actor gradients at this l2 norm")
parser.add_argument('--critic-gradient-clip', type=float, default=None, help="clip critic gradients at this l2 norm")
parser.add_argument('--actor-activation-init-magnitude', type=float, default=0.001,
                    help="weight magnitude for actor final activation. explicitly near zero to force near zero predictions initially")
parser.add_argument('--replay-memory-size', type=int, default=500000, help="max size of replay memory")
parser.add_argument('--eval-action-noise', action='store_true', help="whether to use noise during eval")
parser.add_argument('--action-noise-theta', type=float, default=0.01,
                    help="OrnsteinUhlenbeckNoise theta (rate of change) param for action exploration")
parser.add_argument('--action-noise-sigma', type=float, default=0.2,
                    help="OrnsteinUhlenbeckNoise sigma (magnitude) param for action exploration")
bullet_cartpole.add_opts(parser)
opts = parser.parse_args()
sys.stderr.write("%s\n" % opts)

class Network(object):
  """Common class for actor/critic handling ops for making / updating target networks."""

  def __init__(self, namespace):
    self.namespace = namespace
    self.target_update_op = None

  def _create_variables_copy_op(self, source_namespace, affine_combo_coeff):
    """create an op that does updates all vars in source_namespace to target_namespace"""
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

  def hidden_layers_starting_at(self, layer, config):
    layer_sizes = map(int, config.split(","))
    assert len(layer_sizes) > 0
    for i, size in enumerate(layer_sizes):
      layer = slim.fully_connected(scope="h%d" % i,
                                  inputs=layer,
                                  num_outputs=size,
                                  weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                  activation_fn=tf.nn.relu)
    return layer

class ActorNetwork(Network):
  """ the actor represents the learnt policy mapping states to actions"""

  def __init__(self, namespace, state_shape, action_dim,
               hidden_layer_config,
               explore_theta=0.0, explore_sigma=0.0,
               actor_activation_init_magnitude=1e-3):
    super(ActorNetwork, self).__init__(namespace)
    self.action_dim = action_dim

    batched_state_shape = [None] + list(state_shape)
    self.input_state = tf.placeholder(shape=batched_state_shape,
                                      dtype=tf.float32, name="actor_input_state")

    self.exploration_noise = util.OrnsteinUhlenbeckNoise(action_dim, explore_theta, explore_sigma)

    with tf.variable_scope(namespace):
      # stack of hidden layers
      flat_input = slim.flatten(self.input_state)
      final_hidden = self.hidden_layers_starting_at(flat_input, config=hidden_layer_config)
      # action dim output. note: actors out is (-1, 1) and scaled in environment as required.
      weights_initializer = tf.random_uniform_initializer(-actor_activation_init_magnitude,
                                                          actor_activation_init_magnitude)
      self.output_action = slim.fully_connected(scope='output_action',
                                                inputs=final_hidden,
                                                num_outputs=action_dim,
                                                weights_initializer=weights_initializer,
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
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
#        gradient = tf.Print(gradient, [util.l2_norm(gradient)], 'actor gradient l2_norm ')
        actor_gradients[i] = gradient
      optimiser = tf.train.AdamOptimizer(learning_rate)
      self.train_op = optimiser.apply_gradients(zip(actor_gradients,
                                                    self.trainable_model_vars()))

  def action_given(self, state, add_noise):
    # NOTE: noise is added _outside_ tf graph. we do this simply because the noisy output
    # is never used for any part of computation graph required for online training. it's
    # only used during training after being the replay buffer.
    actions = tf.get_default_session().run(self.output_action,
                                           feed_dict={self.input_state: state})
    if add_noise:
      actions[0] += self.exploration_noise.sample()
      actions = np.clip(1, -1, actions)  # action output is _always_ (-1, 1)
    return actions

  def train(self, state):
    # training actor only requires state since we are trying to maximise the
    # q_value according to the critic.
    return tf.get_default_session().run(self.train_op,
                                        feed_dict={self.input_state: state})


class CriticNetwork(Network):
  """ the critic represents a mapping from state & actors action to a quality score."""

  def __init__(self, namespace, actor, hidden_layer_config, discount=0.99):
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
    self.input_action = tf.stop_gradient(actor.output_action)

    with tf.variable_scope(namespace):
      # TODO: don't add action until later in stack?
      flat_input_state = slim.flatten(self.input_state)
      concat_inputs = tf.concat(1, [flat_input_state, self.input_action])
      # stack of hidden layers
      final_hidden = self.hidden_layers_starting_at(concat_inputs, config=hidden_layer_config)
      # output from critic is a single q-value
      self.q_value = slim.fully_connected(scope='q_value',
                                          inputs=final_hidden,
                                          num_outputs=1,
                                          weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
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
    bellman_rhs = self.reward + (self.terminal_mask * self.discount * target_critic.q_value)

    # note: since we are NOT training target networks we stop gradients flowing to them
    bellman_rhs = tf.stop_gradient(bellman_rhs)

    # the value we are trying to mimimise is the difference between these two; the
    # temporal difference we use a squared loss for optimisation and, as for actor, we
    # wrap optimiser in a namespace so it's not picked up by target network variable
    # handling.
    temporal_difference = bellman_lhs - bellman_rhs
    self.temporal_difference_loss = tf.reduce_mean(tf.pow(temporal_difference, 2))
    with tf.variable_scope("optimiser"):
      optimizer = tf.train.AdamOptimizer(learning_rate)
      gradients = optimizer.compute_gradients(self.temporal_difference_loss)
      for i, (gradient, variable) in enumerate(gradients):
        if gradient is None:  # these are the stop gradient cases; ignore them
          continue
        if gradient_clip_norm is not None:
          gradient = tf.clip_by_norm(gradient, gradient_clip_norm)
#        gradient = tf.Print(gradient, [util.l2_norm(gradient)], 'critic gradient l2_norm ')
        gradients[i] = (gradient, variable)
      self.train_op = optimizer.apply_gradients(gradients)

  def q_gradients_wrt_actions(self):
    """ gradients for the q.value w.r.t just input_action; used for actor training"""
    return tf.gradients(self.q_value, self.input_action)[0]

  def debug_q_value_for(self, state, action=None):
    feed_dict = {self.input_state: state}
    if action is not None:
      feed_dict[self.input_action] = action
    return tf.get_default_session().run(self.q_value, feed_dict=feed_dict)

  def train(self, state_1, action, reward, terminal_mask, state_2):
    tf.get_default_session().run(self.train_op,
                                 feed_dict={self.input_state: state_1,
                                            self.input_action: action,
                                            self.reward: reward,
                                            self.terminal_mask: terminal_mask,
                                            self.input_state_2: state_2})

  def check_loss(self, state_1, action, reward, terminal_mask, state_2):
    return tf.get_default_session().run(self.temporal_difference_loss,
                                        feed_dict={self.input_state: state_1,
                                                   self.input_action: action,
                                                   self.reward: reward,
                                                   self.terminal_mask: terminal_mask,
                                                   self.input_state_2: state_2})


class DeepDeterministicPolicyGradientAgent(object):
  def __init__(self, env, agent_opts):
    self.env = env
    state_shape = self.env.observation_space.shape
    action_dim = self.env.action_space.shape[1]

    # for now, with single machine synchronous training, use a replay memory for training.
    # TODO: switch back to async training with multiple replicas (as in drivebot project)
    self.replay_memory = replay_memory.ReplayMemory(agent_opts.replay_memory_size, 
                                                    state_shape, action_dim)

    # initialise base models for actor / critic and their corresponding target networks
    # target_actor is never used for online sampling so doesn't need explore noise.
    self.actor = ActorNetwork("actor", state_shape, action_dim,
                              agent_opts.actor_hidden_layers,
                              agent_opts.action_noise_theta,
                              agent_opts.action_noise_sigma,
                              agent_opts.actor_activation_init_magnitude)
          
    self.critic = CriticNetwork("critic", self.actor, 
                                agent_opts.critic_hidden_layers)
    self.target_actor = ActorNetwork("target_actor", state_shape, action_dim,
                                     agent_opts.actor_hidden_layers)
    self.target_critic = CriticNetwork("target_critic", self.target_actor,
                                       agent_opts.critic_hidden_layers)

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
    episode_num = 0
    while True:
      # some stats collection
      stats = {"time": int(time.time()), "episode": episode_num}
      rewards = []
      # run an episode
      state_1 = self.env.reset()
      done = False
      # about 1% of the time we record verbose info for entire episode about loss etc
      debug = np.random.random() < 0.01
      while not done:
        # choose action
        action = self.actor.action_given([state_1], add_noise=True)
        # take action step in env
        state_2, reward, done, _ = self.env.step(action)

        if debug:
          print "-----"
          print "state_1", state_1
          print "action", action
          print "reward", reward
          print "done", done
          print "state_2", state_2
          expected_q = float(self.critic.debug_q_value_for([state_1])[0][0])
          expected_target_q = float(self.target_critic.debug_q_value_for([state_1])[0][0])
          print "EXPECTED_Q_VALUES", expected_q, expected_target_q

        # add to replay memory
        self.replay_memory.add(state_1, action, reward, done, state_2)
        # do a training step (after waiting for buffer to fill a bit...)
        if self.replay_memory.size() > batch_size * batches_per_step * 10:
          for _ in xrange(batches_per_step):
            state_1_b, action_b, reward_b, terminal_mask_b, state_2_b = self.replay_memory.random_batch(batch_size)
            self.actor.train(state_1_b)
            self.critic.train(state_1_b, action_b, reward_b, terminal_mask_b, state_2_b)
            self.target_actor.update_weights()
            self.target_critic.update_weights()
            if debug:
              print "Q LOSS", self.critic.check_loss(state_1_b, action_b, reward_b, terminal_mask_b, state_2_b)
        # roll state for next step.
        state_1 = state_2
        rewards.append(reward)

      # dump some stats and progress info
      stats["total_reward"] = np.sum(rewards)
      stats["episode_len"] = len(rewards)
      stats["replay_memory_size"] = self.replay_memory.size()
      print "STATS %s\t%s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                              json.dumps(stats))

      # save if required
      if saver_util is not None:
        saver_util.save_if_required()

      # hack in occasional eval and dumping of weights
      if episode_num % 100 == 0:
        self.run_eval(1)
#      if episode_num % 10000 == 0:
#        self.debug_dump_network_weights()

      # exit when finished
      num_actions_taken += len(rewards)
      if max_num_actions > 0 and num_actions_taken > max_num_actions:
        break
      if max_run_time > 0 and time.time() > start_time + max_run_time:
        break
      episode_num += 1

  def run_eval(self, num_episodes, add_noise=False):
    """ run num_episodes of eval and output episode length and rewards """
    for i in xrange(num_episodes):
      state = self.env.reset()
      total_reward = 0
      steps = 0
      done = False
      while not done:
        action = self.actor.action_given([state], add_noise)
        state, reward, done, _ = self.env.step(action)
#        print "EVALSTEP r%s %s %s %s" % (i, steps, np.linalg.norm(action), reward)
        total_reward += reward
        steps += 1
      print "EVAL", i, steps, total_reward
    self.env.reset()  # just to flush logging, clumsy :/

  def debug_dump_network_weights(self):
    for var in tf.all_variables():
      print "----------------", var.name
      print var.eval()

def main():
  env = bullet_cartpole.BulletCartpole(opts=opts, discrete_actions=False)

  with tf.Session() as sess:  #config=tf.ConfigProto(log_device_placement=True)) as sess:
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

if __name__ == "__main__":
  main()
