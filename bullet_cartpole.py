#!/usr/bin/env python

from collections import *
import gym
from gym import spaces
import numpy as np
import pybullet as p
import sys
import time

np.set_printoptions(precision=3, suppress=True, linewidth=10000)

def add_opts(parser):
  parser.add_argument('--gui', action='store_true')
  parser.add_argument('--delay', type=float, default=0.0)
  parser.add_argument('--action-force', type=float, default=50.0,
                      help="magnitude of action force applied per step")
  parser.add_argument('--initial-force', type=float, default=55.0,
                      help="magnitude of initial push, in random direction")
  parser.add_argument('--no-random-theta', action='store_true')
  parser.add_argument('--action-repeats', type=int, default=3,
                      help="number of action repeats")
  parser.add_argument('--steps-per-repeat', type=int, default=4,
                      help="number of sim steps per repeat")
  parser.add_argument('--num-cameras', type=int, default=1,
                      help="how many camera points to render; 1 or 2")
  parser.add_argument('--event-log-out', type=str, default=None,
                      help="path to record event log.")
  parser.add_argument('--max-episode-len', type=int, default=200,
                      help="maximum episode len for cartpole")
  parser.add_argument('--use-raw-pixels', action='store_true',
                      help="use raw pixels as state instead of cart/pole poses")
  parser.add_argument('--render-width', type=int, default=50,
                      help="if --use-raw-pixels render with this width")
  parser.add_argument('--render-height', type=int, default=50,
                      help="if --use-raw-pixels render with this height")
  parser.add_argument('--reward-calc', type=str, default='fixed',
                      help="'fixed': 1 per step. 'angle': 2*max_angle - ox - oy. 'action': 1.5 - |action|. 'angle_action': both angle and action")

def state_fields_of_pose_of(body_id):
  (x,y,z), (a,b,c,d) = p.getBasePositionAndOrientation(body_id)
  return np.array([x,y,z,a,b,c,d])

class BulletCartpole(gym.Env):

  def __init__(self, opts, discrete_actions):
    self.gui = opts.gui
    self.delay = opts.delay if self.gui else 0.0

    self.max_episode_len = opts.max_episode_len

    # threshold for pole position.
    # if absolute x or y moves outside this we finish episode
    self.pos_threshold = 2.0  # TODO: higher?

    # threshold for angle from z-axis.
    # if x or y > this value we finish episode.
    self.angle_threshold = 0.3  # radians; ~= 12deg

    # force to apply per action simulation step.
    # in the discrete case this is the fixed force applied
    # in the continuous case each x/y is in range (-F, F)
    self.action_force = opts.action_force

    # initial push force. this should be enough that taking no action will always
    # result in pole falling after initial_force_steps but not so much that you
    # can't recover. see also initial_force_steps.
    self.initial_force = opts.initial_force

    # number of sim steps initial force is applied.
    # (see initial_force)
    self.initial_force_steps = 30

    # whether we do initial push in a random direction
    # if false we always push with along x-axis (simplee problem, useful for debugging)
    self.random_theta = not opts.no_random_theta

    # true if action space is discrete; 5 values; no push, left, right, up & down
    # false if action space is continuous; fx, fy both (-action_force, action_force)
    self.discrete_actions = discrete_actions

    # 5 discrete actions: no push, left, right, up, down
    # 2 continuous action elements; fx & fy
    if self.discrete_actions:
      self.action_space = spaces.Discrete(5)
    else:
      self.action_space = spaces.Box(-1.0, 1.0, shape=(1, 2))

    # open event log
    if opts.event_log_out:
      import event_log
      self.event_log = event_log.EventLog(opts.event_log_out, opts.use_raw_pixels)
    else:
      self.event_log = None

    # how many time to repeat each action per step().
    # and how many sim steps to do per state capture
    # (total number of sim steps = action_repeats * steps_per_repeat
    self.repeats = opts.action_repeats
    self.steps_per_repeat = opts.steps_per_repeat

    # how many cameras to render?
    # if 1 just render from front
    # if 2 render from front and 90deg side
    if opts.num_cameras not in [1, 2]:
      raise ValueError("--num-cameras must be 1 or 2")
    self.num_cameras = opts.num_cameras

    # whether we are using raw pixels for state or just pole + cart pose
    self.use_raw_pixels = opts.use_raw_pixels

    # in the use_raw_pixels is set we will be rendering
    self.render_width = opts.render_width
    self.render_height = opts.render_height

    # decide observation space
    if self.use_raw_pixels:
      # in high dimensional case each observation is an RGB images (H, W, 3)
      # we have R repeats and C cameras resulting in (H, W, 3, R, C)
      # final state fed to network is concatenated in depth => (H, W, 3*R*C)
      state_shape = (self.render_height, self.render_width, 3,
                     self.num_cameras, self.repeats)
    else:
      # in the low dimensional case obs space for problem is (R, 2, 7)
      #  R = number of repeats
      #  2 = two items; cart & pole
      #  7d tuple for pos + orientation pose
      state_shape = (self.repeats, 2, 7)
    float_max = np.finfo(np.float32).max
    self.observation_space = gym.spaces.Box(-float_max, float_max, state_shape)

    # check reward type
    assert opts.reward_calc in ['fixed', 'angle', 'action', 'angle_action']
    self.reward_calc = opts.reward_calc

    # no state until reset.
    self.state = np.empty(state_shape, dtype=np.float32)

    # setup bullet
    p.connect(p.GUI if self.gui else p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.loadURDF("models/ground.urdf", 0,0,0, 0,0,0,1)
    self.cart = p.loadURDF("models/cart.urdf", 0,0,0.08, 0,0,0,1)
    self.pole = p.loadURDF("models/pole.urdf", 0,0,0.35, 0,0,0,1)

  def _configure(self, display=None):
    pass

  def _seed(self, seed=None):
    pass

  def _render(self, mode, close):
    pass

  def _step(self, action):
    if self.done:
      print >>sys.stderr, "calling step after done????"
      return np.copy(self.state), 0, True, {}

    info = {}

    # based on action decide the x and y forces
    fx = fy = 0
    if self.discrete_actions:
      if action == 0:
        pass
      elif action == 1:
        fx = self.action_force
      elif action == 2:
        fx = -self.action_force
      elif action == 3:
        fy = self.action_force
      elif action == 4:
        fy = -self.action_force
      else:
        raise Exception("unknown discrete action [%s]" % action)
    else:
      fx, fy = action[0] * self.action_force

    # step simulation forward. at the end of each repeat we set part of the step's
    # state by capture the cart & pole state in some form.
    for r in xrange(self.repeats):
      for _ in xrange(self.steps_per_repeat):
        p.stepSimulation()
        p.applyExternalForce(self.cart, -1, (fx,fy,0), (0,0,0), p.WORLD_FRAME)
        if self.delay > 0:
          time.sleep(self.delay)
      self.set_state_element_for_repeat(r)
    self.steps += 1

    # Check for out of bounds by position or orientation on pole.
    # we (re)fetch pose explicitly rather than depending on fields in state.
    (x, y, _z), orient = p.getBasePositionAndOrientation(self.pole)
    ox, oy, _oz = p.getEulerFromQuaternion(orient)  # roll / pitch / yaw
    if abs(x) > self.pos_threshold or abs(y) > self.pos_threshold:
      info['done_reason'] = 'out of position bounds'
      self.done = True
      reward = 0.0
    elif abs(ox) > self.angle_threshold or abs(oy) > self.angle_threshold:
      # TODO: probably better to do explicit angle from z?
      info['done_reason'] = 'out of orientation bounds'
      self.done = True
      reward = 0.0
    # check for end of episode (by length)
    if self.steps >= self.max_episode_len:
      info['done_reason'] = 'episode length'
      self.done = True

    # calc reward, fixed base of 1.0
    reward = 1.0
    if self.reward_calc == "angle" or self.reward_calc == "angle_action":
      # clip to zero since angles can be past threshold
      reward += max(0, 2 * self.angle_threshold - np.abs(ox) - np.abs(oy))
    if self.reward_calc == "action" or self.reward_calc == "angle_action":
      # max norm will be sqr(2) ~= 1.4.
      # reward is already 1.0 to add another 0.5 as o0.1 buffer from zero
      reward += 0.5 - np.linalg.norm(action[0])

    # log this event.
    # TODO in the --use-raw-pixels case would be nice to have poses in state repeats too.
    if self.event_log:
      self.event_log.add(self.state, action, reward)

    # return observation
    return np.copy(self.state), reward, self.done, info

  def render_rgb(self, camera_idx):
    cameraPos = [(0.0, 0.75, 0.75), (0.75, 0.0, 0.75)][camera_idx]
    targetPos = (0, 0, 0.3)
    cameraUp = (0, 0, 1)
    nearVal, farVal = 1, 20
    fov = 60
    _w, _h, rgba, _depth, _objects = p.renderImage(self.render_width, self.render_height,
                                                   cameraPos, targetPos, cameraUp,
                                                   nearVal, farVal, fov)
    # convert from 1d uint8 array to (H,W,3) hacky hardcode whitened float16 array.
    # TODO: for storage concerns could just store this as uint8 (which it is)
    # and normalise 0->1 + whiten later.
    rgba_img = np.reshape(np.asarray(rgba, dtype=np.float16),
                          (self.render_height, self.render_width, 4))
    rgb_img = rgba_img[:,:,:3]  # slice off alpha, always 1.0
    rgb_img /= 255
    return rgb_img

  def set_state_element_for_repeat(self, repeat):
    if self.use_raw_pixels:
      # high dim caseis (H, W, 3, C, R)
      # H, W, 3 -> height x width, 3 channel RGB image
      # C -> camera_idx; 0 or 1
      # R -> repeat
      for camera_idx in range(self.num_cameras):
        self.state[:,:,:,camera_idx,repeat] = self.render_rgb(camera_idx)
    else:
      # in low dim case state is (R, 2, 7)
      # R -> repeat, 2 -> 2 objects (cart & pole), 7 -> 7d pose
      self.state[repeat][0] = state_fields_of_pose_of(self.cart)
      self.state[repeat][1] = state_fields_of_pose_of(self.pole)

  def _reset(self):
    # reset state
    self.steps = 0
    self.done = False

    # reset pole on cart in starting poses
    p.resetBasePositionAndOrientation(self.cart, (0,0,0.08), (0,0,0,1))
    p.resetBasePositionAndOrientation(self.pole, (0,0,0.35), (0,0,0,1))
    for _ in xrange(100): p.stepSimulation()

    # give a fixed force push in a random direction to get things going...
    theta = (np.random.random() * 2 * np.pi) if self.random_theta else 0.0
    fx, fy = self.initial_force * np.cos(theta), self.initial_force * np.sin(theta)
    for _ in xrange(self.initial_force_steps):
      p.stepSimulation()
      p.applyExternalForce(self.cart, -1, (fx, fy, 0), (0, 0, 0), p.WORLD_FRAME)
      if self.delay > 0:
        time.sleep(self.delay)

    # bootstrap state by running for all repeats
    for i in xrange(self.repeats):
      self.set_state_element_for_repeat(i)

    # reset event log (if applicable) and add entry with only state
    if self.event_log:
      self.event_log.reset()
      self.event_log.add_just_state(self.state)

    # return this state
    return np.copy(self.state)
