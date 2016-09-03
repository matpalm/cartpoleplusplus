#!/usr/bin/env python

from collections import *
import gym
from gym import spaces
import numpy as np
import pybullet as p
import sys
import time

np.set_printoptions(precision=3, suppress=True, linewidth=10000)

def state_fields_of_pose_of(body_id):
  (x,y,z), (a,b,c,d) = p.getBasePositionAndOrientation(body_id)
  return np.array([x,y,z,a,b,c,d])

def render_rgb(width, height):
  # call to pybullet render
  cameraPos = (0.5, 0.5, 0.5)
  targetPos = (0, 0, 0.2)
  cameraUp = (0, 0, 1)
  nearVal, farVal = 1, 20
  fov = 60
  _w, _h, rgba, _depth, _objects = p.renderImage(width, height,
                                                 cameraPos, targetPos, cameraUp,
                                                 nearVal, farVal, fov)
  # convert from 1d array of 1 -> 255 to np array 0.0 -> 1.0
  rgba_img = np.reshape(np.asarray(rgba, dtype=np.float32),
                        (height, width, 4))
  rgb_img = rgba_img[:,:,:3]  # slice off alpha, always 1.0
  return rgb_img / 255

class BulletCartpole(gym.Env):

  # required fo keras-rl hooks, though not used
  metadata = {
   'render.modes': ['human', 'rgb_array'],
   'video.frames_per_second' : 50
  }

  def __init__(self, gui=True, delay=0.0, max_episode_len=200, action_force=50.0,
               initial_force=55.0, random_theta=True, discrete_actions=True,
               event_log_file=None, state_as_raw_pixels=False,
               render_width=160, render_height=120):

    self.gui = gui
    self.delay = delay if gui else 0.0
    self.max_episode_len = max_episode_len

    # threshold for pole position.
    # if absolute x or y moves outside this we finish episode
    self.pos_threshold = 2.0  # TODO: higher?

    # threshold for angle from z-axis.
    # if x or y > this value we finish episode.
    self.angle_threshold = 0.3  # radians; ~= 12deg

    # force to apply per action simulation step.
    # in the discrete case this is the fixed force applied
    # in the continuous case each x/y is in range (-F, F)
    self.action_force = action_force

    # apply action for this many sim time steps.
    self.sim_step_rate = 10

    # initial push force. this should be enough that taking no action will always
    # result in pole falling after initial_force_steps but not so much that you
    # can't recover. see also initial_force_steps.
    self.initial_force = initial_force

    # number of sim steps initial force is applied.
    # (see initial_force)
    self.initial_force_steps = 30

    # whether we do initial push in a random direction
    # if false we always push with along x-axis (simplee problem, useful for debugging)
    self.random_theta = random_theta

    # true if action space is discrete; 5 values; no push, left, right, up & down
    # false if action space is continuous; fx, fy both (-action_force, action_force)
    self.discrete_actions = discrete_actions

    # 5 discrete actions: no push, left, right, up, down
    if self.discrete_actions:
      self.action_space = spaces.Discrete(5)
    else:
      self.action_space = spaces.Box(-1.0, 1.0, shape=(1, 2))

    # open event log
    if event_log_file:
      import event_log
      self.event_log = event_log.EventLog(event_log_file)
    else:
      self.event_log = None

    # whether we are using raw pixels for state or just pole + cart pose
    self.state_as_raw_pixels = state_as_raw_pixels

    # if either event_log_file or state_as_raw_pixels is set we will be rendering
    self.render_width = render_width
    self.render_height = render_height

    # decide observation space
    if self.state_as_raw_pixels:
      # in high dimensional case observation space is _two_ rendered rgb images
      # concatenated; first is current frame, second is previous frame
      self.observation_space = gym.spaces.Box(0, 1, (render_height*2, render_width, 3))
    else:
      # in low dimensional case observation state space is ...
      #  7 tuple for pose (pos x,y,z & quaternion orientation a,b,c,d)
      #  * 2 for cart & pole
      #  * 2 for current / last time step
      # ( don't worry about correct bounds, we won't be sampling from
      # these & just out of bounds obs just causes stdout "warning" noise )
      float_max = np.finfo(np.float32).max
      bound = np.array([float_max] * 7 * 2 * 2)
      self.observation_space = gym.spaces.Box(-bound, bound)

    p.connect(p.GUI if self.gui else p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.loadURDF("models/ground.urdf", 0,0,0, 0,0,0,1)
    self.cart = p.loadURDF("models/cart.urdf", 0,0,0.08, 0,0,0,1)
    self.pole = p.loadURDF("models/pole.urdf", 0,0,0.35, 0,0,0,1)
    self.reset()

  def _configure(self, display=None):
    pass

  def _seed(self, seed=None):
    pass

  def _render(self, mode, close):
    pass

  def _step(self, action):
    if self.done:
      print "calling step after done????"
      return self.observation_state(), 0, True, {}

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

    # step simulation forward a bit.
    for _ in xrange(self.sim_step_rate):
      p.stepSimulation()
      p.applyExternalForce(self.cart, -1, (fx,fy,0), (0,0,0), p.WORLD_FRAME)
      if self.delay > 0:
        time.sleep(self.delay)
    self.steps += 1

    # calculate reward.
    reward = None
    if self.discrete_actions:
      reward = 1.0
    else:
      # base reward of 1.0 but linear ramp up to 5.0 if you don't apply much force.
      max_abs_force = self.action_force * 2
      abs_force = abs(fx) + abs(fy)
      reward = 1.0 + 4.0 - (4.0 * abs_force / max_abs_force)
      assert reward >= 0  # TODO REMOVE

    # Check for out of bounds by position or orientation on pole.
    # we (re)fetch pose explicitly rather than depending on fields in state.
    (x, y, _z), orient = p.getBasePositionAndOrientation(self.pole)
    ox, oy, _oz = p.getEulerFromQuaternion(orient)
    if abs(x) > self.pos_threshold or abs(y) > self.pos_threshold:
      info['done_reason'] = 'out of position bounds'
      self.done = True
      reward = 0.0
    elif abs(ox) > self.angle_threshold or abs(oy) > self.angle_threshold:
      info['done_reason'] = 'out of orientation bounds'
      self.done = True
      reward = 0.0
    # check for end of episode (by length)
    if self.steps >= self.max_episode_len:
      info['done_reason'] = 'episode length'
      self.done = True

    # log this event
    if self.event_log:
      print "TODO!" # need to always call state_fields_of_pose_of pole & cart
      # and pass instead of self.current_state furthermore move this to BELOW
      # update current state in case we can use that cached value
#      self.event_log.add(self.current_state, self.done, action, reward,
#                         render_rgb(self.render_width, self.render_height))

    # update state and return observation
    self.update_current_state()
    return self.observation_state(), reward, self.done, info

  def pole_and_cart_state(self):
    if self.state_as_raw_pixels:
      return render_rgb(self.render_width, self.render_height)
    else:
      return np.concatenate([state_fields_of_pose_of(self.pole),
                             state_fields_of_pose_of(self.cart)])

  def update_current_state(self):
    self.last_state = self.current_state
    self.current_state = self.pole_and_cart_state()

  def observation_state(self):
    return np.concatenate([self.current_state, self.last_state])

  def _reset(self):
    # reset state
    self.steps = 0
    self.done = False

    # reset event log (if applicable)
    if self.event_log:
      self.event_log.reset()

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

    # bootstrap last / current state
    self.last_state = self.pole_and_cart_state()
    self.current_state = self.last_state
    return self.observation_state()


