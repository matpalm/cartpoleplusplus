#!/usr/bin/env python

from collections import *
import gym
from gym import spaces
import pybullet as p
import numpy as np
import time

np.set_printoptions(precision=3, suppress=True, linewidth=10000)
    
def state_fields_of_pose_of(body_id):
  pose = p.getBasePositionAndOrientation(body_id)
  (x, y, _z), (oa, ob, oc, od) = pose
  return np.asarray([x, y, oa, ob, oc, od])

class BulletCartpole(gym.Env):

  # required fo keras-rl hooks, though not used
  metadata = {
   'render.modes': ['human', 'rgb_array'],
   'video.frames_per_second' : 50
  }

  def __init__(self, gui=True, delay=0.0, max_episode_len=200, action_force=50.0,
               initial_force=55.0, include_cart_in_state=True, random_theta=True):
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
    self.action_force = action_force

    # apply action for this many sim time steps.
    self.sim_step_rate = 10

    # initial push force. this should be enough that taking no action
    # will always result in pole falling after initial_force_steps.
    self.initial_force = initial_force

    # number of sim steps initial force is applied.
    # (see initial_force)
    self.initial_force_steps = 30

    # whether to include the cart pose in state too.
    # if false just include pole info.
    self.include_cart_in_state = include_cart_in_state

    # whether we do initial push in a random direction
    # if false we always push with along x-axis (simplee problem, useful for debugging)
    self.random_theta = random_theta

    # 5 discrete actions: no push, left, right, up, down
    self.action_space = spaces.Discrete(5)  

    # obs space for problem is
    # for pole: pos x/y, orientation a,b,c,d
    # and the same for the cart
    # and then the deltas, for the pole, and possibly the cart
    # we give *2 range for cutoff... and ignore bounds on deltas
    float_max = np.finfo(np.float32).max
    # first half of observation state is the current time step info
    bound = [self.pos_threshold*2] * 2     # pole pos x,y  (threshold*2 for slack)
    bound += [float_max] * 4               # pole q orient a,b,c,d
    if self.include_cart_in_state:
      bound += [self.pos_threshold*2] * 2  # cart pos x,y
      bound += [float_max] * 4             # cart q orient a,b,c,d
    # second half of the state is the delta from last state
    # ( since it's current - last it's already been checked for bounds )
    bound += [float_max] * 6               # pole x,y & a,b,c,d
    if self.include_cart_in_state:
      bound += [float_max] * 6             # cart x,y & a,b,c,d
    # use this bound as upper and lower bound
    bound = np.array(bound)
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
    # TODO
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
      print >>sys.stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! unknown action? [%s].... assuming 0"
#      raise Exception("unknown action [%s]" % action)

    # step simulation forward a bit.
    for _ in xrange(self.sim_step_rate):
      p.stepSimulation()
      p.applyExternalForce(self.cart, -1, (fx,fy,0), (0,0,0), p.WORLD_FRAME)
      if self.delay > 0:
        time.sleep(self.delay)
    self.steps += 1

    # Check for out of bounds by position or orientation on pole.
    # we (re)fetch pose explicitly rather than depending on fields in state.
    (x, y, _z), orient = p.getBasePositionAndOrientation(self.pole)
    ox, oy, _oz = p.getEulerFromQuaternion(orient)
    if abs(x) > self.pos_threshold or abs(y) > self.pos_threshold:
      info['done_reason'] = 'out of position bounds'
      self.done = True
    elif abs(ox) > self.angle_threshold or abs(oy) > self.angle_threshold:
      info['done_reason'] = 'out of orientation bounds'
      self.done = True
    # check for end of episode (by length)
    if self.steps >= self.max_episode_len:
      info['done_reason'] = 'episode length'
      self.done = True

    # return observation tuple
    self.update_current_state()
    return self.observation_state(), 1.0, self.done, info

  def pole_and_cart_state(self):
    states = [state_fields_of_pose_of(self.pole)]
    if self.include_cart_in_state:
      states.append(state_fields_of_pose_of(self.cart))
    return np.concatenate(states)

  def update_current_state(self):
    self.last_state = self.current_state
    self.current_state = self.pole_and_cart_state()

  def observation_state(self):
    # TODO: try with current & last (i.e. not an explicit delta)
    return np.concatenate([self.current_state,
                           self.current_state - self.last_state])

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

    # bootstrap last / current state
    self.last_state = self.pole_and_cart_state()
    self.current_state = self.last_state
    return self.observation_state()


