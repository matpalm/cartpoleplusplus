#!/usr/bin/env python
import argparse
import bullet_cartpole
import random
import time

parser = argparse.ArgumentParser()
parser.add_argument('--gui', action='store_true')
parser.add_argument('--initial-force', type=float, default=20.0,
                    help="magnitude of initial push, in random direction")
parser.add_argument('--actions', type=str, default='0,1,2,3,4',
                    help='comma seperated list of actions to pick from, if env is discrete')
parser.add_argument('--num-eval', type=int, default=1000)
parser.add_argument('--delay', type=float, default=0.0)
parser.add_argument('--max-episode-len', type=int, default=None)
parser.add_argument('--action-type', type=str, default='discrete',
                    help="either 'discrete' or 'continuous'")
opts = parser.parse_args()

actions = map(int, opts.actions.split(","))

if opts.action_type == 'discrete':
  discrete_actions = True
elif opts.action_type == 'continuous':
  discrete_actions = False
else:
  raise Exception("Unknown action type [%s]" % opts.action_type)

env = bullet_cartpole.BulletCartpole(gui=opts.gui, initial_force=opts.initial_force,
                                     discrete_actions=discrete_actions)

for _ in xrange(opts.num_eval):
  env.reset()
  done = False
  steps = 0
  while not done:
    if discrete_actions:
      action = random.choice(actions)
    else:
      action = env.action_space.sample()
    _state, _reward, done, info = env.step(action)
    steps += 1
    if opts.delay > 0:
      time.sleep(opts.delay)
    if opts.max_episode_len is not None and steps > opts.max_episode_len:
      break
  print steps

