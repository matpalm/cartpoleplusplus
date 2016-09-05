#!/usr/bin/env python
import argparse
import bullet_cartpole
import random
import time

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--actions', type=str, default='0,1,2,3,4',
                    help='comma seperated list of actions to pick from, if env is discrete')
parser.add_argument('--num-eval', type=int, default=1000)
parser.add_argument('--action-type', type=str, default='discrete',
                    help="either 'discrete' or 'continuous'")
bullet_cartpole.add_opts(parser)
opts = parser.parse_args()

actions = map(int, opts.actions.split(","))

if opts.action_type == 'discrete':
  discrete_actions = True
elif opts.action_type == 'continuous':
  discrete_actions = False
else:
  raise Exception("Unknown action type [%s]" % opts.action_type)

env = bullet_cartpole.BulletCartpole(opts=opts, discrete_actions=discrete_actions)

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
env.reset()  # hack to flush last event log if required
