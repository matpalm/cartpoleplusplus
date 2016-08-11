#!/usr/bin/env python
import argparse
import bullet_cartpole
import random
import time

parser = argparse.ArgumentParser()
parser.add_argument('--gui', type=bool, default=False)
parser.add_argument('--initial-force', type=float, default=20.0, help="magnitude of initial push, in random direction")
parser.add_argument('--actions', type=str, default='0,1,2,3,4', help='comma seperated list of actions to pick from')
parser.add_argument('--num-eval', type=int, default=1000)
parser.add_argument('--delay', type=float, default=0.0)
parser.add_argument('--max-episode-len', type=int, default=None)
opts = parser.parse_args()

actions = map(int, opts.actions.split(","))

env = bullet_cartpole.BulletCartpole(gui=opts.gui, initial_force=opts.initial_force)

for _ in xrange(opts.num_eval):
  env.reset()
  done = False
  steps = 0
  while not done:
    _state, _reward, done, info = env.step(random.choice(actions))
    steps += 1
    if opts.delay > 0:
      time.sleep(opts.delay)
    if opts.max_episode_len is not None and steps > opts.max_episode_len:
      break
  print steps

