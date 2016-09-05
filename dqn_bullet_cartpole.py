#!/usr/bin/env python

# copy pasta from https://github.com/matthiasplappert/keras-rl/blob/master/examples/dqn_cartpole.py
# with some extra arg parsing

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import bullet_cartpole
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num-train', type=int, default=100)
parser.add_argument('--num-eval', type=int, default=0)
parser.add_argument('--load-file', type=str, default=None)
parser.add_argument('--save-file', type=str, default=None)
bullet_cartpole.add_opts(parser)
opts = parser.parse_args()
print "OPTS", opts

ENV_NAME = 'BulletCartpole'

# Get the environment and extract the number of actions.
env = bullet_cartpole.BulletCartpole(opts=opts, discrete_actions=True)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(32))
model.add(Activation('tanh'))
#model.add(Dense(16))
#model.add(Activation('relu'))
#model.add(Dense(16))
#model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

memory = SequentialMemory(limit=50000)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

if opts.load_file is not None:
  print "loading weights from from [%s]" % opts.load_file
  dqn.load_weights(opts.load_file)

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=opts.num_train, visualize=True, verbose=2)

# After training is done, we save the final weights.
if opts.save_file is not None:
  print "saving weights to [%s]" % opts.save_file
  dqn.save_weights(opts.save_file, overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=opts.num_eval, visualize=True)

