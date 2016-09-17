#!/usr/bin/env python
import os
import random

flags = {
 "--actor-hidden-layers": ["50", "100,50", "100,100,50"],
 "--critic-hidden-layers": ["50", "100,50", "100,100,50"],
 "--action-force": [50, 100],
 "--batch-size": [64, 128, 256],
 "--target-update-rate": [0.01, 0.001, 0.0001],
 "--actor-learning-rate": [0.01, 0.001],
 "--critic-learning-rate": [0.01, 0.001],
 "--actor-gradient-clip":  [None, 50, 100, 200],
 "--critic-gradient-clip":  [None, 50, 100, 200],
 "--action-noise-theta":  [ 0.1, 0.01, 0.001 ],
 "--action-noise-sigma":  [ 0.1, 0.2, 0.5]
}

while True:
  run_id = "run_%s" % str(random.random())[2:]
  cmd = "mkdir ckpts/%s;" % run_id
  cmd += " ./ddpg_cartpole.py"
  cmd += " --max-run-time=3600"
  cmd += " --ckpt-dir=ckpts/%s/ckpts" % run_id
  for flag, values in flags.iteritems():
    value = random.choice(values)
    if value is not None:
      cmd += " %s=%s" % (flag, value)
  cmd += " >ckpts/%s/out 2>ckpts/%s/err" % (run_id, run_id)
  print cmd

