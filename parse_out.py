#!/usr/bin/env python
import sys, re, json
import numpy as np

# fields we output
# ones specific to policy gradient include ... r_XXX 
print "\t".join(["run_id", "episode", "reward", "loss", "r_min", "r_mean", "r_max"])

def emit(run_id, episode, reward, loss, r_min=0, r_mean=0, r_max=0):
  print "\t".join(map(str, [run_id, episode, reward, loss, r_min, r_mean, r_max]))

for filename in sys.argv[1:]:
  run_id = filename.replace(".out", "")
  episode = 0  # for pg 
  for line in open(filename, "r"):
    if "episode steps:" in line:
      # looks like entry from dqn_cartpole
      try:
        m = re.match(".*episode: (\d+),.*episode reward: (.*?),.*loss: (.*?),", line)
        emit(run_id=run_id, episode=m.group(1), reward=m.group(2),
             loss="0" if m.group(3) == "--" else m.group(3))
      except AttributeError:
        pass
    elif line.startswith("STATS"):      
      # looks like entry from the policy gradient code...
      try:
        d = json.loads(re.sub("STATS.*?\t", "", line))
        emit(run_id=run_id, episode=d['batch'], reward=d['mean_reward'], loss=d['loss'],
             r_min=np.min(d['rewards']), r_mean=np.mean(d['rewards']), r_max=np.max(d['rewards']))
      except ValueError:
        pass  # partial line?
      
