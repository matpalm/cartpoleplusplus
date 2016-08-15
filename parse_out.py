#!/usr/bin/env python
import argparse, sys, re, json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('files', metavar='F', type=str, nargs='+',
                    help='files to process')
parser.add_argument('--pg-emit-all', action='store_true',
                    help="if set emit all rewards in pg case. if false just emit stats")
opts = parser.parse_args()

# fields we output
# ones specific to policy gradient include ... r_XXX 
print "\t".join(["run_id", "n", "episode", "reward", "loss", "r_min", "r_mean", "r_max"])

# emit a record.
n = 0
def emit(run_id, episode, reward, loss, r_min=0, r_mean=0, r_max=0):
  global n
  print "\t".join(map(str, [run_id, n, episode, reward, loss, r_min, r_mean, r_max]))
  n += 1

for filename in opts.files:
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
        if opts.pg_emit_all:
          for reward in d['rewards']:
            emit(run_id=run_id, episode=d['batch'], reward=reward, loss=d['loss'],
                 r_min=np.min(d['rewards']), r_mean=np.mean(d['rewards']),
                 r_max=np.max(d['rewards']))
        else:
          emit(run_id=run_id, episode=d['batch'], reward=d['mean_reward'], loss=d['loss'],
               r_min=np.min(d['rewards']), r_mean=np.mean(d['rewards']),
               r_max=np.max(d['rewards']))
      except ValueError:
        pass  # partial line?
      
