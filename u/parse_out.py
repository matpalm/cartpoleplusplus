#!/usr/bin/env python
import argparse, sys, re, json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('files', metavar='F', type=str, nargs='+',
                    help='files to process')
parser.add_argument('--pg-emit-all', action='store_true',
                    help="if set emit all rewards in pg case. if false just emit stats")
parser.add_argument('--nth', type=int, default=1, help="emit every nth")
opts = parser.parse_args()

# fields we output
KEYS = ["run_id", "exp", "replica", "episode", "loss", "r_min", "r_mean", "r_max"]

# tsv header
print "\t".join(KEYS)

# emit a record.
n = 0
def emit(data):
  global n
  if n % opts.nth == 0:
    print "\t".join(map(str, [data[key] for key in KEYS]))
  n += 1

for filename in opts.files:
  run_id = filename.replace(".out", "").replace(".stats", "")
  run_info = {"run_id": run_id, "exp": run_id[:-1], "replica": run_id[-1]}

  for line in open(filename, "r"):
    if line.startswith("STATS"):
      # looks like entry from the policy gradient code...
      try:
        d = json.loads(re.sub("STATS.*?\t", "", line))
        d.update(run_info)
        d['episode'] = d['batch']
        d['r_min'] = np.min(d['rewards'])
        d['r_mean'] = np.mean(d['rewards'])
        d['r_max'] = np.max(d['rewards'])
        emit(d)
      except ValueError:
        pass  # partial line?
    else:
      # old file format? or just noise? ignore....
      pass
