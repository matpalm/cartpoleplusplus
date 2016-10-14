#!/usr/bin/env python
import sys, re, json
import numpy as np

timestamp = None
for line in sys.stdin:
  if line.startswith("STATS"):    
    m = re.match("STATS (.*)\t", line)
    timestamp = m.group(1)
  elif line.startswith("EVAL"):    
    if "STEP" not in line:
      if timestamp is None:
        continue
      cols = line.split(" ")
      assert len(cols) == 4
      total_reward = cols[2]
      print "\t".join([timestamp, total_reward])
    
