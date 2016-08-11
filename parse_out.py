#!/usr/bin/env python
import sys, re
for filename in sys.argv[1:]:
  run_id = filename.replace(".out", "")
  for line in open(filename, "r"):
    if not "episode steps:" in line:
      continue
    try:
      m = re.match(".*episode: (\d+),.*episode reward: (.*?),.*loss: (.*?), mean_absolute_error: (.*?), mean_q:(.*)", line)
      fields = list(m.groups())
      fields.insert(0, run_id)
      print "\t".join(fields)
    except AttributeError:
      pass
