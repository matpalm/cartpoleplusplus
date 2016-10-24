#!/usr/bin/env python
# grep EVAL $* | grep -v EVALSTEP  | perl -plne's/runs\///;s/:EVAL / /;s/\/out//;'  | cut -f1,3,4 -d' '
from multiprocessing import Pool
import sys, re

def process(filename):
  col0 = filename.replace("runs/", "").replace("/out", "")
  time = None
  first_time = None
  for line in open(filename).readlines():
    if line.startswith("STATS"):
      m = re.match(".*\"time\": (.*?),", line)
      time = float(m.group(1))
      if first_time is None: first_time = time
      continue
    if not line.startswith("EVAL"): continue
    if line.startswith("EVALSTEP"): continue
    if time is None: continue
    sys.stdout.write("%s %s %s" % (col0, (time-first_time), line.replace("EVAL 0 ", "")))
    sys.stdout.flush()

#if __name__ == '__main__':
#  p = Pool(5)
#  p.map(process, sys.argv[1:])
for filename in sys.argv[1:]:
  process(filename)

