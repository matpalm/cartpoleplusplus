#!/usr/bin/env python
import sys

def second_col(l):
  cols = l.split("\t")
  assert len(cols) == 2 or len(cols) == 3
  return float(cols[1])

print "\t".join(["n", "loss", "val", "adv", "t_val"])

loss = val = adv = t_val = None
n = 0
for line in sys.stdin:
  if line.startswith("--"):
    if loss is not None:
      print "\t".join(map(str, [n, loss, val, adv, t_val]))
      loss = val = adv = t_val = None
      n += 1
  elif line.startswith("loss"):
    loss = second_col(line)
  elif line.startswith("val'"):
    t_val = second_col(line)
  elif line.startswith("val"):
    val = second_col(line)
  elif line.startswith("adv"):
    adv = second_col(line)

