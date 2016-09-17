#!/usr/bin/env python
import sys, re
from collections import Counter
freq = Counter()
for line in sys.stdin:
  m = re.match(".* gradient (.*?) l2_norm \[(.*?)\]", line)
  if not m: continue
  name, val = m.groups()
  print "%d\t%s\t%s" % (freq[name], name, val)
  freq[name] += 1

