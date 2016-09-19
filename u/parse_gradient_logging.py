#!/usr/bin/env python
import sys, re
from collections import Counter
freq = Counter()
nth = 5
for line in sys.stdin:
  m = re.match(".* gradient (.*?) l2_norm \[(.*?)\]", line)
  if not m: continue
  name, val = m.groups()
  freq[name] += 1
  if freq[name] % nth == 0:
    print "%d\t%s\t%s" % (freq[name], name, val)

