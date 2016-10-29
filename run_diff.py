#!/usr/bin/env python
# slurp the argparse debug lines from two 'run/NNN/err' files and show a side by side diff

import sys, re

def config(f):
  c = {}
  for line in open("runs/%s/err" % f, "r"):
    m = re.match("^Namespace\((.*)\)$", line)
    if m:
      config = m.group(1)
      while config:
        if "," in config:
          m = re.match("^((.*?)=(.*?),\s+)", config)
          pair, key, value = m.groups()
          if value.startswith("'"):
            # value is a string, need special handling since it might contain a comma!
            m = re.match("^((.*?)='(.*?)',\s+)", config)
            pair, key, value = m.groups()
        else:
          # final entry  
          pair = config
          key, value = config.split("=")
        # ignore run related keys
        if key not in ['ckpt_dir', 'event_log_out']:
          c[key] = value
        config = config.replace(pair, "")
      return c
  assert False, "no namespace in file?"

c1 = config(sys.argv[1])
c2 = config(sys.argv[2])

#unset_defaults = {"use_dropout": "False"}

data = []
max_widths = [0,0,0]
for key in sorted(set(c1.keys()).union(c2.keys())):
  c1v = c1[key] if key in c1 else " "#unset_defaults[key]
  c2v = c2[key] if key in c2 else " "#unset_defaults[key]
  data.append((key, c1v, c2v))
  max_widths[0] = max(max_widths[0], len(key))
  max_widths[1] = max(max_widths[1], len(c1v))
  max_widths[2] = max(max_widths[2], len(c2v))
for k, c1, c2 in data:
  format_str = "%%%ds %%%ds %%%ds %%s" % tuple(max_widths)
  star = " ***" if c1 != c2 else " "
  print format_str % (k, c1, c2, star)
