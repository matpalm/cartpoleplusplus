#!/usr/bin/env python
import sys, re
import numpy as np

params = [
 "--actor-hidden-layers",
 "--critic-hidden-layers",
 "--action-force",
 "--batch-size",
 "--target-update-rate",
 "--actor-learning-rate",
 "--critic-learning-rate",
 "--actor-gradient-clip",
 "--critic-gradient-clip",
 "--action-noise-theta",
 "--action-noise-sigma",
]
params = [p.replace("--","").replace("-","_") for p in params]

header = ["run_id"] + params + ["eval_type", "eval_val"]
print "\t".join(header)

for run_id in sys.stdin:
  run_id = run_id.strip()  
  row = [run_id]

  for line in open("ckpts/%s/err" % run_id, "r").readlines():
    if line.startswith("Namespace"):
      line = re.sub(r"^Namespace\(", "", line).strip()
      line = re.sub(r"\)$", "", line)
      kvs = {}
      for kv in line.split(", "):
        k, v = kv.split("=")
        kvs[k] = v.replace("'", "")
      for p in params:
        row.append("_"+kvs[p])
      assert len(row) == 12, len(row)

  evals = []
  for line in open("ckpts/%s/out" % run_id, "r"):
    if line.startswith("EVAL"):
      cols = line.strip().split(" ")
      assert len(cols) == 4
      assert cols[0] == "EVAL"
      evals.append(float(cols[3]))
  assert len(evals) > 10
  evals = evals[-10:]

  print "\t".join(map(str, row + ["min", np.min(evals)]))
  print "\t".join(map(str, row + ["mean", np.mean(evals)]))
  print "\t".join(map(str, row + ["max", np.max(evals)]))




  
