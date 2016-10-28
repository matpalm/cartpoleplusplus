#!/usr/bin/env python
import glob
import Image, ImageDraw, ImageChops
import matplotlib.pyplot as plt
import os

# hacktastic stitching of activation renderings from an eval run

step = 1
while True:
  if not os.path.isfile("/tmp/state_s%03d_c0_r0.png" % step):
    print "rendered to step", step
    break
  background = Image.new('RGB', (600, 250), (0, 0, 0))
  canvas = ImageDraw.Draw(background)
  canvas.text((0, 0), str(step))
  canvas.text((30, 0), "c0")
  canvas.text((85, 0), "c1")
  canvas.text((0, 30), "r0")
  canvas.text((0, 85), "r1")
  canvas.text((55, 130), "diffs")
  # draw cameras and repeats up in top corner, with a difference below them
  for c in [0, 1]:
    r0 = Image.open("/tmp/state_s%03d_c%d_r0.png" % (step, c))
    r1 = Image.open("/tmp/state_s%03d_c%d_r1.png" % (step, c))
    background.paste(r0, (15+c*55, 15))
    background.paste(r1, (15+c*55, 70))
    diff = ImageChops.invert(ImageChops.difference(r0, r1))
    background.paste(diff, (15+c*55, 145))
  # 3 conv layers, 10 filters each
  for p in range(3):  
    for f in range(10):
      act = Image.open("/tmp/activation_s%03d_p%d_f%02d.png" % (step, p, f))
      act = act.resize((40, 40))
      background.paste(act, (130+(f*45), 20+(p*45)))
  # down the bottom draw in the force magnitude
  background.paste(Image.open("/tmp/action_%03d.png" % step), (250, 170))
  # write image
  background.save("/tmp/viz_%03d.png" % step)
  step += 1

