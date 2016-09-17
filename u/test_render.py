#!/usr/bin/env python
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt

p.connect(p.DIRECT)
#p.loadURDF("models/ground.urdf", 0,0,0, 0,0,0,1)
#p.loadURDF("models/cart.urdf", 0,0,0.08, 0,0,0,1)
p.loadURDF("models/pole.urdf", 0,0,0.35, 0,0,0,1)
cameraPos = (0.75, 0.75, 0.75)
targetPos = (0, 0, 0.2)
cameraUp = (0, 0, 1)
nearVal, farVal = 1, 20
fov = 60
_w, _h, rgba, _depth, _objects = p.renderImage(50, 50, cameraPos, targetPos, cameraUp, nearVal, farVal, fov)
rgba_img = np.reshape(np.asarray(rgba, dtype=np.float32), (50, 50, 4))
rgb_img = rgba_img[:,:,:3]
rgb_img /= 255
plt.imsave("/tmp/ppp.png", rgb_img)
