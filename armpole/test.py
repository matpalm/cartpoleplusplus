#!/usr/bin/env python
import argparse
import json
import numpy as np
import pybullet as p
import random
import sys
import time

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-episodes', type=int, default=10)
parser.add_argument('--gui', action='store_true')
opts = parser.parse_args()
sys.stderr.write("%s\n" % opts)

ARM = None
POLE = None

def setArmPJoints(positions):
  target_vel = 0
  max_force, kp, kd = 1000, 0.1, 1.0
  for joint_idx, target_pos in enumerate(positions):
    p.setJointMotorControl(ARM, joint_idx,
                           p.POSITION_CONTROL, target_pos, target_vel,
                           max_force, kp, kd)

def setArmVJoints(velocities):
  max_force, kd = 100000, 1.0
  for joint_idx, target_vel in enumerate(velocities):
    p.setJointMotorControl(ARM, joint_idx,
                           p.VELOCITY_CONTROL, target_vel,
                           max_force, kd)

def armState():
  state = []
  for i in xrange(7):
    pos, vel, _torques, motorTorque = p.getJointState(ARM, i)
    state.append((pos, vel, motorTorque))
  return state

def poleState():
  (px,py,pz), (oa,ob,oc,od) = p.getBasePositionAndOrientation(POLE)
  return (px,py,pz, oa,oa,oc,od)

def poleHeight():
  return poleState()[2]

def poleAboveGround():
  return poleHeight() > 0

INITIAL_FORCES = [0] #, 50, 100]

p.connect(p.GUI if opts.gui else p.DIRECT)
p.setGravity(0, 0, -9.81)

ARM = p.loadURDF("models/kuka_iiwa/model.urdf", 0,0,0, 0,0,0,1)
#p.resetJointState(ARM, 1, 3.14/2)
#p.resetJointState(ARM, 3, 3.14/2)
#p.stepSimulation()
POLE = p.loadURDF("models/pole.urdf", 0.5,-0.5,0, 0,0,0,1)

INITIAL_FORCES = [0, 100]

for _ in xrange(opts.num_episodes):
  # reset arm
  for i, js in enumerate([0, 1.57, 0, 1.57, 0, 0, 0]):
    p.resetJointState(ARM, i, js)

  # reset pole just above end effector of arm
  p.resetBasePositionAndOrientation(POLE, (0.42,0,1.1), (0,0,0,1))

  # do initial push in a random direction
  initial_force_magnitude = random.choice(INITIAL_FORCES)
  theta = np.random.random() * 2 * np.pi
  fx, fy = initial_force_magnitude * np.cos(theta), initial_force_magnitude * np.sin(theta)
  for _ in xrange(10):
    p.applyExternalForce(POLE, -1, (fx, fy, 0), (0, 0, 0), p.WORLD_FRAME)
    p.stepSimulation()

  velocities = ((np.random.random(7)*2)-1)  # -1 -> 1
#  velocities *= 2                           # -2 -> 2
#  velocities = np.tanh(velocities)          # squashed back to -1 to 1, but non linearly
  #velocities = [0] * 7

  log_entry = {"initial_force_magnitude": initial_force_magnitude,
               "fx": fx, "fy": fy, "velocities": list(velocities)}
#  log_entry = {"initial_force_magnitude": initial_force_magnitude, "fx": fx, "fy": fy}

  num_steps = 0
  steps_log = []
  while num_steps < 1000:

  #  velocities = ((np.random.random(7)*2)-1)  # -1 -> 1
    setArmVJoints(velocities)
    #setArmPJoints([0, 1.57, 0, 1.57, 0, 0, 0])  # hold initial position

    p.stepSimulation()
 #   if opts.gui:
 #     time.sleep(0.01)

    ps = poleState()    
#    print "ps", ps

    # below ground?
    if ps[2] < 0:
      print "GRODUN!"
      break
    # fallen over?
#    print "???", ps[3:]
#    yaw, pitch, roll = p.getEulerFromQuaternion(ps[3:])
#    r2d = 57.2958
#    print "euler xyz  %0.5f %0.5f %0.5f" % ( yaw*r2d, pitch*r2d, roll*r2d)

#    if abs(pitch) > 0.3 or abs(yaw) > 0.3:
#      print "TILT!", yaw, pitch, roll
#      try:
#        input("sdfsdf")
#      except:
#        pass
#      break

 #   steps_log.append({"arm": armState(), "pole": poleState()})
    num_steps += 1

  log_entry["num_steps"] = num_steps
  log_entry["steps_log"] = steps_log
  print json.dumps(log_entry)


