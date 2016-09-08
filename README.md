# cartpole ++

cartpole++ is a non trivial 3d version of cartpole 
simulated using [bullet physics](http://bulletphysics.org/) where the pole _isn't_ connected to the cart.

![cartpole](cartpole.png)

this repo contains a [gym env](https://gym.openai.com/) for this cartpole as well as example policies trained with ...

* a [deep q network](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) from [keras-rl](https://github.com/matthiasplappert/keras-rl)
* a hand rolled likelihood ratio policy gradient method ( [lrpg_cartpole.py](lrpg_cartpole.py) ) for the discrete control version
* a hand rolled [deep deterministic policy gradient method](http://arxiv.org/abs/1509.02971) ( [ddpg_cartpole.py](ddpg_cartpole.py) ) for the continuous control version

observation state in the low dimensional case is shaped (2, 2, 7)
* axis 0 is a time step. each `env.step` action is repeated in simulation 10 times and `state[0,:,:]` is pose information after step 5 and `state[1,:,:]` is pose information after step 10. the delta between these two can be used to derive velocity.
* axis 1 represents the object. `state[:,0,:]` is the cart, `state[:,1,:]` is the pole.
# axis 2 contains the object's 7d pose (3d position + 4d quaternion orientation)

observation state in the high dimensional case is shaped (50, 50, 6) where
* [:,:,0:3] is a 50x50 pixel RGB rendering of the scene at action repeat step 5 (eg below)
* [:,:,3:6] is a 50x50 pixel RGB rendering of the scene at action repeat step 10

![eg_render](eg_render.png)

( testing the high dimensional case now, see the `pixels` branch )

in general see [the blog post](http://matpalm.com/blog/cartpole_plus_plus/) for more info...

## discrete control version

* 5 actions; go left, right, up, down, do nothing
* +1 reward for each step pole is up.

### random agent

example behaviour of random action agent (click through for video)

[![link](https://img.youtube.com/vi/buSAT-3Q8Zs/0.jpg)](https://www.youtube.com/watch?v=buSAT-3Q8Zs)

```
# some sanity checks...

# no initial push and taking no action (action=0) results in episode timeout of 200 steps
$ ./random_action_agent.py --initial-force=0 --actions="0" --num-eval=100 | ./deciles.py 
[ 200.  200.  200.  200.  200.  200.  200.  200.  200.  200.  200.]

# no initial push and random actions knocks pole over
$ ./random_action_agent.py --initial-force=0 --actions="0,1,2,3,4" --num-eval=100 | ./deciles.py
[ 16.   22.9  26.   28.   31.6  35.   37.4  42.3  48.4  56.1  79. ]

# initial push and no action knocks pole over
$ ./random_action_agent.py --initial-force=55 --actions="0" --num-eval=100 | ./deciles.py
[  6.    7.    7.    8.    8.6   9.   11.   12.3  15.   21.   39. ]

# initial push and random action knocks pole over
$ ./random_action_agent.py --initial-force=55 --actions="0,1,2,3,4" --num-eval=100 | ./deciles.py 
[  3.    5.9   7.    7.7   8.    9.   10.   11.   13.   15.   32. ]
```

### training a dqn

```
$ ./dqn_cartpole.py \
 --num-train=2000000 --num-eval=0 \
 --save-file=ckpt.h5
```

result by numbers...

```
$ ./dqn_cartpole.py \
 --load-file=ckpt.h5 \
 --num-train=0 --num-eval=100 \
 | grep ^Episode | sed -es/.*steps:// | ./deciles.py 
[   5.    35.5   49.8   63.4   79.   104.5  122.   162.6  184.   200.   200. ]
```

result visually (click through for video)

[![link](https://img.youtube.com/vi/zteyMIvhn1U/0.jpg)](https://www.youtube.com/watch?v=zteyMIvhn1U)

```
$ ./dqn_cartpole.py \
 --gui --delay=0.005 \
 --load-file=run11_50.weights.2.h5 \
 --num-train=0 --num-eval=100
```

### training using likelihood ratio policy gradient

policy gradient nails it

```
$ ./lrpg_cartpole.py --rollouts-per-batch=20 --num-train-batches=100 \
 --ckpt-dir=ckpts/foo
```

result by numbers...

```
# deciles
[  13.    70.6  195.8  200.   200.   200.   200.   200.   200.   200.   200. ]
```

result visually (click through for video)

[![link](https://img.youtube.com/vi/aricda9gs2I/0.jpg)](https://www.youtube.com/watch?v=aricda9gs2I)

## continuous control version

* 2d action; force to apply on cart in x & y directions
* +1 base reward for each step pole is up. up to an additional +4 as force applied tends to 0.

### training using deep deterministic policy gradient

```
./ddpg_cartpole.py \
 --actor-hidden-layers="100,100,50" --critic-hidden-layers="100,100,50" \
 --action-force=100 --action-noise-sigma=0.1 --batch-size=256 \
 --max-num-actions=1000000 --ckpt-dir=ckpts/run43
```

result by numbers

```
# episode len deciles
[  30.    48.    56.8   65.    73.    86.   116.4  153.3  200.   200.   200. ]
# reward deciles
[  35.51154724  153.20243076  178.7908135   243.38630372  272.64655323
  426.95298195  519.25360223  856.9702368   890.72279221  913.21068417
  955.50168709]
```

result visually (click through for video)

[![link](https://img.youtube.com/vi/8X05GA5ZKvQ/0.jpg)](https://www.youtube.com/watch?v=8X05GA5ZKvQ)


## general utils

run a random agent, logging events to disk (outputs total rewards per episode)

note: for replay logging will need to compile protobuffer `protoc event.proto --python_out=.`

```
$ ./random_action_agent.py --event-log=test.log --num-eval=10 --action-type=continuous
12
14
...
```

review event.log (either from ddpg training or from random agent)

```
$ ./event_log.py --log-file=test.log --echo
event {
  state {
    cart_pose: 0.116232253611
    cart_pose: 0.0877446383238
    cart_pose: 0.0748709067702
    cart_pose: 1.14359036161e-05
    cart_pose: 5.10180834681e-05
    cart_pose: 0.0653914809227
    cart_pose: 0.997859716415
    pole_pose: 0.000139251351357
    pole_pose: -0.0611916743219
    pole_pose: 0.344804286957
    pole_pose: -0.123383037746
    pole_pose: 0.00611496530473
    pole_pose: 0.0471726879478
    pole_pose: 0.991218447685
    render {
      height: 120
      width: 160
      rgba: "\211PNG\r\n\032\n\000\..."
    }
  }
  is_terminal: false
  action: -0.157108291984
  action: 0.330988258123
  reward: 4.0238070488
}
...
```

generate images from event.log

```
$ ./event_log.py --log-file=test.log --img-output-dir=eg_renders
$ find eg_renders -type f | sort
eg_renders/e_00000/s_00000.png
eg_renders/e_00000/s_00001.png
...
eg_renders/e_00009/s_00018.png
eg_renders/e_00009/s_00019.png
```
