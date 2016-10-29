#!/usr/bin/env python
import datetime, os, time, yaml, sys
import json
import matplotlib.pyplot as plt
import numpy as np
import StringIO
import tensorflow as tf
import time

def add_opts(parser):
   parser.add_argument('--gradient-clip', type=float, default=5,
                       help="do global clipping to this norm")
   parser.add_argument('--print-gradients', action='store_true', 
                       help="whether to verbose print all gradients and l2 norms")
   parser.add_argument('--optimiser', type=str, default="GradientDescent",
                       help="tf.train.XXXOptimizer to use")
   parser.add_argument('--optimiser-args', type=str, default="{\"learning_rate\": 0.001}",
                       help="json serialised args for optimiser constructor")
   parser.add_argument('--use-dropout', action='store_true',
                       help="include a dropout layers after each fully connected layer")

class StopWatch:
  def reset(self):
    self.start = time.time()
  def time(self):
    return time.time() - self.start
#  def __enter__(self):
#    self.start = time.time()
#    return self
#  def __exit__(self, *args):
#    self.time = time.time() - self.start

def l2_norm(tensor):
  """(row wise) l2 norm of a tensor"""
  return tf.sqrt(tf.reduce_sum(tf.pow(tensor, 2)))

def standardise(tensor):
  """standardise a tensor"""
  # is std_dev not an op in tensorflow?!? i must be taking crazy pills...
  mean = tf.reduce_mean(tensor)
  variance = tf.reduce_mean(tf.square(tensor - mean))
  std_dev = tf.sqrt(variance)
  return (tensor - mean) / std_dev

def clip_and_debug_gradients(gradients, opts):
  # extract just the gradients temporarily for global clipping and then rezip
  if opts.gradient_clip is not None:
    just_gradients, variables = zip(*gradients)
    just_gradients, _ = tf.clip_by_global_norm(just_gradients, opts.gradient_clip)
    gradients = zip(just_gradients, variables)
  # verbose debugging
  if opts.print_gradients:
    for i, (gradient, variable) in enumerate(gradients):
      if gradient is not None:
        gradients[i] = (tf.Print(gradient, [l2_norm(gradient)],
                                 "gradient %s l2_norm " % variable.name), variable)
  # done
  return gradients

def collapsed_successive_ranges(values):
  """reduce an array, e.g. [2,3,4,5,13,14,15], to its successive ranges [2-5, 13-15]"""
  last, start, out = None, None, []
  for value in values:
    if start is None:
      start = value
    elif value != last + 1:
      out.append("%d-%d" % (start, last))
      start = value
    last = value
  out.append("%d-%d" % (start, last))
  return ", ".join(out)

def construct_optimiser(opts):
  optimiser_cstr = eval("tf.train.%sOptimizer" % opts.optimiser)
  args = json.loads(opts.optimiser_args)
  return optimiser_cstr(**args)

def shape_and_product_of(t):
  shape_product = 1
  for dim in t.get_shape():
    try:
      shape_product *= int(dim)
    except TypeError:
      # Dimension(None)
      pass
  return "%s #%s" % (t.get_shape(), shape_product)

class SaverUtil(object):
  def __init__(self, sess, ckpt_dir="/tmp", save_freq=60):
    self.sess = sess
    var_list = [v for v in tf.all_variables() if not "replay_memory" in v.name]
    self.saver = tf.train.Saver(var_list=var_list, max_to_keep=1000)
    self.ckpt_dir = ckpt_dir
    if not os.path.exists(self.ckpt_dir):
      os.makedirs(self.ckpt_dir)
    assert save_freq > 0
    self.save_freq = save_freq
    self.load_latest_ckpt_or_init_if_none()

  def load_latest_ckpt_or_init_if_none(self):
    """loads latests ckpt from dir. if there are non run init variables."""
    # if no latest checkpoint init vars and return
    ckpt_info_file = "%s/checkpoint" % self.ckpt_dir
    if os.path.isfile(ckpt_info_file):
      # load latest ckpt
      info = yaml.load(open(ckpt_info_file, "r"))
      assert 'model_checkpoint_path' in info
      most_recent_ckpt = "%s/%s" % (self.ckpt_dir, info['model_checkpoint_path'])
      sys.stderr.write("loading ckpt %s\n" % most_recent_ckpt)
      self.saver.restore(self.sess, most_recent_ckpt)
      self.next_scheduled_save_time = time.time() + self.save_freq
    else:
      # no latest ckpts, init and force a save now
      sys.stderr.write("no latest ckpt in %s, just initing vars...\n" % self.ckpt_dir)
      self.sess.run(tf.initialize_all_variables())
      self.force_save()

  def force_save(self):
    """force a save now."""
    dts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    new_ckpt = "%s/ckpt.%s" % (self.ckpt_dir, dts)
    sys.stderr.write("saving ckpt %s\n" % new_ckpt)
    start_time = time.time()
    self.saver.save(self.sess, new_ckpt)
    print "save_took", time.time() - start_time
    self.next_scheduled_save_time = time.time() + self.save_freq

  def save_if_required(self):
    """check if save is required based on time and if so, save."""
    if time.time() >= self.next_scheduled_save_time:
      self.force_save()


class OrnsteinUhlenbeckNoise(object):
  """generate time correlated noise for action exploration"""

  def __init__(self, dim, theta=0.01, sigma=0.2, max_magnitude=1.5):
    # dim: dimensionality of returned noise
    # theta: how quickly the value moves; near zero => slow, near one => fast
    #   0.01 gives very roughly 2/3 peaks troughs over ~1000 samples
    # sigma: maximum range of values; 0.2 gives approximately the range (-1.5, 1.5)
    #   which is useful for shifting the output of a tanh which is (-1, 1)
    # max_magnitude: max +ve / -ve value to clip at. dft clip at 1.5 (again for
    #   adding to output from tanh. we do this since sigma gives no guarantees
    #   regarding min/max values.
    self.dim = dim
    self.theta = theta
    self.sigma = sigma
    self.max_magnitude = max_magnitude
    self.state = np.zeros(self.dim)

  def sample(self):
    self.state += self.theta * -self.state
    self.state += self.sigma * np.random.randn(self.dim)
    self.state = np.clip(self.max_magnitude, -self.max_magnitude, self.state)
    return np.copy(self.state)


def write_img_to_png_file(img, filename):
  png_bytes = StringIO.StringIO()
  plt.imsave(png_bytes, img)
  print "writing", filename
  with open(filename, "w") as f:
    f.write(png_bytes.getvalue())

# some hacks for writing state to disk for visualisation
def render_state_to_png(step, state, split_channels=False):
  height, width, num_channels, num_cameras, num_repeats = state.shape
  for c_idx in range(num_cameras):
    for r_idx in range(num_repeats):
      if split_channels:
        for channel in range(num_channels):
          single_channel = state[:,:,channel,c_idx,r_idx]
          img = np.zeros((height, width, 3))
          img[:,:,channel] = single_channel
          write_img_to_png_file(img, "/tmp/state_s%03d_ch%s_c%s_r%s.png" % (step, channel, c_idx, r_idx))
      else:
        img = np.empty((height, width, 3))
        img[:,:,0] = state[:,:,0,c_idx,r_idx]
        img[:,:,1] = state[:,:,1,c_idx,r_idx]
        img[:,:,2] = state[:,:,2,c_idx,r_idx]
        write_img_to_png_file(img, "/tmp/state_s%03d_c%s_r%s.png" % (step, c_idx, r_idx))

def render_action_to_png(step, action):
  import Image, ImageDraw
  img = Image.new('RGB', (50, 50), (50, 50, 50))
  canvas = ImageDraw.Draw(img)
  lx, ly = int(25+(action[0][0]*25)), int(25+(action[0][1]*25))
  canvas.line((25,25, lx,ly), fill="black")
  img.save("/tmp/action_%03d.png" % step)
