#!/usr/bin/env python
import datetime, os, time, yaml, sys
import tensorflow as tf

def standardise(tensor):
  """ standardise a tensor. """
  # is std_dev not an op in tensorflow?!? i must be taking crazy pills...
  mean = tf.reduce_mean(tensor)
  variance = tf.reduce_mean(tf.square(tensor - mean))
  std_dev = tf.sqrt(variance)
  return (tensor - mean) / std_dev

class SaverUtil(object):
  def __init__(self, sess, ckpt_dir="/tmp", save_freq=60):
    self.sess = sess
    self.saver = tf.train.Saver()
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
    self.saver.save(self.sess, new_ckpt)
    self.next_scheduled_save_time = time.time() + self.save_freq

  def save_if_required(self):
    """check if save is required based on time and if so, save."""
    if time.time() >= self.next_scheduled_save_time:
      self.force_save()
