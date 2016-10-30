import numpy as np
import operator
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import util

# TODO: move opts only used in this module to an add_opts method
#       requires fixing the bullet-before-slim problem though :/

IS_TRAINING = tf.placeholder(tf.bool, name="is_training")

class Network(object):
  """Common class for handling ops for making / updating target networks."""

  def __init__(self, namespace):
    self.namespace = namespace
    self.target_update_op = None

  def _create_variables_copy_op(self, source_namespace, affine_combo_coeff):
    """create an op that does updates all vars in source_namespace to target_namespace"""
    assert affine_combo_coeff >= 0.0 and affine_combo_coeff <= 1.0
    assign_ops = []
    with tf.variable_scope(self.namespace, reuse=True):
      for src_var in tf.all_variables():
        if not src_var.name.startswith(source_namespace):
          continue
        target_var_name = src_var.name.replace(source_namespace+"/", "").replace(":0", "")
        target_var = tf.get_variable(target_var_name)
        assert src_var.get_shape() == target_var.get_shape()
        assign_ops.append(target_var.assign_sub(affine_combo_coeff * (target_var - src_var)))
    single_assign_op = tf.group(*assign_ops)
    return single_assign_op

  def set_as_target_network_for(self, source_network, target_update_rate):
    """Create an op that will update this networks weights based on a source_network"""
    # first, as a one off, copy _all_ variables across.
    # i.e. initial target network will be a copy of source network.
    op = self._create_variables_copy_op(source_network.namespace, affine_combo_coeff=1.0)
    tf.get_default_session().run(op)
    # next build target update op for running later during training
    self.update_weights_op = self._create_variables_copy_op(source_network.namespace,
                                                           target_update_rate)

  def update_weights(self):
    """called during training to update target network."""
    if self.update_weights_op is None:
      raise Exception("not a target network? or set_source_network not yet called")
    return tf.get_default_session().run(self.update_weights_op)

  def trainable_model_vars(self):
    v = []
    for var in tf.all_variables():
      if var.name.startswith(self.namespace):
        v.append(var)
    return v

  def hidden_layers_starting_at(self, layer, layer_sizes, opts=None):
    # TODO: opts=None => will force exception on old calls....
    if not isinstance(layer_sizes, list):
      layer_sizes = map(int, layer_sizes.split(","))
    assert len(layer_sizes) > 0
    for i, size in enumerate(layer_sizes):
      layer = slim.fully_connected(scope="h%d" % i,
                                  inputs=layer,
                                  num_outputs=size,
                                  weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                  activation_fn=tf.nn.relu)
      if opts.use_dropout:
        layer = slim.dropout(layer, is_training=IS_TRAINING, scope="do%d" % i)
    return layer

  def simple_conv_net_on(self, input_layer, opts):
    if opts.use_batch_norm:
      normalizer_fn = slim.batch_norm
      normalizer_params = { 'is_training': IS_TRAINING }
    else:
      normalizer_fn = None
      normalizer_params = None

    # optionally drop blue channel, in a simple cart pole env we only need r/g
    #if opts.drop_blue_channel:
    #  input_layer = input_layer[:,:,:,0:2,:,:]

    # state is (batch, height, width, rgb, camera_idx, repeat)
    # rollup rgb, camera_idx and repeat into num_channels
    # i.e. (batch, height, width, rgb*camera_idx*repeat)
    height, width = map(int, input_layer.get_shape()[1:3])
    num_channels = input_layer.get_shape()[3:].num_elements()
    input_layer = tf.reshape(input_layer, [-1, height, width, num_channels])
    print >>sys.stderr, "input_layer", util.shape_and_product_of(input_layer)

    # whiten image, per channel, using batch_normalisation layer with
    # params calculated directly from batch.
    axis = list(range(input_layer.get_shape().ndims - 1))
    batch_mean, batch_var = tf.nn.moments(input_layer, axis)  # gives moments per channel
    whitened_input_layer = tf.nn.batch_normalization(input_layer, batch_mean, batch_var,
                                                     scale=None, offset=None,
                                                     variance_epsilon=1e-6)

    # TODO: num_outputs here are really dependant on the incoming channels,
    # which depend on the #repeats & cameras so they should be a param.
    model = slim.conv2d(whitened_input_layer, num_outputs=10, kernel_size=[5, 5],
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        scope='conv1')
    model = slim.max_pool2d(model, kernel_size=[2, 2], scope='pool1')
    self.pool1 = model
    print >>sys.stderr, "pool1", util.shape_and_product_of(model)

    model = slim.conv2d(model, num_outputs=10, kernel_size=[5, 5],
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        scope='conv2')
    model = slim.max_pool2d(model, kernel_size=[2, 2], scope='pool2')
    self.pool2 = model
    print >>sys.stderr, "pool2", util.shape_and_product_of(model)

    model = slim.conv2d(model, num_outputs=10, kernel_size=[3, 3],
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        scope='conv3')
    model = slim.max_pool2d(model, kernel_size=[2, 2], scope='pool2')
    self.pool3 = model
    print >>sys.stderr, "pool3", util.shape_and_product_of(model)

    return model

  def input_state_network(self, input_state, opts):
    # TODO: use in lrpg and ddpg too
    if opts.use_raw_pixels:
      input_state = self.simple_conv_net_on(input_state, opts)
    flattened_input_state = slim.flatten(input_state, scope='flat')
    return self.hidden_layers_starting_at(flattened_input_state, opts.hidden_layers, opts)

  def render_convnet_activations(self, activations, filename_base):
    _batch, height, width, num_filters = activations.shape
    for f_idx in range(num_filters):
      single_channel = activations[0,:,:,f_idx]
      single_channel /= np.max(single_channel)
      img = np.empty((height, width, 3))
      img[:,:,0] = single_channel
      img[:,:,1] = single_channel
      img[:,:,2] = single_channel
      util.write_img_to_png_file(img, "%s_f%02d.png" % (filename_base, f_idx))

  def render_all_convnet_activations(self, step, input_state_placeholder, state):
    activations = tf.get_default_session().run([self.pool1, self.pool2, self.pool3],
                                               feed_dict={input_state_placeholder: [state],
                                                          IS_TRAINING: False})
    filename_base = "/tmp/activation_s%03d" % step
    self.render_convnet_activations(activations[0], filename_base + "_p0")
    self.render_convnet_activations(activations[1], filename_base + "_p1")
    self.render_convnet_activations(activations[2], filename_base + "_p2")
