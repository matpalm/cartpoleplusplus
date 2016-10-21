import tensorflow as tf
import tensorflow.contrib.slim as slim

# TODO: move opts only used in this case to an add_opts method

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

  def hidden_layers_starting_at(self, layer, config):
    layer_sizes = map(int, config.split(","))
    assert len(layer_sizes) > 0
    for i, size in enumerate(layer_sizes):
      layer = slim.fully_connected(scope="h%d" % i,
                                  inputs=layer,
                                  num_outputs=size,
                                  weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                  activation_fn=tf.nn.relu)
    return layer

  def simple_conv_net_on(self, input_layer, opts):
    if opts.use_batch_norm:
      normalizer_fn = slim.batch_norm
      normalizer_params = { 'is_training': IS_TRAINING }
    else:
      normalizer_fn = None
      normalizer_params = None

    # state is (batch, height, width, rgb, camera_idx, repeat)
    # rollup rgb, camera_idx and repeat into num_channels
    height, width = map(int, input_layer.get_shape()[1:3])
    num_channels = input_layer.get_shape()[3:].num_elements()
    input_layer = tf.reshape(input_layer, [-1, height, width, num_channels])

    model = slim.conv2d(input_layer, num_outputs=30, kernel_size=[5, 5],
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        scope='conv1')
    model = slim.max_pool2d(model, kernel_size=[2, 2], scope='pool1')
    model = slim.conv2d(model, num_outputs=20, kernel_size=[5, 5], 
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        scope='conv2')
    model = slim.max_pool2d(model, kernel_size=[2, 2], scope='pool2')
    return slim.flatten(model, scope='flat')

  def input_state_network(self, input_state, opts):
    # TODO: use in lrpg and ddpg too
    if opts.use_raw_pixels:
      conv_net = self.simple_conv_net_on(input_state, opts)
      hidden1 = slim.fully_connected(conv_net, 400, scope='hidden1')
      return slim.fully_connected(hidden1, 50, scope='hidden2')
    else:
      flat_input_state = slim.flatten(input_state, scope='flat')
      return self.hidden_layers_starting_at(flat_input_state, opts.hidden_layers)
