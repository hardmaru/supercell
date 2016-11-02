'''

supercell

https://github.com/hardmaru/supercell/

inspired by http://supercell.jp/

'''

import tensorflow as tf
import numpy as np

# Orthogonal Initializer from
# https://github.com/OlavHN/bnlstm
def orthogonal(shape):
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  return q.reshape(shape)
def orthogonal_initializer(scale=1.0):
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    return tf.constant(orthogonal(shape) * scale, dtype)
  return _initializer
def lstm_identity_initializer(scale=1.0):
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    size = shape[0]
    t = np.zeros(shape)
    t[:, size:size * 2] = np.identity(size)*scale # gate (j) is identity
    t[:, :size] = orthogonal([size, size])
    t[:, size * 2:size * 3] = orthogonal([size, size])
    t[:, size * 3:] = orthogonal([size, size])
    return tf.constant(t, dtype)
  return _initializer
def lstm_ortho_initializer(scale=1.0):
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    size_x = shape[0]
    size_h = shape[1]/4 # assumes lstm.
    t = np.zeros(shape)
    t[:, :size_h] = orthogonal([size_x, size_h])*scale
    t[:, size_h:size_h*2] = orthogonal([size_x, size_h])*scale
    t[:, size_h*2:size_h*3] = orthogonal([size_x, size_h])*scale
    t[:, size_h*3:] = orthogonal([size_x, size_h])*scale
    return tf.constant(t, dtype)
  return _initializer

class LSTMCell(tf.nn.rnn_cell.RNNCell):
  '''
  Vanilla LSTM with ortho initializer,
  and also recurrent dropout without memory loss
  (https://arxiv.org/abs/1603.05118)
  derived from
  https://github.com/OlavHN/bnlstm
  https://github.com/LeavesBreathe/tensorflow_with_latest_papers
  '''
  def __init__(self, num_units, forget_bias=1.0,
    use_recurrent_dropout=False, dropout_keep_prob=0.9):
    self.num_units = num_units
    self.forget_bias=forget_bias
    self.use_recurrent_dropout=use_recurrent_dropout
    self.dropout_keep_prob=dropout_keep_prob

  @property
  def state_size(self):
    return 2 * self.num_units

  @property
  def output_size(self):
    return self.num_units

  def __call__(self, x, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      c, h = tf.split(1, 2, state)

      h_size = self.num_units
      x_size = x.get_shape().as_list()[1]

      #w_init=lstm_ortho_initializer(1.0)
      #w_init=orthogonal_initializer(1.0)
      #w_init=tf.constant_initializer(0.0)
      #w_init=tf.random_normal_initializer(stddev=0.01)
      w_init=None # uniform

      h_init=lstm_ortho_initializer(1.0)
      #h_init=tf.constant_initializer(0.0)
      #h_init=tf.random_normal_initializer(stddev=0.01)
      #h_init=None # uniform

      # Keep W_xh and W_hh separate here as well to use different initialization methods
      W_xh = tf.get_variable('W_xh',
        [x_size, 4 * self.num_units], initializer=w_init)
      W_hh = tf.get_variable('W_hh',
        [self.num_units, 4 * self.num_units], initializer=h_init)
      bias = tf.get_variable('bias',
        [4 * self.num_units], initializer=tf.constant_initializer(0.0))

      concat = tf.concat(1, [x, h])
      W_full = tf.concat(0, [W_xh, W_hh])
      hidden = tf.matmul(concat, W_full) + bias

      i, j, f, o = tf.split(1, 4, hidden)

      if self.use_recurrent_dropout:
        g = tf.nn.dropout(tf.tanh(j), self.dropout_keep_prob)
      else:
        g = tf.tanh(j) 

      new_c = c*tf.sigmoid(f+self.forget_bias) + tf.sigmoid(i)*g
      new_h = tf.tanh(new_c) * tf.sigmoid(o)

      return new_h, tf.concat(1, [new_c, new_h]) # fuk tuples.

# support functions for layer norm
def moments_for_layer_norm(x, axes=1, name=None):
  #output for mean and variance should be [batch_size]
  # from https://github.com/LeavesBreathe/tensorflow_with_latest_papers
  epsilon = 1e-3 # found this works best.
  if not isinstance(axes, list): axes = list(axes)
  with tf.op_scope([x, axes], name, "moments"):
    mean = tf.reduce_mean(x, axes, keep_dims=True)
    variance = tf.sqrt(tf.reduce_mean(tf.square(x-mean), axes, keep_dims=True)+epsilon)
    return mean, variance

def layer_norm(input_tensor, scope="layer_norm", alpha_start=1.0, bias_start=0.0, reuse=False):
  # derived from:
  # https://github.com/LeavesBreathe/tensorflow_with_latest_papers, but simplified.
  with tf.variable_scope(scope):
    if reuse == True:
      tf.get_variable_scope().reuse_variables()
    input_tensor_shape_list = input_tensor.get_shape().as_list()
    num_units = input_tensor_shape_list[1]

    alpha = tf.get_variable('layer_norm_alpha', [num_units],
      initializer=tf.constant_initializer(alpha_start))
    bias = tf.get_variable('layer_norm_bias', [num_units],
      initializer=tf.constant_initializer(bias_start))

    mean, variance = moments_for_layer_norm(input_tensor,
      axes=[1], name = "moments_"+scope)
    output = (alpha * (input_tensor-mean))/(variance)+bias

  return output

def super_linear(x, output_size, scope=None, reuse=False,
  init_w="ortho", weight_start=0.0, use_bias=True, bias_start=0.0):
  # support function doing linear operation.  uses ortho initializer defined earlier.
  shape = x.get_shape().as_list()
  with tf.variable_scope(scope or "linear"):
    if reuse == True:
      tf.get_variable_scope().reuse_variables()

    w_init = None # uniform
    x_size = shape[1]
    h_size = output_size
    if init_w == "zeros":
      w_init=tf.constant_initializer(0.0)
    elif init_w == "constant":
      w_init=tf.constant_initializer(weight_start)
    elif init_w == "gaussian":
      w_init=tf.random_normal_initializer(stddev=weight_start)
    elif init_w == "ortho":
      w_init=lstm_ortho_initializer(1.0)

    w = tf.get_variable("super_linear_w",
      [shape[1], output_size], tf.float32, initializer=w_init)
    if use_bias:
      b = tf.get_variable("super_linear_b", [output_size], tf.float32,
        initializer=tf.constant_initializer(bias_start))
      return tf.matmul(x, w) + b
    return tf.matmul(x, w)

class LayerNormLSTMCell(tf.nn.rnn_cell.RNNCell):
  """
  Layer-Norm, with Ortho Initialization and
  Recurrent Dropout without Memory Loss.

  https://arxiv.org/abs/1607.06450 - Layer Norm

  https://arxiv.org/abs/1603.05118 - Recurrent Dropout without Memory Loss

  derived from
  https://github.com/OlavHN/bnlstm
  https://github.com/LeavesBreathe/tensorflow_with_latest_papers

  """

  def __init__(self, num_units, forget_bias=1.0,
    use_recurrent_dropout=False, dropout_keep_prob=0.90):
    """Initialize the Layer Norm LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (default 1.0).
      use_recurrent_dropout: float, Whether to use Recurrent Dropout (default False)
      dropout_keep_prob: float, dropout keep probability (default 0.90)
    """
    self.num_units = num_units
    self.forget_bias = forget_bias
    self.use_recurrent_dropout = use_recurrent_dropout
    self.dropout_keep_prob = dropout_keep_prob

  @property
  def input_size(self):
    return self.num_units

  @property
  def output_size(self):
    return self.num_units

  @property
  def state_size(self):
    return 2 * self.num_units

  def __call__(self, x, state, timestep = 0, scope=None):
    with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
      h, c = tf.split(1, 2, state)

      h_size = self.num_units
      x_size = x.get_shape().as_list()[1]

      #w_init=lstm_ortho_initializer(1.0)
      #w_init=orthogonal_initializer(1.0)
      #w_init=tf.constant_initializer(0.0)
      #w_init=tf.random_normal_initializer(stddev=0.01)
      w_init=None # uniform

      h_init=lstm_ortho_initializer(1.0)
      #h_init=tf.constant_initializer(0.0)
      #h_init=tf.random_normal_initializer(stddev=0.01)
      #h_init=None # uniform

      W_xh = tf.get_variable('W_xh',
        [x_size, 4 * self.num_units], initializer=w_init)
      W_hh = tf.get_variable('W_hh',
        [self.num_units, 4 * self.num_units], initializer=h_init)
      # no bias, since there's a bias thing inside layer norm
      # and we don't wanna double task variables.

      concat = tf.concat(1, [x, h]) # concat for speed.
      W_full = tf.concat(0, [W_xh, W_hh])
      concat = tf.matmul(concat, W_full) #+ bias # live life without garbage.

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = tf.split(1, 4, concat)

      i = layer_norm(i, 'ln_i')
      j = layer_norm(j, 'ln_j')
      f = layer_norm(f, 'ln_f')
      o = layer_norm(o, 'ln_o')

      if self.use_recurrent_dropout:
        g = tf.nn.dropout(tf.tanh(j), self.dropout_keep_prob)
      else:
        g = tf.tanh(j) 

      new_c = c*tf.sigmoid(f+self.forget_bias) + tf.sigmoid(i)*g
      new_h = tf.tanh(layer_norm(new_c, 'ln_c')) * tf.sigmoid(o)
    
    return new_h, tf.concat(1, [new_h, new_c])

class HyperLSTMCell(tf.nn.rnn_cell.RNNCell):
  '''
  HyperLSTM, with Ortho Initialization,
  Layer Norm and Recurrent Dropout without Memory Loss.
  
  https://arxiv.org/abs/1609.09106
  http://blog.otoro.net/2016/09/28/hyper-networks/
  '''

  def __init__(self, num_units, forget_bias=1.0,
    use_recurrent_dropout=False, dropout_keep_prob=0.90, use_layer_norm=True,
    hyper_num_units=64, hyper_embedding_size=4,
    hyper_use_recurrent_dropout=False):
    """Initialize the Layer Norm HyperLSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (default 1.0).
      use_recurrent_dropout: float, Whether to use Recurrent Dropout (default False)
      dropout_keep_prob: float, dropout keep probability (default 0.90)
      use_layer_norm: boolean. (default True)
        Controls whether we use LayerNorm layers in main LSTM and HyperLSTM cell.
      hyper_num_units: int, number of units in HyperLSTM cell.
        (default is 128, recommend experimenting with 256 for larger tasks)
      hyper_embedding_size: int, size of signals emitted from HyperLSTM cell.
        (default is 4, recommend trying larger values but larger is not always better)
      hyper_use_recurrent_dropout: boolean. (default False)
        Controls whether HyperLSTM cell also uses recurrent dropout. (Not in Paper.)
        Recommend turning this on only if hyper_num_units becomes very large (>= 512)
    """
    self.num_units = num_units
    self.forget_bias = forget_bias
    self.use_recurrent_dropout = use_recurrent_dropout
    self.dropout_keep_prob = dropout_keep_prob
    self.use_layer_norm = use_layer_norm
    self.hyper_num_units = hyper_num_units
    self.hyper_embedding_size = hyper_embedding_size
    self.hyper_use_recurrent_dropout = hyper_use_recurrent_dropout

    self.total_num_units = self.num_units + self.hyper_num_units

    if self.use_layer_norm:
      cell_fn = LayerNormLSTMCell
    else:
      cell_fn = LSTMCell
    self.hyper_cell = cell_fn(hyper_num_units,
      use_recurrent_dropout=hyper_use_recurrent_dropout,
      dropout_keep_prob=dropout_keep_prob)

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self.num_units

  @property
  def state_size(self):
    return 2*self.total_num_units

  def layer_norm(self, layer, scope="layer_norm"):
    # wrapper for layer_norm
    if self.use_layer_norm:
      return layer_norm(layer, scope)
    else:
      return layer

  def hyper_norm(self, layer, scope="hyper", use_bias=True):
    num_units = self.num_units
    embedding_size = self.hyper_embedding_size
    # recurrent batch norm init trick (https://arxiv.org/abs/1603.09025).
    init_gamma = 0.10 # cooijmans' da man.
    with tf.variable_scope(scope):
      zw = super_linear(self.hyper_output, embedding_size, init_w="constant",
        weight_start=0.00, use_bias=True, bias_start=1.0, scope="zw")
      alpha = super_linear(zw, num_units, init_w="constant",
        weight_start=init_gamma / embedding_size, use_bias=False, scope="alpha")
      result = tf.mul(alpha, layer)
      if use_bias:
        zb = super_linear(self.hyper_output, embedding_size, init_w="gaussian",
          weight_start=0.01, use_bias=False, bias_start=0.0, scope="zb")
        beta = super_linear(zb, num_units, init_w="constant",
          weight_start=0.00, use_bias=False, scope="beta")
        result = result + beta
    return result

  def __call__(self, x, state, timestep = 0, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      total_h, total_c = tf.split(1, 2, state)
      h = total_h[:, 0:self.num_units]
      c = total_c[:, 0:self.num_units]
      self.hyper_state = tf.concat(1, [total_h[:,self.num_units:], total_c[:,self.num_units:]])

      x_size = x.get_shape().as_list()[1]
      self._input_size = x_size

      #w_init=lstm_ortho_initializer(1.0)
      #w_init=orthogonal_initializer(1.0)
      #w_init=tf.constant_initializer(0.0)
      #w_init=tf.random_normal_initializer(stddev=0.01)
      w_init=None # uniform

      h_init=lstm_ortho_initializer(1.0)
      #h_init=lstm_identity_initializer(1.0)
      #h_init=tf.constant_initializer(0.0)
      #h_init=tf.random_normal_initializer(stddev=0.01)
      #h_init=None # uniform

      W_xh = tf.get_variable('W_xh',
        [x_size, 4*self.num_units], initializer=w_init)
      W_hh = tf.get_variable('W_hh',
        [self.num_units, 4*self.num_units], initializer=h_init)
      bias = tf.get_variable('bias',
        [4*self.num_units], initializer=tf.constant_initializer(0.0))

      # concatenate the input and hidden states for hyperlstm input
      hyper_input = tf.concat(1, [x, h])
      hyper_output, hyper_new_state = self.hyper_cell(hyper_input, self.hyper_state)
      self.hyper_output = hyper_output
      self.hyper_state = hyper_new_state

      xh = tf.matmul(x, W_xh)
      hh = tf.matmul(h, W_hh)

      # split Wxh contributions
      ix, jx, fx, ox = tf.split(1, 4, xh)
      ix = self.hyper_norm(ix, 'hyper_ix', use_bias=False)
      jx = self.hyper_norm(jx, 'hyper_jx', use_bias=False)
      fx = self.hyper_norm(fx, 'hyper_fx', use_bias=False)
      ox = self.hyper_norm(ox, 'hyper_ox', use_bias=False)

      # split Whh contributions
      ih, jh, fh, oh = tf.split(1, 4, hh)
      ih = self.hyper_norm(ih, 'hyper_ih', use_bias=True)
      jh = self.hyper_norm(jh, 'hyper_jh', use_bias=True)
      fh = self.hyper_norm(fh, 'hyper_fh', use_bias=True)
      oh = self.hyper_norm(oh, 'hyper_oh', use_bias=True)

      # split bias
      ib, jb, fb, ob = tf.split(0, 4, bias) # bias is to be broadcasted.

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i = ix + ih + ib
      j = jx + jh + jb
      f = fx + fh + fb
      o = ox + oh + ob

      i = self.layer_norm(i, 'ln_i')
      j = self.layer_norm(j, 'ln_j')
      f = self.layer_norm(f, 'ln_f')
      o = self.layer_norm(o, 'ln_o')

      if self.use_recurrent_dropout:
        g = tf.nn.dropout(tf.tanh(j), self.dropout_keep_prob)
      else:
        g = tf.tanh(j) 

      new_c = c*tf.sigmoid(f+self.forget_bias) + tf.sigmoid(i)*g
      new_h = tf.tanh(self.layer_norm(new_c, 'ln_c')) * tf.sigmoid(o)
    
      hyper_h, hyper_c = tf.split(1, 2, hyper_new_state)
      new_total_h = tf.concat(1, [new_h, hyper_h])
      new_total_c = tf.concat(1, [new_c, hyper_c])
      new_total_state = tf.concat(1, [new_total_h, new_total_c])
    return new_h, new_total_state

class LayerNormRNNCell(tf.nn.rnn_cell.RNNCell):
  """
  Layer-Norm, with Ortho Initialization, vanilla RNN cell.
  """

  def __init__(self, num_units, activation=tf.tanh):
    self.num_units = num_units
    self.activation = activation

  @property
  def input_size(self):
    return self.num_units

  @property
  def output_size(self):
    return self.num_units

  @property
  def state_size(self):
    return self.num_units

  def __call__(self, x, state, timestep = 0, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      h = state

      h_size = self.num_units
      x_size = x.get_shape().as_list()[1]

      #w_init=orthogonal_initializer(1.0)
      #w_init=tf.constant_initializer(0.0)
      #w_init=tf.random_normal_initializer(stddev=0.01)
      w_init=None # uniform

      h_init=orthogonal_initializer(1.0)
      #h_init=tf.constant_initializer(0.0)
      #h_init=tf.random_normal_initializer(stddev=0.01)
      #h_init=None # uniform

      W_xh = tf.get_variable('W_xh',
        [x_size, self.num_units], initializer=w_init)
      W_hh = tf.get_variable('W_hh',
        [self.num_units, self.num_units], initializer=h_init)
      # no bias, since there's a bias thing inside layer norm
      # and we don't wanna double task variables.

      concat = tf.concat(1, [x, h]) # concat for speed.
      W_full = tf.concat(0, [W_xh, W_hh])
      concat = tf.matmul(concat, W_full)

      new_h = self.activation(layer_norm(concat, 'ln_h'))
    
    return new_h, new_h

class FastRNNCell(tf.nn.rnn_cell.RNNCell):
  """
  Layer-Norm, with Ortho Initialization, Fast RNN cell.
  Alpha version, probably bugs, pls test and see if fixin' needed.
  
  Based on:
  https://arxiv.org/abs/1610.06258
  Assumes S = 1, that was my understanding. (pls write more clearly ...)
  As usual, calling FastRNNCell will return two things: output, state
  
  output is the current hidden state of the RNN, dimensions batch_size x num_units
  
  state will be a packed tensor containing the previous max_history hidden states
  the previous max_history will be scaled gradually by lambda factor.
  
  state's dimensions will be batch_size x [num_units * max_history]
  
  initially, all the previous history will be zeroed out.
  """

  def __init__(self, num_units, eta_factor=0.5, lambda_factor=0.90,
               activation=tf.nn.relu, inner_steps=1, max_history=20):
    """Initialize the FastRNN cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      eta_factor: float, default 0.5
      lambda_factor: float, default 0.90 (try to choose a better greek character next time that is not a python keyword...)
      activation: default tf.nn.relu.  Feel free to try tf.tanh
      inner_steps: default 1, number of inner steps of fast rnn
      max_history: int, default 20. Only considers up to previous [20] states
    """
    self.num_units = num_units
    self.eta = eta_factor
    self.lam = np.sqrt(lambda_factor) # sqrt to be consistent w/ the paper's def
    self.activation = activation
    self.inner_steps = inner_steps
    self.max_history=max_history

  @property
  def input_size(self):
    return self.num_units

  @property
  def output_size(self):
    return self.num_units

  @property
  def state_size(self):
    return self.max_history*self.num_units

  def __call__(self, x, raw_state, timestep = 0, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      batch_size = x.get_shape().as_list()[0]
      max_history = self.max_history
      h_size = self.num_units
      x_size = x.get_shape().as_list()[1]

      state = tf.reshape(raw_state, [batch_size, max_history, h_size])
      h = state[:, 0, :] # get most recent hidden state

      state = self.lam*state # decrease everything by the lambda factor

      #w_init=orthogonal_initializer(1.0)
      #w_init=tf.constant_initializer(0.0)
      #w_init=tf.random_normal_initializer(stddev=0.01)
      w_init=None # uniform

      h_init=orthogonal_initializer(1.0)
      #h_init=tf.constant_initializer(0.0)
      #h_init=tf.random_normal_initializer(stddev=0.01)
      #h_init=None # uniform

      W_xh = tf.get_variable('W_xh',
        [x_size, self.num_units], initializer=w_init)
      W_hh = tf.get_variable('W_hh',
        [self.num_units, self.num_units], initializer=h_init)
      # no bias, since there's a bias thing inside layer norm
      # and we don't wanna double task variables.

      concat = tf.concat(1, [x, h]) # concat for speed.
      W_full = tf.concat(0, [W_xh, W_hh])

      # preliminary vector (eq 2 has no non-linearity, I go with that)
      h0 = tf.matmul(concat, W_full)
      
      h0 = self.activation(h0)
      
      h0 = tf.reshape(h0, [batch_size, 1, h_size])
      
      # eq4, already scaled by sqrt(lambda_factor) in both lines
      h1 = tf.batch_matmul(h0, tf.transpose(state, perm=[0, 2, 1]))
      h1 = tf.batch_matmul(h1, state)

      h0 = tf.reshape(h0, [batch_size, h_size])
      new_h = tf.reshape(h1, [batch_size, h_size])

      # combination of eq2/eq5 and eq4:
      for i in range(self.inner_steps):
        new_h = h0+self.eta*new_h
        new_h = self.activation(layer_norm(new_h, 'ln_h', reuse=(True if i > 0 else False)))

      h_insert = tf.reshape(new_h, [batch_size, 1, h_size])
      state = tf.concat(1, [h_insert, state])  # put in most recent state in the front
      state = state[:, 0:self.max_history, :]  # kick out the last hidden state

      state = tf.reshape(state, [batch_size, max_history*h_size])
    
    return new_h, state


