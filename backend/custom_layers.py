import numpy as np
import tensorflow as tf


class Normalization(tf.keras.layers.Layer):
	"""
	My own Normalization function with more consistent
	functionality than BatchNormalization. Or at the very least functionality that I can understand.
	"""
	def __init__(self, init_mean = 0.0, init_var = 1.0, eps = 0.001, *args, **kwargs):
		super(Normalization, self).__init__(*args, **kwargs)
		self.eps = eps
		self.init_mean = init_mean
		self.init_var = init_var * np.ones_like(init_mean)

	def build(self, input_shape):
		init_mean_tensor = np.zeros(input_shape[1:])
		init_var_tensor = np.ones(input_shape[1:]) * self.init_var

		if type(self.init_mean) == float or self.init_mean.shape == ():
			init_mean_tensor[0] = self.init_mean
		else:
			init_mean_tensor = self.init_mean

		self.mean = tf.Variable(initial_value = init_mean_tensor,
								name = "mean",
		                        dtype = 'float32',
								trainable = self.trainable)

		self.var = tf.Variable(initial_value = init_var_tensor,
							   name = "var",
							   dtype = 'float32',
							   trainable = self.trainable)

	def call(self, x, **kwargs):
		return (x - self.mean) / (self.eps + tf.square(self.var))


class SimpleTradeBlock(tf.keras.layers.Layer):
	"""
	A non-recurrent version of the trade block, this rough
	architecture is taken from Teichmann et al paper Deep Hedging.
	"""
	def __init__(self, output_dim, initial_price = 4.6, activation = 'tanh', extra_nodes = 8, **kwargs):
		self.output_dim = output_dim
		self.extra_nodes = extra_nodes
		super(SimpleTradeBlock, self).__init__(**kwargs)

		self.norm1 = Normalization(initial_price, trainable = True, init_var = 0.4)

		self.dense1 = tf.keras.layers.Dense(self.output_dim + self.extra_nodes,
											activation = activation,
											kernel_initializer =
											tf.keras.initializers.RandomUniform(minval = -0.1, maxval = 0.1),
		                                    name = "dense1_")

		self.dense2 = tf.keras.layers.Dense(self.output_dim + self.extra_nodes,
											activation = activation,
											kernel_initializer =
											tf.keras.initializers.RandomUniform(minval = -0.1, maxval = 0.1),
		                                    name = "dense1_")

		self.norm2 = Normalization(trainable = True)

		# self.dense3 = tf.keras.layers.Dense(self.output_dim + self.extra_nodes,
		# 									activation = activation,
		# 									kernel_initializer =
		# 									tf.keras.initializers.RandomUniform(minval = -0.05, maxval = 0.2),
		# 									name = "dense1_")

		self.final = tf.keras.layers.Dense(self.output_dim, activation = 'linear',
		                                   name = "final_")

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.norm1.build(input_shape)
		self.dense1.build(input_shape)
		self.dense2.build((None, self.output_dim + self.extra_nodes))
		self.norm2.build((None, self.output_dim + self.extra_nodes))
		# self.dense3.build((None, self.output_dim + self.extra_nodes))
		self.final.build((None, self.output_dim + self.extra_nodes))

		super(SimpleTradeBlock, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		x = self.norm1(x)
		x = self.dense1(x)
		x = self.dense2(x)
		x = self.norm2(x)
		x = self.final(x)
		return x

	def compute_output_shape(self, input_shape):
		return input_shape[0], self.output_dim


class TradeBlock(tf.keras.layers.Layer):
	"""
	A recurrent version of the trade block, this rough
	architecture is taken from Teichmann et al paper Deep Hedging.
	"""
	def __init__(self, output_dim, initial_price = 4.6, activation = 'tanh', extra_nodes = 10, **kwargs):
		self.output_dim = output_dim
		self.extra_nodes = extra_nodes
		super(TradeBlock, self).__init__(**kwargs)

		self.norm1 = Normalization(initial_price, trainable = True, init_var = 0.4)

		self.dense1 = tf.keras.layers.Dense(self.output_dim + self.extra_nodes,
		                                    activation = activation,
											kernel_initializer =
											tf.keras.initializers.RandomUniform(minval = -0.1, maxval = 0.2),
		                                    name = "dense1_")

		self.dense2 = tf.keras.layers.Dense(self.output_dim + self.extra_nodes,
		                                    activation = activation,
											kernel_initializer =
											tf.keras.initializers.RandomUniform(minval = -0.1, maxval = 0.2),
		                                    name = "dense2_")

		self.norm2 = Normalization(trainable = True)

		self.final = tf.keras.layers.Dense(self.output_dim, activation = 'linear',
		                                   name = "final_")

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.norm1.build(input_shape)
		self.dense1.build(input_shape)
		self.dense2.build((None, self.output_dim + self.extra_nodes))
		self.norm2.build((None, self.output_dim + self.extra_nodes))
		self.final.build((None, self.output_dim + self.extra_nodes))

		super(TradeBlock, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		y = self.norm1(tf.concat(x, -1))
		y = self.dense1(y)
		y = self.dense2(y)
		y = self.norm2(y)
		y = self.final(y)
		return y

	def compute_output_shape(self, input_shape):
		return input_shape[0], self.output_dim


class OtherTradeBlock(tf.keras.layers.Layer):
	def __init__(self, output_dim, initial_price = 4.6, hidden_size = None, activation = 'tanh', extra_nodes = 8, **kwargs):
		if hidden_size is None:
			self.hidden_size = output_dim
		else:
			self.hidden_size = hidden_size
		self.output_dim = output_dim
		self.extra_nodes = extra_nodes
		super(OtherTradeBlock, self).__init__(**kwargs)

		self.norm1 = Normalization(initial_price, trainable = True, init_var = 0.4)

		self.dense1 = tf.keras.layers.Dense(self.output_dim + self.hidden_size + self.extra_nodes,
											activation = activation,
											kernel_initializer =
											tf.keras.initializers.RandomUniform(minval = -0.1, maxval = 0.2),
											name = "dense1_")

		self.dense2 = tf.keras.layers.Dense(self.output_dim + self.hidden_size + self.extra_nodes,
											activation = activation,
											kernel_initializer =
											tf.keras.initializers.RandomUniform(minval = -0.1, maxval = 0.2),
											name = "dense1_")

		self.norm2 = Normalization()

		self.final_y = tf.keras.layers.Dense(self.output_dim, activation = 'linear', name = "finaly_")
		self.final_h = tf.keras.layers.Dense(self.hidden_size, activation = 'tanh', name = "finalh_")

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.norm1.build(input_shape)
		self.dense1.build(input_shape)
		self.dense2.build((None, self.output_dim + self.hidden_size + self.extra_nodes))
		self.norm2.build((None, self.output_dim + self.hidden_size + self.extra_nodes))
		self.final_y.build((None, self.output_dim + self.hidden_size + self.extra_nodes))
		self.final_h.build((None, self.output_dim + self.hidden_size + self.extra_nodes))

		super(OtherTradeBlock, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		y = self.norm1(tf.concat(x, -1))
		y = self.dense1(y)
		y = self.dense2(y)
		y = self.norm2(y)
		return self.final_y(y), self.final_h(y)


class StopBlock(tf.keras.layers.Layer):
	"""
	A recurrent version of the trade block, this rough
	architecture is taken from Teichmann et al paper Deep Hedging.
	"""
	def __init__(self, output_dim, initial_price = 100., activation = 'tanh', extra_nodes = 4, **kwargs):
		self.output_dim = output_dim
		self.extra_nodes = extra_nodes
		super(StopBlock, self).__init__(**kwargs)

		self.norm1 = Normalization(initial_price, trainable = True)

		self.dense1 = tf.keras.layers.Dense(self.output_dim + self.extra_nodes,
		                                    activation = activation,
											kernel_initializer =
											tf.keras.initializers.RandomUniform(minval = -0.1, maxval = 0.2),
		                                    name = "dense1_")

		self.dense2 = tf.keras.layers.Dense(self.output_dim + self.extra_nodes,
											activation = activation,
											kernel_initializer =
											tf.keras.initializers.RandomUniform(minval = -0.1, maxval = 0.2),
											name = "dense2_")

		self.norm2 = Normalization(trainable = True)

		self.final = tf.keras.layers.Dense(self.output_dim, activation = 'sigmoid', name = "final_")

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.norm1.build(input_shape)
		self.dense1.build(input_shape)
		self.norm2.build((None, self.output_dim + self.extra_nodes))
		self.dense2.build((None, self.output_dim + self.extra_nodes))
		self.final.build((None, self.output_dim + self.extra_nodes))

		super(StopBlock, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		# x = tf.math.log(x)
		y = self.norm1(x)
		y = self.dense1(y)
		y = self.norm2(y)
		y = self.dense2(y)
		y = self.final(y)
		return y

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)
