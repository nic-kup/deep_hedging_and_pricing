import os
from backend.pricer import *
from backend.custom_layers import *
import numpy as np

tf.get_logger().setLevel('ERROR')

# Avoid Error with calculating cur_hedge on the fly
tf.config.experimental_run_functions_eagerly(True)


# TODO: write the train loop into a function


# Profits and Losses at terminal time
def PLT(Z, p_0, delta, S, tc = 0.):
	# TODO: Broken
	"""
	Loss function for pricing
	:param Z:       risky position
	:param S:       Stock price
	:param p_0:     capital
	:param delta:   trading strategy
	:param tc:      transaction cost
	:return:        loss
	"""
	return -Z + p_0 + tf.math.reduce_sum(tf_stoch_int(delta, S)[:, -1, :]) - tc


# Alternative Profit and Losses calculation
def PLT_alt(Z, p_0, delS, tc = 0.):
	"""
	Alternative Loss function which takes stochastic integral as input
	:param Z:       Risk
	:param p_0:     Price paid
	:param delS:    integral
	:param tc:      transaction costs
	:return:        Profits and losses
	"""
	return -Z + p_0 + delS - tc


# Complicated Objective Function from Deep Hedging Paper
def objective(Z, p_0, delta, S, tc = 0, lambda_0 = 0, F = None, alpha_bar = None):
	if alpha_bar is None:
		return -PLT(Z, p_0, delta, S, tc)
	else:
		return -PLT(Z, p_0, delta, S, tc) * tf.exp(F(S)) - alpha_bar(F) - lambda_0 * (tf.exp(F(S)) - 1)


def objective_alt(Z, p_0, delS, tc = 0.):
	return -PLT_alt(Z, p_0, delS, tc = 0.)

# TODO: create a super class for all the Learners, current solution is not so nice!


class SimpleLearner(tf.keras.Model):
	def __init__(self, d, N, option, cost_fct = None,
				 initial_price = 4.6,
				 risk_measure = lambda x: tf.reduce_mean(-x, axis = 0),
				 pricebs = None,
				 input_func = tf.identity,
				 activation = 'tanh',
				 *args, **kwargs):
		"""
		:param d: Market of size d
		"""
		super().__init__(*args, **kwargs)

		# number of stocks to trade
		self.d = d
		# Number of trade days + 1
		self.N = N
		# Risk aversion of Client
		self.risk_measure = risk_measure
		# Function applied to stock before input to NN
		self.input_func = input_func

		# option to price
		self.option_args = None
		self.option = option

		self.string_name = "Simple"

		# Transaction costs
		if cost_fct is None:
			def cost_fct(incr, price):
				return tf.Variable([[0.0]], trainable = False, dtype = 'float32')

			self.cost_fct = cost_fct
		else:
			self.cost_fct = cost_fct

		# Price of Option (if None, will be learned)
		if pricebs is None:
			self.pricebs = tf.Variable(0.0, trainable = True,
									   name = 'price', dtype = 'float32')
		else:
			self.pricebs = pricebs

		self.trade_blocks = [SimpleTradeBlock(self.d,
										initial_price = initial_price,
										activation = activation,
										name = "tb_" + str(_)) for _ in range(N - 1)]

	def build(self, input_shape = None):
		for i in range(self.N - 1):
			self.trade_blocks[i].build((None, self.d))

	def call(self, inputs, training=None, mask=None):
		y = tf.zeros_like(inputs[:, 0, :], dtype = 'float32')
		y_old = tf.zeros_like(inputs[:, 0, :], dtype = 'float32')
		cur_hedge = tf.zeros_like(inputs[:, 0, 0], dtype = 'float32')
		cur_costs = tf.zeros_like(inputs[:, 0, 0], dtype = 'float32')
		# Input given as S
		for i in range(self.N - 1):
			y = self.trade_blocks[i](self.input_func(inputs[:, i, :]))

			cur_hedge = cur_hedge + tf.reduce_sum(y * (inputs[:, i + 1, :] - inputs[:, i, :]), 1)
			cur_costs = cur_costs + tf.reduce_sum(self.cost_fct(y - y_old, inputs[:, i, :]), 1)
			y_old = y

		return self.risk_measure(self.pricebs + cur_hedge
								 - self.option(inputs) - cur_costs)

	def get_pl(self, inputs, training=None, mask=None):
		y = tf.zeros_like(inputs[:, 0, :], dtype = 'float32')
		y_old = tf.zeros_like(inputs[:, 0, :], dtype = 'float32')
		cur_hedge = tf.zeros_like(inputs[:, 0, 0], dtype = 'float32')
		cur_costs = tf.zeros_like(inputs[:, 0, 0], dtype = 'float32')
		# Input given as S
		for i in range(self.N - 1):
			y = self.trade_blocks[i](self.input_func(inputs[:, i, :]))

			cur_hedge = cur_hedge + tf.reduce_sum(y * (inputs[:, i + 1, :] - inputs[:, i, :]), 1)
			cur_costs = cur_costs + tf.reduce_sum(self.cost_fct(y - y_old, inputs[:, i, :]), 1)
			y_old = y

		return self.pricebs + cur_hedge - self.option(inputs) - cur_costs

	def call_k(self, x, k):
		y = tf.zeros_like(x[:, 0, :])
		for i in range(k):
			y = self.trade_blocks[i](self.input_func(x[:, i, :]))

		return y

	def call_only_k(self, x, k):
		assert len(x.shape) == 2
		y = tf.zeros_like(x[:, :])

		y = self.trade_blocks[k - 1](self.input_func(x[:, :]))

		return y

	def get_hedge(self, x):
		if type(x) is np.ndarray:
			delta = np.zeros_like(x)
		else:
			delta = np.zeros_like(x.numpy())
		y = tf.zeros_like(x[:, 0, :])
		# Input given as S
		for i in range(self.N - 1):
			y = self.trade_blocks[i](self.input_func(x[:, i, :]))

			delta[:, i, :] = y
		delta[:, self.N - 1, :] = delta[:, self.N - 2, :]
		return delta

	def get_price(self):
		return self.pricebs


class Learner(tf.keras.Model):
	def __init__(self, d, N, option, cost_fct = None,
				 initial_price = 4.6,
				 risk_measure = lambda x: tf.reduce_mean(-x, axis = 0),
				 pricebs = None,
				 input_func = tf.identity,
				 trade_rest = tf.identity,
				 activation = 'tanh',
				 price_reg = 0.0,
				 *args, **kwargs):
		"""
		:param d: Market of size d
		"""
		super().__init__(*args, **kwargs)

		# number of stocks to trade
		self.d = d
		# Number of trade days + 1
		self.N = N
		# Risk aversion of Client
		self.risk_measure = risk_measure
		# Function applied to stock before input to NN
		self.input_func = input_func

		self.trade_rest = trade_rest

		self.initial_price = initial_price
		self.activation = activation

		# option to price
		self.option_args = None
		self.option = option

		self.string_name = "Markovian"

		# Transaction costs
		if cost_fct is None:
			def cost_fct(incr, price):
				return tf.Variable([[0.0]], trainable = False, dtype = 'float32')
			self.cost_fct = cost_fct
		else:
			self.cost_fct = cost_fct

		# Price of Option (if None, will be learned)
		if pricebs is None:
			self.pricebs = tf.Variable(0.0, trainable = True,
									   name = 'price', dtype = 'float32')
		else:
			self.pricebs = pricebs

		if self.trade_rest is None:
			self.trade_rest = tf.identity

		self.trade_blocks = [TradeBlock(self.d,
										initial_price = initial_price,
										activation = activation,
										name = "tb_" + str(_)) for _ in range(N - 1)]

	def build(self, input_shape = None):
		# The input dimension should be twice the output dimension:
		# Once for the current trading strategy and once for information about the objects being traded
		# Might add more in case more information is needed
		for block in self.trade_blocks:
			block.build((None, self.d * 2))

	def call(self, x):
		y = tf.zeros_like(x[:, 0, :], dtype = 'float32')
		y_old = tf.zeros_like(x[:, 0, :], dtype = 'float32')
		cur_hedge = tf.zeros_like(x[:, 0, 0], dtype = 'float32')
		cur_costs = tf.zeros_like(x[:, 0, 0], dtype = 'float32')
		# Input given as S
		for i in range(self.N - 1):
			y = self.trade_rest(self.trade_blocks[i]([self.input_func(x[:, i, :]), y]))

			cur_hedge = cur_hedge + tf.reduce_sum(y * (x[:, i + 1, :] - x[:, i, :]), 1)
			cur_costs = cur_costs + tf.reduce_sum(self.cost_fct(y - y_old, x[:, i, :]), 1)
			y_old = y

		return self.risk_measure(self.pricebs + cur_hedge
								 - self.option(x) - cur_costs)

	def get_pl(self, x):
		y = tf.zeros_like(x[:, 0, :], dtype = 'float32')
		y_old = tf.zeros_like(x[:, 0, :], dtype = 'float32')
		cur_hedge = tf.zeros_like(x[:, 0, 0], dtype = 'float32')
		cur_costs = tf.zeros_like(x[:, 0, 0], dtype = 'float32')
		# Input given as S
		for i in range(self.N - 1):
			y = self.trade_rest(self.trade_blocks[i]([self.input_func(x[:, i, :]), y]))

			cur_hedge = cur_hedge + tf.reduce_sum(y * (x[:, i + 1, :] - x[:, i, :]), 1)
			cur_costs = cur_costs + tf.reduce_sum(self.cost_fct(y - y_old, x[:, i, :]), 1)
			y_old = y

		return self.pricebs + cur_hedge - self.option(x) - cur_costs

	def call_k(self, x, k):
		y = tf.zeros_like(x[:, 0, :])
		cur_hedge = tf.zeros_like(x[:, 0, 0])
		# Input given as S
		for i in range(k):
			y = self.trade_blocks[i]([self.input_func(x[:, i, :]), y])

		return y

	def call_only_k(self, x, delta, k):
		assert len(x.shape) == 2
		assert len(delta.shape) == 2
		y = tf.zeros_like(x[:, :])

		y = self.trade_blocks[k - 1]([self.input_func(x[:, :]), delta])

		return y

	def get_hedge(self, x):
		if type(x) is np.ndarray:
			delta = np.zeros_like(x)
		else:
			delta = np.zeros_like(x.numpy())
		y = tf.zeros_like(x[:, 0, :])
		cur_hedge = tf.zeros_like(x[:, 0, 0])
		# Input given as S
		for i in range(self.N - 1):
			y = self.trade_rest(self.trade_blocks[i]([self.input_func(x[:, i, :]), y]))

			cur_hedge = cur_hedge + tf.reduce_sum(y * (x[:, i + 1, :] - x[:, i, :]), 1)

			delta[:, i, :] = y

		delta[:, self.N - 1, :] = delta[:, self.N - 2, :]
		return delta

	def get_price(self):
		return self.pricebs

	def fit(self, print_price = False, *args, **kwargs):
		super(Learner, self).fit(*args, **kwargs)
		if print_price:
			if isinstance(self.pricebs, np.float32) or isinstance(self.pricebs, np.ndarray):
				print("Price:{0:9.3f}".format(self.pricebs))
			else:
				print("Price:{0:9.3f}".format(self.pricebs.numpy()))


# Is exactly like learner, except it should handle path dependent options better,
# since it passes on more information between time-steps!
class OtherLearner(Learner):
	def __init__(self, d, N, option,
				 hidden_size = 1,
				 *args, **kwargs):
		super().__init__(d, N, option, dynamic = True, *args, **kwargs)

		self.string_name = "Hidden"

		self.hidden_size = hidden_size

		self.trade_blocks = [OtherTradeBlock(self.d,
										initial_price = self.initial_price,
										hidden_size = hidden_size,
										activation = self.activation,
										name = "tb_" + str(_)) for _ in range(N - 1)]

	def call(self, x):
		hidden = tf.zeros((x.get_shape().as_list()[0], self.hidden_size), dtype = 'float32')
		y = tf.zeros_like(x[:, 0, :], dtype = 'float32')
		y_old = tf.zeros_like(x[:, 0, :], dtype = 'float32')
		cur_hedge = tf.zeros_like(x[:, 0, 0], dtype = 'float32')
		cur_costs = tf.zeros_like(x[:, 0, 0], dtype = 'float32')
		# Input given as S
		for i in range(self.N - 1):
			y, hidden = self.trade_blocks[i]([self.input_func(x[:, i, :]), y, hidden])

			cur_hedge = cur_hedge + tf.reduce_sum(self.trade_rest(y) * (x[:, i + 1, :] - x[:, i, :]), 1)
			cur_costs = cur_costs + tf.reduce_sum(self.cost_fct(self.trade_rest(y) - y_old, x[:, i, :]), 1)
			y_old = self.trade_rest(y)

		return self.risk_measure(self.pricebs + cur_hedge
								 - self.option(x) - cur_costs)

	def get_pl(self, x):
		hidden = tf.zeros((x.get_shape().as_list()[0], self.hidden_size), dtype = 'float32')
		y = tf.zeros_like(x[:, 0, :], dtype = 'float32')
		y_old = tf.zeros_like(x[:, 0, :], dtype = 'float32')
		cur_hedge = tf.zeros_like(x[:, 0, 0], dtype = 'float32')
		cur_costs = tf.zeros_like(x[:, 0, 0], dtype = 'float32')
		# Input given as S
		for i in range(self.N - 1):
			y, hidden = self.trade_blocks[i]([self.input_func(x[:, i, :]), y, hidden])
			y = self.trade_rest(y)

			cur_hedge = cur_hedge + tf.reduce_sum(y * (x[:, i + 1, :] - x[:, i, :]), 1)
			cur_costs = cur_costs + tf.reduce_sum(self.cost_fct(y - y_old, x[:, i, :]), 1)
			y_old = y

		return self.pricebs + cur_hedge - self.option(x) - cur_costs

	def build(self, input_shape = None):
		# The input dimension should be twice the output dimension:
		# Once for the current trading strategy and once for information about the objects being traded
		# Might add more in case more information is needed
		for block in self.trade_blocks:
			block.build((None, self.d * 2 + self.hidden_size))

	def get_hedge(self, x):
		if type(x) is np.ndarray:
			delta = np.zeros_like(x)
		else:
			delta = np.zeros_like(x.numpy())
		hidden = np.zeros((delta.shape[0], self.N, self.hidden_size))

		y = tf.zeros_like(x[:, 0, :])
		hid = tf.zeros_like(hidden[:, 0, :])
		cur_hedge = tf.zeros_like(x[:, 0, 0])
		# Input given as S
		for i in range(self.N - 1):
			y, hid = self.trade_blocks[i]([self.input_func(x[:, i, :]), y, hid])
			y = self.trade_rest(y)

			cur_hedge = cur_hedge + tf.reduce_sum(y * (x[:, i + 1, :] - x[:, i, :]), 1)

			hidden[:, i, :] = hid
			delta[:, i, :]  = y

		delta[:, self.N - 1, :] = delta[:, self.N - 2, :]
		return delta, hidden

	def call_only_k(self, x, delta, hidden, k):
		assert len(x.shape) == 2
		assert len(delta.shape) == 2
		y = tf.zeros_like(x[:, :])

		y, hidden = self.trade_blocks[k - 1]([self.input_func(x[:, :]), delta, hidden])
		y = self.trade_rest(y)

		return y


class Stopper(tf.keras.Model):
	"""
	Model to find the optimal stopping rule for some function g: time x price --> value

	"""

	def __init__(self, d, N, T, initial_price = 100.,
				 value_function = None,
				 risk_measure = lambda x: tf.reduce_mean(-x, axis = 0),
				 activation = 'tanh',
				 *args, **kwargs):
		"""
		:param d: Market of size d
		"""
		super(Stopper, self).__init__(*args, **kwargs)

		# number of stocks to trade
		self.d = d
		# Number of trade days + 1
		self.N = N
		# initial price
		self.initial_price = initial_price
		# Time increment
		self.dt = tf.cast(T / N, tf.float32)
		# Risk measure
		self.risk_measure = risk_measure
		# Which
		self.call_range = None

		if value_function is None:
			def vf(time, price):
				return price

			self.value_function = vf
		else:
			self.value_function = value_function

		self.stopping_prob = 0.5

		self.blocks = [StopBlock(self.d,
								 initial_price = initial_price,
								 activation = activation,
								 name = "sb_" + str(_)) for _ in range(N - 1)]

	def build(self, input_shape = None):
		for block in self.blocks:
			block.build((None, self.d))

	def call(self, inputs):
		for block in self.blocks:
			block.trainable = True

		reward = self.value_function(self.dt * (self.N - 1), inputs[:, self.N - 1, :])
		# reward = 0

		if self.call_range is None:
			up_to = np.random.randint(2, self.N + 1)
		else:
			up_to = self.call_range
			# I freeze weights for all stops after the one I am training
			for i in range(self.N - up_to + 1, self.N - 1):
				self.blocks[i].trainable = False

		for i in range(2, up_to + 1):
			reward = reward + self.blocks[self.N - i](inputs[:, self.N - i, :]) \
					 * (self.value_function(self.dt * (self.N - i), inputs[:, self.N - i, :]) - reward)

		return self.risk_measure(reward)

	def call_k(self, x, k):
		return self.blocks[k](x)

	def all_prob(self, x):
		delta = [tf.ones_like(x[:, 0, :])] * self.N

		# Input given as S
		for i in range(self.N - 1):
			delta[i] = self.blocks[i](x[:, i, :])

		delta = tf.stack(delta, axis = 1)
		return delta

	# This function returns the optimal time to stop as an index
	def get_stop_time(self, x, threshold = None):
		if threshold is None:
			threshold = self.stopping_prob
		delta = tf.math.ceil(self.all_prob(x) - threshold)
		stop_ind = tf.math.reduce_min(tf.math.argmax(delta, 1), 1)
		return stop_ind

	# This function returns the optimal value at the optimally stopped time
	def get_stop_value(self, x, threshold = None):
		if threshold is None:
			threshold = self.stopping_prob
		stop_time = self.get_stop_time(x, threshold)
		# Reformatting `stop_ind` to be compatible with gather_nd
		stop_ind = tf.stack([tf.cumsum(tf.ones_like(stop_time)) - 1,
							 stop_time,
							 tf.zeros_like(stop_time)], axis = 1)
		return self.value_function(self.dt * tf.cast(stop_time, tf.float32), tf.expand_dims(tf.gather_nd(x, stop_ind), 1))
