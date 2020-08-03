from backend.pricer import *
import numpy as np
import matplotlib.pyplot as plt


@tf.function
def loss_function(x, alpha = 0.2):
	return (1. / (1. - alpha)) * tf.maximum(0., x)


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
	return -Z + p_0 + tf.math.reduce_sum(tf_stoch_int(delta, S)[-1]) - tc


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


class RecurrentLearner(tf.keras.Model):
	def __init__(self, d, *args, **kwargs):
		"""
		:param d: Market of size d
		"""
		super().__init__(*args, **kwargs)

		# Structure as proposed in paper except time

		self.dense1 = tf.keras.layers.Dense(d + 15, activation = 'sigmoid',
		                                    kernel_initializer = tf.random_uniform_initializer(minval = 0, maxval = 0.2))

		self.dense2 = tf.keras.layers.Dense(d + 15, activation = 'sigmoid',
		                                    kernel_initializer = tf.random_uniform_initializer(minval = 0, maxval = 0.2))
		self.dense3 = tf.keras.layers.Dense(d + 15, activation = 'tanh',
		                                    kernel_initializer = tf.random_uniform_initializer(minval = 0, maxval = 0.2))

		self.final1 = tf.keras.layers.Dense(d, activation = 'tanh',
		                                    kernel_initializer = tf.random_uniform_initializer(minval = 0, maxval = 0.2))

		self.final2 = tf.keras.layers.Dense(d, activation = None,
		                                    kernel_initializer = tf.random_uniform_initializer(minval = 0, maxval = 0.2))

	def call(self, x):
		x_temp = tf.concat([tf.math.log(x[0]), x[1]], -1)
		y1 = self.dense1(x_temp) * x[2]
		y2 = self.dense2(x_temp) * self.dense3(x_temp)
		c_out = y1 + y2
		delta_out = self.final1(x_temp) * self.final2(c_out)
		return delta_out, c_out

	def get_hedge(self, S, T):
		i = 0

		# v  I don't like this  v
		N = S.shape[0]
		d = S.shape[1]
		batch_size = S.shape[2]
		time = np.linspace(0, T, num = N)
		# ^  I don't like this  ^

		cur_c_val = tf.Variable([[0.0] * (d + 15)] * batch_size, trainable = False, dtype = tf.dtypes.float32)
		cur_int_val = tf.Variable([[0.0] * d] * batch_size, trainable = False, dtype = tf.dtypes.float32)

		delta = np.zeros((N, batch_size, d))
		delta[0], cur_c_val = self.call([
			tf.reshape(tf.Variable(S[i, :, :], dtype = tf.dtypes.float32), (batch_size, d)),
			delta[0], cur_c_val])

		# To add time back in: tf.Variable([[T - time[i]]] * batch_size)

		for i in range(0, N - 1):
			delta[i + 1], cur_c_val = self.call([
				tf.reshape(tf.Variable(S[i, :, :], dtype = tf.dtypes.float32), (batch_size, d)),
				delta[i], cur_c_val])

			cur_int_val = cur_int_val + delta[i] * (S[i + 1] - S[i]).T

		return delta


# met = tf.keras.metrics.CosineSimilarity()
# optimizer = tf.keras.optimizers.Adam()


def train_minibatch(model, specifications, optimizer, utility, option, p_0: float, cost_function,
                    epochs: int, N: int, T: float, batch_size: int = 16):
	for epoch in range(epochs):

		S0 = specifications['S0']
		mu = specifications['mu']
		sigma2 = specifications['sigma2']
		d = mu.shape[0]

		# Note S.shape = (time_steps, len(mu), batch_size)
		S = tf.Variable(generate_gbm(S0 = S0, mu = mu, sigma2 = sigma2, N = N, T = T, M = batch_size), dtype = tf.dtypes.float32)
		time = np.linspace(0, T, num = N)
		with tf.GradientTape() as t:
			i = 0

			cur_c_val = tf.Variable([[0.0] * (d + 15)] * batch_size, trainable = False, dtype = tf.dtypes.float32)

			### (delta \cdot S)_t
			cur_int_val = tf.Variable([[0.0] * d] * batch_size, trainable = False, dtype = tf.dtypes.float32)

			### hedge amount
			cur_delta_val = tf.Variable([[0.0] * d] * batch_size, trainable = False, dtype = tf.dtypes.float32)

			# C_t(\delta)
			# cur_tc = tf.Variable(0.0, trainable = False)

			for i in range(0, N - 1):
				### Adding up transaction costs
				# cur_tc = cur_tc + cost_function(cur_delta_val - prev_delta_val)

				### Calculating Previous Delta (needed for transaction costs)
				# prev_delta_val = cur_delta_val

				### Calculating new delta
				cur_delta_val, cur_c_val = model([
					tf.reshape(S[i, :, :], (batch_size, d)),
					cur_delta_val, cur_c_val])

				### Adding up stochastic integral
				cur_int_val = cur_int_val + cur_delta_val * tf.transpose(S[i + 1] - S[i])

			### Loss function
			loss = utility(option(S[N - 1]), p_0, tf.reduce_sum(cur_int_val, 1))
			loss = tf.math.square(loss)
			loss = tf.reduce_sum(loss / batch_size)

		### Calculating then applying gradients
		dw = t.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(dw, model.trainable_variables))

		if epoch % (epochs // 5) == 0:
			print(str(epoch) + ", Loss = " + str(loss))
	print("Final Loss = " + str(loss))


if __name__ == "__main__":
	K = 100.


	def call_option(x):
		return tf.maximum(0.0, x[0] - K)


	if False:
		T = 0.5
		model = Learner(d = 1)
		S = np.linspace(-2, 2, 200)
		S = S.reshape((1, 1, 200))
		approx = model.get_hedge(S, T).squeeze()
		plt.plot(S.squeeze(), approx)
		plt.show()

	if True:
		# Time until end of contract
		T = 0.5
		# Number of possible trading occasions
		num_steps = 60

		# Black Scholes Model Specifications
		specs = {
			'S0': np.array([110., 100.]),
			'mu': np.array([0.0, 0.0]),
			'sigma2': np.array([[0.2 ** 2, 0.],
			                    [0., 0.18 ** 2]])
		}
		model = RecurrentLearner(d = 2)

		optimizer = tf.keras.optimizers.Adam(0.001)
		train_minibatch(model, specs, optimizer, objective_alt, call_option, 0, lambda x: 0,
		                epochs = 40, N = num_steps, T = 1., batch_size = 2 ** 10)

		S = generate_gbm(S0 = specs['S0'],
		                 mu = specs['mu'],
		                 sigma2 = specs['sigma2'], N = num_steps, T = T, M = 1)
		delta = model.get_hedge(S, T)

		time = np.linspace(0, T, num = num_steps)

		print("Stochastic Integral: " + str(stoch_int(delta, S)[-1]))
		print("Call Option Value: " + str(call_option(S[-1])))
		print("Profits and Losses: " + str(PLT_alt(call_option(S[-1]), 0,
		                                           np.sum(stoch_int(delta, S)[-1]), tc = 0.)))

		plt.plot(time, S.squeeze() / 100, label = "stocks")
		plt.plot(time, delta.squeeze(), label = "trades")
		plt.legend(loc = 'upper left')
		plt.show()
