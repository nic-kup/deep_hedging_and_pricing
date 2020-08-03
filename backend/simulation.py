import numpy as np
import tensorflow as tf
import numpy.random as rand
import matplotlib.pyplot as plt
from backend.BaseModel import *

"""
This file contains functions related to simulation of stock prices and calculation of costs.
Stocks should always be 3-Tensors!
"""


# This function returns correlated gaussians (i.e. BM increments)
# To recover correlated BM from these increments use `corr_gaussians().cumsum(1)`
def corr_gaussians(rho = 0, T = 1., N = 20, M = 1000):
	dt = T / N
	mean = np.array([0., 0.]) * dt
	cov = np.array([[1., rho], [rho, 1.]]) * dt
	BM = rand.multivariate_normal(mean = mean, cov = cov, size = (N * M)).reshape((M, N, 2))
	return BM


# Heston Model class
class HestonModel(BaseModel):
	def __init__(self, mu, a, b, rho, sigma, V0, S0, N: int, T, pref_M = 1):
		if sigma ** 2 >= 2 * a * b:
			print("WARNING: Feller Condition no satisfies")
		self.mu = mu		# Drift
		self.a = a			# Rate of reversion to b
		self.b = b			# Long Variance
		self.rho = rho		# Correlation
		self.sigma = sigma	# Volatility of Volatility
		self.V0 = V0		# Starting Volatility
		self.S0 = S0		# Starting Price

		self.N = N
		self.T = T
		# number of batches if no number is specified
		self.pref_M = pref_M

	def generate_instances(self, N = None, T = None, M = None):
		if N is None:
			N = self.N
		if T is None:
			T = self.T
		if M is None:
			M = self.pref_M
		dt = T / N
		gaussians = corr_gaussians(rho = self.rho, T = T, N = N, M = M)
		gaussians[:, :, 1] *= self.sigma
		SV = np.multiply(np.ones((M, N, 2)), np.array([self.S0, self.V0]))
		for i in range(1, N):
			SV[:, i, 0] = SV[:, i-1, 0] + self.mu * SV[:, i - 1, 0] * dt +\
						   np.sqrt(SV[:, i - 1, 1]) * SV[:, i - 1, 0] * gaussians[:, i, 0]
			SV[:, i, 1] = SV[:, i-1, 1] + self.a * (self.b - SV[:, i - 1, 1]) * dt + np.sqrt(SV[:, i - 1, 1]) * gaussians[:, i, 1]

			# TODO: if you wanna go fancy, there are dedicated schemes for the Heston model
			# 	see http://www.ressources-actuarielles.net/ext/isfa/1226.nsf/d512ad5b22d73cc1c1257052003f1aed/1826b88b152e65a7c12574b000347c74/$FILE/LeifAndersenHeston.pdf
			SV[:, i, 1] = np.maximum(0., SV[:, i, 1])  # Making Sure volatility >= 0

		return SV


# Heston Model class
class HestonModelRestricted(BaseModel):
	def __init__(self, mu, a, b, rho, sigma, V0, S0, N: int, T, pref_M = 1, restricted = 0):
		if sigma ** 2 >= 2 * a * b:
			print("WARNING: Feller Condition no satisfies")
		self.mu = mu		# Drift
		self.a = a			# Rate of reversion to b
		self.b = b			# Long Variance
		self.rho = rho		# Correlation
		self.sigma = sigma	# Volatility of Volatility
		self.V0 = V0		# Starting Volatility
		self.S0 = S0		# Starting Price

		self.N = N
		self.T = T
		# number of batches if no number is specified
		self.pref_M = pref_M

		self.restriction = restricted

	def generate_instances(self, N = None, T = None, M = None):
		if N is None:
			N = self.N
		if T is None:
			T = self.T
		if M is None:
			M = self.pref_M
		dt = T / N
		gaussians = corr_gaussians(rho = self.rho, T = T, N = N, M = M)
		gaussians[:, :, 1] *= self.sigma
		SV = np.multiply(np.ones((M, N, 2)), np.array([self.S0, self.V0]))
		for i in range(1, N):
			SV[:, i, 0] = SV[:, i-1, 0] + self.mu * SV[:, i - 1, 0] * dt +\
						   np.sqrt(SV[:, i - 1, 1]) * SV[:, i - 1, 0] * gaussians[:, i, 0]
			SV[:, i, 1] = SV[:, i-1, 1] + self.a * (self.b - SV[:, i - 1, 1]) * dt + np.sqrt(SV[:, i - 1, 1]) * gaussians[:, i, 1]

			# TODO: if you wanna go fancy, there are dedicated schemes for the Heston model
			# 	see http://www.ressources-actuarielles.net/ext/isfa/1226.nsf/d512ad5b22d73cc1c1257052003f1aed/1826b88b152e65a7c12574b000347c74/$FILE/LeifAndersenHeston.pdf
			SV[:, i, 1] = np.maximum(0., SV[:, i, 1])  # Making Sure volatility >= 0

		return SV[:, :, self.restriction, np.newaxis]


# Black Scholes Model Class
class BlackScholesModel(BaseModel):
	def __init__(self, mu, sigma2, S0, N: int, T, pref_M: int = 1):
		if np.linalg.det(sigma2) < 0:
			assert "sigma2 value is {0:4.4f}, but should be larger that or equal to 0!".format(sigma2)
		self.mu = mu			# Drift
		self.sigma2 = sigma2	# Variance
		self.S0 = S0			# Starting Value

		self.N = N				# Num timesteps
		self.T = T				# Final Time
		self.pref_M = pref_M	# Preferred minibatches M

	def generate_instances(self, N = None, T = None, M = None):
		if N is None:
			N = self.N
		if T is None:
			T = self.T
		if M is None:
			M = self.pref_M

		dt = T / N
		mean = dt * (self.mu - (1 / 2) * np.diag(self.sigma2))
		cov = dt * self.sigma2
		gaussians = rand.multivariate_normal(mean, cov, N * M)
		gaussians = np.reshape(gaussians, (M, N, len(self.mu)))
		S = np.log(self.S0[np.newaxis, np.newaxis, :]) + np.cumsum(gaussians, 1)
		return np.exp(S).astype('float32')

	def generate_pair(self):
		S = self.generate_instances()
		return (S, np.zeros((self.pref_M, 1)))


class LogNormalNoise(BaseModel):
	def __init__(self, mu, sigma2, N: int, T, pref_M: int = 1):
		if np.linalg.det(sigma2) < 0:
			assert "sigma2 value is {0:4.4f}, but should be larger that or equal to 0!".format(sigma2)
		self.mu = mu			# Drift
		self.sigma2 = sigma2	# Variance

		self.N = N				# Num timesteps
		self.T = T				# Final Time
		self.pref_M = pref_M	# Preferred minibatches M

	def generate_instances(self, N = None, T = None, M = None):
		if N is None:
			N = self.N
		if T is None:
			T = self.T
		if M is None:
			M = self.pref_M

		gaussians = rand.multivariate_normal(self.mu, self.sigma2, N * M)
		gaussians = np.reshape(gaussians, (M, N, len(self.mu)))
		return np.exp(gaussians).astype('float32')

	def generate_pair(self):
		S = self.generate_instances()
		return S, np.zeros((self.pref_M, 1))


def tf_diff(X, axis = 0):
	"""
    like np.diff
    :param X:
    :return:
    """
	n = X.shape[1] - 1
	return tf.slice(X, [0, 0, 0], [-1, n, -1]) - tf.slice(X, [0, 1, 0], [-1, n, -1])


def stoch_int(H, S):
	"""
    This function integrates the parameter H with respect to the parameter S.
    It returns the integral over the whole time interval.
    Warning: the output is shifted to the right by one entry
    In particular stoch_int(1, S) != S as you would expect
    """
	if len(H.shape) == 2:
		H = np.reshape(H, (H.shape[0], H.shape[1], 1))
	return np.cumsum(H.reshape((S.shape[0], S.shape[1], S.shape[2])) *
					 np.concatenate((np.diff(S, 1, 1), np.zeros((S.shape[0], 1, S.shape[2]))), 1), 1)


def tf_stoch_int(H, S):
	"""
    Does the same as stoch_int, but works for tensorflow tensors.
    """
	if len(H.shape) == 2:
		print("This thing happened")
		H = tf.reshape(H, (H.shape[0], H.shape[1], 1))
	return tf.math.cumsum(H * tf.concat((tf_diff(S, 1), tf.zeros((S.shape[0], 1, S.shape[2]))), 1), 1)


def basic_call_option(price, strike_price = 100.):
	return np.maximum(0.0, price - strike_price)


def tf_call_option(x, K = 1):
	"""
	A call option with with strike price K
	:param x:
	:param K:
	:return:
	"""
	return tf.math.maximum(0.0, x - K)


# def digi_option(price, strike = 1):
# 	return np.greater_equal(price, strike) * 1.0
#
#
# def tf_digi_option(price, strike = 1):
# 	return tf.math.greater_equal(price, strike) * 1.0


def transaction_costs(c, delta):
	"""
    :param c: function representing the cost
    :param delta: trading strategy
    :return: costs
    """
	return np.sum(c(np.diff(delta)))


def tf_transaction_costs(c, delta):
	return tf.math.reduce_sum(c(tf_diff(delta)))


if __name__ == "__main__":
	N = 2000
	T = 1.

	heston = HestonModel(mu = 0.0,
						 a = 1.0,
						 b = 0.04,
						 rho = -0.7,
						 sigma = 0.2,
						 V0 = 0.04,
						 S0 = 100,
						 N = N, T = T)
	time = np.linspace(0., T, num = N)

	# Advantage of inheritance
	SV = heston.tf_generate_instances(N = N, T = T)

	fig, ax1 = plt.subplots()
	color = 'tab:red'
	ax1.set_xlabel('time')
	ax1.set_ylabel('Stock', color = color)
	ax1.plot(time, SV.numpy()[0, :, 0], color = color)
	ax1.tick_params(axis = 'y', labelcolor = color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:blue'
	ax2.set_ylabel('volatility', color = color)  # we already handled the x-label with ax1
	ax2.plot(time, SV.numpy()[0, :, 1], color = color)
	ax2.tick_params(axis = 'y', labelcolor = color)

	fig.tight_layout()
	plt.show()
