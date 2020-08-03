import scipy.stats as sps
from backend.simulation import *
import matplotlib.pyplot as plt


# Black Scholes price for a call option at a given time
def bs_call_price_ptws(cur_price, K, vol, cur_time, T):
	if cur_time == T:
		return np.maximum(cur_price - K, 0)
	else:
		x = (np.log(cur_price) - np.log(K)) / (vol * np.sqrt(T - cur_time))
		y = (1 / 2) * vol * np.sqrt(T - cur_time)
		return cur_price * sps.norm.cdf(x + y) - K * sps.norm.cdf(x - y)


# Black Scholes delta for a call option at a given time
def bs_call_delta_ptws(cur_price, K, vol, cur_time, T):
	if T == cur_time:
		return 1*(cur_price >= K)
	else:
		return sps.norm.cdf((np.log(cur_price/K) + 0.5 * (vol ** 2) * (T - cur_time))/(vol * np.sqrt(T - cur_time)))


# Black Scholes price for a digital option at a given time
def bs_digi_price_ptws(cur_price, K, vol, cur_time, T):
	if cur_time == T:
		return (cur_price >= K) * 1
	else:
		return 1 - sps.norm.cdf(
			(np.log(K / cur_price) + 0.5 * (vol ** 2) * (T - cur_time)) / (vol * np.sqrt(T - cur_time)))


# Black Scholes delta for a digital option at a given time
def bs_digi_delta_ptws(cur_price, K, vol, cur_time, T):
	if cur_time == T:
		return 0
	else:
		return 1/(cur_price * vol * np.sqrt(T - cur_time)) * \
			   sps.norm.pdf((np.log(cur_price/K) - 0.5 * vol ** 2 * (T - cur_time))/(vol * np.sqrt(T - cur_time)))


def hes_call_price_ptws(cur_price, K, specs, cur_time, T):
	if cur_time == T:
		return np.maximum(cur_price - K, 0)


def cts_prices(ptw_func, prices, K, vol, times, T):
	values = np.zeros_like(times)
	for i in range(times.shape[0]):
		values[i] = ptw_func(prices[i], K, vol, times[i], T)
	return values


def get_multi_hedges(ptw_func, prices, K, vol, times, T):
	values = np.zeros_like(prices)
	for i in range(times.shape[0]):
		values[:, i] = ptw_func(prices[:, i], K, vol, times[i], T)
	return values


if __name__ == "__main__":
	plt.rcParams.update({'font.size': 14})

	mu = np.array([0])
	sigma = np.array([[0.2]])

	T = 20 / 365

	num_steps = 500

	K = 100.
	S = np.linspace(85, 115, num = num_steps)
	time = np.ones_like(S) * (10/365)

	for i in [5, 10, 15, 19, 19.5]:
		time = np.ones_like(S) * (i / 365)
		delta = np.squeeze(cts_prices(bs_digi_delta_ptws, S, K, sigma, time, T))
		plt.plot(S, delta, label = 'Day ' + str(i), lw = 1.6)

		plt.legend()
	plt.title("Digital Delta for 20 day expiry time")
	plt.show()

	print("Call price: " + str(bs_call_price_ptws(100., K, sigma, 0, T)))
	print("Bina price: " + str(bs_digi_price_ptws(100., K, sigma, 0, T)))

	# num_steps = 500
	# time = np.linspace(0, 1, num = num_steps)
	#
	# S0 = np.array([2.])
	# mu = np.array([0])
	# sigma = np.array([[0.2]])
	#
	# BM = BlackScholesModel(mu = np.array([0.0]),
	# 					   sigma2 = np.array([[0.2 ** 2]]),
	# 					   S0 = np.array([2.]),
	# 					   N = num_steps, T = 1)
	#
	# K = 2.
	# S = np.squeeze(BM.generate_instances())
	# S = 1.1 * np.ones_like(time)
	# call_price = np.squeeze(cts_prices(bs_call_price_ptws, S, K, sigma, time, 1))
	# c_dl_price = np.squeeze(cts_prices(bs_call_delta_ptws, S, K, sigma, time, 1))
	#
	# digi_price = np.squeeze(cts_prices(bs_digi_price_ptws, S, K, sigma, time, 1))
	# d_dl_price = np.squeeze(cts_prices(bs_digi_delta_ptws, S, K, sigma, time, 1))
	#
	# print("The max digital delta price is: " + str(np.max(d_dl_price)))
	#
	# plt.plot(time, call_price, color = '#2222ee', linewidth = 1)
	# plt.plot(time, c_dl_price, color = '#1166bb', linewidth = 1)
	#
	# plt.plot(time, digi_price, color = '#22ee22', linewidth = 1)
	# plt.plot(time, d_dl_price, color = '#11bb66', linewidth = 1)
	#
	# plt.hlines(K, color = 'red', xmin = 0, xmax = 1)
	# plt.plot(time, S, color = 'black')
	#
	# plt.ylim((0, 3))
	#
	# plt.show()
