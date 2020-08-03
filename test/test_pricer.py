from backend.pricer import *
from backend.simulation import *


def test_bs_call_price_ptws_end():
	# Testing if it returns the correct value at the end of the time
	np.testing.assert_equal(bs_call_price_ptws(100, 90, 0.3, 1, 1), 10.)
	np.testing.assert_equal(bs_call_price_ptws(100, 95, 0.5, 0.8, 0.8), 5.)
	np.testing.assert_equal(bs_call_price_ptws(80, 95, 0.5, 0.8, 0.8), 0.)


def test_bs_digi_price_ptws_end():
	np.testing.assert_equal(bs_digi_price_ptws(100, 90, 0.3, 1, 1), 1.)
	np.testing.assert_equal(bs_digi_price_ptws(100, 95, 0.5, 0.8, 0.8), 1.)
	np.testing.assert_equal(bs_digi_price_ptws(80, 95, 0.5, 0.8, 0.8), 0.)


def test_bs_call_monte_carlo():
	num_steps = 20
	num_sim = 100000
	T = 0.5

	mu = np.array([0.])
	S0 = np.array([1.1])
	sigma2 = np.array([[0.2 ** 2]])

	results = generate_gbm(S0, mu, sigma2, N = num_steps, T = T, M = num_sim)[:, num_steps - 1, :]

	K = np.array(1.)

	np.testing.assert_almost_equal(np.mean(basic_call_option(results, K)),
								   bs_call_price_ptws(S0, K, np.sqrt(sigma2), 0, T),
								   3)


def test_bs_digi_monte_carlo():
	num_steps = 20
	num_sim = 20000
	T = 0.5

	mu = np.array([0.])
	S0 = np.array([1.1])
	sigma2 = np.array([[0.2]])

	results = generate_gbm(S0, mu, sigma2, num_steps, T, num_sim)[num_steps - 1]

	K = np.array(1.)

	np.testing.assert_almost_equal(np.mean(digi_option(results, K)),
								   bs_digi_price_ptws(S0, K, np.sqrt(sigma2), 0, T),
								   2)


def test_bs_call_delta_ptws():
	epsilon = 0.001

	num_steps = 20
	num_sim = 50000
	T = 0.5

	mu = np.array([0.])
	S0 = np.array([1.1])
	sigma2 = np.array([[0.2]])

	K = np.array(1.)

	results = generate_gbm(S0, mu, sigma2, num_steps, T, num_sim) [num_steps - 1]

	resultsA1 = (1 + epsilon / S0) * results
	resultsA2 = (1 - epsilon / S0) * results

	resultsB1 = (1 + epsilon / (2 * S0)) * results
	resultsB2 = (1 - epsilon / (2 * S0)) * results

	resultsA1 = np.mean(call_option(resultsA1, K))
	resultsA2 = np.mean(call_option(resultsA2, K))

	resultsB1 = np.mean(call_option(resultsB1, K))
	resultsB2 = np.mean(call_option(resultsB2, K))

	T_h = (resultsA1 - resultsA2)/(2 * epsilon)
	T_h2 = (resultsB1 - resultsB2) / epsilon

	# Trapezoidal Rule
	approx_delta = (4/3) * T_h2 - (1/3) * T_h

	np.testing.assert_almost_equal(approx_delta,
	                               bs_call_delta_ptws(S0, K, np.sqrt(sigma2), 0, T),
	                               2)


def test_bs_digi_delta_ptws():
	epsilon = 0.001

	num_steps = 20
	num_sim = 500000

	T = 0.5
	mu = np.array([0.])
	S0 = np.array([1.1])
	sigma2 = np.array([[0.2]])

	K = np.array(1.)

	results = generate_gbm(S0, mu, sigma2, num_steps, T, num_sim)[num_steps - 1]

	resultsA1 = (1 + epsilon / S0) * results
	resultsA2 = (1 - epsilon / S0) * results

	resultsB1 = (1 + epsilon / (2 * S0)) * results
	resultsB2 = (1 - epsilon / (2 * S0)) * results

	resultsA1 = np.mean(digi_option(resultsA1, K))
	resultsA2 = np.mean(digi_option(resultsA2, K))

	resultsB1 = np.mean(digi_option(resultsB1, K))
	resultsB2 = np.mean(digi_option(resultsB2, K))

	T_h = (resultsA1 - resultsA2)/(2 * epsilon)
	T_h2 = (resultsB1 - resultsB2) / epsilon

	# Iterated Trapezoidal Rule
	approx_delta = (4/3) * T_h2 - (1/3) * T_h

	np.testing.assert_almost_equal(approx_delta,
	                               bs_digi_delta_ptws(S0, K, np.sqrt(sigma2), 0, T),
	                               1)