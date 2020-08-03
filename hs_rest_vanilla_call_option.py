import tensorflow as tf
from backend.neural_network import *
from backend.pricer import *
from backend.utility_functions import *
from backend.plot_funcs import *
import matplotlib.pyplot as plt

"""
Here we calculate the price of a vanilla binary option in the Black Scholes model with 20 trading days.
The parameters can easily be changed in the beginning of the script.
"""

# Defining constants time, number of trading days, number of stocks and Strike price
T = 20 / 365  # 30 Trading days:
N = 20 + 1  # Trade once per day
d = 2  # Only one stock
K = 100.  # Strike Price

# Defining trading dates
time = np.linspace(0, T, num = N)


# Defining the option
@tf.function
def call_option(price, strike = K):
	return tf.maximum(0., price[:, N-1, 0] - strike)


@tf.function
def digi_option(price, strike = K, *args):
	# Price input should be a 3-tensor
	return (tf.math.sign(price[:, N - 1, 0] - strike) + 1.) / 2


@tf.function
def cost_functionB(incr, stock):
	return 0.001 * tf.abs(incr * stock)


@tf.function
def entropic_risk(x):
	return tf.math.log(tf.reduce_mean(tf.math.exp(-x), axis = -1))


@tf.function
def power_risk(x):
	return tf.math.sqrt(tf.reduce_mean(tf.square(-x), axis = -1))


@tf.function
def neg_expectation(x):
	return tf.reduce_mean(-x, axis = -1)


@tf.function
def square_first(x):
	return tf.reduce_mean(tf.square(-x), axis = -1)


@tf.function
def restrict_one(strat):
	tbr = tf.Variable([[0., 0.], [0., 1.]], dtype = tf.float32)
	return tf.matmul(strat, tbr)


# Number of training examples
K_train = 2 ** 15

# Specifications for the Black Scholes Model
BM = HestonModel(mu = 0.0,
					 a = 0.8,
					 b = 0.05,
					 rho = -0.65,
					 sigma = 0.20,
					 V0 = 0.07,
					 S0 = 100,
					 N = N, T = T)

# initializing the Model
model = OtherLearner(d = d, N = N, initial_price = 100.,
				option = call_option,
				risk_measure = entropic_risk,
				trade_rest = restrict_one,
				cost_fct = cost_functionB,
				pricebs = 0.,
				activation = 'tanh')

lrs_epoch = [(0.01, 8), (0.004, 8), (0.001, 12)]

for lr, epoch in lrs_epoch:
	print("Learning Rate: " + str(lr))
	optimizer = tf.optimizers.Adam(lr)
	model.compile(loss = no_loss, optimizer = optimizer)

	# Generate Data
	X_train = BM.generate_instances(N = N, T = T, M = K_train)
	y_train = np.zeros((K_train, 0))

	print(X_train.shape)

	model.fit(x = X_train, y = y_train, epochs = epoch, verbose = 2, batch_size = 2 ** 11)

	print("Model Price: " + str(model.pricebs))


# == Plotting ==
print("Internal Model Price: " + str(model.pricebs))

S = BM.tf_generate_instances(M = 100000)
results = model(S)
print("Model Price: " + str(results))

# First we plot some example of trades
hm_plot_example(model, BM, N, T, 5)

model.utility_function = tf.identity

# Next we plot a histogram of results
S = BM.tf_generate_instances(M = 100000)
results = model.get_pl(S)
plt.hist(results.numpy(), bins = 'sqrt', density = True)
plt.title("Histogram of Profits & Losses")
plt.show()

# 2D Histogram
plot_2d_hist_call(model, BM, N, T)

# Strategy Surface
# plot_strategy(model, 300)

# == Plotting Trading Curve ==
approx_fidelity = 300

# Choose trading day
k = 10


# --- #
S = np.zeros((approx_fidelity, 2))
S[:, 0] = np.linspace(85, 115, approx_fidelity, dtype = np.float32)
S[:, 1] += BM.b

# Setting delta to 0 strategy (only needed for Learner)
delta = np.zeros((approx_fidelity, 2))
delta[:, 1] = -100.

# Calculating Approximate Price
approx_delta = model.call_only_k(S,
                                 delta,
                                 k).numpy().squeeze()
# Plotting
plt.plot(S[:, 0].squeeze(), approx_delta[:, 0].squeeze())
plt.show()


S = np.zeros((approx_fidelity, 2))
S[:, 1] += np.linspace(0.01, 0.1, approx_fidelity, dtype = np.float32)

res = np.zeros((approx_fidelity, approx_fidelity))
stock_prices = np.linspace(85, 115, approx_fidelity)

for i in range(approx_fidelity):
	S[:, 0] = stock_prices[i]
	res[:, i] = model.call_only_k(S, delta, k).numpy().squeeze()[:, 1].squeeze()

fig = plt.figure()
ax = plt.axes(projection = '3d')

ax.plot_surface(stock_prices, S[:, 1], res, cmap = 'viridis', edgecolor = 'none')

plt.show()
# --- #
