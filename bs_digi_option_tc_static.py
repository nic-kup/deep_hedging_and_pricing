from backend.neural_network import *
from backend.pricer import *
from backend.utility_functions import *
import matplotlib.pyplot as plt
from backend.plot_funcs import *
from mpl_toolkits import mplot3d

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

"""
This script calculates the price of a binary option with transaction costs under the Black Scholes Model
by fixing a potential price x for the option and then training the neural network with that price in the
loss function. It then does that for every price and returns the price with the lowest loss.

Instead of training a completely new model every time, I simply change the price in the model and
then train for another 5 epochs with a low learning rate.

It is necessary to use a utility function for this to work. The idea is that the correct price is given by:
inf E[ rho( PL ) ] = 0

rho needs to fulfill the following:
    - increasing
    - convex
    - rho(0) = 0
"""

# Defining constants time, number of trading days, number of stocks and Strike price
T = 20 / 365  # 30 Trading days:
N = 20 + 1  # Trade once per day
d = 1  # Only one stock
K = np.float32(100.)  # Strike Price

# Defining trading dates
time = np.linspace(0, T, num = N)

# Specifications for the Black Scholes Model
BM = BlackScholesModel(mu = np.array([0.0]),
                       sigma2 = np.array([[0.2 ** 2]]),
                       S0 = np.array([100.]),
                       N = N, T = T)


# Below we define the following functions
# - the option we want to price
# - The function describing trading frictions
# - The utility function describing trader preferences
#  - Loss
@tf.function
def digi_option(price, strike = K, *args):
    # Price input should be a 3-tensor
    return (tf.math.sign(price[:, N - 2, 0] - strike) + 1.) / 2


@tf.function
def call_option(price, strike = K):
    return tf.maximum(0., price[:, N-1, 0] - strike)


@tf.function
def proportional_costs(incr, stock):
    return 0.001 * tf.abs(incr * stock)


@tf.function
def entropic_risk(x):
    return tf.math.log(tf.reduce_mean(tf.math.exp(-x)))


@tf.function
def smooth_bin_option(price, strike = K):
    return (1/0.1) * (call_option(price, strike - 0.1) - call_option(price, strike))


@tf.function
def neg_expectation(x):
    return tf.reduce_mean(-x)


@tf.function
def worst_case(x):
    return -tf.reduce_min(x)


model, best_price, price_loss = find_model_price(d=d, N=N, T=T, nn_model_class = Learner, market_model = BM,
                                     option = digi_option, risk_measure = entropic_risk,
                                     cost_function = proportional_costs,
                                     range_min = 0.3, range_max = 1.0)

# == Plotting ==
print("The best price is: {0:.4f} with a loss of {1:.4f}".format(best_price, price_loss))

# First we plot some example of trades
plot_example(model, BM, N, T, 5)

# Next we plot a histogram of results
S = BM.tf_generate_instances(M = 100000)
results = model.get_pl(S)
plt.hist(results.numpy(), bins = 'sqrt', density = True)
plt.title("Histogram of Profits & Losses")
plt.show()

# 2D Histogram alt
plot_2d_hist_alt(model, BM, N, T)

# 2D Histogram
plot_2d_hist_digi(model, BM, N, T, (100, 3))

# Strategy Surface
plot_strategy(model, 300)
