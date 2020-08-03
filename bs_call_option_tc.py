from backend.neural_network import *
from backend.pricer import *
from backend.utility_functions import *
import matplotlib.pyplot as plt
from backend.plot_funcs import *
from mpl_toolkits import mplot3d

plt.rcParams.update({'font.size': 13})

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
def call_option(price, strike = K):
    return tf.maximum(0., price[:, N-2, 0] - strike)


@tf.function
def cost_functionA(incr, stock):
    return tf.maximum(0., tf.sign(tf.abs(incr) - 0.001)) * 0.01


tc_const = 0.001


@tf.function
def cost_functionB(incr, stock):
    return tc_const * tf.abs(incr * stock)


@tf.function
def linear_loss(y_true, y_pred):
    return y_pred - y_true


@tf.function
def entropic_risk(x):
    return tf.math.log(tf.reduce_mean(tf.math.exp(-x)))


@tf.function
def worst_case(x):
    return -tf.reduce_min(x)


model, best_price, price_loss = find_model_price(d=d, N=N, T=T, nn_model_class = Learner, market_model = BM,
									 option = call_option, risk_measure = entropic_risk,
									 cost_function = cost_functionB,
									 range_min = 1.5, range_max = 2.5, final_retrain = True)

S = BM.tf_generate_instances(M = 100000)
results = model.get_pl(S)
plt.hist(results.numpy(), bins = 'sqrt', density = True)
plt.title("Histogram of Profits & Losses")
plt.show()

# First we plot some example of trades
plot_example(model, BM, N, T, 5)

# 2D Histogram alt
plot_2d_hist_alt(model, BM, N, T)

# 2D Histogram
plot_2d_hist_call(model, BM, N, T)

# Strategy Surface
plot_strategy(model, 300)
