from backend.neural_network import *
from backend.pricer import *
from backend.utility_functions import *
import matplotlib.pyplot as plt

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

"""
This script calculates the price of an american call option under the Black Scholes Model.
"""

# Defining constants time, number of trading days, number of stocks and Strike price
T = 20 / 365  # 30 Trading days:
N = 20 + 1  # Trade once per day
d = 1  # Only one stock
K = 100.  # Strike Price

# Defining trading dates
time = np.linspace(0, T, num = N)


# Defining the option
@tf.function
def call_option(t, price, strike = K):
    return tf.maximum(0., price[:, 0] - strike)


@tf.function
def cost_functionA(incr, stock):
    return tf.maximum(0., .01 * tf.abs(incr * stock) - 0.1)


@tf.function
def proportional_costs(incr, stock):
    return 0.001 * tf.abs(incr * stock)


@tf.function
def entropic_risk(x):
    return tf.math.log(tf.reduce_mean(tf.math.exp(-x)))


@tf.function
def no_loss(y_true, y_pred):
    return y_pred


# Number of training examples
K_train = 2 ** 14

# Specifications for the Black Scholes Model
BM = BlackScholesModel(mu = np.array([-.5]),
                       sigma2 = np.array([[0.2 ** 2]]),
                       S0 = np.array([100.]),
                       N = N,
                       T = T,
                       pref_M = 2 ** 12)

# Initialize the Stopping Model
stopping_model = Stopper(d = d, N = N, T = T,
                         value_function = call_option,
                         activation = 'tanh')

# == First we have to train the stopping model ==

# Specify for how many epochs you want to train with each lr

lrs_epoch = [(0.03, 8), (0.01, 8)]

for lr, epoch in lrs_epoch:
    print("Train")
    stopping_model.call_range = N

    print("Learning Rate: " + str(lr))
    optimizer = tf.optimizers.Adam(lr)
    stopping_model.compile(loss = no_loss, optimizer = optimizer)

    # Generate Data
    X_train = BM.generate_instances(M = K_train)
    y_train = np.zeros((K_train, 1))

    stopping_model.fit(x = X_train, y = y_train, epochs = epoch, batch_size = 2 ** 9, verbose = 0)

# Learn the best stopping threshold
S = BM.tf_generate_instances(M = 1000000)
print("\nExpected Value for Stopping for thresholds:\n")
best_exp = -np.inf
best_stop = 0.0
for k in np.arange(0.05, 0.99, 0.02):
    temp = tf.reduce_mean(stopping_model.get_stop_value(S, k)).numpy()
    if best_exp < temp:
        best_exp = temp
        best_stop = k

print("The best stopping probability is: {0:.3f}".format(best_stop))
print("With a Expected Value of: " + str(best_exp))
stopping_model.stopping_prob = best_stop
stopping_model.trainable = False

# A histogram 2
S = BM.tf_generate_instances(M = 1000000)
results = stopping_model.get_stop_value(S, best_stop).numpy()
inds = np.where(results > 0.001)[0]
plt.hist(results[inds], bins = 'sqrt', density = True)
plt.title("Distribution of outcomes > 0")
plt.show()

# == == == == ==
# == Finished Training Optimal Stopping - froze weights ==
# == == == == ==

# == Preliminary training for pricing (as in pricing digital options with tc) ==
# == Specifying possible price range ==

# interval_cuts = 4
# total_interval = 3.0
# lower_bound_price = 0.8
# possible_prices = np.linspace(lower_bound_price, lower_bound_price + total_interval,
# 							  num = interval_cuts, dtype = 'float32')
# loss_prices = np.zeros_like(possible_prices)
# loss_prices_var = np.zeros_like(possible_prices)

# Initializing the pricing / hedging model
# Here we use other learner, because the american option is price dependent!!
pricing_model, best_price, price_loss = find_model_price(d = d, N = N, T = T, nn_model_class = OtherLearner,
                                                         market_model = BM,
                                                         option = stopping_model.get_stop_value,
                                                         risk_measure = entropic_risk,
                                                         cost_function = proportional_costs,
                                                         range_min = 1.8, range_max = 2.8)

# Trading examples
num_exmps = 3
for _ in range(num_exmps):
    S = BM.generate_instances(M = 1)
    stop_time = stopping_model.get_stop_time(S).numpy() * T / (N - 1)
    print("Model Call: " + str(pricing_model(S)))
    delta, hidden = pricing_model.get_hedge(S)

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('time')
    ax1.set_ylabel('Stock', color = color)
    ax1.plot(time, S[0, :, 0], color = color)
    ax1.tick_params(axis = 'y', labelcolor = color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('axes', 1.2))
    ax3.spines['right'].set_visible(True)

    color = 'tab:blue'
    ax2.set_ylabel('Trades', color = color)  # we already handled the x-label with ax1
    ax2.plot(time, delta.squeeze(), color = color)
    ax2.tick_params(axis = 'y', labelcolor = color)
    ax2.axvline(stop_time, ymin = 0.0, ymax = 1.0, color = 'black', ls = '--')

    color = 'tab:green'
    ax3.set_ylabel('Hidden', color = color)  # we already handled the x-label with ax1
    ax3.plot(time, hidden.squeeze(), color = color, ls = '--')
    ax3.tick_params(axis = 'y', labelcolor = color)

    fig.tight_layout()
    plt.show()

# Next we plot a histogram of results
S = BM.tf_generate_instances(M = 100000)
results = pricing_model.get_pl(S)
plt.hist(results.numpy(), bins = 'sqrt', density = True)
plt.title("Histogram of Profits & Losses")
plt.show()

# Trading surfaces
approx_fidelity = 300

# Choose trading day

plot_2d_hist_alt(pricing_model, BM, N, T)

for k in [5, 10, 15]:
    # Calculating True Price
    S = np.linspace(85, 115, approx_fidelity, dtype = np.float32).reshape((approx_fidelity, 1))
    delta = np.linspace(0.0, 0.99, approx_fidelity, dtype = np.float32)
    res = np.zeros((approx_fidelity, approx_fidelity))
    # Calculating Approximate Price
    for i in range(approx_fidelity):
        res[:, i] = pricing_model.call_only_k(S,
                                              delta[i] * np.ones((approx_fidelity, 1)),
                                              np.zeros((approx_fidelity, 1)),
                                              k).numpy().squeeze()

    # Plotting
    fig = plt.figure()
    ax = plt.axes(projection = '3d')

    ax.plot_surface(S, delta, res, cmap = 'viridis', edgecolor = 'none')
    ax.set_title('Strategy based on previous strategy and stock price')
    plt.show()
