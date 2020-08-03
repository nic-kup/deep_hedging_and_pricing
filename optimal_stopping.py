from backend.neural_network import *
import matplotlib.pyplot as plt

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
	tf.config.experimental.set_memory_growth(device, True)

# Defining constants time, number of trading days, number of stocks and Strike price
T = 20 / 365  # 30 Trading days:
N = 20 + 1  # Trade once per day
d = 1  # Only one stock
K = 100.  # Strike Price

# Defining trading dates
time = np.linspace(0, T, num = N)

# Number of training examples
K_train = 2 ** 15

# Specifications for the Black Scholes Model
BM = BlackScholesModel(mu = np.array([-0.6]),
					   sigma2 = np.array([[0.2 ** 2]]),
					   S0 = np.array([100.]),
					   N = N,
					   T = T,
					   pref_M = 2 ** 12)


# BM = LogNormalNoise(mu = np.array([0.0]),
# 					   sigma2 = np.array([[0.2 ** 2]]),
# 					   N = N,
# 					   T = T,
# 					   pref_M = 2 ** 12)

# S = BM.tf_generate_instances(M = 1)
# plt.plot(time, S[0, :, 0].numpy())
# plt.show()


# Defining the option
@tf.function
def call_option(t, price, strike = K):
	return tf.maximum(0., price[:, 0] - strike)


@tf.function
def shifted_id(t, price, strike = K):
	return price[:, 0] - strike


@tf.function
def entropic_risk(x):
	return tf.math.log(tf.reduce_mean(tf.math.exp(-x)))


@tf.function
def neg_expectation(x):
	return tf.reduce_mean(-x)


@tf.function
def no_loss(y_true, y_pred):
	return y_pred


# initializing the Model
model = Stopper(d = d, N = N, T = T,
				value_function = call_option,
				risk_measure = neg_expectation,
				activation = 'tanh')

# Specify for how many epochs you want to train with each lr
lrs_epoch = [(0.03, 8)]

for lr, epoch in lrs_epoch:
	print("Train")
	model.call_range = N

	print("Learning Rate: " + str(lr))
	optimizer = tf.optimizers.Adam(lr)
	model.compile(loss = no_loss, optimizer = optimizer)

	# Generate Data
	X_train = BM.generate_instances(M = K_train)
	y_train = np.zeros((K_train, 1))

	model.fit(x = X_train, y = y_train, epochs = epoch, batch_size = 2 ** 9, verbose = 2)


# lrs_epoch = [(0.01, 8)]
# for lr, epoch in lrs_epoch:
# 	for call_range in range(N - 1):
# 		print("On Layer " + str(N - call_range - 2))
# 		model.call_range = call_range + 2
#
# 		print("Learning Rate: " + str(lr))
# 		optimizer = tf.optimizers.Adam(lr)
# 		model.compile(loss = no_loss, optimizer = optimizer)
#
# 		# Generate Data
# 		X_train = BM.generate_instances(M = K_train)
# 		y_train = np.zeros((K_train, 1))
#
# 		model.fit(x = X_train, y = y_train, epochs = epoch, batch_size = 2 ** 11, verbose = 0)

# Learn the best stopping threshold
S = BM.tf_generate_instances(M = 1000000)
print("\nExpected Value for Stopping for thresholds:\n")
best_exp = -np.inf
best_stop = 0.0
for k in np.arange(0.01, 0.99, 0.03):
	temp = tf.reduce_mean(model.get_stop_value(S, k)).numpy()
	if best_exp < temp:
		best_exp = temp
		best_stop = k

print("The best stopping probability is: {0:.3f}".format(best_stop))
print("With a Expected Value of: " + str(best_exp))
model.stopping_prob = best_stop

# =========================
# == Plotting an example ==
# =========================

# A histogram 2
S = BM.tf_generate_instances(M = 1000000)
results = model.get_stop_value(S, best_stop).numpy()
inds = np.where(results > 0.001)[0]
plt.hist(results[inds], bins = 'sqrt', density = True)
plt.title("Distribution of outcomes > 0")
plt.show()

# The examples
num_exmps = 5
for _ in range(num_exmps):
	S = BM.tf_generate_instances(M = 1)
	print("Example " + str(_) + "/" + str(num_exmps - 1) + "\nStop Time: " + str(model.get_stop_time(S)))
	print("Stop Value: " + str(model.get_stop_value(S)))
	delta = model.all_prob(S).numpy()

	fig, ax1 = plt.subplots()
	color = 'tab:red'
	ax1.set_xlabel('time')
	ax1.set_ylabel('Stock', color = color)
	ax1.plot(time, S[0, :, 0].numpy(), color = color)
	# ax1.hlines(100., xmin = 0, xmax = T, color = color)
	ax1.tick_params(axis = 'y', labelcolor = color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:blue'
	ax2.set_ylabel('Stopping Probability', color = color)  # we already handled the x-label with ax1
	ax2.plot(time, delta.squeeze(), color = color)
	ax2.hlines(model.stopping_prob, xmin = 0, xmax = T, color = color)
	ax2.tick_params(axis = 'y', labelcolor = color)

	ax2.axvline(model.get_stop_time(S).numpy() * T / (N - 1), ymin = 0.0, ymax = 1.0, color = 'black', ls = '--')

	fig.tight_layout()
	plt.show()

S = np.linspace(85, 115, num = 250, dtype = 'float32').reshape((250, 1, 1))

plt.figure(figsize = (10, 7))

for k in range(N - 1):
	if k % 1 == 0:
		y = model.call_k(S, k)
		plt.plot(S.squeeze(), y.numpy().squeeze(), label = str(k))
		plt.legend()

plt.hlines(y = model.stopping_prob, xmin = 85, xmax = 115, color = 'black', ls = '--')
plt.show()
