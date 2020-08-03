import tensorflow as tf
from backend.plot_funcs import *


@tf.function
def exp_util(x):
	return 1.0 * tf.exp(1.0 * x)


@tf.function
def log_util(x):
	return -tf.math.log(-x / 2. + 1.)


@tf.function
def power_util(x, alpha = 0.2, k = 5.):
	return -(tf.pow(-x + k, alpha) - tf.pow(k, alpha)) / alpha


@tf.function
def linear_util(x):
	return x


@tf.function
def linear_loss(y_true, y_pred):
	return tf.reduce_mean(y_pred - y_true, axis = -1)


@tf.function
def no_loss(y_true, y_pred):
	return y_pred


def find_model_price(d, N, T, nn_model_class, market_model,
					 option, risk_measure, cost_function,
					 range_min, range_max, trade_rest = None, final_retrain = True):
	print()
	interval_cuts = 5
	total_interval = range_max - range_min - (range_max - range_min) / interval_cuts
	possible_prices = np.linspace(range_min, range_min + total_interval, num = interval_cuts, dtype = 'float32')

	loss_prices = np.zeros_like(possible_prices)
	loss_prices_var = np.zeros_like(possible_prices)

	# initializing the Model
	model = nn_model_class(d = d, N = N,
						   initial_price = 100.,
						   option = option,
						   cost_fct = cost_function,
						   risk_measure = risk_measure,
						   trade_rest = trade_rest,
						   activation = 'tanh',
						   # hidden_size = 1,
						   pricebs = possible_prices[0])

	# Specify loss function
	loss_choice = no_loss

	# Specify for how many epochs you want to train with each lr
	K_train = 2 ** 16
	lrs_epoch = [(0.01, 6), (0.005, 8)]

	# Train once to get a rough strategy
	for lr, epoch in lrs_epoch:
		optimizer = tf.optimizers.Adam(lr)
		model.compile(loss = loss_choice, optimizer = optimizer)

		# Generate Data
		X_train = market_model.generate_instances(N = N, T = T, M = K_train)
		y_train = np.zeros((K_train, 1))

		model.fit(x = X_train, y = y_train, epochs = epoch, verbose = 0, batch_size = 2 ** 11)

	print("Pre-Training done")
	# Retrain for different prices
	K_train = 2 ** 16
	lrs_epoch = [(0.003, 8)]

	for _ in range(3):
		print("Prices Testing: " + str(possible_prices))

		iter_prices = enumerate(possible_prices)
		if _ >= 1:
			# Taking the first entry for loss from the previous loop
			next(iter_prices)

		for i, price in iter_prices:
			model.pricebs = price
			for lr, epoch in lrs_epoch:
				print("\r" + str(i) + ": Learning Rate: " + str(lr), end = '')
				optimizer = tf.optimizers.Adam(lr)
				model.compile(loss = loss_choice, optimizer = optimizer)

				# Generating prices
				X_train = market_model.generate_instances(N = N, T = T, M = K_train)
				y_train = np.zeros((K_train, 1))

				model.fit(x = X_train, y = y_train, epochs = epoch, verbose = 0, batch_size = 2 ** 11)

			X = model(market_model.generate_instances(N = N, T = T, M = 2 * K_train))
			loss_prices[i] = X
			loss_prices_var[i] = np.var(X)

		print("\rLoss for each price: " + str(loss_prices))
		loss_prices = loss_prices - 0.005  # If price is to close to zero rather take the smaller one
		# The min index is the last loss that is not negative
		min_index = next((ind for ind, val in enumerate(loss_prices) if val < 0), interval_cuts)
		min_index = min_index - np.sign(min_index)

		loss_prices = np.ones_like(loss_prices) * loss_prices[min_index]
		best_price = possible_prices[min_index]
		total_interval /= interval_cuts
		possible_prices = np.linspace(best_price, best_price + total_interval, num = interval_cuts, dtype = 'float32')

	min_index = np.argmin(np.square(loss_prices))
	print("The best price is: {0:.4f} with a loss of {1:.4f}".format(best_price, loss_prices[min_index]))

	# == Retrain on optimal price ==
	# model.utility_function = tf.identity
	model.pricebs = best_price

	if final_retrain:
		print("Start retrain with found price")
		K_train = 2 ** 16
		lrs_epoch = [(0.003, 8)]

		# Retrain on "real" price
		for lr, epoch in lrs_epoch:
			optimizer = tf.optimizers.Adam(lr)
			model.compile(loss = loss_choice, optimizer = optimizer)

			# Generate Data
			X_train = market_model.generate_instances(N = N, T = T, M = K_train)
			y_train = np.zeros((K_train, 1))

			model.fit(x = X_train, y = y_train, epochs = epoch, verbose = 0, batch_size = 2 ** 11)

	price_loss = model(market_model.generate_instances(N = N, T = T, M = 2 * K_train))

	return model, best_price, price_loss
