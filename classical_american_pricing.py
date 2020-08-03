import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from backend.simulation import BlackScholesModel
import matplotlib.pyplot as plt

"""
This file calculates the price of an american option using classical monte carlo simulation and backwards induction.
This approach will of course not give the same price as the deep method (in the limit), since this method does not
use utility functions and so weighs risk differently.
"""


# Define parameters
N = 20 + 1
T = 20 / 365
K = 100.

num_sim = 500000


# Define Value function
def call_option(t, x, strike = K):
	return np.maximum(0, x - strike)


# Define Black Scholes model
BM = BlackScholesModel(mu = np.array([-0.6]),
					   sigma2 = np.array([[0.2 ** 2]]),
					   S0 = np.array([100.]),
					   N = N, T = T)

# Make some data
stock = BM.generate_instances(M = num_sim).reshape((num_sim, N))
exercise = call_option(None, stock)

# Define out sk-learn models (one per timestep except last)
pre_poly = PolynomialFeatures(degree = 12)
model = [LinearRegression() for _ in range(N-1)]

# Next we use backwards induction to calculate the price
values = np.zeros_like(stock)

# Set the last value equal to the last exercise
values[:, -1] = exercise[:, -1]
for i in range(N-2, -1, -1):
	print("\rLayer " + str(i), end = '')
	x_poly = pre_poly.fit_transform(stock[:, i, np.newaxis])
	model[i].fit(x_poly, values[:, i + 1])
	c_hat = model[i].predict(x_poly)
	# Take the maximum of the estimated value next timestep and exercising now
	binary = (np.sign(c_hat - exercise[:, i]) + 1)/2
	values[:, i] = binary * values[:, i + 1] + (1 - binary) * exercise[:, i]

# Expected value of stopping
# stock = BM.generate_instances(M = 100000).reshape((num_sim, N))
# exercise = call_option(None, stock)
# x_poly = x_poly = pre_poly.fit_transform(stock[:, i, np.newaxis])


print("\n Expected Value of Stopping")

# Printing stats
print("\nStats:")
print("Price:     " + str(np.mean(values[:, 0])))
print("Variance:  " + str(np.var(values[:, 0])))
print("Min:       " + str(np.min(values[:, 0])))
print("Max:       " + str(np.max(values[:, 0])))

x = np.linspace(85, 120, num = 200).reshape((200, 1))
x_poly = pre_poly.fit_transform(x)

print(x_poly[100, :])

for k in range(N-1):
	if k % (N//5) == 0:
		plt.plot(x, model[k].predict(x_poly), label = str(k))
		plt.legend()

plt.show()