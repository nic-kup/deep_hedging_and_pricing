import numpy as np
import tensorflow as tf

"This file contains an abstract class for all other market models to inherit."

class BaseModel():

	# A function to generate instances of the model
	# Takes as parameters
	# N: number of
	def __init__(self):
		self.pref_M = None
		self.T = None
		self.N = None

	def generate_instances(self, N: int, T, M: int = 1):
		assert "Not implemented"
		return 0

	def tf_generate_instances(self, N = None, T = None, M: int = 1):
		if N is None:
			N = self.N
		if T is None:
			T = self.T
		if M is None:
			M = self.pref_M
		return tf.Variable(self.generate_instances(N, T, M), trainable = False, dtype = tf.dtypes.float32)

	def get_parameters(self):
		assert "Not implemented"
		return 0
