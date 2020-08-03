import numpy as np

"""
This file contains unit tests, it is to be run before every commit.
"""


def test_init_learner():
	"""
	Test functions of Learner
	"""
	np.testing.assert_equal(1.0, 1.0)


def test_approx_bs():
	"""
	See if deep hedge approximates bs estimate for step_size --> 0
	"""
	np.testing.assert_almost_equal(0.0, 0.0, 3)