from backend.simulation import *

"""
This file is for unit tests, to be run before every commit.
"""



def test_corr_gaussians():
    rho = 0.2
    T = 0.5
    N = 1000000
    results = corr_gaussians(rho = rho, T = T, N = N, M = 1)
    results = np.sum(np.diff(results[:, :, 0], 1) * np.diff(results[:, :, 1], 1))
    np.testing.assert_almost_equal(results, rho * T, 3)


def test_generate_hes():
    M = 100000
    N = 30
    T = 0.4
    hes = HestonModel(
        mu = 0.0,
        a = 1.0,
        b = 0.04,
        rho = -0.7,
        sigma = 0.2,
        V0 = 0.04,
        S0 = 100.)

    results = hes.generate_instances(N = N, T = T, M = M)[:, -1, 0]
    np.testing.assert_almost_equal(np.mean(results), hes.S0, 2)
    # TODO: Implement variance check


def test_stoch_int():
    H = np.array([1.0, 1.0, 1.0, 1.0])
    S = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(stoch_int(H, S), np.array([0., 1., 2., 3.]))


def test_generate_gbm():
    num_sims = 1000000
    N = 20
    T = 1.4
    BS = BlackScholesModel(
        mu = np.array([0.2]),
        sigma2 = np.array([[0.2 ** 2]]),
        S0 = np.array([1.2]),
        N = N,
        T = T
    )

    results = BS.generate_instances(M = num_sims)[:, N - 1, :]
    print(results.shape)
    np.testing.assert_almost_equal(np.mean(results), BS.S0 * np.exp(BS.mu * T), decimal = 2)
    np.testing.assert_almost_equal(np.var(results), BS.S0 ** 2 * np.exp(2 * BS.mu * T) * (np.exp(BS.sigma2 * T) - 1),
                                   decimal = 2)


def test_generate_gbm_2d():
    num_sims = 500000
    N = 25
    T = 0.8
    BS = BlackScholesModel(
        S0 = np.array([1.2, 0.8]),
        mu = np.array([0.5, -0.2]),
        sigma2 = np.array([[0.21, 0.02], [0.02, 0.2]]),
        N = N, T = T
    )
    results = BS.generate_instances(M = num_sims)[:, N - 1, :]
    np.testing.assert_almost_equal(np.mean(results, 0), BS.S0 * np.exp(BS.mu * T), decimal = 2)
    # Cov(S^i, S^j) = S0i * S0j * exp((mu^i + mu^j) T)
    #       * (exp((1/2) * sum_k{ sigma_ik sigma_jk t} - 1)
    np.testing.assert_almost_equal(np.cov(results.T), np.outer(BS.S0, BS.S0)
                                   * np.outer(np.exp(BS.mu * T), np.exp(BS.mu * T))
                                   * (np.exp(T * BS.sigma2) - 1), decimal = 2)


def test_tf_diff():
    X = tf.Variable([0., 1., 2., 3.], trainable = False)
    tf.debugging.assert_equal(tf_diff(X), tf.Variable([1., 1., 1.]))


def test_tf_stoch_int():
    H = tf.Variable([1.0, 1.0, 1.0, 1.0])
    S = tf.Variable([1.0, 2.0, 3.0, 4.0])
    tf.debugging.assert_equal(tf_stoch_int(H, S), tf.Variable([0., 1., 2., 3.]))


def test_transaction_costs():
    def c(x):
        return np.minimum(np.absolute(x), 1.)

    delta = np.array([0., 0.2, 0.5, -1., 1.])
    np.testing.assert_equal(transaction_costs(c, delta), 2.5)
