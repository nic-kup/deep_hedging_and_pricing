from backend.neural_network import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

plt.rcParams.update({'font.size': 14})

# mpl.rcParams['figure.dpi'] = 200
# mpl.rcParams['savefig.dpi'] = 300


def hm_plot_example(model, market_model, N, T, num_examples = 5):
    time = np.linspace(0, T, num = N)
    for _ in range(num_examples):
        S = market_model.tf_generate_instances(N = N, T = T, M = 1)
        print("Model Call: " + str(model(S)))

        if isinstance(model, OtherLearner):
            delta, hidden = model.get_hedge(S)
        else:
            delta = model.get_hedge(S)

        fig, ax1 = plt.subplots(2, figsize = (10, 8))
        color = 'tab:red'
        ax1[0].set_xlabel('time')
        ax1[0].set_ylabel('Stock', color = color)
        ax1[0].plot(time, S[0, :, 0].numpy(), color = color)
        ax1[0].tick_params(axis = 'y', labelcolor = color)

        ax1[1].set_xlabel('time')
        ax1[1].set_ylabel('Volatility', color = color)
        ax1[1].plot(time, S[0, :, 1].numpy(), color = color)
        ax1[1].tick_params(axis = 'y', labelcolor = color)

        ax2 = [None] * 2
        ax2[0] = ax1[0].twinx()  # instantiate a second axes that shares the same x-axis
        ax2[1] = ax1[1].twinx()

        color = 'tab:blue'
        ax2[0].set_ylabel('Trades', color = color)  # we already handled the x-label with ax1
        ax2[0].plot(time, delta[0, :, 0].squeeze(), color = color)
        ax2[0].tick_params(axis = 'y', labelcolor = color)

        ax2[1].set_ylabel('Trades', color = color)  # we already handled the x-label with ax1
        ax2[1].plot(time, delta[0, :, 1].squeeze(), color = color)
        ax2[1].tick_params(axis = 'y', labelcolor = color)

        fig.tight_layout()
        plt.show()

        if isinstance(model, OtherLearner) and model.hidden_size == 1:
            ax3 = ax1[0].twinx()
            ax3.spines['right'].set_position(('axes', 1.2))
            ax3.spines['right'].set_visible(True)

            color = 'tab:green'
            ax3.set_ylabel('Hidden', color = color)  # we already handled the x-label with ax1
            ax3.plot(time, hidden.squeeze(), color = color, ls = '--')
            ax3.tick_params(axis = 'y', labelcolor = color)

        fig.tight_layout()
        plt.show()


def plot_example(model, market_model, N, T, num_examples = 5):
    time = np.linspace(0, T, num = N)
    for _ in range(num_examples):
        S = market_model.tf_generate_instances(N = N, T = T, M = 1)
        print("Model Call: " + str(model(S)))

        if isinstance(model, OtherLearner):
            delta, hidden = model.get_hedge(S)
        else:
            delta = model.get_hedge(S)

        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('time')
        ax1.set_ylabel('Stock', color = color)
        ax1.plot(time, S[0, :, 0].numpy(), color = color)
        ax1.tick_params(axis = 'y', labelcolor = color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Trades', color = color)  # we already handled the x-label with ax1
        ax2.plot(time, delta.squeeze(), color = color)
        ax2.tick_params(axis = 'y', labelcolor = color)

        if isinstance(model, OtherLearner) and model.hidden_size == 1:
            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('axes', 1.2))
            ax3.spines['right'].set_visible(True)

            color = 'tab:green'
            ax3.set_ylabel('Hidden', color = color)  # we already handled the x-label with ax1
            ax3.plot(time, hidden.squeeze(), color = color, ls = '--')
            ax3.tick_params(axis = 'y', labelcolor = color)

        fig.tight_layout()
        plt.show()


def plot_multiple_examples(models, market_model, N, T, num_examples = 5):
    time = np.linspace(0, T, num = N)
    for _ in range(num_examples):
        S = market_model.tf_generate_instances(N = N, T = T, M = 1)

        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('time')
        ax1.set_ylabel('Stock', color = color)
        ax1.plot(time, S[0, :, 0].numpy(), color = color)
        ax1.tick_params(axis = 'y', labelcolor = color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        for model in models:
            if isinstance(model, OtherLearner):
                delta, hidden = model.get_hedge(S)
            else:
                delta = model.get_hedge(S)

            ax2.set_ylabel('Trades')  # we already handled the x-label with ax1
            ax2.plot(time, delta.squeeze(), label = model.name)
            ax2.tick_params(axis = 'y')

        ax2.legend()
        fig.tight_layout()
        plt.show()


def plot_strategy(model, approx_fidelity = 300, compare = None):
    if isinstance(model, Learner):
        # Trading surfaces

        # Choose trading day
        for k in [5, 10, 15]:
            # Setting variables for plotting
            S = np.linspace(85, 115, approx_fidelity, dtype = np.float32).reshape((approx_fidelity, 1))
            delta = np.linspace(0.0, 0.9, approx_fidelity, dtype = np.float32)
            res = np.zeros((approx_fidelity, approx_fidelity))

            # Calculating Approximate Price
            for i in range(approx_fidelity):
                if isinstance(model, OtherLearner):
                    res[:, i] = model.call_only_k(S,
                                                  delta[i] * np.ones((approx_fidelity, 1)),
                                                  np.zeros((approx_fidelity, model.hidden_size)),
                                                  # hidden layer = zeros
                                                  k).numpy().squeeze()
                else:
                    res[:, i] = model.call_only_k(S,
                                                  delta[i] * np.ones((approx_fidelity, 1)),
                                                  k).numpy().squeeze()

            # Plotting
            fig = plt.figure()
            ax = plt.axes(projection = '3d')

            ax.plot_surface(S, delta, res, cmap = 'viridis', edgecolor = 'none')
            ax.set_title('Strategy surface for k = ' + str(k))
            plt.show()
    else:
        for k in [5, 10, 15]:
            # Calculating True Price
            S = np.linspace(80, 120, approx_fidelity, dtype = np.float32).reshape((approx_fidelity, 1))

            # Setting delta to 0 strategy (only needed for Learner)
            delta = np.ones((approx_fidelity, 1)) * 0.5

            # Calculating Approximate Price
            approx_delta = model.call_only_k(S, k).numpy().squeeze()

            if compare == "call":
                S = np.linspace(80, 120, approx_fidelity, dtype = np.float32).reshape((approx_fidelity, 1))
                call_delta = cts_prices(bs_call_delta_ptws, S, 100., 0.2, np.ones_like(S)*k/365, 20/365)
                plt.plot(S.squeeze(), call_delta.squeeze())
            elif compare == "digi":
                S = np.linspace(80, 120, approx_fidelity, dtype = np.float32).reshape((approx_fidelity, 1))
                call_delta = cts_prices(bs_digi_delta_ptws, S, 100., 0.2, np.ones_like(S) * k / 365, 20 / 365)
                plt.plot(S.squeeze(), call_delta.squeeze())

            # Plotting
            plt.plot(S.squeeze(), approx_delta.squeeze())
            plt.title("Delta of " + str(k))
            plt.show()


def plot_2d_hist(model, market_model, N, T, bins = 100):
    S = market_model.tf_generate_instances(N = N, T = T, M = 100000)
    x = model.get_pl(S).numpy().squeeze()
    y = model.option(S).numpy().squeeze()

    plt.hist2d(x, y, bins = bins)

    plt.title("Density for PL vs Option Value")
    plt.xlabel("Profits and Losses")
    plt.ylabel("Option Value at Terminal time")
    plt.show()


def plot_2d_hist_digi(model, market_model, N, T, bins = 100):
    S = market_model.tf_generate_instances(N = N, T = T, M = 100000)
    x = model.get_pl(S).numpy().squeeze()
    y = model.option(S).numpy().squeeze()

    inds = np.where(y < 0.5)[0]

    plt.hist(x[inds], bins = 'sqrt', density = True, color = 'blue', label = 'opt = 0', alpha = 0.5)
    plt.hist(x[-inds], bins = 'sqrt', density = True, color = 'green', label = 'opt = 1', alpha = 0.5)
    plt.legend()

    plt.title("Density for PL vs Option Value")
    plt.xlabel("Profits and Losses")
    plt.ylabel("Density")
    plt.show()


def plot_2d_hist_call(model, market_model, N, T, bins = 100, compare = None):
    S = market_model.tf_generate_instances(N = N, T = T, M = 100000)
    x = model.get_pl(S).numpy().squeeze()
    y = model.option(S).numpy().squeeze()

    inds = np.where(y >= 0.0001)[0]

    plt.hist2d(x[inds], y[inds], bins = bins)

    plt.title("Density of PL vs Option Value for Option Value > 0")
    plt.xlabel("Profits and Losses")
    plt.ylabel("Option Value at Terminal time")
    plt.show()

    plt.hist(x[-inds], bins = 'sqrt', density = True)
    plt.title("Density of PL for Option Value = 0")
    plt.xlabel("Profits and Losses")
    plt.show()


def plot_2d_hist_alt(model, market_model, N, T, bins = 100):
    S = market_model.tf_generate_instances(N = N, T = T, M = 100000)
    x = model.get_pl(S).numpy().squeeze()
    y = S[:, N - 1, 0].numpy().squeeze()

    plt.hist2d(x, y, bins = bins)

    plt.title("Density of PL vs terminal stock price")
    plt.xlabel("Profits and Losses")
    plt.ylabel("Stock price at terminal time")
    plt.show()