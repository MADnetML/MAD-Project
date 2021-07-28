from simulation import Y_factory
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


def Y_plot(kernel, activation_map, result):
    """
    Plotting 2 kernels, one activation map and 2 results in one figure
    :param kernel: matrix with shape (s, m1, m2)
    :param activation_map: matrix with shape (n1, n2)
    :param result: matrix with shape (s, n1, n2)
    :return: Nothing
    """
    fig = plt.figure()
    gs = GridSpec(2, 3, width_ratios=[1, 2, 1])
    ker_top_ax = fig.add_subplot(gs[0, 0])
    ker_bot_ax = fig.add_subplot(gs[1, 0])
    activ_ax = fig.add_subplot(gs[:, 1])
    result_top_ax = fig.add_subplot(gs[0, 2])
    result_bot_ax = fig.add_subplot(gs[1, 2])
    ker_top_ax.imshow(kernel[0])
    ker_bot_ax.imshow(kernel[1])
    activ_ax.imshow(activation_map)
    result_top_ax.imshow(result[0])
    result_bot_ax.imshow(result[1])
    plt.tight_layout()
    plt.show()


# testing Y_factory
levels = 3
density = 0.005
SNR = 200
n1, n2 = 185, 185
m1, m2 = 25, 25
Y, A, X = Y_factory(levels, (n1, n2), (m1, m2), density, SNR)

Y_plot(A, X.toarray(), Y)
