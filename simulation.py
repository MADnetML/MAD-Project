import numpy as np
import random
from scipy import sparse, signal


def Y_factory(s, Y_size, A_size, density, SNR=1):
    """
    Returns Measurement, Kernel and Activation map.
    :param s:
    :param Y_size:
    :param A_size:
    :param density:
    :param SNR:
    :return:
    """
    n1, n2 = Y_size
    m1, m2 = A_size
    A = kernel_factory(s, m1, m2)
    X = sparse.random(n1, n2, density)
    X = X / np.sum(X)
    Y = np.zeros([s, n1, n2])

    for level in range(s):
        Y[level] = signal.convolve(X.A, A[level], mode='same')
        eta = np.var(Y[level]) / SNR
        noise = np.random.normal(0, np.sqrt(eta), (n1, n2))
        Y[level] += noise

    return Y, A, X.toarray()


def kernel_factory(s, m1, m2):
    """
    Creates s different kernels of size m1 by m2
    :param s: Number of kernels
    :param m1: Number of rows
    :param m2: Number of columns
    :return: np.array shaped (s, m1, m2)
    """
    m_max = max(m1, m2)
    A = np.zeros([s, m_max, m_max], dtype=complex)
    symmetry = random.choice([2, 3, 4, 6])
    half_sym = np.floor(symmetry / 2).astype('int')
    lowest_k = 0.5
    highest_k = 3
    k = np.zeros([s, symmetry])
    for level in range(s):
        k[level, :] = np.random.uniform(lowest_k, highest_k, symmetry)

    x, y = np.meshgrid(np.linspace(-1, 1, m_max), np.linspace(-1, 1, m_max))
    # dist = np.sqrt(x * x + y * y)
    # theta = np.arctan(x / y)
    arb_angle = np.random.uniform(0, 2 * np.pi)
    for direction in range(symmetry):
        ang = direction * 180 / symmetry
        ang = arb_angle + ang * np.pi / 180
        r = (x * np.cos(ang) + np.sin(ang) * y)
        phi = np.random.uniform(0, 2 * np.pi)
        for i in range(s):
            A[i, :, :] += np.cos(2 * np.pi * k[i, direction % half_sym] * r)

    # Adding normal decay
    sigma = np.random.uniform(0.3, 0.6)
    decay = gaussian_window(m_max, m_max, sigma)
    A = np.multiply(np.abs(A), decay)
    # Normalizing:
    A = sphere_norm_by_layer(A)
    return A


def sphere_norm_by_layer(M):
    """
    Returns your matrix normalized to the unit sphere.
    :param M: A matrix with 2 or 3 dimensions
    :return: M normalized such that the sum of it's elements squared is one.
    """
    M_inner = M
    shape = np.shape(M_inner)
    if len(shape) == 3:
        for i in range(shape[0]):
            norm = np.sqrt(np.sum(M_inner[i, :, :] ** 2))
            M_inner[i, :, :] = M_inner[i, :, :] / norm
        return M_inner
    if len(shape) == 2:
        norm = np.sqrt(np.sum(M_inner ** 2))
        M_inner = M_inner / norm
    return M_inner


def gaussian_window(n1, n2, sig=1, mu=0):
    """
    Returns a n1 by n2 Gaussian window
    :param n1: number or rows
    :param n2: number of columns
    :param sig: STD of the Gaussian window
    :param mu: Center of the window in (-1, 1)
    :return: n1 by n2 np.array
    """
    x, y = np.meshgrid(np.linspace(-1, 1, n1), np.linspace(-1, 1, n2))
    d = np.sqrt(x * x + y * y)
    g = np.exp(-((d - mu) ** 2 / (2.0 * sig ** 2)))
    return g
