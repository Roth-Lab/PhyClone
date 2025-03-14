"""
Created on 8 Dec 2016

@author: Andrew Roth
"""

import math
from functools import lru_cache

import numba
import numpy as np
from scipy.signal import fftconvolve


def bernoulli_rvs(rng: np.random.Generator, p=0.5):
    return rng.random() < p


def discrete_rvs(p, rng):
    p = p / np.sum(p)
    return rng.multinomial(1, p).argmax()


@numba.jit(nopython=True)
def simple_log_factorial(n, arr):
    idxs = np.nonzero(arr == -math.inf)[0]

    for i in idxs:
        if i > n:
            break
        if i == 0:
            arr[i] = np.log(1)
        else:
            arr[i] = np.log(i) + arr[i - 1]


@numba.jit(nopython=True)
def exp_normalize(log_p):
    """Normalize a vector numerically safely.

    Parameters
    ----------
    log_p: array_like (float)
        Unnormalized array of values in log space.

    Returns:
    -------
    p: array_like (float)
        Normalized array of values.
    log_norm: float
        Log normalization constant.
    """
    log_norm = log_sum_exp(log_p)

    p = np.exp(log_p - log_norm)

    # p = p / p.sum()
    p /= p.sum()

    return p, log_norm


@numba.jit("float64(float64[:])", nopython=True, fastmath=True)
def log_sum_exp(log_X):
    """Given a list of values in log space, log_X. Compute exp(log_X[0] + log_X[1] + ... log_X[n])

    This implementation is numerically safer than the naive method.
    """
    max_exp = log_X.max()

    if np.isinf(max_exp):
        return max_exp

    total = 0.0
    for x in log_X:
        total += np.exp(x - max_exp)

    return np.log(total) + max_exp


@numba.jit("float64(float64[:, ::1])", nopython=True, fastmath=True)
def log_sum_exp_over_dims(log_x_arr):

    sum_total = 0.0

    for dim in log_x_arr:
        sum_total += log_sum_exp(dim)

    return sum_total


@numba.jit(nopython=True, fastmath=False)
def log_sum_exp_over_dims_to_arr(log_x_arr):
    num_dims = log_x_arr.shape[0]
    ret_arr = np.empty(num_dims, np.float64)

    for dim, log_x_dim in enumerate(log_x_arr):
        ret_arr[dim] = log_sum_exp(log_x_dim)

    return ret_arr


@numba.jit(nopython=True)
def log_normalize(log_p):
    return log_p - log_sum_exp(log_p)


@numba.vectorize()
def log_gamma(x):
    return math.lgamma(x)


@numba.jit(nopython=True)
def log_beta(a, b):
    if a <= 0 or b <= 0:
        return -np.inf

    return log_gamma(a) + log_gamma(b) - log_gamma(a + b)


@numba.jit(nopython=True)
def log_factorial(x):
    return log_gamma(x + 1)


@lru_cache(maxsize=None)
def cached_log_factorial(x):
    return log_factorial(x)


@numba.jit(nopython=True)
def log_binomial_coefficient(n, x):
    return log_factorial(n) - log_factorial(x) - log_factorial(n - x)


@lru_cache(maxsize=2048)
def cached_log_binomial_coefficient(n, x):
    return log_binomial_coefficient(n, x)


def log_multinomial_coefficient(x):
    """Compute the multinomial coefficient.

    Parameters
    ----------
    x: list
        The number of elements in each category.
    """
    if len(x) == 0:
        return 0

    n = sum(x)

    result = cached_log_factorial(n)

    result -= sum(map(cached_log_factorial, x))

    return result


@numba.jit(nopython=True)
def log_beta_binomial_likelihood(n, x, a, b):
    return log_beta(a + x, b + n - x) - log_beta(a, b)


@numba.jit(nopython=True)
def log_binomial_likelihood(n, x, p):
    if p == 0:
        if x == 0:
            return 0
        else:
            return -np.inf

    if p == 1:
        if x == n:
            return 0
        else:
            return -np.inf

    return x * np.log(p) + (n - x) * np.log(1 - p)


@numba.jit(nopython=True)
def log_binomial_pdf(n, x, p):
    return log_binomial_coefficient(n, x) + log_binomial_likelihood(n, x, p)


@numba.jit(nopython=True)
def log_beta_binomial_pdf(n, x, a, b):
    return log_binomial_coefficient(n, x) + log_beta_binomial_likelihood(n, x, a, b)


@numba.jit("float64[:, ::1](float64[:, ::1], float64[:, ::1], float64[:, ::1])", nopython=True, fastmath=True)
def conv_over_dims(log_x_arr, log_y_arr, ans_arr):
    """Direct convolution in numba-time."""

    n = log_x_arr.shape[-1]
    dims = log_x_arr.shape[0]
    m = n + 1
    log_y_arr = np.ascontiguousarray(log_y_arr[..., ::-1])

    for l in range(dims):
        log_x = log_x_arr[l]
        log_y = log_y_arr[l]
        ans = ans_arr[l]
        for k in range(1, m):
            for j in range(k):
                ans[k - 1] += log_x[j] * log_y[n - (k - j)]

    return ans_arr


def fft_convolve_two_children(child_1, child_2):
    """FFT convolution"""

    result = fftconvolve(child_1, child_2, axes=[-1])

    result = result[..., : child_1.shape[-1]]

    return result


@numba.jit(nopython=True)
def log_pyclone_beta_binomial_pdf(data, f, s):
    t = data.t

    C = len(data.cn)

    population_prior = np.zeros(3)
    population_prior[0] = 1 - t
    population_prior[1] = t * (1 - f)
    population_prior[2] = t * f

    ll = np.ones(C, dtype=np.float64) * np.inf * -1

    for c in range(C):
        e_vaf = 0

        norm_const = 0

        for i in range(3):
            e_cn = population_prior[i] * data.cn[c, i]

            e_vaf += e_cn * data.mu[c, i]

            norm_const += e_cn

        e_vaf /= norm_const

        a = e_vaf * s

        b = s - a

        ll[c] = data.log_pi[c] + log_beta_binomial_pdf(data.a + data.b, data.b, a, b)

    return log_sum_exp(ll)


@numba.jit(nopython=True)
def log_pyclone_binomial_pdf(data, f):
    t = data.t

    C = len(data.cn)

    population_prior = np.zeros(3)
    population_prior[0] = 1 - t
    population_prior[1] = t * (1 - f)
    population_prior[2] = t * f

    ll = np.ones(C, dtype=np.float64) * np.inf * -1

    for c in range(C):
        e_vaf = 0

        norm_const = 0

        for i in range(3):
            e_cn = population_prior[i] * data.cn[c, i]

            e_vaf += e_cn * data.mu[c, i]

            norm_const += e_cn

        e_vaf /= norm_const

        ll[c] = data.log_pi[c] + log_binomial_pdf(data.a + data.b, data.b, e_vaf)

    return log_sum_exp(ll)


def np_conv_dims(child_1, child_2):
    num_dims = child_1.shape[0]

    grid_size = child_1.shape[-1]

    arr_list = [np.convolve(child_2[i, :], child_1[i, :])[:grid_size] for i in range(num_dims)]

    return np.ascontiguousarray(arr_list)


def _np_conv_dims(child_1, child_2):
    num_dims = child_1.shape[0]

    child_1_maxes = np.max(child_1, axis=-1, keepdims=True)

    child_2_maxes = np.max(child_2, axis=-1, keepdims=True)

    child_1_norm = np.exp(child_1 - child_1_maxes)

    child_2_norm = np.exp(child_2 - child_2_maxes)

    grid_size = child_1.shape[-1]

    arr_list = [np.convolve(child_2_norm[i, :], child_1_norm[i, :])[:grid_size] for i in range(num_dims)]

    log_D = np.ascontiguousarray(arr_list)

    log_D[log_D <= 0] = 1e-100

    log_D = np.log(log_D, order="C", dtype=np.float64, out=log_D)

    log_D += child_1_maxes

    log_D += child_2_maxes

    return np.ascontiguousarray(log_D)
