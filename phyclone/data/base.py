import numpy as np
from scipy.special import logsumexp as log_sum_exp

from phyclone.tree.utils import _sub_compute_S


class DataPoint(object):
    __slots__ = (
        "idx",
        "value",
        "name",
        "outlier_prob",
        "outlier_marginal_prob",
        "outlier_prob_not",
    )

    def __init__(self, idx, value, name=None, outlier_prob=0, outlier_prob_not=1):
        self.idx = idx

        self.value = value

        if name is None:
            name = idx

        self.name = name

        self.outlier_prob = outlier_prob

        self.outlier_prob_not = outlier_prob_not

        log_prior = -np.log(value.shape[1])

        tmp = self.value + log_prior

        sub_comp = _sub_compute_S(tmp)

        self.outlier_marginal_prob = np.sum(log_sum_exp(sub_comp + log_prior, axis=1))

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        self_key = self.name

        other_key = other.name

        return self_key == other_key

    @property
    def grid_size(self):
        return self.shape

    @property
    def shape(self):
        return self.value.shape
