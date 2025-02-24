import numpy as np
from phyclone.utils.math import log_sum_exp_over_dims_to_arr


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

        self.value = np.ascontiguousarray(value)

        if name is None:
            name = idx

        self.name = name

        self.outlier_prob = outlier_prob

        self.outlier_prob_not = outlier_prob_not

        log_prior = -np.log(value.shape[1])
        sub_comp = self.value + log_prior
        sub_comp = np.logaddexp.accumulate(sub_comp, out=sub_comp, axis=-1)
        sub_comp += log_prior
        sub_comp = np.ascontiguousarray(sub_comp)
        self.outlier_marginal_prob = log_sum_exp_over_dims_to_arr(sub_comp).sum()


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
