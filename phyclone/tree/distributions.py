import sys
import numpy as np

from phyclone.utils.math_utils import cached_log_factorial, log_sum_exp_over_dims


class FSCRPDistribution(object):
    """FSCRP prior distribution on trees."""

    __slots__ = ("_alpha", "log_alpha", "_c_const", "_smallest_alpha")

    def __init__(self, alpha, c_const=1000):
        self.alpha = alpha
        self.c_const = c_const
        self._smallest_alpha = sys.float_info.min

    def __eq__(self, other):
        alpha_check = self.alpha == other.alpha
        return alpha_check

    def __hash__(self):
        return hash(self.alpha)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if alpha == 0.0:
            alpha = self._smallest_alpha
        self._alpha = alpha
        self.log_alpha = np.log(alpha)

    @property
    def c_const(self):
        return self._c_const

    @c_const.setter
    def c_const(self, c_const):
        self._c_const = np.log(c_const)

    def log_p(self, tree):
        log_p, num_nodes = self.compute_CRP_prior(tree)

        log_p -= (num_nodes - 1) * np.log(num_nodes + 1)

        log_p -= tree.multiplicity

        return log_p

    def compute_CRP_prior(self, tree):
        tree_node_data = tree.node_data

        outlier_node_name = tree.outlier_node_name
        log_p = 0.0
        # CRP prior
        num_nodes = tree.get_number_of_nodes()
        log_p += num_nodes * self.log_alpha
        log_p += sum(cached_log_factorial(len(v) - 1) for k, v in tree_node_data.items() if k != outlier_node_name)
        return log_p, num_nodes

    def log_p_one(self, tree):
        # log_p, num_nodes = self.compute_CRP_prior(tree)
        log_p = self.log_p(tree)

        tree_roots = tree.roots

        r_term = self._compute_r_term(len(tree_roots), tree.get_number_of_nodes())

        num_ways = 0

        for root in tree_roots:
            curr_num_nodes = tree.get_number_of_descendants(root) + 1
            num_sub_trees = (curr_num_nodes - 1) * np.log(curr_num_nodes)
            num_ways += num_sub_trees

        log_p += -num_ways + r_term

        # log_p -= tree.multiplicity

        return log_p

    def _compute_z_term(self, num_roots, num_nodes):
        log_one = 0.0

        a_term = log_one * num_nodes

        la = log_one
        log_const = self.c_const

        if num_roots == 0:
            res = a_term
        else:

            r_term_numerator = log_one - (log_const * num_roots)
            r_term_denominator = log_one - (log_const * 1)

            r_term_numerator = la + np.log1p(-np.exp(r_term_numerator - la))
            r_term_denominator = la + np.log1p(-np.exp(r_term_denominator - la))

            res = a_term + (r_term_numerator - r_term_denominator)
        return res

    def _compute_r_term(self, num_roots, num_nodes):
        z_term = self._compute_z_term(num_roots, num_nodes)
        log_const = self.c_const

        log_one = 0.0

        if num_roots == 0:
            num_roots = 1

        return log_one - (z_term + (log_const * (num_roots - 1)))


class TreeJointDistribution(object):
    __slots__ = "prior", "outlier_modelling_active"

    def __init__(self, prior: FSCRPDistribution, outlier_modelling_active=False):
        self.prior = prior
        self.outlier_modelling_active = outlier_modelling_active

    def __eq__(self, other):
        return self.prior == other.prior

    def __hash__(self):
        return hash(self.prior)

    def log_p(self, tree):
        """The log likelihood of the data marginalized over root node parameters."""

        log_p = self.prior.log_p(tree)

        log_p += self.outlier_prior(tree)

        if tree.get_number_of_children(tree.root_node_name) > 0:
            log_p += log_sum_exp_over_dims(tree.data_log_likelihood)

        log_p += self.outlier_marginal_prob(tree)

        return log_p

    def log_p_one(self, tree):
        """The log likelihood of the data conditioned on the root having value 1.0 in all dimensions."""

        log_p_one = self.prior.log_p_one(tree)

        log_p_one += self.outlier_prior(tree)

        if tree.get_number_of_children(tree.root_node_name) > 0:
            log_p_one += tree.data_log_likelihood[:, -1].sum()

        log_p_one += self.outlier_marginal_prob(tree)

        return log_p_one

    def outlier_marginal_prob(self, tree):
        outliers_marginal_prob = sum(data_point.outlier_marginal_prob for data_point in tree.outliers)
        return outliers_marginal_prob

    def outlier_prior(self, tree):
        log_p = 0.0

        if self.outlier_modelling_active:
            outlier_node_name = tree.outlier_node_name
            tree_node_data = tree.node_data
            for node, node_data in tree_node_data.items():
                if node == outlier_node_name:
                    log_p += sum(data_point.outlier_prob for data_point in node_data)
                else:
                    log_p += sum(data_point.outlier_prob_not for data_point in node_data)
        return log_p
