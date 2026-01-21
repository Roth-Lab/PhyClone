import unittest

import numpy as np

from phyclone.smc.swarm import TreeHolder
from phyclone.tree.tree_shell_node_adder import TreeShellNodeAdder
from phyclone.tests.utilities.simulate import simulate_binomial_data
from phyclone.tree import FSCRPDistribution, Tree, TreeJointDistribution
from phyclone.tree.utils import get_clades


class TestTreeShellNodeAdder(unittest.TestCase):

    def setUp(self):
        self.outlier_prob = 0.0

        self._rng = np.random.default_rng(12345)

        # self.tree_dist = FSCRPDistribution(1.0)

        self.tree_dist = TreeJointDistribution(FSCRPDistribution(1.0), outlier_modelling_active=True)

    def build_cherry_tree(self, tree_class, data=None):
        if data is None:
            data = self.build_six_datapoints()

        tree = tree_class.get_single_node_tree(data[:2])

        exp_n_1 = tree.create_root_node(children=[], data=data[2:4])

        expected_tree_roots = tree.roots

        exp_n_2 = tree.create_root_node(children=expected_tree_roots, data=data[4:])

        return tree

    def build_linear_tree(self, tree_class, data=None):
        if data is None:
            data = self.build_six_datapoints()

        tree = tree_class.get_single_node_tree(data[:2])

        expected_tree_roots = tree.roots

        exp_n_1 = tree.create_root_node(children=expected_tree_roots, data=data[2:4])

        expected_tree_roots = tree.roots

        exp_n_2 = tree.create_root_node(children=[exp_n_1], data=data[4:])

        return tree

    def build_six_datapoints(self):
        n = 100
        p = 1.0
        data = self._create_data_points(6, n, p)
        return data

    def _create_new_node_no_children(self, expected_tree, n, p, tree_shell):
        new_data = self._create_data_points(1, n, p)
        expected_tree.create_root_node(children=[], data=new_data)
        expected_tree_holder = TreeHolder(expected_tree, self.tree_dist, None)
        actual_tree_builder = tree_shell.create_tree_holder_with_new_node(None, new_data[0])
        actual_tree_holder = actual_tree_builder.build()
        return actual_tree_builder, actual_tree_holder, expected_tree_holder

    def _add_a_dp_to_outliers(self, expected_tree, n, p, tree_shell, start_idx=0):
        new_data = self._create_data_points(1, n, p, start_idx=start_idx)
        expected_tree.add_data_point_to_outliers(new_data[0])
        expected_tree_holder = TreeHolder(expected_tree, self.tree_dist, None)
        actual_tree_builder = tree_shell.create_tree_holder_with_datapoint_added_to_outliers(new_data[0])
        actual_tree_holder = actual_tree_builder.build()
        return actual_tree_builder, actual_tree_holder, expected_tree_holder

    def _run_asserts(self, actual_tree_builder, actual_tree_holder, expected_tree, expected_tree_holder):
        self.assertEqual(actual_tree_builder.multiplicity, expected_tree.multiplicity)
        self.assertEqual(expected_tree_holder, actual_tree_holder)
        self.assertEqual(expected_tree_holder.log_p, actual_tree_holder.log_p)
        self.assertEqual(expected_tree_holder.tree, actual_tree_holder.tree)
        self.assertEqual(expected_tree_holder.log_p_one, actual_tree_holder.log_p_one)
        self.assertEqual(expected_tree_holder.log_pdf, actual_tree_holder.log_pdf)
        np.testing.assert_array_equal(expected_tree_holder.tree_roots.sort(), actual_tree_holder.tree_roots.sort())
        np.testing.assert_array_equal(expected_tree_holder.tree_nodes.sort(), actual_tree_holder.tree_nodes.sort())
        self.assertListEqual(expected_tree.outliers, list(actual_tree_builder.outliers))
        self.assertEqual(self.tree_dist.log_p(expected_tree), self.tree_dist.log_p(actual_tree_builder))
        self.assertEqual(self.tree_dist.log_p_one(expected_tree), self.tree_dist.log_p_one(actual_tree_builder))

    def test_zero_node_tree_hash_no_outliers(self):
        grid_size = (10, 101)
        actual_tree = Tree(grid_size)
        tree_shell = TreeShellNodeAdder(actual_tree, self.tree_dist)
        self.assertEqual(hash(tree_shell), hash(actual_tree))

    def test_zero_node_tree_hash_one_outlier_dp(self):
        n = 100
        p = [1.0] * 10

        data = self._create_data_points(1, n, p)
        grid_size = data[0].grid_size
        actual_tree = Tree(grid_size)
        actual_tree.add_data_point_to_outliers(data[0])
        tree_shell = TreeShellNodeAdder(actual_tree, self.tree_dist)
        self.assertEqual(hash(tree_shell), hash(actual_tree))

    def test_zero_node_tree_hash_three_outlier_dps(self):
        n = 100
        p = [1.0] * 10

        data = self._create_data_points(3, n, p)
        grid_size = data[0].grid_size
        expected = Tree(grid_size)
        for dp in data:
            expected.add_data_point_to_outliers(dp)
        tree_shell = TreeShellNodeAdder(expected, self.tree_dist)
        self.assertEqual(hash(tree_shell), hash(expected))

    def test_zero_node_tree_no_outliers_create_node(self):
        grid_size = (10, 101)
        expected_tree = Tree(grid_size)
        tree_shell = TreeShellNodeAdder(expected_tree, self.tree_dist)

        n = 100
        p = [1.0] * 10
        actual_tree_builder, actual_tree_holder, expected_tree_holder = self._create_new_node_no_children(
            expected_tree, n, p, tree_shell
        )

        self._run_asserts(actual_tree_builder, actual_tree_holder, expected_tree, expected_tree_holder)

    def test_zero_node_tree_no_outliers_add_to_outliers(self):
        grid_size = (10, 101)
        expected_tree = Tree(grid_size)
        tree_shell = TreeShellNodeAdder(expected_tree, self.tree_dist)

        n = 100
        p = [1.0] * 10

        actual_tree_builder, actual_tree_holder, expected_tree_holder = self._add_a_dp_to_outliers(
            expected_tree, n, p, tree_shell
        )

        self._run_asserts(actual_tree_builder, actual_tree_holder, expected_tree, expected_tree_holder)

    def test_zero_node_tree_one_outlier_dp_create_node(self):
        n = 100
        p = [1.0] * 10

        data = self._create_data_points(1, n, p)
        grid_size = data[0].grid_size
        expected_tree = Tree(grid_size)
        expected_tree.add_data_point_to_outliers(data[0])
        tree_shell = TreeShellNodeAdder(expected_tree, self.tree_dist)

        actual_tree_builder, actual_tree_holder, expected_tree_holder = self._create_new_node_no_children(
            expected_tree, n, p, tree_shell
        )

        self._run_asserts(actual_tree_builder, actual_tree_holder, expected_tree, expected_tree_holder)

    def test_zero_node_tree_one_outlier_add_to_outliers(self):
        n = 100
        p = [1.0] * 10

        data = self._create_data_points(1, n, p)
        grid_size = data[0].grid_size
        expected_tree = Tree(grid_size)
        expected_tree.add_data_point_to_outliers(data[0])
        tree_shell = TreeShellNodeAdder(expected_tree, self.tree_dist)

        actual_tree_builder, actual_tree_holder, expected_tree_holder = self._add_a_dp_to_outliers(
            expected_tree, n, p, tree_shell, 1
        )

        self._run_asserts(actual_tree_builder, actual_tree_holder, expected_tree, expected_tree_holder)

    def test_zero_node_tree_three_outlier_dps_create_node(self):
        n = 100
        p = [1.0] * 10

        data = self._create_data_points(3, n, p)
        grid_size = data[0].grid_size
        expected_tree = Tree(grid_size)
        for dp in data:
            expected_tree.add_data_point_to_outliers(dp)
        tree_shell = TreeShellNodeAdder(expected_tree, self.tree_dist)

        actual_tree_builder, actual_tree_holder, expected_tree_holder = self._create_new_node_no_children(
            expected_tree, n, p, tree_shell
        )

        self._run_asserts(actual_tree_builder, actual_tree_holder, expected_tree, expected_tree_holder)

    def test_zero_node_tree_three_outlier_dps_add_to_outliers(self):
        n = 100
        p = [1.0] * 10

        data = self._create_data_points(3, n, p)
        grid_size = data[0].grid_size
        expected_tree = Tree(grid_size)
        for dp in data:
            expected_tree.add_data_point_to_outliers(dp)
        tree_shell = TreeShellNodeAdder(expected_tree, self.tree_dist)

        actual_tree_builder, actual_tree_holder, expected_tree_holder = self._add_a_dp_to_outliers(
            expected_tree, n, p, tree_shell, 3
        )

        self._run_asserts(actual_tree_builder, actual_tree_holder, expected_tree, expected_tree_holder)

    def test_single_node_tree_hash(self):
        n = 100
        p = 1.0

        data = self._create_data_points(2, n, p)

        actual_tree_built = Tree.get_single_node_tree(data)

        tree_shell = TreeShellNodeAdder(actual_tree_built, self.tree_dist)

        self.assertEqual(hash(tree_shell), hash(actual_tree_built))

    def test_single_node_tree_add_node_to_root(self):
        n = 100
        p = 1.0

        data = self._create_data_points(3, n, p)

        expected_tree = Tree.get_single_node_tree(data[:-1])

        tree_shell = TreeShellNodeAdder(expected_tree, self.tree_dist)

        expected_tree.create_root_node(children=[], data=[data[-1]])

        actual_tree_builder = tree_shell.create_tree_holder_with_new_node(children=[], datapoint=data[-1])

        actual_tree_holder = actual_tree_builder.build()

        expected_tree_holder = TreeHolder(expected_tree, self.tree_dist, None)

        self._run_asserts(actual_tree_builder, actual_tree_holder, expected_tree, expected_tree_holder)

    def test_single_node_tree_add_to_outliers(self):
        n = 100
        p = 1.0

        data = self._create_data_points(3, n, p)

        expected_tree = Tree.get_single_node_tree(data[:-1])

        tree_shell = TreeShellNodeAdder(expected_tree, self.tree_dist)

        actual_tree_builder, actual_tree_holder, expected_tree_holder = self._add_a_dp_to_outliers(
            expected_tree, n, p, tree_shell, 3
        )

        self._run_asserts(actual_tree_builder, actual_tree_holder, expected_tree, expected_tree_holder)

    def test_cherry_tree_add_node_to_root(self):
        n = 100
        p = 1.0

        data = self._create_data_points(7, n, p)

        expected_tree = self.build_cherry_tree(Tree, data[:-1])

        tree_shell = TreeShellNodeAdder(expected_tree, self.tree_dist)

        expected_tree.create_root_node(children=[], data=[data[-1]])

        actual_tree_builder = tree_shell.create_tree_holder_with_new_node(children=[], datapoint=data[-1])

        actual_tree_holder = actual_tree_builder.build()

        expected_tree_holder = TreeHolder(expected_tree, self.tree_dist, None)

        self._run_asserts(actual_tree_builder, actual_tree_holder, expected_tree, expected_tree_holder)

    def test_cherry_tree_add_to_outliers(self):
        n = 100
        p = 1.0

        data = self._create_data_points(7, n, p)

        expected_tree = self.build_cherry_tree(Tree, data[:-1])

        tree_shell = TreeShellNodeAdder(expected_tree, self.tree_dist)

        actual_tree_builder, actual_tree_holder, expected_tree_holder = self._add_a_dp_to_outliers(
            expected_tree, n, p, tree_shell, 7
        )

        self._run_asserts(actual_tree_builder, actual_tree_holder, expected_tree, expected_tree_holder)

    def test_cherry_tree_add_node_to_node(self):
        n = 100
        p = [1.0, 1.0]

        data = self._create_data_points(7, n, p)

        expected_tree = self.build_cherry_tree(Tree, data[:-1])

        tree_shell = TreeShellNodeAdder(expected_tree, self.tree_dist)

        actual_tree_built_roots = expected_tree.roots

        # root_clade_hash = {rt: hash(expected_tree.get_node_clade(rt)) for rt in actual_tree_built_roots}

        actual_tree_builder = tree_shell.create_tree_holder_with_new_node(
            children=[actual_tree_built_roots[0]], datapoint=data[-1]
        )

        actual_tree_holder = actual_tree_builder.build()

        expected_tree_holder = TreeHolder(expected_tree, self.tree_dist, None)

        expected_tree = expected_tree_holder.tree
        expected_tree.create_root_node(children=[actual_tree_built_roots[0]], data=[data[-1]])
        expected_tree_holder = TreeHolder(expected_tree, self.tree_dist, None)

        self._run_asserts(actual_tree_builder, actual_tree_holder, expected_tree, expected_tree_holder)

    def test_cherry_tree_add_datapoint_to_node(self):
        n = 100
        p = 1.0

        data = self._create_data_points(7, n, p)

        actual_tree_built = self.build_cherry_tree(Tree, data[:-1])

        tree_shell = TreeShellNodeAdder(actual_tree_built, self.tree_dist)

        actual_tree_built_roots = actual_tree_built.roots

        # root_clade_hash = {rt:hash(actual_tree_built.get_node_clade(rt)) for rt in actual_tree_built_roots}

        actual_tree_builder = tree_shell.create_tree_holder_with_datapoint_added_to_node(
            actual_tree_built_roots[0], datapoint=data[-1]
        )

        self.assertEqual(actual_tree_builder.multiplicity, actual_tree_built.multiplicity)

        actual_tree_holder = actual_tree_builder.build()

        expected_tree_holder = TreeHolder(actual_tree_built, self.tree_dist, None)

        expected_tree = expected_tree_holder.tree
        expected_tree.add_data_point_to_node(data[-1], actual_tree_built_roots[0])
        expected_tree_holder = TreeHolder(expected_tree, self.tree_dist, None)

        self._run_asserts(actual_tree_builder, actual_tree_holder, expected_tree, expected_tree_holder)

    def _create_data_point(self, idx, n, p):
        return simulate_binomial_data(idx, n, p, self._rng, self.outlier_prob)

    def _create_data_points(self, size, n, p, start_idx=0):
        result = []

        for i in range(size):
            result.append(self._create_data_point(i + start_idx, n, p))

        return result


def tree_eq(self_tree, other):
    self_key = (get_clades(self_tree), frozenset(self_tree.outliers))

    other_key = (get_clades(other), frozenset(other.outliers))

    return self_key == other_key


if __name__ == "__main__":
    unittest.main()
