import unittest

import numpy as np
import rustworkx as rx


from phyclone.data.base import DataPoint
from phyclone.tree import FSCRPDistribution, Tree, TreeJointDistribution
from phyclone.utils.math_utils import log_factorial
from phyclone.tests.utilities.simulate import simulate_binomial_data


class TestTree(unittest.TestCase):

    def setUp(self):
        grid_size = (1, 10)

        self.tree = Tree(grid_size)

        self.tree_dist = FSCRPDistribution(1.0)

        self.tree_joint_dist = TreeJointDistribution(self.tree_dist)

    def test_create_root_node(self):
        node = self.tree.create_root_node([])

        self.assertEqual(node, 0)

        self.assertListEqual(self.tree.nodes, [0])

        self.assertListEqual(self.tree.roots, [0])

    def test_simple_log_p(self):
        self.assertEqual(self.tree_joint_dist.log_p_one(self.tree), 0)

    def test_one_data_point_sigma(self):
        data_point = self._create_data_point(0)

        node = self.tree.create_root_node([])

        self.tree.add_data_point_to_node(data_point, node)

        self.assertEqual(self.tree_dist.log_p(self.tree), 0)

    def test_two_data_point_one_cluster_sigma(self):
        """CRP term is $\alpha * log(x - 1) = 1.0 * log(2)! and tree term is (n+1)^(n-1) = 2^0"""
        data = self._create_data_points(3)

        node = self.tree.create_root_node([])

        self.tree.add_data_point_to_node(data[0], node)

        self.tree.add_data_point_to_node(data[1], node)

        self.tree.add_data_point_to_node(data[2], node)

        tree_multiplicity = self.tree.multiplicity

        actual = log_factorial(2)

        actual -= tree_multiplicity

        self.assertEqual(self.tree_dist.log_p(self.tree), actual)

    def test_two_data_point_chain_sigma(self):
        """CRP is 0 and tree term is (n+1)^(n-1) = 3^1"""
        data = self._create_data_points(2)

        node = self.tree.create_root_node([])

        self.tree.add_data_point_to_node(data[0], node)

        node = self.tree.create_root_node([node])

        self.tree.add_data_point_to_node(data[1], node)

        tree_multiplicity = self.tree.multiplicity

        actual = -np.log(3)

        actual -= tree_multiplicity

        self.assertEqual(self.tree_dist.log_p(self.tree), actual)

    def test_two_data_point_separate_sigma(self):
        """CRP is 0 and tree term is (n+1)^(n-1) = 3^1"""
        data = self._create_data_points(2)

        node = self.tree.create_root_node([])

        self.tree.add_data_point_to_node(data[0], node)

        node = self.tree.create_root_node([])

        self.tree.add_data_point_to_node(data[1], node)

        tree_multiplicity = self.tree.multiplicity

        actual = -np.log(3)

        actual -= tree_multiplicity

        self.assertEqual(self.tree_dist.log_p(self.tree), actual)

    def _create_data_point(self, idx):
        return DataPoint(idx, np.zeros(self.tree.grid_size))

    def _create_data_points(self, size):
        result = []

        for i in range(size):
            result.append(self._create_data_point(i))

        return result


class TestTreeDictIO(unittest.TestCase):

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

    def _create_data_point(self, idx, n, p):
        return simulate_binomial_data(idx, n, p, self._rng, self.outlier_prob)

    def _create_data_points(self, size, n, p, start_idx=0):
        result = []

        for i in range(size):
            result.append(self._create_data_point(i + start_idx, n, p))

        return result

    def tree_dict_equal(self, dict_1, dict_2):
        self.assertEqual(dict_1["log_prior"], dict_2["log_prior"])
        self.assertEqual(dict_1["grid_size"], dict_2["grid_size"])
        self.assertEqual(set(dict_1["graph"]), set(dict_2["graph"]))
        self.assertEqual(dict_1["node_idx"], dict_2["node_idx"])
        self.assertEqual(set(dict_1["node_data"].keys()), set(dict_2["node_data"].keys()))

        for k, v in dict_1["node_data"].items():
            v_2 = dict_2["node_data"][k]
            self.assertEqual(set(v), set(v_2))

        return True

    def test_zero_node_tree_no_outliers(self):
        grid_size = (10, 101)

        expected_graph = rx.PyDiGraph(multigraph=False)

        expected_dict = {
            "graph": expected_graph.edge_list(),
            "node_idx": {Tree._ROOT_NODE_NAME: 0},
            "node_data": {},
            "grid_size": grid_size,
            "log_prior": -np.log(grid_size[1]),
        }

        expected_tree = Tree(grid_size)
        actual_dict = expected_tree.to_dict()

        actual_tree = Tree.from_dict(actual_dict)

        self.assertTrue(self.tree_dict_equal(expected_dict, actual_dict))
        self.assertEqual(expected_tree, actual_tree)

    def test_zero_node_tree_one_outlier_dp(self):
        n = 100
        p = [1.0] * 10

        data = self._create_data_points(1, n, p)
        grid_size = data[0].grid_size

        expected_tree = Tree(grid_size)
        expected_tree.add_data_point_to_outliers(data[0])
        actual_dict = expected_tree.to_dict()

        expected_dict = {
            "graph": expected_tree._graph.edge_list(),
            "node_idx": {Tree._ROOT_NODE_NAME: 0},
            "node_data": {Tree._OUTLIER_NODE_NAME: [data[0]]},
            "grid_size": grid_size,
            "log_prior": -np.log(grid_size[1]),
        }

        actual_tree = Tree.from_dict(actual_dict)

        self.assertTrue(self.tree_dict_equal(expected_dict, actual_dict))
        self.assertEqual(expected_tree, actual_tree)

    def test_single_node_tree_no_outliers(self):
        n = 100
        p = 1.0

        data = self._create_data_points(2, n, p)
        grid_size = data[0].grid_size

        expected_tree = Tree.get_single_node_tree(data)

        actual_dict = expected_tree.to_dict()

        expected_dict = {
            "graph": expected_tree._graph.edge_list(),
            "node_idx": {Tree._ROOT_NODE_NAME: 0, 0: 1},
            "node_data": {0: list(data)},
            "grid_size": grid_size,
            "log_prior": -np.log(grid_size[1]),
        }

        actual_tree = Tree.from_dict(actual_dict)

        self.assertTrue(self.tree_dict_equal(expected_dict, actual_dict))
        self.assertEqual(expected_tree, actual_tree)

    def test_single_node_tree_one_outlier_dp(self):
        n = 100
        p = 1.0

        data = self._create_data_points(3, n, p)
        grid_size = data[0].grid_size

        expected_tree = Tree.get_single_node_tree(data[:-1])
        expected_tree.add_data_point_to_outliers(data[-1])
        actual_dict = expected_tree.to_dict()

        expected_dict = {
            "graph": expected_tree._graph.edge_list(),
            "node_idx": {Tree._ROOT_NODE_NAME: 0, 0: 1},
            "node_data": {Tree._OUTLIER_NODE_NAME: [data[-1]], 0: list(data[:-1])},
            "grid_size": grid_size,
            "log_prior": -np.log(grid_size[1]),
        }

        actual_tree = Tree.from_dict(actual_dict)

        self.assertTrue(self.tree_dict_equal(expected_dict, actual_dict))
        self.assertEqual(expected_tree, actual_tree)

    def test_cherry_tree_no_outliers(self):
        data = self.build_six_datapoints()
        grid_size = data[0].grid_size

        expected_tree = self.build_cherry_tree(Tree, data)
        actual_dict = expected_tree.to_dict()

        expected_dict = {
            "graph": expected_tree._graph.edge_list(),
            "node_idx": {Tree._ROOT_NODE_NAME: 0, 0: 1, 1: 2, 2: 3},
            "node_data": {0: list(data[:2]), 1: list(data[2:4]), 2: list(data[4:])},
            "grid_size": grid_size,
            "log_prior": -np.log(grid_size[1]),
        }

        actual_tree = Tree.from_dict(actual_dict)

        self.assertTrue(self.tree_dict_equal(expected_dict, actual_dict))
        self.assertEqual(expected_tree, actual_tree)

    def test_cherry_tree_one_outlier_dp(self):
        n = 100
        p = 1.0

        data = self._create_data_points(7, n, p)
        grid_size = data[0].grid_size
        cherry_data = data[:-1]

        expected_tree = self.build_cherry_tree(Tree, cherry_data)
        expected_tree.add_data_point_to_outliers(data[-1])
        actual_dict = expected_tree.to_dict()

        expected_dict = {
            "graph": expected_tree._graph.edge_list(),
            "node_idx": {Tree._ROOT_NODE_NAME: 0, 0: 1, 1: 2, 2: 3},
            "node_data": {
                Tree._OUTLIER_NODE_NAME: [data[-1]],
                0: list(cherry_data[:2]),
                1: list(cherry_data[2:4]),
                2: list(cherry_data[4:]),
            },
            "grid_size": grid_size,
            "log_prior": -np.log(grid_size[1]),
        }

        actual_tree = Tree.from_dict(actual_dict)

        self.assertTrue(self.tree_dict_equal(expected_dict, actual_dict))
        self.assertEqual(expected_tree, actual_tree)

    def test_linear_tree_no_outliers(self):
        data = self.build_six_datapoints()
        grid_size = data[0].grid_size

        expected_tree = self.build_linear_tree(Tree, data)
        actual_dict = expected_tree.to_dict()

        expected_dict = {
            "graph": expected_tree._graph.edge_list(),
            "node_idx": {Tree._ROOT_NODE_NAME: 0, 0: 1, 1: 2, 2: 3},
            "node_data": {0: list(data[:2]), 1: list(data[2:4]), 2: list(data[4:])},
            "grid_size": grid_size,
            "log_prior": -np.log(grid_size[1]),
        }

        actual_tree = Tree.from_dict(actual_dict)

        self.assertTrue(self.tree_dict_equal(expected_dict, actual_dict))
        self.assertEqual(expected_tree, actual_tree)

    def test_linear_tree_one_outlier_dp(self):
        n = 100
        p = 1.0

        data = self._create_data_points(7, n, p)
        grid_size = data[0].grid_size
        cherry_data = data[:-1]

        expected_tree = self.build_linear_tree(Tree, cherry_data)
        expected_tree.add_data_point_to_outliers(data[-1])
        actual_dict = expected_tree.to_dict()

        expected_dict = {
            "graph": expected_tree._graph.edge_list(),
            "node_idx": {Tree._ROOT_NODE_NAME: 0, 0: 1, 1: 2, 2: 3},
            "node_data": {
                Tree._OUTLIER_NODE_NAME: [data[-1]],
                0: list(cherry_data[:2]),
                1: list(cherry_data[2:4]),
                2: list(cherry_data[4:]),
            },
            "grid_size": grid_size,
            "log_prior": -np.log(grid_size[1]),
        }

        actual_tree = Tree.from_dict(actual_dict)

        self.assertTrue(self.tree_dict_equal(expected_dict, actual_dict))
        self.assertEqual(expected_tree, actual_tree)


if __name__ == "__main__":
    unittest.main()
