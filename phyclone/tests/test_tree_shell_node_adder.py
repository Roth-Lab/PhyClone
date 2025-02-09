import unittest

import numpy as np

from phyclone.smc.swarm import TreeHolder
# from phyclone.tests.old_implementations import OldTree
from phyclone.smc.swarm.tree_shell_node_adder import TreeShellNodeAdder
from phyclone.tests.simulate import simulate_binomial_data
from phyclone.tree import FSCRPDistribution, Tree, TreeJointDistribution
from phyclone.tree.utils import get_clades


class TestTreeShellNodeAdder(unittest.TestCase):

    def setUp(self):
        self.outlier_prob = 0.0

        self._rng = np.random.default_rng(12345)

        self.tree_dist = FSCRPDistribution(1.0)

        self.tree_joint_dist = TreeJointDistribution(self.tree_dist)

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

    # def test_copy(self):
    #     original_tree = self.build_cherry_tree(Tree)
    #
    #     copied_tree = original_tree.copy()
    #
    #     original_nodes = original_tree._graph.nodes()
    #     copied_nodes = copied_tree._graph.nodes()
    #
    #     for orig_node, copied_node in zip(original_nodes, copied_nodes):
    #         self.assertIsNot(orig_node, copied_node)
    #         np.testing.assert_array_equal(copied_node.log_r, orig_node.log_r)
    #         np.testing.assert_array_equal(copied_node.log_p, orig_node.log_p)
    #         self.assertIsNot(copied_node.log_r, orig_node.log_r)
    #         self.assertIsNot(copied_node.log_p, orig_node.log_p)
    #         self.assertEqual(copied_node.node_id, orig_node.node_id)
    #
    #         copied_node.log_r += 1
    #
    #         self.assertFalse(np.array_equal(copied_node.log_r, orig_node.log_r))
    #
    #     self.assertEqual(copied_tree, original_tree)
    #     self.assertIsNot(copied_tree, original_tree)

    def test_single_node_tree_hash(self):
        n = 100
        p = 1.0

        data = self._create_data_points(2, n, p)

        # expected_tree = OldTree.get_single_node_tree(data)

        actual_tree_built = Tree.get_single_node_tree(data)

        tree_shell = TreeShellNodeAdder(actual_tree_built)

        self.assertEqual(hash(tree_shell), hash(actual_tree_built))


    def test_single_node_tree_add_node_to_root(self):
        n = 100
        p = 1.0

        data = self._create_data_points(3, n, p)

        actual_tree_built = Tree.get_single_node_tree(data[:-1])

        tree_shell = TreeShellNodeAdder(actual_tree_built)

        actual_tree_built.create_root_node(children=[], data=[data[-1]])

        tree_holder_builder = tree_shell.create_tree_holder_with_new_node(children=[],
                                                                          datapoint=data[-1],
                                                                          tree_dist=self.tree_joint_dist)

        self.assertEqual(tree_holder_builder.multiplicity, actual_tree_built.multiplicity)

        actual_tree_holder = tree_holder_builder.build()

        expected_tree_holder = TreeHolder(actual_tree_built, self.tree_joint_dist, None)

        self.assertEqual(actual_tree_holder, expected_tree_holder)

        self.assertEqual(self.tree_joint_dist.log_p(actual_tree_built), self.tree_joint_dist.log_p(tree_holder_builder))
        self.assertEqual(self.tree_joint_dist.log_p_one(actual_tree_built), self.tree_joint_dist.log_p_one(tree_holder_builder))


        # self.assertEqual(hash(tree_shell), hash(actual_tree_built))

        # actual_tree_dict = actual_tree_built.to_dict()
        #
        # actual_tree = Tree.from_dict(actual_tree_dict)
        #
        # self.assertTrue(tree_eq(expected_tree, actual_tree))

    # def test_cherry_tree_from_dict_representation(self):
    #     data = self.build_six_datapoints()
    #
    #     # expected_tree = self.build_cherry_tree(OldTree, data)
    #
    #     actual_tree_built = self.build_cherry_tree(Tree, data)
    #
    #     actual_tree_dict = actual_tree_built.to_dict()
    #
    #     actual_tree = Tree.from_dict(actual_tree_dict)
    #
    #     self.assertTrue(tree_eq(expected_tree, actual_tree))
    #
    # def test_linear_tree_from_dict_representation(self):
    #
    #     data = self.build_six_datapoints()
    #
    #     # expected_tree = self.build_linear_tree(OldTree, data)
    #
    #     actual_tree_built = self.build_linear_tree(Tree, data)
    #
    #     actual_tree_dict = actual_tree_built.to_dict()
    #
    #     actual_tree = Tree.from_dict(actual_tree_dict)
    #
    #     self.assertTrue(tree_eq(expected_tree, actual_tree))


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
