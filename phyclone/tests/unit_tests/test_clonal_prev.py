import unittest

import networkx as nx
import numpy as np
import pandas as pd

from phyclone.process_trace.map import get_map_node_clonal_prevs_dict
from phyclone.tests.utilities.fscrp import add_clonal_prev, add_cellular_prev


class TestClonalPrevOutput(unittest.TestCase):
    seed = 244310326493402386435613023297139050129
    grid_size = 101

    def setUp(self):
        self.rng = np.random.default_rng(self.seed)

    def build_random_tree(self, num_nodes, num_samples):
        graph = nx.random_labeled_rooted_tree(num_nodes, seed=self.rng)
        tree = nx.DiGraph()
        if num_nodes == 1:
            tree.add_nodes_from(graph.nodes)
        tree.add_edges_from(nx.dfs_edges(graph, source=graph.graph["root"]))
        tree.graph["root"] = graph.graph["root"]
        add_clonal_prev(num_samples, tree, self.rng)
        add_cellular_prev(tree)
        tree.nodes[tree.graph["root"]]["log_R"] = np.zeros((num_samples, self.grid_size))
        grid_div = self.grid_size - 1
        for node in tree.nodes:
            tree.nodes[node]["max_idx"] = tree.nodes[node]["cellular_prev"] * grid_div
        return tree

    @staticmethod
    def expected_tree_vs_actual_dict(tree, prev_dict):
        for node in tree.nodes:
            expected_clonal_prev = tree.nodes[node]["clonal_prev"]
            actual_clonal_prev = prev_dict[node]
            np.testing.assert_allclose(actual_clonal_prev, expected_clonal_prev, verbose=True)

    @staticmethod
    def dataframe_sum_actual_vs_expected(actual_dict):
        df = pd.DataFrame(actual_dict)
        df["sum"] = df[df.columns].sum(axis=1)
        df["expected_sum"] = np.float64(1.0)
        np.testing.assert_allclose(df["sum"], df["expected_sum"], verbose=True)

    def _run_test(self, test_tree):
        actual_dict = get_map_node_clonal_prevs_dict(test_tree, test_tree.graph["root"])
        with self.subTest(msg="Actual Clonal prev values are float-close to expected"):
            self.expected_tree_vs_actual_dict(test_tree, actual_dict)

        with self.subTest(msg="Actual Clonal prev values sum to ~1 in all samples"):
            self.dataframe_sum_actual_vs_expected(actual_dict)

    def run_test(self, test_tree, struct_msg=None):
        if struct_msg is None:
            print("Tree structure:")
        else:
            print(struct_msg)
        nx.write_network_text(test_tree)
        self._run_test(test_tree)

    def test_single_node_one_sample(self):
        test_tree = self.build_random_tree(1, 1)
        self.run_test(test_tree)

    def test_three_nodes_one_sample(self):
        test_tree = self.build_random_tree(3, 1)
        self.run_test(test_tree)

    def test_single_node_two_samples(self):
        test_tree = self.build_random_tree(1, 2)
        self.run_test(test_tree)

    def test_three_nodes_two_samples(self):
        test_tree = self.build_random_tree(3, 2)
        self.run_test(test_tree)

    def test_eight_nodes_two_samples(self):
        for tree_iter in range(5):
            with self.subTest(msg="Random tree {}".format(tree_iter), random_tree=tree_iter):
                test_tree = self.build_random_tree(8, 2)
                self.run_test(test_tree, struct_msg="Random tree {} structure:".format(tree_iter))


if __name__ == "__main__":
    unittest.main()
