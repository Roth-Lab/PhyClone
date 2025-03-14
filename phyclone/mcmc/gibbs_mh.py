import numpy as np
from phyclone.tree import Tree, TreeJointDistribution
from phyclone.utils.math import exp_normalize, log_normalize


class DataPointSampler(object):
    """Gibbs sample a new assignment for a data point.

    TODO: Confirm this is valid since we have a special condition to avoid creating empty nodes.
    """

    __slots__ = ("tree_dist", "outliers", "_rng")

    def __init__(self, tree_dist: TreeJointDistribution, rng: np.random.Generator, outliers: bool = False):
        self.tree_dist = tree_dist

        self.outliers = outliers

        self._rng = rng

    def sample_tree(self, tree: Tree):
        tree_labels = tree.labels
        data_idxs = list(tree_labels.keys())

        self._rng.shuffle(data_idxs)

        for data_idx in data_idxs:
            old_node = tree_labels[data_idx]
            if tree.get_data_len(old_node) > 1:
                tree = self._sample_tree(data_idx, tree, old_node)
                tree_labels = tree.labels

        return tree

    def _sample_tree(self, data_idx, tree, old_node):
        data_point = tree.data[data_idx]

        assert data_point.idx == data_idx

        new_trees = []

        rem_tree = tree.copy()
        rem_tree.remove_data_point_from_node(data_point, old_node)

        for new_node in tree.nodes:
            new_tree = rem_tree.copy()

            new_tree.add_data_point_to_node(data_point, new_node)

            new_trees.append(new_tree)

        if self.outliers:
            new_tree = rem_tree.copy()

            new_tree.add_data_point_to_outliers(data_point)

            new_trees.append(new_tree)

        log_q = np.array([self.tree_dist.log_p_one(x) for x in new_trees])

        log_q = log_normalize(log_q)

        q = np.exp(log_q)

        q = q / sum(q)

        tree_idx = self._rng.multinomial(1, q).argmax()

        return new_trees[tree_idx]


class PruneRegraphSampler(object):
    """Prune a subtree and regraph by Gibbs sampling possible attachment points"""

    __slots__ = ("tree_dist", "_rng")

    def __init__(self, tree_dist: TreeJointDistribution, rng: np.random.Generator):
        self.tree_dist = tree_dist

        self._rng = rng

    def sample_tree(self, tree: Tree):
        if tree.get_number_of_nodes() <= 1:
            return tree

        remaining_nodes, pruned_tree, subtree = self._get_subtree_and_pruned_tree(tree)

        if len(remaining_nodes) == 0:
            return tree

        trees = self._create_sampled_trees_array(remaining_nodes, pruned_tree, subtree)

        log_p = np.array([np.log(n + 1) + self.tree_dist.log_p_one(x) for n, x in trees])

        p, _ = exp_normalize(log_p)

        idx = self._rng.multinomial(1, p).argmax()

        return trees[idx][1]

    @staticmethod
    def _create_sampled_trees_array(
        remaining_nodes: list[int | str | None],
        pruned_tree: Tree,
        subtree: Tree,
    ) -> list[tuple[int, Tree]]:
        trees = []
        # Descendant from dummy normal node
        remaining_nodes.append(None)
        for parent in remaining_nodes:
            new_tree = pruned_tree.copy()

            if parent is None:
                nc = new_tree.get_number_of_children(new_tree.root_node_name)

            else:
                nc = new_tree.get_number_of_children(parent)

            new_tree.add_subtree(subtree, parent=parent)

            new_tree.update()

            trees.append((nc, new_tree))
        return trees

    def _get_subtree_and_pruned_tree(self, tree: Tree) -> tuple[list[int | str | None], Tree, Tree]:
        pruned_tree = tree.copy()
        subtree_root = self._rng.choice(pruned_tree.nodes)
        subtree = pruned_tree.get_subtree(subtree_root)
        pruned_tree.remove_subtree(subtree)
        remaining_nodes = pruned_tree.nodes
        return remaining_nodes, pruned_tree, subtree
