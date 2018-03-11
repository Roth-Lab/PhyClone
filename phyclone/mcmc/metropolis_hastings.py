import numpy as np
import random

import phyclone.math_utils
from phyclone.math_utils import exp_normalize, discrete_rvs


class DataPointSwapSampler(object):
    def sample_tree(self, tree):
        data = tree.data

        new_tree = tree.copy()

        labels = new_tree.labels

        idx_1, idx_2 = random.sample(list(labels.keys()), 2)

        node_1 = labels[idx_1]

        node_2 = labels[idx_2]

        if node_1 == node_2:
            return tree

        data_point_1 = data[idx_1]

        assert data_point_1.idx == idx_1

        data_point_2 = data[idx_2]

        assert data_point_2.idx == idx_2

        new_tree.remove_data_point_from_node(data_point_1, node_1)

        new_tree.remove_data_point_from_node(data_point_2, node_2)

        new_tree.add_data_point_to_node(data_point_1, node_2)

        new_tree.add_data_point_to_node(data_point_2, node_1)

        u = random.random()

        if new_tree.log_p_one - tree.log_p_one > np.log(u):
            tree = new_tree

        return tree


class NodeSwap(object):
    def sample_tree(self, tree):
        if len(tree.nodes) == 1:
            return tree

        trees = [tree]

        nodes = tree.nodes

        if len(tree._data[-1]) > 0:
            nodes.append(-1)

        swap_node = random.choice(nodes)

        for node in nodes:
            if node == swap_node:
                continue

            new_tree = tree.copy()

            new_tree._data[node] = list(tree._data[swap_node])

            new_tree._data[swap_node] = list(tree._data[node])

            if node != -1:
                new_tree._graph.nodes[node]['log_p'] = sum([x.value for x in new_tree._data[node]]) + \
                    new_tree._log_prior

            if swap_node != -1:
                new_tree._graph.nodes[swap_node]['log_p'] = sum([x.value for x in new_tree._data[swap_node]]) + \
                    new_tree._log_prior

            new_tree.update()

            trees.append(new_tree)

        log_p = np.array([x.log_p for x in trees])

        p, _ = exp_normalize(log_p)

        idx = discrete_rvs(p)

        return trees[idx]


class ParentChildSwap(object):
    def sample_tree(self, tree):
        if len(tree.nodes) == 1:
            return tree

        new_tree = tree.copy()

        node_1 = random.choice(list(new_tree.nodes))

        node_2 = new_tree.get_parent(node_1)

        if node_2 is None:
            return tree

        node_2_parent = new_tree.get_parent(node_2)

        if node_2_parent is None:
            return tree

        node_1_children = new_tree.get_children(node_1)

        node_2_children = new_tree.get_children(node_2)

        node_2_children.remove(node_1)

        for child in node_1_children:
            new_tree._graph.remove_edge(node_1, child)

            new_tree._graph.add_edge(node_2, child)

        for child in node_2_children:
            new_tree._graph.remove_edge(node_2, child)

            new_tree._graph.add_edge(node_1, child)

        new_tree._graph.remove_edge(node_2_parent, node_2)

        new_tree._graph.remove_edge(node_2, node_1)

        new_tree._graph.add_edge(node_2_parent, node_1)

        new_tree._graph.add_edge(node_1, node_2)

        new_tree.update()

        u = random.random()

        if new_tree.log_p_one - tree.log_p_one > np.log(u):
            tree = new_tree

        return tree
#
#
# class SimpleSampler(object):
#     def sample_tree(self, node_data, tree):
#         new_tree = tree.copy()
#
#         node = random.choice(list(new_tree.nodes.values()))
#
#         if len(node.node_data) == 1:
#             parent = new_tree.get_parent_node(node)
#
#             if parent is None:
#                 return tree
#
#             new_tree.remove_subtree(new_tree.get_subtree(node))
#
#             new_tree.add_subtree(
#                 Tree.create_tree_from_nodes(new_tree.alpha, new_tree.grid_size, Tree.get_nodes(node)[1:], [])
#             )
#
#             parent.add_data_point(node.node_data[0])
#
#         else:
#             data_point = random.choice(node.node_data)
#
#             node.remove_data_point(data_point)
#
#             if phyclone.math_utils.bernoulli_rvs():
#                 new_node = random.choice(list(new_tree.nodes.values()))
#
#                 new_node.add_data_point(data_point)
#
#                 assert data_point in new_node.node_data
#
#                 assert data_point.idx in new_tree.data_points
#
#             else:
#                 idx = new_tree.new_node_idx
#
#                 new_node = MarginalNode(idx, new_tree.grid_size, [])
#
#                 new_node.add_data_point(data_point)
#
#                 node.add_child_node(new_node)
#
#                 new_tree._nodes[new_node.idx] = new_node
#
#                 new_tree._graph.add_edge(node.idx, new_node.idx)
#
#             assert data_point.idx in new_tree.data_points
#
#         new_tree.update_likelihood()
#
#         u = random.random()
#
# #         print(new_tree.log_p_one, tree.log_p_one)
#
#         if new_tree.log_p_one - tree.log_p_one > np.log(u):
#             tree = new_tree
#
#         return tree


class OutlierSampler(object):
    # TODO: Tree prior term
    def sample_tree(self, tree):
        data = list(tree.data)

        data_point = random.choice(data)

        node_data = tree.node_data

        for node in node_data:
            if data_point in node_data[node]:
                if (len(node_data[node]) == 1) and (node != -1):
                    return tree

                orig_node = node

        tree.remove_data_point_from_node(data_point, orig_node)

        log_p = {}

        for node in node_data:
            tree.add_data_point_to_node(data_point, node)

            log_p[node] = tree.log_p_one

            tree.remove_data_point_from_node(data_point, node)

        p, _ = phyclone.math_utils.exp_normalize(np.array(list(log_p.values())).astype(float))

        x = phyclone.math_utils.discrete_rvs(p)

        node = list(log_p.keys())[x]

        tree.add_data_point_to_node(data_point, node)

        return tree


class PruneRegraphSampler(object):
    def sample_tree(self, tree):
        if len(tree.nodes) <= 1:
            return tree

        new_tree = tree.copy()

        subtree_root = random.choice(new_tree.nodes)

        subtree = new_tree.get_subtree(subtree_root)

        new_tree.remove_subtree(subtree)

        remaining_nodes = new_tree.nodes

        if len(remaining_nodes) == 0:
            return tree

        trees = [tree]

        remaining_nodes.append(None)

        for parent in remaining_nodes:
            new_tree = tree.copy()

            subtree = new_tree.get_subtree(subtree_root)

            new_tree.remove_subtree(subtree)

            new_tree.add_subtree(subtree, parent=parent)

            new_tree.update()

            trees.append(new_tree)

        log_p = np.array([x.log_p for x in trees])

        p, _ = exp_normalize(log_p)

        idx = discrete_rvs(p)

        return trees[idx]


class OutlierNodeSampler(object):
    def sample_tree(self, tree):
        trees = [tree]

        for node in tree.nodes:
            trees.append(self._try_node(tree, node))

        log_p = np.array([x.log_p for x in trees])

        p, _ = exp_normalize(log_p)

        idx = discrete_rvs(p)

        return trees[idx]

    def _try_node(self, tree, node):
        new_tree = tree.copy()

        parent = new_tree.get_parent(node)

        children = new_tree.get_children(node)

        new_tree._graph.remove_node(node)

        for child in children:
            new_tree._graph.add_edge(parent, child)

        new_tree._data[-1].extend(new_tree._data[node])

        del new_tree._data[node]

        new_tree.update()

        return new_tree
