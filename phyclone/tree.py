from collections import defaultdict

import networkx as nx
import numpy as np

from phyclone.consensus import get_clades
from phyclone.math_utils import log_sum_exp, log_factorial

from phyclone.tree_utils import add_to_log_p, subtract_from_log_p, compute_log_R, compute_log_S, add_to_log_R


class FSCRPDistribution(object):
    """ FSCRP prior distribution on trees.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def log_p(self, tree):
        log_p = 0

        # CRP prior
        num_nodes = len(tree.nodes)

        log_p += num_nodes * np.log(self.alpha)

        for node, node_data in tree.node_data.items():
            if node == -1:
                continue

            num_data_points = len(node_data)

            log_p += log_factorial(num_data_points - 1)

        # Uniform prior on toplogies
        log_p -= (num_nodes - 1) * np.log(num_nodes + 1)

        return log_p


class TreeJointDistribution(object):

    def __init__(self, prior):
        self.prior = prior

    def log_p(self, tree):
        """ The log likelihood of the data marginalized over root node parameters.
        """
        log_p = self.prior.log_p(tree)

        # Outlier prior
        for node, node_data in tree.node_data.items():
            for data_point in node_data:
                if data_point.outlier_prob != 0:
                    if node == -1:
                        # log_p += np.log(data_point.outlier_prob)
                        log_p += data_point.outlier_prob

                    else:
                        # log_p += np.log1p(-data_point.outlier_prob)
                        log_p += data_point.outlier_prob_not

        if len(tree.roots) > 0:
            for i in range(tree.grid_size[0]):
                log_p += log_sum_exp(tree.data_log_likelihood[i, :])

        for data_point in tree.outliers:
            log_p += data_point.outlier_marginal_prob

        return log_p

    def log_p_one(self, tree):
        """ The log likelihood of the data conditioned on the root having value 1.0 in all dimensions.
        """
        log_p = self.prior.log_p(tree)

        # Outlier prior
        for node, node_data in tree.node_data.items():
            for data_point in node_data:
                if data_point.outlier_prob != 0:
                    if node == -1:
                        # log_p += np.log(data_point.outlier_prob)
                        log_p += data_point.outlier_prob

                    else:
                        log_p += data_point.outlier_prob_not

        if len(tree.roots) > 0:
            for i in range(tree.grid_size[0]):
                log_p += tree.data_log_likelihood[i, -1]

        for data_point in tree.outliers:
            log_p += data_point.outlier_marginal_prob

        return log_p


class Tree(object):

    def __init__(self, grid_size, memo_logs):
        self.grid_size = grid_size

        # self.factorial_arr = factorial_arr

        self._data = defaultdict(list)

        self._log_prior = -np.log(grid_size[1])

        self._graph = nx.DiGraph()

        self._log_p_comp_memo = memo_logs["log_p"]

        self._set_log_p_memo()

        self._add_node("root")

        self.memo_logs = memo_logs

    def __hash__(self):
        return hash((get_clades(self), frozenset(self.outliers)))

    def __eq__(self, other):
        self_key = (get_clades(self), frozenset(self.outliers))

        other_key = (get_clades(other), frozenset(other.outliers))

        return self_key == other_key

    def _set_log_p_memo(self):
        tmp_hash = "log_p"
        tmp_hash_2 = "zeros"
        if tmp_hash not in self._log_p_comp_memo:
            self._log_p_comp_memo[tmp_hash] = np.ascontiguousarray(np.ones(self.grid_size) * self._log_prior)
        if tmp_hash_2 not in self._log_p_comp_memo:
            self._log_p_comp_memo[tmp_hash_2] = np.zeros(self.grid_size, order='C')

    @staticmethod
    def get_single_node_tree(data, memo_logs):
        """ Load a tree with all data points assigned single node.

        Parameters
        ----------
        data: list
            Data points.
        """
        tree = Tree(data[0].grid_size, memo_logs)

        node = tree.create_root_node([])

        for data_point in data:
            tree.add_data_point_to_node(data_point, node)

        return tree

    @property
    def graph(self):
        result = self._graph.copy()

        result.remove_node("root")

        return result

    @property
    def data(self):
        result = []

        for node in self._data:
            result.extend(self._data[node])

        result = sorted(result, key=lambda x: x.idx)

        return result

    @property
    def data_log_likelihood(self):
        """ The log likelihood grid of the data for all values of the root node.
        """
        return self._graph.nodes["root"]["log_R"]

    @property
    def labels(self):
        result = {}

        for node, node_data in self.node_data.items():
            for data_point in node_data:
                result[data_point.idx] = node

        return result

    @property
    def nodes(self):
        result = list(self._graph.nodes())

        result.remove("root")

        return result

    @property
    def node_data(self):
        result = self._data.copy()

        if "root" in result:
            del result["root"]

        return result

    @property
    def outliers(self):
        return list(self._data[-1])

    @property
    def roots(self):
        return list(self._graph.successors("root"))

    @staticmethod
    def from_dict(data, tree_dict, memo_logs=None):
        if memo_logs is None:
            memo_logs = {"log_p": {}}

        new = Tree(data[0].grid_size, memo_logs)

        new._graph = nx.DiGraph(tree_dict["graph"])

        data = dict(zip([x.idx for x in data], data))

        for node in new._graph.nodes:
            new._add_node(node)

        for idx, node in tree_dict["labels"].items():
            new._data[node].append(data[idx])

            if node != -1:
                # new._graph.nodes[node]["log_p"] += data[idx].value
                # new._graph.nodes[node]["log_R"] += data[idx].value
                new._graph.nodes[node]["log_p"] = add_to_log_p(new._graph.nodes[node]["log_p"], data[idx].value)
                new._graph.nodes[node]["log_R"] = add_to_log_R(new._graph.nodes[node]["log_R"], data[idx].value)
                # new._graph.nodes[node]["log_R"] = compute_log_R(new._graph.nodes[node]["log_R"], data[idx].value)

        new.update()

        return new

    def to_dict(self):
        return {
            "graph": nx.to_dict_of_dicts(self._graph),
            "labels": self.labels
        }

    def add_data_point_to_node(self, data_point, node):
        assert data_point.idx not in self.labels.keys()

        self._data[node].append(data_point)

        if node != -1:
            # self._graph.nodes[node]["log_p"] += data_point.value
            # self._graph.nodes[node]["log_R"] += data_point.value
            # self._graph.nodes[node]["log_R"] = compute_log_R(self._graph.nodes[node]["log_R"], data_point.value)
            self._graph.nodes[node]["log_R"] = add_to_log_R(self._graph.nodes[node]["log_R"], data_point.value)
            self._graph.nodes[node]["log_p"] = add_to_log_p(self._graph.nodes[node]["log_p"], data_point.value)

            self._update_path_to_root(self.get_parent(node))

    def add_data_point_to_outliers(self, data_point):
        self._data[-1].append(data_point)

    def add_subtree(self, subtree, parent=None):
        first_label = max(self.nodes + subtree.nodes + [-1, ]) + 1

        node_map = {}

        subtree = subtree.copy()

        for new_node, old_node in enumerate(subtree.nodes, first_label):
            node_map[old_node] = new_node

            self._data[new_node] = subtree._data[old_node]

        nx.relabel_nodes(subtree._graph, node_map, copy=False)

        self._graph = nx.compose(self._graph, subtree.graph)

        # Connect subtree
        if parent is None:
            parent = "root"

        for node in subtree.roots:
            self._graph.add_edge(parent, node)

        self._update_path_to_root(parent)

    def create_root_node(self, children=None, data=None):
        """ Create a new root node in the forest.

        Parameters
        ----------
        children: list
            Children of the new node.
        data: list
            Data points to add to new node.
        """
        if data is None:
            data = []
        if children is None:
            children = []

        node = nx.number_of_nodes(self._graph) - 1

        self._add_node(node)

        for data_point in data:
            self._data[node].append(data_point)

            # self._graph.nodes[node]["log_p"] += data_point.value
            self._graph.nodes[node]["log_p"] = add_to_log_p(self._graph.nodes[node]["log_p"], data_point.value)

        self._graph.add_edge("root", node)

        for child in children:
            self._graph.remove_edge("root", child)

            self._graph.add_edge(node, child)

        self._update_path_to_root(node)

        return node

    def copy(self):

        cls = self.__class__

        new = cls.__new__(cls)

        new.grid_size = self.grid_size

        new._data = defaultdict(list)

        new.memo_logs = self.memo_logs

        # new.factorial_arr = self.factorial_arr

        new._log_p_comp_memo = self.memo_logs["log_p"]

        for node in self._data:
            new._data[node] = list(self._data[node])

        new._log_prior = self._log_prior

        new._graph = self._graph.copy()

        for node in new._graph:
            new._graph.nodes[node]["log_R"] = self._graph.nodes[node]["log_R"]
            new._graph.nodes[node]["log_p"] = self._graph.nodes[node]["log_p"]
            # new._graph.nodes[node]["log_p"] = self._graph.nodes[node]["log_p"].copy()
            #
            # new._graph.nodes[node]["log_S"] = self._graph.nodes[node]["log_S"].copy()
            #
            # new._graph.nodes[node]["log_R"] = self._graph.nodes[node]["log_R"].copy()

        return new

    def get_children(self, node):
        return list(self._graph.successors(node))

    def get_descendants(self, source="root"):
        return nx.descendants(self._graph, source=source)

    def get_parent(self, node):
        if node == "root":
            return None

        else:
            return list(self._graph.predecessors(node))[0]

    def get_data(self, node):
        return list(self._data[node])

    def get_subtree(self, subtree_root):
        if subtree_root == "root":
            return self.copy()

        new = Tree(self.grid_size, self.memo_logs)

        subtree_graph = nx.dfs_tree(self._graph, subtree_root)

        new._graph = nx.compose(new._graph, subtree_graph)

        new._graph.add_edge("root", subtree_root)

        for node in new.nodes:
            new._data[node] = list(self._data[node])
            new._graph.nodes[node]["log_p"] = self._graph.nodes[node]["log_p"]

            # new._graph.nodes[node]["log_p"] = self._graph.nodes[node]["log_p"].copy()

        new.update()

        return new

    def get_subtree_data(self, node):
        data = self.get_data(node)

        for desc in self.get_descendants(node):
            data.extend(self.get_data(desc))

        return data

    def relabel_nodes(self):
        node_map = {}

        data = defaultdict(list)

        data[-1] = self._data[-1]

        new_node = 0

        for old_node in nx.dfs_preorder_nodes(self._graph, source="root"):
            if old_node == "root":
                continue

            node_map[old_node] = new_node

            data[new_node] = self._data[old_node]

            new_node += 1

        self._data = data

        self._graph = nx.relabel_nodes(self._graph, node_map)

    def remove_data_point_from_node(self, data_point, node):
        self._data[node].remove(data_point)

        if node != -1:
            # self._graph.nodes[node]["log_p"] -= data_point.value
            self._graph.nodes[node]["log_p"] = subtract_from_log_p(self._graph.nodes[node]["log_p"], data_point.value)

            self._update_path_to_root(node)

    def remove_data_point_from_outliers(self, data_point):
        self._data[-1].remove(data_point)

    def remove_subtree(self, subtree):
        if subtree == self:
            self.__init__(self.grid_size, self.memo_logs)

        else:
            assert len(subtree.roots) == 1

            parent = self.get_parent(subtree.roots[0])

            self._graph.remove_nodes_from(subtree.nodes)

            for node in subtree.nodes:
                del self._data[node]

            self._update_path_to_root(parent)

    def update(self):
        for node in nx.dfs_postorder_nodes(self._graph, "root"):
            self._update_node(node)

    def _add_node(self, node):
        self._graph.add_node(node)

        # self._graph.nodes[node]["log_p"] = self._log_p_comp_memo[get_set_hash({"log_p"})]
        # self._graph.nodes[node]["log_R"] = self._log_p_comp_memo[get_set_hash({"zeros"})]
        self._graph.nodes[node]["log_p"] = self._log_p_comp_memo["log_p"]
        self._graph.nodes[node]["log_R"] = self._log_p_comp_memo["zeros"]

        # self._graph.nodes[node]["log_p"] = np.ones(self.grid_size) * self._log_prior
        #
        # self._graph.nodes[node]["log_R"] = np.zeros(self.grid_size)
        #
        # self._graph.nodes[node]["log_S"] = np.zeros(self.grid_size)

    def _update_path_to_root(self, source):
        """ Update recursion values for all nodes on the path between the source node and root inclusive.
        """
        paths = list(nx.all_simple_paths(self._graph, "root", source))

        if len(paths) == 0:
            assert source == "root"

            paths = [["root"]]

        assert len(paths) == 1

        path = paths[0]

        assert path[-1] == source

        assert path[0] == "root"

        for source in reversed(path):
            self._update_node(source)

    def _update_node(self, node):
        child_log_r_values = [self._graph.nodes[child]["log_R"] for child in self._graph.successors(node)]

        if len(child_log_r_values) == 0:
            log_s = np.zeros(self.grid_size, order='C')
        else:
            log_s = compute_log_S(child_log_r_values)

        log_p = self._graph.nodes[node]["log_p"]

        self._graph.nodes[node]["log_R"] = compute_log_R(log_p, log_s)


