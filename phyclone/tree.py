from scipy.misc import logsumexp as log_sum_exp
from scipy.signal import fftconvolve

import networkx as nx
import numpy as np

from phyclone.math_utils import log_factorial


class Tree(object):
    """ FSCRP tree data structure.

    This structure includes the dummy root node.
    """

    def __init__(self, alpha, grid_size, nodes, outliers):
        """
        Parameters
        ----------
        alpha: float
            CRP concentration parameter.
        grid_size: tuple
            The size of the grid used for likelihood.
        nodes: list
            A list of MarginalNodes that form the tree.
        outliers: list
            A list of outlier data points.
        """
        self.alpha = alpha

        self.grid_size = grid_size

        self.outliers = outliers

        self._init_graph(nodes)

        self._validate()

    @staticmethod
    def get_nodes(source):
        """ Recursively fetch all nodes in the subtree rooted at source node.

        Parameters
        ----------
        source: MarginalNode
            Root node to fetch descendants below.
        """
        nodes = [source, ]

        for child in source.children:
            nodes.extend(Tree.get_nodes(child))

        return nodes

    @staticmethod
    def get_single_node_tree(data):
        """ Load a tree with all data points assigned single node.

        Parameters
        ----------
        data: list
            Data points.
        """
        nodes = [MarginalNode(0, data[0].shape, []), ]

        nodes[0].data = data

        return Tree(1.0, nodes[0].grid_size, nodes, [])

    @property
    def data_points(self):
        """ Set of data points in the tree.
        """
        return set(self.labels.keys())

    @property
    def graph(self):
        """ NetworkX graph representing tree.
        """
        graph = self._graph.copy()

        graph.remove_node('root')

        return graph

    @property
    def labels(self):
        """ Cluster assignment of data points.

        Outliers are numbered -1. All other clusters are from 0 onwards.
        """
        labels = {}

        for node in self.nodes.values():
            for data_point in node.data:
                labels[data_point.idx] = node.idx

        for data_point in self.outliers:
            labels[data_point.idx] = -1

        return labels

    @property
    def log_likelihood(self):
        """ The log likelihood of the data marginalized over root node parameters.
        """
        log_p = self._nodes['root'].log_p

        for data_point in self.outliers:
            log_norm = np.log(data_point.value.shape[1])

            log_p += np.sum(log_sum_exp(data_point.value - log_norm, axis=1))

        return log_p

    @property
    def log_likelihood_one(self):
        """ The log likelihood of the data conditioned on the root having value 1.0 in all dimensions.
        """
        log_p = self._nodes['root'].log_p_one

        for data_point in self.outliers:
            log_norm = np.log(data_point.value.shape[1])

            log_p += np.sum(log_sum_exp(data_point.value - log_norm, axis=1))

        return log_p

    @property
    def log_p(self):
        """ Log joint probability.
        """
        return self.log_p_prior + self.log_likelihood

    @property
    def log_p_one(self):
        """ The joint probability of the data conditioned on the root having value 1.0 in all dimensions.
        """
        return self.log_p_prior + self.log_likelihood_one

    @property
    def log_p_prior(self):
        """ Log prior from the FSCRP and outlier contributions.
        """
        log_p = 0

        # Outlier prior
        for data_point in self.outliers:
            log_p += np.log(data_point.outlier_prob)

        for node in self.nodes.values():
            for data_point in node.data:
                log_p += np.log(1 - data_point.outlier_prob)

        # CRP prior
        num_nodes = len(self.nodes)

        log_p += num_nodes * np.log(self.alpha)

        for node in self.nodes.values():
            num_data_points = len(node.data)

            log_p += log_factorial(num_data_points - 1)

        # Uniform prior on toplogies
        log_p -= (num_nodes - 1) * np.log(num_nodes + 1)

        return log_p

    @property
    def log_p_sigma(self):
        """ Log probability of the permutation.
        """
        # TODO: Check this. Are we missing a term for the outliers.
        # Correction for auxillary distribution
        log_p = 0

        for node in self.nodes.values():
            log_p += log_factorial(sum([len(child.data) for child in node.children]))

            for child in node.children:
                log_p -= log_factorial(len(child.data))

            log_p += log_factorial(len(node.data))

        return -log_p

    @property
    def nodes(self):
        """ Dictionary of nodes keyed by node index.
        """
        nodes = self._nodes.copy()

        del nodes['root']

        return nodes

    @property
    def node_sizes(self):
        """ Dictionary mapping nodes to number of data points.
        """
        cluster_sizes = {}

        for node in self.nodes.values():
            cluster_sizes[node.idx] = len(node.data)

        return cluster_sizes

    @property
    def num_nodes(self):
        """ Number of nodes in the forest.
        """
        return len(self.nodes)

    @property
    def roots(self):
        """ List of nodes attached to the dummy root.

        These are the actual root nodes in the forest.
        """
        return [self._nodes[idx] for idx in self._graph.successors('root')]

    def copy(self):
        """ Make a copy of the tree which shares no memory with the original.
        """
        root = self._nodes['root'].copy()

        return Tree(self.alpha, self.grid_size, Tree.get_nodes(root)[1:], list(self.outliers))

    def draw(self, ax=None):
        """ Draw the tree.
        """
        nx.draw(
            self.graph,
            ax=ax,
            pos=nx.drawing.nx_agraph.graphviz_layout(self.graph, prog='dot'),
            with_labels=True
        )

    def add_subtree(self, subtree, parent=None):
        """ Add a subtree to the current tree.

        Parameters:
            subtree: Tree
                The subtree to add to tree.
            parent: MarginalNode
                The node in the tree to use as a parent. If this none the subtree is joined to the dummy root.
        """
        if parent is None:
            parent = self._nodes['root']

        assert parent in self._nodes.values()

        for n in subtree.nodes.values():
            assert n.idx not in self._nodes.keys()

            assert n not in self._nodes.values()

            self._nodes[n.idx] = n

            for child in n.children:
                self._graph.add_edge(n.idx, child.idx)

        for node in subtree.roots:
            self._graph.add_edge(parent.idx, node.idx)

            parent.add_child_node(node)

            assert parent == self.get_parent_node(node)

            assert node in parent.children

        self._update_ancestor_nodes(parent)

        self._validate()

    def get_parent_node(self, node):
        """ Retrieve the parent of a node.
        """
        if node.idx == 'root':
            return None

        parent_idxs = list(self._graph.predecessors(node.idx))

        assert len(parent_idxs) == 1

        parent_idx = parent_idxs[0]

        return self._nodes[parent_idx]

    def get_subtree(self, subtree_root):
        """ Get a subtree.

        subtree_root: MarginalNode
            Root node of the subtree.
        """
        subtree_node_idxs = list(nx.dfs_tree(self._graph, subtree_root.idx))

        nodes = []

        for node_idx in subtree_node_idxs:
            nodes.append(self._nodes[node_idx])

        t = Tree(self.alpha, subtree_root.grid_size, nodes, [])

        t._validate()

        return t

    def remove_subtree(self, tree):
        """ Remove a subtree of the current tree.

        Parameters
        ----------
        tree: Tree
            Subtree to remove from the tree. This should come from the get_subtree method.
        """
        for root in tree.roots:
            subtree = tree.get_subtree(root)

            self._remove_subtree(subtree)

    def relabel_nodes(self, min_value=0):
        """ Relabel all nodes in the tree so their indexes are sequential from a minimum value.

        Parameters
        ----------
        min_value: int
            The lowest node index in the tree.
        """
        # TODO: Fix this by copy the dicts
        node_map = {}

        self._data_points = {'root': []}

        self._nodes = {'root': self._nodes['root'].copy()}

        old_nodes = Tree.get_nodes(self._nodes['root'])

        for new_idx, old in enumerate(old_nodes[1:], min_value):
            node_map[old.idx] = new_idx

            self._nodes[new_idx] = old

            self._nodes[new_idx].idx = new_idx

        self._graph = nx.relabel_nodes(self._graph, node_map)

        for node_idx in nx.dfs_preorder_nodes(self._graph, 'root'):
            node = self._nodes[node_idx]

            node._children = {}

            for child_idx in self._graph.successors(node_idx):
                child = self._nodes[child_idx]

                node._children[child_idx] = child

                assert child_idx == child.idx

        self._validate()

    def _init_graph(self, nodes):
        # Create graph
        G = nx.DiGraph()

        for node in nodes:
            G.add_node(node.idx)

            for child in node.children:
                G.add_edge(node.idx, child.idx)

        # Connect roots to dummy root
        G.add_node('root')

        for node in nodes:
            if G.in_degree(node.idx) == 0:
                G.add_edge('root', node.idx)

        self._graph = G

        # Instantiate dummy node
        roots = []

        for node in nodes:
            if node.idx in self._graph.successors('root'):
                roots.append(node)

        dummy_root = MarginalNode(
            'root',
            self.grid_size,
            children=roots
        )

        self._nodes = {}

        for n in Tree.get_nodes(dummy_root):
            self._nodes[n.idx] = n

    def _remove_subtree(self, subtree):
        assert len(subtree.roots) == 1

        subtree_root = subtree.roots[0]

        parent = self.get_parent_node(subtree_root)

        parent.remove_child_node(subtree_root)

        assert subtree_root not in parent.children

        self._update_ancestor_nodes(parent)

        subtree_node_idxs = list(nx.dfs_tree(self._graph, subtree_root.idx))

        # Update data structures
        for node_idx in subtree_node_idxs:
            del self._nodes[node_idx]

            self._graph.remove_node(node_idx)

        self._validate()

    def _update_ancestor_nodes(self, source):
        """ Update all ancestor nodes of source sequentially from source to root.

        Parameters
        ----------
        source: MarginalNode
            The node to begin updating nodes from.
        """
        while source is not None:
            self._nodes[source.idx].update()

            source = self.get_parent_node(source)

    def _validate(self):
        for node in self._nodes.values():
            assert set(self._graph.successors(node.idx)) == set([x.idx for x in node.children])

            for child in node.children:
                parents = list(self._graph.predecessors(child.idx))

                assert len(parents) == 1

                assert parents[0] == node.idx


class MarginalNode(object):
    """ A node in FS-CRP forest with parameters marginalized.
    """

    def __init__(self, idx, grid_size, children=None):
        self.idx = idx

        self.grid_size = grid_size

        self._children = {}

        if children is not None:
            for child in children:
                self._children[child.idx] = child

        self.data = []

        self.log_likelihood = np.ones(grid_size) * -np.log(grid_size[1])

        self.log_R = np.zeros(grid_size)

        self.update()

    @property
    def children(self):
        """ List of child nodes.
        """
        return list(self._children.values())

    @property
    def log_p(self):
        """ Log probability of sub-tree rooted at node, marginalizing node parameter.
        """
        log_p = 0

        for i in range(self.grid_size[0]):
            log_p += log_sum_exp(self.log_R[i, :])

        return log_p

    @property
    def log_p_one(self):
        """ Log probability of sub-tree rooted at node, conditioned on node parameter of one.
        """
        log_p = 0

        for i in range(self.grid_size[0]):
            log_p += self.log_R[i, -1]

        return log_p

    def add_child_node(self, node):
        """ Add a child node.
        """
        assert node.idx not in self.children

        self._children[node.idx] = node

        self.update()

    def add_data_point(self, data_point):
        """ Add a data point to the collection at this node.
        """
        self.data.append(data_point)

        self.log_likelihood += data_point.value

        self._update_log_R()

    def copy(self, deep=True):
        """ Make a deep copy of the node.

        Parameters
        ----------
        deep: bool
            If true then children will be copied, otherwise they will be shared pointers with the original.
        """
        new = MarginalNode.__new__(MarginalNode)

        if deep:
            new._children = {}

            for child in self.children:
                new._children[child.idx] = child.copy()

        else:
            new._children = self._children.copy()

        new.grid_size = self.grid_size

        new.idx = self.idx

        new.data = list(self.data)

        new.log_likelihood = np.copy(self.log_likelihood)

        new.log_R = np.copy(self.log_R)

        new.log_S = np.copy(self.log_S)

        return new

    def remove_child_node(self, node):
        """ Remove a child node.
        """
        del self._children[node.idx]

        self.update()

    def remove_data_point(self, data_point):
        """ Remove a data point to the collection at this node.
        """
        self.data.remove(data_point)

        self.log_likelihood -= data_point.value

        self._update_log_R()

    def update_children(self, children):
        """ Set the node children
        """
        self._children = {}

        if children is not None:
            for child in children:
                self._children[child.idx] = child

        self.log_R = np.zeros(self.grid_size)

        self.update()

    def update(self):
        """ Update the arrays required for the recursion.
        """
        self._update_log_S()

        self._update_log_R()

    def _compute_log_D(self):
        for child_id, child in enumerate(self.children):
            if child_id == 0:
                log_D = child.log_R.copy()

            else:
                for i in range(self.grid_size[0]):
                    log_D[i, :] = _compute_log_D_n(child.log_R[i, :], log_D[i, :])

        return log_D

    def _update_log_R(self):
        self.log_R = self.log_likelihood + self.log_S

    def _update_log_S(self):
        self.log_S = np.zeros(self.grid_size)

        if len(self.children) > 0:
            log_D = self._compute_log_D()

            for i in range(self.grid_size[0]):
                self.log_S[i, :] = np.logaddexp.accumulate(log_D[i, :])


def _compute_log_D_n(child_log_R, prev_log_D_n):
    """ Compute the recursion over D using the FFT.
    """
    log_R_max = child_log_R.max()

    log_D_max = prev_log_D_n.max()

    R_norm = np.exp(child_log_R - log_R_max)

    D_norm = np.exp(prev_log_D_n - log_D_max)

    result = fftconvolve(R_norm, D_norm)

    result = result[:len(child_log_R)]

    result[result <= 0] = 1e-100

    return np.log(result) + log_D_max + log_R_max
