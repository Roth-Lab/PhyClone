from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import rustworkx as rx

from phyclone.data.base import DataPoint
from phyclone.smc.swarm import TreeHolder
from phyclone.smc.utils import RootPermutationDistribution
from phyclone.tree import Tree, TreeJointDistribution
from phyclone.tree.tree_node import TreeNode
from phyclone.tree.visitors import GraphToCladesVisitor
from phyclone.utils.math_utils import cached_log_factorial


@dataclass(slots=True)
class TreeInfo:
    graph: rx.EdgeList
    node_idx: dict
    node_idx_rev: dict
    node_data: defaultdict[list[DataPoint]]
    grid_size: tuple[int, int]
    log_prior: float

    def build_graph_shell(self):
        graph = rx.PyDiGraph(multigraph=False)
        if len(self.graph) == 0:
            graph.add_node(0)
        else:
            graph.extend_from_edge_list(self.graph)
        return graph

    def get_node_idx_dict(self):
        return self.node_idx.copy()

    def get_node_idx_rev_dict(self):
        return self.node_idx_rev.copy()

    def get_node_data(self):
        node_data = defaultdict(list)
        node_data.update({k: v.copy() for k, v in self.node_data.items()})
        return node_data

    def get_num_datapoints(self):
        count = sum(map(len, self.node_data.values()))
        return count


class TreeHolderBuilder(object):
    __slots__ = (
        "_graph",
        "_multiplicity",
        "_roots",
        "_outlier_node_name",
        "_nodes",
        "_data",
        "_node_idx",
        "_node_idx_rev",
        "grid_size",
        "_log_prior",
        "_root_node_name",
        "_labels",
        "_outliers",
        "roots_num_children",
        "roots_num_desc",
        "_data_log_likelihood",
        "_roots",
        "_hash_val",
        "_tree_dist",
        "_number_of_nodes",
        "_log_pdf",
    )

    def __init__(
        self,
        outlier_node_name: str | int,
        root_node_name: str | int,
        grid_size: tuple[int, int],
        log_prior: float,
        nodes: list[str | int],
    ):
        self._number_of_nodes = 0
        self.roots_num_desc = None
        self.roots_num_children = None
        self._labels = None
        self._graph = None
        self._multiplicity = None
        self._roots = None
        self._outlier_node_name = outlier_node_name
        self._root_node_name = root_node_name
        self._nodes = set(nodes)
        self._data = None
        self._node_idx = None
        self._node_idx_rev = None
        self.grid_size = grid_size
        self._log_prior = log_prior
        self._outliers = None
        self._data_log_likelihood = None
        self._roots = None
        self._hash_val = None
        self._tree_dist = None
        self._log_pdf = 0.0

    def __hash__(self):
        return self._hash_val

    def __eq__(self, other):
        hash_check = hash(self) == hash(other)
        return hash_check

    def set_hash_val(self):
        self._hash_val = hash((self.get_clades(), frozenset(self.outliers)))

    def with_graph(self, graph):
        self._graph = graph
        self.multiplicity = graph
        self._number_of_nodes = graph.num_nodes() - 1
        return self

    def with_node_data(self, node_data):
        self._data = node_data
        self.labels = node_data
        self.outliers = node_data
        return self

    def with_log_pdf(self, log_pdf):
        self._log_pdf = log_pdf

    def with_node_last_added_to(self, node_last_added_to):
        if node_last_added_to != self._outlier_node_name:
            self._nodes.add(node_last_added_to)
            # if node_last_added_to not in self._nodes:
            #     self._nodes.append(node_last_added_to)

        return self

    def with_root_node_object(self, root_node_obj):
        self.data_log_likelihood = root_node_obj
        return self

    def with_roots_num_children(self, roots_num_children):
        self.roots_num_children = roots_num_children
        return self

    def with_roots_num_desc(self, roots_num_desc):
        self.roots_num_desc = roots_num_desc
        return self

    def with_roots(self, roots):
        self._roots = roots
        return self

    def with_node_idx(self, node_idx):
        self._node_idx = node_idx
        return self

    def with_node_idx_rev(self, node_idx_rev):
        self._node_idx_rev = node_idx_rev
        return self

    def with_tree_dist(self, tree_dist: TreeJointDistribution):
        self._tree_dist = tree_dist
        return self

    def build(self) -> TreeHolder:

        ret = TreeHolder(self, self._tree_dist, None)
        ret.log_pdf = self._log_pdf
        return ret

    def get_clades(self) -> frozenset:
        visitor = GraphToCladesVisitor(self._node_idx_rev, self._data, self._root_node_name)
        root_idx = self._node_idx[self._root_node_name]
        rx.dfs_search(self._graph, [root_idx], visitor)
        vis_clades = frozenset(visitor.clades)
        return vis_clades

    def get_node_clade(self, node) -> frozenset:
        visitor = GraphToCladesVisitor(self._node_idx_rev, self._data, node)
        root_idx = self._node_idx[node]
        rx.dfs_search(self._graph, [root_idx], visitor)
        visitor.clades.add(frozenset(visitor.dict_of_sets[node]))
        vis_clades = frozenset(visitor.clades)
        return vis_clades

    def get_number_of_descendants(self, node):
        return self.roots_num_desc[node]

    def get_number_of_children(self, node):
        return self.roots_num_children[node]

    def get_number_of_nodes(self):
        return self._number_of_nodes

    def to_dict(self):
        tree_dict = {
            "graph": self._graph.edge_list(),
            "node_idx": self._node_idx.copy(),
            "node_idx_rev": self._node_idx_rev.copy(),
            "node_data": {k: v.copy() for k, v in self._data.items()},
            "grid_size": self.grid_size,
            "log_prior": self._log_prior,
        }
        return tree_dict

    @property
    def data_log_likelihood(self):
        return self._data_log_likelihood

    @data_log_likelihood.setter
    def data_log_likelihood(self, root_node_obj: TreeNode):
        self._data_log_likelihood = root_node_obj.log_r.copy()

    @property
    def nodes(self):
        return list(self._nodes)

    @property
    def roots(self):
        return list(self._roots)

    @property
    def node_data(self):
        result = self._data.copy()

        if self._root_node_name in result:
            del result[self._root_node_name]

        return result

    @property
    def labels(self):
        return self._labels.copy()

    @labels.setter
    def labels(self, node_data):
        self._labels = {dp.idx: k for k, l in node_data.items() for dp in l}

    @property
    def multiplicity(self):
        return self._multiplicity

    @multiplicity.setter
    def multiplicity(self, graph: rx.PyDiGraph):
        self._multiplicity = Tree.compute_multiplicity_from_graph(graph)

    @property
    def outliers(self):
        return self._outliers

    @outliers.setter
    def outliers(self, node_data):
        self._outliers = list(node_data[self._outlier_node_name])

    @property
    def outlier_node_name(self):
        return self._outlier_node_name

    @property
    def root_node_name(self):
        return self._root_node_name


class TreeShellNodeAdder(object):
    __slots__ = (
        "_root_nodes_dict",
        "_tree_info",
        "_next_node_id",
        "_dummy_root_obj",
        "_hash_val",
        "_grid_size",
        "_log_prior",
        "_root_idx",
        "_root_node_name",
        "_outlier_node_name",
        "_root_node_names_set",
        "roots_num_children",
        "roots_num_desc",
        "_nodes",
        "tree_dist",
        "perm_dist",
        "_perm_dist_dict",
        "_num_datapoints",
        # "_root_clade_dict",
    )

    def __init__(self, tree: Tree, tree_dist: TreeJointDistribution, perm_dist: RootPermutationDistribution = None):
        self._hash_val = hash(tree)
        self._tree_info = TreeInfo(**tree.to_dict())
        self._grid_size = self._tree_info.grid_size
        self._log_prior = self._tree_info.log_prior
        self._root_nodes_dict = tree.get_root_tree_node_dict()
        self._next_node_id = tree.get_number_of_nodes()
        self._dummy_root_obj = tree.get_tree_node_object(tree.root_node_name)
        self._root_idx = self._tree_info.node_idx[tree.root_node_name]
        self._root_node_name = tree.root_node_name
        self._outlier_node_name = tree.outlier_node_name
        self._root_node_names_set = self._root_nodes_dict.keys()
        self._nodes = tree.nodes
        self.perm_dist = perm_dist
        self.tree_dist = tree_dist
        self._perm_dist_dict = None
        self._num_datapoints = 0
        # self._root_clade_dict = {hash(tree.get_node_clade(rt)): rt for rt in self._root_nodes_dict.keys()}

        for subroot in self._root_nodes_dict.values():
            subroot.make_internal_arrays_read_only()
        self._dummy_root_obj.make_internal_arrays_read_only()

        if perm_dist:
            self._num_datapoints = self._tree_info.get_num_datapoints()
            self._perm_dist_dict = self._precompute_subroot_log_counts(tree)

        self.roots_num_children = {
            sub_root: tree.get_number_of_children(sub_root) for sub_root in self._root_node_names_set
        }
        self.roots_num_desc = {
            sub_root: tree.get_number_of_descendants(sub_root) for sub_root in self._root_node_names_set
        }

    def _precompute_subroot_log_counts(self, tree):
        perm_dist_dict = {
            sub_root: (self.perm_dist.log_count(tree, sub_root), tree.get_subtree_data_len(sub_root))
            for sub_root in self._root_node_names_set
        }
        return perm_dist_dict

    def __hash__(self):
        return self._hash_val

    def __eq__(self, other):
        return hash(self) == hash(other)

    def create_tree_holder_with_datapoint_added_to_outliers(self, datapoint: DataPoint):

        node_id = self._outlier_node_name
        tree_holder_builder = self._add_num_children_and_node_id(node_id)

        graph = self._tree_info.build_graph_shell()
        node_idx_dict = self._tree_info.get_node_idx_dict()
        node_idx_rev_dict = self._tree_info.get_node_idx_rev_dict()

        node_data = self._tree_info.get_node_data()
        node_data[node_id].append(datapoint)
        tree_holder_builder.with_node_data(node_data)
        tree_holder_builder.with_graph(graph)
        tree_holder_builder.with_node_idx(node_idx_dict)
        tree_holder_builder.with_node_idx_rev(node_idx_rev_dict)

        root_node_obj = self._dummy_root_obj.copy()
        tree_holder_builder.with_root_node_object(root_node_obj)

        roots_num_children = self.roots_num_children.copy()
        roots_num_children[self._root_node_name] = len(graph.successors(self._root_idx))
        tree_holder_builder.with_roots_num_children(roots_num_children)

        roots_num_desc = self.roots_num_desc.copy()
        roots_num_desc[self._root_node_name] = len(rx.descendants(graph, self._root_idx))
        tree_holder_builder.with_roots_num_desc(roots_num_desc)

        tree_holder_builder.with_roots(list(self._root_node_names_set))

        self._compute_new_log_pdf_added_outlier(self._root_node_names_set, tree_holder_builder)

        tree_holder_builder.set_hash_val()

        return tree_holder_builder

    def create_tree_holder_with_datapoint_added_to_node(self, node_id: int | str, datapoint: DataPoint):
        # node_id = self._root_clade_dict[node_id]
        tree_holder_builder = self._add_num_children_and_node_id(node_id)

        node_obj = self._root_nodes_dict[node_id].copy()

        graph = self._tree_info.build_graph_shell()
        node_idx_dict = self._tree_info.get_node_idx_dict()
        node_idx_rev_dict = self._tree_info.get_node_idx_rev_dict()

        node_data = self._add_node_data_graph_and_index_dicts(
            datapoint,
            graph,
            node_id,
            node_idx_dict,
            node_idx_rev_dict,
            node_obj,
            tree_holder_builder,
        )

        self._update_likelihood_computations_added_datapoint(node_id, node_obj, tree_holder_builder)

        roots_num_children = self.roots_num_children.copy()
        roots_num_children[self._root_node_name] = len(graph.successors(self._root_idx))
        tree_holder_builder.with_roots_num_children(roots_num_children)

        roots_num_desc = self.roots_num_desc.copy()
        roots_num_desc[self._root_node_name] = len(rx.descendants(graph, self._root_idx))
        tree_holder_builder.with_roots_num_desc(roots_num_desc)

        tree_holder_builder.with_roots(list(self._root_node_names_set))

        self._compute_new_log_pdf_added_datapoint(node_id, self._root_node_names_set, tree_holder_builder, node_data)

        tree_holder_builder.set_hash_val()

        return tree_holder_builder

    def _update_likelihood_computations_added_datapoint(self, node_id, node_obj, tree_holder_builder):
        root_node_obj = self._dummy_root_obj.copy()
        sub_roots = self._root_node_names_set - {node_id}
        sub_root_log_r_values = [self._root_nodes_dict[sub_root].log_r for sub_root in sub_roots]
        sub_root_log_r_values.append(node_obj.log_r)
        root_node_obj.update_node_from_child_r_vals(sub_root_log_r_values)
        tree_holder_builder.with_root_node_object(root_node_obj)

    def _add_node_data_graph_and_index_dicts(
        self,
        datapoint,
        graph,
        node_id,
        node_idx_dict,
        node_idx_rev_dict,
        node_obj,
        tree_holder_builder,
    ):
        node_obj.add_data_point(datapoint)
        node_data = self._tree_info.get_node_data()
        node_data[node_id].append(datapoint)
        tree_holder_builder.with_node_data(node_data)
        tree_holder_builder.with_graph(graph)
        tree_holder_builder.with_node_idx(node_idx_dict)
        tree_holder_builder.with_node_idx_rev(node_idx_rev_dict)
        return node_data

    def create_tree_holder_with_new_node(
        self,
        children: frozenset[int | str] | list[int | str] | None,
        datapoint: DataPoint,
    ):
        if children is None:
            children = set()
        else:
            children = set(children)

        # children = {self._root_clade_dict[c] for c in children}

        node_id = self._next_node_id

        tree_holder_builder = self._add_num_children_and_node_id(node_id)

        graph = self._tree_info.build_graph_shell()
        node_idx_dict = self._tree_info.get_node_idx_dict()
        node_idx_rev_dict = self._tree_info.get_node_idx_rev_dict()

        new_node_obj, node_idx = self._attach_new_node_to_graph(
            children, graph, node_id, node_idx_dict, node_idx_rev_dict
        )

        _ = self._add_node_data_graph_and_index_dicts(
            datapoint,
            graph,
            node_id,
            node_idx_dict,
            node_idx_rev_dict,
            new_node_obj,
            tree_holder_builder,
        )

        sub_roots = self._root_node_names_set - children

        self._update_likelihood_computations_new_node(new_node_obj, children, tree_holder_builder, sub_roots)
        self._add_roots_num_children(graph, new_node_obj, node_idx, sub_roots, tree_holder_builder)
        self._add_roots_num_desc(graph, new_node_obj, node_idx, sub_roots, tree_holder_builder)
        roots_list = list(sub_roots)
        roots_list.append(node_id)
        tree_holder_builder.with_roots(roots_list)

        self._compute_new_log_pdf_new_node(children, node_id, roots_list, tree_holder_builder)

        tree_holder_builder.set_hash_val()

        return tree_holder_builder

    def _attach_new_node_to_graph(self, children, graph, node_id, node_idx_dict, node_idx_rev_dict):
        new_node_obj = TreeNode(self._grid_size, self._log_prior, node_id)
        node_idx = graph.add_child(self._root_idx, new_node_obj, None)
        node_idx_dict[node_id] = node_idx
        node_idx_rev_dict[node_idx] = node_id
        if len(children) > 0:
            child_indices = [node_idx_dict[child] for child in children]
            graph.insert_node_on_in_edges_multiple(node_idx, child_indices)
        return new_node_obj, node_idx

    def _add_num_children_and_node_id(self, node_id):
        tree_holder_builder = self._get_initial_tree_holder_builder()
        tree_holder_builder.with_node_last_added_to(node_id)
        tree_holder_builder.with_tree_dist(self.tree_dist)
        return tree_holder_builder

    def _compute_new_log_pdf_added_outlier(self, roots, tree_holder_builder):
        if self.perm_dist:
            new_num_datapoints = self._num_datapoints + 1
            num_outliers = len(tree_holder_builder.outliers)
            log_pdf = -self.perm_dist.log_count_root(self._perm_dist_dict, roots, new_num_datapoints, num_outliers)
        else:
            log_pdf = 0.0
        tree_holder_builder.with_log_pdf(log_pdf)

    def _compute_new_log_pdf_added_datapoint(self, node_id, roots, tree_holder_builder, node_data):
        if self.perm_dist:
            orig_tuple = self._perm_dist_dict[node_id]
            node_data_len = len(node_data[node_id])
            old_node_data_len = node_data_len - 1
            old_log_factorial = cached_log_factorial(old_node_data_len)
            new_log_factorial = cached_log_factorial(node_data_len)
            new_log_count = orig_tuple[0] - old_log_factorial + new_log_factorial
            self._perm_dist_dict[node_id] = (new_log_count, orig_tuple[1] + 1)
            new_num_datapoints = self._num_datapoints + 1
            num_outliers = len(tree_holder_builder.outliers)
            log_pdf = -self.perm_dist.log_count_root(self._perm_dist_dict, roots, new_num_datapoints, num_outliers)
            self._perm_dist_dict[node_id] = orig_tuple
        else:
            log_pdf = 0.0
        tree_holder_builder.with_log_pdf(log_pdf)

    def _compute_new_log_pdf_new_node(self, children, node_id, roots, tree_holder_builder):
        if self.perm_dist:
            count = self.perm_dist.log_count_subroot_from_precomp(self._perm_dist_dict, children, 1)
            subtree_data_len = sum(self._perm_dist_dict[child][1] for child in children) + 1
            self._perm_dist_dict[node_id] = (count, subtree_data_len)
            new_num_datapoints = self._num_datapoints + 1
            num_outliers = len(tree_holder_builder.outliers)
            log_pdf = -self.perm_dist.log_count_root(self._perm_dist_dict, roots, new_num_datapoints, num_outliers)
            self._perm_dist_dict.pop(node_id)
        else:
            log_pdf = 0.0
        tree_holder_builder.with_log_pdf(log_pdf)

    def _add_roots_num_desc(
        self,
        graph: rx.PyDiGraph,
        new_node_obj: TreeNode,
        node_idx,
        sub_roots,
        tree_holder_builder: TreeHolderBuilder,
    ):
        roots_num_desc = {root: self.roots_num_desc[root] for root in sub_roots}
        roots_num_desc[new_node_obj.node_id] = len(rx.descendants(graph, node_idx))
        roots_num_desc[self._root_node_name] = len(rx.descendants(graph, self._root_idx))
        tree_holder_builder.with_roots_num_desc(roots_num_desc)

    def _add_roots_num_children(
        self,
        graph: rx.PyDiGraph,
        new_node_obj: TreeNode,
        node_idx,
        sub_roots,
        tree_holder_builder: TreeHolderBuilder,
    ):
        roots_num_children = {root: self.roots_num_children[root] for root in sub_roots}
        roots_num_children[new_node_obj.node_id] = len(graph.successors(node_idx))
        roots_num_children[self._root_node_name] = len(graph.successors(self._root_idx))
        tree_holder_builder.with_roots_num_children(roots_num_children)

    def _get_initial_tree_holder_builder(self) -> TreeHolderBuilder:
        tree_holder_builder = TreeHolderBuilder(
            self._outlier_node_name,
            self._root_node_name,
            self._grid_size,
            self._log_prior,
            self._nodes,
        )
        return tree_holder_builder

    def _update_likelihood_computations_new_node(
        self,
        node_obj: TreeNode,
        children,
        tree_holder_builder: TreeHolderBuilder,
        sub_roots,
    ):
        child_log_r_values = [self._root_nodes_dict[child].log_r for child in children]
        node_obj.update_node_from_child_r_vals(child_log_r_values)

        root_node_obj = self._dummy_root_obj.copy()

        sub_root_log_r_values = [self._root_nodes_dict[sub_root].log_r for sub_root in sub_roots]
        sub_root_log_r_values.append(node_obj.log_r)
        root_node_obj.update_node_from_child_r_vals(sub_root_log_r_values)

        tree_holder_builder.with_root_node_object(root_node_obj)
