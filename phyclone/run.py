"""
Created on 2012-02-08

@author: Andrew Roth
"""
import Bio.Phylo
from io import StringIO
import gzip
import numpy as np
import pandas as pd
import pickle
from numba import set_num_threads
from dataclasses import dataclass
import os

from phyclone.concentration import GammaPriorConcentrationSampler
from phyclone.map import get_map_node_ccfs
from phyclone.mcmc.gibbs_mh import DataPointSampler, PruneRegraphSampler
from phyclone.mcmc.particle_gibbs import ParticleGibbsSubtreeSampler, ParticleGibbsTreeSampler
from phyclone.smc.kernels import BootstrapKernel, FullyAdaptedKernel, SemiAdaptedKernel
from phyclone.smc.samplers import SMCSampler
from phyclone.smc.utils import RootPermutationDistribution
from phyclone.tree import FSCRPDistribution, Tree, TreeJointDistribution
from phyclone.utils import Timer
from phyclone.tree_utils import create_cache_info_file, clear_function_caches
from phyclone.consensus import get_consensus_tree

import phyclone.data.pyclone
import phyclone.math_utils
from phyclone.math_utils import discrete_rvs, exp_normalize


def write_map_results(in_file, out_table_file, out_tree_file, out_log_probs_file=None, topology_report=False):
    set_num_threads(1)
    with gzip.GzipFile(in_file, "rb") as fh:
        results = pickle.load(fh)

    map_iter = 0

    map_val = float("-inf")

    topologies = []

    for i, x in enumerate(results["trace"]):
        if x["log_p"] > map_val:
            map_iter = i

            map_val = x["log_p"]

        if topology_report:
            count_topology(topologies, x)

    data = results["data"]

    tree = Tree.from_dict(data, results["trace"][map_iter]["tree"])

    clusters = results.get("clusters", None)

    table = get_clone_table(data, results["samples"], tree, clusters=clusters)

    _create_results_output_files(out_log_probs_file, out_table_file, out_tree_file, results, table, tree)
    if topology_report:
        _create_topology_result_file(topologies, out_table_file, data)


def count_topology(topologies, x):
    found = False
    for topology in topologies:
        top = topology['topology']
        if top == x['tree']:
            topology['count'] += 1
            curr_log_p = x['log_p']
            if curr_log_p > topology['log_p_max']:
                topology['log_p_max'] = curr_log_p
            found = True
            break
    if not found:
        topologies.append({'topology': x['tree'], 'count': 1, 'log_p_max': x['log_p']})


def _create_results_output_files(out_log_probs_file, out_table_file, out_tree_file, results, table, tree):
    table.to_csv(out_table_file, index=False, sep="\t")
    Bio.Phylo.write(get_bp_tree_from_graph(tree.graph), out_tree_file, "newick", plain=True)
    if out_log_probs_file:
        log_probs_table = pd.DataFrame(results["trace"], columns=['iter', 'time', 'log_p'])
        log_probs_table.to_csv(out_log_probs_file, index=False, sep="\t")


def _create_topology_result_file(topologies, out_table_file, data):
    tmp_str_io = StringIO()
    for topology in topologies:
        tree = Tree.from_dict(data, topology['topology'])
        Bio.Phylo.write(get_bp_tree_from_graph(tree.graph), tmp_str_io, "newick", plain=True)
        as_str = tmp_str_io.getvalue().rstrip()
        topology['topology'] = as_str
        tmp_str_io.seek(0)

    out_file = os.path.join(os.path.dirname(out_table_file), 'topology_info.tsv')
    df = pd.DataFrame(topologies)
    df.to_csv(out_file, index=False, sep="\t")


def write_consensus_results(in_file, out_table_file, out_tree_file, out_log_probs_file=None, consensus_threshold=0.5):
    set_num_threads(1)
    with gzip.GzipFile(in_file, "rb") as fh:
        results = pickle.load(fh)

    data = results["data"]

    trees = [Tree.from_dict(data, x["tree"]) for x in results["trace"]]

    probs = np.array([x["log_p"] for x in results["trace"]])

    probs, norm = exp_normalize(probs)

    graph = get_consensus_tree(trees, data=data, threshold=consensus_threshold, log_p_list=probs)

    tree = get_tree_from_consensus_graph(data, graph)

    clusters = results.get("clusters", None)

    table = get_clone_table(data, results["samples"], tree, clusters=clusters)

    table = pd.DataFrame(table)

    _create_results_output_files(out_log_probs_file, out_table_file, out_tree_file, results, table, tree)


def get_clades(tree, source=None):
    if source is None:
        roots = []

        for node in tree.nodes:
            if tree.in_degree(node) == 0:
                roots.append(node)

        children = []
        for node in roots:
            children.append(get_clades(tree, source=node))

        clades = Bio.Phylo.BaseTree.Clade(name="root", clades=children)

    else:
        children = []

        for child in tree.successors(source):
            children.append(get_clades(tree, source=child))

        clades = Bio.Phylo.BaseTree.Clade(name=str(source), clades=children)

    return clades


def get_bp_tree_from_graph(tree):
    return Bio.Phylo.BaseTree.Tree(root=get_clades(tree), rooted=True)


def get_tree_from_consensus_graph(data, graph):
    labels = {}

    for node in graph.nodes:
        for idx in graph.nodes[node]["idxs"]:
            labels[idx] = node

    for x in data:
        if x.idx not in labels:
            labels[x.idx] = -1

    graph = graph.copy()

    nodes = list(graph.nodes)

    for node in nodes:
        if len(list(graph.predecessors(node))) == 0:
            graph.add_edge("root", node)

    tree = Tree.from_dict(data, {"graph": graph, "labels": labels})

    tree.update()

    return tree


def get_clone_table(data, samples, tree, clusters=None):
    labels = get_labels_table(data, tree, clusters=clusters)

    ccfs = get_map_node_ccfs(tree)

    table = []

    for _, row in labels.iterrows():
        for i, sample_id in enumerate(samples):
            new_row = row.copy()

            new_row["sample_id"] = sample_id

            if new_row["clone_id"] in ccfs:
                new_row["ccf"] = ccfs[new_row["clone_id"]][i]

            else:
                new_row["ccf"] = -1

            table.append(new_row)

    return pd.DataFrame(table)


def get_labels_table(data, tree, clusters=None):
    df = []

    clone_muts = set()

    if clusters is None:
        for idx in tree.labels:
            df.append({
                "mutation_id": data[idx].name,
                "clone_id": tree.labels[idx],
            })

            clone_muts.add(data[idx].name)

        for x in data:
            if x.name not in clone_muts:
                df.append({
                    "mutation_id": x.name,
                    "clone_id": -1
                })

        df = pd.DataFrame(df, columns=["mutation_id", "clone_id"])

        df = df.sort_values(by=["clone_id", "mutation_id"])

    else:
        for idx in tree.labels:
            muts = clusters[clusters["cluster_id"] == int(data[idx].name)]["mutation_id"]

            for mut in muts:
                df.append({
                    "mutation_id": mut,
                    "clone_id": tree.labels[idx],
                    "cluster_id": int(data[idx].name)
                })

                clone_muts.add(mut)

        clusters = clusters.set_index("mutation_id")

        for mut in clusters.index.values:
            if mut not in clone_muts:
                df.append({
                    "mutation_id": mut,
                    "clone_id": -1,
                    "cluster_id": clusters.loc[mut].values[0]
                })

        df = pd.DataFrame(df, columns=["mutation_id", "clone_id", "cluster_id"])

        df = df.sort_values(by=["clone_id", "cluster_id", "mutation_id"])

    return df


def run(
        in_file,
        out_file,
        burnin=100,
        cluster_file=None,
        concentration_value=1.0,
        concentration_update=True,
        density="beta-binomial",
        grid_size=101,
        max_time=float("inf"),
        num_iters=1000,
        num_particles=20,
        num_samples_data_point=1,
        num_samples_prune_regraph=1,
        outlier_prob=0,
        precision=1.0,
        print_freq=100,
        proposal="fully-adapted",
        resample_threshold=0.5,
        seed=None,
        subtree_update_prob=0,
        thin=1,
        num_threads=1,
        mitochondrial=False):
    rng = instantiate_and_seed_RNG(seed)

    set_num_threads(num_threads)

    data, samples = phyclone.data.pyclone.load_data(
        in_file, cluster_file=cluster_file, density=density, grid_size=grid_size, outlier_prob=outlier_prob,
        precision=precision, mitochondrial=mitochondrial
    )

    tree_dist = TreeJointDistribution(FSCRPDistribution(concentration_value))

    kernel = setup_kernel(outlier_prob, proposal, rng, tree_dist)

    samplers = setup_samplers(kernel,
                              num_particles,
                              outlier_prob,
                              resample_threshold,
                              rng,
                              tree_dist)

    tree = Tree.get_single_node_tree(data, kernel.memo_logs)

    timer = Timer()

    # =========================================================================
    # Burnin
    # =========================================================================
    tree = _run_burnin(burnin, max_time, num_samples_data_point, num_samples_prune_regraph, print_freq, samplers, timer,
                       tree, tree_dist)

    # =========================================================================
    # Main sampler
    # =========================================================================

    trace = setup_trace(timer, tree, tree_dist)

    results = _run_main_sampler(concentration_update, data, max_time, num_iters, num_samples_data_point,
                                num_samples_prune_regraph, print_freq, rng, samplers, samples, subtree_update_prob,
                                thin, timer, trace, tree, tree_dist)

    _create_main_run_output(cluster_file, out_file, results)


def _create_main_run_output(cluster_file, out_file, results):
    if cluster_file is not None:
        results["clusters"] = pd.read_csv(cluster_file, sep="\t")[["mutation_id", "cluster_id"]].drop_duplicates()
    with gzip.GzipFile(out_file, mode="wb") as fh:
        pickle.dump(results, fh)

    cache_txt_file = os.path.join(os.path.dirname(out_file), 'cache_info.txt')
    create_cache_info_file(cache_txt_file)


def _run_main_sampler(concentration_update, data, max_time, num_iters, num_samples_data_point,
                      num_samples_prune_regraph, print_freq, rng, samplers, samples, subtree_update_prob, thin, timer,
                      trace, tree, tree_dist):
    dp_sampler = samplers.dp_sampler
    prg_sampler = samplers.prg_sampler
    subtree_sampler = samplers.subtree_sampler
    tree_sampler = samplers.tree_sampler
    conc_sampler = samplers.conc_sampler

    for i in range(num_iters):
        with timer:
            if i % print_freq == 0:
                print_stats(i, tree, tree_dist)

            if rng.random() < subtree_update_prob:
                tree = subtree_sampler.sample_tree(tree)

            else:
                tree = tree_sampler.sample_tree(tree)

            for _ in range(num_samples_data_point):
                tree = dp_sampler.sample_tree(tree)

            for _ in range(num_samples_prune_regraph):
                tree = prg_sampler.sample_tree(tree)

            tree.relabel_nodes()

            if concentration_update:
                node_sizes = []

                for node, node_data in tree.node_data.items():
                    if node == -1:
                        continue

                    node_sizes.append(len(node_data))

                tree_dist.prior.alpha = conc_sampler.sample(tree_dist.prior.alpha, len(tree.nodes), sum(node_sizes))

            if i % thin == 0:
                trace.append({
                    "iter": i,
                    "time": timer.elapsed,
                    "alpha": tree_dist.prior.alpha,
                    "log_p": tree_dist.log_p_one(tree),
                    "tree": tree.to_dict()
                })

            if timer.elapsed >= max_time:
                break
    results = {"data": data, "samples": samples, "trace": trace}
    return results


def setup_trace(timer, tree, tree_dist):
    trace = []
    trace.append({
        "iter": 0,
        "time": timer.elapsed,
        "alpha": tree_dist.prior.alpha,
        "log_p": tree_dist.log_p_one(tree),
        "tree": tree.to_dict()
    })
    return trace


def _run_burnin(burnin, max_time, num_samples_data_point, num_samples_prune_regraph, print_freq, samplers, timer, tree,
                tree_dist):
    burnin_sampler = samplers.burnin_sampler
    dp_sampler = samplers.dp_sampler
    prg_sampler = samplers.prg_sampler
    if burnin > 0:
        print("#" * 100)
        print("Burnin")
        print("#" * 100)

        for i in range(burnin):
            with timer:
                if i % print_freq == 0:
                    print_stats(i, tree, tree_dist)

                tree = burnin_sampler.sample_tree(tree)

                for _ in range(num_samples_data_point):
                    tree = dp_sampler.sample_tree(tree)

                for _ in range(num_samples_prune_regraph):
                    tree = prg_sampler.sample_tree(tree)

                tree.relabel_nodes()

                if timer.elapsed > max_time:
                    break

    print()
    print("#" * 100)
    print("Post-burnin")
    print("#" * 100)
    print()

    # clear_function_caches()
    return tree


class UnconditionalSMCSampler(object):

    def __init__(self, kernel, num_particles=20, resample_threshold=0.5):
        self.kernel = kernel

        self.num_particles = num_particles

        self.resample_threshold = resample_threshold

        self._rng = kernel.rng

    def sample_tree(self, tree):
        data_sigma = RootPermutationDistribution.sample(tree, self._rng)

        smc_sampler = SMCSampler(
            data_sigma, self.kernel, num_particles=self.num_particles, resample_threshold=self.resample_threshold
        )

        swarm = smc_sampler.sample()

        idx = discrete_rvs(swarm.weights, self._rng)

        return swarm.particles[idx].tree


@dataclass
class SamplersHolder:
    dp_sampler: DataPointSampler
    prg_sampler: PruneRegraphSampler
    conc_sampler: GammaPriorConcentrationSampler
    burnin_sampler: UnconditionalSMCSampler
    tree_sampler: ParticleGibbsTreeSampler
    subtree_sampler: ParticleGibbsSubtreeSampler


def setup_samplers(kernel, num_particles, outlier_prob, resample_threshold, rng, tree_dist):
    dp_sampler = DataPointSampler(tree_dist, rng, outliers=(outlier_prob > 0))
    prg_sampler = PruneRegraphSampler(tree_dist, rng)
    conc_sampler = GammaPriorConcentrationSampler(0.01, 0.01, rng=rng)
    burn_in_particles = int(max(1, np.rint(num_particles / 2)))
    burnin_sampler = UnconditionalSMCSampler(
        kernel, num_particles=burn_in_particles, resample_threshold=resample_threshold
    )
    tree_sampler = ParticleGibbsTreeSampler(
        kernel, rng, num_particles=num_particles, resample_threshold=resample_threshold
    )
    subtree_sampler = ParticleGibbsSubtreeSampler(
        kernel, rng, num_particles=num_particles, resample_threshold=resample_threshold
    )
    return SamplersHolder(dp_sampler,
                          prg_sampler,
                          conc_sampler,
                          burnin_sampler,
                          tree_sampler,
                          subtree_sampler)


def setup_kernel(outlier_prob, proposal, rng, tree_dist):
    if outlier_prob > 0:
        outlier_proposal_prob = 0.1
    else:
        outlier_proposal_prob = 0
    kernel_cls = FullyAdaptedKernel
    if proposal == "bootstrap":
        kernel_cls = BootstrapKernel
    elif proposal == "fully-adapted":
        kernel_cls = FullyAdaptedKernel
    elif proposal == "semi-adapted":
        kernel_cls = SemiAdaptedKernel
    memo_logs = {"log_p": {}, "log_r": {}, "log_s": {}}
    kernel = kernel_cls(tree_dist, memo_logs, rng, outlier_proposal_prob=outlier_proposal_prob)
    return kernel


def instantiate_and_seed_RNG(seed):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    return rng


def print_stats(iter_id, tree, tree_dist):
    string_template = 'iter: {}, alpha: {}, log_p: {}, num_nodes: {}, num_outliers: {}, num_roots: {}'
    print(string_template.format(iter_id, round(tree_dist.prior.alpha, 3), round(tree_dist.log_p_one(tree), 3),
          len(tree.nodes), len(tree.outliers), len(tree.roots)))
