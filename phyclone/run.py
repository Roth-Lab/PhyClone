"""
Created on 2012-02-08

@author: Andrew Roth
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np

from phyclone.data.pyclone import load_data
from phyclone.mcmc.concentration import GammaPriorConcentrationSampler
from phyclone.mcmc.gibbs_mh import DataPointSampler, PruneRegraphSampler
from phyclone.mcmc.particle_gibbs import (
    ParticleGibbsTreeSampler,
    ParticleGibbsSubtreeSampler,
)
from phyclone.process_trace import create_main_run_output
from phyclone.smc.kernels import BootstrapKernel, FullyAdaptedKernel, SemiAdaptedKernel
from phyclone.smc.samplers import UnconditionalSMCSampler
from phyclone.tree import FSCRPDistribution, Tree, TreeJointDistribution
from phyclone.utils import Timer
from phyclone.utils.dev import clear_proposal_dist_caches, clear_convolution_caches


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
    num_iters=5000,
    num_particles=100,
    num_samples_data_point=1,
    num_samples_prune_regraph=1,
    outlier_prob=0,
    precision=1.0,
    print_freq=100,
    proposal="semi-adapted",
    resample_threshold=0.5,
    seed=None,
    thin=1,
    num_chains=1,
    subtree_update_prob=0.0,
    low_loss_prob=0.0001,
    high_loss_prob=0.4,
    assign_loss_prob=False,
    user_provided_loss_prob=False,
    run_pre_burn=False,
):

    rng_main = instantiate_and_seed_RNG(seed)

    if assign_loss_prob and user_provided_loss_prob:
        raise Exception(
            "Cannot use both --assign-loss-prob and --user-provided-loss-prob, these options are mutually exclusive."
        )

    if assign_loss_prob and outlier_prob == 0:
        outlier_prob = 0.0001

    if user_provided_loss_prob and outlier_prob == 0:
        outlier_prob = 0.0001

    outlier_modelling_active = outlier_prob > 0

    print_welcome_message(
        burnin, density, num_chains, num_iters, num_particles, seed, outlier_modelling_active, rng_main, proposal
    )

    data, samples, init_sigma = load_data(
        in_file,
        rng_main,
        low_loss_prob,
        high_loss_prob,
        assign_loss_prob,
        cluster_file=cluster_file,
        density=density,
        grid_size=grid_size,
        outlier_prob=outlier_prob,
        precision=precision,
    )

    if not run_pre_burn:
        init_sigma = None

    results = {}

    if num_chains == 1:
        results[0] = run_phyclone_chain(
            burnin,
            concentration_update,
            concentration_value,
            data,
            max_time,
            num_iters,
            num_particles,
            num_samples_data_point,
            num_samples_prune_regraph,
            outlier_modelling_active,
            print_freq,
            proposal,
            resample_threshold,
            rng_main,
            samples,
            thin,
            0,
            subtree_update_prob,
            init_sigma,
        )

        print("Finished chain", 0)

    else:

        rng_list = rng_main.spawn(num_chains)

        with ProcessPoolExecutor(max_workers=num_chains, mp_context=get_context("spawn")) as pool:
            chain_results = [
                pool.submit(
                    run_phyclone_chain,
                    burnin,
                    concentration_update,
                    concentration_value,
                    data,
                    max_time,
                    num_iters,
                    num_particles,
                    num_samples_data_point,
                    num_samples_prune_regraph,
                    outlier_modelling_active,
                    print_freq,
                    proposal,
                    resample_threshold,
                    rng,
                    samples,
                    thin,
                    chain_num,
                    subtree_update_prob,
                    init_sigma,
                )
                for chain_num, rng in enumerate(rng_list)
            ]

            for future in as_completed(chain_results):
                exception = future.exception()
                if exception is not None:
                    raise exception
                else:
                    result = future.result()
                    res_chain = result["chain_num"]
                    results[res_chain] = result
                    print("Finished chain", res_chain)

    create_main_run_output(cluster_file, out_file, results)


def print_welcome_message(
    burnin,
    density,
    num_chains,
    num_iters,
    num_particles,
    seed,
    outlier_modelling_active,
    rng_main,
    proposal_kernel,
):
    print()
    print("#" * 100)
    print("PhyClone - Analysis Run")
    print("#" * 100)
    print()
    print("Running with the following parameters:\n")
    print("Number of independent chains: {}".format(num_chains))
    print("Number of PG particles: {}".format(num_particles))
    print("Density: {}".format(density))
    print("Proposal distribution: {}".format(proposal_kernel))
    print("Number of burn-in iterations: {}".format(burnin))
    print("Number of MCMC iterations: {}".format(num_iters))
    if seed is not None:
        seed_msg = "(user-provided)"
    else:
        seed_msg = "(machine-entropy)"
    print("Random seed: {} {}".format(rng_main.bit_generator.seed_seq.entropy, seed_msg))
    print("Outlier modelling allowed: {}".format(outlier_modelling_active))
    print()
    print("#" * 100)
    print()


def run_phyclone_chain(
    burnin,
    concentration_update,
    concentration_value,
    data,
    max_time,
    num_iters,
    num_particles,
    num_samples_data_point,
    num_samples_prune_regraph,
    outlier_modelling_active,
    print_freq,
    proposal,
    resample_threshold,
    rng,
    samples,
    thin,
    chain_num,
    subtree_update_prob,
    init_sigma,
):
    tree_dist = TreeJointDistribution(FSCRPDistribution(concentration_value), outlier_modelling_active)
    kernel = setup_kernel(outlier_modelling_active, proposal, rng, tree_dist)
    samplers = setup_samplers(kernel, num_particles, outlier_modelling_active, resample_threshold, rng, tree_dist)
    tree = Tree.get_single_node_tree(data)
    timer = Timer()
    if init_sigma:
        tree = _run_sigma_init_iter(
            max_time,
            num_samples_data_point,
            num_samples_prune_regraph,
            samplers,
            timer,
            tree,
            tree_dist,
            chain_num,
            init_sigma,
            concentration_update,
        )
    tree = _run_burnin(
        burnin,
        max_time,
        num_samples_data_point,
        num_samples_prune_regraph,
        print_freq,
        samplers,
        timer,
        tree,
        tree_dist,
        chain_num,
    )
    results = _run_main_sampler(
        concentration_update,
        data,
        max_time,
        num_iters,
        num_samples_data_point,
        num_samples_prune_regraph,
        print_freq,
        samplers,
        samples,
        thin,
        timer,
        tree,
        tree_dist,
        chain_num,
        rng,
        subtree_update_prob,
    )
    return results


def _run_main_sampler(
    concentration_update,
    data,
    max_time,
    num_iters,
    num_samples_data_point,
    num_samples_prune_regraph,
    print_freq,
    samplers,
    samples,
    thin,
    timer,
    tree,
    tree_dist,
    chain_num,
    rng,
    subtree_update_prob,
):
    clear_convolution_caches()
    trace = setup_trace(timer, tree, tree_dist)

    dp_sampler = samplers.dp_sampler
    prg_sampler = samplers.prg_sampler
    tree_sampler = samplers.tree_sampler
    conc_sampler = samplers.conc_sampler
    subtree_sampler = samplers.subtree_sampler

    for i in range(num_iters):
        with timer:
            if i % print_freq == 0:
                print_stats(i, tree, tree_dist, chain_num)

            clear_proposal_dist_caches()

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
                update_concentration_value(conc_sampler, tree, tree_dist)

            if i % thin == 0:
                append_to_trace(i, timer, trace, tree, tree_dist)

            if timer.elapsed >= max_time:
                break
    results = {"data": data, "samples": samples, "trace": trace, "chain_num": chain_num}
    return results


def append_to_trace(i, timer, trace, tree, tree_dist):
    trace.append(
        {
            "iter": i,
            "time": timer.elapsed,
            "alpha": tree_dist.prior.alpha,
            "log_p_one": tree_dist.log_p_one(tree),
            "tree": tree.to_dict(),
        }
    )


def update_concentration_value(conc_sampler, tree, tree_dist):
    node_sizes = []
    outlier_node_name = tree.outlier_node_name
    for node, node_data in tree.node_data.items():
        if node == outlier_node_name:
            continue

        node_sizes.append(len(node_data))

    tree_dist.prior.alpha = conc_sampler.sample(tree_dist.prior.alpha, len(node_sizes), sum(node_sizes))


def setup_trace(timer, tree, tree_dist):
    trace = []
    append_to_trace(0, timer, trace, tree, tree_dist)
    return trace


def _run_burnin(
    burnin,
    max_time,
    num_samples_data_point,
    num_samples_prune_regraph,
    print_freq,
    samplers,
    timer,
    tree,
    tree_dist,
    chain_num,
):
    burnin_sampler = samplers.burnin_sampler
    dp_sampler = samplers.dp_sampler
    prg_sampler = samplers.prg_sampler
    best_tree = tree

    if burnin > 0:
        best_score = -np.inf
        print("#" * 100)
        print("Burnin")
        print("#" * 100)

        for i in range(burnin):
            with timer:
                if i % print_freq == 0:
                    print_stats(i, tree, tree_dist, chain_num)

                clear_proposal_dist_caches()

                tree = burnin_sampler.sample_tree(tree)

                for _ in range(num_samples_data_point):
                    tree = dp_sampler.sample_tree(tree)

                for _ in range(num_samples_prune_regraph):
                    tree = prg_sampler.sample_tree(tree)

                tree.relabel_nodes()

                tree_score = tree_dist.log_p_one(tree)
                if tree_score > best_score:
                    best_score = tree_score
                    best_tree = tree

                if timer.elapsed > max_time:
                    break
        print_stats(burnin, tree, tree_dist, chain_num)
    print()
    print("#" * 100)
    print("Post-burnin")
    print("#" * 100)
    print()

    return best_tree


def _run_sigma_init_iter(
    max_time,
    num_samples_data_point,
    num_samples_prune_regraph,
    samplers,
    timer,
    tree,
    tree_dist,
    chain_num,
    sigma_init,
    concentration_update,
):
    burnin_sampler = samplers.burnin_sampler
    dp_sampler = samplers.dp_sampler
    prg_sampler = samplers.prg_sampler
    # conc_sampler = samplers.conc_sampler

    sigma_init_iters = 100
    print_freq = round(sigma_init_iters / 2)

    best_tree = tree
    best_score = -np.inf
    # assoc_alpha = tree_dist.prior.alpha

    print("#" * 100)
    print("Pre-Burn: Single Sigma Data Initialization")
    print("#" * 100)

    for i in range(sigma_init_iters):
        with timer:
            if i % print_freq == 0:
                print_stats(i, tree, tree_dist, chain_num)

            clear_proposal_dist_caches()

            tree = burnin_sampler.sample_tree(tree, sigma_init)

            for _ in range(num_samples_data_point):
                tree = dp_sampler.sample_tree(tree)

            for _ in range(num_samples_prune_regraph):
                tree = prg_sampler.sample_tree(tree)

            tree.relabel_nodes()

            # if concentration_update:
            #     update_concentration_value(conc_sampler, tree, tree_dist)

            tree_score = tree_dist.log_p_one(tree)
            if tree_score > best_score:
                best_score = tree_score
                best_tree = tree
                # assoc_alpha = tree_dist.prior.alpha

            if timer.elapsed > max_time:
                break
    print_stats(sigma_init_iters, tree, tree_dist, chain_num)

    clear_proposal_dist_caches()

    # tree_dist.prior.alpha = assoc_alpha
    print()
    print("~" * 100)
    print("Post Pre-Burn")
    print("~" * 100)
    print()

    return best_tree


@dataclass
class SamplersHolder:
    dp_sampler: DataPointSampler
    prg_sampler: PruneRegraphSampler
    conc_sampler: GammaPriorConcentrationSampler
    burnin_sampler: UnconditionalSMCSampler
    tree_sampler: ParticleGibbsTreeSampler
    subtree_sampler: ParticleGibbsSubtreeSampler


def setup_samplers(kernel, num_particles, outlier_modelling_active, resample_threshold, rng, tree_dist):
    dp_sampler = DataPointSampler(tree_dist, rng, outliers=outlier_modelling_active)
    prg_sampler = PruneRegraphSampler(tree_dist, rng)
    conc_sampler = GammaPriorConcentrationSampler(0.01, 0.01, rng=rng)
    burnin_sampler = UnconditionalSMCSampler(kernel, num_particles=num_particles, resample_threshold=resample_threshold)
    tree_sampler = ParticleGibbsTreeSampler(
        kernel, rng, num_particles=num_particles, resample_threshold=resample_threshold
    )
    subtree_sampler = ParticleGibbsSubtreeSampler(
        kernel, rng, num_particles=num_particles, resample_threshold=resample_threshold
    )
    return SamplersHolder(
        dp_sampler,
        prg_sampler,
        conc_sampler,
        burnin_sampler,
        tree_sampler,
        subtree_sampler,
    )


def setup_kernel(outlier_modelling_active, proposal, rng, tree_dist):
    kernel_cls = SemiAdaptedKernel
    if proposal == "bootstrap":
        kernel_cls = BootstrapKernel
    elif proposal == "fully-adapted":
        kernel_cls = FullyAdaptedKernel
    elif proposal == "semi-adapted":
        kernel_cls = SemiAdaptedKernel

    kernel = kernel_cls(tree_dist, rng, outlier_modelling_active=outlier_modelling_active, perm_dist=None)
    return kernel


def instantiate_and_seed_RNG(seed):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    return rng


def print_stats(iter_id, tree, tree_dist, chain_num):
    string_template = "chain: {} || iter: {}, alpha: {}, log_p: {}, num_nodes: {}, num_outliers: {}, num_roots: {}"
    print(
        string_template.format(
            chain_num,
            iter_id,
            round(tree_dist.prior.alpha, 3),
            round(tree_dist.log_p_one(tree), 3),
            tree.get_number_of_nodes(),
            len(tree.outliers),
            tree.get_number_of_children(tree.root_node_name),
        )
    )
