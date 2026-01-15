"""
Created on 2012-02-08

@author: Andrew Roth
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
import click

from phyclone.data.pyclone import load_data
from phyclone.mcmc.concentration import GammaPriorConcentrationSampler
from phyclone.mcmc.gibbs_mh import DataPointSampler, PruneRegraphSampler
from phyclone.mcmc.particle_gibbs import (
    ParticleGibbsTreeSampler,
    ParticleGibbsSubtreeSampler,
)

from phyclone.smc.kernels import BootstrapKernel, FullyAdaptedKernel, SemiAdaptedKernel
from phyclone.smc.samplers import UnconditionalSMCSampler
from phyclone.tree import FSCRPDistribution, Tree, TreeJointDistribution
from phyclone.utils import Timer, TraceEntry
from phyclone.utils.cache import clear_proposal_dist_caches, clear_all_caches
from phyclone.utils.save_hdf5 import save_trace_to_h5df
from phyclone.utils.utils import print_command_header


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
    high_loss_prob=0.4,
    assign_loss_prob=False,
    user_provided_loss_prob=False,
):

    rng_main = instantiate_and_seed_RNG(seed)

    outlier_modelling_active = outlier_prob > 0

    rng_seed = print_welcome_message(
        burnin,
        density,
        num_chains,
        num_iters,
        num_particles,
        seed,
        outlier_modelling_active,
        rng_main,
        proposal,
    )

    data, samples, minimal_cluster_df = load_data(
        in_file,
        rng_main,
        high_loss_prob,
        assign_loss_prob,
        user_provided_loss_prob,
        cluster_file=cluster_file,
        density=density,
        grid_size=grid_size,
        outlier_prob=outlier_prob,
        precision=precision,
    )

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
            thin,
            0,
            subtree_update_prob,
        )

        click.echo("Finished chain 0")

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
                    thin,
                    chain_num,
                    subtree_update_prob,
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
                    click.echo(f"Finished chain {res_chain}")

    save_trace_to_h5df(results, out_file, minimal_cluster_df, rng_seed, samples, data)


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
    print_command_header("Analysis Run")
    click.echo("Running with the following parameters:\n")
    click.echo("Number of independent chains: {}".format(num_chains))
    click.echo("Number of PG particles: {}".format(num_particles))
    click.echo("Density: {}".format(density))
    click.echo("Proposal distribution: {}".format(proposal_kernel))
    click.echo("Number of burn-in iterations: {}".format(burnin))
    click.echo("Number of MCMC iterations: {}".format(num_iters))
    if seed is not None:
        seed_msg = "(user-provided)"
    else:
        seed_msg = "(machine-entropy)"
        seed = rng_main.bit_generator.seed_seq.entropy
    click.echo("Random seed: {} {}".format(seed, seed_msg))
    click.echo("Outlier modelling allowed: {}".format(outlier_modelling_active))
    click.echo()
    click.echo("#" * 100)
    click.echo()
    return seed


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
    thin,
    chain_num,
    subtree_update_prob,
):
    tree_dist = TreeJointDistribution(FSCRPDistribution(concentration_value), outlier_modelling_active)
    kernel = setup_kernel(outlier_modelling_active, proposal, rng, tree_dist)
    samplers = setup_samplers(kernel, num_particles, outlier_modelling_active, resample_threshold, rng, tree_dist)
    tree = Tree.get_single_node_tree(data)
    timer = Timer()
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
        max_time,
        num_iters,
        num_samples_data_point,
        num_samples_prune_regraph,
        print_freq,
        samplers,
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
    max_time,
    num_iters,
    num_samples_data_point,
    num_samples_prune_regraph,
    print_freq,
    samplers,
    thin,
    timer,
    tree,
    tree_dist,
    chain_num,
    rng,
    subtree_update_prob,
):
    clear_all_caches()
    trace = []

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

            if i % thin == 0:
                append_to_trace(i, timer, trace, tree, tree_dist)

            if timer.elapsed >= max_time:
                break

            if concentration_update:
                update_concentration_value(conc_sampler, tree, tree_dist)

    results = {"trace": trace, "chain_num": chain_num}
    clear_all_caches()
    return results


def append_to_trace(i, timer, trace, tree, tree_dist):
    trace.append(TraceEntry(i, timer, tree, tree_dist))


def update_concentration_value(conc_sampler, tree, tree_dist):
    node_sizes = []
    outlier_node_name = tree.outlier_node_name
    for node, node_data in tree.node_data.items():
        if node == outlier_node_name:
            continue

        node_sizes.append(len(node_data))

    tree_dist.prior.alpha = conc_sampler.sample(tree_dist.prior.alpha, len(node_sizes), sum(node_sizes))


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
        click.echo("#" * 100)
        click.echo(f"Burnin - Chain {chain_num}")
        click.echo("#" * 100)

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

                tree_score = tree_dist.log_p_one(tree)
                if tree_score > best_score:
                    best_score = tree_score
                    best_tree = tree

                if timer.elapsed > max_time:
                    break
        print_stats(burnin, tree, tree_dist, chain_num)

    click.echo()
    click.echo("#" * 100)
    click.echo(f"Post-burnin - Chain {chain_num}")
    click.echo("#" * 100)
    click.echo()

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
    click.echo(
        string_template.format(
            chain_num,
            iter_id,
            round(tree_dist.prior.alpha, 3),
            round(tree_dist.log_p_one(tree), 3),
            tree.get_number_of_nodes(),
            tree.get_number_of_outliers(),
            tree.get_number_of_children(tree.root_node_name),
        ),
    )
