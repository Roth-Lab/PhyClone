from sys import maxsize

import click

from phyclone.process_trace import (
    write_map_results,
    write_consensus_results,
    write_topology_report,
)
from phyclone.run import run as run_prog


# =========================================================================
# Consensus Tree Output
# =========================================================================


@click.command(context_settings={"max_content_width": 120})
@click.option(
    "-i",
    "--in-file",
    required=True,
    type=click.Path(resolve_path=True, exists=True),
    help="""Path to trace file from MCMC analysis. Format is gzip compressed Python pickle file.""",
)
@click.option(
    "-o",
    "--out-table-file",
    required=True,
    type=click.Path(resolve_path=True, writable=True),
)
@click.option(
    "-t",
    "--out-tree-file",
    required=True,
    type=click.Path(resolve_path=True, writable=True),
)
@click.option(
    "--consensus-threshold",
    default=0.5,
    type=click.FloatRange(0.0, 1.0, clamp=True),
    show_default=True,
    help="""Consensus threshold to keep an SNV.""",
)
@click.option(
    "-w",
    "--weight-type",
    default="joint-likelihood",
    type=click.Choice(["counts", "joint-likelihood"]),
    show_default=True,
    help="""Which measure to use as the consensus tree weights. Counts is the same as an unweighted consensus.""",
)
def consensus(**kwargs):
    """Build consensus results."""
    write_consensus_results(**kwargs)


# =========================================================================
# MAP Tree Output
# =========================================================================


@click.command(context_settings={"max_content_width": 120})
@click.option(
    "-i",
    "--in-file",
    required=True,
    type=click.Path(resolve_path=True, exists=True),
    help="""Path to trace file from MCMC analysis. Format is gzip compressed Python pickle file.""",
)
@click.option(
    "-o",
    "--out-table-file",
    required=True,
    type=click.Path(resolve_path=True, writable=True),
)
@click.option(
    "-t",
    "--out-tree-file",
    required=True,
    type=click.Path(resolve_path=True, writable=True),
)
@click.option(
    "--map-type",
    default="joint-likelihood",
    type=click.Choice(["joint-likelihood", "frequency"]),
    show_default=True,
    help="""Which measure to use as for MAP computation.""",
)
def map(**kwargs):
    """Build MAP results."""
    write_map_results(**kwargs)


# =========================================================================
# Topology Output
# =========================================================================


@click.command(context_settings={"max_content_width": 120})
@click.option(
    "-i",
    "--in-file",
    required=True,
    type=click.Path(resolve_path=True, exists=True),
    help="""Path to trace file from MCMC analysis. Format is gzip compressed Python pickle file.""",
)
@click.option(
    "-o",
    "--out-file",
    required=True,
    type=click.Path(resolve_path=True, writable=True),
    help="""Path/filename to where topology report will be written in .tsv format""",
)
@click.option(
    "-t",
    "--topologies-archive",
    default=None,
    type=click.Path(resolve_path=True, writable=True),
    help="""To produce the results tables and newick trees for each uniquely sampled topology in the report, provide a
    path to where the archive file will be written in tar.gz compressed format.""",
)
@click.option(
    "--top-trees",
    default=maxsize,
    type=click.IntRange(1, clamp=True),
    help="""Number of uniquely sampled topologies to archive. Default is to produce an archive of all unique 
    topologies.""",
)
def topology_report(**kwargs):
    """Build topology report."""
    write_topology_report(**kwargs)


# =========================================================================
# Analysis
# =========================================================================
@click.command(context_settings={"max_content_width": 120}, name="run")
@click.option(
    "-i",
    "--in-file",
    required=True,
    type=click.Path(exists=True, resolve_path=True),
    help="""Path to TSV format file with copy number and allele count information for all samples. 
    See the examples directory in the GitHub repository for format.""",
)
@click.option(
    "-o",
    "--out-file",
    required=True,
    type=click.Path(resolve_path=True, writable=True),
    help="""Path to where trace file will be written in gzip compressed pickle format.""",
)
@click.option(
    "-b",
    "--burnin",
    default=100,
    type=click.IntRange(1, clamp=True),
    show_default=True,
    help="""Number of burnin iterations using unconditional SMC sampler. Default is 100.""",
)
@click.option(
    "-n",
    "--num-iters",
    default=5000,
    type=click.IntRange(1, clamp=True),
    show_default=True,
    help="""Number of iterations of the MCMC sampler to perform. Default is 5,000.""",
)
@click.option(
    "-t",
    "--thin",
    default=1,
    type=click.IntRange(1, clamp=True),
    show_default=True,
    help="""Thinning parameter for storing entries in trace. Default is 1.""",
)
@click.option(
    "--num-chains",
    default=1,
    type=click.IntRange(1, clamp=True),
    help="""Number of parallel chains for sampling. Recommended to use 4. Default is 1.""",
)
@click.option(
    "-c",
    "--cluster-file",
    default=None,
    type=click.Path(resolve_path=True, exists=True),
    help="""Path to file with pre-computed cluster assignments of mutations.""",
)
@click.option(
    "-d",
    "--density",
    default="beta-binomial",
    type=click.Choice(["binomial", "beta-binomial"]),
    show_default=True,
    help="""Allele count density in the PyClone model. Use beta-binomial for most cases. Default beta-binomial.""",
)
@click.option(
    "-l",
    "--outlier-prob",
    default=0,
    type=click.FloatRange(0.0, 1.0, clamp=True),
    show_default=True,
    help="""Global prior probability that data points are outliers and don't fit tree. Default is 0.0""",
)
@click.option(
    "-p",
    "--proposal",
    default="semi-adapted",
    type=click.Choice(["bootstrap", "fully-adapted", "semi-adapted"]),
    show_default=True,
    help="""
    Proposal distribution to use for PG sampling.
    Fully adapted is the most computationally expensive but also likely to lead to the best performance per iteration.
    For large datasets it may be necessary to use one of the other proposals.
    """,
)
@click.option(
    "-t",
    "--max-time",
    default=float("inf"),
    type=float,
    show_default=True,
    help="""Maximum running time in seconds.""",
)
@click.option(
    "--concentration-update/--no-concentration-update",
    default=True,
    show_default=True,
    help="Whether the concentration parameter should be updated during sampling.",
)
@click.option(
    "--concentration-value",
    default=1.0,
    type=float,
    show_default=True,
    help="""The (initial) concentration of the Dirichlet process. Higher values will encourage more clusters, 
    lower values have the opposite effect. Default is 1.0.""",
)
@click.option(
    "--grid-size",
    default=101,
    type=click.IntRange(11, clamp=True),
    show_default=True,
    help="""Grid size for discrete approximation. This will numerically marginalise the cancer cell fraction. 
    Higher values lead to more accurate approximations at the expense of run time.""",
)
@click.option(
    "--num-particles",
    default=100,
    type=click.IntRange(1, clamp=True),
    show_default=True,
    help="""Number of particles to use during PG sampling.""",
)
@click.option(
    "--num-samples-data-point",
    default=1,
    type=int,
    show_default=True,
    help="""Number of Gibbs updates to reassign data points per SMC iteration. Default is 1.""",
)
@click.option(
    "--num-samples-prune-regraph",
    default=1,
    type=int,
    show_default=True,
    help="""Number of prune-regraph updates per SMC iteration. Default is 1.""",
)
@click.option(
    "-s",
    "--subtree-update-prob",
    default=0.0,
    type=click.FloatRange(0.0, 1.0, clamp=True),
    show_default=True,
    help="""Probability of updating a subtree (instead of whole tree) using PG sampler. Default is 0.0""",
)
@click.option(
    "--precision",
    default=400,
    type=float,
    show_default=True,
    help="""The (initial) precision parameter of the Beta-Binomial density. 
    The higher the value the more similar the Beta-Binomial is to a Binomial. Default is 400.""",
)
@click.option(
    "--print-freq",
    default=100,
    type=int,
    show_default=True,
    help="""How frequently to print information about fitting. Default every 10 iterations.""",
)
@click.option(
    "--resample-threshold",
    default=0.5,
    type=click.FloatRange(0.0, 1.0, clamp=True),
    show_default=True,
    help="""ESS threshold to trigger resampling. Default is 0.5.""",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="""Set random seed so results can be reproduced. By default a random seed is chosen.""",
)
@click.option(
    "--assign-loss-prob/--no-assign-loss-prob",
    default=False,
    show_default=True,
    help="Whether to assign loss probability prior from the cluster data."
    "Note: This option is incompatible with --user-provided-loss-prob",
)
@click.option(
    "--user-provided-loss-prob/--no-user-provided-loss-prob",
    default=False,
    show_default=True,
    help="Whether to use user-provided cluster loss probability prior from the cluster file."
    "Requires that the 'outlier_prob' column be present and populated in the cluster file."
    "Note: This option is incompatible with --assign-loss-prob",
)
@click.option(
    "--low-loss-prob",
    default=0.0001,
    type=click.FloatRange(0.0001, 1.0, clamp=True),
    show_default=True,
    help="""Lower loss probability setting. 
    Used when allowing PhyClone to assign loss prior probability from cluster data.
    Unless combined with the --assign-loss-prob option and a cluster input file, this does nothing.""",
)
@click.option(
    "--high-loss-prob",
    default=0.4,
    type=click.FloatRange(0.0001, 1.0, clamp=True),
    show_default=True,
    help="""Higher loss probability setting. 
    Used when allowing PhyClone to assign loss prior probability from cluster data.
    Unless combined with the --assign-loss-prob option and a cluster input file, this does nothing.""",
)
def run(**kwargs):
    """Run a new PhyClone analysis."""
    run_prog(**kwargs)


# =========================================================================
# Setup main interface
# =========================================================================
@click.group(name="phyclone")
@click.version_option()
def main():
    pass


main.add_command(consensus)
main.add_command(map)
main.add_command(topology_report)
main.add_command(run)
