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
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="""Path to trace file from MCMC analysis. Format is HDF5.""",
)
@click.option(
    "-o",
    "--out-table-file",
    required=True,
    type=click.Path(resolve_path=True, writable=True, file_okay=True, dir_okay=False),
)
@click.option(
    "-t",
    "--out-tree-file",
    required=True,
    type=click.Path(resolve_path=True, writable=True, file_okay=True, dir_okay=False),
    help="""Path to where tree will be written in minimal newick format.""",
)
@click.option(
    "-s",
    "--out-sample-prev-table",
    required=True,
    type=click.Path(resolve_path=True, writable=True, file_okay=True, dir_okay=False),
    help="""Path to where sample prevalence table will be written in .tsv format.""",
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
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="""Path to trace file from MCMC analysis. Format is HDF5.""",
)
@click.option(
    "-o",
    "--out-table-file",
    required=True,
    type=click.Path(resolve_path=True, writable=True, file_okay=True, dir_okay=False),
)
@click.option(
    "-t",
    "--out-tree-file",
    required=True,
    type=click.Path(resolve_path=True, writable=True, file_okay=True, dir_okay=False),
    help="""Path to where tree will be written in minimal newick format.""",
)
@click.option(
    "-s",
    "--out-sample-prev-table",
    required=True,
    type=click.Path(resolve_path=True, writable=True, file_okay=True, dir_okay=False),
    help="""Path to where sample prevalence table will be written in .tsv format.""",
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
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="""Path to trace file from MCMC analysis. Format is HDF5.""",
)
@click.option(
    "-o",
    "--out-file",
    required=True,
    type=click.Path(resolve_path=True, writable=True, file_okay=True, dir_okay=False),
    help="""Path/filename to where topology report will be written in .tsv format""",
)
@click.option(
    "-t",
    "--topologies-archive",
    default=None,
    type=click.Path(resolve_path=True, writable=True, file_okay=True, dir_okay=False),
    help="""To produce the results tables and newick trees for each uniquely sampled topology in the report, provide a
    path to where the archive file will be written in tar.gz compressed format.""",
)
@click.option(
    "--top-trees",
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


def _validate_thin(ctx, param, value):
    num_iters = ctx.params["num_iters"]
    if value >= num_iters:
        raise click.BadParameter(
            "Cannot be greater than or equal to number of iterations, as this will result in an empty trace."
        )
    return value


def _validate_assign_loss_prob(ctx, param, value):
    if value:
        if "user_provided_loss_prob" in ctx.params:
            user_provided_loss_prob = ctx.params["user_provided_loss_prob"]
            if user_provided_loss_prob:
                raise click.BadParameter(
                    "Cannot be used with '--user-provided-loss-prob', as these options are mutually exclusive."
                )
    return value


def _validate_user_provided_loss_prob(ctx, param, value):
    if value:
        if "assign_loss_prob" in ctx.params:
            assign_loss_prob = ctx.params["assign_loss_prob"]
            if assign_loss_prob:
                raise click.BadParameter(
                    "Cannot be used with '--assign-loss-prob', as these options are mutually exclusive."
                )
    return value


def _validate_outlier_prob(ctx, param, value):
    if value > 0:
        if ctx.params["assign_loss_prob"]:
            if "high_loss_prob" in ctx.params:
                high_loss_prob = ctx.params["high_loss_prob"]
                if value >= high_loss_prob:
                    raise click.BadParameter(
                        "Value must be lesser than '--high-loss-prob' when using '--assign-loss-prob'"
                    )
        return value
    if ctx.params["assign_loss_prob"] or ctx.params["user_provided_loss_prob"]:
        min_val = 1e-4
        click.echo()
        click.echo(f"As outlier modelling is active, changing '--outlier-prob' from 0.0 to {min_val}.")
        return min_val
    return value


def _validate_high_loss_prob(ctx, param, value):
    if ctx.params["assign_loss_prob"]:
        if "outlier_prob" in ctx.params:
            outlier_prob = ctx.params["outlier_prob"]
            if value <= outlier_prob:
                raise click.BadParameter("Value must be greater than '--outlier-prob'")
    return value


def _validate_positive_value(ctx, param, value):
    if value < 0:
        raise click.BadParameter("Value must be positive.")
    return value


@click.command(context_settings={"max_content_width": 120}, name="run")
@click.option(
    "-i",
    "--in-file",
    required=True,
    type=click.Path(exists=True, resolve_path=True, readable=True, file_okay=True, dir_okay=False),
    help="""Path to TSV format file with copy number and allele count information for all samples. 
    See the examples directory in the GitHub repository for format.""",
)
@click.option(
    "-o",
    "--out-file",
    required=True,
    type=click.Path(resolve_path=True, writable=True, file_okay=True, dir_okay=False),
    help="""Path to where trace file will be written in HDF5 format.""",
)
@click.option(
    "-b",
    "--burnin",
    default=1000,
    type=click.IntRange(1, clamp=True),
    show_default=True,
    help="""Number of burn-in iterations using unconditional SMC sampler.""",
)
@click.option(
    "-n",
    "--num-iters",
    default=10000,
    type=click.IntRange(1, clamp=True),
    show_default=True,
    is_eager=True,
    help="""Number of iterations of the MCMC sampler to perform.""",
)
@click.option(
    "--thin",
    default=1,
    type=click.IntRange(1, clamp=True),
    show_default=True,
    callback=_validate_thin,
    help="""Thinning parameter for storing entries in trace.""",
)
@click.option(
    "--num-chains",
    default=1,
    type=click.IntRange(1, clamp=True),
    show_default=True,
    help="""Number of parallel chains for sampling. Recommended to use at least 4.""",
)
@click.option(
    "-c",
    "--cluster-file",
    default=None,
    type=click.Path(resolve_path=True, exists=True, readable=True, file_okay=True, dir_okay=False),
    help="""Path to file with pre-computed cluster assignments of mutations.""",
)
@click.option(
    "-d",
    "--density",
    default="beta-binomial",
    type=click.Choice(["binomial", "beta-binomial"]),
    show_default=True,
    help="""Allele count density in the PyClone model. Use beta-binomial for most cases.""",
)
@click.option(
    "-l",
    "--outlier-prob",
    default=0,
    type=click.FloatRange(0.0, 1.0, clamp=True),
    show_default=True,
    callback=_validate_outlier_prob,
    help="""Prior probability that data points are outliers and don't fit tree.""",
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
    callback=_validate_positive_value,
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
    callback=_validate_positive_value,
    help="""The (initial) concentration of the Dirichlet process. Higher values will encourage more clusters, 
    lower values have the opposite effect.""",
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
    type=click.IntRange(0, clamp=True),
    show_default=True,
    help="""Number of Gibbs updates to reassign data points per SMC iteration.""",
)
@click.option(
    "--num-samples-prune-regraph",
    default=1,
    type=click.IntRange(0, clamp=True),
    show_default=True,
    help="""Number of prune-regraph updates per SMC iteration.""",
)
@click.option(
    "-s",
    "--subtree-update-prob",
    default=0.0,
    type=click.FloatRange(0.0, 1.0, clamp=True),
    show_default=True,
    help="""Probability of updating a subtree (instead of whole tree) using PG sampler.""",
)
@click.option(
    "--precision",
    default=400,
    type=float,
    show_default=True,
    callback=_validate_positive_value,
    help="""The (initial) precision parameter of the Beta-Binomial density. 
    The higher the value the more similar the Beta-Binomial is to a Binomial.""",
)
@click.option(
    "--print-freq",
    default=100,
    type=click.IntRange(1, clamp=True),
    show_default=True,
    help="""How frequently to print information about fitting.""",
)
@click.option(
    "--resample-threshold",
    default=0.5,
    type=click.FloatRange(0.0, 1.0, clamp=True),
    show_default=True,
    help="""ESS threshold to trigger resampling.""",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="""Set random seed so results can be reproduced. By default, a random seed is chosen.""",
)
@click.option(
    "--assign-loss-prob/--no-assign-loss-prob",
    default=False,
    show_default=True,
    callback=_validate_assign_loss_prob,
    is_eager=True,
    help="""Whether to assign loss probability prior from the cluster data.
    Note: This option is incompatible with --user-provided-loss-prob""",
)
@click.option(
    "--user-provided-loss-prob/--no-user-provided-loss-prob",
    default=False,
    show_default=True,
    callback=_validate_user_provided_loss_prob,
    is_eager=True,
    help="""Whether to use user-provided cluster loss probability prior from the cluster file.
    Requires that the 'outlier_prob' column be present and populated in the cluster file.
    Note: This option is incompatible with --assign-loss-prob""",
)
@click.option(
    "--high-loss-prob",
    default=0.4,
    type=click.FloatRange(0.0002, 1.0, clamp=True),
    show_default=True,
    callback=_validate_high_loss_prob,
    help="""Higher loss probability setting. 
    Used when allowing PhyClone to assign loss prior probability from cluster data.""",
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
