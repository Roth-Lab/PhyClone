import itertools
from collections import OrderedDict, defaultdict
import numba
import numpy as np
import pandas as pd
import phyclone.data.base
from phyclone.data.cluster_outlier_probabilities import _assign_out_prob
from phyclone.utils.exceptions import MajorCopyNumberError
from phyclone.utils.math import log_pyclone_beta_binomial_pdf, log_pyclone_binomial_pdf
from phyclone.data.validator import create_cluster_input_validator_instance, create_data_input_validator_instance


def load_data(
    file_name,
    rng,
    low_loss_prob,
    high_loss_prob,
    assign_loss_prob,
    cluster_file=None,
    density="beta-binomial",
    grid_size=101,
    outlier_prob=1e-4,
    precision=400,
    min_clust_size=4,
):

    pyclone_data, samples, data_df = load_pyclone_data(file_name)

    init_sigma = None

    if cluster_file is None:
        data = []

        for idx, (mut, val) in enumerate(pyclone_data.items()):
            out_probs = compute_outlier_prob(outlier_prob, 1)
            data_point = phyclone.data.base.DataPoint(
                idx,
                val.to_likelihood_grid(density, grid_size, precision=precision),
                name=mut,
                outlier_prob=out_probs[0],
                outlier_prob_not=out_probs[1],
            )

            data.append(data_point)

    else:
        cluster_df = _setup_cluster_df(
            cluster_file,
            outlier_prob,
            rng,
            low_loss_prob,
            high_loss_prob,
            assign_loss_prob,
            min_clust_size,
            data_df,
        )

        cluster_sizes = cluster_df["cluster_id"].value_counts().to_dict()

        clusters = cluster_df.set_index("mutation_id")["cluster_id"].to_dict()

        cluster_outlier_probs = cluster_df.set_index("cluster_id")["outlier_prob"].to_dict()

        print("\nUsing input clustering with {} clusters\n".format(cluster_df["cluster_id"].nunique()))

        data = _create_clustered_data_arr(
            cluster_outlier_probs,
            cluster_sizes,
            clusters,
            density,
            grid_size,
            precision,
            pyclone_data,
        )

        if "order_rank" in cluster_df.columns:
            datapoint_name_dict = {d.name: d for d in data}
            cluster_order = cluster_df.set_index("order_rank")["cluster_id"].to_dict()
            init_sigma = [datapoint_name_dict[str(cluster_order[i])] for i in range(len(datapoint_name_dict))]

    return data, samples, init_sigma


def _create_clustered_data_arr(
    cluster_outlier_probs,
    cluster_sizes,
    clusters,
    density,
    grid_size,
    precision,
    pyclone_data,
):
    raw_data = defaultdict(list)
    for mut, val in pyclone_data.items():
        raw_data[clusters[mut]].append(val.to_likelihood_grid(density, grid_size, precision=precision))
    data = []
    for idx, cluster_id in enumerate(sorted(raw_data.keys())):
        val = np.sum(np.array(raw_data[cluster_id]), axis=0)
        cluster_outlier_prob = cluster_outlier_probs[cluster_id]
        out_probs = compute_outlier_prob(cluster_outlier_prob, cluster_sizes[cluster_id])

        data_point = phyclone.data.base.DataPoint(
            idx,
            val,
            name="{}".format(cluster_id),
            outlier_prob=out_probs[0],
            outlier_prob_not=out_probs[1],
        )

        data.append(data_point)
    return data


def _setup_cluster_df(
    cluster_file,
    outlier_prob,
    rng,
    low_loss_prob,
    high_loss_prob,
    assign_loss_prob,
    min_clust_size,
    data_df,
):
    cluster_df = _get_raw_cluster_df(cluster_file, data_df)
    cluster_prob_status_msg = ""
    if "outlier_prob" not in cluster_df.columns:
        cluster_prob_status_msg += "\nCluster level outlier probability column not found. "
        if assign_loss_prob:
            column_checks = "chrom" in cluster_df.columns and "cellular_prevalence" in cluster_df.columns
            if column_checks:
                cluster_prob_status_msg += "Assigning from data.\n"
                print(cluster_prob_status_msg)
                _assign_out_prob(cluster_df, rng, low_loss_prob, high_loss_prob, min_clust_size)
            else:
                cluster_prob_status_msg += "\nMutation chrom position column also not found."
                cluster_prob_status_msg += "\nOutlier probability cannot be assigned from data."
                cluster_prob_status_msg += " Setting values to {p}.\n".format(p=low_loss_prob)
                print(cluster_prob_status_msg)
                cluster_df.loc[:, "outlier_prob"] = low_loss_prob
        else:
            cluster_prob_status_msg += "Setting values to {p}\n".format(p=outlier_prob)
            print(cluster_prob_status_msg)
            cluster_df.loc[:, "outlier_prob"] = outlier_prob
    else:
        cluster_prob_status_msg += "\nCluster level outlier probability column is present.\n"
        cluster_prob_status_msg += "Using user-supplied outlier probability prior values.\n"
        print(cluster_prob_status_msg)
    if not assign_loss_prob:
        if outlier_prob == 0:
            cluster_df.loc[:, "outlier_prob"] = outlier_prob
        else:
            cluster_df.loc[cluster_df["outlier_prob"] == 0, "outlier_prob"] = outlier_prob
    if "order_rank" in cluster_df.columns:
        cluster_df = cluster_df[["mutation_id", "cluster_id", "outlier_prob", "order_rank"]].drop_duplicates()
    else:
        cluster_df = cluster_df[["mutation_id", "cluster_id", "outlier_prob"]].drop_duplicates()
    return cluster_df


def _get_raw_cluster_df(cluster_file, data_df):
    cluster_input_validator = create_cluster_input_validator_instance(cluster_file)
    cluster_input_validator.validate()
    cluster_df = cluster_input_validator.df
    if "chrom" not in cluster_df.columns:
        if "chrom" in data_df.columns:
            data_df_filtered = data_df[["mutation_id", "chrom"]].drop_duplicates()
            cluster_df = pd.merge(cluster_df, data_df_filtered, how="inner", on=["mutation_id"])
            cluster_df = cluster_df.drop_duplicates()
    return cluster_df


def compute_outlier_prob(outlier_prob, cluster_size):
    if outlier_prob == 0:
        return outlier_prob, np.log(1.0)
    else:
        res = np.log(outlier_prob) * cluster_size
        if outlier_prob == 1:
            res_not = -np.inf
        else:
            res_not = np.log1p(-outlier_prob) * cluster_size
        return res, res_not


def load_pyclone_data(file_name):
    df = _create_raw_data_df(file_name)

    df = _remove_cn_zero_mutations(df)

    samples = sorted(df["sample_id"].unique())

    df = _remove_duplicated_and_partially_absent_mutations(df, samples)

    mutations = df["mutation_id"].unique()

    print("Num mutations: {}".format(len(mutations)))

    _process_required_cols_on_df(df, samples)

    data = _create_loaded_pyclone_data_dict(df, samples)

    return data, samples, df


def _remove_duplicated_and_partially_absent_mutations(df, samples):
    samples_len = len(samples)
    group_transform = df.groupby(df["mutation_id"])["sample_id"].transform("size")
    num_not_present_in_all = len(df[group_transform < samples_len]["mutation_id"].unique())
    num_duplicates = len(df[group_transform > samples_len]["mutation_id"].unique())
    if num_duplicates > 0:
        if num_duplicates == 1:
            pl = ""
        else:
            pl = "s"
        print("Removing {} duplicate mutation ID{}".format(num_duplicates, pl))
    if num_not_present_in_all > 0:
        if num_not_present_in_all == 1:
            pl = ("", "is")
        else:
            pl = ("s", "are")
        print(
            "Removing {} mutation{} that {} not present in all samples".format(
                num_not_present_in_all,
                pl[0],
                pl[1],
            )
        )
    df = df.loc[group_transform == samples_len]
    return df


def _remove_cn_zero_mutations(df):
    num_dels = len(df.loc[df["major_cn"] == 0])
    if num_dels > 0:
        if num_dels == 1:
            pl = ""
        else:
            pl = "s"
        print("Removing {} mutation{} with major copy number zero".format(num_dels, pl))
    df = df.loc[df["major_cn"] > 0]
    return df


def _process_required_cols_on_df(df, samples):
    print("Num Samples: {}".format(len(samples)))
    if len(samples) > 10:
        print("Samples: {}...".format(" ".join(samples[:4])))
    else:
        print("Samples: {}".format(" ".join(samples)))
    if "error_rate" not in df.columns:
        df.loc[:, "error_rate"] = 1e-3
    if "tumour_content" not in df.columns:
        print("Tumour content column not found. Setting values to 1.0.")
        df.loc[:, "tumour_content"] = 1.0


def _create_raw_data_df(file_name):
    data_input_validator = create_data_input_validator_instance(file_name)
    data_input_validator.validate()
    df = data_input_validator.df
    df["sample_id"] = df["sample_id"].astype("string")
    return df


def _create_loaded_pyclone_data_dict(df, samples):
    data = OrderedDict()
    df = df.sort_values(by="mutation_id", ascending=True)
    grouped = df.groupby("mutation_id", sort=False)

    for mutation, group in grouped:
        sample_data_points = []

        group.set_index("sample_id", inplace=True)

        for sample in samples:

            a = group.at[sample, "ref_counts"]

            b = group.at[sample, "alt_counts"]

            cn, mu, log_pi = get_major_cn_prior(
                group.at[sample, "major_cn"],
                group.at[sample, "minor_cn"],
                group.at[sample, "normal_cn"],
                error_rate=group.at[sample, "error_rate"],
            )

            sample_data_points.append(SampleDataPoint(a, b, cn, mu, log_pi, group.at[sample, "tumour_content"]))

        data[mutation] = DataPoint(samples, sample_data_points)

    return data


def get_major_cn_prior(major_cn, minor_cn, normal_cn, error_rate=1e-3):
    total_cn = major_cn + minor_cn

    if major_cn < minor_cn:
        raise MajorCopyNumberError(major_cn, minor_cn)

    # Consider all possible mutational genotypes consistent with mutation before CN change
    cn = [(normal_cn, normal_cn, total_cn) for _ in range(1, major_cn + 1)]
    mu = [(error_rate, error_rate, min(1 - error_rate, x / total_cn)) for x in range(1, major_cn + 1)]

    # Consider mutational genotype of mutation before CN change if not already added
    if total_cn != normal_cn:
        mutation_after_cn = (normal_cn, total_cn, total_cn)
        cn.append(mutation_after_cn)
        mu.append((error_rate, error_rate, min(1 - error_rate, 1 / total_cn)))
        assert len(set(cn)) == 2

    cn = np.array(cn, dtype=int)
    mu = np.array(mu, dtype=float)

    log_pi_val = -np.log(len(cn))
    log_pi = np.full(len(cn), log_pi_val)

    return cn, mu, log_pi


class DataPoint(object):

    def __init__(self, samples, sample_data_points):
        self.samples = samples

        self.sample_data_points = sample_data_points

    def get_ccf_grid(self, grid_size):
        return np.linspace(0, 1, grid_size)

    def to_dict(self):
        return OrderedDict(zip(self.samples, self.sample_data_points))

    def to_likelihood_grid(self, density, grid_size, precision=None):
        if (density == "beta-binomial") and (precision is None):
            raise Exception("Precision must be set when using Beta-Binomial.")

        shape = (len(self.samples), grid_size)

        log_ll = np.zeros(shape)

        sample_data_points = self.sample_data_points
        ccf_grid = self.get_ccf_grid(grid_size)

        _compute_liklihood_grid(ccf_grid, density, log_ll, precision, numba.typed.List(sample_data_points))

        return log_ll


@numba.jit(nopython=True)
def _compute_liklihood_grid(ccf_grid, density, log_ll, precision, sample_data_points):
    for s_idx, data_point in enumerate(sample_data_points):
        for i, ccf in enumerate(ccf_grid):
            if density == "beta-binomial":
                log_ll[s_idx, i] = log_pyclone_beta_binomial_pdf(data_point, ccf, precision)

            elif density == "binomial":
                log_ll[s_idx, i] = log_pyclone_binomial_pdf(data_point, ccf)


@numba.experimental.jitclass(
    [
        ("a", numba.int64),
        ("b", numba.int64),
        ("cn", numba.int64[:, :]),
        ("mu", numba.float64[:, :]),
        ("log_pi", numba.float64[:]),
        ("t", numba.float64),
    ]
)
class SampleDataPoint(object):

    def __init__(self, a, b, cn, mu, log_pi, t):
        self.a = a
        self.b = b
        self.cn = cn
        self.mu = mu
        self.log_pi = log_pi
        self.t = t
