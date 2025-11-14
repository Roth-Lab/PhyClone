import os
import tempfile
import unittest
from unittest.mock import MagicMock
import sys

import numpy as np
import pandas as pd

import phyclone
from phyclone.data.pyclone import (
    _setup_cluster_df,
    compute_outlier_prob,
    _create_clustered_data_arr,
    _create_loaded_pyclone_data_dict,
    load_data,
)

EPS = sys.float_info.min
LOG_EPS = np.log(EPS)


def build_standard_cluster_df():
    df_dict = {
        "mutation_id": ["m1", "m1", "m1", "m2", "m2", "m2", "m3", "m3", "m3"],
        "sample_id": ["s1", "s2", "s3", "s1", "s2", "s3", "s1", "s2", "s3"],
        "cluster_id": [0, 0, 0, 1, 1, 1, 0, 0, 0],
        "cellular_prevalence": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        "chrom": ["chr1", "chr1", "chr1", "chr2", "chr2", "chr2", "chr3", "chr3", "chr3"],
    }
    return pd.DataFrame(df_dict)


def build_standard_data_df():
    df_dict = {
        "mutation_id": ["m1", "m1", "m1", "m2", "m2", "m2", "m3", "m3", "m3"],
        "sample_id": ["s1", "s2", "s3", "s1", "s2", "s3", "s1", "s2", "s3"],
        "ref_counts": [20, 4, 104, 5, 8, 96, 7, 6, 3],
        "alt_counts": [8, 16, 45, 78, 56, 15, 65, 56, 12],
        "major_cn": [2, 2, 4, 3, 4, 4, 6, 4, 2],
        "minor_cn": [1, 2, 3, 2, 1, 3, 2, 1, 1],
        "normal_cn": [2, 2, 2, 2, 2, 2, 2, 2, 2],
        "tumour_content": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "error_rate": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    }
    return pd.DataFrame(df_dict)


def build_lossy_cluster_df():
    df1 = build_standard_cluster_df()
    df_dict = {
        "mutation_id": ["m4", "m4", "m4", "m5", "m5", "m5", "m6", "m6", "m6"],
        "sample_id": ["s1", "s2", "s3", "s1", "s2", "s3", "s1", "s2", "s3"],
        "cluster_id": [0, 0, 0, 1, 1, 1, 0, 0, 0],
        "cellular_prevalence": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        "chrom": ["chr1", "chr1", "chr1", "chr2", "chr2", "chr2", "chr3", "chr3", "chr3"],
    }
    return pd.concat([df1, pd.DataFrame(df_dict)])


def build_lossy_data_df():
    df1 = build_standard_data_df()
    df_dict = {
        "mutation_id": ["m4", "m4", "m4", "m5", "m5", "m5", "m6", "m6", "m6"],
        "sample_id": ["s1", "s2", "s3", "s1", "s2", "s3", "s1", "s2", "s3"],
        "ref_counts": [20, 4, 104, 5, 8, 96, 7, 6, 3],
        "alt_counts": [8, 16, 45, 78, 56, 15, 65, 56, 12],
        "major_cn": [2, 2, 4, 3, 4, 4, 6, 4, 2],
        "minor_cn": [1, 2, 3, 2, 1, 3, 2, 1, 1],
        "normal_cn": [2, 2, 2, 2, 2, 2, 2, 2, 2],
        "tumour_content": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "error_rate": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    }
    return pd.concat([df1, pd.DataFrame(df_dict)])


def emulate_assign_out_prob_output(df, rng, low_loss_prob, high_loss_prob, min_clust_size=4):
    clusters = df["cluster_id"].unique()
    prob_arr = np.array([low_loss_prob, high_loss_prob])
    low_prob_clusts = rng.choice(prob_arr, size=len(clusters))

    df["outlier_prob"] = 0.0

    for i, loss_prob in enumerate(low_prob_clusts):
        curr_clust = clusters[i]
        df.loc[df["cluster_id"] == curr_clust, "outlier_prob"] = loss_prob


class BaseLoadDataTest(object):
    class TestLoadDataIntegration(unittest.TestCase):
        def setUp(self):
            self.rng = np.random.default_rng(self.seed)

        def __init__(self, method_name: str = ...):
            super().__init__(method_name)
            self.seed = 244310326493402386435613023297139050129
            self.density = None
            self.precision = 400
            self.grid_size = 101

        def run_clustered_test(self, actual_samples, cluster_df, data, expected_samples):
            self.assertListEqual(expected_samples, actual_samples)
            cluster_df = cluster_df[["mutation_id", "cluster_id", "outlier_prob"]].drop_duplicates()
            cluster_sizes = cluster_df["cluster_id"].value_counts().to_dict()
            cluster_outlier_probs = cluster_df.set_index("cluster_id")["outlier_prob"].to_dict()
            for dp in data:
                cluster_id = int(dp.name)
                cluster_size = cluster_sizes[cluster_id]
                cluster_outlier_prob = cluster_outlier_probs[cluster_id]
                if cluster_outlier_prob == 0:
                    outlier_prob = np.log(EPS) * cluster_size
                    outlier_prob_not = 0
                else:
                    outlier_prob = np.log(cluster_outlier_prob) * cluster_size
                    if cluster_outlier_prob == 1:
                        outlier_prob_not = np.log(EPS) * cluster_size
                    else:
                        outlier_prob_not = np.log1p(-cluster_outlier_prob) * cluster_size
                self.assertEqual(outlier_prob, dp.outlier_prob)
                self.assertEqual(outlier_prob_not, dp.outlier_prob_not)

        def run_clustered_load_data(
            self,
            assign_loss_prob,
            cluster_df,
            data_df,
            high_loss_prob,
            outlier_prob,
            min_clust_size=4,
        ):
            with tempfile.TemporaryDirectory() as tmp_dir:
                data_file_path = os.path.join(tmp_dir, "data.tsv")
                data_df.to_csv(data_file_path, sep="\t", index=False)

                cluster_file_path = os.path.join(tmp_dir, "clusters.tsv")
                cluster_df.to_csv(cluster_file_path, sep="\t", index=False)

                data, actual_samples, _ = load_data(
                    data_file_path,
                    self.rng,
                    high_loss_prob,
                    assign_loss_prob,
                    cluster_file_path,
                    self.density,
                    self.grid_size,
                    outlier_prob,
                    self.precision,
                    min_clust_size,
                )
            return actual_samples, data

        def run_unclustered_load_data(self, assign_loss_prob, data_df, high_loss_prob, outlier_prob):
            with tempfile.TemporaryDirectory() as tmp_dir:
                data_file_path = os.path.join(tmp_dir, "data.tsv")
                data_df.to_csv(data_file_path, sep="\t", index=False)

                data, actual_samples, _ = load_data(
                    data_file_path,
                    self.rng,
                    high_loss_prob,
                    assign_loss_prob,
                    None,
                    self.density,
                    self.grid_size,
                    outlier_prob,
                    self.precision,
                )
            return actual_samples, data

        def run_unclustered_test(self, actual_samples, data, expected_samples, prob):
            self.assertListEqual(expected_samples, actual_samples)
            for dp in data:
                if prob == 0:
                    outlier_prob = np.log(EPS)
                    outlier_prob_not = 0
                else:
                    outlier_prob = np.log(prob)
                    if prob == 1:
                        outlier_prob_not = np.log(EPS)
                    else:
                        outlier_prob_not = np.log1p(-prob)
                self.assertEqual(outlier_prob, dp.outlier_prob)
                self.assertEqual(outlier_prob_not, dp.outlier_prob_not)

        def test_clustered__no_outliers(self):
            cluster_df = build_standard_cluster_df()
            data_df = build_standard_data_df()
            high_loss_prob = 0.4
            outlier_prob = 0.001
            assign_loss_prob = False

            actual_samples, data = self.run_clustered_load_data(
                assign_loss_prob,
                cluster_df,
                data_df,
                high_loss_prob,
                outlier_prob,
            )

            expected_samples = sorted(data_df["sample_id"].unique())
            cluster_df["outlier_prob"] = outlier_prob

            self.run_clustered_test(actual_samples, cluster_df, data, expected_samples)

        def test_clustered_outlier__prob_col_supplied(self):
            data_df = build_standard_data_df()
            high_loss_prob = 0.4
            outlier_prob = 0.001
            assign_loss_prob = False
            cluster_df = build_standard_cluster_df()
            emulate_assign_out_prob_output(cluster_df, self.rng, outlier_prob, high_loss_prob)

            actual_samples, data = self.run_clustered_load_data(
                assign_loss_prob,
                cluster_df,
                data_df,
                high_loss_prob,
                outlier_prob,
            )

            expected_samples = sorted(data_df["sample_id"].unique())

            self.run_clustered_test(actual_samples, cluster_df, data, expected_samples)

        def test_clustered__global_outlier_prior(self):
            cluster_df = build_standard_cluster_df()
            data_df = build_standard_data_df()
            high_loss_prob = 0.4
            outlier_prob = 0.05
            assign_loss_prob = False

            actual_samples, data = self.run_clustered_load_data(
                assign_loss_prob,
                cluster_df,
                data_df,
                high_loss_prob,
                outlier_prob,
            )

            expected_samples = sorted(data_df["sample_id"].unique())
            cluster_df["outlier_prob"] = outlier_prob

            self.run_clustered_test(actual_samples, cluster_df, data, expected_samples)

        def test_clustered__assign_from_data_no_loss(self):
            cluster_df = build_standard_cluster_df()
            data_df = build_standard_data_df()
            high_loss_prob = 0.4
            outlier_prob = 0.001
            assign_loss_prob = True

            actual_samples, data = self.run_clustered_load_data(
                assign_loss_prob,
                cluster_df,
                data_df,
                high_loss_prob,
                outlier_prob,
            )

            expected_samples = sorted(data_df["sample_id"].unique())
            cluster_df["outlier_prob"] = outlier_prob

            self.run_clustered_test(actual_samples, cluster_df, data, expected_samples)

        def test_clustered__assign_from_data_with_loss(self):
            cluster_df = build_lossy_cluster_df()
            data_df = build_lossy_data_df()
            high_loss_prob = 0.4
            outlier_prob = 0.001
            assign_loss_prob = True

            actual_samples, data = self.run_clustered_load_data(
                assign_loss_prob,
                cluster_df,
                data_df,
                high_loss_prob,
                outlier_prob,
                min_clust_size=1,
            )

            expected_samples = sorted(data_df["sample_id"].unique())
            cluster_df["outlier_prob"] = outlier_prob
            cluster_df.loc[cluster_df["cluster_id"] == 1, "outlier_prob"] = high_loss_prob

            self.run_clustered_test(actual_samples, cluster_df, data, expected_samples)

        def test_unclustered__global_outlier_prior(self):
            data_df = build_standard_data_df()
            high_loss_prob = 0.4
            outlier_prob = 0.001
            assign_loss_prob = False

            actual_samples, data = self.run_unclustered_load_data(
                assign_loss_prob,
                data_df,
                high_loss_prob,
                outlier_prob,
            )

            expected_samples = sorted(data_df["sample_id"].unique())
            self.run_unclustered_test(actual_samples, data, expected_samples, outlier_prob)

        def test_unclustered__no_outliers(self):
            data_df = build_standard_data_df()
            high_loss_prob = 0.4
            outlier_prob = 0.0
            assign_loss_prob = False

            actual_samples, data = self.run_unclustered_load_data(
                assign_loss_prob,
                data_df,
                high_loss_prob,
                outlier_prob,
            )

            expected_samples = sorted(data_df["sample_id"].unique())
            self.run_unclustered_test(actual_samples, data, expected_samples, outlier_prob)


class BaseClusteredDataArrTest(object):
    class TestCreateClusteredDataArr(unittest.TestCase):
        def setUp(self):
            self.rng = np.random.default_rng(self.seed)

        def __init__(self, method_name: str = ...):
            super().__init__(method_name)
            self.seed = 244310326493402386435613023297139050129
            self.density = None
            self.precision = 400

        def build_cluster_df(self, low_loss_prob, high_loss_prob):
            cluster_df = build_standard_cluster_df()
            emulate_assign_out_prob_output(cluster_df, self.rng, low_loss_prob, high_loss_prob)
            return cluster_df

        def run_test(self, grid_size, high_loss_prob, low_loss_prob):
            cluster_df = self.build_cluster_df(low_loss_prob, high_loss_prob)
            cluster_sizes = cluster_df["cluster_id"].value_counts().to_dict()
            clusters = cluster_df.set_index("mutation_id")["cluster_id"].to_dict()
            cluster_outlier_probs = cluster_df.set_index("cluster_id")["outlier_prob"].to_dict()
            data_df = build_standard_data_df()
            samples = sorted(data_df["sample_id"].unique())
            pyclone_data = _create_loaded_pyclone_data_dict(data_df, samples)
            data = _create_clustered_data_arr(
                cluster_outlier_probs,
                cluster_sizes,
                clusters,
                self.density,
                grid_size,
                self.precision,
                pyclone_data,
            )
            self.assertEqual(len(data), len(cluster_sizes))
            for dp in data:
                cluster_id = dp.idx
                cluster_size = cluster_sizes[cluster_id]
                cluster_outlier_prob = cluster_outlier_probs[cluster_id]
                if cluster_outlier_prob == 0:
                    outlier_prob = np.log(EPS) * cluster_size
                    outlier_prob_not = 0
                else:
                    outlier_prob = np.log(cluster_outlier_prob) * cluster_size
                    if cluster_outlier_prob == 1:
                        outlier_prob_not = np.log(EPS) * cluster_size
                    else:
                        outlier_prob_not = np.log1p(-cluster_outlier_prob) * cluster_size
                self.assertEqual(dp.outlier_prob, outlier_prob)
                self.assertEqual(dp.outlier_prob_not, outlier_prob_not)

        def test_create_clustered_data_arr__low_0_high_1(self):
            low_loss_prob = 0.0
            high_loss_prob = 1.0
            grid_size = 101
            self.run_test(grid_size, high_loss_prob, low_loss_prob)

        def test_create_clustered_data_arr__low_45_high_5(self):
            low_loss_prob = 0.45
            high_loss_prob = 0.5
            grid_size = 101
            self.run_test(grid_size, high_loss_prob, low_loss_prob)

        def test_create_clustered_data_arr__low_05_high_9(self):
            low_loss_prob = 0.05
            high_loss_prob = 0.9
            grid_size = 101
            self.run_test(grid_size, high_loss_prob, low_loss_prob)


class TestComputeOutlierProb(unittest.TestCase):

    def test_compute_outlier_prob__prior_0_1_mutation(self):
        prob = 0
        cluster_size = 1
        expected = (np.log(EPS) * cluster_size, np.log(1.0))
        actual = compute_outlier_prob(prob, cluster_size)
        self.assertTupleEqual(expected, actual)

    def test_compute_outlier_prob__prior_0_100_mutations(self):
        prob = 0
        cluster_size = 100
        expected = (np.log(EPS) * cluster_size, np.log(1.0))
        actual = compute_outlier_prob(prob, cluster_size)
        self.assertTupleEqual(expected, actual)

    def test_compute_outlier_prob__prior_5_1_mutation(self):
        prob = 0.5
        cluster_size = 1
        expected = self.compute_expected(cluster_size, prob)
        actual = compute_outlier_prob(prob, cluster_size)
        self.assertTupleEqual(expected, actual)

    def test_compute_outlier_prob__prior_5_100_mutations(self):
        prob = 0.5
        cluster_size = 100
        expected = self.compute_expected(cluster_size, prob)
        actual = compute_outlier_prob(prob, cluster_size)
        self.assertTupleEqual(expected, actual)

    def test_compute_outlier_prob__prior_1_1_mutation(self):
        prob = 1.0
        cluster_size = 1
        expected = self.compute_expected(cluster_size, prob)
        actual = compute_outlier_prob(prob, cluster_size)
        self.assertTupleEqual(expected, actual)

    def test_compute_outlier_prob__prior_1_100_mutations(self):
        prob = 1.0
        cluster_size = 100
        expected = self.compute_expected(cluster_size, prob)
        actual = compute_outlier_prob(prob, cluster_size)
        self.assertTupleEqual(expected, actual)

    @staticmethod
    def compute_expected(cluster_size, prob):
        if prob == 1:
            res_not = np.log(EPS) * cluster_size
        else:
            res_not = np.log1p(-prob) * cluster_size
        expected = (np.log(prob) * cluster_size, res_not)
        return expected


class TestSetupClusterDF(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(self.seed)

    def __init__(self, method_name: str = ...):
        super().__init__(method_name)
        self.seed = 244310326493402386435613023297139050129
        self.outlier_prob = 1e-4
        self.low_loss_prob = 1e-4
        self.high_loss_prob = 0.4
        self.min_clust_size = 4

    def test_no_outliers_no_optional_cols(self):
        df = build_standard_cluster_df()
        df = df.drop(columns=["cellular_prevalence", "chrom"])
        data_df = build_standard_data_df()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "data.tsv")
            df.to_csv(file_path, sep="\t")
            actual_df, _ = _setup_cluster_df(
                file_path,
                0,
                self.rng,
                self.high_loss_prob,
                False,
                self.min_clust_size,
                data_df,
            )
        self.assertTrue(np.all(actual_df["outlier_prob"] == 0))

    def test_no_outliers_all_optional_cols(self):
        df = build_standard_cluster_df()
        data_df = build_standard_data_df()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "data.tsv")
            df.to_csv(file_path, sep="\t")
            actual_df, _ = _setup_cluster_df(
                file_path,
                0,
                self.rng,
                self.high_loss_prob,
                False,
                self.min_clust_size,
                data_df,
            )
        self.assertTrue(np.all(actual_df["outlier_prob"] == 0))

    def test_global_outlier_val(self):
        df = build_standard_cluster_df()
        data_df = build_standard_data_df()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "data.tsv")
            df.to_csv(file_path, sep="\t")
            actual_df, _ = _setup_cluster_df(
                file_path,
                self.outlier_prob,
                self.rng,
                self.high_loss_prob,
                False,
                self.min_clust_size,
                data_df,
            )
        self.assertTrue(np.all(actual_df["outlier_prob"] == self.outlier_prob))

    def test_assign_loss_prob__valid_cols(self):
        df = build_standard_cluster_df()
        data_df = build_standard_data_df()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "data.tsv")
            df.to_csv(file_path, sep="\t")
            phyclone.data.pyclone._assign_out_prob = MagicMock(side_effect=emulate_assign_out_prob_output)
            actual_df, _ = _setup_cluster_df(
                file_path,
                self.outlier_prob,
                self.rng,
                self.high_loss_prob,
                True,
                self.min_clust_size,
                data_df,
            )
        self.assertTrue(np.all(actual_df["outlier_prob"] != 0))

    def test_assign_loss_prob__chrom_missing(self):
        df = build_standard_cluster_df()
        df = df.drop(columns=["chrom"])
        data_df = build_standard_data_df()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "data.tsv")
            df.to_csv(file_path, sep="\t")
            phyclone.data.pyclone._assign_out_prob = MagicMock(side_effect=emulate_assign_out_prob_output)
            actual_df, _ = _setup_cluster_df(
                file_path,
                self.outlier_prob,
                self.rng,
                self.high_loss_prob,
                True,
                self.min_clust_size,
                data_df,
            )
        self.assertTrue(np.all(actual_df["outlier_prob"] == self.low_loss_prob))

    def test_assign_loss_prob__chrom_missing_present_on_data_df(self):
        df = build_standard_cluster_df()
        df = df.drop(columns=["chrom"])
        data_df = build_standard_data_df()
        data_df["chrom"] = ["chr1", "chr1", "chr1", "chr2", "chr2", "chr2", "chr3", "chr3", "chr3"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "data.tsv")
            df.to_csv(file_path, sep="\t")
            phyclone.data.pyclone._assign_out_prob = MagicMock(side_effect=emulate_assign_out_prob_output)
            actual_df, _ = _setup_cluster_df(
                file_path,
                self.outlier_prob,
                self.rng,
                self.high_loss_prob,
                True,
                self.min_clust_size,
                data_df,
            )
        self.assertTrue(np.all(actual_df["outlier_prob"] != 0))

    def test_assign_loss_prob__cellular_prevalence_missing(self):
        df = build_standard_cluster_df()
        df = df.drop(columns=["cellular_prevalence"])
        data_df = build_standard_data_df()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "data.tsv")
            df.to_csv(file_path, sep="\t")
            phyclone.data.pyclone._assign_out_prob = MagicMock(side_effect=emulate_assign_out_prob_output)
            actual_df, _ = _setup_cluster_df(
                file_path,
                self.outlier_prob,
                self.rng,
                self.high_loss_prob,
                True,
                self.min_clust_size,
                data_df,
            )
        self.assertTrue(np.all(actual_df["outlier_prob"] == self.low_loss_prob))

    def test_assign_loss_prob__both_chrom_and_ccf_missing(self):
        df = build_standard_cluster_df()
        data_df = build_standard_data_df()
        df = df.drop(columns=["cellular_prevalence", "chrom"])
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "data.tsv")
            df.to_csv(file_path, sep="\t")
            phyclone.data.pyclone._assign_out_prob = MagicMock(side_effect=emulate_assign_out_prob_output)
            actual_df, _ = _setup_cluster_df(
                file_path,
                self.outlier_prob,
                self.rng,
                self.high_loss_prob,
                True,
                self.min_clust_size,
                data_df,
            )
        self.assertTrue(np.all(actual_df["outlier_prob"] == self.low_loss_prob))


class BinomialDistTest(BaseClusteredDataArrTest.TestCreateClusteredDataArr):

    def setUp(self):
        super().setUp()
        self.density = "binomial"


class BetaBinomialDistTest(BaseClusteredDataArrTest.TestCreateClusteredDataArr):

    def setUp(self):
        super().setUp()
        self.density = "beta-binomial"


class BinomialDistLoadDataIntegrationTest(BaseLoadDataTest.TestLoadDataIntegration):

    def setUp(self):
        super().setUp()
        self.density = "binomial"


class BetaBinomialDistLoadDataIntegrationTest(BaseLoadDataTest.TestLoadDataIntegration):

    def setUp(self):
        super().setUp()
        self.density = "beta-binomial"


if __name__ == "__main__":
    unittest.main()
