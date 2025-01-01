import tempfile
import unittest
import numpy as np
from phyclone.utils.exceptions import MajorCopyNumberError
from phyclone.utils.math import log_normalize
from phyclone.data.pyclone import (
    _create_raw_data_df,
    _remove_cn_zero_mutations,
    _remove_duplicated_and_partially_absent_mutations,
    _process_required_cols_on_df,
    _create_loaded_pyclone_data_dict,
    get_major_cn_prior,
    load_pyclone_data,
)
import pandas as pd
import os


def tester_get_major_cn_prior(major_cn, minor_cn, normal_cn, error_rate=1e-3):
    total_cn = major_cn + minor_cn

    cn = []

    mu = []

    log_pi = []

    if major_cn < minor_cn:
        raise MajorCopyNumberError(major_cn, minor_cn)

    # Consider all possible mutational genotypes consistent with mutation before CN change
    for x in range(1, major_cn + 1):
        cn.append((normal_cn, normal_cn, total_cn))

        mu.append((error_rate, error_rate, min(1 - error_rate, x / total_cn)))

        log_pi.append(0)

    # Consider mutational genotype of mutation before CN change if not already added
    mutation_after_cn = (normal_cn, total_cn, total_cn)

    if mutation_after_cn not in cn:
        cn.append(mutation_after_cn)

        mu.append((error_rate, error_rate, min(1 - error_rate, 1 / total_cn)))

        log_pi.append(0)

        assert len(set(cn)) == 2

    cn = np.array(cn, dtype=int)

    mu = np.array(mu, dtype=float)

    log_pi = log_normalize(np.array(log_pi, dtype=float))

    return cn, mu, log_pi


class TestLoadPyCloneData(unittest.TestCase):

    def _assert_datapoint_dict_against_df_dict(self, actual_df, data, df_dict, skip_dict):
        self.assertEqual(len(actual_df["mutation_id"].unique()), len(data))
        self.assertListEqual(sorted(actual_df["mutation_id"].unique()), list(data.keys()))
        for mut, dp in data.items():
            curr_list = dp.sample_data_points
            skips = skip_dict[mut]
            for i, sdp in enumerate(curr_list):
                idx = i + skips
                self.assertEqual(sdp.a, df_dict["ref_counts"][idx])
                self.assertEqual(sdp.b, df_dict["alt_counts"][idx])
                cn, mu, log_pi = tester_get_major_cn_prior(df_dict["major_cn"][idx], df_dict["minor_cn"][idx], 2)
                np.testing.assert_array_equal(sdp.cn, cn)
                np.testing.assert_array_equal(sdp.mu, mu)
                np.testing.assert_array_equal(sdp.log_pi, log_pi)

    def test_create_raw_data_df__valid_tab_sep(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            df_dict = {
                "mutation_id": ["m1", "m2", "m3"],
                "sample_id": ["s1", "s2", "s3"],
                "ref_counts": [20, 4, 104],
                "alt_counts": [8, 16, 45],
                "major_cn": [2, 2, 4],
                "minor_cn": [1, 2, 3],
                "normal_cn": [2, 2, 2],
                "tumour_content": [1.0, 0.2, 0.3],
                "error_rate": [0.001, 0.002, 0.001],
            }
            df = pd.DataFrame(df_dict)
            file_path = os.path.join(tmp_dir, "data.tsv")
            df.to_csv(file_path, sep="\t", index=False)
            df = df.astype({"mutation_id": "string", "sample_id": "string"})
            actual_df = _create_raw_data_df(file_path)
        pd.testing.assert_frame_equal(df, actual_df)

    def test_create_raw_data_df__valid_comma_sep(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            df_dict = {
                "mutation_id": ["m1", "m2", "m3"],
                "sample_id": ["s1", "s2", "s3"],
                "ref_counts": [20, 4, 104],
                "alt_counts": [8, 16, 45],
                "major_cn": [2, 2, 4],
                "minor_cn": [1, 2, 3],
                "normal_cn": [2, 2, 2],
                "tumour_content": [1.0, 0.2, 0.3],
                "error_rate": [0.001, 0.002, 0.001],
            }
            df = pd.DataFrame(df_dict)
            file_path = os.path.join(tmp_dir, "data.tsv")
            df.to_csv(file_path, index=False)
            df = df.astype({"mutation_id": "string", "sample_id": "string"})
            actual_df = _create_raw_data_df(file_path)
        pd.testing.assert_frame_equal(df, actual_df)

    def test_remove_cn_zero_mutations__none_removed(self):
        df_dict = {
            "mutation_id": ["m1", "m2", "m3"],
            "sample_id": ["s1", "s2", "s3"],
            "ref_counts": [20, 4, 104],
            "alt_counts": [8, 16, 45],
            "major_cn": [2, 2, 4],
            "minor_cn": [1, 2, 3],
            "normal_cn": [2, 2, 2],
            "tumour_content": [1.0, 0.2, 0.3],
            "error_rate": [0.001, 0.002, 0.001],
        }
        df = pd.DataFrame(df_dict)
        actual_df = df.copy()
        actual_df = _remove_cn_zero_mutations(actual_df)
        pd.testing.assert_frame_equal(df, actual_df)

    def test_remove_cn_zero_mutations__one_removed(self):
        df_dict = {
            "mutation_id": ["m1", "m2", "m3"],
            "sample_id": ["s1", "s2", "s3"],
            "ref_counts": [20, 4, 104],
            "alt_counts": [8, 16, 45],
            "major_cn": [2, 0, 4],
            "minor_cn": [1, 2, 3],
            "normal_cn": [2, 2, 2],
            "tumour_content": [1.0, 0.2, 0.3],
            "error_rate": [0.001, 0.002, 0.001],
        }
        df = pd.DataFrame(df_dict)
        actual_df = df.copy()
        actual_df = _remove_cn_zero_mutations(actual_df)
        self.assertGreater(len(df), len(actual_df))
        self.assertEqual(len(actual_df), 2)

    def test_remove_cn_zero_mutations__two_removed(self):
        df_dict = {
            "mutation_id": ["m1", "m2", "m3"],
            "sample_id": ["s1", "s2", "s3"],
            "ref_counts": [20, 4, 104],
            "alt_counts": [8, 16, 45],
            "major_cn": [2, 0, 0],
            "minor_cn": [1, 2, 3],
            "normal_cn": [2, 2, 2],
            "tumour_content": [1.0, 0.2, 0.3],
            "error_rate": [0.001, 0.002, 0.001],
        }
        df = pd.DataFrame(df_dict)
        actual_df = df.copy()
        actual_df = _remove_cn_zero_mutations(actual_df)
        self.assertGreater(len(df), len(actual_df))
        self.assertEqual(len(actual_df), 1)

    def test_remove_duplicated_and_partially_absent_mutations__none_removed(self):
        df_dict = {
            "mutation_id": ["m1", "m1", "m1", "m2", "m2", "m2", "m3", "m3", "m3"],
            "sample_id": ["s1", "s2", "s3", "s1", "s2", "s3", "s1", "s2", "s3"],
            "ref_counts": [20, 4, 104, 20, 4, 104, 20, 4, 104],
            "alt_counts": [8, 16, 45, 8, 16, 45, 8, 16, 45],
            "major_cn": [2, 2, 4, 2, 2, 4, 2, 2, 4],
            "minor_cn": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "normal_cn": [2, 2, 2, 2, 2, 2, 2, 2, 2],
            "tumour_content": [1.0, 0.2, 0.3, 1.0, 0.2, 0.3, 1.0, 0.2, 0.3],
            "error_rate": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        }
        df = pd.DataFrame(df_dict)
        actual_df = df.copy()
        samples = sorted(df["sample_id"].unique())
        actual_df = _remove_duplicated_and_partially_absent_mutations(actual_df, samples)
        pd.testing.assert_frame_equal(df, actual_df)

    def test_remove_duplicated_and_partially_absent_mutations__one_removed_absent(self):
        df_dict = {
            "mutation_id": ["m1", "m1", "m1", "m2", "m2", "m2", "m3", "m3", "m3"],
            "sample_id": ["s1", "s2", "s3", "s1", "s2", "s3", "s1", "s2", "s3"],
            "ref_counts": [20, 4, 104, 20, 4, 104, 20, 4, 104],
            "alt_counts": [8, 16, 45, 8, 16, 45, 8, 16, 45],
            "major_cn": [2, 2, 4, 2, 2, 4, 2, 2, 4],
            "minor_cn": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "normal_cn": [2, 2, 2, 2, 2, 2, 2, 2, 2],
            "tumour_content": [1.0, 0.2, 0.3, 1.0, 0.2, 0.3, 1.0, 0.2, 0.3],
            "error_rate": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        }
        df = pd.DataFrame(df_dict)
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "mutation_id": "m4",
                            "sample_id": "s1",
                            "ref_counts": 10,
                            "alt_counts": 12,
                            "major_cn": 4,
                            "minor_cn": 2,
                            "normal_cn": 2,
                            "tumour_content": 1.0,
                            "error_rate": 0.001,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        actual_df = df.copy()
        samples = sorted(df["sample_id"].unique())
        actual_df = _remove_duplicated_and_partially_absent_mutations(actual_df, samples)
        self.assertGreater(len(df), len(actual_df))
        self.assertEqual(len(actual_df["mutation_id"].unique()), 3)

    def test_remove_duplicated_and_partially_absent_mutations__two_removed_absent(self):
        df_dict = {
            "mutation_id": ["m1", "m1", "m1", "m2", "m2", "m2", "m3", "m3", "m3"],
            "sample_id": ["s1", "s2", "s3", "s1", "s2", "s3", "s1", "s2", "s3"],
            "ref_counts": [20, 4, 104, 20, 4, 104, 20, 4, 104],
            "alt_counts": [8, 16, 45, 8, 16, 45, 8, 16, 45],
            "major_cn": [2, 2, 4, 2, 2, 4, 2, 2, 4],
            "minor_cn": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "normal_cn": [2, 2, 2, 2, 2, 2, 2, 2, 2],
            "tumour_content": [1.0, 0.2, 0.3, 1.0, 0.2, 0.3, 1.0, 0.2, 0.3],
            "error_rate": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        }
        df = pd.DataFrame(df_dict)
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "mutation_id": "m4",
                            "sample_id": "s1",
                            "ref_counts": 10,
                            "alt_counts": 12,
                            "major_cn": 4,
                            "minor_cn": 2,
                            "normal_cn": 2,
                            "tumour_content": 1.0,
                            "error_rate": 0.001,
                        },
                        {
                            "mutation_id": "m5",
                            "sample_id": "s2",
                            "ref_counts": 10,
                            "alt_counts": 12,
                            "major_cn": 4,
                            "minor_cn": 2,
                            "normal_cn": 2,
                            "tumour_content": 1.0,
                            "error_rate": 0.001,
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )
        actual_df = df.copy()
        samples = sorted(df["sample_id"].unique())
        actual_df = _remove_duplicated_and_partially_absent_mutations(actual_df, samples)
        self.assertGreater(len(df), len(actual_df))
        self.assertEqual(len(actual_df["mutation_id"].unique()), 3)

    def test_remove_duplicated_and_partially_absent_mutations__one_removed_dup_id(self):
        df_dict = {
            "mutation_id": ["m1", "m1", "m1", "m2", "m2", "m2", "m1", "m1", "m1"],
            "sample_id": ["s1", "s2", "s3", "s1", "s2", "s3", "s1", "s2", "s3"],
            "ref_counts": [20, 4, 104, 20, 4, 104, 20, 4, 104],
            "alt_counts": [8, 16, 45, 8, 16, 45, 8, 16, 45],
            "major_cn": [2, 2, 4, 2, 2, 4, 2, 2, 4],
            "minor_cn": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "normal_cn": [2, 2, 2, 2, 2, 2, 2, 2, 2],
            "tumour_content": [1.0, 0.2, 0.3, 1.0, 0.2, 0.3, 1.0, 0.2, 0.3],
            "error_rate": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        }
        df = pd.DataFrame(df_dict)
        actual_df = df.copy()
        samples = sorted(df["sample_id"].unique())
        actual_df = _remove_duplicated_and_partially_absent_mutations(actual_df, samples)
        self.assertGreater(len(df), len(actual_df))
        self.assertEqual(len(actual_df["mutation_id"].unique()), 1)

    def test_remove_duplicated_and_partially_absent_mutations__two_removed_dup_id(self):
        df_dict = {
            "mutation_id": ["m1", "m1", "m1", "m2", "m2", "m2", "m1", "m1", "m1", "m3", "m3", "m3", "m3", "m3", "m3"],
            "sample_id": ["s1", "s2", "s3", "s1", "s2", "s3", "s1", "s2", "s3", "s1", "s2", "s3", "s1", "s2", "s3"],
            "ref_counts": [20, 4, 104, 20, 4, 104, 20, 4, 104, 20, 4, 104, 20, 4, 104],
            "alt_counts": [8, 16, 45, 8, 16, 45, 8, 16, 45, 8, 16, 45, 8, 16, 45],
            "major_cn": [2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4],
            "minor_cn": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            "normal_cn": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "tumour_content": [1.0, 0.2, 0.3, 1.0, 0.2, 0.3, 1.0, 0.2, 0.3, 1.0, 0.2, 0.3, 1.0, 0.2, 0.3],
            "error_rate": [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4],
        }
        df = pd.DataFrame(df_dict)
        actual_df = df.copy()
        samples = sorted(df["sample_id"].unique())
        actual_df = _remove_duplicated_and_partially_absent_mutations(actual_df, samples)
        self.assertGreater(len(df), len(actual_df))
        self.assertEqual(len(actual_df["mutation_id"].unique()), 1)

    def test_process_required_cols_on_df__none_missing(self):
        df_dict = {
            "mutation_id": ["m1", "m1", "m1", "m2", "m2", "m2", "m3", "m3", "m3"],
            "sample_id": ["s1", "s2", "s3", "s1", "s2", "s3", "s1", "s2", "s3"],
            "ref_counts": [20, 4, 104, 20, 4, 104, 20, 4, 104],
            "alt_counts": [8, 16, 45, 8, 16, 45, 8, 16, 45],
            "major_cn": [2, 2, 4, 2, 2, 4, 2, 2, 4],
            "minor_cn": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "normal_cn": [2, 2, 2, 2, 2, 2, 2, 2, 2],
            "tumour_content": [1.0, 0.2, 0.3, 1.0, 0.2, 0.3, 1.0, 0.2, 0.3],
            "error_rate": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        }
        df = pd.DataFrame(df_dict)
        actual_df = df.copy()
        samples = sorted(df["sample_id"].unique())
        _process_required_cols_on_df(actual_df, samples)
        pd.testing.assert_frame_equal(df, actual_df)

    def test_process_required_cols_on_df__tumour_content_missing(self):
        df_dict = {
            "mutation_id": ["m1", "m1", "m1", "m2", "m2", "m2", "m3", "m3", "m3"],
            "sample_id": ["s1", "s2", "s3", "s1", "s2", "s3", "s1", "s2", "s3"],
            "ref_counts": [20, 4, 104, 20, 4, 104, 20, 4, 104],
            "alt_counts": [8, 16, 45, 8, 16, 45, 8, 16, 45],
            "major_cn": [2, 2, 4, 2, 2, 4, 2, 2, 4],
            "minor_cn": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "normal_cn": [2, 2, 2, 2, 2, 2, 2, 2, 2],
            "error_rate": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        }
        df = pd.DataFrame(df_dict)
        actual_df = df.copy()
        samples = sorted(df["sample_id"].unique())
        _process_required_cols_on_df(actual_df, samples)
        self.assertLess(len(df.columns), len(actual_df.columns))
        self.assertIn("tumour_content", actual_df.columns)

    def test_process_required_cols_on_df__error_rate_missing(self):
        df_dict = {
            "mutation_id": ["m1", "m1", "m1", "m2", "m2", "m2", "m3", "m3", "m3"],
            "sample_id": ["s1", "s2", "s3", "s1", "s2", "s3", "s1", "s2", "s3"],
            "ref_counts": [20, 4, 104, 20, 4, 104, 20, 4, 104],
            "alt_counts": [8, 16, 45, 8, 16, 45, 8, 16, 45],
            "major_cn": [2, 2, 4, 2, 2, 4, 2, 2, 4],
            "minor_cn": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "normal_cn": [2, 2, 2, 2, 2, 2, 2, 2, 2],
            "tumour_content": [1.0, 0.2, 0.3, 1.0, 0.2, 0.3, 1.0, 0.2, 0.3],
        }
        df = pd.DataFrame(df_dict)
        actual_df = df.copy()
        samples = sorted(df["sample_id"].unique())
        _process_required_cols_on_df(actual_df, samples)
        self.assertLess(len(df.columns), len(actual_df.columns))
        self.assertIn("error_rate", actual_df.columns)

    def test_create_loaded_pyclone_data_dict(self):
        df_dict = {
            "mutation_id": ["m1", "m1", "m1", "m2", "m2", "m2"],
            "sample_id": ["s1", "s2", "s3", "s1", "s2", "s3"],
            "ref_counts": [20, 4, 104, 5, 8, 96],
            "alt_counts": [8, 16, 45, 78, 56, 15],
            "major_cn": [2, 2, 4, 3, 4, 4],
            "minor_cn": [1, 2, 3, 2, 1, 3],
            "normal_cn": [2, 2, 2, 2, 2, 2],
            "tumour_content": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "error_rate": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        }
        skip_dict = {"m1": 0, "m2": 3}
        actual_df = pd.DataFrame(df_dict)
        samples = sorted(actual_df["sample_id"].unique())
        data = _create_loaded_pyclone_data_dict(actual_df, samples)
        self._assert_datapoint_dict_against_df_dict(actual_df, data, df_dict, skip_dict)

    def test_load_pyclone_data(self):
        df_dict = {
            "mutation_id": ["m1", "m1", "m1", "m2", "m2", "m2"],
            "sample_id": ["s1", "s2", "s3", "s1", "s2", "s3"],
            "ref_counts": [20, 4, 104, 5, 8, 96],
            "alt_counts": [8, 16, 45, 78, 56, 15],
            "major_cn": [2, 2, 4, 3, 4, 4],
            "minor_cn": [1, 2, 3, 2, 1, 3],
            "normal_cn": [2, 2, 2, 2, 2, 2],
            "tumour_content": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "error_rate": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        }
        skip_dict = {"m1": 0, "m2": 3}
        actual_df = pd.DataFrame(df_dict)
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "data.tsv")
            actual_df.to_csv(file_path, sep="\t", index=False)
            data, samples = load_pyclone_data(file_path)

        self._assert_datapoint_dict_against_df_dict(actual_df, data, df_dict, skip_dict)

    def test_get_major_cn_prior(self):
        maj_vals = [2, 2, 8, 10, 4, 1]
        min_vals = [1, 0, 3, 1, 4, 1]

        for cn_iter in range(len(maj_vals)):
            major = maj_vals[cn_iter]
            minor = min_vals[cn_iter]

            with self.subTest(
                msg="Major CN prior test {}".format(cn_iter),
                maj_cn_prior_test=cn_iter,
                major_val=major,
                minor_val=minor,
            ):
                actual_vals = get_major_cn_prior(major, minor, 2)
                expected_vals = tester_get_major_cn_prior(major, minor, 2)
                for i in range(len(actual_vals)):
                    np.testing.assert_array_equal(actual_vals[i], expected_vals[i])


if __name__ == "__main__":
    unittest.main()
