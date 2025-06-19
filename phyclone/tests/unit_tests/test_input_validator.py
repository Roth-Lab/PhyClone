import os
import secrets
import tarfile
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from phyclone.data.validator import create_cluster_input_validator_instance, create_data_input_validator_instance
from phyclone.data.validator.input_validator import InputValidator
from phyclone.data.validator.schema_error_builder import SchemaErrors
from phyclone.utils.exceptions import InputFormatError


class TesterInputValidator(InputValidator):
    def __init__(self, test_df, schema):
        self.df = test_df
        self.required_columns = set(schema["required"])
        self.optional_columns = set(schema["properties"]) - self.required_columns
        self.column_rules = schema["properties"]
        self.error_helper = SchemaErrors("Dummy_file_path.tsv")


class TestInputValidatorLoaders(unittest.TestCase):

    def test_data_validator_loads(self):
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
                "chrom": ["chr1", "chr2", "chr3"],
            }
            df = pd.DataFrame(df_dict)
            file_path = os.path.join(tmp_dir, "data.tsv")
            df.to_csv(file_path, sep="\t")
            input_validator = create_data_input_validator_instance(file_path)
        self.assertTrue(input_validator.validate())

    def test_data_validator_loads__trigger_delim_warning(self):
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
                "chrom": ["chr1", "chr2", "chr3"],
            }
            df = pd.DataFrame(df_dict)
            file_path = os.path.join(tmp_dir, "data.tsv")
            df.to_csv(file_path)
            with self.assertWarns(UserWarning) as w:
                input_validator = create_data_input_validator_instance(file_path)
        print(w.warning)
        self.assertTrue(input_validator.validate())

    def test_data_validator_loads__gzip(self):
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
                "chrom": ["chr1", "chr2", "chr3"],
            }
            df = pd.DataFrame(df_dict)
            file_path = os.path.join(tmp_dir, "data.tsv.gz")
            df.to_csv(file_path, sep="\t")
            input_validator = create_data_input_validator_instance(file_path)
        self.assertTrue(input_validator.validate())

    def test_data_validator_loads__gzip_trigger_delim_warning(self):
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
                "chrom": ["chr1", "chr2", "chr3"],
            }
            df = pd.DataFrame(df_dict)
            file_path = os.path.join(tmp_dir, "data.tsv.gz")
            df.to_csv(file_path)
            with self.assertWarns(UserWarning) as w:
                input_validator = create_data_input_validator_instance(file_path)
        print(w.warning)
        self.assertTrue(input_validator.validate())

    def test_data_validator_loads__bz2_trigger_delim_warning(self):
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
                "chrom": ["chr1", "chr2", "chr3"],
            }
            df = pd.DataFrame(df_dict)
            file_path = os.path.join(tmp_dir, "data.tsv.bz2")
            df.to_csv(file_path)
            with self.assertWarns(UserWarning) as w:
                input_validator = create_data_input_validator_instance(file_path)
        print(w.warning)
        self.assertTrue(input_validator.validate())

    def test_data_validator_loads__lzma_trigger_delim_warning(self):
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
                "chrom": ["chr1", "chr2", "chr3"],
            }
            df = pd.DataFrame(df_dict)
            file_path = os.path.join(tmp_dir, "data.tsv.xz")
            df.to_csv(file_path)
            with self.assertWarns(UserWarning) as w:
                input_validator = create_data_input_validator_instance(file_path)
        print(w.warning)
        self.assertTrue(input_validator.validate())

    def test_data_validator_loads__tar_trigger_delim_warning(self):
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
                "chrom": ["chr1", "chr2", "chr3"],
            }
            df = pd.DataFrame(df_dict)
            file_path = os.path.join(tmp_dir, "data.tar")
            df.to_csv(file_path)
            with self.assertWarns(UserWarning) as w:
                input_validator = create_data_input_validator_instance(file_path)
        print(w.warning)
        self.assertTrue(input_validator.validate())

    def test_data_validator_loads__tar_gz_trigger_delim_warning(self):
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
                "chrom": ["chr1", "chr2", "chr3"],
            }
            df = pd.DataFrame(df_dict)
            file_path = os.path.join(tmp_dir, "data.tar.gz")
            df.to_csv(file_path)
            with self.assertWarns(UserWarning) as w:
                input_validator = create_data_input_validator_instance(file_path)
        print(w.warning)
        self.assertTrue(input_validator.validate())

    def test_data_validator_loads__tar_bz2_trigger_delim_warning(self):
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
                "chrom": ["chr1", "chr2", "chr3"],
            }
            df = pd.DataFrame(df_dict)
            file_path = os.path.join(tmp_dir, "data.tar.bz2")
            df.to_csv(file_path)
            with self.assertWarns(UserWarning) as w:
                input_validator = create_data_input_validator_instance(file_path)
        print(w.warning)
        self.assertTrue(input_validator.validate())

    def test_data_validator_loads__tar_lzma_trigger_delim_warning(self):
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
                "chrom": ["chr1", "chr2", "chr3"],
            }
            df = pd.DataFrame(df_dict)
            file_path = os.path.join(tmp_dir, "data.tar.xz")
            df.to_csv(file_path)
            with self.assertWarns(UserWarning) as w:
                input_validator = create_data_input_validator_instance(file_path)
        print(w.warning)
        self.assertTrue(input_validator.validate())

    def test_data_validator_loads__tar_gz_trigger_tarchive_error(self):
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
                "chrom": ["chr1", "chr2", "chr3"],
            }
            df = pd.DataFrame(df_dict)
            file_path = os.path.join(tmp_dir, "data.tsv")
            file_path2 = os.path.join(tmp_dir, "data2.tsv")
            tar_path = os.path.join(tmp_dir, "data.tar.gz")
            df.to_csv(file_path, sep="\t", index=False)
            df.to_csv(file_path2, sep="\t", index=False)
            with tarfile.open(tar_path, "x:gz") as tar:
                tar.add(file_path)
                tar.add(file_path2)
            with self.assertRaises(ValueError) as e:
                _ = create_data_input_validator_instance(tar_path)
                self.assertTrue(
                    str(e.exception).startswith("Multiple files found in TAR archive. Only one file per TAR archive:")
                )
        print(e.exception)

    def test_data_validator_loads__tar_uncompressed_trigger_tarchive_error(self):
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
                "chrom": ["chr1", "chr2", "chr3"],
            }
            df = pd.DataFrame(df_dict)
            file_path = os.path.join(tmp_dir, "data.tsv")
            file_path2 = os.path.join(tmp_dir, "data2.tsv")
            tar_path = os.path.join(tmp_dir, "data.tar")
            df.to_csv(file_path, sep="\t", index=False)
            df.to_csv(file_path2, sep="\t", index=False)
            with tarfile.open(tar_path, "x:") as tar:
                tar.add(file_path)
                tar.add(file_path2)
            with self.assertRaises(ValueError) as e:
                _ = create_data_input_validator_instance(tar_path)
                self.assertTrue(
                    str(e.exception).startswith("Multiple files found in TAR archive. Only one file per TAR archive:")
                )
        print(e.exception)

    def test_data_validator_loads__zip_trigger_zip_warning(self):
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
                "chrom": ["chr1", "chr2", "chr3"],
            }
            df = pd.DataFrame(df_dict)
            file_path = os.path.join(tmp_dir, "data.zip")
            df.to_csv(file_path)
            with self.assertWarns(UserWarning) as w:
                input_validator = create_data_input_validator_instance(file_path)
        self.assertTrue(
            str(w.warning).startswith(
                "Zip file input detected, allowing pandas to infer parser dialect via python engine - "
            )
        )
        print(w.warning)
        self.assertTrue(input_validator.validate())

    # def test_data_validator_loads__zstd_trigger_delim_warning(self):
    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         df_dict = {
    #             "mutation_id": ["m1", "m2", "m3"],
    #             "sample_id": ["s1", "s2", "s3"],
    #             "ref_counts": [20, 4, 104],
    #             "alt_counts": [8, 16, 45],
    #             "major_cn": [2, 2, 4],
    #             "minor_cn": [1, 2, 3],
    #             "normal_cn": [2, 2, 2],
    #             "tumour_content": [1.0, 0.2, 0.3],
    #             "error_rate": [0.001, 0.002, 0.001],
    #             "chrom": ["chr1", "chr2", "chr3"],
    #         }
    #         df = pd.DataFrame(df_dict)
    #         file_path = os.path.join(tmp_dir, "data.zst")
    #         df.to_csv(file_path)
    #         with self.assertWarns(UserWarning) as w:
    #             input_validator = create_data_input_validator_instance(file_path)
    #     print(w.warning)
    #     self.assertTrue(input_validator.validate())

    def test_data_validator_loads_extra_col(self):
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
                "chrom": ["chr1", "chr2", "chr3"],
                "extra_col": [1, 2, 3],
            }
            df = pd.DataFrame(df_dict)
            file_path = os.path.join(tmp_dir, "data.tsv")
            df.to_csv(file_path, sep="\t")
            input_validator = create_data_input_validator_instance(file_path)
        self.assertTrue(input_validator.validate())

    def test_data_validator_loads__invalid_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            df_dict = {
                "mutation_id": ["m1", "m2", "m3"],
                "sample_id": ["s1", "s2", "s3"],
                "ref_counts": [20, -4, 104],
                "alt_counts": [8, 16, 45],
                "major_cn": [2, 2, 4],
                "minor_cn": [1, 2, 3],
                "normal_cn": [2, 2, 2],
                "tumour_content": [1.0, 0.2, 0.3],
                "error_rate": [0.001, 0.002, 0.001],
                "chrom": ["chr1", "chr2", "chr3"],
            }
            df = pd.DataFrame(df_dict)
            file_path = os.path.join(tmp_dir, "data.tsv")
            df.to_csv(file_path, sep="\t")
            input_validator = create_data_input_validator_instance(file_path)
        with self.assertRaises(InputFormatError) as error:
            input_validator.validate()
        print(error.exception)

    def test_cluster_validator_loads(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            df_dict = {
                "mutation_id": ["m1", "m2", "m3"],
                "sample_id": ["s1", "s2", "s3"],
                "cluster_id": [20, 4, 104],
                "cellular_prevalence": [0.001, 0.002, 0.001],
                "chrom": ["chr1", "chr2", "chr3"],
            }
            df = pd.DataFrame(df_dict)
            file_path = os.path.join(tmp_dir, "cluster.tsv")
            df.to_csv(file_path, sep="\t")
            input_validator = create_cluster_input_validator_instance(file_path)
        self.assertTrue(input_validator.validate())

    def test_cluster_validator_loads__gzip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            df_dict = {
                "mutation_id": ["m1", "m2", "m3"],
                "sample_id": ["s1", "s2", "s3"],
                "cluster_id": [20, 4, 104],
                "cellular_prevalence": [0.001, 0.002, 0.001],
                "chrom": ["chr1", "chr2", "chr3"],
            }
            df = pd.DataFrame(df_dict)
            file_path = os.path.join(tmp_dir, "cluster.tsv.gz")
            df.to_csv(file_path, sep="\t")
            input_validator = create_cluster_input_validator_instance(file_path)
        self.assertTrue(input_validator.validate())

    def test_cluster_validator_loads__invalid_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            df_dict = {
                "mutation_id": ["m1", "m2", "m3"],
                "sample_id": ["", "s2", "s3"],
                "cluster_id": ["20", "", "104"],
                "cellular_prevalence": [0.001, 0.002, 0.001],
                "chrom": ["chr1", "chr2", "chr3"],
            }
            df = pd.DataFrame(df_dict)
            file_path = os.path.join(tmp_dir, "cluster.tsv")
            df.to_csv(file_path, sep="\t")
            input_validator = create_cluster_input_validator_instance(file_path)
        with self.assertRaises(InputFormatError) as error:
            input_validator.validate()
        print(error.exception)


class BaseTest(object):
    class TestInputValidatorMethods(unittest.TestCase):
        def __init__(self, method_name: str = ...):
            super().__init__(method_name)
            self.schema = dict()
            self.set_schema()
            self.required_columns = set(self.schema["required"])
            self.optional_columns = set(self.schema["properties"]) - self.required_columns
            self.column_rules = self.schema["properties"]
            self.col_categories_dict = self.create_col_type_dict_from_schema()
            self.integer_col = self._get_a_col_for_category("integer")
            self.float_col = self._get_a_col_for_category("number")
            self.string_col = self._get_a_col_for_category("string")
            self.list_type_schema_col = self._get_a_col_for_category("list")
            self.string_col_w_min = self._get_a_col_for_category_with_min("string", "minLength", 1)
            self.integer_col_w_min = self._get_a_col_for_category_with_min("integer", "minimum", -np.inf)
            self.pos_integer_col = self._get_a_col_for_category_with_min("integer", "minimum", 0)

        def set_schema(self):
            raise NotImplementedError

        def create_input_validator_instance(self, df):
            input_validator = TesterInputValidator(df, self.schema)
            return input_validator

        def _run_raise_error_on_validate_test(self, input_validator):
            with self.assertRaises(InputFormatError) as error:
                input_validator.validate()
            print(error.exception)

        def create_col_type_dict_from_schema(self):
            list_type_cols = []
            integer_type_cols = []
            float_type_cols = []
            string_type_cols = []
            longest_list_type_length = 0
            longest_list_type_col_name = None
            for col, col_rule in self.column_rules.items():
                if isinstance(col_rule["type"], list):
                    list_type_cols.append(col)
                    curr_type_list_length = len(col_rule["type"])
                    if curr_type_list_length > longest_list_type_length:
                        longest_list_type_length = curr_type_list_length
                        longest_list_type_col_name = col
                else:
                    col_type = col_rule["type"]
                    if col_type == "integer":
                        integer_type_cols.append(col)
                    elif col_type == "number":
                        float_type_cols.append(col)
                    elif col_type == "string":
                        string_type_cols.append(col)

            col_categories_dict = {
                "integer": integer_type_cols,
                "number": float_type_cols,
                "string": string_type_cols,
                "list": list_type_cols,
                "longest_list_type_col_name": longest_list_type_col_name,
            }
            return col_categories_dict

        def _get_a_col_for_category(self, check_type):
            category_list = self.col_categories_dict[check_type]

            if len(category_list) == 0:
                col_name = "{ck}_col_{r}".format(ck=check_type, r=secrets.randbits(8))
                tmp = {col_name: {"type": check_type}}
                if check_type == "integer" or check_type == "number":
                    tmp[col_name]["minimum"] = 0
                elif check_type == "string":
                    tmp[col_name]["minLength"] = 1
                elif check_type == "list":
                    tmp[col_name]["type"] = ["string", "number", "integer"]
                self.schema["properties"].update(tmp)
                self.col_categories_dict[check_type].append(col_name)
                return col_name
            else:
                return category_list[0]

        def _get_a_col_for_category_with_min(self, check_type, min_type, min_val):
            category_list = self.col_categories_dict[check_type]

            for col in category_list:
                col_rule = self.column_rules[col]
                if min_type in col_rule:
                    curr_min = col_rule[min_type]
                    if curr_min >= min_val:
                        return col

            if min_val == -np.inf:
                min_val = 0

            col_name = "{ck}_min_col_{r}".format(ck=check_type, r=secrets.randbits(8))
            tmp = {col_name: {"type": check_type, min_type: min_val}}
            self.schema["properties"].update(tmp)
            self.col_categories_dict[check_type].append(col_name)
            return col_name

        @staticmethod
        def create_valid_col_data_for_simple_type(col_type, min_val):
            if col_type == "integer":
                min_val = int(min_val)
                return [min_val + i for i in range(3)]
            elif col_type == "number":
                min_val = float(min_val)
                return [min_val + float(i * 0.25) for i in range(3)]
            elif col_type == "string":
                min_val = int(min_val)
                return [str(i) * min_val for i in range(3)]
            else:
                raise NotImplementedError("Column type {} not supported".format(col_type))

        def create_valid_data_for_any_col(self, col, list_type_idx=0):
            col_rule = self.column_rules[col]
            col_type = col_rule["type"]
            if isinstance(col_type, list):
                if list_type_idx >= len(col_type):
                    list_type_idx = -1
                col_type = col_type[list_type_idx]
            if col_type == "integer" or col_type == "number":
                if "minimum" in col_rule:
                    min_val = col_rule["minimum"]
                else:
                    min_val = 0
            elif col_type == "string":
                if "minLength" in col_rule:
                    min_val = col_rule["minLength"]
                else:
                    min_val = 1
            else:
                min_val = 1
            return self.create_valid_col_data_for_simple_type(col_type, min_val)

        def test_validate_base_type__not_implemented_type(self):
            df = pd.DataFrame({self.integer_col: [1, 2, 3]})
            input_validator = self.create_input_validator_instance(df)
            with self.assertRaises(NotImplementedError):
                input_validator._validate_base_type("bool", self.integer_col)

        def test_validate_base_type__integer_valid_input(self):
            df = pd.DataFrame({self.integer_col: [1, 2, 3]})
            input_validator = self.create_input_validator_instance(df)
            self.assertTrue(input_validator._validate_base_type("integer", self.integer_col))

        def test_validate_base_type__integer_invalid_input_float(self):
            df = pd.DataFrame({self.integer_col: [1.5, 2.5, 3.5]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_base_type("integer", self.integer_col))

        def test_validate_base_type__integer_invalid_input_string(self):
            df = pd.DataFrame({self.integer_col: ["1.5", "2.5", "3.5"]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_base_type("integer", self.integer_col))

        def test_check_column_minimum__integer_invalid_input_NaN(self):
            df = pd.DataFrame({self.integer_col: [1, None, 3]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(
                input_validator._check_column_minimum(self.column_rules[self.integer_col], "integer", self.integer_col)
            )

        def test_validate_base_type__number_valid_input(self):
            df = pd.DataFrame({self.float_col: [1.5, 2.5, 3.5]})
            input_validator = self.create_input_validator_instance(df)
            self.assertTrue(input_validator._validate_base_type("number", self.float_col))

        def test_validate_base_type__number_invalid_input_integer(self):
            df = pd.DataFrame({self.float_col: [1, 2, 3]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_base_type("number", self.float_col))

        def test_validate_base_type__number_invalid_input_string(self):
            df = pd.DataFrame({self.float_col: ["1", "2", "3"]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_base_type("number", self.float_col))

        def test_check_column_minimum__number_invalid_input_NaN(self):
            df = pd.DataFrame({self.float_col: [1.5, 2.5, None]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(
                input_validator._check_column_minimum(self.column_rules[self.float_col], "number", self.float_col)
            )

        def test_validate_base_type__string_valid_input(self):
            df = pd.DataFrame({self.string_col: ["m1", "m2", "m3"]})
            input_validator = self.create_input_validator_instance(df)
            self.assertTrue(input_validator._validate_base_type("string", self.string_col))

        def test_validate_base_type__string_invalid_input_integer(self):
            df = pd.DataFrame({self.string_col: [1, 2, 3]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_base_type("string", self.string_col))

        def test_validate_base_type__string_invalid_input_float(self):
            df = pd.DataFrame({self.string_col: [1.5, 2.5, 3.5]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_base_type("string", self.string_col))

        def test_validate_column__simple_type_valid(self):
            df = pd.DataFrame({self.integer_col: [1, 2, 3]})
            input_validator = self.create_input_validator_instance(df)
            input_validator._validate_base_type = MagicMock(return_value=True)
            self.assertTrue(input_validator._validate_column(self.integer_col))

        def test_validate_column__simple_type_invalid(self):
            df = pd.DataFrame({self.integer_col: ["1", "2", "3"]})
            input_validator = self.create_input_validator_instance(df)
            input_validator._validate_base_type = MagicMock(return_value=False)
            self.assertFalse(input_validator._validate_column(self.integer_col))

        def test_validate_column__list_type_valid(self):
            df = pd.DataFrame({self.list_type_schema_col: [1, 2, 3]})
            input_validator = self.create_input_validator_instance(df)
            col_rule_dict = self.column_rules[self.list_type_schema_col]
            type_list_len = len(col_rule_dict["type"])
            side_effect_list = [False] * type_list_len
            side_effect_list[-1] = True
            input_validator._validate_base_type = MagicMock(side_effect=side_effect_list)
            self.assertTrue(input_validator._validate_column(self.list_type_schema_col))

        def test_validate_column__list_type_invalid(self):
            df = pd.DataFrame({self.list_type_schema_col: [1, 2, 3]})
            input_validator = self.create_input_validator_instance(df)
            col_rule_dict = self.column_rules[self.list_type_schema_col]
            type_list_len = len(col_rule_dict["type"])
            side_effect_list = [False] * type_list_len
            input_validator._validate_base_type = MagicMock(side_effect=side_effect_list)
            self.assertFalse(input_validator._validate_column(self.list_type_schema_col))

        def test_validate_column__simple_type_valid_integration(self):
            col_data = self.create_valid_data_for_any_col(self.integer_col)
            df = pd.DataFrame({self.integer_col: col_data})
            input_validator = self.create_input_validator_instance(df)
            self.assertTrue(input_validator._validate_column(self.integer_col))

        def test_validate_column__simple_type_invalid_integration(self):
            df = pd.DataFrame({self.integer_col: ["1", "2", "3"]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_column(self.integer_col))

        def test_validate_column__list_type_valid_integration(self):
            col_data = self.create_valid_data_for_any_col(self.list_type_schema_col)
            df = pd.DataFrame({self.list_type_schema_col: col_data})
            input_validator = self.create_input_validator_instance(df)
            self.assertTrue(input_validator._validate_column(self.list_type_schema_col))

        def test_validate_column__list_type_invalid_integration(self):
            df = pd.DataFrame({self.list_type_schema_col: [True, False, True]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_column(self.list_type_schema_col))

        def test_validate_column__minLength_valid(self):
            df = pd.DataFrame({self.string_col_w_min: ["1", "2", "3"]})
            input_validator = self.create_input_validator_instance(df)
            self.assertTrue(input_validator._validate_column(self.string_col))

        def test_validate_column__minLength_invalid(self):
            df = pd.DataFrame({self.string_col_w_min: ["1", "2", ""]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_column(self.string_col))

        def test_validate_column__minimum_valid(self):
            col_data = self.create_valid_data_for_any_col(self.integer_col_w_min)
            df = pd.DataFrame({self.integer_col_w_min: col_data})
            input_validator = self.create_input_validator_instance(df)
            self.assertTrue(input_validator._validate_column(self.integer_col_w_min))

        def test_validate_column__minimum_valid_all_zero(self):
            df = pd.DataFrame({self.pos_integer_col: [0, 0, 0]})
            input_validator = self.create_input_validator_instance(df)
            self.assertTrue(input_validator._validate_column(self.pos_integer_col))

        def test_validate_column__minimum_invalid(self):
            df = pd.DataFrame({self.pos_integer_col: [1, 2, -3]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_column(self.pos_integer_col))

        def test_validate_required_column_presence__all_present(self):
            df_dict = {}
            for col in self.required_columns:
                df_dict[col] = [True]
            for opt_col in self.optional_columns:
                df_dict[opt_col] = [True]
            df = pd.DataFrame(df_dict)
            input_validator = self.create_input_validator_instance(df)
            self.assertTrue(input_validator._validate_required_column_presence())

        def test_validate_required_column_presence__only_req_present(self):
            df_dict = {}
            for col in self.required_columns:
                df_dict[col] = [True]
            df = pd.DataFrame(df_dict)
            input_validator = self.create_input_validator_instance(df)
            self.assertTrue(input_validator._validate_required_column_presence())

        def test_validate_required_column_presence__only_opt_present(self):
            df_dict = {}
            for opt_col in self.optional_columns:
                df_dict[opt_col] = [True]
            df = pd.DataFrame(df_dict)
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_required_column_presence())

        def test_validate_required_column_presence__some_req_missing(self):
            df_dict = {}
            num_req_cols = len(self.required_columns)
            stop_iter = num_req_cols - 1
            for col in self.required_columns:
                if len(df_dict) == stop_iter:
                    break
                df_dict[col] = [True]
            for opt_col in self.optional_columns:
                df_dict[opt_col] = [True]
            df = pd.DataFrame(df_dict)
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_required_column_presence())

        def test_validate__all_valid_and_present(self):
            longest_list_type_col = self.col_categories_dict["longest_list_type_col_name"]
            if longest_list_type_col is not None:
                list_types = self.column_rules[longest_list_type_col]["type"]

                for i, list_col_type in enumerate(list_types):
                    with self.subTest(msg="List column type {}".format(list_col_type), list_type_used=list_col_type):
                        df_dict = self.create_all_schema_cols_valid(i)
                        df = pd.DataFrame(df_dict)
                        input_validator = self.create_input_validator_instance(df)
                        self.assertTrue(input_validator.validate())
            else:
                df_dict = self.create_all_schema_cols_valid(0)
                df = pd.DataFrame(df_dict)
                input_validator = self.create_input_validator_instance(df)
                self.assertTrue(input_validator.validate())

        def test_validate__one_req_col_missing(self):
            df_dict = self.create_all_schema_cols_valid(0)
            for req_col in self.required_columns:
                with self.subTest(msg="Required col missing {}".format(req_col), missing_req_col=req_col):
                    req_col_val = df_dict.pop(req_col)
                    df = pd.DataFrame(df_dict)
                    input_validator = self.create_input_validator_instance(df)
                    self._run_raise_error_on_validate_test(input_validator)
                    df_dict[req_col] = req_col_val

        def test_validate__all_req_col_missing(self):
            df_dict = self.create_all_schema_opt_cols_valid(0)
            df = pd.DataFrame(df_dict)
            input_validator = self.create_input_validator_instance(df)
            self._run_raise_error_on_validate_test(input_validator)

        def test_validate_all_columns__one_invalid_col(self):
            df_dict = self.create_all_schema_cols_valid(0)
            all_cols = self.optional_columns.union(self.required_columns)
            for col in all_cols:
                invalid_data_list = self.create_invalid_data_for_col(col)
                for invalid_data_tuple in invalid_data_list:
                    invalid_data = invalid_data_tuple[1]
                    how_invalid = invalid_data_tuple[0]
                    with self.subTest(
                        msg="Invalid col: {}, {}".format(col, how_invalid),
                        invalid_col=col,
                        how_invalid=how_invalid,
                    ):
                        valid_col_val = df_dict[col]
                        df_dict[col] = invalid_data
                        df = pd.DataFrame(df_dict)
                        input_validator = self.create_input_validator_instance(df)
                        self.assertFalse(input_validator._validate_all_columns())
                        df_dict[col] = valid_col_val

        def test_validate__one_invalid_col(self):
            df_dict = self.create_all_schema_cols_valid(0)
            all_cols = self.optional_columns.union(self.required_columns)
            for col in all_cols:
                invalid_data_list = self.create_invalid_data_for_col(col)
                for invalid_data_tuple in invalid_data_list:
                    invalid_data = invalid_data_tuple[1]
                    how_invalid = invalid_data_tuple[0]
                    with self.subTest(
                        msg="Invalid col: {}, {}".format(col, how_invalid),
                        invalid_col=col,
                        how_invalid=how_invalid,
                    ):
                        valid_col_val = df_dict[col]
                        df_dict[col] = invalid_data
                        df = pd.DataFrame(df_dict)
                        input_validator = self.create_input_validator_instance(df)
                        self._run_raise_error_on_validate_test(input_validator)
                        df_dict[col] = valid_col_val

        def create_invalid_data_for_col(self, col):
            invalid_data_list = [("Invalid type", [True, False, True])]
            col_rule = self.column_rules[col]
            if "minimum" in col_rule:
                min_val = col_rule["minimum"]
                invalid_data = [min_val - i for i in range(3)]
                invalid_data_list.append(("Min value invalid", invalid_data))
            elif "minLength" in col_rule:
                min_val = col_rule["minLength"]
                if min_val > 0:
                    invalid_data = ["", "", ""]
                    invalid_data_list.append(("Min str length invalid", invalid_data))
            return invalid_data_list

        def create_all_schema_req_cols_valid(self, list_type_idx):
            df_dict = {col: self.create_valid_data_for_any_col(col, list_type_idx) for col in self.required_columns}
            return df_dict

        def create_all_schema_opt_cols_valid(self, list_type_idx):
            df_dict = {col: self.create_valid_data_for_any_col(col, list_type_idx) for col in self.optional_columns}
            return df_dict

        def create_all_schema_cols_valid(self, list_type_idx):
            df_dict = self.create_all_schema_req_cols_valid(list_type_idx)
            df_dict.update(self.create_all_schema_opt_cols_valid(list_type_idx))
            return df_dict


class TestDataInputValidator(BaseTest.TestInputValidatorMethods):

    def set_schema(self):
        self.schema = {
            "properties": {
                "mutation_id": {"type": ["number", "string", "integer"], "minLength": 1},
                "sample_id": {"type": ["number", "string", "integer"], "minLength": 1},
                "ref_counts": {"type": "integer", "minimum": 0},
                "alt_counts": {"type": "integer", "minimum": 0},
                "major_cn": {"type": "integer", "minimum": 0},
                "minor_cn": {"type": "integer", "minimum": 0},
                "normal_cn": {"type": "integer", "minimum": 0},
                "tumour_content": {"type": "number", "minimum": 0.0, "default": 1.0},
                "error_rate": {"type": "number", "minimum": 0.0, "default": 0.001},
                "chrom": {"type": ["number", "string", "integer"], "minLength": 1},
            },
            "required": ["mutation_id", "sample_id", "ref_counts", "alt_counts", "major_cn", "minor_cn", "normal_cn"],
        }

    def test_validate__all_invalid_cols(self):
        df_dict = {
            "mutation_id": [True, False, True],
            "sample_id": [True, False, True],
            "ref_counts": [-20, -4, -104],
            "alt_counts": [-8, -16, -45],
            "major_cn": [-2, -2, -4],
            "minor_cn": [-1, -2, -3],
            "normal_cn": [2, 2, -2],
            "tumour_content": [1, 2, 3],
            "error_rate": ["0.001", "0.002", "0.001"],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self._run_raise_error_on_validate_test(input_validator)

    def test_validate__all_invalid_cols_one_req_missing(self):
        df_dict = {
            "mutation_id": [True, False, True],
            "ref_counts": [-20, -4, -104],
            "alt_counts": [-8, -16, -45],
            "major_cn": [-2, -2, -4],
            "minor_cn": [-1, -2, -3],
            "normal_cn": [-2, -2, -2],
            "tumour_content": [1, 2, 3],
            "error_rate": ["0.001", "0.002", "0.001"],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self._run_raise_error_on_validate_test(input_validator)


class TestClusterInputValidator(BaseTest.TestInputValidatorMethods):

    def set_schema(self):
        self.schema = {
            "properties": {
                "mutation_id": {"type": ["number", "string", "integer"], "minLength": 1},
                "sample_id": {"type": ["number", "string", "integer"], "minLength": 1},
                "cluster_id": {"type": ["number", "string", "integer"], "minLength": 1},
                "cellular_prevalence": {"type": "number"},
                "chrom": {"type": ["number", "string", "integer"], "minLength": 1},
            },
            "required": ["mutation_id", "sample_id", "cluster_id"],
        }

    def test_validate__all_invalid_cols(self):
        df_dict = {
            "mutation_id": [True, False, True],
            "sample_id": ["", None, "1"],
            "cluster_id": ["", "4", "104"],
            "cellular_prevalence": ["0.001", "0.002", "0.001"],
            "chrom": [True, False, True],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self._run_raise_error_on_validate_test(input_validator)

    def test_validate__all_invalid_cols_one_req_missing(self):
        df_dict = {
            "sample_id": ["", None, "1"],
            "cluster_id": ["20", "4", ""],
            "cellular_prevalence": ["0.001", "0.002", "0.001"],
            "chrom": [True, False, True],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self._run_raise_error_on_validate_test(input_validator)


if __name__ == "__main__":
    unittest.main()
