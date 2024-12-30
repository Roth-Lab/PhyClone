import os
import tempfile
import unittest
from phyclone.data.validator.input_validator import InputValidator
from phyclone.data.validator.schema_error_builder import SchemaErrors
from phyclone.data.validator import create_cluster_input_validator_instance, create_data_input_validator_instance
import pandas as pd
from unittest.mock import MagicMock
from phyclone.utils.exceptions import InputFormatError


class TesterInputValidator(InputValidator):
    def __init__(self, test_df, schema):
        self.df = test_df
        self.required_columns = set(schema["required"])
        self.optional_columns = set(schema["properties"]) - self.required_columns
        self.column_rules = schema["properties"]
        self.error_helper = SchemaErrors()

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
                "extra_col": [1, 2, 3]
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

    def test_cluster_validator_loads__invalid_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            df_dict = {
                "mutation_id": ["m1", "m2", "m3"],
                "sample_id": ["", "s2", "s3"],
                "cluster_id": [20, None, 104],
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
            self.schema = None
            self.required_columns = None
            self.optional_columns = None
            self.integer_col = None
            self.float_col = None
            self.string_col = None
            self.list_type_schema_col = None
            self.column_rules = None

        def create_input_validator_instance(self, df):
            input_validator = TesterInputValidator(df, self.schema)
            return input_validator

        def test_validate_base_type_not_implemented_type(self):
            df = pd.DataFrame({self.integer_col: [1, 2, 3]})
            input_validator = self.create_input_validator_instance(df)
            with self.assertRaises(NotImplementedError):
                input_validator._validate_base_type("bool", self.integer_col)

        def test_validate_base_type_integer_valid_input(self):
            df = pd.DataFrame({self.integer_col: [1, 2, 3]})
            input_validator = self.create_input_validator_instance(df)
            self.assertTrue(input_validator._validate_base_type("integer", self.integer_col))

        def test_validate_base_type_integer_invalid_input_float(self):
            df = pd.DataFrame({self.integer_col: [1.5, 2.5, 3.5]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_base_type("integer", self.integer_col))

        def test_validate_base_type_integer_invalid_input_string(self):
            df = pd.DataFrame({self.integer_col: ["1.5", "2.5", "3.5"]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_base_type("integer", self.integer_col))

        def test_validate_base_type_number_valid_input(self):
            df = pd.DataFrame({self.float_col: [1.5, 2.5, 3.5]})
            input_validator = self.create_input_validator_instance(df)
            self.assertTrue(input_validator._validate_base_type("number", self.float_col))

        def test_validate_base_type_number_invalid_input_integer(self):
            df = pd.DataFrame({self.float_col: [1, 2, 3]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_base_type("number", self.float_col))

        def test_validate_base_type_number_invalid_input_string(self):
            df = pd.DataFrame({self.float_col: ["1", "2", "3"]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_base_type("number", self.float_col))

        def test_validate_base_type_string_valid_input(self):
            df = pd.DataFrame({self.string_col: ["m1", "m2", "m3"]})
            input_validator = self.create_input_validator_instance(df)
            self.assertTrue(input_validator._validate_base_type("string", self.string_col))

        def test_validate_base_type_string_invalid_input_integer(self):
            df = pd.DataFrame({self.string_col: [1, 2, 3]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_base_type("string", self.string_col))

        def test_validate_base_type_string_invalid_input_float(self):
            df = pd.DataFrame({self.string_col: [1.5, 2.5, 3.5]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_base_type("string", self.string_col))

        def test_validate_column_simple_type_valid(self):
            df = pd.DataFrame({self.integer_col: [1, 2, 3]})
            input_validator = self.create_input_validator_instance(df)
            input_validator._validate_base_type = MagicMock(return_value=True)
            self.assertTrue(input_validator._validate_column(self.integer_col))

        def test_validate_column_simple_type_invalid(self):
            df = pd.DataFrame({self.integer_col: ["1", "2", "3"]})
            input_validator = self.create_input_validator_instance(df)
            input_validator._validate_base_type = MagicMock(return_value=False)
            self.assertFalse(input_validator._validate_column(self.integer_col))

        def test_validate_column_list_type_valid(self):
            df = pd.DataFrame({self.list_type_schema_col: [1, 2, 3]})
            input_validator = self.create_input_validator_instance(df)
            col_rule_dict = self.column_rules[self.list_type_schema_col]
            type_list_len = len(col_rule_dict["type"])
            side_effect_list = [False] * type_list_len
            side_effect_list[-1] = True
            input_validator._validate_base_type = MagicMock(side_effect=side_effect_list)
            self.assertTrue(input_validator._validate_column(self.list_type_schema_col))

        def test_validate_column_list_type_invalid(self):
            df = pd.DataFrame({self.list_type_schema_col: [1, 2, 3]})
            input_validator = self.create_input_validator_instance(df)
            col_rule_dict = self.column_rules[self.list_type_schema_col]
            type_list_len = len(col_rule_dict["type"])
            side_effect_list = [False] * type_list_len
            input_validator._validate_base_type = MagicMock(side_effect=side_effect_list)
            self.assertFalse(input_validator._validate_column(self.list_type_schema_col))

        def test_validate_column_simple_type_valid_integration(self):
            df = pd.DataFrame({self.integer_col: [1, 2, 3]})
            input_validator = self.create_input_validator_instance(df)
            self.assertTrue(input_validator._validate_column(self.integer_col))

        def test_validate_column_simple_type_invalid_integration(self):
            df = pd.DataFrame({self.integer_col: ["1", "2", "3"]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_column(self.integer_col))

        def test_validate_column_list_type_valid_integration(self):
            df = pd.DataFrame({self.list_type_schema_col: [1, 2, 3]})
            input_validator = self.create_input_validator_instance(df)
            self.assertTrue(input_validator._validate_column(self.list_type_schema_col))

        def test_validate_column_list_type_invalid_integration(self):
            df = pd.DataFrame({self.list_type_schema_col: [True, False, True]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_column(self.list_type_schema_col))

        def test_validate_column_minLength_valid(self):
            df = pd.DataFrame({self.string_col: ["1", "2", "3"]})
            input_validator = self.create_input_validator_instance(df)
            self.assertTrue(input_validator._validate_column(self.string_col))

        def test_validate_column_minLength_invalid(self):
            df = pd.DataFrame({self.string_col: ["1", "2", ""]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_column(self.string_col))

        def test_validate_column_minimum_valid(self):
            df = pd.DataFrame({self.integer_col: [1, 2, 3]})
            input_validator = self.create_input_validator_instance(df)
            self.assertTrue(input_validator._validate_column(self.integer_col))

        def test_validate_column_minimum_valid_all_zero(self):
            df = pd.DataFrame({self.integer_col: [0, 0, 0]})
            input_validator = self.create_input_validator_instance(df)
            self.assertTrue(input_validator._validate_column(self.integer_col))

        def test_validate_column_minimum_invalid(self):
            df = pd.DataFrame({self.integer_col: [1, 2, -3]})
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_column(self.integer_col))

        def test_validate_required_column_presence_all_present(self):
            df_dict = {}
            for col in self.required_columns:
                df_dict[col] = [True]
            for opt_col in self.optional_columns:
                df_dict[opt_col] = [True]
            df = pd.DataFrame(df_dict)
            input_validator = self.create_input_validator_instance(df)
            self.assertTrue(input_validator._validate_required_column_presence())

        def test_validate_required_column_presence_only_req_present(self):
            df_dict = {}
            for col in self.required_columns:
                df_dict[col] = [True]
            df = pd.DataFrame(df_dict)
            input_validator = self.create_input_validator_instance(df)
            self.assertTrue(input_validator._validate_required_column_presence())

        def test_validate_required_column_presence_only_opt_present(self):
            df_dict = {}
            for opt_col in self.optional_columns:
                df_dict[opt_col] = [True]
            df = pd.DataFrame(df_dict)
            input_validator = self.create_input_validator_instance(df)
            self.assertFalse(input_validator._validate_required_column_presence())

        def test_validate_required_column_presence_some_req_missing(self):
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

        def _run_raise_error_on_validate_test(self, input_validator):
            with self.assertRaises(InputFormatError) as error:
                input_validator.validate()
            print(error.exception)


class TestDataInputValidator(BaseTest.TestInputValidatorMethods):

    def setUp(self):
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
        self.required_columns = set(self.schema["required"])
        self.optional_columns = set(self.schema["properties"]) - self.required_columns
        self.integer_col = "ref_counts"
        self.float_col = "tumour_content"
        self.string_col = "mutation_id"
        self.list_type_schema_col = "sample_id"
        self.column_rules = self.schema["properties"]

    def test_validate_valid_and_present(self):
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
        input_validator = self.create_input_validator_instance(df)
        self.assertTrue(input_validator.validate())

    def test_validate_valid_one_req_col_missing(self):
        df_dict = {
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
        input_validator = self.create_input_validator_instance(df)
        self._run_raise_error_on_validate_test(input_validator)

    def test_validate_valid_all_req_col_missing(self):
        df_dict = {
            "tumour_content": [1.0, 0.2, 0.3],
            "error_rate": [0.001, 0.002, 0.001],
            "chrom": ["chr1", "chr2", "chr3"],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self._run_raise_error_on_validate_test(input_validator)

    def test_validate_all_columns_valid_string_cols_used(self):
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
        input_validator = self.create_input_validator_instance(df)
        self.assertTrue(input_validator._validate_all_columns())

    def test_validate_all_columns_valid_number_cols_used(self):
        df_dict = {
            "mutation_id": [1.0, 2.0, 3.0],
            "sample_id": [1.0, 2.0, 3.0],
            "ref_counts": [20, 4, 104],
            "alt_counts": [8, 16, 45],
            "major_cn": [2, 2, 4],
            "minor_cn": [1, 2, 3],
            "normal_cn": [2, 2, 2],
            "tumour_content": [1.0, 0.2, 0.3],
            "error_rate": [0.001, 0.002, 0.001],
            "chrom": [1.0, 2.0, 3.0],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self.assertTrue(input_validator._validate_all_columns())

    def test_validate_all_columns_valid_integer_cols_used(self):
        df_dict = {
            "mutation_id": [1, 2, 3],
            "sample_id": [1, 2, 3],
            "ref_counts": [20, 4, 104],
            "alt_counts": [8, 16, 45],
            "major_cn": [2, 2, 4],
            "minor_cn": [1, 2, 3],
            "normal_cn": [2, 2, 2],
            "tumour_content": [1.0, 0.2, 0.3],
            "error_rate": [0.001, 0.002, 0.001],
            "chrom": [1, 2, 3],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self.assertTrue(input_validator._validate_all_columns())

    def test_validate_all_columns_valid_mixed_cols_used(self):
        df_dict = {
            "mutation_id": ["m1", "m2", "m3"],
            "sample_id": [1, 2, 3],
            "ref_counts": [20, 4, 104],
            "alt_counts": [8, 16, 45],
            "major_cn": [2, 2, 4],
            "minor_cn": [1, 2, 3],
            "normal_cn": [2, 2, 2],
            "tumour_content": [1.0, 0.2, 0.3],
            "error_rate": [0.001, 0.002, 0.001],
            "chrom": [1.0, 2.0, 3.0],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self.assertTrue(input_validator._validate_all_columns())

    def test_validate_all_columns_one_invalid_col(self):
        df_dict = {
            "mutation_id": ["m1", "m2", "m3"],
            "sample_id": ["s1", "s2", "s3"],
            "ref_counts": [-20, 4, 104],
            "alt_counts": [8, 16, 45],
            "major_cn": [2, 2, 4],
            "minor_cn": [1, 2, 3],
            "normal_cn": [2, 2, 2],
            "tumour_content": [1.0, 0.2, 0.3],
            "error_rate": [0.001, 0.002, 0.001],
            "chrom": ["chr1", "chr2", "chr3"],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self.assertFalse(input_validator._validate_all_columns())

    def test_validate_all_columns_two_invalid_cols(self):
        df_dict = {
            "mutation_id": ["m1", "m2", "m3"],
            "sample_id": ["s1", "s2", "s3"],
            "ref_counts": [-20, -4, -104],
            "alt_counts": [-8, -16, -45],
            "major_cn": [2, 2, 4],
            "minor_cn": [1, 2, 3],
            "normal_cn": [2, 2, 2],
            "tumour_content": [1.0, 0.2, 0.3],
            "error_rate": [0.001, 0.002, 0.001],
            "chrom": ["chr1", "chr2", "chr3"],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self.assertFalse(input_validator._validate_all_columns())

    def test_validate_all_columns_all_invalid_cols(self):
        df_dict = {
            "mutation_id": [True, False, True],
            "sample_id": [True, False, True],
            "ref_counts": [-20, -4, -104],
            "alt_counts": [-8, -16, -45],
            "major_cn": [-2, -2, -4],
            "minor_cn": [-1, -2, -3],
            "normal_cn": [-2, -2, -2],
            "tumour_content": [1, 2, 3],
            "error_rate": ["0.001", "0.002", "0.001"],
            "chrom": [True, False, True],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self.assertFalse(input_validator._validate_all_columns())

    def test_validate_one_invalid_col(self):
        df_dict = {
            "mutation_id": ["m1", "m2", "m3"],
            "sample_id": ["s1", "s2", "s3"],
            "ref_counts": [-20, 4, 104],
            "alt_counts": [8, 16, 45],
            "major_cn": [2, 2, 4],
            "minor_cn": [1, 2, 3],
            "normal_cn": [2, 2, 2],
            "tumour_content": [1.0, 0.2, 0.3],
            "error_rate": [0.001, 0.002, 0.001],
            "chrom": ["chr1", "chr2", "chr3"],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self._run_raise_error_on_validate_test(input_validator)

    def test_validate_two_invalid_cols(self):
        df_dict = {
            "mutation_id": ["m1", "m2", "m3"],
            "sample_id": ["s1", "s2", "s3"],
            "ref_counts": [-20, -4, -104],
            "alt_counts": [-8, -16, -45],
            "major_cn": [2, 2, 4],
            "minor_cn": [1, 2, 3],
            "normal_cn": [2, 2, 2],
            "tumour_content": [1.0, 0.2, 0.3],
            "error_rate": [0.001, 0.002, 0.001],
            "chrom": ["chr1", "chr2", "chr3"],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self._run_raise_error_on_validate_test(input_validator)

    def test_validate_all_invalid_cols(self):
        df_dict = {
            "mutation_id": [True, False, True],
            "sample_id": [True, False, True],
            "ref_counts": [-20, -4, -104],
            "alt_counts": [-8, -16, -45],
            "major_cn": [-2, -2, -4],
            "minor_cn": [-1, -2, -3],
            "normal_cn": [-2, -2, -2],
            "tumour_content": [1, 2, 3],
            "error_rate": ["0.001", "0.002", "0.001"],
            "chrom": [True, False, True],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self._run_raise_error_on_validate_test(input_validator)

    def test_validate_all_invalid_cols_one_req_missing(self):
        df_dict = {
            "mutation_id": [True, False, True],
            "ref_counts": [-20, -4, -104],
            "alt_counts": [-8, -16, -45],
            "major_cn": [-2, -2, -4],
            "minor_cn": [-1, -2, -3],
            "normal_cn": [-2, -2, -2],
            "tumour_content": [1, 2, 3],
            "error_rate": ["0.001", "0.002", "0.001"],
            "chrom": ["", None, True],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self._run_raise_error_on_validate_test(input_validator)


class TestClusterInputValidator(BaseTest.TestInputValidatorMethods):

    def setUp(self):
        self.schema = {
            "properties": {
                "mutation_id": {"type": ["number", "string", "integer"], "minLength": 1},
                "sample_id": {"type": ["number", "string", "integer"], "minLength": 1},
                "cluster_id": {"type": "integer"},
                "cellular_prevalence": {"type": "number"},
                "chrom": {"type": ["number", "string", "integer"]},
            },
            "required": ["mutation_id", "sample_id", "cluster_id"],
        }
        self.required_columns = set(self.schema["required"])
        self.optional_columns = set(self.schema["properties"]) - self.required_columns
        self.integer_col = "cluster_id"
        self.float_col = "cellular_prevalence"
        self.string_col = "mutation_id"
        self.list_type_schema_col = "sample_id"
        self.column_rules = self.schema["properties"]

    def test_validate_column_minimum_invalid(self):
        self.skipTest("Cluster schema lacks integer col with defined minimum")

    def test_validate_valid_and_present(self):
        df_dict = {
            "mutation_id": ["m1", "m2", "m3"],
            "sample_id": ["s1", "s2", "s3"],
            "cluster_id": [20, 4, 104],
            "cellular_prevalence": [0.001, 0.002, 0.001],
            "chrom": ["chr1", "chr2", "chr3"],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self.assertTrue(input_validator.validate())

    def test_validate_valid_one_req_col_missing(self):
        df_dict = {
            "sample_id": ["s1", "s2", "s3"],
            "cluster_id": [20, 4, 104],
            "cellular_prevalence": [0.001, 0.002, 0.001],
            "chrom": ["chr1", "chr2", "chr3"],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self._run_raise_error_on_validate_test(input_validator)

    def test_validate_valid_all_req_col_missing(self):
        df_dict = {
            "cellular_prevalence": [0.001, 0.002, 0.001],
            "chrom": ["chr1", "chr2", "chr3"],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self._run_raise_error_on_validate_test(input_validator)

    def test_validate_all_columns_valid_string_cols_used(self):
        df_dict = {
            "mutation_id": ["m1", "m2", "m3"],
            "sample_id": ["s1", "s2", "s3"],
            "cluster_id": [20, 4, 104],
            "cellular_prevalence": [0.001, 0.002, 0.001],
            "chrom": ["chr1", "chr2", "chr3"],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self.assertTrue(input_validator._validate_all_columns())

    def test_validate_all_columns_valid_number_cols_used(self):
        df_dict = {
            "mutation_id": [1.0, 2.0, 3.0],
            "sample_id": [1.0, 2.0, 3.0],
            "cluster_id": [20, 4, 104],
            "cellular_prevalence": [0.001, 0.002, 0.001],
            "chrom": [1.0, 2.0, 3.0],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self.assertTrue(input_validator._validate_all_columns())

    def test_validate_all_columns_valid_integer_cols_used(self):
        df_dict = {
            "mutation_id": [1, 2, 3],
            "sample_id": [1, 2, 3],
            "cluster_id": [20, 4, 104],
            "cellular_prevalence": [0.001, 0.002, 0.001],
            "chrom": [1, 2, 3],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self.assertTrue(input_validator._validate_all_columns())

    def test_validate_all_columns_valid_mixed_cols_used(self):
        df_dict = {
            "mutation_id": ["m1", "m2", "m3"],
            "sample_id": [1, 2, 3],
            "cluster_id": [20, 4, 104],
            "cellular_prevalence": [0.001, 0.002, 0.001],
            "chrom": [1.0, 2.0, 3.0],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self.assertTrue(input_validator._validate_all_columns())

    def test_validate_all_columns_one_invalid_col(self):
        df_dict = {
            "mutation_id": ["m1", "m2", "m3"],
            "sample_id": ["s1", "s2", "s3"],
            "cluster_id": [20, 4, 104],
            "cellular_prevalence": [1, 2, 1],
            "chrom": ["chr1", "chr2", "chr3"],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self.assertFalse(input_validator._validate_all_columns())

    def test_validate_all_columns_two_invalid_cols(self):
        df_dict = {
            "mutation_id": [True, False, True],
            "sample_id": ["s1", "s2", "s3"],
            "cluster_id": [20, 4, 104],
            "cellular_prevalence": [1, 2, 1],
            "chrom": ["chr1", "chr2", "chr3"],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self.assertFalse(input_validator._validate_all_columns())

    def test_validate_all_columns_all_invalid_cols(self):
        df_dict = {
            "mutation_id": [True, False, True],
            "sample_id": [True, False, True],
            "cluster_id": ["20", "4", "104"],
            "cellular_prevalence": ["0.001", "0.002", "0.001"],
            "chrom": [True, False, True],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self.assertFalse(input_validator._validate_all_columns())

    def test_validate_one_invalid_col(self):
        df_dict = {
            "mutation_id": ["m1", "m2", "m3"],
            "sample_id": ["s1", "s2", "s3"],
            "cluster_id": [20, 4, 104],
            "cellular_prevalence": [1, 2, 1],
            "chrom": ["chr1", "chr2", "chr3"],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self._run_raise_error_on_validate_test(input_validator)

    def test_validate_two_invalid_cols(self):
        df_dict = {
            "mutation_id": [True, False, True],
            "sample_id": ["s1", "s2", "s3"],
            "cluster_id": [20, 4, 104],
            "cellular_prevalence": [1, 2, 1],
            "chrom": ["chr1", "chr2", "chr3"],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self._run_raise_error_on_validate_test(input_validator)

    def test_validate_all_invalid_cols(self):
        df_dict = {
            "mutation_id": [True, False, True],
            "sample_id": ["", None, "1"],
            "cluster_id": ["20", "4", "104"],
            "cellular_prevalence": ["0.001", "0.002", "0.001"],
            "chrom": [True, False, True],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self._run_raise_error_on_validate_test(input_validator)

    def test_validate_all_invalid_cols_one_req_missing(self):
        df_dict = {
            "sample_id": ["", None, "1"],
            "cluster_id": ["20", "4", "104"],
            "cellular_prevalence": ["0.001", "0.002", "0.001"],
            "chrom": [True, False, True],
        }
        df = pd.DataFrame(df_dict)
        input_validator = self.create_input_validator_instance(df)
        self._run_raise_error_on_validate_test(input_validator)


if __name__ == "__main__":
    unittest.main()
