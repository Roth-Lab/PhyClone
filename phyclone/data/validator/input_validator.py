import pandas as pd
import csv
import warnings
import json
from phyclone.data.validator.schema_error_builder import SchemaErrors
import gzip


class InputValidator(object):

    def __init__(self, file_path, schema_file):
        self.df = self.load_df(file_path)
        schema = self.load_json_schema(schema_file)
        self.required_columns = set(schema["required"])
        self.optional_columns = set(schema["properties"]) - self.required_columns
        self.column_rules = schema["properties"]
        self.error_helper = SchemaErrors()

    @staticmethod
    def load_json_schema(schema_file):
        with open(schema_file) as f:
            schema = json.load(f)
        return schema

    @staticmethod
    def load_df(file_path):

        try:
            with open(file_path, "r") as csv_file:
                dialect = csv.Sniffer().sniff(csv_file.readline())
                csv_delim = str(dialect.delimiter)
        except UnicodeDecodeError:
            with gzip.open(file_path, "rt") as csv_file:
                dialect = csv.Sniffer().sniff(csv_file.readline())
                csv_delim = str(dialect.delimiter)

        if csv_delim != "\t":
            warnings.warn(
                "Input should be tab-delimited, supplied file is delimited by {delim}\n"
                "Will attempt parsing with the current delimiter, {delim}\n".format(delim=repr(csv_delim)),
                stacklevel=2,
                category=UserWarning,
            )
        return pd.read_csv(file_path, sep=csv_delim)

    def validate(self):
        are_required_columns_present = self._validate_required_column_presence()
        are_all_columns_valid = self._validate_all_columns()
        self.error_helper.raise_errors()
        return are_required_columns_present and are_all_columns_valid

    def _validate_required_column_presence(self):
        curr_cols = set(self.df.columns)
        missing_cols = set(self.required_columns) - curr_cols
        for col in missing_cols:
            self.error_helper.add_missing_column_error(col)
        return len(missing_cols) == 0

    def _validate_all_columns(self):
        curr_cols = self.df.columns
        all_defined_cols = self.optional_columns.union(self.required_columns)
        all_defined_cols = all_defined_cols.intersection(set(curr_cols))
        are_all_cols_valid = True
        for col in all_defined_cols:
            is_curr_col_valid = self._validate_column(col)
            are_all_cols_valid = are_all_cols_valid and is_curr_col_valid
        return are_all_cols_valid

    def _validate_base_type(self, base_type, column):
        if base_type == "integer":
            return self._validate_int(self.df[column].dtype)
        elif base_type == "string":
            col_dtype = self.df[column].dtype
            if col_dtype == object:
                self.df[column] = self.df[column].astype("string")
                col_dtype = self.df[column].dtype
            return self._validate_str(col_dtype)
        elif base_type == "number":
            return self._validate_num(self.df[column].dtype)
        else:
            raise NotImplementedError("Base type {} is not implemented yet".format(base_type))

    @staticmethod
    def _validate_int(col_dtype):
        return col_dtype == int

    @staticmethod
    def _validate_str(col_dtype):
        return col_dtype == pd.StringDtype()

    @staticmethod
    def _validate_num(col_dtype):
        return col_dtype == float

    def _validate_column(self, column):
        col_rule = self.column_rules[column]
        required_type = col_rule["type"]

        is_type_valid, col_type = self._check_column_type(col_rule, column)

        if is_type_valid:
            is_min_valid = self._check_column_minimum(col_rule, col_type, column)
        else:
            is_min_valid = False
            invalid_type_msg = "Column is of invalid type {}, valid type(s): {}".format(col_type, required_type)
            self.error_helper.add_invalid_column_error(column, invalid_type_msg)

        return is_type_valid and is_min_valid

    def _check_column_minimum(self, col_rule, col_type, column):
        is_min_valid = True
        invalid_type_msg = "Column minimum value/length violation"
        if col_type != "string":
            is_min_valid = pd.notnull(self.df[column]).all()
            invalid_type_msg = "Column contains missing or NaN values"
        if "minimum" in col_rule and (col_type == "integer" or col_type == "number"):
            min_value = col_rule["minimum"]
            is_min_valid = (self.df[column] >= min_value).all(skipna=False)
            invalid_type_msg = "Column contains elements that violate the required minimum value of {}".format(
                min_value
            )
        elif "minLength" in col_rule and col_type == "string":
            min_length = col_rule["minLength"]
            self.df[column] = self.df[column].fillna(value="")
            is_min_valid = self.df[column].str.len().ge(min_length).all(skipna=False)
            invalid_type_msg = "Column contains elements that violate the required minimum string length of {}".format(
                min_length
            )
        if not is_min_valid:
            self.error_helper.add_invalid_column_error(column, invalid_type_msg)
        return is_min_valid

    def _check_column_type(self, col_rule, column):
        is_type_valid = False
        col_type = col_rule["type"]
        loaded_col_type = self.df[column].dtype.name
        if isinstance(col_type, list):
            for chk_type in col_type:
                is_type_valid = self._validate_base_type(chk_type, column)
                if is_type_valid:
                    loaded_col_type = chk_type
                    break
        else:
            is_type_valid = self._validate_base_type(col_type, column)
            if is_type_valid:
                loaded_col_type = col_type
        return is_type_valid, loaded_col_type
