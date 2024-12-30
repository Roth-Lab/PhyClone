import pandas as pd
import csv
import warnings
import json

class InputValidator(object):

    def __init__(self, file_path, schema_file):
        self.df = self.load_df(file_path)
        schema = self.load_json_schema(schema_file)
        self.required_columns = set(schema['required'])
        self.optional_columns = set(schema['properties']) - self.required_columns
        self.column_rules = schema['properties']

    @staticmethod
    def load_json_schema(schema_file):
        with open(schema_file) as f:
            schema = json.load(f)
        return schema

    @staticmethod
    def load_df(file_path):
        csv_delim = ','
        with open(file_path, 'r') as csv_file:
            dialect = csv.Sniffer().sniff(csv_file.readline())
            csv_delim = str(dialect.delimiter)

        if csv_delim != "\t":
            warnings.warn("Input should be tab-delimited, supplied file is delimited by {}\n"
                          "Will attempt parsing with the current delimiter\n".format(repr(csv_delim)),
                          stacklevel=2)
        return pd.read_csv(file_path, sep=csv_delim)

    def validate(self):
        are_required_columns_present = self.validate_required_column_presence()
        are_all_columns_valid = self.validate_all_columns()
        return are_required_columns_present and are_all_columns_valid

    def validate_required_column_presence(self):
        curr_cols = set(self.df.columns)
        missing_cols = set(self.required_columns) - curr_cols
        return len(missing_cols) == 0

    def validate_all_columns(self):
        curr_cols = self.df.columns
        are_all_cols_valid = True
        for col in curr_cols:
            is_curr_col_valid = self.validate_column(col)
            are_all_cols_valid = are_all_cols_valid and is_curr_col_valid
        return are_all_cols_valid

    def validate_base_type(self, base_type, column):
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

    def validate_column(self, column):
        col_rule = self.column_rules[column]

        is_type_valid, col_type = self._check_column_type(col_rule, column)

        if is_type_valid:
            is_min_valid = self._check_column_minimum(col_rule, col_type, column)
        else:
            is_min_valid = False

        return is_type_valid and is_min_valid

    def _check_column_minimum(self, col_rule, col_type, column):
        is_min_valid = True
        if "minimum" in col_rule and (col_type == "integer" or col_type == "number"):
            is_min_valid = (self.df[column] >= 0).all()
        elif "minLength" in col_rule and col_type == "string":
            is_min_valid = self.df[column].str.len().ge(col_rule["minLength"]).all(skipna=False)
        return is_min_valid

    def _check_column_type(self, col_rule, column):
        is_type_valid = False
        col_type = col_rule['type']
        loaded_col_type = col_rule['type']
        if isinstance(col_type, list):
            for chk_type in col_type:
                is_type_valid = self.validate_base_type(chk_type, column)
                if is_type_valid:
                    loaded_col_type = chk_type
                    break
        else:
            is_type_valid = self.validate_base_type(col_type, column)
        return is_type_valid, loaded_col_type
