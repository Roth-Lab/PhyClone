import pandas as pd
import csv
import warnings
import json

# ['cellular_prevalence']



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
            warnings.warn("Input should be tab-delimited, supplied file is delimited by {}".format(repr(csv_delim)),
                          stacklevel=2)
        return pd.read_csv(file_path, sep=csv_delim)

    # def validate_base_type(self, col_dtype, base_type):
    #     pass

    def validate_base_type(self, base_type, column):
        if base_type == "integer":
            return self.validate_int(self.df[column].dtype.name)
        elif base_type == "string":
            col_dtype = self.df[column].dtype.name
            if col_dtype == "object":
                self.df[column] = self.df[column].astype(str)
                col_dtype = self.df[column].dtype.name
            return self.validate_str(col_dtype)
        elif base_type == "number":
            is_num = self.validate_int(self.df[column].dtype.name)
            if not is_num:
                is_num = self.validate_num(self.df[column].dtype.name)
            return is_num



    @staticmethod
    def validate_int(col_dtype):
        return isinstance(col_dtype, int)

    @staticmethod
    def validate_str(col_dtype):
        pass

    @staticmethod
    def validate_num(col_dtype):
        pass

    def validate_column(self, column):
        col_rule = self.column_rules[column]

        #test type
        col_type = col_rule['type']
        # col_dtype = self.df[column].dtype.name
        if isinstance(col_type, list):
            for chk_type in col_type:
                # self.validate_base_type(col_dtype, chk_type)
                self.validate_base_type(chk_type, column)

        #test min

        #test default


if __name__ == "__main__":
    tmp_val = InputValidator("/home/emilia/RothLab/Projects/phyclone/examples/data/mixing.tsv", "PhyClone_schema.json")

    tmp_val.validate_column('mutation_id')

    print("d")