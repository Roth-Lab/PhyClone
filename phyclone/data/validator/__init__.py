from importlib.resources import files, as_file

from .input_validator import InputValidator


def create_cluster_input_validator_instance(file_path):
    source = files("phyclone.data.validator").joinpath("cluster_file_schema.json")
    with as_file(source) as fspath:
        validator = InputValidator(file_path, fspath)
    return validator


def create_data_input_validator_instance(file_path):
    source = files("phyclone.data.validator").joinpath("PhyClone_schema.json")
    with as_file(source) as fspath:
        validator = InputValidator(file_path, fspath)
    return validator
