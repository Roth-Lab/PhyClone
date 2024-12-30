class MajorCopyNumberError(Exception):
    def __init__(self, major_cn, minor_cn):
        error_msg = (
            "Major copy number should not be smaller than minor copy number"
            "\n Major_CN: {maj}, Minor CN: {minor}".format(maj=major_cn, minor=minor_cn)
        )
        super().__init__(error_msg)


class InputFormatError(Exception):
    def __init__(self, errors_list):
        error_msg = "\n"
        for error in errors_list:
            error_msg += str(error) + "\n\n"
        super().__init__(error_msg)
