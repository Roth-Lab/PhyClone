class MajorCopyNumberError(Exception):
    def __init__(self, major_cn, minor_cn):
        error_msg = (
            "\nMajorCopyNumberError:\n"
            "Major copy number should not be smaller than minor copy number."
            "\n-> Major CN: {maj}, Minor CN: {minor}".format(maj=major_cn, minor=minor_cn)
        )
        super().__init__(error_msg)
        self.__suppress_context__ = True


class InputFormatError(Exception):
    def __init__(self, errors_list):
        error_msg = "\n\nInputFormatError:\n"
        errors_list_string = "\n\n".join(map(str, errors_list))
        error_msg += errors_list_string
        super().__init__(error_msg)
        self.__suppress_context__ = True
