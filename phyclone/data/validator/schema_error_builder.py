from dataclasses import dataclass
from phyclone.utils.exceptions import InputFormatError


@dataclass(slots=True)
class SchemaError(object):
    error_type: str
    column_errors: dict
    error_count: int

    def __str__(self):
        error_msg = "{type}, affected column count: {count}".format(type=self.error_type, count=self.error_count)
        error_template = "\n\t- {col_name}: {error}"
        for col_name, error in self.column_errors.items():
            error_msg += error_template.format(col_name=col_name, error=error)

        return error_msg


class SchemaErrorBuilder(object):
    __slots__ = ("_error_type", "_column_errors")

    def __init__(self):
        self._error_type = ""
        self._column_errors = {}

    def with_column_error(self, column_name, error_reason):
        self._column_errors[column_name] = error_reason
        return self

    def with_error_type(self, error_type):
        self._error_type = error_type
        return self

    def build(self):
        return SchemaError(self._error_type, self._column_errors, len(self._column_errors))


class SchemaErrors(object):
    __slots__ = ("errors", "_missing_column_error", "_invalid_column_error")

    def __init__(self):
        self.errors = set()
        self._missing_column_error = SchemaErrorBuilder().with_error_type("Missing column(s)")
        self._invalid_column_error = SchemaErrorBuilder().with_error_type("Invalid column(s)")

    def add_invalid_column_error(self, column_name, error_reason):
        error_reason = error_reason.replace("number", "float")
        self._invalid_column_error.with_column_error(column_name, error_reason)
        self.errors.add(self._invalid_column_error)

    def add_missing_column_error(self, column_name):
        self._missing_column_error.with_column_error(column_name, "Required column missing")
        self.errors.add(self._missing_column_error)

    def raise_errors(self):
        if len(self.errors) > 0:
            built_errors = [err.build() for err in self.errors]
            raise InputFormatError(built_errors)
