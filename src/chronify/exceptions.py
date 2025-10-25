class ChronifyExceptionBase(Exception):
    """Base class for exceptions in this package"""


class ConflictingInputsError(ChronifyExceptionBase):
    """Raised when user inputs conflict with each other."""


class InvalidTable(ChronifyExceptionBase):
    """Raised when a table does not match its schema."""


class InvalidOperation(ChronifyExceptionBase):
    """Raised when an invalid operation is requested."""


class InvalidModel(ChronifyExceptionBase):
    """Raised when an invalid model is passed."""


class InvalidParameter(ChronifyExceptionBase):
    """Raised when an invalid parameter is passed."""


class InvalidValue(ChronifyExceptionBase):
    """Raised when an invalid value is passed."""


class MissingValue(ChronifyExceptionBase):
    """Raised when an expecting value is missing."""


class MissingParameter(ChronifyExceptionBase):
    """Raised when a parameter is not found or missing."""


class TableAlreadyExists(ChronifyExceptionBase):
    """Raised when a table already exists in engine."""


class TableNotStored(ChronifyExceptionBase):
    """Raised when a table is not stored."""
