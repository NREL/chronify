class ChronifyExceptionBase(Exception):
    """Base class for exceptions in this package"""


class ConflictingInputsError(ChronifyExceptionBase):
    """Raised when user inputs conflict with each other."""


class InvalidTable(ChronifyExceptionBase):
    """Raised when a table does not match its schema."""


class InvalidOperation(ChronifyExceptionBase):
    """Raised when an invalid operation is requested."""


class InvalidParameter(ChronifyExceptionBase):
    """Raised when an invalid parameter is passed."""


class MissingParameter(ChronifyExceptionBase):
    """Raised when a parameter is not found or missing."""


class TableAlreadyExists(ChronifyExceptionBase):
    """Raised when a table already exists in engine."""


class TableNotStored(ChronifyExceptionBase):
    """Raised when a table is not stored."""
