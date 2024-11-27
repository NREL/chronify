class ChronifyExceptionBase(Exception):
    """Base class for exceptions in this package"""


class ConflictingInputsError(ChronifyExceptionBase):
    """Raised when user inputs conflict with each other."""


class InvalidTable(ChronifyExceptionBase):
    """Raised when a table does not match its schema."""


class InvalidParameter(ChronifyExceptionBase):
    """Raised when an invalid parameter is passed."""


class MissingParameter(ChronifyExceptionBase):
    """Raised when a parameter is not found or missing."""


class TableAlreadyExists(ChronifyExceptionBase):
    """Raised when a table already exists in engine."""
