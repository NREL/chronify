class ChronifyExceptionBase(Exception):
    """Base class for exceptions in this package"""


class InvalidTable(ChronifyExceptionBase):
    """Raised when a table does not match its schema."""


class InvalidParameter(ChronifyExceptionBase):
    """Raised when an invalid parameter is passed."""
