class ChronifyExceptionBase(Exception):
    """Base class for exceptions in this package"""


class InvalidTable(ChronifyExceptionBase):
    """Raised when a table does not match its schema."""
