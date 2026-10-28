# Converted from src/pipeline/Utility.h and src/pipeline/ErrorTracker.h

from enum import IntEnum

# String length limits (from Utility.h)
MAX_LEN_ATOM_NAME = 5
MAX_LEN_ATOM_TYPE = 5
MAX_LEN_ATOM_HYBRID = 5
MAX_LEN_ATOM_DONOR = 5
MAX_LEN_ATOM_ACCEPTOR = 5
MAX_LEN_RES_NAME = 5
MAX_LEN_CHAIN_NAME = 3
MAX_LEN_STRUCTURE_NAME = 5
MAX_LEN_ONE_LINE_CONTENT = 1024
MAX_LEN_FILE_NAME = 100
MAX_LEN_ERR_MSG = 1024

COMMENT_LINE_SYMBOL1 = '!'
COMMENT_LINE_SYMBOL2 = '#'


class ErrorCode(IntEnum):
    Success = 0
    Warning = 1
    NameError = -1
    IOError = -2
    FormatError = -3
    IndexError = -4
    ValueError = -5
    ZeroDivisonError = -6
    DataNotExistError = -7
    InvalidInputError = -8
    Exception = -10


def failed(error_code: int) -> bool:
    """Return True if error_code indicates failure."""
    return error_code < 0



# Copyright (c) Xiaoqiang Huang