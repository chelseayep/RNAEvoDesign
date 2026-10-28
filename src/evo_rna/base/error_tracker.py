# Converted from src/pipeline/ErrorTracker.h

from .constants import ErrorCode, MAX_LEN_ERR_MSG, failed


def trace_error(err_msg: str, error_code: int) -> int:
    """Log error and return error code."""
    if len(err_msg) > MAX_LEN_ERR_MSG:
        err_msg = err_msg[:MAX_LEN_ERR_MSG]
    # In C this writes to stderr; in Python we could use logging or print
    print(f"Error: {err_msg} (code={error_code})", flush=True)
    return error_code