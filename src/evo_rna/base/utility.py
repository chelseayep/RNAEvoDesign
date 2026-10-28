
# Converted from src/pipeline/Utility.h

from typing import List, Tuple

from .constants import (
    MAX_LEN_ATOM_NAME,
    MAX_LEN_CHAIN_NAME,
    MAX_LEN_ONE_LINE_CONTENT,
    MAX_LEN_RES_NAME,
    MAX_LEN_STRUCTURE_NAME,
    ErrorCode,
    failed,
)

class StringArray:
    """List of strings with C API-style methods."""

    def __init__(self) -> None:
        self._strings: List[str] = []

    def get_count(self) -> int:
        return len(self._strings)

    def get(self, index: int) -> str:
        return self._strings[index]

    def set(self, index: int, src_string: str) -> int:
        self._strings[index] = src_string
        return ErrorCode.Success

    def find(self, src_string: str) -> Tuple[int, int]:
        """Returns (ErrorCode, position). Position -1 if not found."""
        try:
            pos = self._strings.index(src_string)
            return (ErrorCode.Success, pos)
        except ValueError:
            return (ErrorCode.DataNotExistError, -1)

    def insert(self, pos: int, src_string: str) -> int:
        self._strings.insert(pos, src_string)
        return ErrorCode.Success

    def remove(self, pos: int) -> int:
        self._strings.pop(pos)
        return ErrorCode.Success

    def append(self, src_string: str) -> int:
        self._strings.append(src_string)
        return ErrorCode.Success

    def copy_from(self, other: "StringArray") -> int:
        self._strings = list(other._strings)
        return ErrorCode.Success

    def split_string(self, src_str: str, splitter: str) -> int:
        self._strings = [s.strip() for s in src_str.split(splitter) if s.strip()]
        return ErrorCode.Success


class IntArray:
    def __init__(self, length: int = 0) -> None:
        self._content: List[int] = [0] * length if length else []

    def get_length(self) -> int:
        return len(self._content)

    def get(self, index: int) -> int:
        return self._content[index]

    def set(self, index: int, value: int) -> int:
        self._content[index] = value
        return ErrorCode.Success

    def resize(self, new_length: int) -> int:
        if new_length > len(self._content):
            self._content.extend([0] * (new_length - len(self._content)))
        else:
            self._content = self._content[:new_length]
        return ErrorCode.Success

    def insert(self, index: int, value: int) -> int:
        self._content.insert(index, value)
        return ErrorCode.Success

    def remove(self, index: int) -> int:
        self._content.pop(index)
        return ErrorCode.Success

    def append(self, value: int) -> int:
        self._content.append(value)
        return ErrorCode.Success

    def find(self, num: int) -> int:
        try:
            return self._content.index(num)
        except ValueError:
            return -1

    def copy_from(self, other: "IntArray") -> int:
        self._content = list(other._content)
        return ErrorCode.Success


class DoubleArray:
    def __init__(self, length: int = 0) -> None:
        self._content: List[float] = [0.0] * length if length else []

    def get_length(self) -> int:
        return len(self._content)

    def get(self, index: int) -> float:
        return self._content[index]

    def set(self, index: int, value: float) -> int:
        self._content[index] = value
        return ErrorCode.Success

    def resize(self, new_length: int) -> int:
        if new_length > len(self._content):
            self._content.extend([0.0] * (new_length - len(self._content)))
        else:
            self._content = self._content[:new_length]
        return ErrorCode.Success

    def insert(self, index: int, value: float) -> int:
        self._content.insert(index, value)
        return ErrorCode.Success

    def remove(self, index: int) -> int:
        self._content.pop(index)
        return ErrorCode.Success

    def append(self, value: float) -> int:
        self._content.append(value)
        return ErrorCode.Success

    def copy_from(self, other: "DoubleArray") -> int:
        self._content = list(other._content)
        return ErrorCode.Success

    def get_all(self) -> List[float]:
        return self._content


class FileReader:
    """Reads a file into lines and supports line-by-line iteration."""

    def __init__(self) -> None:
        self._lines: List[str] = []
        self._position: int = 0

    def create(self, path: str) -> int:
        try:
            with open(path) as f:
                self._lines = [line.rstrip("\r\n") for line in f]
            self._position = 0
            return ErrorCode.Success
        except OSError:
            return ErrorCode.IOError

    def get_line_count(self) -> int:
        return len(self._lines)

    def get_line(self, index: int) -> str:
        return self._lines[index]

    def get_current_pos(self) -> int:
        return self._position

    def set_current_pos(self, index: int) -> int:
        self._position = index
        return ErrorCode.Success

    def get_next_line(self) -> Tuple[int, str]:
        """Returns (ErrorCode, line). Line empty if EOF."""
        if self._position >= len(self._lines):
            return (ErrorCode.DataNotExistError, "")
        line = self._lines[self._position]
        self._position += 1
        return (ErrorCode.Success, line)

    def end_of_file(self) -> bool:
        return self._position >= len(self._lines)


def extract_target_string_from_source_string(
    src: str, start: int, length: int
) -> str:
    """Extract substring; C's ExtractTargetStringFromSourceString(dest, src, start, length)."""
    end = min(start + length, len(src))
    return src[start:end].strip()


def extract_first_string_from_source_string(src: str) -> str:
    """First whitespace-delimited token."""
    return src.split()[0] if src.split() else ""
