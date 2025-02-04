from __future__ import annotations

from typing import Optional, TextIO
from typing_extensions import Self


class FileIterator:
    """
    File iterator helper class.

    Parameters
    ----------
    f : TextIO
        File handle.
    count : int, default 0
        Line count.

    """
    __name__: str = "FileIterator"
    __qualname__: str = "toughio.FileIterator"

    def __init__(self, f: TextIO, count: int = 0) -> None:
        """Initialize a file iterator."""
        self.f = f
        self.count = count
        self.fiter = iter(f.readline, "")

    def __iter__(self) -> Self:
        """Return iterator."""
        return self

    def __next__(self) -> str:
        """Return next item."""
        self.count += 1
        self.line = next(self.fiter)

        return self.line

    def next(self, skip_empty: bool = False, comments: Optional[str] = None) -> str:
        """Return next line."""
        if skip_empty:
            while True:
                line = self.__next__().strip()

                if comments:
                    if line and not line.startswith(comments):
                        self.line = line

                        return line

                elif line:
                    self.line = line

                    return line

        elif comments:
            while True:
                line = self.__next__().strip()

                if not line.startswith(comments):
                    self.line = line

                    return line

        else:
            self.line = self.__next__()

            return self.line

    def seek(self, i: int, increment: int) -> None:
        """Set file's position."""
        self.count += increment
        self.f.seek(i)

    def tell(self) -> int:
        """Return current position of file."""
        return self.f.tell()
