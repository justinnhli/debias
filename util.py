from pathlib import Path
from typing import Iterator, Set, List

def read_file(path):
    # type: (Path) -> Iterator[str]
    """Read lines from a file, ignoring comments that begin with "#".

    Parameters:
        path (Path): The path of the text file.

    Yields:
        str: Each line, stripped of beginning and ending whitespace.
    """
    with path.open() as fd:
        for line in fd:
            line = line.strip()
            if not line.startswith('#'):
                yield line


def read_word_list(path):
    # type: (Path) -> Set[str]
    """Read a gender pairs file.

    Parameters:
        path (Path): The path of the words file.

    Returns:
        Set[str]: A list of the words in the file.
    """
    words = set()
    for line in read_file(path):
        words |= set(line.split())
    return words


def read_word_groups(path):
    # type: (Path) -> List[Set[str]]
    """Read a word groups file.

    Parameters:
        path (Path): The path of the word groups  file.

    Returns:
        List[Set[str]]: A list of group of words.
    """
    groups = []
    for line in read_file(path):
        groups.append(set(line.split()))
    return groups 
