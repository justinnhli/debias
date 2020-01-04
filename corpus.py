from pathlib import Path
from random import Random
from typing import Any, Iterable, Sequence, Iterator


def _randomly_swap_words(word_groups, *strings, random_seed=8675309, **kwargs):
    # type: (Iterable[Iterable[str]], *str, Any, **Any) -> Iterator[str]
    """Swap words in strings with another randomly chosen word.

    Parameters:
        word_groups (Iterable[Iterable[str]]): A collection of word groups to swap.
        *strings (str): The strings to swap in.
        random_seed (Any): The random seed for the swapping. Optional.
        **kwargs: Other keyword parameters.

    Yields:
        str: The strings with the words swapped.
    """
    # TODO deal with duplicates, eg. [['him', 'her'], ['his', 'her']]
    rng = Random(random_seed)
    candidates = {}
    for group in word_groups:
        group = sorted(group)
        for word in group:
            candidates[word] = group
    for string in strings:
        result = []
        for word in string.split():
            if word in candidates:
                result.append(rng.choice(candidates[word]))
            else:
                result.append(word)
        yield ' '.join(result)


def create_swapped_corpus(corpus_file, word_groups, out_file=None, **kwargs):
    # type: (Path, Iterable[Iterable[str]], Path, **Any) -> Path
    """Create a randomized, word-swapped corpus.

    Parameters:
        corpus_file (Path): The path of the input corpus file.
        word_groups (Iterable[Iterable[str]]): A collection of word groups to swap.
        out_file (Path): The path of the resulting corpus file. Optional.
        **kwargs: Other keyword parameters.

    Returns:
        Path: The path of the resulting corpus file.
    """
    if out_file is None:
        out_file = corpus_file.parent.joinpath(corpus_file.name + '.random')
    if out_file.exists():
        return out_file
    with corpus_file.open() as in_fd:
        lines = (line.strip() for line in in_fd.readlines())
        with out_file.open('w') as out_fd:
            for line in _randomly_swap_words(word_groups, *lines):
                out_fd.write(line)
                out_fd.write('\n')
    return out_file
