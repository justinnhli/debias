"""Highest level functions for word embedding debiasing."""

from pathlib import Path
from typing import Any, Iterable, Tuple

from corpus import create_swapped_corpus
from training import create_fasttext_model
from debias import bolukbasi_debias_original
from word_embedding import WordEmbedding


def create_baseline_model(corpus_file, **kwargs):
    # type: (Path, **Any) -> WordEmbedding
    """Create a baseline FastText model.

    Parameters:
        corpus_file (Path): The text corpus.
        **kwargs: Other keyword arguments.

    Returns:
        WordEmbedding: The model built from the corpus.
    """
    return create_fasttext_model(corpus_file, **kwargs)


def create_generalized_bolukbasi_model(corpus_file, word_groups, excludes=None, **kwargs):
    """Create a model debiased with a generalization of Bolukbasi's method.

    Parameters:
        corpus_file (Path): The text corpus.
        word_pairs (Iterable[Tuple[str, str]]):
            A list of gender words to define the subpsace.
        excludes (Iterable[Tuple[str, str]]):
            A collection of words to be excluded from the debiasing
        **kwargs: Other keyword arguments.

    Returns:
        WordEmbedding: The model built from the corpus.
    """
    baseline_model = create_baseline_model(corpus_file)
    return bolukbasi_debias_generalized(baseline_model, gender_pairs, mirrors=mirrors, **kwargs)


def create_original_bolukbasi_model(corpus_file, gender_pairs, mirrors=None, excludes=None, **kwargs):
    # type: (Path, Iterable[Tuple[str, str]], Iterable[Tuple[str, str]], **Any) -> WordEmbedding
    """Create a model debiased with Bolukbasi's original method.

    Parameters:
        corpus_file (Path): The text corpus.
        word_pairs (Iterable[Tuple[str, str]]):
            A list of gender words to define the subpsace.
        mirrors (Iterable[Tuple[str, str]]):
            A list of words to equalize.
        excludes (Iterable[Tuple[str, str]]):
            A collection of words to be excluded from the debiasing
        **kwargs: Other keyword arguments.

    Returns:
        WordEmbedding: The model built from the corpus.
    """
    baseline_model = create_baseline_model(corpus_file)
    return bolukbasi_debias_original(baseline_model, gender_pairs, mirrors=mirrors, **kwargs)


def create_swapped_model(corpus_file, word_pairs, **kwargs):
    # type: (Path, Iterable[Tuple[str, str]], **Any) -> WordEmbedding
    """Create a model debiased by word swapping.

    Parameters:
        corpus_file (Path): The text corpus.
        word_pairs (Iterable[Tuple[str, str]]): A list of words to swap.
        **kwargs: Other keyword arguments.

    Returns:
        WordEmbedding: The model built from the corpus.
    """
    swapped_corpus = create_swapped_corpus(corpus_file, word_pairs)
    return create_fasttext_model(swapped_corpus, **kwargs)
