import subprocess
from typing import Any
from pathlib import Path

from constants import MODELS_PATH
from word_embedding import WordEmbedding


def create_fasttext_model(corpus_file, method='cbow', out_file=None, **kwargs):
    # type: (Path, str, Path, **Any) -> WordEmbedding
    """Load or create a FastText word embedding.

    Parameters:
        corpus_file (Path): The path of the corpus file.
        method (str): The model type. Must be either 'cbow' or 'skipgram'.
        out_file (Path): The output path of the model. Optional.
        **kwargs: Other keyword arguments.

    Returns:
        WordEmbedding: The trained FastText model.

    Raises:
        ValueError: If method is not 'cbow' or 'skipgram'.
    """
    if method not in {'cbow', 'skipgram'}:
        raise ValueError(f'method must be "cbow" or "skipgram" but got "{method}"')
    if out_file is None:
        out_file = MODELS_PATH.joinpath(corpus_file.name + f'.fasttext.{method}')
    if not out_file.exists():
        binary_file = out_file.parent.joinpath(out_file.name + '.bin')
        if not binary_file.exists():
            subprocess.run(
                [
                    'fasttext', method,
                    '-input', str(corpus_file),
                    '-output', str(out_file),
                ],
                check=True,
            )
        embedding = WordEmbedding.load_fasttext_file(binary_file)
        embedding.save(out_file)
    return WordEmbedding.load_word2vec_file(out_file)
