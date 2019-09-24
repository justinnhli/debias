import os
import re
import subprocess
from contextlib import redirect_stderr
from functools import lru_cache
from itertools import product
from pathlib import Path
from random import Random
from statistics import mean

import numpy as np
from clusterun import sequencerun
from gensim.models import Word2Vec
from gensim.models.fasttext import load_facebook_vectors
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors, Word2VecKeyedVectors
from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from gensim.scripts.glove2word2vec import glove2word2vec
from permspace import PermutationSpace
from sklearn.decomposition import PCA

from constants import CORPORA_PATH, MODELS_PATH, DATA_PATH, RESULTS_PATH

# IO utilities


def read_file(path):
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


def read_gender_pairs(path):
    """Read a gender pairs file.

    Parameters:
        path (Path): The path of the gender pairs file.

    Returns:
        List[Tuple[str, str]]: A list of (male, female) words.
    """
    pairs = []
    for line in read_file(path):
        pairs.append(tuple(line.split()))
    return pairs


def read_word_groups(path):
    """Read in a word group file and augment with common contractions.

    Parameters:
        path (Path): The path of the word groups file.

    Returns:
        List[List[str]]: A list of word groups.
    """
    contractions = ["'ll", "'s", "'d", "'ve"]
    root_swaps = [tuple(line.split()) for line in read_file(path)]
    return root_swaps + [
        (words[0] + contraction, words[1] + contraction)
        for words, contraction
        in product(root_swaps, contractions)
    ]


# vector utilities


def normalize(vectors):
    """Normalize vectors.

    Parameters:
        vectors (numpy.ndarray): The vectors, as rows.

    Returns:
        numpy.ndarray: The normalized vec.
    """
    flat = (len(vectors.shape) == 1)
    if flat:
        vectors = vectors[np.newaxis, :]
    result = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    if flat:
        return result[0]
    else:
        return result


def project(vectors, bases, change_coords=False):
    """Project the vectors on to the subspace formed by the bases.

    Parameters:
        vectors (numpy.ndarray): The vectors to project, as rows.
        bases (numpy.ndarray): The bases to project on to.
        change_coords (bool): If True, the result will be in the coordinate
            system defined by the bases. Defaults to False.

    Returns:
        numpy.ndarray: The projection.
    """
    flat = (len(vectors.shape) == 1)
    if flat:
        vectors = vectors[np.newaxis, :]
    if len(bases.shape) == 1:
        bases = bases[np.newaxis, :]
    try:
        result = np.linalg.inv(bases @ bases.T) @ bases @ vectors.T
    except ValueError:
        breakpoint()
    if not change_coords:
        result = bases.T @ result
    result = result.T
    if flat:
        return result[0]
    else:
        return result



def reject(vectors, bases):
    """Reject the vectors from the subspace formed by the bases.

    Parameters:
        vectors (numpy.ndarray): The vector to reject.
        bases (numpy.ndarray): The vector to reject from.

    Returns:
        numpy.ndarray: The rejection.
    """
    return vectors - project(vectors, bases)


def recenter(vectors):
    """Redefine vectors as coming from their centroid.

    Parameters:
        vectors (numpy.ndarray): The vectors, as rows

    Returns:
        numpy.ndarray: The new vectors.
    """
    centroid = np.mean(vectors, axis=0)
    extrusion = np.repeat(centroid[np.newaxis, :], [vectors.shape[0]], axis=0)
    return vectors - extrusion


# classes


class WordEmbedding:
    """A wrapper around gensim.models.keyedvectors.

    The main goal of this wrapper/adaptor class is to provide more Pythonic,
    dictionary-like access. This class was written by examining the gensim
    source code is and therefore prone to breakage.

    https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/keyedvectors.py
    """

    def __init__(self, dimensions=None, gensim_obj=None, source=None):
        """Initialize a word embedding.

        At least one of dimensions and gensim_obj must be provided. If both are
        used, dimensions is ignored.

        Parameters:
            dimensions (int): The number of dimensions of the embedding.
            gensim_obj (gensim.Word2VecKeyedVectors):
                A gensim word embedding or related model.
            source (Path): The path of the source file.

        Raises:
            ValueError:
                If neither dimensions nor gensim_obj is provided.
                If dimensions is not a positive integer.
                If the word vectors in the gensim_obj cannot be determined.
        """
        if dimensions is None and gensim_obj is None:
            raise ValueError('one of dimensions or gensim_obj must be provided')
        if gensim_obj is None:
            if not isinstance(dimensions, int) and dimensions > 0:
                raise ValueError('dimensions must be a positive integer')
            self.keyed_vectors = Word2VecKeyedVectors(dimensions)
        elif isinstance(gensim_obj, WordEmbeddingsKeyedVectors):
            if not hasattr(gensim_obj, 'save_word2vec_format'):
                raise ValueError(f'gensim_obj {type(gensim_obj)} does not have attribute "save_word2vec_format"')
            self.keyed_vectors = gensim_obj
        elif isinstance(gensim_obj, BaseWordEmbeddingsModel):
            if not hasattr(gensim_obj, 'wv'):
                raise ValueError(f'gensim_obj {type(gensim_obj)} does not have attribute "wv"')
            self.keyed_vectors = gensim_obj.wv
        else:
            raise ValueError(f'unable to determine word vectors in gensim object {gensim_obj}')
        self.source = source
        # forcefully normalize the vectors
        self.keyed_vectors.vectors = normalize(self.keyed_vectors.vectors)

    @property
    def dimensions(self):
        """Get the dimensions of the word embedding.

        Returns:
            int: The dimensions of the word embedding.
        """
        return self.keyed_vectors.vector_size

    @property
    def words(self):
        """Get the words in the word embedding.

        Returns:
            List[str]: The words in the word embedding.
        """
        return self.keyed_vectors.index2entity

    @property
    def vectors(self):
        """Get the vectors in the word embedding.

        Returns:
            numpy.ndarray: The vectors in the word embedding.
        """
        return self.keyed_vectors.vectors

    def __len__(self):
        return len(self.keyed_vectors.vocab)

    def __contains__(self, word):
        return word in self.keyed_vectors.vocab

    def __setitem__(self, word, vector):
        if not isinstance(word, str):
            raise ValueError(f'word must be a str but got {str}')
        self.keyed_vectors[word] = vector

    def __getitem__(self, word):
        if not isinstance(word, str):
            raise ValueError(f'word must be a str but got {str}')
        if word not in self:
            raise KeyError(f'word "{word}" is not in the embedding')
        return self.keyed_vectors[word]

    def index(self, word):
        """Get the index of the word in the internal representation

        Returns:
            int: The index of the word in the Word2VecKeyedVectors.
        """
        return self.keyed_vectors.vocab[word].index

    def items(self):
        """Get the words and vectors in the word embedding.

        Yields:
            Tuple[str, numpy.ndarray]: The words and vectors in the word
                embedding.
        """
        for word in self.words:
            yield word, self[word]

    def save(self, path=None):
        """Save the word embedding in word2vec format.

        Parameters:
            path (Path): The path to save to. Defaults to the word embedding source.

        Raises:
            ValueError: If neither the path nor the source is set.
        """
        if path is None:
            path = self.source
        if path is None:
            raise ValueError('neither path nor WordEmbedding.source is set')
        self.keyed_vectors.save_word2vec_format(str(path))

    def words_near_vector(self, vector, k=10):
        """Find words near a given vector.

        Parameters:
            vector (numpy.ndarray): The word vector.
            k (int): The number of words to return. Defaults to 10.

        Returns:
            List[Tuple[str, float]]: A list of word and their cosine similarity.
        """
        return self.keyed_vectors.similar_by_vector(vector, topn=k)

    def words_near_word(self, word, k=10):
        """Find words near a given word.

        Parameters:
            word (str): The word.
            k (int): The number of words to return. Defaults to 10.

        Returns:
            List[Tuple[str, float]]: A list of word and their cosine similarity.
        """
        return self.words_near_vector(self[word], k=k)

    def distance(self, word1, word2):
        """Get the distance between two words.

        Parameters:
            word1 (str): The first word.
            word2 (str): The second word.

        Returns:
            float: The distance between the words.
        """
        return self.keyed_vectors.distance(word1, word2)

    @staticmethod
    def load_fasttext_file(path):
        """Load from a FastText .bin file.

        Parameters:
            path (Path): The path to the binary file.

        Returns:
            WordEmbedding: The resulting word embedding.
        """
        with redirect_stderr(open(os.devnull)):
            gensim_obj = load_facebook_vectors(path)
        return WordEmbedding(
            gensim_obj=gensim_obj,
            source=path,
        )

    @staticmethod
    def load_word2vec_file(path):
        """Load from a word2vec file.

        Parameters:
            path (Path): The path to the word2vec file.

        Returns:
            WordEmbedding: The resulting word embedding.
        """
        with redirect_stderr(open(os.devnull)):
            gensim_obj = Word2VecKeyedVectors.load_word2vec_format(str(path))
        return WordEmbedding(
            gensim_obj=gensim_obj,
            source=path,
        )

    @staticmethod
    def from_vectors(words, vectors):
        """Create a WordEmbedding from words and their vectors.

        Parameters:
            words (List[str]): The words in the embedding.
            vectors (numpy.ndarray): The vectors, in the order of words.

        Returns:
            WordEmbedding: The resulting WordEmbedding object.

        Raises:
            ValueError: If the length of words and vectors don't match.
        """
        embedding = WordEmbedding(dimensions=vectors.shape[1])
        embedding.keyed_vectors.add(words, vectors)
        return embedding


# corpus transforms


def swap_words(word_pairs, *strings):
    """Swap words in the strings.

    Parameters:
        word_pairs (Iterable[Tuple[str, str]]): A collection of word pairs to swap.
        *strings (str): The strings to swap in.

    Yields:
        str: The strings with the words swapped.
    """
    prefix = '__PREFIX__'
    for string in strings:
        for word1, word2 in word_pairs:
            # "forwards"
            result = re.sub(
                r'\b' + word1 + r'\b',
                prefix + word2,
                string,
                flags=re.IGNORECASE,
            )
            # "backwards"
            result = re.sub(
                r'\b' + word2 + r'\b',
                prefix + word1,
                result,
                flags=re.IGNORECASE,
            )
        yield result.replace(prefix, '')


def randomly_swap_words(word_groups, *strings, random_seed=8675309):
    """Swap words in strings with another randomly chosen word.

    Parameters:
        word_groups (Iterable[Sequence[str, str]]): A collection of word groups to swap.
        *strings (str): The strings to swap in.
        random_seed (any): The random seed for the swapping. Optional.

    Yields:
        str: The strings with the words swapped.
    """
    # FIXME deal with duplicates, eg. [['him', 'her'], ['his', 'her']]
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


def replace_swap_words(word_groups, *strings):
    """Swap words in strings by replacing them with the first in a group.

    Parameters:
        word_groups (Iterable[Sequence[str, str]]): A collection of word groups to swap.
        *strings (str): The strings to swap in.

    Yields:
        str: The strings with the words replaced.
    """
    # FIXME deal with duplicates, eg. [['him', 'her'], ['his', 'her']]
    candidates = {}
    for group in word_groups:
        group = sorted(group)
        for word in group:
            candidates[word] = group
    for string in strings:
        result = []
        for word in string.split():
            if word in candidates:
                result.append(candidates[word][0])
            else:
                result.append(word)
        yield ' '.join(result)


def create_duplicated_swapped_corpus(corpus, word_pairs, out_path=None):
    """Create a double-length word-swapped corpus.

    Parameters:
        corpus (Path): The path of the input corpus file.
        word_pairs (Iterable[Tuple[str, str]]): A collection of word pairs to swap.
        out_path (Path): The path of the resulting corpus file. Optional.

    Returns:
        Path: The path of the resulting corpus file.
    """
    if out_path is None:
        out_path = corpus.parent.joinpath(corpus.name + '.duplicate')
    if out_path.exists():
        return out_path
    with out_path.open('w') as out_fd:
        with corpus.open() as in_fd:
            for line in in_fd.readlines():
                out_fd.write(line.strip())
                out_fd.write('\n')
        with corpus.open() as in_fd:
            lines = (line.strip() for line in in_fd.readlines())
            for line in swap_words(word_pairs, *lines):
                out_fd.write(line)
                out_fd.write('\n')
    return out_path


def create_randomized_swapped_corpus(corpus, word_groups, out_path=None):
    """Create a randomized, word-swapped corpus.

    Parameters:
        corpus (Path): The path of the input corpus file.
        word_groups (Iterable[Iterable[str, str]]): A collection of word groups to swap.
        out_path (Path): The path of the resulting corpus file. Optional.

    Returns:
        Path: The path of the resulting corpus file.
    """
    if out_path is None:
        out_path = corpus.parent.joinpath(corpus.name + '.random')
    if out_path.exists():
        return out_path
    with corpus.open() as in_fd:
        lines = (line.strip() for line in in_fd.readlines())
        with out_path.open('w') as out_fd:
            for line in randomly_swap_words(word_groups, *lines):
                out_fd.write(line)
                out_fd.write('\n')
    return out_path


def create_replaced_swapped_corpus(corpus, word_groups, out_path=None):
    """Create a randomized, word-swapped corpus.

    Parameters:
        corpus (Path): The path of the input corpus file.
        word_groups (Iterable[Iterable[str, str]]): A collection of word groups to swap.
        out_path (Path): The path of the resulting corpus file. Optional.

    Returns:
        Path: The path of the resulting corpus file.
    """
    if out_path is None:
        out_path = corpus.parent.joinpath(corpus.name + '.replace')
    if out_path.exists():
        return out_path
    with corpus.open() as in_fd:
        lines = (line.strip() for line in in_fd.readlines())
        with out_path.open('w') as out_fd:
            for line in replace_swap_words(word_groups, *lines):
                out_fd.write(line)
                out_fd.write('\n')
    return out_path


def debias_corpus(corpus, params):
    """Debias the corpus.

    Parameters:
        corpus (Path): The path of the corpus file.
        params (NameSpace): The experiment parameters.

    Returns:
        Path: The path of the resulting corpus file.

    Raises:
        ValueError: If the params contain an unknown corpus_transform.
    """
    if params.corpus_transform == 'none':
        return corpus
    words_file = Path(params.swap_words_file)
    word_groups = read_word_groups(words_file)
    out_path = corpus.parent.joinpath(corpus.name + f'.{params.corpus_transform}.{words_file.name}')
    if params.corpus_transform == 'duplicate':
        return create_duplicated_swapped_corpus(corpus, word_groups, out_path)
    elif params.corpus_transform == 'random':
        return create_randomized_swapped_corpus(corpus, word_groups, out_path)
    elif params.corpus_transform == 'replace':
        return create_replaced_swapped_corpus(corpus, word_groups, out_path)
    else:
        raise ValueError(f'unknown corpus transform {params.corpus_transform}')


# word embeddings

@lru_cache(maxsize=16)
def load_word2vec_embedding(corpus, out_path=None):
    """Load or create a word2vec word embedding.

    Parameters:
        corpus (Path): The path of the corpus file.
        out_path (Path): The output path of the model. Optional.

    Returns:
        WordEmbedding: The trained word2vec model.
    """
    if out_path is None:
        out_path = MODELS_PATH.joinpath(corpus.name + '.w2v')
    if out_path.exists():
        return WordEmbedding.load_word2vec_file(out_path)
    model = Word2Vec(corpus_file=str(corpus), size=100, window=5, min_count=1, workers=4)
    model.wv.save_word2vec_format(str(out_path))
    return WordEmbedding(gensim_obj=model, source=out_path)


@lru_cache(maxsize=16)
def load_glove_embedding(corpus, out_path=None):
    """Load or create a GloVe word embedding.

    Parameters:
        corpus (Path): The path of the corpus file.
        out_path (Path): The output path of the model. Optional.

    Returns:
        WordEmbedding: The trained GloVe model.

    Raises:
        ValueError: If GloVe cannot be found.
    """
    glove_path = Path(os.environ['HOME'], 'git', 'GloVe')
    vocab_file_path = MODELS_PATH.joinpath(corpus.name + '.glove.vocab')
    cooccur_file_path = MODELS_PATH.joinpath(corpus.name + '.glove.cooccur')
    shuffle_file_path = MODELS_PATH.joinpath(corpus.name + '.glove.shuffle')
    raw_model_file_path = MODELS_PATH.joinpath(corpus.name + '.glove.raw')
    binary_file_path = MODELS_PATH.joinpath(corpus.name + '.glove.raw.bin')
    text_file_path = MODELS_PATH.joinpath(corpus.name + '.glove.raw.txt')
    if out_path is None:
        model_file_path = MODELS_PATH.joinpath(corpus.name + '.glove')
    else:
        model_file_path = out_path
    if not glove_path.exists():
        raise ValueError(f'GloVe does not exist at {glove_path}')
    if not vocab_file_path.exists():
        with corpus.open('rb') as stdin:
            with vocab_file_path.open('wb') as stdout:
                subprocess.run(
                    [
                        str(glove_path.joinpath('build', 'vocab_count')),
                        '-min-count', '5',
                        '-verbose', '2',
                    ],
                    stdin=stdin,
                    stdout=stdout,
                )
    if not cooccur_file_path.exists():
        with corpus.open('rb') as stdin:
            with cooccur_file_path.open('wb') as stdout:
                subprocess.run(
                    [
                        str(glove_path.joinpath('build', 'cooccur')),
                        '-memory', '4.0',
                        '-vocab-file', str(vocab_file_path),
                        '-verbose', '2',
                        '-window-size', '15',
                    ],
                    stdin=stdin,
                    stdout=stdout,
                )
    if not shuffle_file_path.exists():
        with cooccur_file_path.open('rb') as stdin:
            with shuffle_file_path.open('wb') as stdout:
                subprocess.run(
                    [
                        str(glove_path.joinpath('build', 'shuffle')),
                        '-memory', '4.0',
                        '-verbose', '2',
                    ],
                    stdin=stdin,
                    stdout=stdout,
                )
    if not binary_file_path.exists():
        subprocess.run([
            str(glove_path.joinpath('build', 'glove')),
            '-save-file', str(raw_model_file_path),
            '-threads', '8',
            '-input-file', str(shuffle_file_path),
            '-x-max', '10',
            '-iter', '15',
            '-vector-size', '50',
            '-binary', '2',
            '-vocab-file', str(vocab_file_path),
            '-verbose', '2',
        ])
    if not model_file_path.exists():
        glove2word2vec(str(text_file_path), str(model_file_path))
    return WordEmbedding.load_word2vec_file(model_file_path)


@lru_cache(maxsize=16)
def load_fasttext_embedding(corpus, method, out_path=None):
    """Load or create a FastText word embedding.

    Parameters:
        corpus (Path): The path of the corpus file.
        method (str): The model type. Must be either 'cbow' or 'skipgram'.
        out_path (Path): The output path of the model. Optional.

    Returns:
        WordEmbedding: The trained FastText model.

    Raises:
        ValueError: If method is not 'cbow' or 'skipgram'.
    """
    if method not in {'cbow', 'skipgram'}:
        raise ValueError(f'model_type must be "cbow" or "skipgram" but got "{method}"')
    file_name = corpus.name + f'.fasttext.{method}'
    if out_path is None:
        out_path = MODELS_PATH.joinpath(file_name)
    elif out_path.suffix == '.bin':
        out_path = out_path.parent.joinpath(out_path.stem)
    binary_path = out_path.parent.joinpath(out_path.name + '.bin')
    if not binary_path.exists():
        subprocess.run([
            'fasttext', method,
            '-input', str(corpus),
            '-output', str(out_path),
        ])
    binary_path = out_path.parent.joinpath(out_path.name + '.bin')
    return WordEmbedding.load_fasttext_file(binary_path)


def embed(corpus, params):
    """Create a word embedding.

    Parameters:
        corpus (Path): The path of the corpus file.
        params (NameSpace): The experiment parameters.

    Returns:
        WordEmbedding: The resulting word embedding.

    Raises:
        ValueError: If the params contain an unknown embed_method.
    """
    if params.embed_method == 'word2vec':
        return load_word2vec_embedding(corpus)
    elif params.embed_method == 'glove':
        return load_glove_embedding(corpus)
    elif params.embed_method == 'fasttext':
        return load_fasttext_embedding(corpus, params.fasttext_method)
    else:
        raise ValueError(f'unknown embedding method {params.embed_method}')


# word embedding utilities


def define_mean_gender_direction(embedding, gender_pairs):
    """Calculate the gender direction using the Euclidean mean.

    Parameters:
        embedding (WordEmbedding): A word embedding.
        gender_pairs (Iterable[Tuple[str, str]]): A list of male-female word pairs.

    Returns:
        numpy.ndarray: A female->male vector.

    Raises:
        ValueError: If none of the gender pairs are in the embedding.
    """
    diff_vectors = []
    for male_word, female_word in gender_pairs:
        if male_word not in embedding or female_word not in embedding:
            continue
        diff_vectors.append(embedding[male_word] - embedding[female_word])
    if not diff_vectors:
        raise ValueError('embedding does not contain any gender pairs.')
    gender_direction = normalize(np.mean(np.array(diff_vectors), axis=0))
    return align_gender_direction(embedding, gender_direction, gender_pairs)


def define_pca_gender_direction(embedding, gender_pairs):
    """Calculate the gender direction using PCA.

    Parameters:
        embedding (WordEmbedding): A word embedding.
        gender_pairs (Iterable[Tuple[str, str]]): A list of male-female word pairs.

    Returns:
        numpy.ndarray: A female->male vector.

    Raises:
        ValueError: If none of the gender pairs are in the embedding.
    """
    matrix = []
    for male_word, female_word in gender_pairs:
        if male_word not in embedding or female_word not in embedding:
            continue
        matrix.extend(recenter(
            np.array([embedding[male_word], embedding[female_word]])
        ))
    if not matrix:
        raise ValueError('embedding does not contain any gender pairs.')
    matrix = np.array(matrix)
    pca = PCA(n_components=10)
    pca.fit(matrix)
    gender_direction = normalize(pca.components_[0])
    return align_gender_direction(embedding, gender_direction, gender_pairs)


def align_gender_direction(embedding, gender_direction, gender_pairs):
    """Make sure the direction is female->male, not vice versa.

    Parameters:

    Parameters:
        embedding (WordEmbedding): A word embedding.
        gender_direction (numpy.ndarray): A male->female or female->male vector.
        gender_pairs (Iterable[Tuple[str, str]]): A list of male-female word pairs.

    Returns:
        numpy.ndarray: A female->male vector.
    """
    total = 0
    for male_word, female_word in gender_pairs:
        if male_word not in embedding or female_word not in embedding:
            continue
        male_vector = embedding[male_word]
        female_vector = embedding[female_word]
        total += (male_vector - female_vector).dot(gender_direction)
    if total < 0:
        gender_direction = -gender_direction
    return gender_direction


def define_gender_direction(embedding, subspace_aggregation, gender_pairs_file):
    """Calculate the gender direction.

    Parameters:
        embedding (WordEmbedding): A word embedding.
        subspace_aggregation (str): Method of determining the gender subspace.
        gender_pairs_file (Path): Path to a list of gendered word pairs.

    Returns:
        numpy.ndarray: A female->male vector.

    Raises:
        ValueError: If the subspace_aggregation is unknown.
    """
    gender_pairs = read_gender_pairs(Path(gender_pairs_file))
    if subspace_aggregation == 'mean':
        return define_mean_gender_direction(embedding, gender_pairs)
    elif subspace_aggregation == 'pca':
        return define_pca_gender_direction(embedding, gender_pairs)
    else:
        raise ValueError(
            f'unknown gender subspace aggregation method {subspace_aggregation}'
        )


# embedding transforms


def _bolukbasi_debias(embedding, direction, exclusions=None):
    if exclusions is None:
        exclusions = set()
    # debias the entire space first
    new_vectors = reject(embedding.vectors, direction)
    # put the exclusions back in
    for word in exclusions:
        if word in embedding:
            new_vectors[embedding.index(word)] = embedding[word]
    return normalize(new_vectors)


def _bolukbasi_equalize(embedding, vectors, direction, word_pairs):
    for male_root_word, female_root_word in word_pairs:
        variants = [
            (male_root_word.lower(), female_root_word.lower()),
            (male_root_word.title(), female_root_word.title()),
            (male_root_word.upper(), female_root_word.upper()),
        ]
        for (male_word, female_word) in variants:
            if male_word not in embedding or female_word not in embedding:
                continue
            male_index = embedding.index(male_word)
            female_index = embedding.index(female_word)
            male_vector = vectors[male_index]
            female_vector = vectors[female_index]
            mean_vector = (male_vector + female_vector) / 2
            gender_component = project(mean_vector, direction)
            non_gender_component = mean_vector - gender_component
            if (male_vector - female_vector).dot(gender_component) < 0:
                gender_component = -gender_component
            vectors[male_index] = gender_component + non_gender_component
            vectors[female_index] = -gender_component + non_gender_component
    return normalize(vectors)


def debias_bolukbasi(embedding, subspace, out_path, exclusions=None):
    """Debiasing a word embedding by zeroing the vectors along a subspace.

    Usecase: allow debiasing along high-dimensional, non-gender directions.

    Parameters:
        embedding (WordEmbedding): The word embedding to debias.
        subspace (numpy.ndarray): The subspace to zero out.
        out_path (Path):
            The path to save the new embedding to.
        exclusions (Iterable[str]):
            Words that should not be zeroed out.

    Returns:
        WordEmbedding: The debiased word embedding.

    Raises:
        ValueError: If the subspace dimensions do not match that of the
            embedding.
    """
    if out_path.exists():
        return WordEmbedding.load_word2vec_file(out_path)
    if len(subspace) != embedding.vectors.shape[1]:
        raise ValueError(
            f'Subspace has {len(subspace)} dimensions '
            f'but vectors have {embedding.vectors.shape[1]}.'
        )
    new_vectors = _bolukbasi_debias(embedding, subspace, exclusions)
    new_vectors = normalize(new_vectors)
    new_embedding = WordEmbedding.from_vectors(embedding.words, new_vectors)
    new_embedding.source = out_path
    with redirect_stderr(open(os.devnull)):
        new_embedding.save()
    return new_embedding


def debias_bolukbasi_original(embedding, gender_pairs, gendered_words, equalize_pairs, out_path):
    """Debiasing a word embedding using Bolukbasi's original algorithm

    Adapted from https://github.com/tolga-b/debiaswe/blob/master/debiaswe/debias.py#L19
    Commit 10277b23e187ee4bd2b6872b507163ef4198686b on 2018-04-02

    Parameters:
        embedding (WordEmbedding): The word embedding to debias.
        gender_pairs (Iterable[Tuple[str, str]]):
            A list of male-female word pairs.
        gendered_words (Iterable[str]):
            A collection of words that should be gendered.
        equalize_pairs (Iterable[Tuple[str, str]]):
            Specific words that should be equidistant.
        out_path (Path):
            The path to save the new embedding to.

    Returns:
        WordEmbedding: The debiased word embedding.
    """
    if out_path.exists():
        return WordEmbedding.load_word2vec_file(out_path)
    gender_direction = define_pca_gender_direction(embedding, gender_pairs)
    gender_direction = gender_direction[np.newaxis, :]
    new_vectors = _bolukbasi_debias(embedding, gender_direction, gendered_words)
    new_vectors = _bolukbasi_equalize(embedding, new_vectors, gender_direction, equalize_pairs)
    new_vectors = normalize(new_vectors)
    new_embedding = WordEmbedding.from_vectors(embedding.words, new_vectors)
    new_embedding.source = out_path
    with redirect_stderr(open(os.devnull)):
        new_embedding.save()
    return new_embedding


def debias_embedding(embedding, params):
    """Debias a word embedding.

    Parameters:
        embedding (WordEmbedding): A word embedding.
        params (NameSpace): The experiment parameters.

    Returns:
        WordEmbedding: The resulting WordEmbedding.

    Raises:
        ValueError: If the params contain an unknown embedding_transform.
    """
    if params.embedding_transform == 'none':
        return embedding
    elif params.embedding_transform == 'bolukbasi':
        gender_pairs_path = Path(params.bolukbasi_subspace_words_file)
        out_path_parts = [
            embedding.source.name,
            gender_pairs_path.stem,
            params.bolukbasi_subspace_aggregation,
        ]
        gender_pairs = read_gender_pairs(gender_pairs_path)
        if params.bolukbasi_gendered_words_file == 'none':
            gendered_words = set()
            out_path_parts.append('none')
        else:
            gendered_words_path = Path(params.bolukbasi_gendered_words_file)
            gendered_words = read_word_list(gendered_words_path)
            out_path_parts.append(gendered_words_path.stem)
        if params.bolukbasi_equalize_pairs_file == 'none':
            equalize_pairs = []
            out_path_parts.append('none')
        else:
            equalize_pairs_path = Path(params.bolukbasi_equalize_pairs_file)
            equalize_pairs = read_gender_pairs(equalize_pairs_path)
            out_path_parts.append(equalize_pairs_path.stem)
        out_path = MODELS_PATH.joinpath('.'.join(out_path_parts))
        return debias_bolukbasi_original(embedding, gender_pairs, gendered_words, equalize_pairs, out_path)
    else:
        raise ValueError(f'unknown embedding transform {params.embedding_transform}')


# bias measurements

def measure_projection_bias(embedding, subspace_aggregation, gender_pairs_file, biased_words_file, bias_strictness):
    """Measure the bias by projecting onto the gender direction.

    Parameters:
        embedding (WordEmbedding): A word embedding.
        subspace_aggregation (str): Method of determining the gender subspace.
        gender_pairs_file (Path): Path to a list of gendered word pairs.
        biased_words_file (Path): Path to a list of potentially biased words.
        bias_strictness (float): Bias strictness parameter.

    Returns:
        float: The measured bias.
    """
    direction = define_gender_direction(
        embedding,
        subspace_aggregation,
        gender_pairs_file,
    )
    direction_vector = normalize(direction)
    biased_words = read_word_list(biased_words_file)
    word_biases = []
    for word in biased_words:
        if word not in embedding:
            continue
        word_biases.append(
            abs(np.dot(embedding[word], direction_vector))
            ** bias_strictness
        )
    return mean(word_biases)


def measure_analogy_bias(embedding, gender_pairs_file, biased_words_file):
    """Measure the bias by projecting onto the gender direction.

    Parameters:
        embedding (WordEmbedding): A word embedding.
        gender_pairs_file (Path): Path to a list of gendered word pairs.
        biased_words_file (Path): Path to a list of potentially biased words.

    Returns:
        float: The measured bias.
    """
    gender_pairs = read_gender_pairs(gender_pairs_file)
    biased_words = read_word_list(biased_words_file)
    biased_words = [word for word in biased_words if word in embedding]
    distances = []
    for male_word, female_word in gender_pairs:
        if male_word not in embedding or female_word not in embedding:
            continue
        for word in biased_words:
            diff_vector = embedding[male_word] - embedding[female_word]
            # this gives List[Tuple[str, float]]
            targets = embedding.words_near_vector(embedding[word] + diff_vector)
            target = max(targets, key=(lambda pair: pair[1]))[0]
            distances.append(embedding.distance(word, target))
    return mean(distances)


def measure_bias(embedding, params):
    if params.bias_metric == 'projection':
        return measure_projection_bias(
            embedding,
            params.subspace_aggregation,
            params.subspace_words_file,
            params.biased_words_file,
            params.bias_strictness,
        )
    elif params.bias_metric == 'analogy':
        return measure_analogy_bias(
            embedding,
            params.subspace_words_file,
            params.biased_words_file,
        )
    else:
        raise ValueError(f'unknown embedding transform {params.bias_metric}')


# main


def xor(a, b):
    # pylint: disable = invalid-name
    return (a and not b) or (b and not a)


def run_experiment(params):
    """Run a word embedding bias experiment.

    Parameters:
        params (NameSpace): The experiment parameters.
    """
    corpus = Path(params.corpus_file).expanduser().resolve()
    corpus = debias_corpus(corpus, params)
    embedding = embed(corpus, params)
    embedding = debias_embedding(embedding, params)
    bias = measure_bias(embedding, params)
    print(' '.join(str(part) for part in [
        Path(params.corpus_file).name,
        params.corpus_transform,
        Path(params.swap_words_file).name,
        params.embed_method,
        params.fasttext_method,
        params.embedding_transform,
        Path(params.bolukbasi_subspace_words_file).name,
        params.bolukbasi_subspace_aggregation,
        Path(params.bolukbasi_gendered_words_file).name,
        Path(params.bolukbasi_equalize_pairs_file).name,
        params.bias_metric,
        Path(params.subspace_words_file).name,
        params.subspace_aggregation,
        Path(params.biased_words_file).name,
        bias,
    ]))


CORPUS_FILES = [
    str(CORPORA_PATH.joinpath('wikipedia-1')),
    str(CORPORA_PATH.joinpath('alice-in-wonderland')),
]
GENDER_PAIRS_FILES = [
    str(DATA_PATH.joinpath('gender-pairs', 'pronouns')),
    str(DATA_PATH.joinpath('gender-pairs', 'definitional')),
]
GENDERED_WORDS_FILES = [
    str(DATA_PATH.joinpath('gendered-words', 'gender_specific_seed')),
]
EQUALIZE_PAIRS_FILES = [
    str(DATA_PATH.joinpath('gender-pairs', 'equalize')),
]
EVALUATION_PAIRS_FILES = [
    str(DATA_PATH.joinpath('gender-pairs', 'definitional')),
]
BIASED_WORDS_FILES = [
    str(DATA_PATH.joinpath('biased-words', 'adjectives')),
    str(DATA_PATH.joinpath('biased-words', 'occupations')),
]
PSPACE = PermutationSpace(
    [
        'corpus_file',
        'corpus_transform',
        'swap_words_file',
        'embedding_transform',
        'bolukbasi_subspace_words_file',
        'bolukbasi_subspace_aggregation',
        'bolukbasi_gendered_words_file',
        'bolukbasi_equalize_pairs_file',
        'embed_method',
        'fasttext_method',
        'bias_metric',
        'subspace_words_file',
        'subspace_aggregation',
        'biased_words_file',
    ],
    # corpus parameters
    corpus_file=CORPUS_FILES,
    # corpus transform parameters
    corpus_transform=['none', 'replace', 'duplicate', 'random'],
    swap_words_file=['none', *GENDER_PAIRS_FILES],
    # embedding parameters
    embed_method=['word2vec', 'glove', 'fasttext'],
    fasttext_method=['none', 'cbow', 'skipgram'],
    # embedding transform parameters
    embedding_transform=['none', 'bolukbasi'],
    bolukbasi_subspace_words_file=['none', *GENDER_PAIRS_FILES],
    bolukbasi_subspace_aggregation=['none', 'pca'],
    bolukbasi_gendered_words_file=['none', *GENDERED_WORDS_FILES],
    bolukbasi_equalize_pairs_file=['none', *EQUALIZE_PAIRS_FILES],
    # bias evaluation parameters
    bias_metric=['projection', 'analogy'],
    subspace_words_file=EVALUATION_PAIRS_FILES,
    subspace_aggregation=['mean', 'pca'],
    biased_words_file=BIASED_WORDS_FILES,
    bias_strictness=0.8,
).filter(
    lambda corpus_transform, embedding_transform:
        corpus_transform == 'none' or embedding_transform == 'none'
).filter(
    lambda corpus_transform, swap_words_file:
        xor(corpus_transform != 'none', swap_words_file == 'none')
).filter(
    lambda embed_method, fasttext_method:
        xor(embed_method == 'fasttext', fasttext_method == 'none')
).filter(
    lambda embedding_transform,
            bolukbasi_subspace_words_file, bolukbasi_subspace_aggregation,
            bolukbasi_gendered_words_file, bolukbasi_equalize_pairs_file:
        xor(embedding_transform == 'bolukbasi', bolukbasi_subspace_words_file == 'none')
        and xor(embedding_transform == 'bolukbasi', bolukbasi_subspace_aggregation == 'none')
        and xor(embedding_transform == 'bolukbasi', bolukbasi_gendered_words_file == 'none')
        and xor(embedding_transform == 'bolukbasi', bolukbasi_equalize_pairs_file == 'none')
).filter_if(
    (lambda embedding_transform: embedding_transform == 'bolukbasi'),
    (lambda bolukbasi_subspace_words_file, bolukbasi_subspace_aggregation, bolukbasi_gendered_words_file:
        not any(
            param == 'none' for param in [
                bolukbasi_subspace_words_file,
                bolukbasi_subspace_aggregation,
                bolukbasi_gendered_words_file,
            ]
        )
    ),
).filter(
    # FIXME fix multiple parameters for testing purposes
    lambda corpus_file, corpus_transform, embed_method, fasttext_method,
            bolukbasi_subspace_aggregation, subspace_aggregation:
        corpus_file == CORPUS_FILES[0]
        and corpus_transform == 'replace'
        and embed_method == 'fasttext'
        and fasttext_method == 'cbow'
        and bolukbasi_subspace_aggregation != 'pca'
        and subspace_aggregation == 'pca'
)


if __name__ == '__main__':
    sequencerun(run_experiment, 'PSPACE')
