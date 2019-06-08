import os
import re
import subprocess
from contextlib import redirect_stderr
from functools import lru_cache
from itertools import chain, product
from pathlib import Path
from random import Random
from statistics import mean

import numpy as np
from sklearn.decomposition import PCA
from gensim.models.fasttext import load_facebook_vectors
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors, Word2VecKeyedVectors
from permspace import PermutationSpace

from clusterun import sequencerun

# constants

CODE_PATH = Path(__file__).resolve().parent
CORPORA_PATH = CODE_PATH.joinpath('corpora')
MODELS_PATH = CODE_PATH.joinpath('models')
DATA_PATH = CODE_PATH.joinpath('data')
RESULTS_PATH = CODE_PATH.joinpath('results')

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


def normalize(vec):
    """Normalize a vec.

    Parameters:
        vec (numpy.ndarray): The vec.


    Returns:
        numpy.ndarray: The normalized vec.
    """
    return vec / np.linalg.norm(vec, ord=1)


def project(vec1, vec2):
    """Project vec1 on to vec2.

    Parametesr:
        vec1 (numpy.ndarray): The vector to project.
        vec2 (numpy.ndarray): The vector to project on to.

    Returns:
        numpy.ndarray: The projection.
    """
    return vec2 * vec1.dot(vec2) / vec2.dot(vec2)


def reject(vec1, vec2):
    """Reject vec1 from vec2.

    Parametesr:
        vec1 (numpy.ndarray): The vector to reject.
        vec2 (numpy.ndarray): The vector to reject from.

    Returns:
        numpy.ndarray: The rejection.
    """
    return vec1 - project(vec1, vec2)


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
        else:
            raise ValueError(f'unable to determine word vectors in gensim object {gensim_obj}')
        self.source = source
        # forcefully normalize the vectors
        self.normalize()

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

    def normalize(self):
        """Normalize all vectors in the word embedding."""
        new_vectors = np.array(self.vectors, dtype='float32')
        new_vectors /= np.linalg.norm(new_vectors, axis=1)[:, np.newaxis]
        self.keyed_vectors.vectors = new_vectors

    def keys(self):
        """Get the words in the word embedding.

        Returns:
            List[str]: The words in the word embedding.
        """
        return self.keyed_vectors.index2entity

    def values(self):
        """Get the vectors in the word embedding.

        Returns:
            numpy.ndarray: The vectors in the word embedding.
        """
        return self.keyed_vectors.vectors

    def items(self):
        """Get the words and vectors in the word embedding.

        Yields:
            Tuple[str, numpy.ndarray]: The words and vectors in the word
                embedding.
        """
        for word in self.keys():
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


def create_duplicated_swapped_corpus(corpus, word_pairs):
    """Create a double-length word-swapped corpus.

    Parameters:
        corpus (Path): The path of the input corpus file.
        word_pairs (Iterable[Tuple[str, str]]): A collection of word pairs to swap.

    Returns:
        Path: The path of the resulting corpus file.
    """
    out_path = corpus.parent.joinpath(corpus.name + '.duplicate-swapped')
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


def create_randomized_swapped_corpus(corpus, word_groups):
    """Create a randomized, word-swapped corpus.

    Parameters:
        corpus (Path): The path of the input corpus file.
        word_groups (Iterable[Iterable[str, str]]): A collection of word groups to swap.

    Returns:
        Path: The path of the resulting corpus file.
    """
    out_path = corpus.parent.joinpath(corpus.name + '.random-swapped')
    if out_path.exists():
        return out_path
    with corpus.open() as in_fd:
        lines = (line.strip() for line in in_fd.readlines())
        with out_path.open('w') as out_fd:
            for line in randomly_swap_words(word_groups, *lines):
                out_fd.write(line)
                out_fd.write('\n')
    return out_path


def create_replaced_swapped_corpus(corpus, word_groups):
    """Create a randomized, word-swapped corpus.

    Parameters:
        corpus (Path): The path of the input corpus file.
        word_groups (Iterable[Iterable[str, str]]): A collection of word groups to swap.

    Returns:
        Path: The path of the resulting corpus file.
    """
    out_path = corpus.parent.joinpath(corpus.name + '.replace-swapped')
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
    word_groups = read_word_groups(Path(params.swap_words_file).expanduser().resolve())
    if params.corpus_transform == 'duplicate':
        return create_duplicated_swapped_corpus(corpus, word_groups)
    elif params.corpus_transform == 'random':
        return create_randomized_swapped_corpus(corpus, word_groups)
    elif params.corpus_transform == 'replace':
        return create_replaced_swapped_corpus(corpus, word_groups)
    else:
        raise ValueError(f'unknown corpus transform {params.corpus_transform}')


# word embeddings


@lru_cache(maxsize=16)
def load_fasttext_embedding(corpus, method):
    """Load a FastText word embedding.

    Parameters:
        corpus (Path): The path of the corpus file.
        method (str): The model type. Must be either 'cbow' or 'skipgram'.

    Returns:
        WordEmbedding: The trained FastText model.

    Raises:
        ValueError: If method is not 'cbow' or 'skipgram'.
    """
    if method not in {'cbow', 'skipgram'}:
        raise ValueError(f'model_type must be "cbow" or "skipgram" but got "{method}"')
    file_name = corpus.name + f'.fasttext.{method}'
    binary_file_path = MODELS_PATH.joinpath(file_name + '.bin')
    if not binary_file_path.exists():
        subprocess.run([
            'fasttext', method,
            '-input', str(corpus),
            '-output', str(MODELS_PATH.joinpath(file_name)),
        ])
    return WordEmbedding.load_fasttext_file(binary_file_path)


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
    if params.embed_method == 'fasttext':
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
        diff_vector = embedding[male_word] - embedding[female_word]
        diff_vectors.append(normalize(diff_vector))
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
        fem_vec_norm = normalize(embedding[female_word])
        male_vec_norm = normalize(embedding[male_word])
        center = (fem_vec_norm + male_vec_norm) / 2
        matrix.append(fem_vec_norm - center)
        matrix.append(male_vec_norm - center)
    if not matrix:
        raise ValueError('embedding does not contain any gender pairs.')
    matrix = np.array(matrix)
    pca = PCA()
    pca.fit(matrix)
    gender_direction = normalize(pca.components_[0])
    return align_gender_direction(embedding, gender_direction, gender_pairs)


def align_gender_direction(embedding, gender_direction, gender_pairs):
    # if result is opposite the average female->male vector, flip it
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


def define_gender_direction(embedding, params):
    """Calculate the gender direction.

    Parameters:
        embedding (WordEmbedding): A word embedding.
        params (NameSpace): The experiment parameters.

    Returns:
        numpy.ndarray: A female->male vector.

    Raises:
        ValueError: If the params contain an unknown subspace_aggregation.
    """
    gender_pairs = read_gender_pairs(Path(params.subspace_words_file))
    if params.subspace_aggregation == 'mean':
        return define_mean_gender_direction(embedding, gender_pairs)
    elif params.subspace_aggregation == 'pca':
        return define_pca_gender_direction(embedding, gender_pairs)
    else:
        raise ValueError(
            f'unknown gender subspace aggregation method {params.subspace_aggregation}'
        )


# embedding transforms


def debias_bolukbasi(embedding, gender_pairs, gendered_words, equalize_pairs, out_path=None):
    """Debiasing a word embedding by zeroing the gender vector.

    Adapted from https://github.com/tolga-b/debiaswe/blob/master/debiaswe/debias.py#L19
    Commit 10277b23e187ee4bd2b6872b507163ef4198686b on 2018-04-02

    Parameters:
        embedding (WordEmbedding): The word embedding to debias.
        gender_pairs (Iterable[Tuple[str, str]]):
            A list of male-female word pairs.
        gendered_words (Set[str]):
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
    for male_word, female_word in gender_pairs:
        gendered_words.add(male_word)
        gendered_words.add(female_word)
        equalize_pairs.append((male_word, female_word))
    # expand the equalized pairs with their variants
    equalized_pair_variants = []
    for male_root_word, female_root_word in equalize_pairs:
        equalized_pair_variants.extend([
            (male_root_word.lower(), female_root_word.lower()),
            (male_root_word.title(), female_root_word.title()),
            (male_root_word.upper(), female_root_word.upper()),
        ])
    # pre-calculate the indices of the gendered and equalization words
    indices = {}
    equalize_words = set(chain(*equalized_pair_variants))
    for index, word in enumerate(embedding.keys()):
        if word in gendered_words or word in equalize_words:
            indices[word] = index
    # convert flat array to array in higher dimension
    gender_direction = define_pca_gender_direction(embedding, gender_pairs)
    gender_direction = gender_direction[np.newaxis, :]
    # debias the entire space first, then add back the origin gendered words
    vectors = embedding.values()
    scale = (vectors @ gender_direction.T) / (gender_direction @ gender_direction.T)
    extrusion = np.repeat(gender_direction, [vectors.shape[0]], axis=0)
    projection = scale * extrusion
    new_vectors = vectors - projection
    # put the gendered words back in
    for word in gendered_words:
        if word in indices:
            new_vectors[indices[word]] = embedding[word]
    # normalize the new vectors
    new_vectors = np.array(new_vectors, dtype='float32')
    new_vectors /= np.linalg.norm(new_vectors, axis=1)[:, np.newaxis]
    # equalize some words
    for (male_word, female_word) in equalized_pair_variants:
        if male_word not in indices or female_word not in indices:
            continue
        male_index = indices[male_word]
        female_index = indices[female_word]
        male_vector = new_vectors[male_index]
        female_vector = new_vectors[female_index]
        mean_vector = (male_vector + female_vector) / 2
        gender_component = project(mean_vector, gender_direction[0])
        non_gender_component = mean_vector - gender_component
        if (male_vector - female_vector).dot(gender_component) < 0:
            gender_component = -gender_component
        new_vectors[male_index] = gender_component + non_gender_component
        new_vectors[female_index] = -gender_component + non_gender_component
    # normalize the new vectors again
    new_vectors = np.array(new_vectors, dtype='float32')
    new_vectors /= np.linalg.norm(new_vectors, axis=1)[:, np.newaxis]
    # save the new embedding to disk
    new_embedding = WordEmbedding.from_vectors(embedding.keys(), new_vectors)
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
        return debias_bolukbasi(embedding, gender_pairs, gendered_words, equalize_pairs, out_path)
    else:
        raise ValueError(f'unknown embedding transform {params.embedding_transform}')


# bias measurements


def measure_bias(embedding, params):
    """Measure the bias of a word embedding.

    Parameters:
        embedding (WordEmbedding): A word embedding.
        params (NameSpace): The experiment parameters.

    Returns:
        float: The measured bias.
    """
    direction = define_gender_direction(embedding, params)
    biased_words = read_word_list(Path(params.biased_words_file))
    word_biases = []
    for word in biased_words:
        if word not in embedding:
            continue
        word_vector = embedding[word]
        direction_vector = normalize(direction)
        raw_bias = abs(np.dot(word_vector, direction_vector))
        word_biases.append(raw_bias**params.bias_strictness)
    return mean(word_biases)


# main


def run_experiment(params):
    """Run a word embedding bias experiment.

    Parameters:
        params (NameSpace): The experiment parameters.
    """
    print(params)
    corpus = Path(params.corpus_file).expanduser().resolve()
    corpus = debias_corpus(corpus, params)
    embedding = embed(corpus, params)
    embedding = debias_embedding(embedding, params)
    bias = measure_bias(embedding, params)
    print(bias)


CORPUS_FILES = [
    str(CORPORA_PATH.joinpath('wikipedia-1')),
    str(CORPORA_PATH.joinpath('alice-in-wonderland')),
]
GENDER_PAIRS_FILES = [
    str(DATA_PATH.joinpath('gender-pairs', 'all')),
]
GENDERED_WORDS_FILES = [
    str(DATA_PATH.joinpath('gendered-words', 'gender_specific_seed')),
]
EQUALIZE_PAIRS_FILES = [
    str(DATA_PATH.joinpath('gender-pairs', 'bolukbasi-equalize')),
]
EVALUATION_PAIRS_FILES = [
    str(DATA_PATH.joinpath('gender-pairs', 'bolukbasi-definitional')),
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
        'embed_method',
        'fasttext_method',
        'embedding_transform',
        'bolukbasi_subspace_words_file',
        'bolukbasi_subspace_aggregation',
        'bolukbasi_gendered_words_file',
        'bolukbasi_equalize_pairs_file',
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
    embed_method=['fasttext'],
    fasttext_method=['none', 'cbow', 'skipgram'],
    # embedding transform parameters
    embedding_transform=['none', 'bolukbasi'],
    bolukbasi_subspace_words_file=['none', *GENDER_PAIRS_FILES],
    bolukbasi_subspace_aggregation=['none', 'mean', 'pca'],
    bolukbasi_gendered_words_file=['none', *GENDERED_WORDS_FILES],
    bolukbasi_equalize_pairs_file=['none', *EQUALIZE_PAIRS_FILES],
    # bias evaluation parameters
    subspace_words_file=EVALUATION_PAIRS_FILES,
    subspace_aggregation=['mean', 'pca'],
    biased_words_file=BIASED_WORDS_FILES,
    bias_strictness=0.8,
).filter(
    lambda corpus_transform, embedding_transform:
        corpus_transform == 'none' or embedding_transform == 'none'
).filter_if(
    (lambda corpus_transform: corpus_transform == 'none'),
    (lambda swap_words_file: swap_words_file == 'none'),
).filter_if(
    (lambda corpus_transform: corpus_transform != 'none'),
    (lambda swap_words_file: swap_words_file != 'none'),
).filter_if(
    (lambda embed_method: embed_method != 'fasttext'),
    (lambda fasttext_method: fasttext_method == 'none'),
).filter_if(
    (lambda embedding_transform: embedding_transform != 'bolukbasi'),
    (lambda bolukbasi_subspace_words_file, bolukbasi_subspace_aggregation,
            bolukbasi_gendered_words_file, bolukbasi_equalize_pairs_file:
        all(
            param == 'none' for param in [
                bolukbasi_subspace_words_file,
                bolukbasi_subspace_aggregation,
                bolukbasi_gendered_words_file,
                bolukbasi_equalize_pairs_file,
            ]
        )
    ),
).filter_if(
    (lambda embedding_transform: embedding_transform == 'bolukbasi'),
    (lambda bolukbasi_subspace_words_file, bolukbasi_subspace_aggregation,
            bolukbasi_gendered_words_file, bolukbasi_equalize_pairs_file:
        not any(
            param == 'none' for param in [
                bolukbasi_subspace_words_file,
                bolukbasi_subspace_aggregation,
                bolukbasi_gendered_words_file,
                bolukbasi_equalize_pairs_file,
            ]
        )
    ),
).filter(
    # FIXME fix multiple parameters for testing purposes
    lambda corpus_file, corpus_transform, embed_method, fasttext_method,
            bolukbasi_subspace_aggregation, subspace_aggregation:
        corpus_file == corpus_files[0]
        and corpus_transform == 'replace'
        and embed_method == 'fasttext'
        and fasttext_method == 'cbow'
        and bolukbasi_subspace_aggregation != 'pca'
        and subspace_aggregation == 'pca'
)


if __name__ == '__main__':
    sequencerun(run_experiment, 'PSPACE')
