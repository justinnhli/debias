import numpy as np
from sklearn.decomposition import PCA

from linalg import recenter, normalize, reject
from word_embedding import WordEmbedding


def _define_mean_bias_subspace(embedding, word_pairs, **kwargs):
    # type: (WordEmbedding, Iterable[Tuple[str, str]], **Any) -> numpy.ndarray
    """Calculate the gender direction using the Euclidean mean.

    Parameters:
        embedding (WordEmbedding): A word embedding.
        word_pairs (Iterable[Tuple[str, str]]): A list of male-female word pairs.
        **kwargs: Other keyword arguments.

    Returns:
        numpy.ndarray: A female->male vector.

    Raises:
        ValueError: If none of the gender pairs are in the embedding.
    """
    diff_vectors = []
    for male_word, female_word in word_pairs:
        if male_word not in embedding or female_word not in embedding:
            continue
        diff_vectors.append(embedding[male_word] - embedding[female_word])
    if not diff_vectors:
        raise ValueError('embedding does not contain any gender pairs.')
    return normalize(np.mean(np.array(diff_vectors), axis=0))


def _define_pca_bias_subspace(embedding, words, subspace_dimensions, **kwargs):
    # type: (WordEmbedding, Iterable[Iterable[str]], int, **Any) -> numpy.ndarray
    """Calculate the gender direction using PCA.

    Parameters:
        embedding (WordEmbedding): A word embedding.
        words (Iterable[Iterable[str]]): A collection of definitional words.
        subspace_dimensions (int): The number of principle components to use.
        **kwargs: Other keyword arguments.

    Returns:
        numpy.ndarray: A basis of the bias subspace.

    Raises:
        ValueError: If none of the words are in the embedding.
    """
    matrix = recenter(np.array([
        embedding[word] for word in words
        if word in embedding
        ]))
    if not matrix:
        raise ValueError('embedding does not contain any of the definitional words.')
    matrix = np.array(matrix)
    pca = PCA(n_components=subspace_dimensions)
    pca.fit(matrix)
    return normalize(pca.components_) # FIXME likely incorrect


def _align_gender_direction(embedding, gender_direction, gender_pairs):
    # type: (WordEmbedding, numpy.ndarray, Iterable[Tuple[str, str]]) -> numpy.ndarray
    """Make sure the direction is female->male, not vice versa.

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

def define_bias_subspace(embedding, word_groups, subspace_method='pca', subspace_dimensions=1, **kwargs):
    # type: (WordEmbedding, Iterable[Iterable[str]], Optional[str], Optional[int], **Any) -> numpy.ndarray
    """Define a bias subspace.

    Parameters:
        embedding (WordEmbedding): A word embedding.
        word_groups (Iterable[Iterable[str]]): A collection of definitional words.
        subspace_method (Optional[str]): The method to crate the subspace.
            Must be 'pca' or 'mean'.
        subspace_dimensions (Optional[int]): The number of principle components to use.
        **kwargs: Other keyword arguments.

    Returns:
        numpy.ndarray: The bias subspace.

    Raises:
        ValueError: If subspace_method is invalid.
    """
    if subspace_method == 'mean':
        return _define_mean_bias_subspace(embedding, word_groups, **kwargs)
    elif subspace_method == 'pca':
        return _define_pca_bias_subspace(embedding, word_groups, subspace_dimensions=subspace_dimensions, **kwargs)
    else:
        raise ValueError(f'unknown bias subspace definition method {subspace_method}')


def bolukbasi_debias_original(embedding, word_pairs, out_file, excludes=None, mirrors=None, **kwargs):
    # type: (WordEmbedding, Iterable[Tuple[str, str]], Path, Iterable[str], Iterable[Tuple[str, str]], **Any) -> WordEmbedding
    """Debias a word embedding using Bolukbasi's original algorithm.

    Adapted from https://github.com/tolga-b/debiaswe/blob/master/debiaswe/debias.py#L19
    Commit 10277b23e187ee4bd2b6872b507163ef4198686b on 2018-04-02

    Parameters:
        embedding (WordEmbedding): The word embedding to debias.
        word_pairs (Iterable[Tuple[str, str]]):
            A list of word pairs that define the bias subspace.
        out_file (Path):
            The path to save the new embedding to.
        excludes (Iterable[str]):
            A collection of words to be excluded from the debiasing
        mirrors (Iterable[Tuple[str, str]]):
            Specific words that should be equidistant.
        **kwargs: Other keyword arguments.

    Returns:
        WordEmbedding: The debiased word embedding.
    """
    if out_file.exists():
        return WordEmbedding.load_word2vec_file(out_file)

    # define the bias subspace
    bias_subspace = define_bias_subspace(embedding, word_pairs, **kwargs)
    bias_subspace = _align_gender_direction(embedding, bias_subspace, word_pairs)
    bias_subspace = bias_subspace[np.newaxis, :]

    # debias by rejecting the subspace and reverting the excluded words
    if excludes is None:
        excludes = set()
    new_vectors = reject(embedding.vectors, bias_subspace)
    for word in excludes:
        if word in embedding:
            new_vectors[embedding.index(word)] = embedding[word]
    new_vectors = normalize(new_vectors)

    # FIXME does equalizing make sense in higher dimensions?
    #new_vectors = _bolukbasi_equalize(embedding, new_vectors, bias_subspace, mirrors)

    # create a word embedding from the new vectors
    new_embedding = WordEmbedding.from_vectors(embedding.words, new_vectors)
    new_embedding.source = out_file
    new_embedding.save()
    return new_embedding
