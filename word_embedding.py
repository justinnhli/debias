from os import devnull
from contextlib import redirect_stderr

from gensim.models.fasttext import load_facebook_vectors
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors, Word2VecKeyedVectors
from gensim.models.base_any2vec import BaseWordEmbeddingsModel

from linalg import normalize

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
        with redirect_stderr(open(devnull)):
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
        with redirect_stderr(open(devnull)):
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
