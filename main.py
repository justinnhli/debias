from corpus import create_swapped_corpus
from training import create_fasttext_model
from debias import bolukbasi_debias_original


def create_baseline_model(corpus_file, **kwargs):
    return create_fasttext_model(corpus_file, **kwargs)

def create_bolukbasi_model(corpus_file, gender_pairs, equalize_pairs, **kwargs):
    baseline_model = create_baseline_model(corpus_file)
    return bolukbasi_debias_original(baseline_model, gender_pairs, equalize_pairs, **kwargs)

def create_swapped_model(corpus_file, word_pairs, **kwargs):
    swapped_corpus = create_swapped_corpus(corpus_file, word_pairs)
    return create_fasttext_model(swapped_corpus, **kwargs)
