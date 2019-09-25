from pathlib import Path

from experiments import read_word_list, read_gender_pairs
from experiments import load_word2vec_embedding, debias_bolukbasi_original, create_randomized_swapped_corpus
from experiments import measure_projection_bias, measure_analogy_bias


def validate_approach():
    # settings
    corpus_path = Path('corpora/wikipedia-1')
    # load in the relevant data
    gender_pairs_file = Path('data/gender-pairs/definitional')
    gender_pairs = read_gender_pairs(gender_pairs_file)
    gendered_words = read_word_list(Path('data/gendered-words/gender_specific_seed'))
    equalize_pairs = []
    # create the models
    print('loading baseline')
    baseline_model = load_word2vec_embedding(
        corpus_path,
        out_path=Path('models/paper-validation-baseline.w2v')
    )
    print('loading bolukbasi')
    bolukbasi_model = debias_bolukbasi_original(
        baseline_model,
        gender_pairs,
        gendered_words,
        equalize_pairs,
        out_path=Path('models/paper-validation-bolukbasi.w2v'),
    )
    print('loading swapped')
    swapped_model = load_word2vec_embedding(
        create_randomized_swapped_corpus(
            corpus_path,
            gender_pairs,
            out_path=Path('corpora/wikipedia-1-paper-swapped'),
        ),
        out_path=Path('models/paper-validation-swapped.w2v')
    )
    print('loading swapped')
    # prepare evaluation data
    adjectives_file = Path('data/biased-words/adjectives')
    occupations_file = Path('data/biased-words/occupations')
    print(' '.join(['METRIC', 'DATASET', 'BASELINE', 'SWAPPED', 'BOLUKBASI']))
    metric = 'projection'
    for biased_words_file in [adjectives_file, occupations_file]:
        line = [metric, biased_words_file.name]
        for model in [baseline_model, swapped_model, bolukbasi_model]:
            bias = measure_projection_bias(model, 'pca', gender_pairs_file, biased_words_file, 1)
            line.append(str(bias))
        print(' '.join(line))
    metric = 'analogy'
    for biased_words_file in [adjectives_file, occupations_file]:
        line = [metric, biased_words_file.name]
        for model in [baseline_model, swapped_model, bolukbasi_model]:
            bias = measure_analogy_bias(model, gender_pairs_file, biased_words_file)
            line.append(str(bias))
        print(' '.join(line))


def main():
    validate_approach()


if __name__ == '__main__':
    main()
