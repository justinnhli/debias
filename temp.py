from pathlib import Path

from main import create_baseline_model, create_bolukbasi_model, create_swapped_model
from util import read_word_groups, read_word_list

corpus_file = Path('corpora/wikipedia-1')
# load in the relevant data
gender_pairs_file = Path('data/gender-pairs/definitional')
gender_pairs = read_word_groups(gender_pairs_file)
gendered_words = read_word_list(Path('data/gendered-words/gender_specific_seed'))
equalize_pairs = read_word_groups(Path('data/gender-pairs/equalize'))


create_bolukbasi_model(
    corpus_file,
    gender_pairs,
    out_file=Path('models/paper-bolukbasi-original.w2v'),
    excludes=gendered_words,
    mirrors=equalize_pairs,
)
