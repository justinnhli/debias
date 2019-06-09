from pathlib import Path

CODE_PATH = Path(__file__).resolve().parent
CORPORA_PATH = CODE_PATH.joinpath('corpora')
MODELS_PATH = CODE_PATH.joinpath('models')
DATA_PATH = CODE_PATH.joinpath('data')
RESULTS_PATH = CODE_PATH.joinpath('results')

for path in [CORPORA_PATH, MODELS_PATH, DATA_PATH, RESULTS_PATH]:
    path.mkdir(exist_ok=True)
