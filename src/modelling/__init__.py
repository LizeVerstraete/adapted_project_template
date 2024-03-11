from pathlib import Path

# Defined here to avoid partially initialized module import errors.
RESULTS_DIR = Path(__file__).absolute().parents[2] / 'results'
IMAGE_OUT_DIR = RESULTS_DIR / 'images'
AV_TENSOR_OUT_DIR = RESULTS_DIR / 'av_tensors'
NAMELESS_MODEL = ""

from .cyclegan import CycleGan

MODEL_DIR = Path(__file__).absolute().parents[2] / 'models'
NL_MODELS = []
L_MODELS = [CycleGan.__name__]
MODELS = NL_MODELS + L_MODELS

MODEL_INS = {"cyclegan": CycleGan}

HE2MT = {"cyclegan": str(MODEL_DIR / 'cyclegan_he2mt.pth')}
MT2HE = {k: v.replace('_he2mt', '_mt2he') if v else v for k, v in HE2MT.items()}
MODEL_WEIGHTS = {'a2b': HE2MT, 'b2a': MT2HE}
