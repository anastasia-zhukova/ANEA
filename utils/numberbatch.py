from pymagnitude import Magnitude
from config import NUMBERBATCH_PATH, NUMBERBATCH

_model = {}


def get_model():
    if NUMBERBATCH not in _model:
        _model[NUMBERBATCH] = Magnitude(NUMBERBATCH_PATH)
    return _model[NUMBERBATCH]


def del_model():
    if NUMBERBATCH in _model:
        del _model[NUMBERBATCH]
