# from pymagnitude import Magnitude
import json
from config import NUMBERBATCH_VOC_PATH, NUMBERBATCH

_model = {}


def get_model():
    if NUMBERBATCH not in _model:
        with open(NUMBERBATCH_VOC_PATH, "r") as file:
            _model[NUMBERBATCH] = json.load(file)

    return _model[NUMBERBATCH]


def del_model():
    if NUMBERBATCH in _model:
        del _model[NUMBERBATCH]
