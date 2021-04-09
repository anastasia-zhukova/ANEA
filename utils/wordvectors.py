from typing import List
import numpy as np
from gensim.models.fasttext import load_facebook_model
from config import FASTTEXT_WE, FASTTEXT_PATH


wordvectors = {
    FASTTEXT_WE: FASTTEXT_PATH
}


class WordEmbeddings:

    def __init__(self, we_name=FASTTEXT_WE):
        print("The word vector model is being loaded.")
        self.we_name = we_name
        self._model = load_facebook_model(wordvectors[we_name])
        self.vector_size = self._model.vector_size

    def query(self, tokens: List[str], coefs=None):
        if not len(tokens):
            return [np.zeros(self._model.vector_size)]

        vectors = self._model[tokens]
        if coefs is None:
            return [np.mean(vectors, axis=0)]
        return [np.mean(np.array(coefs).reshape(-1,1) * vectors, axis=0)]

    def get_vector(self, word: str):
        return self._model[word]


model = None


def get_model():
    global model
    if model is None:
        model = WordEmbeddings()
        return model
    else:
        return model
