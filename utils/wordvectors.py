# import gdown
from typing import List, Set
import numpy as np
from gensim.models.fasttext import load_facebook_model


# from pymagnitude import *
from config import FASTTEXT_WE, ELMO_WE, FASTTEXT_PATH
# from sklearn.metrics.pairwise import cosine_similarity as cs


wordvectors = {
    FASTTEXT_WE: FASTTEXT_PATH,
    ELMO_WE: ""
}


class WordEmbeddings:

    def __init__(self, we_name=FASTTEXT_WE):
        print("The word vector model is being loaded.")
        self.we_name = we_name
        self._model = load_facebook_model(wordvectors[we_name])
        self.vector_size = self._model.vector_size

    def n_similarity(self, tokens_1: List[str], tokens_2: List[str]) -> float:
        if not len(tokens_1) or not len(tokens_2):
            return 0

        word_vectors_1 = self._model[tokens_1]
        word_vectors_2 = self._model[tokens_2]

        phrase_vector_1 = np.mean(word_vectors_1, axis=0)
        phrase_vector_2 = np.mean(word_vectors_2, axis=0)
        sim = cs([phrase_vector_1], [phrase_vector_2])
        return sim[0][0]

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
