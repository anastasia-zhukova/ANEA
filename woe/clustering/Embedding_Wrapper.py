import fasttext
import spacy


class Embedding_Wrapper:
    def __init__(self, mode):
        self.mode = mode

        if mode == 'fasttext':
            self.embedder = fasttext.load_model("../fasttext/cc.de.300.bin")
        if mode == 'nlp':
            self.embedder = spacy.load('de_core_news_sm')

    def get_word_vector(self, word):
        if self.mode == 'fasttext':
            return self.embedder.get_word_vector(word)
        if self.mode == 'nlp':
            return self.embedder(word).tensor.squeeze()
