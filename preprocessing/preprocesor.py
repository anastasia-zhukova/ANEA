import progressbar
import ftfy
from typing import List
import re
from config import DOC_LIM
from preprocessing.term import Term
from utils.parser import nlp
import string

# -------------
POS = ["NOUN", "PROPN"]
# -------------


class Preprocessor:

    def __init__(self):
        self.word_dict = {}

    def preprocess(self, orig_texts: List[str]):
        annot_text = self._apply_spacy(orig_texts)
        len_words = len([t for doc in annot_text for t in doc])
        print("Total number of words in a text collection: " + str(len_words))
        terms = self._filter_terms(annot_text)
        return terms, len_words

    def _apply_spacy(self, orig_texts: List[str]):
        preprocessed_docs = []
        widgets = [progressbar.FormatLabel(
            "PROGRESS: Processing %(value)d-th (%(percentage)d %%) doc/entry (in: %(elapsed)s).")]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=len(orig_texts)).start()
        for i, text in enumerate(orig_texts[:DOC_LIM] if DOC_LIM is not None else orig_texts):
            doc = nlp()(ftfy.fix_text(text))
            if doc.doc is not None:
                preprocessed_docs.append(doc)
            bar.update(i)
        bar.finish()
        return preprocessed_docs

    def _filter_terms(self, annot_text):
        term_dict = {}
        widgets = [progressbar.FormatLabel(
            "PROGRESS: Processing %(value)d-th/%(max_value)d (%(percentage)d %%) doc/entry (in: %(elapsed)s).")]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=len(annot_text)).start()

        for i, doc in enumerate(annot_text):
            for t in doc:
                if t.pos_ not in POS:
                    if t.pos_ != "VERB":
                        continue
                    else:
                        if t.orth_[-2:] != "er" and t.orth_[-3:] != "ung":
                            continue

                if len(t.orth_) < 4 or len(set(t.orth_)) < 2:
                    # ignore too short words and test cases like "aaaaa"
                    continue

                if len(re.findall(r'[a-zA-Z]+\d+', t.orth_)):
                    # ignore funcloc codes like A123
                    continue

                if len(re.findall(r'\d', t.orth_)):
                    # ignore if there is a digit
                    continue

                if len(re.findall(r'#|:', t.orth_)):
                    # ignore if there is "#" or ":"
                    continue

                word = t.orth_[1:].capitalize() if t.orth_[0] in string.punctuation else t.orth_
                word = word.split(";")[-1].capitalize() if len(word.split(";")) > 1 else word

                if word not in term_dict:
                    # term = Term(word=word, token=t)
                    term = Term(word=word, token=None)
                    term_dict[word.capitalize()] = term
                else:
                    term_dict[word.capitalize()].increase_counter()
            bar.update(i)
        bar.finish()
        return term_dict
