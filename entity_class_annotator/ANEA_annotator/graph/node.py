import re

from config import *


DEF = "definitions"
SYN = "synonyms"
HYPER = "hypernyms"
HYPON = "hyponyms"
EX = "examples"
WF = "wordforms"
N = "name"
P = "pos"


TAG_MATCH = {MEANING: DEF,
             SYNOMYM: SYN,
             HYPERNYM: HYPER,
             HYPONYM: HYPON,
             EXAMPLE: EX,
             WORDFORM: WF,
             NAME: N,
             POS: P}


class Node:

    def __init__(self, word, leaf, url, content_dict=None, term_id=None):
        for tag in list(TAG_MATCH.values()):
            setattr(self, tag, None)

        self.word = word
        self.term_id = term_id
        self.is_leaf = leaf
        self.url = url
        self.lang = re.split(r'https://', url)[-1][:2] if self.url is not None else LANGUAGE

        if content_dict is not None:
            self._add_wikt_properties(content_dict)
        a = 1

    def __repr__(self):
        return self.word + "_leaf" if self.is_leaf else self.word

    def _add_wikt_properties(self, content_dict):
        # add attributes to the object
        for tag, attr in TAG_MATCH.items():
            if tag not in content_dict:
                continue
            if getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr) + content_dict[tag])
            else:
                setattr(self, attr, content_dict[tag])
