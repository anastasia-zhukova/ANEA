import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data")
NUMBERBATCH = "numberbatch"
ELMO_WE = "elmo"
FASTTEXT_WE = "fasttext"
NUMBERBATCH_PATH = 'D:\\numberbatch-word_vectors\\numberbatch.magnitude'
MAX_COMPOUND_CANDS = 4
MAX_REC_LEVEL = 2
WIKTIONARY_PAGE = "https://de.wiktionary.org/wiki/{0}"
DOC_LIM = 400 # or None
CLUSTER_AREAS_MIN = 6
CLUSTER_AREAS_MAX = 13
TOP_FREQ_AREAS = 5
ITER_GROW_GRAPH = 1
MEANING = "Bedeutungen"
SYNOMYM = "Synonyme"
HYPERNYM = "Oberbegriffe"
HYPONYM = "Unterbegriffe"
EXAMPLE = "Beispiele"
WORDFORM = "Wortbildungen"
NAME = "Name"
POS = "POS"
LANGUAGE = "de"