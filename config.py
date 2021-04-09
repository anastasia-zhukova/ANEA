import os
import gdown
import logging

logger = logging.getLogger("model")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
f = logging.Formatter("%(asctime)s [%(process)d] [%(levelname)s] %(message)s", datefmt="[%Y-%m-%d %H:%M:%S %z]")
ch.setFormatter(f)
logger.addHandler(ch)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data")
SILVER_PATH = os.path.join(ROOT, "evaluation", "silver_dataset")
CSVS_PATH = os.path.join(ROOT, "evaluation", "csvs")
EXEC_RES_PATH = os.path.join(ROOT, "evaluation", "execution_results")
EXCEL_PATH = os.path.join(ROOT, "evaluation", "excels")
RESOURCE_PATH = os.path.join(ROOT, "resources")
EVAL_PATH = os.path.join(ROOT, "evaluation", "eval_results")

FASTTEXT_WE = "fasttext"
# can be adjusted if model is in another folder
FASTTEXT_FOLDER = os.path.join(RESOURCE_PATH, FASTTEXT_WE)
FASTTEXT_PATH = os.path.join(FASTTEXT_FOLDER, "cc.de.300.bin")

if "cc.de.300.bin" not in os.listdir(FASTTEXT_FOLDER):
    FASTTEXT_PATH = "C:\\Users\\annaz\\PycharmProjects\\DL_data_representation\\cc.de.300.bin\\cc.de.300.bin"
#     gdown.download("https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz", FASTTEXT_PATH, quiet=False)

NUMBERBATCH = "numberbatch"
NUMBERBATCH_VOC_PATH = os.path.join(RESOURCE_PATH, NUMBERBATCH, "numberbatch_voc.txt")


WIKTIONARY_PAGE = "https://de.wiktionary.org/wiki/{0}"
MAX_COMPOUND_CANDS = 4
MAX_REC_LEVEL = 2
DOC_LIM = None # number or None for all docs
TERM_GROUPS = 3 # a default number of fraction, e.g., 3 for 1/3, or None for all term groups
CLUSTER_AREAS_MIN = 6
CLUSTER_AREAS_MAX = 13
TOP_FREQ_AREAS = 5
ITER_GROW_GRAPH = 1
MIN_WORDS = 5

MEANING = "Bedeutungen"
SYNOMYM = "Synonyme"
HYPERNYM = "Oberbegriffe"
HYPONYM = "Unterbegriffe"
EXAMPLE = "Beispiele"
WORDFORM = "Wortbildungen"
NAME = "Name"
POS = "POS"
LANGUAGE = "de"


TOPIC = "topic"
FRAC = "fraction"
TERMS = "terms"
HEADS = "heads"
CORE_TERMS = "core_terms"
GROUPS = "groups"
AVG_SIZE = "avg_size"
WEIGHT_GR = "weighted_group"
WEIGHT_L = "weighted_label"
APPR = "approach"
WEIGHT_TYPE = "w_type"
SCORE = "score"
ANEA = "ANEA"
HC = "HC"
PART = "participant"
BAL_GR = "balanced_group"
BAL_L = "balanced_label"

VOCAB = "vocab"
LABELS = "labels"
