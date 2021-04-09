import spacy
import sys
from config import logger

_nlp = None


def nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load('de_core_news_sm')
            logger.info("Spacy loaded")
        except OSError:
            logger.info("Download the german model first and then restart the script: "
                        "\"python -m spacy download de_core_news_sm\".")
            sys.exit(0)
    return _nlp
