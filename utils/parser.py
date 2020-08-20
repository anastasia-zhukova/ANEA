import spacy
import sys

try:
    _nlp = spacy.load('de_core_news_sm')
    print("Spacy loaded")
except OSError:
    print(
        "Download the german model first and then restart the script: \"python -m spacy download de_core_news_sm\".")
    sys.exit(0)


def nlp():
    return _nlp
