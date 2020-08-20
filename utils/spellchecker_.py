from spellchecker import SpellChecker

checker = SpellChecker(language='de')


def correct(word):
    return checker.correction(word)
