import shortuuid
from utils.spellchecker_ import correct
import string
import numpy as np
import re
from utils.CharSplit import char_split
from utils.numberbatch import get_model
from utils.get_wikt import get_page
from config import MAX_COMPOUND_CANDS, MAX_REC_LEVEL, WIKTIONARY_PAGE
import ftfy
import requests
from bs4 import BeautifulSoup


class Term:

    def __init__(self, word, token):
        self.id = shortuuid.uuid()
        self.model = get_model()
        self.word = word.capitalize() if not word[0].isupper() else word
        self.token = token
        self.counter = 1
        self.correct = self.word # correct(lemma)
        self.wikt_page = None
        self.conceptnet_page = None
        self.head = None
        self.conceptnet_page_head, (self.wikt_page_head, self.head) = self._get_head(self.correct)
        # self.head = self.wikt_page_head.split("/")[-1].title() if self.wikt_page_head is not None else self.word
        self.head = self.head.title() if self.head is not None else self.word
        self.concept = None

    def __str__(self):
        return self.word

    def increase_counter(self):
        self.counter += 1

    def _get_head(self, orig_word):
        heads = self._compound_splitter(orig_word, 1)
        if len(heads):
            return heads[0][0], self._check_url(heads[0][1]) if heads[0][2] is None else (heads[0][1], heads[0][2])
        return None, (None, None)

    @staticmethod
    def _convert_format(word, lang='de'):
        return ftfy.fix_text("/c/" + lang + "/" + word.lower())

    def _compound_splitter(self, orig_lemma, rec_level):
        """
        Destillatablaufschwankungen --> [('/c/de/schwankungen', 'de.wiktionary.org/wiki/Schwankungen')]
        :param orig_lemma:
        :return:
        """
        if orig_lemma.isspace() or orig_lemma.isnumeric() or orig_lemma in string.punctuation:
            return []

        cn_origword = Term._convert_format(orig_lemma)
        if cn_origword in self.model:
            self.conceptnet_page = cn_origword if rec_level == 1 else self.conceptnet_page
            page = get_page(orig_lemma.capitalize())
            if page.status_code == 200:
                if rec_level == 1:
                    self.wikt_page, self.head = self._check_url(WIKTIONARY_PAGE.format(orig_lemma.capitalize()))
                return [(self.conceptnet_page, self.wikt_page, self.head)]

        if any([l in string.punctuation for l in orig_lemma]):
            # split at any non-letter character
            compound_list = [re.split(r'[!"#$%&\'()*+,-.:;<=>?@[\\^_`{|}~/]', orig_lemma)]
            strt_index = 0

        else:
            # split with compound splitter tool
            compound_list = list(filter(lambda x: x[0] > -3, char_split.split_compound(orig_lemma)))
            strt_index = 1

        if len(compound_list) == 0 or rec_level == MAX_REC_LEVEL:
            return []

        all_candidates = []
        for compounds in compound_list[:MAX_COMPOUND_CANDS]:
            head_candidates = []
            if strt_index == 1:
                if compounds[1] == compounds[2]:
                    return head_candidates

            potential_head = compounds[-1]
            if potential_head in string.punctuation or potential_head.isnumeric():
                continue

            try:
                found_word = False
                for  head_modified in [potential_head, potential_head.replace("ß", "ss"),
                                      potential_head[:-1]]:
                    head_modified_numberb = Term._convert_format(head_modified)
                    if head_modified_numberb in self.model:
                        page = get_page(potential_head.capitalize())
                        if page.status_code == 200:
                            head_candidates.append((head_modified_numberb, WIKTIONARY_PAGE.format(potential_head.capitalize()), head_modified))
                            found_word = True
                            break

                if not found_word:
                    head_candidates.extend(self._compound_splitter(potential_head, rec_level + 1))
            except ValueError:
                continue

            if len(head_candidates):
                all_candidates.append(head_candidates)
                if len(head_candidates[0][0].split("/")[-1]) < 4:
                    # too short head candidate
                    continue
                return head_candidates

        if not len(all_candidates):
            return []

        # if many heads available, choose the best
        max_val = [np.max(np.array([len(l[-1]) for l in ls]), axis=0) for ls in all_candidates]
        mean_val = [np.mean(np.array([len(l[-1]) for l in ls]), axis=0) for ls in all_candidates]
        best_match = np.argmax([max_ + mean_ for max_, mean_ in zip(max_val, mean_val)])
        return all_candidates[int(best_match)]

    def _check_url(self, url):
        # check is there is a normalized form of a word available
        if url is None:
            return None, None

        # page = requests.get(url)
        page = get_page(url)
        soup = BeautifulSoup(page.content, "html.parser")
        orig_forms = [s for s in soup.find_all(title="Grammatische Merkmale")]
        if len(orig_forms) == 0:
            return url, url.split("/")[-1]

        if "#" in url:
            return url, url.split("/")[-1].split("#")[0]

        try:
            cand = None
            for orig_form in [s for s in soup.find_all("li")]:
                for cand in list(orig_form.find_all("a")):
                    if ":" in cand.attrs["href"]:
                        continue

                    if "#" in cand.attrs["href"]:
                        return url + cand.attrs["href"], url.split("/")[-1]

                    return "/".join(url.split("/")[:-2]) + cand.attrs["href"], cand.attrs["title"]

            if cand is None:
                return url, url.split("/")[-1]

        except IndexError:
            return url, url.split("/")[-1]
