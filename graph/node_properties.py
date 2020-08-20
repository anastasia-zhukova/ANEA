import re
import requests
from bs4 import BeautifulSoup
import ftfy
from typing import Dict

from utils.parser import nlp
from config import *

WIKI_URL = "https://{0}.wiktionary.org"


class WikiLink:

    def __init__(self, text: str, link: str):
        self.text = text
        self.link = link
        if link is not None:
            if "http" not in link:
                self.link = WIKI_URL.format(LANGUAGE) + link
            else:
                self.link = link

    def __repr__(self):
        return self.link if self.link is not None else self.text


class WikiProperties:

    def __init__(self, text: str, links: Dict[str, WikiLink], senses: str = "none", areas: str = "none", node=None):
        self.text = text.lstrip()
        self.links = links
        self.senses = senses
        self.areas = areas
        self.node = node

    def __repr__(self):
        return self.text

    @staticmethod
    def extract_wiki_properties(url, properties_dict, hypernyms):
        content_dict = {}
        if hypernyms is not None:
            # add hypernyms-heads extracted at the preprocessing step
            content_dict[HYPERNYM] = []
            for hypernym_link in hypernyms:
                text = hypernym_link.split("/")[-1]
                if text in properties_dict:
                    prop = properties_dict[text]
                else:
                    prop = WikiProperties(text=text, links={text: WikiLink(text=text, link=hypernym_link)})
                    properties_dict[text] = prop
                content_dict[HYPERNYM].append(prop)

        if url is None:
            return content_dict

        # parse a Wiktionary page
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")

        # two sections which are always (in most cases) are present on each wikt page
        content_dict[NAME] = [s for s in soup.find_all("h1")][0].text
        content_dict[POS] = re.split(r'[\W]+', [s for s in soup.find_all("h3")][0].text)[0]

        # extract other sections which are optional for each page
        for p, dl in zip([s for s in soup.find_all("p") if "title" in s.attrs or len(s.text) > 3],
                         [s for s in soup.find_all("dl") if s.parent.name == "div"
                                                            and [ss for ss in s.previous_siblings][1].name == "p"]):
            # section name
            sect = p.text[:-1]
            if sect in content_dict:
                continue

            content_dict[sect] = []

            # iterate over items in each section
            for d in dl.find_all("dd"):
                # collect links
                links = {}
                for a in d.find_all("a"):
                    if "href" not in a.attrs:
                        continue
                    if "title" not in a.attrs:
                        continue
                    # ignore the inactive links
                    if "edit&redlink" in a.attrs["href"]:
                        continue
                    links.update({a.text:  WikiLink(text=a.attrs["title"].split("(")[0], link=a.attrs["href"])})

                # collect senses assigned to each section item
                senses = []
                text = d.text
                # sense candidates that are listed as comma separated, e.g., [1,2]
                sense_candidates = re.findall(r'\[[0-9\-?\,\s]{3,}\]', d.text)
                if len(sense_candidates):
                    text = re.sub(r'[\[0-9\-?,\s\]]{3,}', "", d.text)
                    if "-" in sense_candidates[0]:
                        codes = re.findall(r'[0-9\-\s]{3,}', sense_candidates[0])[0]
                        if all([a.strip().isdigit() for a in codes.split("-")]):
                            codes_ = [int(a.strip()) for a in codes.split("-")]
                            senses = [str(j) for j in range(codes_[0], codes_[-1] + 1)]
                    elif "," in sense_candidates[0]:
                        codes = re.findall(r'[0-9\,\s]{3,}', sense_candidates[0])[0]
                        senses = [a.strip() for a in codes.split(",")]
                    else:
                        senses = re.findall(r'[0-9]{3,}', sense_candidates[0])[0]
                else:
                    # senses listed as single items, e.g., [1]
                    sense_candidates = re.findall(r'[\[0-9a-z?]{2,}\]', d.text)
                    if len(sense_candidates):
                        senses = [sense_candidates[0][1:-1]]
                        text = re.sub(r'[\[0-9a-z?]{2,}\]', "", d.text)

                # extract areas and text from layout "[1] Technik: Text of the description."
                split_type = re.split(r':', text)
                if len(split_type) > 1:
                    areas = split_type[0].strip().split(",")
                    text = ":".join(split_type[1:])
                else:
                    areas = "none"

                # create wiki properties
                if sect in [HYPERNYM, HYPONYM, SYNOMYM]:
                    # use each link to create a wiki property
                    for link_text, link in links.items():
                        if link_text == content_dict[NAME]:
                            continue
                        # ignore if a verb
                        if not link_text[0].isupper() and link_text[-1] == "n":
                            continue
                        if link_text in properties_dict:
                            prop = properties_dict[link_text]
                        else:
                            prop = WikiProperties(text=link_text, links={link_text: link}, senses=senses,
                                                  areas=areas)
                            properties_dict[link_text] = prop
                        content_dict[sect].append(prop)
                else:
                    # create unique properties for other section items
                    prop = WikiProperties(text=text, links=links, senses=senses, areas=areas)
                    content_dict[sect].append(prop)

        if MEANING in content_dict:
            # extract hypernyms from the text of definitions
            hypernym_cands = WikiProperties._extract_extra_hypernyms(content_dict[MEANING],
                                                          content_dict[HYPERNYM] if HYPERNYM in content_dict else [])
            for hypernym_cand, sense in hypernym_cands:
                if hypernym_cand == content_dict[NAME]:
                    continue

                if HYPONYM in content_dict:
                    if hypernym_cand in [v.text for v in content_dict[HYPONYM]]:
                        continue

                if hypernym_cand in properties_dict:
                    prop = properties_dict[hypernym_cand]
                else:
                    page = requests.get(WIKTIONARY_PAGE.format(hypernym_cand))
                    if page.status_code == 200:
                        prop = WikiProperties(text=hypernym_cand,
                                              links={hypernym_cand: WikiLink(text=hypernym_cand,
                                                                  link=WIKTIONARY_PAGE.format(hypernym_cand))},
                                              senses=sense, areas="none")
                        properties_dict[hypernym_cand] = prop
                    else:
                        # if a hypernym doesn't have a page in Wiktionary, add it to the wiki properties anyway
                        prop = WikiProperties(text=hypernym_cand,
                                          links={},
                                          senses=sense, areas="none")
                        properties_dict[hypernym_cand] = prop

                if HYPERNYM not in content_dict:
                    content_dict[HYPERNYM] = []
                content_dict[HYPERNYM].append(prop)

        if MEANING in content_dict and HYPERNYM in content_dict:
            # add types/areas of concepts from definitions to hypernyms via linking with senses
            senses = {}
            for definition in content_dict[MEANING]:
                if type(definition.senses) == str or type(definition.areas) == str:
                    continue
                senses.update(zip(definition.senses, definition.areas))

            for hypernym in content_dict[HYPERNYM]:
                if not len(hypernym.senses):
                    continue
                hypernym.areas = []
                for s in hypernym.senses:
                    if s not in senses:
                        continue
                    hypernym.areas.append(senses[s])

        return content_dict

    @staticmethod
    def _extract_extra_hypernyms(definitions, old_hypernyms):
        # extract hypernyms from definitions
        hypernyms_cands = []

        if definitions is None:
            return set()

        for definition in definitions:
            doc = nlp()(ftfy.fix_text(definition.text))
            hypernym_cand = ''
            for token in doc:
                if token.dep_ in ["ROOT", "oa", "oa2", "app", "cj"] and token.pos_ in ["NOUN", "PROPN"]:
                    hypernym_cand = token.orth_
                    break

            if not hypernym_cand:
                continue

            hypernym_id_cands = [hypernym_cand in o.text for o in old_hypernyms]
            if sum(hypernym_id_cands) == 0:
                # use only unique hypernyms
                hypernyms_cands.append((hypernym_cand.strip(), definition.senses))
        return hypernyms_cands


class Areas:
    '''
    Groups of wiki areas and their scores of similarity and frequency of usage in wiki properties.
    '''
    def __init__(self, areas=None, score=0, count=0):
        self.areas = areas if areas is not None else []
        self.score = score
        self.count = count

    def __repr__(self):
        return self.areas
