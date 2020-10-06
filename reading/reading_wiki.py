from bs4 import BeautifulSoup
import os, json
import wikipedia as wiki
import nltk
nltk.download('punkt')
from textblob_de import TextBlobDE
import requests
from typing import List
import string
wiki.set_lang("de")

from reading.reader import DocReader
from config import DATA_PATH

WIKI_LINK = "https://de.wikipedia.org/w/index.php?title=Spezial:Suche&limit=100&offset=0&ns0=1&search=incategory%3A{0}&ns0=1"
MAX_LEN = 7000
MAX_ART_LEN = 2500
MIN_ART_LEN = 220
TOPIC_NAME = "Datenbanken"


class WikiReader(DocReader):
    def __init__(self, topic=None):
        if topic is None:
            raise("No topic for Wikipedia parsing given!")
        else:
            self.topic = topic
            self.list_wiki = WIKI_LINK.format(self.topic)
            super().__init__()

    def _read_text(self) -> List[str]:
        print("Data for " + self.topic)
        data_file_path = os.path.join(DATA_PATH, self.topic + ".json")
        if os.path.exists(data_file_path):
            with open(data_file_path, "r", encoding='utf8') as file:
                records = json.load(file)
                if len(records):
                    return records

        page = requests.get(self.list_wiki)
        soup = BeautifulSoup(page.content, "html.parser")
        results = soup.findAll("div", {"class": 'mw-search-result-heading'})

        records = []
        length_ = 0

        for res in results:
            try:
                page = wiki.page(res.text.strip())
                blob = TextBlobDE(page.content)
                if MIN_ART_LEN > len(blob.tokens) or len(blob.tags) > MAX_ART_LEN:
                    continue

                if len(blob.tags) + length_ > MAX_LEN:
                    break

                records.append(page.content)
                length_ += len(blob.tokens)
            except wiki.exceptions.PageError:
                print(res.text)
                continue
        with open(data_file_path, "w", encoding='utf8') as file:
            json.dump(records, file)
        return records


if __name__ == '__main__':
    data = WikiReader()
    a = 0