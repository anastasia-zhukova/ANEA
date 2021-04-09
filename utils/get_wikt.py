import requests
from config import WIKTIONARY_PAGE, logger
import time
import ftfy

wikt_dict = {}

def get_page(term):
    if "http" in term:
        term_ = term.split("/")[-1]
    else:
        term_ = term

    if term_ not in wikt_dict:
        try_again = True
        while try_again:
            try:
                wikt_dict[term_.capitalize()] = requests.get(WIKTIONARY_PAGE.format(term_.capitalize()))
                try_again = False
            except requests.exceptions.SSLError:
                logger.warning("Needed to wait for the \"{}\" page.".format(ftfy.fix_text(term_.capitalize())))
                time.sleep(15)
    return wikt_dict[term_.capitalize()]
