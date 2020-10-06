import pandas as pd
from typing import List
import os

from reading.reader import DocReader
from config import DATA_PATH


class SCCSVReader(DocReader):
    """A reader of SC selected data."""
    def _read_text(self) -> List[str]:
        try:
            data_df = pd.read_csv(os.path.join(DATA_PATH, "sc_sample.csv"), index_col=[0])
            return [v for v in list(data_df["Description"].values) if type(v) == str]
        except FileNotFoundError:
            print("No file for \"Processing\" given. WIll be skipped. \n")
            return []


if __name__ == '__main__':
    SCCSVReader()