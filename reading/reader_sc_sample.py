import pandas as pd
from typing import List
import os

from reading.reader import DocReader
from config import DATA_PATH


class SCSampleReader(DocReader):
    """A reader of SC sample data."""
    def _read_text(self) -> List[str]:
        data_df = pd.read_csv(os.path.join(DATA_PATH, "sc_sample.csv"), index_col=[0])
        return [v for v in list(data_df["Description"].values) if type(v) == str]


if __name__ == '__main__':
    SCSampleReader()