import pandas as pd
from typing import List
import os
import re

from reading.reader import DocReader
from config import DATA_PATH
from reading.connect_db import read_data_db

# ---------
READ_DB = False
# ---------
FILE_NAME = "sc_data_short.csv"


class SCReader(DocReader):
    """
    A reader of a old SC data.
    """
    def _read_text(self) -> List[str]:
        if READ_DB or FILE_NAME not in os.listdir(DATA_PATH):
            read_data_db()
        data_df = pd.read_csv(os.path.join(DATA_PATH, FILE_NAME), index_col=[0])
        output_list = []
        for v in list(data_df["text"].values):
            if type(v) != str:
                continue

            v_edit = re.sub(r'\n\n', "", v)
            v_edit = re.sub(r'\n', ". ", v_edit)
            output_list.append(v_edit)
        return output_list
