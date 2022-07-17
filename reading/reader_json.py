import os, json
from typing import List
from reading.reader import DocReader
from config import DATA_PATH


class JsonReader(DocReader):
    def __init__(self, file_name: str = None):
        if file_name is None:
            raise("No file for annotation given, give a name of a JSON file with a list of strigs placed in the"
                  " {0} folder".format(DATA_PATH))
        else:
            self.file_name = file_name
            super().__init__()

    def _read_text(self) -> List[str]:
        with open(os.path.join(DATA_PATH, self.file_name), "r", encoding='utf8') as file:
            records = json.load(file)
        return records
