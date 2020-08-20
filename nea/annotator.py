from typing import Dict, List


class Annotator:

    def __init__(self):
        pass

    def extract_labels(self) -> Dict[str, List[str]]:
        raise NotImplementedError
