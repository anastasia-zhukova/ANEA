from typing import List


class DocReader:
    """
    A general class to read information for the various sources.
    """
    def __init__(self):
        self.text_collection = self._read_text()

    def _read_text(self) -> List[str]:
        """

        :return: a list or strings for later preprocesor.py
        """
        raise NotImplementedError
