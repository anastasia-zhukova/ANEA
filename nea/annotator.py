from typing import Dict, List


class Annotator:

    def extract_labels(self) -> (Dict[str, List[str]], List[str]):
        '''

        :return: a dictionary <label: list of terms>, a list of outliers, i.e., non-resolved terms
        '''
        raise NotImplementedError
