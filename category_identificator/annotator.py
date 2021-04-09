from typing import Dict, List
import os, json
import datetime

from config import EXEC_RES_PATH


class Annotator:

    def __init__(self, topic=None, date=None):
        self.topic = topic
        self.params = {}
        self.terms_to_annotate = None
        self.cluster_all = False
        if type(date) != str:
            now = datetime.datetime.now()
            self.date = now.strftime("%Y-%m-%d")
        else:
            self.date = date

    def extract_labels(self, terms_to_annotate=None, cluster_all: bool = False) -> (Dict[str, List[str]], List[str]):
        '''
        Main execution method.
        :return: a dictionary <label: list of terms>, a list of outliers, i.e., non-resolved terms
        '''
        self.terms_to_annotate = terms_to_annotate
        self.cluster_all = cluster_all
        label_dict, outliers = self._extract_labels()
        name = self.__class__.__name__
        now = datetime.datetime.now()

        date_folder = os.path.join(EXEC_RES_PATH, self.date)
        if not os.path.exists(date_folder):
            os.makedirs(date_folder)

        topic_folder = os.path.join(date_folder, self.topic)
        if not os.path.exists(topic_folder):
            os.makedirs(topic_folder)

        with open(os.path.join(topic_folder, now.strftime("%Y-%m-%d_%H-%M") + "_" + name + "_" +
                                             self.topic + "_labels" + ".json"), "w") as file:
            json.dump(label_dict, file)

        with open(os.path.join(topic_folder, now.strftime("%Y-%m-%d_%H-%M") + "_" + name + "_" +
                                             self.topic + "_outliers" + ".json"), "w") as file:
            json.dump(outliers, file)

        return label_dict, outliers

    def _extract_labels(self) -> (Dict[str, List[str]], List[str]):
        raise NotImplementedError
