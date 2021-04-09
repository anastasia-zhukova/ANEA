from preprocessing.preprocesor import Preprocessor
from category_identificator.ANEA_annotator.graph.graph import Graph
from category_identificator.ANEA_annotator.annotator_ANEA import ANEAAnnotator
from reading.reading_wiki import WikiReader
from reading.reader_json import JsonReader
from config import EXEC_RES_PATH, logger
import datetime
import os, json, sys
import pandas as pd
import numpy as np


def _combine_results(exec_results_dict):
    # voting
    all_vocab = set().union(*[vv for v in exec_results_dict.values() for k, vv in v.items()])
    all_labels = set().union(*[v.keys() for v in exec_results_dict.values()])
    vocab_df = pd.DataFrame(np.zeros((len(all_vocab), len(all_vocab))), columns=all_vocab,index=all_vocab)
    label_df = pd.DataFrame(np.zeros((len(all_vocab), len(all_labels))), columns=all_labels, index=all_vocab)

    for k, v in enumerate(exec_results_dict.values()):
        for col, t_gr in v.items():
            for t in t_gr:
                vocab_df.loc[t, t_gr] += 1
            label_df.loc[t_gr, col] += 1


    def _expand_chain_count(set_to_check, round_, recursion_level=1):
        output = set()
        for w in set_to_check:
            if round_ == 0:
                if w in checked_vocab:
                    continue
            else:
                if w in checked_vocab - set_to_check:
                    continue
            suitable_df = vocab_df[vocab_df[w] >= 2]
            checked_vocab.add(w)
            diff = {v for v in set(suitable_df.index) - set_to_check if v not in checked_vocab}
            output = output.union(diff)

        if len(output) > 1 and recursion_level <= 1:
            output = output.union(_expand_chain_count(output, round_=round_, recursion_level=recursion_level+1))

        return output.union(set_to_check)

    j = 0
    classes_list = []
    checked_vocab = set()
    for word in list(vocab_df.mean(axis=0).sort_values(ascending=False).index):
        if word in checked_vocab:
            continue

        chain = list(_expand_chain_count({word}, round_=0))

        for w in chain:
            checked_vocab.add(w)

        if len(chain) < 5:
            continue

        potential_labels_df = label_df.loc[chain]
        pot_labels_dict = {}
        for col in list(label_df.loc[chain].columns):
            values = [v for v in potential_labels_df[col].values if v > 0]
            if len(values) < 2:
                continue
            pot_labels_dict[col] = np.mean(values)

        label = max(pot_labels_dict, key=pot_labels_dict.get) if len(pot_labels_dict) else "cl_" + str(j)
        j += 1
        classes_list.append({"label": label,
                             "terms": chain})

    classes_dict = {}
    for cl in classes_list:
        if cl["label"] not in classes_dict:
            if not len(cl["terms"]):
                continue
            classes_dict[cl["label"]] = []
        classes_dict[cl["label"]] += cl["terms"]
    return classes_dict


def run():
    now = datetime.datetime.now()
    date_ = now.strftime("%Y-%m-%d")
    date_folder = os.path.join(EXEC_RES_PATH, date_)
    if not os.path.exists(date_folder):
        os.makedirs(date_folder)

    inp = ""
    while not len(inp):
        inp = input("Type a name of a json file (with a list of document strings) from the data folder to be processed or a topic for "
                "Wikipedia articles (with a capital letter):")

    if ".json" in inp:
        reader = JsonReader(inp)
        topic = inp.split(".json")[0]
    else:
        reader = WikiReader(inp)
        topic = inp

    raw_text = reader.text_collection
    if not len(raw_text):
        sys.exit("Input text is empty! The execution is aborted.")

    preprocessor = Preprocessor()
    noun_terms, len_words = preprocessor.preprocess(raw_text)
    logger.info("Unique terms: {0}.".format(str(len(noun_terms))))

    # empirically derived number of top heads for the voting strategy across multiple run executions
    topN_terms_mean = int(158 + 0.167 * len(noun_terms))

    if topN_terms_mean > 40:
        topN_all = [topN_terms_mean - 40, topN_terms_mean, topN_terms_mean + 40]
    else:
        topN_all = [topN_terms_mean, topN_terms_mean + 40]

    exec_results_dict = {}

    for i, topN in enumerate(topN_all):
        logger.info("Graph contruction, iteration {0}.\n".format(i))
        graph = Graph(noun_terms, topN_terms=topN)
        graph.grow_graph()

        logger.info("ANEA execution.\n")
        anea = ANEAAnnotator(graph=graph, topic=topic, date=date_)
        labels_anea, outliers_anea = anea.extract_labels()
        exec_results_dict[i] = {k.strip(): [v.strip() for v in vv] for k, vv in labels_anea.items()}

    classes_dict = _combine_results(exec_results_dict)

    with open(os.path.join(date_folder, topic, "final_categories.json"), "w") as file:
        json.dump(classes_dict, file)

    logger.info("Execution is over, the results are saved to {}".format(os.path.join(date_folder, "final_categories.json")))


if __name__ == '__main__':
    run()
