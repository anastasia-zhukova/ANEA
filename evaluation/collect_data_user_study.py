from reading.reader_sc_sample import SCCSVReader
from preprocessing.preprocesor import Preprocessor
from category_identificator.ANEA_annotator.graph.graph import Graph
from category_identificator.ANEA_annotator.annotator_ANEA import ANEAAnnotator
from category_identificator.baselines.annotator_clustering import Clustering
from reading.reading_wiki import WikiReader
from config import EXEC_RES_PATH, CSVS_PATH, logger
import datetime
import os, json
import pandas as pd


# dataset setup
# data for "processing" cannot be shared die to the privacy reasons
datasets ={
    "processing": SCCSVReader(),
    "databases": WikiReader("Datenbanken"),
    "softwaredev": WikiReader("Programmierung"),
    "travel": WikiReader("Reise")
}

params = {}

now = datetime.datetime.now()
date_ = now.strftime("%Y-%m-%d")
date_folder = os.path.join(EXEC_RES_PATH, date_)
if not os.path.exists(date_folder):
    os.makedirs(date_folder)

for topic, reader in datasets.items():
    logger.info("Topic: " + topic)
    raw_text = reader.text_collection

    if not len(raw_text):
        continue

    preprocessor = Preprocessor()
    noun_terms, len_words = preprocessor.preprocess(raw_text)

    logger.info("Unique terms: {0}.".format(str(len(noun_terms))))

    params[topic] = {"all_words": len_words,
                     "unique_terms": len(noun_terms)}
    run_params = {}

    if len(noun_terms) < 1000:
        frac_freq_heads_list = [2, 3, 4, 5]
    elif 1000 <= len(noun_terms) < 1200:
        frac_freq_heads_list = [3, 4, 5, 7]
    else:
        frac_freq_heads_list = [4, 5, 7, 9]

    for frac_freq_heads in frac_freq_heads_list:
        # build a graph on the 1/frac_frec_heads
        logger.info("Graph contruction.\n")
        graph = Graph(noun_terms, frac_freq_heads)
        graph.grow_graph()
        params[topic].update({"unique_heads": graph.num_freq_gr_all})
        terms_to_annotate = [n for n, node in graph.all_nodes.items() if node.term_id is not None]

        logger.info("ANEA execution.\n")
        anea = ANEAAnnotator(graph=graph, topic=topic, date=date_)
        labels_anea, outliers_anea = anea.extract_labels()

        logger.info("HC execution.\n")
        clust = Clustering(topic=topic, date=date_)
        labels_cl, outliers_cl = clust.extract_labels(terms_to_annotate=set(terms_to_annotate))

        run_params[frac_freq_heads] = {
            "sel_unique_heads": graph.num_freq_gr_sel,
            "terms_to_cluster": len(terms_to_annotate),
            "clustered_sem": len(terms_to_annotate) - len(outliers_anea),
            "groups_sem": len(labels_anea),
            "clustered_agl": len(terms_to_annotate) - len(outliers_cl),
            "groups_cl": len(labels_cl),
            "clust_params": clust.params
        }

    params[topic].update({"run_params": run_params})


with open(os.path.join(date_folder, "run_results.json"), "w") as file:
    json.dump(params, file)
logger.info("Execution results saved to " + date_folder + ".\n")

# convert json into csv
CSV_PATH = os.path.join(CSVS_PATH, date_)
if not os.path.isdir(CSV_PATH):
    os.makedirs(CSV_PATH)

all_vocab = {}
all_labels = {}


for topic in os.listdir(date_folder):
    if os.path.isfile(os.path.join(date_folder, topic)):
        continue

    dirs = os.listdir(os.path.join(os.getcwd(), "input", topic))
    fracs = list(params[topic]["run_params"])
    j = -1

    vocab = set()
    local_labels = {}

    for i, dir_ in enumerate(dirs):
        if i % 2 != 0:
            continue

        if i % 4 == 0:
            j += 1

        with open(os.path.join(date_folder, topic, dirs[i]), "r") as file:
            labels_dict = json.load(file)

        with open(os.path.join(date_folder, topic, dirs[i + 1]), "r") as file:
            outliers = json.load(file)

        max_ = max([len(v) for v in labels_dict.values()] + [len(outliers)])
        labels_dict = {k: v for k,v in sorted(labels_dict.items(), reverse=True, key=lambda x: len(x[1]))}
        new_labels_dict = {k: [""] * 4 + v + [""] * (max_ - len(v)) for k,v in labels_dict.items()}

        w_spaces_labels_dict = {}
        for i, (k, v) in enumerate(new_labels_dict.items()):
            w_spaces_labels_dict.update({
                k: v,
                "bl1_" + str(i): [""] * len(v)
            })

        data_df = pd.DataFrame(w_spaces_labels_dict)
        now = datetime.datetime.now()
        data_df.to_csv(os.path.join(CSV_PATH,
                                    now.strftime("%Y-%m-%d_%H-%M") + "_" + str(fracs[j]) + "_" +
                                    dir_.split("_")[-3] + "_" + topic + ".csv"),
                       encoding='utf-8-sig')

        vocab = vocab.union(set(outliers))
        vocab = vocab.union(set([l for group in list(labels_dict.values()) for l in group]))

        for label in list(labels_dict):
            if "cl_" in label:
                continue
            if label not in all_labels:
                local_labels[label] = labels_dict[label]
            else:
                terms_union = set(local_labels[label]).union(set(labels_dict[label]))
                local_labels[label] = list(terms_union)

    all_vocab[topic] = [v for v in list(vocab) if len(v)]
    all_labels[topic] = local_labels

with open(os.path.join(date_folder, date_ + "_allvocab.json"), "w") as file:
    json.dump(all_vocab, file)

with open(os.path.join(date_folder, date_ + "_alllabels.json"), "w") as file:
    json.dump(all_labels, file)

logger.info("JSONs to CSV converted to the folder: " + CSV_PATH)
