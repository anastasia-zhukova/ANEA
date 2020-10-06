from reading.reader_sc_sample import SCCSVReader
from preprocessing.preprocesor import Preprocessor
from entity_class_annotator.ANEA_annotator.graph.graph import Graph
from entity_class_annotator.ANEA_annotator.annotator_ANEA import ANEAAnnotator
from entity_class_annotator.baselines.annotator_clustering import Clustering
from reading.reading_wiki import WikiReader
from config import EXEC_RES_PATH
import datetime
import os, json




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
    print("Topic: " + topic)
    raw_text = reader.text_collection

    if not len(raw_text):
        continue

    preprocessor = Preprocessor()
    noun_terms, len_words = preprocessor.preprocess(raw_text)

    print("Unique terms: {0}.".format(str(len(noun_terms))))

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
        print("Graph contruction.\n")
        graph = Graph(noun_terms, frac_freq_heads)
        graph.grow_graph()
        params[topic].update({"unique_heads": graph.num_freq_gr_all})
        terms_to_annotate = [n for n, node in graph.all_nodes.items() if node.term_id is not None]

        print("ANEA execution.\n")
        anea = ANEAAnnotator(graph=graph, topic=topic, date=date_)
        labels_anea, outliers_anea = anea.extract_labels()

        print("HC execution.\n")
        clust = Clustering(set(terms_to_annotate), topic=topic, date=date_)
        labels_cl, outliers_cl = clust.extract_labels()

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


now = datetime.datetime.now()
with open(os.path.join(date_folder, now.strftime("%Y-%m-%d_%H-%M") + "_run_results.json"), "w") as file:
    json.dump(params, file)
print("Execution results saved to " + date_folder + ".\n")
